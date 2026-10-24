#pragma once

// Lab 07 — visual inspection baseline. For each image:
//   [RGB -> gray] -> normalize -> box smooth (radius R) -> residual = |normalized - smoothed|
//   -> mask = residual > pixel_threshold -> score = mask pixels / image pixels
// The stages reuse your Lab 05 and Lab 06 kernels; the new piece is the mask-count reduction.
// Every stage is integer-exact, so CPU and GPU mask counts must be identical.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/image.hpp"
#include "lab05/lab05.hpp"
#include "lab06/lab06.hpp"

namespace lab07 {

struct DetectorParams {
    int lo = 0;                 // normalization levels (Lab 05); 0 and 255 leave values unchanged
    int hi = 255;
    int radius = 3;             // box-filter radius (Lab 06)
    int pixel_threshold = 30;   // a residual above this marks the pixel
    dim3 block = dim3(16, 16);  // thread-block shape for every kernel
};

// Device-side phase times for one image, in milliseconds (CUDA events).
struct GpuPhaseTimes {
    double h2d_ms = 0.0;
    double compute_ms = 0.0;
    double d2h_ms = 0.0;
};

// ---- Implemented in src/lab07/gpu_inspect.cu (yours) or solutions/lab07/gpu_inspect.cu ------
// Zeroes *count (a device counter), then adds the number of nonzero bytes of the pitched device
// mask to it. block.x * block.y must be a power of two. Enqueues work without synchronizing.
void launch_count_mask(common::ConstImageView mask, unsigned long long* count, dim3 block,
                       cudaStream_t stream = nullptr);

// Runs the whole detector for one host image on the GPU and returns its number of mask pixels.
// When mask_out is not null, also downloads the mask into it.
unsigned long long score_gpu(const common::HostImage& image, const DetectorParams& params, GpuPhaseTimes& times,
                             common::HostImage* mask_out);

// ---- CPU reference: the same stages with the Lab 05 and Lab 06 CPU functions -----------------
inline unsigned long long score_cpu(const common::HostImage& image, const DetectorParams& params,
                                    common::HostImage* mask_out) {
    const int w = image.width;
    const int h = image.height;
    common::HostImage gray(w, h, 1), norm(w, h, 1), smooth(w, h, 1), residual(w, h, 1), mask(w, h, 1);
    const common::HostImage* source = &image;
    if (image.channels == 3) {
        lab05::rgb_to_gray_cpu(image, gray);
        source = &gray;
    }
    lab05::normalize_cpu(*source, norm, params.lo, params.hi);
    lab06::box_filter_cpu(norm, smooth, params.radius);
    lab05::absdiff_mask_cpu(norm, smooth, residual, mask, params.pixel_threshold);
    unsigned long long count = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) count += mask.at(x, y) != 0;
    }
    if (mask_out != nullptr) *mask_out = mask;
    return count;
}

// ---- Image lists --------------------------------------------------------------------------------
// A CSV with at least the columns image_id,file,split,label,defect_type, as written by
// scripts/mvtec_to_pgm.py and scripts/lab07_synthetic.py. `file` is relative to the list's
// directory, label is good or defect, and split is train, dev, or test. No quoted fields.
struct ImageEntry {
    std::string image_id;
    std::string path;  // `file` resolved against the list's directory
    std::string split;
    std::string label;
    std::string defect_type;
};

inline std::vector<std::string> split_csv_line(std::string line) {
    if (!line.empty() && line.back() == '\r') line.pop_back();  // tolerate Windows line endings
    std::vector<std::string> fields;
    std::size_t start = 0;
    while (true) {
        const auto comma = line.find(',', start);
        fields.push_back(line.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
        if (comma == std::string::npos) return fields;
        start = comma + 1;
    }
}

inline std::vector<ImageEntry> read_image_list(const std::string& list_path) {
    std::ifstream in(list_path);
    if (!in) throw std::runtime_error("Cannot open image list " + list_path + '.');
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error(list_path + " is empty.");
    const std::vector<std::string> header = split_csv_line(line);
    const auto column = [&](const std::string& name) {
        for (std::size_t i = 0; i < header.size(); ++i) {
            if (header[i] == name) return i;
        }
        throw std::runtime_error(list_path + ": missing column " + name + '.');
    };
    const std::size_t id_col = column("image_id");
    const std::size_t file_col = column("file");
    const std::size_t split_col = column("split");
    const std::size_t label_col = column("label");
    const std::size_t type_col = column("defect_type");
    const auto slash = list_path.find_last_of("/\\");
    const std::string base = slash == std::string::npos ? "" : list_path.substr(0, slash + 1);

    std::vector<ImageEntry> entries;
    std::set<std::string> ids;
    std::size_t line_number = 1;
    while (std::getline(in, line)) {
        ++line_number;
        if (line.empty() || line == "\r") continue;
        const std::vector<std::string> fields = split_csv_line(line);
        const std::string where = list_path + ':' + std::to_string(line_number);
        if (fields.size() != header.size()) {
            throw std::runtime_error(where + ": expected " + std::to_string(header.size()) + " fields.");
        }
        ImageEntry entry;
        entry.image_id = fields[id_col];
        const std::string& file = fields[file_col];
        entry.path = !file.empty() && file[0] == '/' ? file : base + file;
        entry.split = fields[split_col];
        entry.label = fields[label_col];
        entry.defect_type = fields[type_col];
        if (entry.image_id.empty() || file.empty()) throw std::runtime_error(where + ": empty image_id or file.");
        if (entry.label != "good" && entry.label != "defect") {
            throw std::runtime_error(where + ": label must be good or defect.");
        }
        if (entry.split != "train" && entry.split != "dev" && entry.split != "test") {
            throw std::runtime_error(where + ": split must be train, dev, or test.");
        }
        if (!ids.insert(entry.image_id).second) throw std::runtime_error(where + ": duplicate image_id " + entry.image_id + '.');
        entries.push_back(entry);
    }
    return entries;
}

}  // namespace lab07
