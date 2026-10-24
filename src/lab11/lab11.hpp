#pragma once

// Lab 11 — the capstone application `inspect`: one documented command per workflow.
//   inspect images  --input DIR|LIST.csv|IMAGE.pgm --output REPORT.csv [--config FILE] [options]
//   inspect sensors --input RECORDING.f32 --output REPORT.csv [--config FILE] [options]
// Images reuse Labs 05-07 and 09; sensors reuse Labs 02-04. The two workflows use independent
// datasets: never pair MVTec images with CWRU windows as if they described the same machine.

#include <dirent.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "lab04/lab04.hpp"
#include "lab07/lab07.hpp"

namespace lab11 {

enum class Backend { cpu, cuda, both };

struct ImagesConfig {
    std::string input;          // a directory of .pgm/.ppm images, an image-list .csv, or one image
    std::string split = "all";  // rows of a list input: train | dev | test | all
    std::string output;         // report CSV
    std::string mask_dir;       // when set, write <mask_dir>/<image_id>-mask.pgm
    Backend backend = Backend::cuda;
    lab07::DetectorParams detector;
    double image_threshold = std::numeric_limits<double>::quiet_NaN();  // NaN: no decision column
};

struct SensorsConfig {
    std::string input;   // a float32 recording (.f32)
    std::string id;      // recording id written to the report
    std::string output;  // report CSV
    Backend backend = Backend::cuda;
    lab04::PipelineConfig pipeline;
    lab04::AlarmRule rule;
};

// What one run did; the harness turns it into the summary line and the run record.
struct RunSummary {
    std::size_t items = 0;          // images, or windows of the recording
    std::size_t flagged = 0;        // predicted defects, or alarmed windows
    std::size_t disagreements = 0;  // --backend both: inputs (or windows) where CPU and GPU differ
    double decode_ms = 0.0;         // reading and decoding inputs
    double compute_ms = 0.0;        // scoring, including host-device transfers on the GPU
    std::vector<double> item_ms;    // decode + compute of each input, for latency percentiles
};

// ---- Implemented in src/lab11/app.cu (yours) or solutions/lab11/app.cu ------------------------
// Report columns for images: image_id,source,width,height,mask_pixels,score,predicted_defect
// Report for sensors: the Lab 04 format written by lab04::write_report.
RunSummary run_images(const ImagesConfig& config);
RunSummary run_sensors(const SensorsConfig& config);

// ---- Provided helpers -------------------------------------------------------------------------
inline bool ends_with(const std::string& text, const std::string& suffix) {
    return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// File name without its directories.
inline std::string base_name(const std::string& path) {
    const auto slash = path.find_last_of('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

// File name without directories or extension: the stable id of a directory or single-file input.
inline std::string stem(const std::string& path) {
    const std::string name = base_name(path);
    const auto dot = name.find_last_of('.');
    return dot == std::string::npos || dot == 0 ? name : name.substr(0, dot);
}

// The .pgm and .ppm files directly inside `directory`, sorted by name: readdir's order is not
// defined, and a report must enumerate its inputs the same way on every run.
inline std::vector<std::string> list_images(const std::string& directory) {
    DIR* dir = opendir(directory.c_str());
    if (dir == nullptr) throw std::runtime_error("Cannot open input directory " + directory + '.');
    std::vector<std::string> names;
    while (const dirent* entry = readdir(dir)) {
        const std::string name = entry->d_name;
        if (ends_with(name, ".pgm") || ends_with(name, ".ppm")) names.push_back(name);
    }
    closedir(dir);
    std::sort(names.begin(), names.end());
    const std::string prefix = ends_with(directory, "/") ? directory : directory + "/";
    for (auto& name : names) name = prefix + name;
    return names;
}

}  // namespace lab11
