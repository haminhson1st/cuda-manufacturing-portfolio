// Lab 07 application: score every image of a list with the inspection baseline and write one CSV
// row per image, then print one summary row with phase timings. You should not need to edit this
// file; your work is src/lab07/gpu_inspect.cu plus the threshold study described in the tutorial.

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/bench.hpp"
#include "common/cli.hpp"
#include "common/cuda_utils.cuh"
#include "lab07/lab07.hpp"

namespace {

const char* const usage = R"(Usage: score_images --list IMAGES.csv --output SCORES.csv [options]
  --split S               train | dev | test | all: rows of the list to score (default all)
  --backend B             cpu | cuda | both (both checks that CPU and GPU mask counts agree; default cuda)
  --lo L --hi H           normalization levels, 0 <= lo < hi <= 255 (default 0, 255)
  --radius R              smoothing radius, 0..16 (default 3)
  --pixel-threshold T     a residual above T marks the pixel, 0..254 (default 30)
  --image-threshold S     predicted_defect = 1 when score > S (default: column left empty)
  --block XxY             thread-block shape with a power-of-two thread count (default 16x16)
  --save-masks DIR        write DIR/<image_id>-mask.pgm for every image (DIR must exist)
  --header 0|1            print the CSV header of the summary line (default 1)
)";

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto start = common::Clock::now();
        common::Args args(argc, argv);
        if (args.help()) {
            std::cout << usage;
            return 0;
        }
        const std::string list = args.text("--list", "");
        const std::string output = args.text("--output", "");
        const std::string split = args.choice("--split", "all", {"train", "dev", "test", "all"});
        const std::string backend = args.choice("--backend", "cuda", {"cpu", "cuda", "both"});
        lab07::DetectorParams params;
        params.lo = static_cast<int>(args.count("--lo", 0, 0, 255));
        params.hi = static_cast<int>(args.count("--hi", 255, 0, 255));
        params.radius = static_cast<int>(args.count("--radius", 3, 0, 16));
        params.pixel_threshold = static_cast<int>(args.count("--pixel-threshold", 30, 0, 254));
        const double image_threshold = args.number("--image-threshold", std::numeric_limits<double>::quiet_NaN());
        unsigned bx = 0, by = 0;
        common::parse_block(args.text("--block", "16x16"), bx, by);
        params.block = dim3(bx, by);
        const std::string mask_dir = args.text("--save-masks", "");
        const bool header = args.choice("--header", "1", {"0", "1"}) == "1";
        args.finish();
        if (list.empty() || output.empty()) throw std::invalid_argument("--list and --output are required.");
        if (params.lo >= params.hi) throw std::invalid_argument("--lo must be below --hi.");
        const unsigned threads = bx * by;
        if ((threads & (threads - 1)) != 0) {
            throw std::invalid_argument("--block needs a power-of-two thread count (the mask-count reduction requires it).");
        }

        std::vector<lab07::ImageEntry> entries;
        for (const auto& entry : lab07::read_image_list(list)) {
            if (split == "all" || entry.split == split) entries.push_back(entry);
        }
        if (entries.empty()) throw std::runtime_error("No images in split '" + split + "' of " + list + '.');

        const bool run_cpu = backend != "cuda";
        const bool run_gpu = backend != "cpu";
        const bool want_masks = !mask_dir.empty();
        double decode_ms = 0.0, cpu_ms = 0.0, gpu_wall_ms = 0.0, pixels = 0.0;
        lab07::GpuPhaseTimes gpu_total;
        std::size_t mismatches = 0;
        std::string first_mismatch;

        std::ofstream out(output);
        if (!out) throw std::runtime_error("Cannot open " + output + '.');
        out << "image_id,split,label,defect_type,width,height,mask_pixels,score,predicted_defect\n" << std::setprecision(9);
        for (const auto& entry : entries) {
            auto phase = common::Clock::now();
            const common::HostImage image = common::read_pnm(entry.path);
            decode_ms += common::elapsed_ms(phase, common::Clock::now());
            const double image_pixels = static_cast<double>(image.width) * image.height;
            pixels += image_pixels;

            common::HostImage cpu_mask, gpu_mask;
            unsigned long long cpu_count = 0, gpu_count = 0;
            if (run_cpu) {
                phase = common::Clock::now();
                cpu_count = lab07::score_cpu(image, params, want_masks ? &cpu_mask : nullptr);
                cpu_ms += common::elapsed_ms(phase, common::Clock::now());
            }
            if (run_gpu) {
                lab07::GpuPhaseTimes times;
                phase = common::Clock::now();  // includes device allocation for this image
                gpu_count = lab07::score_gpu(image, params, times, want_masks ? &gpu_mask : nullptr);
                gpu_wall_ms += common::elapsed_ms(phase, common::Clock::now());
                gpu_total.h2d_ms += times.h2d_ms;
                gpu_total.compute_ms += times.compute_ms;
                gpu_total.d2h_ms += times.d2h_ms;
            }
            if (run_cpu && run_gpu && cpu_count != gpu_count && mismatches++ == 0) {
                first_mismatch = entry.image_id + ": CPU " + std::to_string(cpu_count) + " mask pixels, GPU "
                                 + std::to_string(gpu_count);
            }
            const unsigned long long count = run_gpu ? gpu_count : cpu_count;
            const double score = static_cast<double>(count) / image_pixels;
            out << entry.image_id << ',' << entry.split << ',' << entry.label << ',' << entry.defect_type << ','
                << image.width << ',' << image.height << ',' << count << ',' << score << ',';
            if (!std::isnan(image_threshold)) out << (score > image_threshold ? 1 : 0);
            out << '\n';
            if (want_masks) common::write_pnm(mask_dir + "/" + entry.image_id + "-mask.pgm", run_gpu ? gpu_mask : cpu_mask);
        }
        out.close();
        if (!out) throw std::runtime_error("Failed writing " + output + '.');

        const double total_ms = common::elapsed_ms(start, common::Clock::now());
        const bool agree = mismatches == 0;
        if (header) {
            std::cout << "split,backend,images,megapixels,decode_ms,cpu_ms,h2d_ms,gpu_compute_ms,d2h_ms,gpu_wall_ms,"
                         "total_ms,images_per_s,mismatches,agree\n";
        }
        std::cout << std::setprecision(6) << std::boolalpha << split << ',' << backend << ',' << entries.size() << ','
                  << pixels / 1.0e6 << ',' << decode_ms << ',';
        if (run_cpu) std::cout << cpu_ms;
        std::cout << ',';
        if (run_gpu) {
            std::cout << gpu_total.h2d_ms << ',' << gpu_total.compute_ms << ',' << gpu_total.d2h_ms << ',' << gpu_wall_ms;
        } else {
            std::cout << ",,,";
        }
        std::cout << ',' << total_ms << ',' << static_cast<double>(entries.size()) / (total_ms / 1000.0) << ','
                  << mismatches << ',' << agree << '\n';
        if (!agree) {
            std::cerr << mismatches << " images disagree. First: " << first_mismatch << '\n';
            return 1;
        }
        return 0;
    } catch (const std::invalid_argument& error) {
        std::cerr << "Error: " << error.what() << "\n\n" << usage;
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 2;
    }
}
