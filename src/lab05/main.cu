// Lab 05 harness: builds a reference image and a test image with a known defect rectangle, runs
// the CPU and GPU pipelines, compares every stage byte for byte, checks that no kernel wrote into
// row padding, and prints one CSV row. You should not need to edit this file.

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/bench.hpp"
#include "common/cli.hpp"
#include "common/cuda_utils.cuh"
#include "common/device_image.cuh"
#include "lab05/lab05.hpp"

namespace {

const char* const usage = R"(Usage: image_ops [options]
  --width W --height H   image size in pixels, 1..16384 each (default 1920 x 1080)
  --channels C           1 (gray) or 3 (RGB, converted to gray first) (default 3)
  --pattern P            gradient | constant | checker | noise (default gradient)
  --host-pad B           extra bytes after each host row: a padded-stride input (default 0)
  --block XxY            thread-block shape, at most 1024 threads (default 16x16)
  --defect on|off        add a brightened rectangle to the test image (default on)
  --defect-delta D       brightness added inside the rectangle, 1..255 (default 80)
  --lo L --hi H          normalization levels, 0 <= lo < hi <= 255 (default 16, 235)
  --threshold T          mask where |test - reference| > T, 0..254 (default 40)
  --check-defect 0|1     also require the mask to equal the defect rectangle (default 0)
  --save PREFIX          write PREFIX-test.p?m, -reference, -diff.pgm, -mask.pgm
  --reps N --warmups N   timed repetitions and warm-up runs (default 20, 3)
  --raw FILE             also write every timed repetition to FILE
  --header 0|1           print the CSV header line (default 1)
)";

constexpr std::uint8_t guard = 0xA5;  // device rows are pre-filled with this, padding included

struct Rect {
    int x0, y0, x1, y1;  // half-open
    bool contains(int x, int y) const { return x >= x0 && x < x1 && y >= y0 && y < y1; }
};

}  // namespace

int main(int argc, char** argv) {
    try {
        common::Args args(argc, argv);
        if (args.help()) {
            std::cout << usage;
            return 0;
        }
        const int width = static_cast<int>(args.count("--width", 1920, 1, 16384));
        const int height = static_cast<int>(args.count("--height", 1080, 1, 16384));
        const int channels = static_cast<int>(args.count("--channels", 3, 1, 3));
        const std::string pattern = args.choice("--pattern", "gradient", {"gradient", "constant", "checker", "noise"});
        const std::size_t host_pad = args.count("--host-pad", 0, 0, 4096);
        unsigned bx = 0, by = 0;
        common::parse_block(args.text("--block", "16x16"), bx, by);
        const bool defect = args.choice("--defect", "on", {"on", "off"}) == "on";
        const int delta = static_cast<int>(args.count("--defect-delta", 80, 1, 255));
        const int lo = static_cast<int>(args.count("--lo", 16, 0, 255));
        const int hi = static_cast<int>(args.count("--hi", 235, 0, 255));
        const int threshold = static_cast<int>(args.count("--threshold", 40, 0, 254));
        const bool check_defect = args.choice("--check-defect", "0", {"0", "1"}) == "1";
        const std::string save = args.text("--save", "");
        const int reps = static_cast<int>(args.count("--reps", 20, 1, 100000));
        const int warmups = static_cast<int>(args.count("--warmups", 3, 0, 1000));
        const std::string raw_path = args.text("--raw", "");
        const bool header = args.choice("--header", "1", {"0", "1"}) == "1";
        args.finish();
        if (channels == 2) throw std::invalid_argument("--channels must be 1 or 3.");
        if (lo >= hi) throw std::invalid_argument("--lo must be below --hi.");
        if (check_defect && !defect) throw std::invalid_argument("--check-defect 1 needs --defect on.");
        const dim3 block(bx, by);

        // Reference ("golden") image and test image = reference + defect rectangle (saturating).
        const common::HostImage reference = common::make_pattern(pattern, width, height, channels, host_pad);
        common::HostImage test = reference;
        const Rect rect{width / 3, height / 3, width / 3 + std::max(1, width / 5), height / 3 + std::max(1, height / 5)};
        if (defect) {
            for (int y = rect.y0; y < rect.y1; ++y) {
                for (int x = rect.x0; x < rect.x1; ++x) {
                    for (int c = 0; c < channels; ++c) test.at(x, y, c) = static_cast<std::uint8_t>(std::min(255, test.at(x, y, c) + delta));
                }
            }
        }

        // CPU pipeline.
        common::HostImage cpu_gray_test(width, height, 1), cpu_gray_ref(width, height, 1);
        common::HostImage cpu_norm_test(width, height, 1), cpu_norm_ref(width, height, 1);
        common::HostImage cpu_diff(width, height, 1), cpu_mask(width, height, 1);
        const auto cpu_times = common::time_host(warmups, reps, [&] {
            const common::HostImage* t = &test;
            const common::HostImage* r = &reference;
            if (channels == 3) {
                lab05::rgb_to_gray_cpu(test, cpu_gray_test);
                lab05::rgb_to_gray_cpu(reference, cpu_gray_ref);
                t = &cpu_gray_test;
                r = &cpu_gray_ref;
            }
            lab05::normalize_cpu(*t, cpu_norm_test, lo, hi);
            lab05::normalize_cpu(*r, cpu_norm_ref, lo, hi);
            lab05::absdiff_mask_cpu(cpu_norm_test, cpu_norm_ref, cpu_diff, cpu_mask, threshold);
        });

        // GPU pipeline. Inputs and outputs are pitched device images; outputs start as `guard`.
        common::DeviceImage d_test(width, height, channels), d_ref(width, height, channels);
        common::DeviceImage d_gray_test(width, height, 1), d_gray_ref(width, height, 1);
        common::DeviceImage d_norm_test(width, height, 1), d_norm_ref(width, height, 1);
        common::DeviceImage d_diff(width, height, 1), d_mask(width, height, 1);
        common::DeviceImage* outputs[] = {&d_gray_test, &d_gray_ref, &d_norm_test, &d_norm_ref, &d_diff, &d_mask};
        for (auto* image : outputs) image->fill_all(guard);
        d_test.upload(test);
        d_ref.upload(reference);

        const auto run = [&] {
            common::ConstImageView t = d_test.cview();
            common::ConstImageView r = d_ref.cview();
            if (channels == 3) {
                lab05::launch_rgb_to_gray(d_test.cview(), d_gray_test.view(), block);
                CUDA_CHECK_LAUNCH();
                lab05::launch_rgb_to_gray(d_ref.cview(), d_gray_ref.view(), block);
                CUDA_CHECK_LAUNCH();
                t = d_gray_test.cview();
                r = d_gray_ref.cview();
            }
            lab05::launch_normalize(t, d_norm_test.view(), lo, hi, block);
            CUDA_CHECK_LAUNCH();
            lab05::launch_normalize(r, d_norm_ref.view(), lo, hi, block);
            CUDA_CHECK_LAUNCH();
            lab05::launch_absdiff_mask(d_norm_test.cview(), d_norm_ref.cview(), d_diff.view(), d_mask.view(), threshold,
                                       block);
            CUDA_CHECK_LAUNCH();
        };
        const auto kernel_times = common::time_device(warmups, reps, run);

        // Compare every stage so a failure names the first wrong kernel.
        std::size_t mismatches = 0;
        std::string first;
        const auto compare = [&](const char* stage, const common::HostImage& expected, const common::DeviceImage& device) {
            common::HostImage got(width, height, 1);
            device.download(got);
            std::string where;
            const std::size_t differences = common::count_differences(expected, got, where);
            if (differences != 0 && first.empty()) first = std::string(stage) + " at " + where;
            mismatches += differences;
        };
        if (channels == 3) compare("rgb_to_gray", cpu_gray_test, d_gray_test);
        compare("normalize", cpu_norm_test, d_norm_test);
        compare("normalize(reference)", cpu_norm_ref, d_norm_ref);
        compare("absdiff", cpu_diff, d_diff);
        compare("mask", cpu_mask, d_mask);
        std::size_t guard_violations = 0;
        for (auto* image : outputs) guard_violations += image->padding_changes(guard);

        // Transfers + compute: upload both inputs, run the pipeline, download diff and mask.
        common::HostImage host_diff(width, height, 1), host_mask(width, height, 1);
        const auto transfer_times = common::time_host(warmups, reps, [&] {
            d_test.upload(test);
            d_ref.upload(reference);
            run();
            d_diff.download(host_diff);
            d_mask.download(host_mask);
            CUDA_CHECK(cudaDeviceSynchronize());
        });

        std::size_t mask_pixels = 0;
        bool defect_match = defect;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const bool on = cpu_mask.at(x, y) != 0;
                mask_pixels += on;
                if (defect && on != rect.contains(x, y)) defect_match = false;
            }
        }
        if (!save.empty()) {
            const char* ext = channels == 3 ? ".ppm" : ".pgm";
            common::HostImage packed_test(width, height, channels), packed_ref(width, height, channels);
            for (int y = 0; y < height; ++y) {
                std::copy(test.row(y), test.row(y) + test.row_bytes(), packed_test.row(y));
                std::copy(reference.row(y), reference.row(y) + reference.row_bytes(), packed_ref.row(y));
            }
            common::write_pnm(save + "-test" + ext, packed_test);
            common::write_pnm(save + "-reference" + ext, packed_ref);
            common::write_pnm(save + "-diff.pgm", host_diff);
            common::write_pnm(save + "-mask.pgm", host_mask);
        }
        if (!raw_path.empty()) {
            common::write_observations(raw_path, {{"cpu_ms", cpu_times}, {"kernel_ms", kernel_times},
                                                  {"transfer_compute_ms", transfer_times}});
        }

        const bool correct = mismatches == 0 && guard_violations == 0 && (!check_defect || defect_match);
        const double kernel_ms = common::median(kernel_times);
        if (header) {
            std::cout << "lab,pattern,width,height,channels,host_pad,device_pitch,block,repetitions,cpu_ms,kernel_ms,"
                         "transfer_compute_ms,mpix_per_s,mask_pixels,defect_match,mismatches,guard_violations,correct\n";
        }
        std::cout << std::setprecision(6) << std::boolalpha << "lab05," << pattern << ',' << width << ',' << height << ','
                  << channels << ',' << host_pad << ',' << d_mask.pitch() << ',' << bx << 'x' << by << ',' << reps << ','
                  << common::median(cpu_times) << ',' << kernel_ms << ',' << common::median(transfer_times) << ','
                  << static_cast<double>(width) * height / (kernel_ms * 1000.0) << ',' << mask_pixels << ','
                  << defect_match << ',' << mismatches << ',' << guard_violations << ',' << correct << '\n';
        if (!correct) {
            if (!first.empty()) std::cerr << "First mismatch: " << first << '\n';
            if (guard_violations != 0) std::cerr << guard_violations << " padding bytes were written: guard x and y.\n";
            if (check_defect && !defect_match) std::cerr << "The mask does not equal the defect rectangle.\n";
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
