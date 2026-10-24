// Lab 09 harness: decodes an image list once, then runs the Lab 07 baseline (lab07::score_gpu,
// fresh buffers per image) and/or your buffer-reusing lab09::Inspector over it several times.
// It checks every count against the CPU reference, checks how often the Inspector allocated, and
// prints one CSV row per variant. NVTX ranges label each phase on an Nsight Systems timeline.
// You should not need to edit this file.

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/bench.hpp"
#include "common/cli.hpp"
#include "common/cuda_utils.cuh"
#include "common/nvtx.hpp"
#include "lab09/lab09.hpp"

namespace {

const char* const usage = R"(Usage: profile_inspect --list IMAGES.csv [options]
  --split S               train | dev | test | all: rows of the list to process (default all)
  --variant V             baseline | reuse | both (default both)
  --repeat N              process the list N times, 1..1000 (default 3)
  --verify 0|1            check every count against the CPU reference (default 1; use 0 under a profiler)
  --lo L --hi H           normalization levels, 0 <= lo < hi <= 255 (default 0, 255)
  --radius R              smoothing radius, 0..16 (default 3)
  --pixel-threshold T     a residual above T marks the pixel, 0..254 (default 30)
  --block XxY             thread-block shape with a power-of-two thread count (default 16x16)
  --header 0|1            print the CSV header line (default 1)
)";

struct VariantResult {
    lab07::GpuPhaseTimes phases;
    double wall_ms = 0.0;
    std::size_t mismatches = 0;
    std::string first_mismatch;
    std::vector<unsigned long long> counts;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        common::Args args(argc, argv);
        if (args.help()) {
            std::cout << usage;
            return 0;
        }
        const std::string list = args.text("--list", "");
        const std::string split = args.choice("--split", "all", {"train", "dev", "test", "all"});
        const std::string variant = args.choice("--variant", "both", {"baseline", "reuse", "both"});
        const std::size_t repeat = args.count("--repeat", 3, 1, 1000);
        const bool verify = args.choice("--verify", "1", {"0", "1"}) == "1";
        lab07::DetectorParams params;
        params.lo = static_cast<int>(args.count("--lo", 0, 0, 255));
        params.hi = static_cast<int>(args.count("--hi", 255, 0, 255));
        params.radius = static_cast<int>(args.count("--radius", 3, 0, 16));
        params.pixel_threshold = static_cast<int>(args.count("--pixel-threshold", 30, 0, 254));
        unsigned bx = 0, by = 0;
        common::parse_block(args.text("--block", "16x16"), bx, by);
        params.block = dim3(bx, by);
        const bool header = args.choice("--header", "1", {"0", "1"}) == "1";
        args.finish();
        if (list.empty()) throw std::invalid_argument("--list is required.");
        if (params.lo >= params.hi) throw std::invalid_argument("--lo must be below --hi.");
        const unsigned threads = bx * by;
        if ((threads & (threads - 1)) != 0) throw std::invalid_argument("--block needs a power-of-two thread count.");

        // Decode once, outside the variants, so they compare only device-side work.
        std::vector<common::HostImage> images;
        double decode_ms = 0.0;
        {
            common::NvtxRange range("decode");
            const auto start = common::Clock::now();
            for (const auto& entry : lab07::read_image_list(list)) {
                if (split == "all" || entry.split == split) images.push_back(common::read_pnm(entry.path));
            }
            decode_ms = common::elapsed_ms(start, common::Clock::now());
        }
        if (images.empty()) throw std::runtime_error("No images in split '" + split + "' of " + list + '.');

        std::vector<unsigned long long> expected;
        if (verify) {
            common::NvtxRange range("cpu reference");
            for (const auto& image : images) expected.push_back(lab07::score_cpu(image, params, nullptr));
        }

        // The Inspector allocates once per change of shape in the processing order.
        std::size_t expected_allocations = 0;
        for (std::size_t r = 0; r < repeat; ++r) {
            for (std::size_t i = 0; i < images.size(); ++i) {
                const std::size_t previous = i > 0 ? i - 1 : images.size() - 1;
                const bool first = r == 0 && i == 0;
                const auto& a = images[i];
                const auto& b = images[previous];
                if (first || a.width != b.width || a.height != b.height || a.channels != b.channels) {
                    ++expected_allocations;
                }
            }
        }

        CUDA_CHECK(cudaFree(nullptr));  // create the context before either variant is timed
        const auto run = [&](const std::string& name, auto&& score_one) {
            common::NvtxRange range(name.c_str());
            VariantResult result;
            const auto start = common::Clock::now();
            for (std::size_t r = 0; r < repeat; ++r) {
                for (std::size_t i = 0; i < images.size(); ++i) {
                    common::NvtxRange image_range("image");
                    lab07::GpuPhaseTimes times;
                    const unsigned long long count = score_one(images[i], times);
                    result.phases.h2d_ms += times.h2d_ms;
                    result.phases.compute_ms += times.compute_ms;
                    result.phases.d2h_ms += times.d2h_ms;
                    if (r == 0) result.counts.push_back(count);
                    if (verify && count != expected[i] && result.mismatches++ == 0) {
                        result.first_mismatch = "image " + std::to_string(i) + ": expected " + std::to_string(expected[i])
                                                + " mask pixels, got " + std::to_string(count);
                    }
                }
            }
            result.wall_ms = common::elapsed_ms(start, common::Clock::now());
            return result;
        };

        if (header) {
            std::cout << "lab,variant,images,repeat,decode_ms,h2d_ms,compute_ms,d2h_ms,gpu_wall_ms,images_per_s,"
                         "allocations,expected_allocations,mismatches,correct\n";
        }
        bool all_correct = true;
        std::vector<unsigned long long> baseline_counts;
        const auto report = [&](const std::string& name, const VariantResult& result, std::size_t allocations,
                                std::size_t wanted_allocations) {
            const bool correct = result.mismatches == 0 && allocations == wanted_allocations;
            all_correct = all_correct && correct;
            const double processed = static_cast<double>(images.size() * repeat);
            std::cout << std::setprecision(6) << std::boolalpha << "lab09," << name << ',' << images.size() << ','
                      << repeat << ',' << decode_ms << ',' << result.phases.h2d_ms << ',' << result.phases.compute_ms
                      << ',' << result.phases.d2h_ms << ',' << result.wall_ms << ','
                      << processed / (result.wall_ms / 1000.0) << ',' << allocations << ',' << wanted_allocations << ','
                      << result.mismatches << ',' << correct << '\n';
            if (result.mismatches != 0) std::cerr << name << ": " << result.first_mismatch << '\n';
            if (allocations != wanted_allocations) {
                std::cerr << name << ": allocated " << allocations << " times; expected " << wanted_allocations
                          << " (once per change of image shape).\n";
            }
        };

        if (variant != "reuse") {
            const VariantResult baseline = run("baseline", [&](const common::HostImage& image, lab07::GpuPhaseTimes& times) {
                return lab07::score_gpu(image, params, times, nullptr);
            });
            baseline_counts = baseline.counts;
            // lab07::score_gpu allocates one set of buffers per call by design.
            report("baseline", baseline, images.size() * repeat, images.size() * repeat);
        }
        if (variant != "baseline") {
            lab09::Inspector inspector(params);
            const VariantResult reuse = run("reuse", [&](const common::HostImage& image, lab07::GpuPhaseTimes& times) {
                return inspector.score(image, times, nullptr);
            });
            report("reuse", reuse, inspector.allocations(), expected_allocations);
            if (!baseline_counts.empty() && baseline_counts != reuse.counts) {
                std::cerr << "reuse: counts differ from the baseline.\n";
                all_correct = false;
            }
        }
        return all_correct ? 0 : 1;
    } catch (const std::invalid_argument& error) {
        std::cerr << "Error: " << error.what() << "\n\n" << usage;
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 2;
    }
}
