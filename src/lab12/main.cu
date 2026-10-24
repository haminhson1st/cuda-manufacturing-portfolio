// Lab 12 harness: measures the frozen detector four ways on the same decoded images, verifies every
// count against the CPU reference, and writes raw.csv, summary.csv, and environment.json; then
// scripts/lab12_report.py turns them into report.md. You should not need to edit this file.

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/bench.hpp"
#include "common/cli.hpp"
#include "common/cuda_utils.cuh"
#include "common/json.hpp"
#include "common/nvtx.hpp"
#include "lab09/lab09.hpp"
#include "lab12/lab12.hpp"

namespace {

const char* const usage = R"(Usage: release_bench --list IMAGES.csv --output-dir DIR [options]
  --split S               rows of the list to use: train | dev | test | all (default all)
  --config FILE           detector values from a frozen configuration file (as for inspect)
  --variants LIST         comma-separated subset of cpu,cpu-parallel,cuda-baseline,cuda (default: all four)
  --sizes LIST            workload sizes in images, comma-separated; "all" = every selected image (default 1,all)
  --cpu-threads N         threads for cpu-parallel (default: SLURM_CPUS_PER_TASK, else all hardware threads)
  --repeats N             timed repetitions per variant and size (default 5)
  --warmups N             untimed repetitions before them (default 1)
  --lo L --hi H           normalization levels (default 0, 255)
  --radius R              smoothing radius, 0..16 (default 3)
  --pixel-threshold T     residual threshold, 0..254 (default 30)
  --image-threshold S     recorded with the results (not used for timing)
  --block XxY             thread-block shape with a power-of-two thread count (default 16x16)
  --header 0|1            print the CSV header line (default 1)
Writes DIR/raw.csv (every repetition), DIR/summary.csv, and DIR/environment.json.
)";

const std::vector<std::string> known_variants{"cpu", "cpu-parallel", "cuda-baseline", "cuda"};

std::vector<std::string> split_list(const std::string& text) {
    std::vector<std::string> items;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) items.push_back(item);
    return items;
}

// Nearest-rank quantile of a non-empty sample.
double quantile(std::vector<double> values, double p) {
    std::sort(values.begin(), values.end());
    const auto rank = static_cast<std::size_t>(std::ceil(p * static_cast<double>(values.size())));
    return values[std::min(values.size() - 1, rank == 0 ? 0 : rank - 1)];
}

std::string cpu_model() {
    std::ifstream info("/proc/cpuinfo");
    std::string line;
    while (std::getline(info, line)) {
        if (line.compare(0, 10, "model name") == 0) {
            const auto colon = line.find(':');
            const auto start = colon == std::string::npos ? colon : line.find_first_not_of(" \t", colon + 1);
            if (start != std::string::npos) return line.substr(start);
        }
    }
    return "unknown";
}

std::string host_name() {
    char name[256] = {};
    return gethostname(name, sizeof name - 1) == 0 ? name : "unknown";
}

std::string env_text(const char* name) {
    const char* value = std::getenv(name);
    return common::json_text(value != nullptr ? value : "");
}

std::string gpu_record() {
    int device = 0, runtime = 0, driver = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    return common::JsonObject()
        .add("name", common::json_text(prop.name))
        .add("compute_capability", common::json_text(std::to_string(prop.major) + '.' + std::to_string(prop.minor)))
        .add("memory_gib", common::json_number(static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0)))
        .add("cuda_runtime_version", std::to_string(runtime))
        .add("cuda_driver_version", std::to_string(driver))
        .str("    ");
}

std::string build_record() {
    std::string host = "unknown";
#if defined(__GNUC__) && !defined(__clang__)
    host = "gcc " + std::to_string(__GNUC__) + '.' + std::to_string(__GNUC_MINOR__);
#endif
    return common::JsonObject()
        .add("nvcc", common::json_text(std::to_string(__CUDACC_VER_MAJOR__) + '.' + std::to_string(__CUDACC_VER_MINOR__)))
        .add("host_compiler", common::json_text(host))
        .str("    ");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        common::Args args(argc, argv);
        if (args.help()) {
            std::cout << usage;
            return 0;
        }
        const std::string config_file = args.text("--config", "");
        if (!config_file.empty()) common::apply_config_file(args, config_file);
        const std::string list = args.text("--list", "");
        const std::string output_dir = args.text("--output-dir", "");
        const std::string split = args.choice("--split", "all", {"train", "dev", "test", "all"});
        const std::vector<std::string> variants = split_list(args.text("--variants", "cpu,cpu-parallel,cuda-baseline,cuda"));
        const std::vector<std::string> size_texts = split_list(args.text("--sizes", "1,all"));
        const int cpu_threads = static_cast<int>(args.count("--cpu-threads", lab12::default_cpu_threads(), 1, 1024));
        const int repeats = static_cast<int>(args.count("--repeats", 5, 1, 1000));
        const int warmups = static_cast<int>(args.count("--warmups", 1, 0, 100));
        lab07::DetectorParams params;
        params.lo = static_cast<int>(args.count("--lo", 0, 0, 255));
        params.hi = static_cast<int>(args.count("--hi", 255, 0, 255));
        params.radius = static_cast<int>(args.count("--radius", 3, 0, 16));
        params.pixel_threshold = static_cast<int>(args.count("--pixel-threshold", 30, 0, 254));
        const double image_threshold = args.number("--image-threshold", std::numeric_limits<double>::quiet_NaN());
        const std::string block = args.text("--block", "16x16");
        unsigned bx = 0, by = 0;
        common::parse_block(block, bx, by);
        params.block = dim3(bx, by);
        const bool header = args.choice("--header", "1", {"0", "1"}) == "1";
        args.finish();
        if (list.empty() || output_dir.empty()) throw std::invalid_argument("--list and --output-dir are required.");
        if (params.lo >= params.hi) throw std::invalid_argument("--lo must be below --hi.");
        if (((bx * by) & (bx * by - 1)) != 0) throw std::invalid_argument("--block needs a power-of-two thread count.");
        std::set<std::string> seen;
        bool uses_gpu = false;
        for (const auto& variant : variants) {
            if (std::find(known_variants.begin(), known_variants.end(), variant) == known_variants.end()) {
                throw std::invalid_argument("Unknown variant '" + variant + "' (use cpu, cpu-parallel, cuda-baseline, cuda).");
            }
            if (!seen.insert(variant).second) throw std::invalid_argument("Variant '" + variant + "' is listed twice.");
            uses_gpu = uses_gpu || variant.compare(0, 4, "cuda") == 0;
        }
        if (variants.empty()) throw std::invalid_argument("--variants is empty.");

        // Decode once; every variant then works on the same images in host memory.
        std::vector<common::HostImage> images;
        std::set<std::string> shapes;
        const auto decode_start = common::Clock::now();
        for (const auto& entry : lab07::read_image_list(list)) {
            if (split != "all" && entry.split != split) continue;
            images.push_back(common::read_pnm(entry.path));
            const auto& image = images.back();
            shapes.insert(std::to_string(image.width) + 'x' + std::to_string(image.height) + 'x' + std::to_string(image.channels));
        }
        const double decode_total_ms = common::elapsed_ms(decode_start, common::Clock::now());
        if (images.empty()) throw std::runtime_error("No images in split '" + split + "' of " + list + '.');
        const double decode_per_image_ms = decode_total_ms / static_cast<double>(images.size());

        std::vector<std::size_t> sizes;
        for (const auto& text : size_texts) {
            std::size_t size = images.size();
            if (text != "all") {
                if (text.empty() || text.size() > 7 || text.find_first_not_of("0123456789") != std::string::npos || text == "0") {
                    throw std::invalid_argument("--sizes takes positive integers or 'all' (got '" + text + "').");
                }
                size = std::min<std::size_t>(std::stoul(text), images.size());
            }
            if (std::find(sizes.begin(), sizes.end(), size) == sizes.end()) sizes.push_back(size);
        }
        if (sizes.empty()) throw std::invalid_argument("--sizes is empty.");

        std::vector<unsigned long long> expected;  // the single-thread CPU result is the reference
        for (const auto& image : images) expected.push_back(lab07::score_cpu(image, params, nullptr));
        if (uses_gpu) CUDA_CHECK(cudaFree(nullptr));  // create the context before anything is timed

        if (::mkdir(output_dir.c_str(), 0755) != 0 && errno != EEXIST) {
            throw std::runtime_error("Cannot create " + output_dir + ": " + std::strerror(errno));
        }
        std::ofstream raw(output_dir + "/raw.csv");
        std::ofstream summary(output_dir + "/summary.csv");
        if (!raw || !summary) throw std::runtime_error("Cannot write into " + output_dir + '.');
        const char* const columns =
            "variant,images,threads,repeats,median_ms,q1_ms,q3_ms,min_ms,max_ms,images_per_s,"
            "device_compute_median_ms,decode_ms,correct\n";
        raw << "variant,images,repetition,processing_ms,device_compute_ms\n" << std::setprecision(9);
        summary << columns << std::setprecision(9) << std::boolalpha;
        if (header) std::cout << columns;
        std::cout << std::setprecision(6) << std::boolalpha;

        bool all_correct = true;
        for (const std::size_t size : sizes) {
            const std::vector<common::HostImage> subset(images.begin(), images.begin() + static_cast<std::ptrdiff_t>(size));
            for (const std::string& variant : variants) {
                const std::string range_name = variant + " x" + std::to_string(size);
                common::NvtxRange range(range_name.c_str());
                const bool gpu = variant.compare(0, 4, "cuda") == 0;
                std::unique_ptr<lab09::Inspector> inspector;
                if (variant == "cuda") inspector = std::make_unique<lab09::Inspector>(params);
                std::vector<unsigned long long> counts;
                double device_ms = 0.0;
                const auto run_once = [&] {
                    counts.clear();
                    device_ms = 0.0;
                    if (variant == "cpu") {
                        for (const auto& image : subset) counts.push_back(lab07::score_cpu(image, params, nullptr));
                    } else if (variant == "cpu-parallel") {
                        counts = lab12::score_all_cpu_parallel(subset, params, cpu_threads);
                    } else {
                        for (const auto& image : subset) {
                            lab07::GpuPhaseTimes times;
                            counts.push_back(variant == "cuda" ? inspector->score(image, times, nullptr)
                                                               : lab07::score_gpu(image, params, times, nullptr));
                            device_ms += times.compute_ms;
                        }
                    }
                };
                for (int i = 0; i < warmups; ++i) run_once();
                std::vector<double> processing, device;
                bool correct = true;
                for (int r = 0; r < repeats; ++r) {
                    const auto start = common::Clock::now();
                    run_once();
                    processing.push_back(common::elapsed_ms(start, common::Clock::now()));
                    device.push_back(device_ms);
                    correct = correct && counts.size() == size && std::equal(counts.begin(), counts.end(), expected.begin());
                    raw << variant << ',' << size << ',' << r << ',' << processing.back() << ',';
                    if (gpu) raw << device_ms;
                    raw << '\n';
                }
                all_correct = all_correct && correct;
                const std::string threads = variant == "cpu" ? "1"
                                            : variant == "cpu-parallel"
                                                ? std::to_string(std::min<std::size_t>(static_cast<std::size_t>(cpu_threads), size))
                                                : "gpu";
                const double median_ms = common::median(processing);
                std::ostringstream row;
                row << std::setprecision(9) << std::boolalpha << variant << ',' << size << ',' << threads << ',' << repeats
                    << ',' << median_ms << ',' << quantile(processing, 0.25) << ',' << quantile(processing, 0.75) << ','
                    << *std::min_element(processing.begin(), processing.end()) << ','
                    << *std::max_element(processing.begin(), processing.end()) << ','
                    << static_cast<double>(size) / (median_ms / 1000.0) << ',';
                if (gpu) row << common::median(device);
                row << ',' << decode_per_image_ms * static_cast<double>(size) << ',' << correct << '\n';
                summary << row.str();
                std::cout << row.str();
                if (!correct) std::cerr << variant << " with " << size << " images: counts differ from the CPU reference.\n";
            }
        }

        std::vector<std::string> encoded_variants, encoded_sizes, encoded_shapes;
        for (const auto& variant : variants) encoded_variants.push_back(common::json_text(variant));
        for (const auto size : sizes) encoded_sizes.push_back(std::to_string(size));
        for (const auto& shape : shapes) encoded_shapes.push_back(common::json_text(shape));
        const char* commit = std::getenv("SOURCE_COMMIT");
        const std::string record =
            common::JsonObject()
                .add("tool", common::json_text("release_bench"))
                .add("created_utc", common::json_text(common::utc_now()))
                .add("source_commit", common::json_text(commit != nullptr ? commit : "unknown"))
                .add("host", common::json_text(host_name()))
                .add("slurm_job_id", env_text("SLURM_JOB_ID"))
                .add("slurm_partition", env_text("SLURM_JOB_PARTITION"))
                .add("cpu_model", common::json_text(cpu_model()))
                .add("hardware_threads", std::to_string(std::thread::hardware_concurrency()))
                .add("cpu_threads", std::to_string(cpu_threads))
                .add("gpu", uses_gpu ? gpu_record() : "null")
                .add("build", build_record())
                .add("list", common::json_text(list))
                .add("split", common::json_text(split))
                .add("images", std::to_string(images.size()))
                .add("image_shapes", common::json_array(encoded_shapes))
                .add("decode_ms_per_image", common::json_number(decode_per_image_ms))
                .add("config_file", common::json_text(config_file))
                .add("detector", common::JsonObject()
                                     .add("lo", std::to_string(params.lo))
                                     .add("hi", std::to_string(params.hi))
                                     .add("radius", std::to_string(params.radius))
                                     .add("pixel_threshold", std::to_string(params.pixel_threshold))
                                     .add("block", common::json_text(block))
                                     .str("    "))
                .add("image_threshold", common::json_number(image_threshold))
                .add("variants", common::json_array(encoded_variants))
                .add("sizes", common::json_array(encoded_sizes))
                .add("repeats", std::to_string(repeats))
                .add("warmups", std::to_string(warmups))
                .add("timing_boundary",
                     common::json_text("processing: decoded images in host memory -> mask counts on the host. Includes "
                                       "host-device transfers and, for cuda-baseline, per-image device allocation. "
                                       "Excludes decoding (decode_ms, measured once) and report writing."))
                .add("correct", all_correct ? "true" : "false")
                .str();
        std::ofstream environment(output_dir + "/environment.json");
        environment << record << '\n';
        raw.close();
        summary.close();
        if (!environment || !raw || !summary) throw std::runtime_error("Failed writing into " + output_dir + '.');
        return all_correct ? 0 : 1;
    } catch (const std::invalid_argument& error) {
        std::cerr << "Error: " << error.what() << "\n\n" << usage;
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 2;
    }
}
