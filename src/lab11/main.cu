// Lab 11 harness: the `inspect` command line. It parses the subcommand, options, and an optional
// configuration file, calls your run_images / run_sensors, writes a run record (REPORT.csv.json)
// with the effective configuration and environment, and prints one summary line.
// You should not need to edit this file.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/bench.hpp"
#include "common/cli.hpp"
#include "common/cuda_utils.cuh"
#include "common/json.hpp"
#include "lab11/lab11.hpp"

namespace {

const char* const usage = R"(Usage:
  inspect images  --input DIR|LIST.csv|IMAGE.pgm --output REPORT.csv [options]
  inspect sensors --input RECORDING.f32 --output REPORT.csv [options]
  inspect images --help | inspect sensors --help

Common options:
  --config FILE           read "name = value" defaults from FILE; command-line options override it
  --backend B             cpu | cuda | both (both checks that CPU and GPU agree; default cuda)
  --header 0|1            print the CSV header of the summary line (default 1)
Every run also writes REPORT.csv.json: the effective configuration, environment, and summary.
)";

const char* const images_usage = R"(Images options (Labs 05-07 and 09):
  --split S               rows of a list input: train | dev | test | all (default all)
  --lo L --hi H           normalization levels, 0 <= lo < hi <= 255 (default 0, 255)
  --radius R              smoothing radius, 0..16 (default 3)
  --pixel-threshold T     a residual above T marks the pixel, 0..254 (default 30)
  --image-threshold S     predicted_defect = 1 when score > S (default: column left empty)
  --block XxY             thread-block shape with a power-of-two thread count (default 16x16)
  --save-masks DIR        write DIR/<image_id>-mask.pgm (DIR must exist)
)";

const char* const sensors_usage = R"(Sensors options (Labs 02-04):
  --id NAME               recording id (default: input file name without extension)
  --window W              samples per window, 1..67108864 (default 2048)
  --gain G --offset B     calibration y = G * x + B (default 1, 0)
  --rms-threshold T       alarm when window RMS > T (default: disabled)
  --peak-threshold T      alarm when window peak > T (default: disabled)
  --crest-threshold T     alarm when peak / RMS > T (default: disabled)
  --threads T             threads per block, a power of two in [32, 1024] (default 256)
)";

using common::json_number;
using common::json_text;
using common::JsonObject;

std::string gpu_record() {
    int device = 0, runtime = 0, driver = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    return JsonObject()
        .add("name", json_text(prop.name))
        .add("compute_capability", json_text(std::to_string(prop.major) + '.' + std::to_string(prop.minor)))
        .add("cuda_runtime_version", std::to_string(runtime))
        .add("cuda_driver_version", std::to_string(driver))
        .str("    ");
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const auto rank = static_cast<std::size_t>(std::ceil(p * static_cast<double>(values.size())));
    return values[std::min(values.size() - 1, rank == 0 ? 0 : rank - 1)];
}

}  // namespace

int main(int argc, char** argv) {
    std::string mode;
    try {
        if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
            std::cout << usage;
            return argc < 2 ? 2 : 0;
        }
        mode = argv[1];
        if (mode != "images" && mode != "sensors") throw std::invalid_argument("Unknown subcommand '" + mode + "'.");
        common::Args args(argc - 1, argv + 1);
        if (args.help()) {
            std::cout << usage << '\n' << (mode == "images" ? images_usage : sensors_usage);
            return 0;
        }
        const std::string config_file = args.text("--config", "");
        if (!config_file.empty()) common::apply_config_file(args, config_file);
        const std::string backend_text = args.choice("--backend", "cuda", {"cpu", "cuda", "both"});
        const auto backend = backend_text == "cpu" ? lab11::Backend::cpu
                             : backend_text == "cuda" ? lab11::Backend::cuda : lab11::Backend::both;
        const bool header = args.choice("--header", "1", {"0", "1"}) == "1";
        const std::string input = args.text("--input", "");
        const std::string output = args.text("--output", "");
        if (input.empty() || output.empty()) throw std::invalid_argument("--input and --output are required.");

        JsonObject settings;  // the effective configuration, recorded with the report
        lab11::ImagesConfig images;
        lab11::SensorsConfig sensors;
        if (mode == "images") {
            images.input = input;
            images.output = output;
            images.backend = backend;
            images.split = args.choice("--split", "all", {"train", "dev", "test", "all"});
            images.detector.lo = static_cast<int>(args.count("--lo", 0, 0, 255));
            images.detector.hi = static_cast<int>(args.count("--hi", 255, 0, 255));
            images.detector.radius = static_cast<int>(args.count("--radius", 3, 0, 16));
            images.detector.pixel_threshold = static_cast<int>(args.count("--pixel-threshold", 30, 0, 254));
            images.image_threshold = args.number("--image-threshold", images.image_threshold);
            const std::string block = args.text("--block", "16x16");
            unsigned bx = 0, by = 0;
            common::parse_block(block, bx, by);
            images.detector.block = dim3(bx, by);
            images.mask_dir = args.text("--save-masks", "");
            args.finish();
            if (images.detector.lo >= images.detector.hi) throw std::invalid_argument("--lo must be below --hi.");
            if (((bx * by) & (bx * by - 1)) != 0) throw std::invalid_argument("--block needs a power-of-two thread count.");
            settings.add("split", json_text(images.split))
                .add("lo", std::to_string(images.detector.lo))
                .add("hi", std::to_string(images.detector.hi))
                .add("radius", std::to_string(images.detector.radius))
                .add("pixel_threshold", std::to_string(images.detector.pixel_threshold))
                .add("image_threshold", json_number(images.image_threshold))
                .add("block", json_text(block))
                .add("save_masks", json_text(images.mask_dir));
        } else {
            sensors.input = input;
            sensors.output = output;
            sensors.backend = backend;
            sensors.id = args.text("--id", lab11::stem(input));
            sensors.pipeline.window = args.count("--window", 2048, 1, 67108864);
            sensors.pipeline.gain = static_cast<float>(args.number("--gain", 1.0));
            sensors.pipeline.offset = static_cast<float>(args.number("--offset", 0.0));
            sensors.rule.rms = args.number("--rms-threshold", sensors.rule.rms);
            sensors.rule.peak = args.number("--peak-threshold", sensors.rule.peak);
            sensors.rule.crest = args.number("--crest-threshold", sensors.rule.crest);
            sensors.pipeline.threads = static_cast<int>(args.count("--threads", 256, 32, 1024));
            args.finish();
            if ((sensors.pipeline.threads & (sensors.pipeline.threads - 1)) != 0) {
                throw std::invalid_argument("--threads must be a power of two.");
            }
            if (sensors.id.empty() || sensors.id.find_first_of(",\n\r\"") != std::string::npos) {
                throw std::invalid_argument("--id must be non-empty and contain no commas or quotes.");
            }
            settings.add("id", json_text(sensors.id))
                .add("window", std::to_string(sensors.pipeline.window))
                .add("gain", json_number(sensors.pipeline.gain))
                .add("offset", json_number(sensors.pipeline.offset))
                .add("rms_threshold", json_number(sensors.rule.rms))
                .add("peak_threshold", json_number(sensors.rule.peak))
                .add("crest_threshold", json_number(sensors.rule.crest))
                .add("threads", std::to_string(sensors.pipeline.threads));
        }

        const auto start = common::Clock::now();
        const lab11::RunSummary summary = mode == "images" ? lab11::run_images(images) : lab11::run_sensors(sensors);
        const double total_ms = common::elapsed_ms(start, common::Clock::now());
        const bool ok = summary.disagreements == 0;

        const char* commit = std::getenv("SOURCE_COMMIT");
        JsonObject record;
        record.add("tool", json_text("inspect"))
            .add("mode", json_text(mode))
            .add("created_utc", json_text(common::utc_now()))
            .add("source_commit", json_text(commit != nullptr ? commit : "unknown"))
            .add("config_file", json_text(config_file))
            .add("input", json_text(input))
            .add("output", json_text(output))
            .add("backend", json_text(backend_text))
            .add("settings", settings.str("    "))
            .add("gpu", backend == lab11::Backend::cpu ? "null" : gpu_record())
            .add("summary", JsonObject()
                                .add("items", std::to_string(summary.items))
                                .add("flagged", std::to_string(summary.flagged))
                                .add("disagreements", std::to_string(summary.disagreements))
                                .add("decode_ms", json_number(summary.decode_ms))
                                .add("compute_ms", json_number(summary.compute_ms))
                                .add("total_ms", json_number(total_ms))
                                .str("    "));
        std::ofstream json(output + ".json");
        json << record.str() << '\n';
        if (!json) throw std::runtime_error("Failed writing " + output + ".json.");

        if (header) {
            std::cout << "mode,backend,items,flagged,decode_ms,compute_ms,total_ms,items_per_s,item_p50_ms,item_p95_ms,"
                         "disagreements,ok\n";
        }
        std::cout << std::setprecision(6) << std::boolalpha << mode << ',' << backend_text << ',' << summary.items << ','
                  << summary.flagged << ',' << summary.decode_ms << ',' << summary.compute_ms << ',' << total_ms << ','
                  << static_cast<double>(summary.items) / (total_ms / 1000.0) << ','
                  << percentile(summary.item_ms, 0.5) << ',' << percentile(summary.item_ms, 0.95) << ','
                  << summary.disagreements << ',' << ok << '\n';
        if (!ok) std::cerr << summary.disagreements << " CPU/GPU disagreements; see the report and the Lab 04/07 checks.\n";
        return ok ? 0 : 1;
    } catch (const std::invalid_argument& error) {
        std::cerr << "Error: " << error.what() << "\n\n" << usage;
        if (mode == "images") std::cerr << '\n' << images_usage;
        if (mode == "sensors") std::cerr << '\n' << sensors_usage;
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 2;
    }
}
