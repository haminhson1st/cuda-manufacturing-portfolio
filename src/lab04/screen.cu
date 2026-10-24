// Lab 04 application: screen one vibration recording and write a per-window CSV report.
// Prints one summary row with phase timings. You should not need to edit this file; your
// work is src/lab04/gpu_pipeline.cu plus the threshold study described in the tutorial.

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/bench.hpp"
#include "common/cli.hpp"
#include "common/cuda_utils.cuh"
#include "lab04/lab04.hpp"

namespace {

const char* const usage = R"(Usage: screen --input RECORDING.f32 --output REPORT.csv [options]
  --id NAME              recording id written to the report (default: input file name without extension)
  --window W             samples per window, 1..67108864 (default 2048)
  --gain G               calibration gain: y = G * x + B (default 1)
  --offset B             calibration offset (default 0)
  --rms-threshold T      alarm when window RMS > T (default: disabled)
  --peak-threshold T     alarm when window peak |y| > T (default: disabled)
  --crest-threshold T    alarm when peak / RMS > T (default: disabled)
  --backend B            cpu | cuda | both (both runs the two paths and checks agreement; default cuda)
  --threads T            threads per block for the GPU kernels: power of two in [32, 1024] (default 256)
  --header 0|1           print the CSV header of the summary line (default 1)
)";

std::string stem_of(const std::string& path) {
    const auto slash = path.find_last_of("/\\");
    const std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    const auto dot = name.find_last_of('.');
    return dot == std::string::npos || dot == 0 ? name : name.substr(0, dot);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto start = common::Clock::now();
        common::Args args(argc, argv);
        if (args.help()) {
            std::cout << usage;
            return 0;
        }
        const std::string input = args.text("--input", "");
        const std::string output = args.text("--output", "");
        std::string id = args.text("--id", "");
        lab04::PipelineConfig config;
        config.window = args.count("--window", 2048, 1, 67108864);
        config.gain = static_cast<float>(args.number("--gain", 1.0));
        config.offset = static_cast<float>(args.number("--offset", 0.0));
        lab04::AlarmRule rule;
        rule.rms = args.number("--rms-threshold", rule.rms);
        rule.peak = args.number("--peak-threshold", rule.peak);
        rule.crest = args.number("--crest-threshold", rule.crest);
        const std::string backend = args.choice("--backend", "cuda", {"cpu", "cuda", "both"});
        config.threads = static_cast<int>(args.count("--threads", 256, 32, 1024));
        const bool header = args.choice("--header", "1", {"0", "1"}) == "1";
        args.finish();
        if (input.empty() || output.empty()) throw std::invalid_argument("--input and --output are required.");
        if ((config.threads & (config.threads - 1)) != 0) throw std::invalid_argument("--threads must be a power of two.");
        if (id.empty()) id = stem_of(input);
        if (id.find_first_of(",\n\r\"") != std::string::npos) throw std::invalid_argument("--id must not contain commas or quotes.");

        auto phase = common::Clock::now();
        const std::vector<float> raw = lab04::read_f32(input);
        const double read_ms = common::elapsed_ms(phase, common::Clock::now());

        const bool run_cpu = backend != "cuda";
        const bool run_gpu = backend != "cpu";
        std::vector<lab03::Reference> cpu_stats;
        double cpu_ms = 0.0;
        if (run_cpu) {
            phase = common::Clock::now();
            cpu_stats = lab04::run_cpu_pipeline(raw, config);
            cpu_ms = common::elapsed_ms(phase, common::Clock::now());
        }
        std::vector<lab03::WindowStats> gpu_stats;
        lab04::GpuPhaseTimes gpu_times;
        double gpu_wall_ms = 0.0;
        if (run_gpu) {
            phase = common::Clock::now();  // includes device allocation and, on the first call, context setup
            gpu_stats = lab04::run_gpu_pipeline(raw, config, gpu_times);
            gpu_wall_ms = common::elapsed_ms(phase, common::Clock::now());
            if (gpu_stats.size() != lab03::window_count(raw.size(), config.window)) {
                throw std::logic_error("run_gpu_pipeline returned the wrong number of windows.");
            }
        }

        // Agreement: statistics within the Lab 03 error bound, then identical decisions except
        // where a feature lies so close to a threshold that rounding may flip it.
        std::size_t stat_failures = 0, decision_mismatches = 0, borderline = 0;
        std::string first_failure;
        if (run_cpu && run_gpu) {
            const auto check = lab03::validate(gpu_stats, cpu_stats, config.window, config.threads);
            stat_failures = check.failures;
            first_failure = check.first_failure;
            for (std::size_t w = 0; w < cpu_stats.size(); ++w) {
                const lab03::WindowStats& g = gpu_stats[w];
                const lab03::Reference& c = cpu_stats[w];
                if (lab04::alarm(g.rms, g.peak, rule) == lab04::alarm(c.rms, c.peak, rule)) continue;
                if (lab04::near_threshold(c.rms, c.peak, rule, 1e-5)) {
                    ++borderline;
                } else {
                    ++decision_mismatches;
                }
            }
        }

        std::vector<lab03::WindowStats> report;
        if (run_gpu) {
            report = gpu_stats;
        } else {
            for (const auto& r : cpu_stats) report.push_back(lab03::to_window_stats(r));
        }
        std::size_t alarms = 0;
        for (const auto& s : report) alarms += lab04::alarm(s.rms, s.peak, rule);
        phase = common::Clock::now();
        lab04::write_report(output, id, config.window, report, rule);
        const double report_ms = common::elapsed_ms(phase, common::Clock::now());
        const double total_ms = common::elapsed_ms(start, common::Clock::now());

        const bool agree = stat_failures == 0 && decision_mismatches == 0;
        if (header) {
            std::cout << "recording_id,backend,samples,window,windows,alarms,read_ms,cpu_ms,h2d_ms,gpu_compute_ms,"
                         "d2h_ms,gpu_wall_ms,report_ms,total_ms,stat_failures,decision_mismatches,borderline,agree\n";
        }
        std::cout << std::setprecision(6) << std::boolalpha << id << ',' << backend << ',' << raw.size() << ','
                  << config.window << ',' << report.size() << ',' << alarms << ',' << read_ms << ',';
        if (run_cpu) std::cout << cpu_ms;
        std::cout << ',';
        if (run_gpu) std::cout << gpu_times.h2d_ms << ',' << gpu_times.compute_ms << ',' << gpu_times.d2h_ms << ',' << gpu_wall_ms;
        else std::cout << ",,,";
        std::cout << ',' << report_ms << ',' << total_ms << ',' << stat_failures << ',' << decision_mismatches << ','
                  << borderline << ',' << agree << '\n';
        if (!agree) {
            if (!first_failure.empty()) std::cerr << "First failing " << first_failure << '\n';
            if (decision_mismatches != 0) std::cerr << decision_mismatches << " alarm decisions differ between CPU and GPU.\n";
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
