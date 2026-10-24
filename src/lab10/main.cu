// Lab 10 harness: builds a recorded item sequence, runs the serial baseline and/or your streamed
// pipeline, checks the baseline against the CPU reference and your results against the baseline
// bit for bit, and prints one CSV row per mode with throughput and per-item latency.
// You should not need to edit this file.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/bench.hpp"
#include "common/cli.hpp"
#include "common/cuda_utils.cuh"
#include "lab10/lab10.hpp"

namespace {

const char* const usage = R"(Usage: stream_batches [options]
  --items N           items in the recorded sequence, 1..100000 (default 256)
  --item-samples S    samples per item, a multiple of --window (default 65536)
  --window W          samples per statistics window (default 1024)
  --batch B           items per batch, 1..4096 (default 16)
  --mode M            serial | streams | both (default both)
  --interval-ms T     simulated arrival interval per item; 0 = offline, all items available (default 0)
  --threads T         threads per block: a power of two in [32, 1024] (default 256)
  --header 0|1        print the CSV header line (default 1)
)";

// Every item appears in exactly one batch, in order, and no batch exceeds the batch size.
std::string coverage_problem(const lab10::RunResult& result, std::size_t items, std::size_t batch) {
    std::size_t next = 0;
    for (const auto& record : result.batches) {
        if (record.first_item != next) return "batch starting at item " + std::to_string(record.first_item)
                                              + " where item " + std::to_string(next) + " was expected";
        if (record.count == 0 || record.count > batch) return "batch with " + std::to_string(record.count) + " items";
        next += record.count;
    }
    return next == items ? "" : "batches cover " + std::to_string(next) + " of " + std::to_string(items) + " items";
}

double percentile(std::vector<double> values, double p) {
    std::sort(values.begin(), values.end());
    const std::size_t index = static_cast<std::size_t>(std::ceil(p * static_cast<double>(values.size()))) - 1;
    return values[std::min(index, values.size() - 1)];
}

}  // namespace

int main(int argc, char** argv) {
    try {
        common::Args args(argc, argv);
        if (args.help()) {
            std::cout << usage;
            return 0;
        }
        lab10::Workload workload;
        workload.items = args.count("--items", 256, 1, 100000);
        workload.item_samples = args.count("--item-samples", 65536, 1, 16777216);
        workload.window = args.count("--window", 1024, 1, 16777216);
        const std::size_t batch = args.count("--batch", 16, 1, 4096);
        const std::string mode = args.choice("--mode", "both", {"serial", "streams", "both"});
        const double interval_ms = args.number("--interval-ms", 0.0);
        workload.threads = static_cast<int>(args.count("--threads", 256, 32, 1024));
        const bool header = args.choice("--header", "1", {"0", "1"}) == "1";
        args.finish();
        if (workload.item_samples % workload.window != 0) {
            throw std::invalid_argument("--item-samples must be a multiple of --window.");
        }
        if ((workload.threads & (workload.threads - 1)) != 0) throw std::invalid_argument("--threads must be a power of two.");
        if (interval_ms < 0.0) throw std::invalid_argument("--interval-ms must not be negative.");
        if (workload.items * workload.item_samples > 400000000) {
            throw std::invalid_argument("items * item-samples must be <= 400000000.");
        }

        lab10::make_source(workload);
        const std::vector<lab03::Reference> reference = lab10::reference(workload);
        CUDA_CHECK(cudaFree(nullptr));  // create the context before anything is timed

        if (header) {
            std::cout << "lab,mode,items,item_samples,window,batch,interval_ms,batches,total_ms,items_per_s,"
                         "latency_median_ms,latency_p95_ms,latency_max_ms,identical_to_serial,correct\n";
        }
        bool all_correct = true;
        lab10::RunResult serial;
        const auto report = [&](const std::string& name, const lab10::RunResult& result, bool correct,
                                const std::string& identical) {
            std::vector<double> latency;
            for (const auto& record : result.batches) {
                for (std::size_t item = record.first_item; item < record.first_item + record.count; ++item) {
                    latency.push_back(record.done_ms - lab10::arrival_ms(item, interval_ms));
                }
            }
            std::cout << std::setprecision(6) << std::boolalpha << "lab10," << name << ',' << workload.items << ','
                      << workload.item_samples << ',' << workload.window << ',' << batch << ',' << interval_ms << ','
                      << result.batches.size() << ',' << result.total_ms << ','
                      << static_cast<double>(workload.items) / (result.total_ms / 1000.0) << ','
                      << percentile(latency, 0.5) << ',' << percentile(latency, 0.95) << ','
                      << percentile(latency, 1.0) << ',' << identical << ',' << correct << '\n';
            all_correct = all_correct && correct;
        };

        // The baseline runs in either mode: it is the reference for the streamed results.
        serial = lab10::run_serial(workload, batch, interval_ms);
        const auto serial_check = lab03::validate(serial.stats, reference, workload.window, workload.threads);
        const std::string serial_coverage = coverage_problem(serial, workload.items, batch);
        if (serial_check.failures != 0 || !serial_coverage.empty()) {
            std::cerr << "serial baseline: " << (serial_coverage.empty() ? serial_check.first_failure : serial_coverage)
                      << "\nThe baseline uses your Lab 02 and Lab 03 kernels; make those labs pass first.\n";
        }
        if (mode != "streams") report("serial", serial, serial_check.failures == 0 && serial_coverage.empty(), "");

        if (mode != "serial") {
            const lab10::RunResult streamed = lab10::run_streamed(workload, batch, interval_ms);
            const std::string coverage = coverage_problem(streamed, workload.items, batch);
            const bool identical = streamed.stats.size() == serial.stats.size()
                                   && std::memcmp(streamed.stats.data(), serial.stats.data(),
                                                  serial.stats.size() * sizeof(lab03::WindowStats)) == 0;
            if (!coverage.empty()) std::cerr << "streams: " << coverage << '\n';
            if (!identical) {
                for (std::size_t w = 0; w < std::min(streamed.stats.size(), serial.stats.size()); ++w) {
                    if (std::memcmp(&streamed.stats[w], &serial.stats[w], sizeof(lab03::WindowStats)) != 0) {
                        std::cerr << "streams: first difference from the serial baseline in window " << w << " (item "
                                  << w / workload.windows_per_item() << ").\n";
                        break;
                    }
                }
            }
            const bool correct = identical && coverage.empty() && serial_check.failures == 0;
            report("streams", streamed, correct, identical ? "true" : "false");
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
