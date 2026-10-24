// Lab 08 harness: builds a deterministic mask, uploads it once, then measures each backend's
// setup cost, first call, and steady state, checking every result against the CPU count.
// You should not need to edit this file.

#include <thrust/version.h>

#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/bench.hpp"
#include "common/cli.hpp"
#include "common/cuda_utils.cuh"
#include "lab08/lab08.hpp"

namespace {

const char* const usage = R"(Usage: reduce_compare [options]
  --bytes N         mask length in bytes, 1..2147483647 (default 100000000)
  --pattern P       zeros | ones | sparse | dense | random | values (default random)
  --backend B       custom | thrust | cub | all (default all); run one backend per process for
                    clean first-call times
  --threads T       threads per block for the custom kernel: a power of two in [32, 1024] (default 256)
  --reps N          timed repetitions (default 20)
  --warmups N       untimed runs between the first call and the timed ones (default 3)
  --cpu-reps N      timed CPU repetitions (default 3)
  --raw FILE        also write every timed repetition to FILE
  --header 0|1      print the CSV header line (default 1)
)";

}  // namespace

int main(int argc, char** argv) {
    try {
        common::Args args(argc, argv);
        if (args.help()) {
            std::cout << usage;
            return 0;
        }
        const std::size_t n = args.count("--bytes", 100000000, 1, 2147483647);
        const std::string pattern =
            args.choice("--pattern", "random", {"zeros", "ones", "sparse", "dense", "random", "values"});
        const std::string backend_name = args.choice("--backend", "all", {"custom", "thrust", "cub", "all"});
        const int threads = static_cast<int>(args.count("--threads", 256, 32, 1024));
        const int reps = static_cast<int>(args.count("--reps", 20, 1, 100000));
        const int warmups = static_cast<int>(args.count("--warmups", 3, 0, 1000));
        const int cpu_reps = static_cast<int>(args.count("--cpu-reps", 3, 1, 1000));
        const std::string raw_path = args.text("--raw", "");
        const bool header = args.choice("--header", "1", {"0", "1"}) == "1";
        args.finish();
        if ((threads & (threads - 1)) != 0) throw std::invalid_argument("--threads must be a power of two.");

        const std::vector<std::uint8_t> mask = lab08::make_mask(lab08::parse_pattern(pattern), n);
        unsigned long long expected = 0;
        const auto cpu_times = common::time_host(0, cpu_reps, [&] { expected = lab08::count_cpu(mask); });

        CUDA_CHECK(cudaFree(nullptr));  // create the context now, so no backend's first call pays for it
        common::DeviceBuffer<std::uint8_t> d_mask(n);
        const auto upload_start = common::Clock::now();
        d_mask.upload(mask);
        const double h2d_ms = common::elapsed_ms(upload_start, common::Clock::now());
        common::DeviceBuffer<unsigned long long> d_count(1);
        const auto read_count = [&d_count] {
            std::vector<unsigned long long> host;
            d_count.download(host);
            return host[0];
        };

        const std::vector<std::string> backends = backend_name == "all"
                                                      ? std::vector<std::string>{"custom", "thrust", "cub"}
                                                      : std::vector<std::string>{backend_name};
        const std::string thrust_version = std::to_string(THRUST_MAJOR_VERSION) + '.'
                                           + std::to_string(THRUST_MINOR_VERSION) + '.'
                                           + std::to_string(THRUST_SUBMINOR_VERSION);
        if (header) {
            std::cout << "lab,backend,pattern,bytes,threads,repetitions,temp_bytes,prepare_ms,first_ms,steady_ms,"
                         "h2d_ms,cpu_ms,gbps,count,expected,correct,thrust_version\n";
        }
        common::Series series{{"cpu_ms", cpu_times}};
        bool all_correct = true;
        for (const std::string& backend : backends) {
            std::size_t temp_bytes = 0;
            double prepare_ms = 0.0;
            common::DeviceBuffer<std::uint8_t> d_temp;
            std::function<unsigned long long()> run;
            if (backend == "custom") {
                run = [&] {
                    lab08::count_custom(d_mask.get(), n, d_count.get(), threads);
                    CUDA_CHECK_LAUNCH();
                    return read_count();
                };
            } else if (backend == "thrust") {
                run = [&] { return lab08::count_thrust(d_mask.get(), n); };
            } else {
                // Setup outside the timed loop: ask CUB for its storage size, then allocate it once.
                const auto prepare_start = common::Clock::now();
                temp_bytes = lab08::cub_temp_bytes(d_mask.get(), n);
                // CUB reads a null storage pointer as a size query, so never hand it one here.
                d_temp = common::DeviceBuffer<std::uint8_t>(std::max<std::size_t>(temp_bytes, 1));
                prepare_ms = common::elapsed_ms(prepare_start, common::Clock::now());
                run = [&] {
                    lab08::count_cub(d_temp.get(), temp_bytes, d_mask.get(), n, d_count.get());
                    CUDA_CHECK_LAUNCH();
                    return read_count();
                };
            }
            d_count.fill_bytes(0xFF);  // a launcher that forgets to zero the counter returns garbage
            CUDA_CHECK(cudaDeviceSynchronize());
            const auto first_start = common::Clock::now();
            const unsigned long long first = run();  // includes loading this backend's kernels
            const double first_ms = common::elapsed_ms(first_start, common::Clock::now());
            unsigned long long last = first;
            const auto times = common::time_host(warmups, reps, [&] { last = run(); });
            const bool correct = first == expected && last == expected;
            all_correct = all_correct && correct;
            series.push_back({backend + "_ms", times});

            // Every timing boundary: mask already on the device -> count available on the host.
            const double steady_ms = common::median(times);
            std::cout << std::setprecision(6) << std::boolalpha << "lab08," << backend << ',' << pattern << ',' << n
                      << ',' << threads << ',' << reps << ',';
            if (backend != "thrust") std::cout << temp_bytes;  // Thrust's storage is internal and unreported
            std::cout << ',' << prepare_ms << ',' << first_ms << ',' << steady_ms << ',' << h2d_ms << ','
                      << common::median(cpu_times) << ',' << static_cast<double>(n) / (steady_ms * 1.0e6) << ','
                      << last << ',' << expected << ',' << correct << ',' << thrust_version << '\n';
            if (!correct) {
                std::cerr << backend << ": expected " << expected << ", got " << first << " on the first call and "
                          << last << " on the last.\n";
            }
        }
        if (!raw_path.empty()) common::write_observations(raw_path, series);
        return all_correct ? 0 : 1;
    } catch (const std::invalid_argument& error) {
        std::cerr << "Error: " << error.what() << "\n\n" << usage;
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 2;
    }
}
