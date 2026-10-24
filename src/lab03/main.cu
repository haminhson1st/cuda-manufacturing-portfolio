// Lab 03 harness: generates a signal, times the double-precision CPU reference and your
// kernels, checks every window against an error bound, and prints one CSV row.
// You should not need to edit this file.

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/bench.hpp"
#include "common/cli.hpp"
#include "common/cuda_utils.cuh"
#include "lab03/lab03.hpp"

namespace {

const char* const usage = R"(Usage: window_stats [options]
  --samples N       signal length, 1..100000000 (default 1000000)
  --window W        samples per window, 1..67108864 (default 4096); the last window may be partial
  --pattern P       zeros | constant | alternating | impulse | ramp | vibration (default vibration)
  --mode M          fused | separate (default fused)
  --threads T       threads per block: a power of two in [32, 1024] (default 256)
  --reps N          timed repetitions (default 20)
  --warmups N       untimed warm-up runs (default 3)
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
        const std::size_t n = args.count("--samples", 1000000, 1, 100000000);
        const std::size_t window = args.count("--window", 4096, 1, 67108864);
        const std::string pattern =
            args.choice("--pattern", "vibration", {"zeros", "constant", "alternating", "impulse", "ramp", "vibration"});
        const std::string mode_name = args.choice("--mode", "fused", {"fused", "separate"});
        const int threads = static_cast<int>(args.count("--threads", 256, 32, 1024));
        const int reps = static_cast<int>(args.count("--reps", 20, 1, 100000));
        const int warmups = static_cast<int>(args.count("--warmups", 3, 0, 1000));
        const std::string raw_path = args.text("--raw", "");
        const bool header = args.choice("--header", "1", {"0", "1"}) == "1";
        args.finish();
        if ((threads & (threads - 1)) != 0) throw std::invalid_argument("--threads must be a power of two.");

        const auto mode = mode_name == "fused" ? lab03::Mode::fused : lab03::Mode::separate;
        const std::vector<float> x = lab03::make_signal(pattern, n);
        const std::size_t windows = lab03::window_count(n, window);

        std::vector<lab03::Reference> reference;
        const auto cpu_times = common::time_host(warmups, reps, [&] { lab03::reference_stats(x, window, reference); });

        common::DeviceBuffer<float> d_x(n);
        common::DeviceBuffer<lab03::WindowStats> d_out(windows);
        d_x.upload(x);
        d_out.fill_bytes(0xFF);  // NaN statistics and count 0xFFFFFFFF unless your kernel writes them

        const auto kernel_times = common::time_device(warmups, reps, [&] {
            lab03::launch_window_stats(d_x.get(), n, window, d_out.get(), threads, mode);
            CUDA_CHECK_LAUNCH();
        });
        std::vector<lab03::WindowStats> out;
        d_out.download(out);
        const auto check = lab03::validate(out, reference, window, threads);

        // Transfers + compute: H2D signal, kernels, D2H statistics (pageable host memory).
        const auto transfer_times = common::time_host(warmups, reps, [&] {
            d_x.upload(x);
            lab03::launch_window_stats(d_x.get(), n, window, d_out.get(), threads, mode);
            CUDA_CHECK_LAUNCH();
            d_out.download(out);
            CUDA_CHECK(cudaDeviceSynchronize());
        });

        if (!raw_path.empty()) {
            common::write_observations(raw_path, {{"cpu_ms", cpu_times}, {"kernel_ms", kernel_times},
                                                  {"transfer_compute_ms", transfer_times}});
        }
        const double kernel_ms = common::median(kernel_times);
        if (header) {
            std::cout << "lab,mode,pattern,samples,window,windows,threads,repetitions,cpu_ms,kernel_ms,"
                         "transfer_compute_ms,windows_per_s,max_mean_error,max_rms_rel_error,peak_mismatches,"
                         "failures,correct\n";
        }
        std::cout << std::setprecision(6) << std::boolalpha << "lab03," << mode_name << ',' << pattern << ',' << n
                  << ',' << window << ',' << windows << ',' << threads << ',' << reps << ','
                  << common::median(cpu_times) << ',' << kernel_ms << ',' << common::median(transfer_times) << ','
                  << static_cast<double>(windows) / (kernel_ms / 1000.0) << ',' << check.max_mean_error << ','
                  << check.max_rms_rel_error << ',' << check.peak_mismatches << ',' << check.failures << ','
                  << (check.failures == 0) << '\n';
        if (check.failures != 0) {
            std::cerr << "First failing " << check.first_failure << '\n';
            if (check.failures == windows && out[0].count == 0xFFFFFFFFu) {
                std::cerr << "No window was written. Implement the kernels in src/lab03/kernels.cu.\n";
            }
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
