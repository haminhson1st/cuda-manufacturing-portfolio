// Lab 02 harness: builds a multi-channel batch, times the CPU baseline and your kernel, checks
// every reading and alarm, and prints one CSV row. You should not need to edit this file.

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/bench.hpp"
#include "common/cli.hpp"
#include "common/cuda_utils.cuh"
#include "lab02/lab02.hpp"

namespace {

const char* const usage = R"(Usage: calibrate_channels [options]
  --channels C      sensor channels, 1..65536 (default 32)
  --samples S       samples per channel (default 100003); channels * samples <= 100000000
  --layout L        channel | sample: memory order of the batch (default channel)
  --threads T       threads per block, a multiple of 32 in [32, 1024] (default 256)
  --blocks B        fixed number of blocks; 0 = one thread per element (default 0)
  --low X           inclusive lower limit (default -100)
  --high Y          inclusive upper limit (default 100)
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
        const std::size_t channels = args.count("--channels", 32, 1, 65536);
        const std::size_t samples = args.count("--samples", 100003, 1, 100000000);
        const std::string layout_name = args.choice("--layout", "channel", {"channel", "sample"});
        const int threads = static_cast<int>(args.count("--threads", 256, 32, 1024));
        const int blocks = static_cast<int>(args.count("--blocks", 0, 0, 2147483647));
        const lab02::Limits limits{static_cast<float>(args.number("--low", -100.0)),
                                   static_cast<float>(args.number("--high", 100.0))};
        const int reps = static_cast<int>(args.count("--reps", 20, 1, 100000));
        const int warmups = static_cast<int>(args.count("--warmups", 3, 0, 1000));
        const std::string raw_path = args.text("--raw", "");
        const bool header = args.choice("--header", "1", {"0", "1"}) == "1";
        args.finish();
        if (threads % 32 != 0) throw std::invalid_argument("--threads must be a multiple of 32 (a whole number of warps).");
        if (channels * samples > 100000000) throw std::invalid_argument("channels * samples must be <= 100000000.");
        if (!(limits.low < limits.high)) throw std::invalid_argument("--low must be below --high.");

        const auto layout = layout_name == "channel" ? lab02::Layout::channel_major : lab02::Layout::sample_major;
        const std::size_t n = channels * samples;

        // Inputs arrive interleaved. A channel-major batch costs a host transpose, timed on its own.
        const std::vector<float> interleaved = lab02::make_interleaved(channels, samples, limits);
        std::vector<float> raw;
        std::vector<double> convert_times;
        if (layout == lab02::Layout::channel_major) {
            raw.resize(n);
            convert_times = common::time_host(warmups, reps, [&] {
                lab02::interleaved_to_channel_major(interleaved, raw, channels, samples);
            });
        } else {
            raw = interleaved;
        }
        std::vector<float> gain(channels);
        std::vector<float> offset(channels);
        for (std::size_t c = 0; c < channels; ++c) {
            gain[c] = lab02::channel_gain(c);
            offset[c] = lab02::channel_offset(c);
        }

        // CPU baseline, in the same storage order as the GPU batch.
        std::vector<float> cpu_out(n);
        std::vector<std::uint8_t> cpu_alarm(n);
        const auto cpu_times = common::time_host(warmups, reps, [&] {
            lab02::calibrate_cpu(raw, gain, offset, cpu_out, cpu_alarm, channels, samples, layout, limits);
        });
        const auto cpu_check = lab02::validate(cpu_out, cpu_alarm, channels, samples, layout, limits);
        if (cpu_check.mismatches != 0) throw std::logic_error("CPU baseline failed: " + cpu_check.first_mismatch);

        common::DeviceBuffer<float> d_raw(n), d_gain(channels), d_offset(channels), d_out(n);
        common::DeviceBuffer<std::uint8_t> d_alarm(n);
        d_raw.upload(raw);
        d_gain.upload(gain);
        d_offset.upload(offset);
        // 0xFF bytes read back as NaN floats and alarm 255, so elements your kernel skips fail validation.
        d_out.fill_bytes(0xFF);
        d_alarm.fill_bytes(0xFF);

        const lab02::Batch batch{d_raw.get(), d_gain.get(), d_offset.get(), d_out.get(), d_alarm.get(),
                                 channels, samples, layout};
        const lab02::LaunchConfig config{threads, blocks};
        const auto kernel_times = common::time_device(warmups, reps, [&] {
            lab02::launch_calibrate(batch, limits, config);
            CUDA_CHECK_LAUNCH();
        });
        std::vector<float> out;
        std::vector<std::uint8_t> alarm;
        d_out.download(out);
        d_alarm.download(alarm);
        const auto check = lab02::validate(out, alarm, channels, samples, layout, limits);

        // Transfers + compute: H2D raw, kernel, D2H values and alarms. Pageable host memory;
        // excludes allocation, input generation, and the layout conversion above.
        const auto transfer_times = common::time_host(warmups, reps, [&] {
            d_raw.upload(raw);
            lab02::launch_calibrate(batch, limits, config);
            CUDA_CHECK_LAUNCH();
            d_out.download(out);
            d_alarm.download(alarm);
            CUDA_CHECK(cudaDeviceSynchronize());
        });

        if (!raw_path.empty()) {
            common::write_observations(raw_path, {{"convert_ms", convert_times}, {"cpu_ms", cpu_times},
                                                  {"kernel_ms", kernel_times},
                                                  {"transfer_compute_ms", transfer_times}});
        }
        // Logical bytes per element: 4 read + 4 written + 1 alarm byte (gain/offset are negligible).
        const double kernel_ms = common::median(kernel_times);
        const double gbps = static_cast<double>(n) * 9.0 / (kernel_ms * 1.0e6);
        if (header) {
            std::cout << "lab,layout,channels,samples,threads,blocks,repetitions,convert_ms,cpu_ms,kernel_ms,"
                         "transfer_compute_ms,kernel_gbps,alarms_low,alarms_high,mismatches,correct\n";
        }
        std::cout << std::setprecision(6) << std::boolalpha << "lab02," << layout_name << ',' << channels << ','
                  << samples << ',' << threads << ',' << blocks << ',' << reps << ',';
        if (!convert_times.empty()) std::cout << common::median(convert_times);
        std::cout << ',' << common::median(cpu_times) << ',' << kernel_ms << ',' << common::median(transfer_times)
                  << ',' << gbps << ',' << check.alarms_low << ',' << check.alarms_high << ',' << check.mismatches
                  << ',' << (check.mismatches == 0) << '\n';
        if (check.mismatches != 0) {
            std::cerr << "First mismatch: " << check.first_mismatch << '\n';
            if (check.mismatches == n) {
                std::cerr << "No element was written. Implement launch_calibrate in src/lab02/kernels.cu.\n";
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
