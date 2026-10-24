// Lab 06 harness: filters a synthetic image with your direct or tiled kernel, compares every pixel
// with the CPU reference, checks row padding, and prints one CSV row with the memory-traffic
// figures needed for the direct-versus-tiled discussion. You should not need to edit this file.

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/bench.hpp"
#include "common/cli.hpp"
#include "common/cuda_utils.cuh"
#include "common/device_image.cuh"
#include "lab06/lab06.hpp"

namespace {

const char* const usage = R"(Usage: box_filter [options]
  --width W --height H   image size in pixels, 1..16384 each (default 1920 x 1080)
  --radius R             filter radius, 0..32: a (2R+1) x (2R+1) box (default 1)
  --variant V            direct | tiled (default direct)
  --pattern P            noise | impulse | corner | constant | gradient | checker (default noise)
  --block XxY            thread-block shape, at most 1024 threads (default 16x16)
  --host-pad B           extra bytes after each host row (default 0)
  --reps N --warmups N   timed GPU repetitions and warm-up runs (default 20, 3)
  --cpu-reps N           timed CPU repetitions; the CPU reference is slow for large R (default 3)
  --raw FILE             also write every timed repetition to FILE
  --header 0|1           print the CSV header line (default 1)
)";

constexpr std::uint8_t guard = 0xA5;

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
        const int radius = static_cast<int>(args.count("--radius", 1, 0, 32));
        const std::string variant = args.choice("--variant", "direct", {"direct", "tiled"});
        const std::string pattern =
            args.choice("--pattern", "noise", {"noise", "impulse", "corner", "constant", "gradient", "checker"});
        unsigned bx = 0, by = 0;
        const std::string block_text = args.text("--block", "16x16");
        common::parse_block(block_text, bx, by);
        const std::size_t host_pad = args.count("--host-pad", 0, 0, 4096);
        const int reps = static_cast<int>(args.count("--reps", 20, 1, 100000));
        const int warmups = static_cast<int>(args.count("--warmups", 3, 0, 1000));
        const int cpu_reps = static_cast<int>(args.count("--cpu-reps", 3, 1, 1000));
        const std::string raw_path = args.text("--raw", "");
        const bool header = args.choice("--header", "1", {"0", "1"}) == "1";
        args.finish();
        const dim3 block(bx, by);
        const bool tiled = variant == "tiled";

        // Stay within the default 48 KiB of dynamic shared memory per block (no opt-in needed).
        const std::size_t shared_bytes = tiled ? lab06::tile_bytes(radius, block) : 0;
        if (shared_bytes > 48 * 1024) throw std::invalid_argument("The tile needs more than 48 KiB: use a smaller block or radius.");

        const common::HostImage input = common::make_pattern(pattern, width, height, 1, host_pad);
        common::HostImage expected(width, height, 1);
        const auto cpu_times = common::time_host(0, cpu_reps, [&] { lab06::box_filter_cpu(input, expected, radius); });

        common::DeviceImage d_in(width, height, 1), d_out(width, height, 1);
        d_in.upload(input);
        d_out.fill_all(guard);
        const auto kernel_times = common::time_device(warmups, reps, [&] {
            if (tiled) {
                lab06::launch_box_tiled(d_in.cview(), d_out.view(), radius, block);
            } else {
                lab06::launch_box_direct(d_in.cview(), d_out.view(), radius, block);
            }
            CUDA_CHECK_LAUNCH();
        });
        common::HostImage got(width, height, 1);
        d_out.download(got);
        std::string first;
        const std::size_t mismatches = common::count_differences(expected, got, first);
        const std::size_t guard_violations = d_out.padding_changes(guard);

        if (!raw_path.empty()) {
            common::write_observations(raw_path, {{"cpu_ms", cpu_times}, {"kernel_ms", kernel_times}});
        }
        // Global-memory loads issued per output pixel: every neighbor for the direct kernel; the
        // tile (outputs plus halo) divided by the outputs per block for the tiled kernel.
        const double area = lab06::box_area(radius);
        const double loads_per_pixel = tiled ? static_cast<double>(shared_bytes) / (bx * by) : area;
        const double kernel_ms = common::median(kernel_times);
        const bool correct = mismatches == 0 && guard_violations == 0;
        if (header) {
            std::cout << "lab,variant,pattern,width,height,radius,block,shared_bytes,global_loads_per_pixel,repetitions,"
                         "cpu_ms,kernel_ms,mpix_per_s,mismatches,guard_violations,correct\n";
        }
        std::cout << std::setprecision(6) << std::boolalpha << "lab06," << variant << ',' << pattern << ',' << width
                  << ',' << height << ',' << radius << ',' << block_text << ',' << shared_bytes << ',' << loads_per_pixel
                  << ',' << reps << ',' << common::median(cpu_times) << ',' << kernel_ms << ','
                  << static_cast<double>(width) * height / (kernel_ms * 1000.0) << ',' << mismatches << ','
                  << guard_violations << ',' << correct << '\n';
        if (!correct) {
            if (!first.empty()) std::cerr << "First mismatch: " << first << '\n';
            if (guard_violations != 0) std::cerr << guard_violations << " padding bytes were written: guard the store.\n";
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
