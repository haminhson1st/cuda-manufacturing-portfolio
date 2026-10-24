// Lab 12 — YOUR multicore CPU baseline. A GPU speedup is only as honest as the CPU it is measured
// against. Build: make lab12   Test: make test-lab12   Measure: make bench-lab12
// Read tutorials/12-portfolio-release.md first. The reference solution is in
// solutions/lab12/cpu_parallel.cu.

#include <stdexcept>

#include "lab12/lab12.hpp"

namespace lab12 {

std::vector<unsigned long long> score_all_cpu_parallel(const std::vector<common::HostImage>& images,
                                                      const lab07::DetectorParams& params, int threads) {
    // TODO(lab12): score the images on up to `threads` std::thread workers.
    //  - std::vector<unsigned long long> counts(images.size());
    //  - a std::atomic<std::size_t> next{0}: each worker loops "i = next++; stop when i >= images.size();
    //    counts[i] = lab07::score_cpu(images[i], params, nullptr);". Workers write distinct elements,
    //    so they need no lock, and the result order is the input order whatever the timing.
    //  - Start min(threads, images.size()) workers and join every one of them before returning.
    //  - An exception escaping a std::thread terminates the whole program: catch it inside the
    //    worker, keep the first one in a std::exception_ptr (guarded by a std::mutex), and
    //    std::rethrow_exception it after joining.
    (void)images;
    (void)params;
    (void)threads;
    throw std::runtime_error("score_all_cpu_parallel is not implemented yet: edit src/lab12/cpu_parallel.cu.");
}

}  // namespace lab12
