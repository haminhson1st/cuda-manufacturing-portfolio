// Lab 12 reference solution. Compare it with your src/lab12/cpu_parallel.cu after a working attempt.

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>

#include "lab12/lab12.hpp"

namespace lab12 {

std::vector<unsigned long long> score_all_cpu_parallel(const std::vector<common::HostImage>& images,
                                                      const lab07::DetectorParams& params, int threads) {
    std::vector<unsigned long long> counts(images.size());
    std::atomic<std::size_t> next{0};
    std::exception_ptr failure;
    std::mutex failure_mutex;

    // Images are independent: each worker claims the next unscored index until none are left.
    const auto worker = [&] {
        try {
            for (std::size_t i = next++; i < images.size(); i = next++) {
                counts[i] = lab07::score_cpu(images[i], params, nullptr);
            }
        } catch (...) {
            const std::lock_guard<std::mutex> lock(failure_mutex);
            if (!failure) failure = std::current_exception();
        }
    };

    const std::size_t workers = std::min<std::size_t>(static_cast<std::size_t>(std::max(threads, 1)), images.size());
    std::vector<std::thread> pool;
    pool.reserve(workers);
    try {
        for (std::size_t t = 0; t < workers; ++t) pool.emplace_back(worker);
    } catch (...) {
        for (auto& thread : pool) thread.join();  // never destroy a joinable thread
        throw;
    }
    for (auto& thread : pool) thread.join();
    if (failure) std::rethrow_exception(failure);
    return counts;
}

}  // namespace lab12
