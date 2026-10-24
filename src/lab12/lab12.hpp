#pragma once

// Lab 12 — release measurements. The same frozen detector runs four ways on the same decoded images:
//   cpu            lab07::score_cpu on one thread
//   cpu-parallel   your multicore baseline (below); the fair comparison for any GPU claim
//   cuda-baseline  the straightforward Lab 07 version (fresh buffers per image)
//   cuda           the final Lab 09 version (buffers reused)

#include <algorithm>
#include <cstdlib>
#include <thread>
#include <vector>

#include "lab07/lab07.hpp"

namespace lab12 {

// Implemented in src/lab12/cpu_parallel.cu (yours) or solutions/lab12/cpu_parallel.cu.
// Scores every image with lab07::score_cpu on up to `threads` CPU threads; counts[i] belongs to
// images[i], exactly as a sequential loop would produce them.
std::vector<unsigned long long> score_all_cpu_parallel(const std::vector<common::HostImage>& images,
                                                      const lab07::DetectorParams& params, int threads);

// The CPU threads this job may use: SLURM_CPUS_PER_TASK when set, otherwise the hardware count.
// Using more threads than the allocation grants would measure contention, not the baseline.
inline int default_cpu_threads() {
    if (const char* slurm = std::getenv("SLURM_CPUS_PER_TASK")) {
        const int value = std::atoi(slurm);
        if (value > 0) return std::min(value, 1024);
    }
    const unsigned hardware = std::thread::hardware_concurrency();
    return hardware > 0 ? static_cast<int>(std::min(hardware, 1024u)) : 1;
}

}  // namespace lab12
