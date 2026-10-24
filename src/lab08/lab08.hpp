#pragma once

// Lab 08 — one aggregation, three implementations: count the nonzero bytes of a packed mask
// (the aggregation behind Lab 07's score).
//   custom   your kernel: grid-stride loop, block reduction, one atomicAdd per block
//   thrust   thrust::count_if; Thrust manages its temporary storage internally
//   cub      cub::DeviceReduce::Sum in two phases; you query and own the temporary storage
// Integer addition is associative, so every backend must return exactly the CPU count.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace lab08 {

// ---- Implemented in src/lab08/reductions.cu (yours) or solutions/lab08/reductions.cu --------
// Zeroes *d_count, then adds the number of nonzero bytes in d_mask[0, n). Must not allocate.
// threads is a power of two in [32, 1024]. Enqueues work in `stream` without synchronizing.
void count_custom(const std::uint8_t* d_mask, std::size_t n, unsigned long long* d_count, int threads,
                  cudaStream_t stream = nullptr);

// Returns the count on the host; thrust::count_if synchronizes before it returns.
unsigned long long count_thrust(const std::uint8_t* d_mask, std::size_t n);

// CUB phase 1: the bytes of temporary storage the reduction needs for this input. No work is done.
std::size_t cub_temp_bytes(const std::uint8_t* d_mask, std::size_t n);

// CUB phase 2: the reduction itself, in caller-owned storage of temp_bytes bytes; writes *d_count.
void count_cub(void* d_temp, std::size_t temp_bytes, const std::uint8_t* d_mask, std::size_t n,
               unsigned long long* d_count, cudaStream_t stream = nullptr);

// ---- Deterministic masks ----------------------------------------------------------------------
enum class Pattern { zeros, ones, sparse, dense, random, values };

inline Pattern parse_pattern(const std::string& name) {
    if (name == "zeros") return Pattern::zeros;
    if (name == "ones") return Pattern::ones;
    if (name == "sparse") return Pattern::sparse;
    if (name == "dense") return Pattern::dense;
    if (name == "random") return Pattern::random;
    if (name == "values") return Pattern::values;
    throw std::invalid_argument("Unknown mask pattern '" + name + "'.");
}

inline std::uint64_t mix64(std::uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// zeros: all 0. ones: all 255. sparse: about 1 in 1000 set. dense: about 9 in 10 set.
// random: about half set. values: about half set to arbitrary nonzero bytes 1..254, which
// catches "sum the bytes and divide by 255" shortcuts.
inline std::vector<std::uint8_t> make_mask(Pattern pattern, std::size_t n) {
    std::vector<std::uint8_t> mask(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t r = mix64(i);
        switch (pattern) {
            case Pattern::zeros: mask[i] = 0; break;
            case Pattern::ones: mask[i] = 255; break;
            case Pattern::sparse: mask[i] = r % 1000 == 0 ? 255 : 0; break;
            case Pattern::dense: mask[i] = r % 10 != 0 ? 255 : 0; break;
            case Pattern::random: mask[i] = (r & 1) != 0 ? 255 : 0; break;
            case Pattern::values: mask[i] = (r & 1) != 0 ? static_cast<std::uint8_t>(1 + (r >> 8) % 254) : 0; break;
        }
    }
    return mask;
}

inline unsigned long long count_cpu(const std::vector<std::uint8_t>& mask) {
    unsigned long long count = 0;
    for (const std::uint8_t v : mask) count += v != 0;
    return count;
}

}  // namespace lab08
