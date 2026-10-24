// Lab 08 — YOUR three implementations of one aggregation.
// Build: make lab08   Test: make test-lab08   Measure: make bench-lab08
// Read tutorials/08-library-comparison.md first. The reference solution is in
// solutions/lab08/reductions.cu; make a time-boxed attempt before opening it.

#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/device/device_reduce.cuh>

#include <stdexcept>

#include "common/cuda_utils.cuh"
#include "lab08/lab08.hpp"

namespace lab08 {
namespace {

// TODO(lab08, step 1): the custom kernel. Start from your Lab 07 count kernel, adapted to a packed
// 1D array: a grid-stride loop (Lab 02) accumulates a per-thread count, the block reduces those
// counts in shared memory (Lab 03), and one thread per block calls atomicAdd(count, block_total).
// The launch below reserves blockDim.x unsigned long longs of dynamic shared memory.
__global__ void count_nonzero_kernel(const std::uint8_t* mask, std::size_t n, unsigned long long* count) {
    // Your code here.
}

// TODO(lab08, steps 2 and 3): functors for the libraries. Device code calls them, so declare
// operator() as __host__ __device__ and const:
//   a predicate for thrust::count_if:        bool operator()(std::uint8_t v) const { return v != 0; }
//   a transform for cub::DeviceReduce::Sum:  unsigned long long operator()(std::uint8_t v) const
//                                            { return v != 0 ? 1 : 0; }

}  // namespace

void count_custom(const std::uint8_t* d_mask, std::size_t n, unsigned long long* d_count, int threads,
                  cudaStream_t stream) {
    // TODO(lab08, step 1): zero *d_count with cudaMemsetAsync, then choose `blocks`. With a
    // grid-stride loop a few blocks per SM are plenty, and fewer blocks means fewer atomics:
    // cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) gives the SM count.
    const int blocks = 0;
    if (blocks == 0) throw std::runtime_error("count_custom is not implemented yet: edit src/lab08/reductions.cu.");
    count_nonzero_kernel<<<blocks, threads, threads * sizeof(unsigned long long), stream>>>(d_mask, n, d_count);
}

unsigned long long count_thrust(const std::uint8_t* d_mask, std::size_t n) {
    // TODO(lab08, step 2): return thrust::count_if(thrust::device, d_mask, d_mask + n, <predicate>);
    (void)d_mask;
    (void)n;
    throw std::runtime_error("count_thrust is not implemented yet: edit src/lab08/reductions.cu.");
}

std::size_t cub_temp_bytes(const std::uint8_t* d_mask, std::size_t n) {
    // TODO(lab08, step 3): call cub::DeviceReduce::Sum(nullptr, bytes, input, output, num_items) and
    // return `bytes`. A null storage pointer makes CUB only report the size it needs. Build `input`
    // with thrust::make_transform_iterator(d_mask, <transform>) so each nonzero byte counts as 1;
    // pass an unsigned long long* output (null is fine in this phase) so CUB accumulates in
    // 64 bits; num_items is an int (the harness keeps n below 2^31).
    (void)d_mask;
    (void)n;
    throw std::runtime_error("cub_temp_bytes is not implemented yet: edit src/lab08/reductions.cu.");
}

void count_cub(void* d_temp, std::size_t temp_bytes, const std::uint8_t* d_mask, std::size_t n,
               unsigned long long* d_count, cudaStream_t stream) {
    // TODO(lab08, step 3): the same call with the real storage, d_count, and stream.
    (void)d_temp;
    (void)temp_bytes;
    (void)d_mask;
    (void)n;
    (void)d_count;
    (void)stream;
    throw std::runtime_error("count_cub is not implemented yet: edit src/lab08/reductions.cu.");
}

}  // namespace lab08
