// Lab 08 reference solution. Compare it with your src/lab08/reductions.cu after a working attempt.

#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/device/device_reduce.cuh>

#include <algorithm>

#include "common/cuda_utils.cuh"
#include "lab08/lab08.hpp"

namespace lab08 {
namespace {

__global__ void count_nonzero_kernel(const std::uint8_t* mask, std::size_t n, unsigned long long* count) {
    extern __shared__ unsigned long long partial[];
    unsigned long long local = 0;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        local += mask[i] != 0;
    }
    partial[threadIdx.x] = local;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0 && partial[0] != 0) atomicAdd(count, partial[0]);
}

struct NonZero {
    __host__ __device__ bool operator()(std::uint8_t v) const { return v != 0; }
};

struct CountNonZero {
    __host__ __device__ unsigned long long operator()(std::uint8_t v) const { return v != 0 ? 1ull : 0ull; }
};

int sm_count() {
    static const int sms = [] {
        int device = 0, value = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device));
        return value;
    }();
    return sms;
}

}  // namespace

void count_custom(const std::uint8_t* d_mask, std::size_t n, unsigned long long* d_count, int threads,
                  cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(unsigned long long), stream));
    if (n == 0) return;
    const auto per_block = static_cast<std::size_t>(threads);
    // A grid-stride loop lets a few blocks per SM cover any n, with one atomic per block.
    const std::size_t blocks = std::min<std::size_t>((n + per_block - 1) / per_block, static_cast<std::size_t>(sm_count()) * 8);
    count_nonzero_kernel<<<static_cast<unsigned>(blocks), threads, per_block * sizeof(unsigned long long), stream>>>(
        d_mask, n, d_count);
}

unsigned long long count_thrust(const std::uint8_t* d_mask, std::size_t n) {
    return static_cast<unsigned long long>(thrust::count_if(thrust::device, d_mask, d_mask + n, NonZero()));
}

std::size_t cub_temp_bytes(const std::uint8_t* d_mask, std::size_t n) {
    std::size_t bytes = 0;
    unsigned long long* no_output = nullptr;
    CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, bytes, thrust::make_transform_iterator(d_mask, CountNonZero()),
                                      no_output, static_cast<int>(n)));
    return bytes;
}

void count_cub(void* d_temp, std::size_t temp_bytes, const std::uint8_t* d_mask, std::size_t n,
               unsigned long long* d_count, cudaStream_t stream) {
    CUDA_CHECK(cub::DeviceReduce::Sum(d_temp, temp_bytes, thrust::make_transform_iterator(d_mask, CountNonZero()),
                                      d_count, static_cast<int>(n), stream));
}

}  // namespace lab08
