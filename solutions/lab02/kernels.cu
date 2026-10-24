// Lab 02 reference solution. Compare it with your src/lab02/kernels.cu after a working attempt.

#include <algorithm>

#include "common/cuda_utils.cuh"
#include "lab02/lab02.hpp"

namespace lab02 {
namespace {

__global__ void calibrate_kernel(Batch batch, Limits limits) {
    const std::size_t n = batch.channels * batch.samples;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        // Neighboring threads touch neighboring addresses in either layout; only the channel lookup differs.
        const std::size_t channel =
            batch.layout == Layout::channel_major ? i / batch.samples : i % batch.channels;
        const float value = fmaf(batch.gain[channel], batch.raw[i], batch.offset[channel]);
        batch.alarm[i] = value < limits.low ? alarm_low : (value > limits.high ? alarm_high : alarm_none);
        batch.out[i] = fminf(fmaxf(value, limits.low), limits.high);
    }
}

}  // namespace

void launch_calibrate(const Batch& batch, Limits limits, LaunchConfig config, cudaStream_t stream) {
    const std::size_t n = batch.channels * batch.samples;
    if (n == 0) return;
    int blocks = config.blocks;
    if (blocks == 0) {
        const std::size_t threads = static_cast<std::size_t>(config.threads);
        blocks = static_cast<int>(std::min<std::size_t>((n + threads - 1) / threads, 2147483647));
    }
    calibrate_kernel<<<blocks, config.threads, 0, stream>>>(batch, limits);
}

}  // namespace lab02
