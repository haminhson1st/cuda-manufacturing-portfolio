// Lab 02 — YOUR kernel. Build: make lab02   Test: make test-lab02   Memcheck: make sanitize-lab02
// Read tutorials/02-sensor-calibration.md first. The reference solution is in
// solutions/lab02/kernels.cu; make a time-boxed attempt before opening it.

#include "common/cuda_utils.cuh"
#include "lab02/lab02.hpp"

namespace lab02 {
namespace {

// TODO(lab02): calibrate every element of the batch.
//
// For each flat index i in [0, channels * samples):
//   1. Find the channel of element i from batch.layout (element_index in lab02.hpp shows both maps):
//        channel-major: i = channel * samples + sample
//        sample-major:  i = sample * channels + channel
//   2. value = fmaf(gain[channel], raw[i], offset[channel])   (one rounding, like the CPU's std::fma)
//   3. alarm[i] = alarm_low if value < limits.low, alarm_high if value > limits.high, else alarm_none
//   4. out[i] = value clamped to [limits.low, limits.high]   (fminf / fmaxf)
//
// Use a grid-stride loop so that ANY grid size covers every element:
//   for (size_t i = <global thread index>; i < n; i += <total threads in the grid>) { ... }
// Compute the index in size_t: blockIdx.x * blockDim.x is 32-bit unsigned arithmetic and can overflow.
__global__ void calibrate_kernel(Batch batch, Limits limits) {
    // Your code here.
}

}  // namespace

void launch_calibrate(const Batch& batch, Limits limits, LaunchConfig config, cudaStream_t stream) {
    const std::size_t n = batch.channels * batch.samples;
    // TODO(lab02): when config.blocks == 0, choose enough blocks for one thread per element
    // (round up), but never more than 2147483647 (the grid's x-dimension limit).
    const int blocks = config.blocks;
    if (n == 0 || blocks == 0) return;  // delete this line once `blocks` is computed above
    calibrate_kernel<<<blocks, config.threads, 0, stream>>>(batch, limits);
}

}  // namespace lab02
