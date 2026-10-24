// Lab 03 — YOUR window-statistics kernels.
// Build: make lab03   Test: make test-lab03   Sanitize: make sanitize-lab03 (memcheck, racecheck, synccheck)
// Read tutorials/03-window-statistics.md first. The reference solution is in
// solutions/lab03/kernels.cu; make a time-boxed attempt before opening it.

#include "common/cuda_utils.cuh"
#include "lab03/lab03.hpp"

namespace lab03 {
namespace {

// TODO(lab03, step 1): fused statistics, one block per window.
//
//   Window w = blockIdx.x covers samples [w * window, min(n, (w + 1) * window)); m = valid samples.
//   1. Each thread accumulates sum, sum of squares, and max |x| over the samples
//        begin + threadIdx.x, begin + threadIdx.x + blockDim.x, ...
//      so neighboring threads read neighboring addresses. Start from the identities (0, 0, 0):
//      threads with no samples then contribute nothing.
//   2. The launch reserves 3 * blockDim.x floats of dynamic shared memory. Declare
//        extern __shared__ float shared[];
//      and use shared, shared + blockDim.x, and shared + 2 * blockDim.x as the three arrays.
//      Store each thread's partial results there, then __syncthreads().
//   3. Tree reduction:
//        for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
//            if (threadIdx.x < stride) { combine element threadIdx.x with threadIdx.x + stride }
//            __syncthreads();   // OUTSIDE the if: every thread must reach every barrier
//        }
//   4. Thread 0 writes out[w]: mean = sum / m, rms = sqrtf(sumsq / m), peak, count = m.
__global__ void fused_kernel(const float* x, std::size_t n, std::size_t window, WindowStats* out) {
    // Your code here.
}

// TODO(lab03, step 2): one statistic per launch, for the fused-versus-separate experiment.
// Same structure as fused_kernel with one shared array of blockDim.x floats. `stat` selects the
// statistic: 0 = sum (write out[w].mean), 1 = sum of squares (write out[w].rms), 2 = peak
// (write out[w].peak). Thread 0 also writes out[w].count.
__global__ void single_stat_kernel(const float* x, std::size_t n, std::size_t window, int stat, WindowStats* out) {
    // Your code here.
}

}  // namespace

void launch_window_stats(const float* x, std::size_t n, std::size_t window, WindowStats* out, int threads,
                         Mode mode, cudaStream_t stream) {
    const std::size_t windows = window_count(n, window);
    if (windows == 0) return;
    const auto blocks = static_cast<unsigned int>(windows);
    if (mode == Mode::fused) {
        const std::size_t shared_bytes = 3 * static_cast<std::size_t>(threads) * sizeof(float);
        fused_kernel<<<blocks, threads, shared_bytes, stream>>>(x, n, window, out);
    } else {
        const std::size_t shared_bytes = static_cast<std::size_t>(threads) * sizeof(float);
        for (int stat = 0; stat < 3; ++stat) {
            single_stat_kernel<<<blocks, threads, shared_bytes, stream>>>(x, n, window, stat, out);
        }
    }
}

}  // namespace lab03
