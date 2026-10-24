// Lab 03 reference solution. Compare it with your src/lab03/kernels.cu after a working attempt.

#include "common/cuda_utils.cuh"
#include "lab03/lab03.hpp"

namespace lab03 {
namespace {

__device__ std::size_t window_end(std::size_t begin, std::size_t window, std::size_t n) {
    return n - begin < window ? n : begin + window;
}

__global__ void fused_kernel(const float* x, std::size_t n, std::size_t window, WindowStats* out) {
    extern __shared__ float shared[];
    float* s_sum = shared;
    float* s_sumsq = shared + blockDim.x;
    float* s_peak = shared + 2 * blockDim.x;

    const unsigned tid = threadIdx.x;
    const std::size_t begin = static_cast<std::size_t>(blockIdx.x) * window;
    const std::size_t end = window_end(begin, window, n);

    // Identities: threads past the end of a short window contribute 0 to every statistic.
    float sum = 0.0f, sumsq = 0.0f, peak = 0.0f;
    for (std::size_t i = begin + tid; i < end; i += blockDim.x) {
        const float v = x[i];
        sum += v;
        sumsq = fmaf(v, v, sumsq);
        peak = fmaxf(peak, fabsf(v));
    }
    s_sum[tid] = sum;
    s_sumsq[tid] = sumsq;
    s_peak[tid] = peak;
    __syncthreads();

    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sumsq[tid] += s_sumsq[tid + stride];
            s_peak[tid] = fmaxf(s_peak[tid], s_peak[tid + stride]);
        }
        __syncthreads();  // reached by every thread of the block
    }

    if (tid == 0) {
        const auto m = static_cast<float>(end - begin);
        WindowStats result;
        result.mean = s_sum[0] / m;
        result.rms = sqrtf(s_sumsq[0] / m);
        result.peak = s_peak[0];
        result.count = static_cast<std::uint32_t>(end - begin);
        out[blockIdx.x] = result;
    }
}

__global__ void single_stat_kernel(const float* x, std::size_t n, std::size_t window, int stat, WindowStats* out) {
    extern __shared__ float shared[];
    const unsigned tid = threadIdx.x;
    const std::size_t begin = static_cast<std::size_t>(blockIdx.x) * window;
    const std::size_t end = window_end(begin, window, n);

    float acc = 0.0f;
    for (std::size_t i = begin + tid; i < end; i += blockDim.x) {
        const float v = x[i];
        if (stat == 0) {
            acc += v;
        } else if (stat == 1) {
            acc = fmaf(v, v, acc);
        } else {
            acc = fmaxf(acc, fabsf(v));
        }
    }
    shared[tid] = acc;
    __syncthreads();

    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = stat == 2 ? fmaxf(shared[tid], shared[tid + stride]) : shared[tid] + shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const auto m = static_cast<float>(end - begin);
        WindowStats& result = out[blockIdx.x];
        if (stat == 0) {
            result.mean = shared[0] / m;
        } else if (stat == 1) {
            result.rms = sqrtf(shared[0] / m);
        } else {
            result.peak = shared[0];
        }
        result.count = static_cast<std::uint32_t>(end - begin);
    }
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
