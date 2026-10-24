// Lab 07 reference solution. Compare it with your src/lab07/gpu_inspect.cu after a working attempt.

#include <stdexcept>
#include <vector>

#include "common/cuda_utils.cuh"
#include "common/device_image.cuh"
#include "lab07/lab07.hpp"

namespace lab07 {
namespace {

__global__ void count_mask_kernel(common::ConstImageView mask, unsigned long long* count) {
    extern __shared__ unsigned int partial[];
    const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned threads = blockDim.x * blockDim.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    // No early return: threads outside the image contribute 0 and still reach every barrier.
    const bool inside = x < mask.width && y < mask.height;
    partial[tid] = inside && mask.data[static_cast<std::size_t>(y) * mask.pitch + x] != 0 ? 1u : 0u;
    __syncthreads();
    for (unsigned stride = threads / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    // One atomic per block: blocks cannot synchronize with each other, but they can all add.
    if (tid == 0 && partial[0] != 0) atomicAdd(count, static_cast<unsigned long long>(partial[0]));
}

}  // namespace

void launch_count_mask(common::ConstImageView mask, unsigned long long* count, dim3 block, cudaStream_t stream) {
    const unsigned threads = block.x * block.y;
    if (threads == 0 || (threads & (threads - 1)) != 0) {
        throw std::invalid_argument("launch_count_mask needs a power-of-two number of threads per block.");
    }
    CUDA_CHECK(cudaMemsetAsync(count, 0, sizeof(unsigned long long), stream));
    const dim3 grid((static_cast<unsigned>(mask.width) + block.x - 1) / block.x,
                    (static_cast<unsigned>(mask.height) + block.y - 1) / block.y);
    count_mask_kernel<<<grid, block, threads * sizeof(unsigned int), stream>>>(mask, count);
}

unsigned long long score_gpu(const common::HostImage& image, const DetectorParams& params, GpuPhaseTimes& times,
                             common::HostImage* mask_out) {
    const int w = image.width;
    const int h = image.height;
    // Allocated per image: simple and correct. Lab 09 measures whether reusing buffers matters.
    common::DeviceImage d_input(w, h, image.channels), d_gray(w, h, 1), d_norm(w, h, 1), d_smooth(w, h, 1),
        d_residual(w, h, 1), d_mask(w, h, 1);
    common::DeviceBuffer<unsigned long long> d_count(1);
    common::EventTimer timer;

    timer.start();
    d_input.upload(image);
    times.h2d_ms = timer.stop();

    timer.start();
    common::ConstImageView source = d_input.cview();
    if (image.channels == 3) {
        lab05::launch_rgb_to_gray(d_input.cview(), d_gray.view(), params.block);
        CUDA_CHECK_LAUNCH();
        source = d_gray.cview();
    }
    lab05::launch_normalize(source, d_norm.view(), params.lo, params.hi, params.block);
    CUDA_CHECK_LAUNCH();
    lab06::launch_box_tiled(d_norm.cview(), d_smooth.view(), params.radius, params.block);
    CUDA_CHECK_LAUNCH();
    lab05::launch_absdiff_mask(d_norm.cview(), d_smooth.cview(), d_residual.view(), d_mask.view(),
                               params.pixel_threshold, params.block);
    CUDA_CHECK_LAUNCH();
    launch_count_mask(d_mask.cview(), d_count.get(), params.block);
    CUDA_CHECK_LAUNCH();
    times.compute_ms = timer.stop();

    timer.start();
    std::vector<unsigned long long> count;
    d_count.download(count);
    if (mask_out != nullptr) {
        *mask_out = common::HostImage(w, h, 1);
        d_mask.download(*mask_out);
    }
    times.d2h_ms = timer.stop();
    return count[0];
}

}  // namespace lab07
