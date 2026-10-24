// Lab 07 — YOUR GPU detector: a mask-count reduction plus the composition of your Lab 05 and
// Lab 06 kernels. Build: make lab07   Test: make test-lab07   Sanitize: make sanitize-lab07
// Read tutorials/07-image-project.md first. The reference solution is in
// solutions/lab07/gpu_inspect.cu; make a time-boxed attempt before opening it.

#include <stdexcept>

#include "common/cuda_utils.cuh"
#include "common/device_image.cuh"
#include "lab07/lab07.hpp"

namespace lab07 {
namespace {

// TODO(lab07, step 1): count the nonzero bytes of a pitched mask.
//   One thread per pixel on a 2D grid, as in Lab 05. A thread outside the image contributes 0,
//   but it must NOT return early: it still has to reach the barriers below.
//   Store each thread's 0 or 1 in the dynamic shared array the launcher reserves
//   (blockDim.x * blockDim.y unsigned ints; flatten the thread index as
//   threadIdx.y * blockDim.x + threadIdx.x), reduce it with the Lab 03 tree, and let ONE thread
//   per block call
//       atomicAdd(count, static_cast<unsigned long long>(block_total));
//   Integer addition is associative, so the total is exact whatever order the blocks finish in.
__global__ void count_mask_kernel(common::ConstImageView mask, unsigned long long* count) {
    // Your code here.
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
    // TODO(lab07, step 2): compose the detector on the device, as Lab 04 composed its pipeline.
    //  1. Allocate common::DeviceImage buffers: the input (image.channels) and 1-channel gray,
    //     normalized, smoothed, residual, and mask images; plus a common::DeviceBuffer<unsigned long long>(1).
    //  2. H2D: upload the image, bracketed by a common::EventTimer -> times.h2d_ms.
    //  3. Compute, one EventTimer interval -> times.compute_ms, checking every launch with
    //     CUDA_CHECK_LAUNCH() and passing params.block to each launcher:
    //       lab05::launch_rgb_to_gray      only when image.channels == 3
    //       lab05::launch_normalize        params.lo, params.hi
    //       lab06::launch_box_tiled        params.radius
    //       lab05::launch_absdiff_mask     normalized vs smoothed -> residual and mask, params.pixel_threshold
    //       launch_count_mask              mask -> the device counter
    //  4. D2H (times.d2h_ms): download the counter, and the mask into *mask_out when it is not null
    //     (assign it a 1-channel HostImage of the right size first). Return the count.
    (void)image;
    (void)params;
    (void)times;
    (void)mask_out;
    throw std::runtime_error("score_gpu is not implemented yet: edit src/lab07/gpu_inspect.cu "
                             "(or build the reference with make SOLUTION=1).");
}

}  // namespace lab07
