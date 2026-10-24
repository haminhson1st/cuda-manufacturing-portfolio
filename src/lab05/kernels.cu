// Lab 05 — YOUR per-pixel kernels. Build: make lab05   Test: make test-lab05   Memcheck: make sanitize-lab05
// Read tutorials/05-image-indexing.md first. The reference solution is in
// solutions/lab05/kernels.cu; make a time-boxed attempt before opening it.

#include "common/cuda_utils.cuh"
#include "lab05/lab05.hpp"

namespace lab05 {
namespace {

// TODO(lab05, step 1): one thread per pixel.
//   int x = blockIdx.x * blockDim.x + threadIdx.x;   and the same for y with the .y fields
//   if (x >= rgb.width || y >= rgb.height) return;   guard BOTH coordinates (no barrier here)
//   Rows start `pitch` BYTES apart: const std::uint8_t* row = rgb.data + (size_t)y * rgb.pitch;
//   RGB pixel x occupies row[3x], row[3x + 1], row[3x + 2] in R, G, B order.
//   gray = (77 R + 150 G + 29 B + 128) >> 8, computed in int; store it at gray.data[y * gray.pitch + x].
__global__ void rgb_to_gray_kernel(common::ConstImageView rgb, common::ImageView gray) {
    // Your code here.
}

// TODO(lab05, step 2): level stretch with exactly the integer rule of normalize_level() in lab05.hpp.
// (Host functions cannot be called from a kernel; write the device version here.)
__global__ void normalize_kernel(common::ConstImageView in, common::ImageView out, int lo, int hi) {
    // Your code here.
}

// TODO(lab05, step 3): diff = |test - reference| computed in int; mask = 255 where diff > threshold, else 0.
__global__ void absdiff_mask_kernel(common::ConstImageView test, common::ConstImageView reference,
                                    common::ImageView diff, common::ImageView mask, int threshold) {
    // Your code here.
}

// TODO(lab05, step 0): enough blocks to cover width x height (round up in each dimension).
// As written, the grid is a single block, so only the top-left corner of the image is processed.
dim3 make_grid(int width, int height, dim3 block) {
    (void)width;
    (void)height;
    (void)block;
    return dim3(1, 1);
}

}  // namespace

void launch_rgb_to_gray(common::ConstImageView rgb, common::ImageView gray, dim3 block, cudaStream_t stream) {
    rgb_to_gray_kernel<<<make_grid(rgb.width, rgb.height, block), block, 0, stream>>>(rgb, gray);
}

void launch_normalize(common::ConstImageView in, common::ImageView out, int lo, int hi, dim3 block,
                      cudaStream_t stream) {
    normalize_kernel<<<make_grid(in.width, in.height, block), block, 0, stream>>>(in, out, lo, hi);
}

void launch_absdiff_mask(common::ConstImageView test, common::ConstImageView reference, common::ImageView diff,
                         common::ImageView mask, int threshold, dim3 block, cudaStream_t stream) {
    absdiff_mask_kernel<<<make_grid(test.width, test.height, block), block, 0, stream>>>(test, reference, diff, mask,
                                                                                       threshold);
}

}  // namespace lab05
