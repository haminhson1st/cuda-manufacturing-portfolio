// Lab 05 reference solution. Compare it with your src/lab05/kernels.cu after a working attempt.

#include "common/cuda_utils.cuh"
#include "lab05/lab05.hpp"

namespace lab05 {
namespace {

// Byte address of pixel (x, y) in a 1-channel view: rows are `pitch` bytes apart.
__device__ std::size_t offset(std::size_t pitch, int x, int y) { return static_cast<std::size_t>(y) * pitch + x; }

__global__ void rgb_to_gray_kernel(common::ConstImageView rgb, common::ImageView gray) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= rgb.width || y >= rgb.height) return;
    const std::uint8_t* pixel = rgb.data + static_cast<std::size_t>(y) * rgb.pitch + 3 * static_cast<std::size_t>(x);
    const int value = (77 * pixel[0] + 150 * pixel[1] + 29 * pixel[2] + 128) >> 8;
    gray.data[offset(gray.pitch, x, y)] = static_cast<std::uint8_t>(value);
}

__global__ void normalize_kernel(common::ConstImageView in, common::ImageView out, int lo, int hi) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= in.width || y >= in.height) return;
    const int v = in.data[offset(in.pitch, x, y)];
    int result;
    if (v <= lo) {
        result = 0;
    } else if (v >= hi) {
        result = 255;
    } else {
        const int range = hi - lo;
        result = ((v - lo) * 255 + range / 2) / range;
    }
    out.data[offset(out.pitch, x, y)] = static_cast<std::uint8_t>(result);
}

__global__ void absdiff_mask_kernel(common::ConstImageView test, common::ConstImageView reference,
                                    common::ImageView diff, common::ImageView mask, int threshold) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= test.width || y >= test.height) return;
    const int a = test.data[offset(test.pitch, x, y)];
    const int b = reference.data[offset(reference.pitch, x, y)];
    const int d = a > b ? a - b : b - a;
    diff.data[offset(diff.pitch, x, y)] = static_cast<std::uint8_t>(d);
    mask.data[offset(mask.pitch, x, y)] = d > threshold ? 255 : 0;
}

dim3 make_grid(int width, int height, dim3 block) {
    return dim3((static_cast<unsigned>(width) + block.x - 1) / block.x,
                (static_cast<unsigned>(height) + block.y - 1) / block.y);
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
