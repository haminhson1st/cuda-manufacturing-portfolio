// Lab 06 reference solution. Compare it with your src/lab06/kernels.cu after a working attempt.

#include "common/cuda_utils.cuh"
#include "lab06/lab06.hpp"

namespace lab06 {
namespace {

__device__ int clamp_coord(int v, int size) { return min(max(v, 0), size - 1); }

__global__ void box_direct_kernel(common::ConstImageView in, common::ImageView out, int radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= in.width || y >= in.height) return;
    int sum = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        const std::uint8_t* row = in.data + static_cast<std::size_t>(clamp_coord(y + dy, in.height)) * in.pitch;
        for (int dx = -radius; dx <= radius; ++dx) sum += row[clamp_coord(x + dx, in.width)];
    }
    const int area = (2 * radius + 1) * (2 * radius + 1);
    out.data[static_cast<std::size_t>(y) * out.pitch + x] = static_cast<std::uint8_t>((sum + area / 2) / area);
}

__global__ void box_tiled_kernel(common::ConstImageView in, common::ImageView out, int radius) {
    extern __shared__ std::uint8_t tile[];
    const int tile_w = static_cast<int>(blockDim.x) + 2 * radius;
    const int tile_h = static_cast<int>(blockDim.y) + 2 * radius;
    // Image coordinates of tile entry (0, 0): the block's first output pixel minus the halo.
    const int origin_x = static_cast<int>(blockIdx.x * blockDim.x) - radius;
    const int origin_y = static_cast<int>(blockIdx.y * blockDim.y) - radius;

    // 1. Cooperative load of outputs plus halo; consecutive threads read consecutive bytes of a row.
    const int threads = static_cast<int>(blockDim.x * blockDim.y);
    for (int i = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x); i < tile_w * tile_h; i += threads) {
        const int gx = clamp_coord(origin_x + i % tile_w, in.width);
        const int gy = clamp_coord(origin_y + i / tile_w, in.height);
        tile[i] = in.data[static_cast<std::size_t>(gy) * in.pitch + gx];
    }
    // 2. Every thread reaches the barrier, including those whose pixel is outside the image.
    __syncthreads();

    // 3. Guarded store.
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < in.width && y < in.height) {
        int sum = 0;
        for (int dy = 0; dy <= 2 * radius; ++dy) {
            const std::uint8_t* row = tile + (static_cast<int>(threadIdx.y) + dy) * tile_w + threadIdx.x;
            for (int dx = 0; dx <= 2 * radius; ++dx) sum += row[dx];
        }
        const int area = (2 * radius + 1) * (2 * radius + 1);
        out.data[static_cast<std::size_t>(y) * out.pitch + x] = static_cast<std::uint8_t>((sum + area / 2) / area);
    }
}

dim3 make_grid(int width, int height, dim3 block) {
    return dim3((static_cast<unsigned>(width) + block.x - 1) / block.x,
                (static_cast<unsigned>(height) + block.y - 1) / block.y);
}

}  // namespace

void launch_box_direct(common::ConstImageView in, common::ImageView out, int radius, dim3 block, cudaStream_t stream) {
    box_direct_kernel<<<make_grid(in.width, in.height, block), block, 0, stream>>>(in, out, radius);
}

void launch_box_tiled(common::ConstImageView in, common::ImageView out, int radius, dim3 block, cudaStream_t stream) {
    box_tiled_kernel<<<make_grid(in.width, in.height, block), block, tile_bytes(radius, block), stream>>>(in, out,
                                                                                                       radius);
}

}  // namespace lab06
