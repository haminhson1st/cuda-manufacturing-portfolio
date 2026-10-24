// Lab 06 — YOUR box-filter kernels.
// Build: make lab06   Test: make test-lab06   Sanitize: make sanitize-lab06 (memcheck, racecheck, synccheck)
// Read tutorials/06-tiled-filter.md first. The reference solution is in
// solutions/lab06/kernels.cu; make a time-boxed attempt before opening it.

#include "common/cuda_utils.cuh"
#include "lab06/lab06.hpp"

namespace lab06 {
namespace {

// TODO(lab06, step 1): direct filter, one thread per output pixel.
//   Compute (x, y) and return early if it lies outside the image (fine here: no barrier).
//   Sum the (2R+1)^2 neighbors straight from global memory, clamping each coordinate to the
//   image (write a __device__ clamp: host functions are not callable from a kernel).
//   Store (sum + area / 2) / area at out.data[y * out.pitch + x].
__global__ void box_direct_kernel(common::ConstImageView in, common::ImageView out, int radius) {
    // Your code here.
}

// TODO(lab06, step 2): tiled filter. The launch reserves tile_bytes(radius, blockDim) of dynamic
// shared memory for a (blockDim.x + 2R) x (blockDim.y + 2R) tile, stored row-major.
//   extern __shared__ std::uint8_t tile[];
//   1. Cooperative load. The tile has more entries than the block has threads, so loop:
//        for (int i = flat_thread_id; i < tile_w * tile_h; i += blockDim.x * blockDim.y)
//      Map i to (tx, ty), then to the image pixel (block origin + (tx, ty) - R), clamped to the
//      image, and copy it into tile[i]. Compute the origin in int: blockIdx.x * blockDim.x is unsigned.
//   2. __syncthreads(). EVERY thread of the block must reach it, including threads whose own
//      pixel lies outside the image, so do NOT return before this barrier.
//   3. If (x, y) is inside the image, sum the window from the tile (thread (tx, ty) owns tile
//      entries [ty .. ty + 2R] x [tx .. tx + 2R]) and store the rounded mean.
__global__ void box_tiled_kernel(common::ConstImageView in, common::ImageView out, int radius) {
    // Your code here.
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
