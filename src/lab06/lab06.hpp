#pragma once

// Lab 06 — (2R+1) x (2R+1) box filter on 8-bit gray images, with clamp-to-edge borders:
//   out(x, y) = round_half_up( sum over dx, dy in [-R, R] of in(clamp(x + dx), clamp(y + dy)) / (2R+1)^2 )
// The sum is an integer and the division is (sum + area / 2) / area, so CPU and GPU match exactly.

#include <cuda_runtime.h>

#include <cstddef>

#include "common/image.hpp"

namespace lab06 {

inline int clamp_index(int v, int size) { return v < 0 ? 0 : (v >= size ? size - 1 : v); }
inline int box_area(int radius) { return (2 * radius + 1) * (2 * radius + 1); }

// Bytes of dynamic shared memory for one (block.x + 2R) x (block.y + 2R) tile of pixels.
inline std::size_t tile_bytes(int radius, dim3 block) {
    return static_cast<std::size_t>(block.x + 2 * radius) * static_cast<std::size_t>(block.y + 2 * radius);
}

// Implemented in src/lab06/kernels.cu (yours) or solutions/lab06/kernels.cu. Views hold DEVICE
// pointers. Each launcher enqueues one kernel in `stream` and returns without synchronizing.
void launch_box_direct(common::ConstImageView in, common::ImageView out, int radius, dim3 block,
                       cudaStream_t stream = nullptr);
void launch_box_tiled(common::ConstImageView in, common::ImageView out, int radius, dim3 block,
                      cudaStream_t stream = nullptr);

// CPU reference: the definition above, written as directly as possible.
inline void box_filter_cpu(const common::HostImage& in, common::HostImage& out, int radius) {
    const int area = box_area(radius);
    for (int y = 0; y < in.height; ++y) {
        for (int x = 0; x < in.width; ++x) {
            int sum = 0;
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    sum += in.at(clamp_index(x + dx, in.width), clamp_index(y + dy, in.height));
                }
            }
            out.at(x, y) = static_cast<std::uint8_t>((sum + area / 2) / area);
        }
    }
}

}  // namespace lab06
