#pragma once

// Lab 05 — per-pixel inspection pipeline on pitched 8-bit images:
//   RGB -> gray (3-channel input only) -> level normalization -> |test - reference| -> binary mask
// Every step uses integer arithmetic, so CPU and GPU results must match exactly.

#include <cuda_runtime.h>

#include <cstdint>

#include "common/image.hpp"

namespace lab05 {

// ---- The per-pixel rules. Your kernels must reproduce them exactly. ------------------------
// Channel order is R, G, B. BT.601 luma weights in 8-bit fixed point (77 + 150 + 29 = 256):
inline std::uint8_t gray_from_rgb(int r, int g, int b) {
    return static_cast<std::uint8_t>((77 * r + 150 * g + 29 * b + 128) >> 8);
}

// Level stretch: v <= lo -> 0, v >= hi -> 255, otherwise round((v - lo) * 255 / (hi - lo)).
inline std::uint8_t normalize_level(int v, int lo, int hi) {
    if (v <= lo) return 0;
    if (v >= hi) return 255;
    const int range = hi - lo;
    return static_cast<std::uint8_t>(((v - lo) * 255 + range / 2) / range);
}

// diff = |a - b| computed in int (unsigned bytes would wrap); mask = 255 where diff > threshold.
inline std::uint8_t abs_difference(int a, int b) { return static_cast<std::uint8_t>(a > b ? a - b : b - a); }
inline std::uint8_t mask_value(int diff, int threshold) { return diff > threshold ? 255 : 0; }

// ---- Kernels: implemented in src/lab05/kernels.cu (yours) or solutions/lab05/kernels.cu -----
// Views hold DEVICE pointers; pitch is in bytes. `rgb` has 3 bytes per pixel, the others 1.
// Each launcher enqueues one kernel in `stream` and returns without synchronizing.
void launch_rgb_to_gray(common::ConstImageView rgb, common::ImageView gray, dim3 block, cudaStream_t stream = nullptr);
void launch_normalize(common::ConstImageView in, common::ImageView out, int lo, int hi, dim3 block,
                      cudaStream_t stream = nullptr);
void launch_absdiff_mask(common::ConstImageView test, common::ConstImageView reference, common::ImageView diff,
                         common::ImageView mask, int threshold, dim3 block, cudaStream_t stream = nullptr);

// ---- CPU reference --------------------------------------------------------------------------
inline void rgb_to_gray_cpu(const common::HostImage& rgb, common::HostImage& gray) {
    for (int y = 0; y < rgb.height; ++y) {
        for (int x = 0; x < rgb.width; ++x) gray.at(x, y) = gray_from_rgb(rgb.at(x, y, 0), rgb.at(x, y, 1), rgb.at(x, y, 2));
    }
}

inline void normalize_cpu(const common::HostImage& in, common::HostImage& out, int lo, int hi) {
    for (int y = 0; y < in.height; ++y) {
        for (int x = 0; x < in.width; ++x) out.at(x, y) = normalize_level(in.at(x, y), lo, hi);
    }
}

inline void absdiff_mask_cpu(const common::HostImage& a, const common::HostImage& b, common::HostImage& diff,
                             common::HostImage& mask, int threshold) {
    for (int y = 0; y < a.height; ++y) {
        for (int x = 0; x < a.width; ++x) {
            diff.at(x, y) = abs_difference(a.at(x, y), b.at(x, y));
            mask.at(x, y) = mask_value(diff.at(x, y), threshold);
        }
    }
}

}  // namespace lab05
