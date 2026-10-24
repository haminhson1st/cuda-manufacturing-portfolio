#pragma once

// Lab 09 — profile the Lab 07 application, then change exactly one thing. The change prepared
// here keeps the device buffers alive across images instead of allocating them for every image,
// as lab07::score_gpu does. Profile first: the timeline, not this comment, should convince you.

#include <cstddef>
#include <memory>

#include "common/cuda_utils.cuh"
#include "common/device_image.cuh"
#include "lab07/lab07.hpp"

namespace lab09 {

class Inspector {
public:
    explicit Inspector(const lab07::DetectorParams& params) : params_(params) {}

    // Scores one image exactly like lab07::score_gpu, but reuses every device buffer while the image
    // shape (width, height, channels) stays the same. Implemented in src/lab09/inspector.cu.
    unsigned long long score(const common::HostImage& image, lab07::GpuPhaseTimes& times,
                             common::HostImage* mask_out);

    // How many times the buffers were (re)allocated. The harness compares it with the number of
    // shape changes in the image sequence.
    std::size_t allocations() const { return allocations_; }

private:
    // Allocates the six device images for this shape, unless the current ones already match.
    // Implemented in src/lab09/inspector.cu.
    void ensure_buffers(int width, int height, int channels);

    lab07::DetectorParams params_;
    int width_ = 0;
    int height_ = 0;
    int channels_ = 0;
    std::unique_ptr<common::DeviceImage> input_, gray_, norm_, smooth_, residual_, mask_;
    common::DeviceBuffer<unsigned long long> count_{1};  // allocated once, with the inspector
    common::EventTimer timer_;                             // events created once, too
    std::size_t allocations_ = 0;
};

}  // namespace lab09
