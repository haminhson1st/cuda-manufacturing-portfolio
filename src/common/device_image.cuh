#pragma once

// Pitched device image for Labs 05–07. cudaMallocPitch chooses an aligned row pitch, so
// the device rows usually carry padding; the guard check verifies kernels never write it.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "common/cuda_utils.cuh"
#include "common/image.hpp"

namespace common {

class DeviceImage {
public:
    DeviceImage(int width, int height, int channels) : width_(width), height_(height), channels_(channels) {
        CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&data_), &pitch_, row_bytes(),
                                   static_cast<std::size_t>(height)));
    }
    ~DeviceImage() {
        if (data_ != nullptr) cudaFree(data_);
    }
    DeviceImage(const DeviceImage&) = delete;
    DeviceImage& operator=(const DeviceImage&) = delete;

    ImageView view() { return {data_, pitch_, width_, height_}; }
    ConstImageView cview() const { return {data_, pitch_, width_, height_}; }
    std::size_t pitch() const { return pitch_; }
    std::size_t row_bytes() const { return static_cast<std::size_t>(width_) * static_cast<std::size_t>(channels_); }
    std::size_t allocation_bytes() const { return pitch_ * static_cast<std::size_t>(height_); }

    // cudaMemcpy2D copies row_bytes() per row and honors both pitches.
    void upload(const HostImage& host) {
        check_shape(host);
        CUDA_CHECK(cudaMemcpy2D(data_, pitch_, host.bytes.data(), host.pitch, row_bytes(),
                                static_cast<std::size_t>(height_), cudaMemcpyHostToDevice));
    }
    void download(HostImage& host) const {
        check_shape(host);
        CUDA_CHECK(cudaMemcpy2D(host.bytes.data(), host.pitch, data_, pitch_, row_bytes(),
                                static_cast<std::size_t>(height_), cudaMemcpyDeviceToHost));
    }

    // Sets every byte of the allocation, including the padding after each row.
    void fill_all(int value) { CUDA_CHECK(cudaMemset(data_, value, allocation_bytes())); }

    // Counts padding bytes that no longer hold `value` (a kernel wrote past width).
    std::size_t padding_changes(std::uint8_t value) const {
        std::vector<std::uint8_t> raw(allocation_bytes());
        CUDA_CHECK(cudaMemcpy(raw.data(), data_, raw.size(), cudaMemcpyDeviceToHost));
        std::size_t changed = 0;
        for (int y = 0; y < height_; ++y) {
            const std::uint8_t* row = raw.data() + static_cast<std::size_t>(y) * pitch_;
            for (std::size_t b = row_bytes(); b < pitch_; ++b) changed += row[b] != value;
        }
        return changed;
    }

private:
    void check_shape(const HostImage& host) const {
        if (host.width != width_ || host.height != height_ || host.channels != channels_) {
            throw std::logic_error("DeviceImage: host image shape differs.");
        }
    }

    std::uint8_t* data_ = nullptr;
    std::size_t pitch_ = 0;
    int width_;
    int height_;
    int channels_;
};

}  // namespace common
