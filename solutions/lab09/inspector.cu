// Lab 09 reference solution. Compare it with your src/lab09/inspector.cu after a working attempt.

#include <memory>
#include <vector>

#include "common/nvtx.hpp"
#include "lab09/lab09.hpp"

namespace lab09 {

void Inspector::ensure_buffers(int width, int height, int channels) {
    if (input_ && width == width_ && height == height_ && channels == channels_) return;
    common::NvtxRange range("allocate");
    input_ = std::make_unique<common::DeviceImage>(width, height, channels);
    gray_ = std::make_unique<common::DeviceImage>(width, height, 1);
    norm_ = std::make_unique<common::DeviceImage>(width, height, 1);
    smooth_ = std::make_unique<common::DeviceImage>(width, height, 1);
    residual_ = std::make_unique<common::DeviceImage>(width, height, 1);
    mask_ = std::make_unique<common::DeviceImage>(width, height, 1);
    width_ = width;
    height_ = height;
    channels_ = channels;
    ++allocations_;
}

unsigned long long Inspector::score(const common::HostImage& image, lab07::GpuPhaseTimes& times,
                                    common::HostImage* mask_out) {
    ensure_buffers(image.width, image.height, image.channels);
    {
        common::NvtxRange range("upload");
        timer_.start();
        input_->upload(image);
        times.h2d_ms = timer_.stop();
    }
    {
        common::NvtxRange range("kernels");
        timer_.start();
        common::ConstImageView source = input_->cview();
        if (image.channels == 3) {
            lab05::launch_rgb_to_gray(input_->cview(), gray_->view(), params_.block);
            CUDA_CHECK_LAUNCH();
            source = gray_->cview();
        }
        lab05::launch_normalize(source, norm_->view(), params_.lo, params_.hi, params_.block);
        CUDA_CHECK_LAUNCH();
        lab06::launch_box_tiled(norm_->cview(), smooth_->view(), params_.radius, params_.block);
        CUDA_CHECK_LAUNCH();
        lab05::launch_absdiff_mask(norm_->cview(), smooth_->cview(), residual_->view(), mask_->view(),
                                   params_.pixel_threshold, params_.block);
        CUDA_CHECK_LAUNCH();
        lab07::launch_count_mask(mask_->cview(), count_.get(), params_.block);
        CUDA_CHECK_LAUNCH();
        times.compute_ms = timer_.stop();
    }
    common::NvtxRange range("download");
    timer_.start();
    std::vector<unsigned long long> count;
    count_.download(count);
    if (mask_out != nullptr) {
        *mask_out = common::HostImage(image.width, image.height, 1);
        mask_->download(*mask_out);
    }
    times.d2h_ms = timer_.stop();
    return count[0];
}

}  // namespace lab09
