// Lab 09 — YOUR optimization: keep device buffers alive across images.
// Build: make lab09   Test: make test-lab09   Profile: make profile-lab09 (inside a GPU allocation)
// Read tutorials/09-profiling.md first, and profile the baseline BEFORE you write this.
// The reference solution is in solutions/lab09/inspector.cu.

#include <memory>
#include <stdexcept>

#include "common/nvtx.hpp"
#include "lab09/lab09.hpp"

namespace lab09 {

void Inspector::ensure_buffers(int width, int height, int channels) {
    // TODO(lab09, step 1): if the buffers exist and (width, height, channels) equals the current
    // shape, return. Otherwise create the six images with std::make_unique<common::DeviceImage>(...):
    // the input with `channels` channels; gray, normalized, smoothed, residual, and mask with 1.
    // Replacing a unique_ptr frees the old buffer. Remember the new shape and ++allocations_.
    (void)width;
    (void)height;
    (void)channels;
}

unsigned long long Inspector::score(const common::HostImage& image, lab07::GpuPhaseTimes& times,
                                    common::HostImage* mask_out) {
    // TODO(lab09, step 2): the same stages as lab07::score_gpu (look at your Lab 07 version), but:
    //   - call ensure_buffers(image.width, image.height, image.channels) first;
    //   - use the member images, count_, and timer_ instead of creating new ones;
    //   - wrap the upload, the kernels, and the download each in a common::NvtxRange
    //     ("upload", "kernels", "download") so the Nsight Systems timeline names them.
    // lab07::launch_count_mask zeroes the counter itself, so reusing count_ is safe.
    (void)image;
    (void)times;
    (void)mask_out;
    throw std::runtime_error("Inspector::score is not implemented yet: edit src/lab09/inspector.cu.");
}

}  // namespace lab09
