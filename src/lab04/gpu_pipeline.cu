// Lab 04 — YOUR GPU pipeline. Build: make lab04   Test: make test-lab04
// Read tutorials/04-sensor-project.md first. The reference solution is in
// solutions/lab04/gpu_pipeline.cu; make a time-boxed attempt before opening it.

#include <limits>
#include <stdexcept>

#include "common/cuda_utils.cuh"
#include "lab04/lab04.hpp"

namespace lab04 {

std::vector<lab03::WindowStats> run_gpu_pipeline(const std::vector<float>& raw, const PipelineConfig& config,
                                                 GpuPhaseTimes& times) {
    // TODO(lab04): compose the kernels you already wrote, keeping the calibrated signal on the GPU.
    //
    //  1. Allocate device buffers with common::DeviceBuffer: the raw signal, the calibrated signal,
    //     one alarm byte per sample (Lab 02 writes it; it is unused here), a one-element gain and
    //     offset, and lab03::window_count(raw.size(), config.window) WindowStats.
    //  2. H2D: upload raw, gain, and offset. Bracket the copies with a common::EventTimer and store
    //     the result in times.h2d_ms.
    //  3. Compute, timed as one interval in times.compute_ms:
    //       lab02::launch_calibrate with channels = 1, samples = raw.size(), Layout::channel_major,
    //         limits = {-infinity, +infinity} (no clamping), LaunchConfig{config.threads, 0}
    //       lab03::launch_window_stats on the CALIBRATED buffer with Mode::fused and config.threads
    //     Call CUDA_CHECK_LAUNCH() after each launch. Nothing returns to the host in between.
    //  4. D2H: download the statistics into a std::vector (times.d2h_ms) and return it.
    (void)raw;
    (void)config;
    (void)times;
    throw std::runtime_error("run_gpu_pipeline is not implemented yet: edit src/lab04/gpu_pipeline.cu "
                             "(or build the reference with make SOLUTION=1).");
}

}  // namespace lab04
