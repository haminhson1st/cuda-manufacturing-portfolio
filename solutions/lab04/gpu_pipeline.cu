// Lab 04 reference solution. Compare it with your src/lab04/gpu_pipeline.cu after a working attempt.

#include <limits>

#include "common/cuda_utils.cuh"
#include "lab04/lab04.hpp"

namespace lab04 {

std::vector<lab03::WindowStats> run_gpu_pipeline(const std::vector<float>& raw, const PipelineConfig& config,
                                                 GpuPhaseTimes& times) {
    const std::size_t n = raw.size();
    const std::size_t windows = lab03::window_count(n, config.window);
    const std::vector<float> gain{config.gain};
    const std::vector<float> offset{config.offset};

    common::DeviceBuffer<float> d_raw(n), d_calibrated(n), d_gain(1), d_offset(1);
    common::DeviceBuffer<std::uint8_t> d_alarm(n);
    common::DeviceBuffer<lab03::WindowStats> d_stats(windows);
    common::EventTimer timer;

    timer.start();
    d_raw.upload(raw);
    d_gain.upload(gain);
    d_offset.upload(offset);
    times.h2d_ms = timer.stop();

    const float inf = std::numeric_limits<float>::infinity();
    const lab02::Batch batch{d_raw.get(), d_gain.get(), d_offset.get(), d_calibrated.get(), d_alarm.get(),
                             1, n, lab02::Layout::channel_major};
    timer.start();
    lab02::launch_calibrate(batch, lab02::Limits{-inf, inf}, lab02::LaunchConfig{config.threads, 0});
    CUDA_CHECK_LAUNCH();
    // The calibrated signal stays on the device: the second kernel reads what the first wrote.
    lab03::launch_window_stats(d_calibrated.get(), n, config.window, d_stats.get(), config.threads,
                               lab03::Mode::fused);
    CUDA_CHECK_LAUNCH();
    times.compute_ms = timer.stop();

    std::vector<lab03::WindowStats> stats;
    timer.start();
    d_stats.download(stats);
    times.d2h_ms = timer.stop();
    return stats;
}

}  // namespace lab04
