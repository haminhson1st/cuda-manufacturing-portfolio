// Lab 10 baseline, provided: one batch at a time on the legacy default stream, copying straight
// from pageable host memory. Every step waits for the previous one, so nothing overlaps. Read it
// before writing run_streamed: your version must return identical results.

#include <algorithm>
#include <limits>

#include "common/cuda_utils.cuh"
#include "common/nvtx.hpp"
#include "lab10/lab10.hpp"

namespace lab10 {

RunResult run_serial(const Workload& workload, std::size_t batch, double interval_ms) {
    const std::size_t wpi = workload.windows_per_item();
    const std::size_t batch_samples = batch * workload.item_samples;
    RunResult result;
    result.stats.resize(workload.items * wpi);

    // Everything is allocated before the clock starts.
    common::DeviceBuffer<float> d_raw(batch_samples), d_calibrated(batch_samples), d_gain(1), d_offset(1);
    common::DeviceBuffer<std::uint8_t> d_alarm(batch_samples);
    common::DeviceBuffer<lab03::WindowStats> d_stats(batch * wpi);
    d_gain.upload(std::vector<float>{workload.gain});
    d_offset.upload(std::vector<float>{workload.offset});
    const float inf = std::numeric_limits<float>::infinity();
    common::Event start_event, done_event;

    CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = common::Clock::now();
    CUDA_CHECK(cudaEventRecord(start_event.get(), nullptr));
    for (std::size_t first = 0; first < workload.items; first += batch) {
        const std::size_t count = std::min(batch, workload.items - first);
        const std::size_t samples = count * workload.item_samples;
        BatchRecord record{first, count, arrival_ms(first + count - 1, interval_ms), 0.0};
        wait_until(start, record.ready_ms);
        common::NvtxRange range("serial batch");

        CUDA_CHECK(cudaMemcpy(d_raw.get(), workload.item(first), samples * sizeof(float), cudaMemcpyHostToDevice));
        const lab02::Batch calibrate{d_raw.get(), d_gain.get(), d_offset.get(), d_calibrated.get(), d_alarm.get(),
                                     1, samples, lab02::Layout::channel_major};
        lab02::launch_calibrate(calibrate, lab02::Limits{-inf, inf}, lab02::LaunchConfig{workload.threads, 0});
        CUDA_CHECK_LAUNCH();
        lab03::launch_window_stats(d_calibrated.get(), samples, workload.window, d_stats.get(), workload.threads,
                                   lab03::Mode::fused);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(result.stats.data() + first * wpi, d_stats.get(), count * wpi * sizeof(lab03::WindowStats),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(done_event.get(), nullptr));
        CUDA_CHECK(cudaEventSynchronize(done_event.get()));
        record.done_ms = common::event_ms(start_event, done_event);
        result.batches.push_back(record);
    }
    result.total_ms = common::elapsed_ms(start, common::Clock::now());
    return result;
}

}  // namespace lab10
