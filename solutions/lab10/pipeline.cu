// Lab 10 reference solution. Compare it with your src/lab10/pipeline.cu after a working attempt.

#include <algorithm>
#include <limits>
#include <memory>

#include "common/cuda_utils.cuh"
#include "common/nvtx.hpp"
#include "lab10/lab10.hpp"

namespace lab10 {
namespace {

// Everything one in-flight batch needs. A slot's buffers are reused only after its event says the
// previous batch that used them has finished.
struct Slot {
    Slot(std::size_t batch_samples, std::size_t batch_windows)
        : host_in(batch_samples), host_out(batch_windows), d_raw(batch_samples), d_calibrated(batch_samples),
          d_alarm(batch_samples), d_stats(batch_windows) {}

    common::PinnedBuffer<float> host_in;
    common::PinnedBuffer<lab03::WindowStats> host_out;
    common::DeviceBuffer<float> d_raw, d_calibrated;
    common::DeviceBuffer<std::uint8_t> d_alarm;
    common::DeviceBuffer<lab03::WindowStats> d_stats;
    common::Stream stream;
    common::Event done;
    bool busy = false;
    BatchRecord record{0, 0, 0.0, 0.0};
};

}  // namespace

RunResult run_streamed(const Workload& workload, std::size_t batch, double interval_ms) {
    const std::size_t wpi = workload.windows_per_item();
    RunResult result;
    result.stats.resize(workload.items * wpi);

    // Allocate everything before the clock starts: allocations between stream commands block overlap.
    common::DeviceBuffer<float> d_gain(1), d_offset(1);
    d_gain.upload(std::vector<float>{workload.gain});
    d_offset.upload(std::vector<float>{workload.offset});
    std::unique_ptr<Slot> slots[2];
    for (auto& slot : slots) slot = std::make_unique<Slot>(batch * workload.item_samples, batch * wpi);
    const float inf = std::numeric_limits<float>::infinity();
    common::Event start_event;

    // Wait for a slot's batch, then move its results out so the slot can be refilled.
    const auto retire = [&](Slot& slot) {
        common::NvtxRange range("retire");
        CUDA_CHECK(cudaEventSynchronize(slot.done.get()));
        slot.record.done_ms = common::event_ms(start_event, slot.done);
        std::copy(slot.host_out.get(), slot.host_out.get() + slot.record.count * wpi,
                  result.stats.begin() + static_cast<std::ptrdiff_t>(slot.record.first_item * wpi));
        result.batches.push_back(slot.record);
        slot.busy = false;
    };

    CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = common::Clock::now();
    CUDA_CHECK(cudaEventRecord(start_event.get(), nullptr));
    std::size_t k = 0;
    for (std::size_t first = 0; first < workload.items; first += batch, ++k) {
        Slot& slot = *slots[k % 2];
        if (slot.busy) retire(slot);
        const std::size_t count = std::min(batch, workload.items - first);
        const std::size_t samples = count * workload.item_samples;
        slot.record = BatchRecord{first, count, arrival_ms(first + count - 1, interval_ms), 0.0};
        wait_until(start, slot.record.ready_ms);
        {
            common::NvtxRange range("fill pinned input");
            std::copy(workload.item(first), workload.item(first) + samples, slot.host_in.get());
        }
        const cudaStream_t stream = slot.stream.get();
        CUDA_CHECK(cudaMemcpyAsync(slot.d_raw.get(), slot.host_in.get(), samples * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        const lab02::Batch calibrate{slot.d_raw.get(), d_gain.get(), d_offset.get(), slot.d_calibrated.get(),
                                     slot.d_alarm.get(), 1, samples, lab02::Layout::channel_major};
        lab02::launch_calibrate(calibrate, lab02::Limits{-inf, inf}, lab02::LaunchConfig{workload.threads, 0}, stream);
        CUDA_CHECK_LAUNCH();
        lab03::launch_window_stats(slot.d_calibrated.get(), samples, workload.window, slot.d_stats.get(),
                                   workload.threads, lab03::Mode::fused, stream);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpyAsync(slot.host_out.get(), slot.d_stats.get(), count * wpi * sizeof(lab03::WindowStats),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(slot.done.get(), stream));
        slot.busy = true;
    }
    for (auto& slot : slots) {
        if (slot->busy) retire(*slot);  // drain, including a final partial batch
    }
    std::sort(result.batches.begin(), result.batches.end(),
              [](const BatchRecord& a, const BatchRecord& b) { return a.first_item < b.first_item; });
    result.total_ms = common::elapsed_ms(start, common::Clock::now());
    return result;
}

}  // namespace lab10
