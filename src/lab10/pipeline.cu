// Lab 10 — YOUR double-buffered stream pipeline.
// Build: make lab10   Test: make test-lab10   Sanitize: make sanitize-lab10   Measure: make bench-lab10
// Read tutorials/10-streaming.md and src/lab10/serial.cu first. The reference solution is in
// solutions/lab10/pipeline.cu; make a time-boxed attempt before opening it.

#include <stdexcept>

#include "common/cuda_utils.cuh"
#include "common/nvtx.hpp"
#include "lab10/lab10.hpp"

namespace lab10 {

RunResult run_streamed(const Workload& workload, std::size_t batch, double interval_ms) {
    // TODO(lab10): two slots used alternately, batch k in slot k % 2.
    //
    // A slot owns, all allocated BEFORE the clock starts (an allocation between stream commands
    // also blocks concurrency):
    //   common::PinnedBuffer<float>              batch * item_samples raw samples
    //   common::PinnedBuffer<lab03::WindowStats> batch * windows_per_item results
    //   device buffers: raw, calibrated, one alarm byte per sample, statistics
    //   common::Stream and common::Event, plus bookkeeping: busy, first item, count, ready_ms.
    // Gain and offset (one float each) can be shared by both slots.
    //
    // 1. Record a start event (cudaEventRecord(start.get(), nullptr)) and note the host start time,
    //    as run_serial does.
    // 2. For each batch [first, first + count):
    //    a. If its slot is busy, RETIRE it: cudaEventSynchronize(slot event), done_ms =
    //       common::event_ms(start, slot event), copy the pinned results into
    //       result.stats[first * windows_per_item ...], and append its BatchRecord. Only now
    //       may the slot's buffers be refilled.
    //    b. wait_until(start, arrival of the batch's last item), then std::copy the items from
    //       workload.item(first) into the slot's pinned input.
    //    c. Enqueue in the SLOT'S stream: cudaMemcpyAsync host->device, lab02::launch_calibrate
    //       and lab03::launch_window_stats (pass the stream as their last argument!),
    //       cudaMemcpyAsync device->host into the pinned results, and cudaEventRecord(slot event,
    //       slot stream). Mark the slot busy.
    // 3. Drain: retire every busy slot, so the final (possibly partial) batch is not dropped.
    // 4. Sort result.batches by first_item and set result.total_ms from the host clock.
    (void)workload;
    (void)batch;
    (void)interval_ms;
    throw std::runtime_error("run_streamed is not implemented yet: edit src/lab10/pipeline.cu "
                             "(or build the reference with make SOLUTION=1).");
}

}  // namespace lab10
