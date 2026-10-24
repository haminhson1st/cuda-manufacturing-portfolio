#pragma once

// Lab 10 — process a recorded sequence of sensor items in batches: first serially, then with two
// stream "slots" so that one batch's host work and transfers overlap another batch's kernels.
// Each item is calibrated (Lab 02) and reduced to window statistics (Lab 03). An item's length is a
// multiple of the window, so windows never cross items and concatenating items changes nothing.

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "common/bench.hpp"
#include "lab02/lab02.hpp"
#include "lab03/lab03.hpp"

namespace lab10 {

struct Workload {
    std::size_t items = 256;
    std::size_t item_samples = 65536;  // a multiple of window
    std::size_t window = 1024;
    float gain = 1.25f;
    float offset = -0.5f;
    int threads = 256;
    std::vector<float> source;  // the "recording": items * item_samples raw samples, item after item

    std::size_t windows_per_item() const { return item_samples / window; }
    const float* item(std::size_t id) const { return source.data() + id * item_samples; }
};

// Item `id` is a vibration-like signal plus an offset unique to the item, so a batch that reads or
// writes the wrong buffer produces visibly wrong statistics.
inline void make_source(Workload& workload) {
    workload.source.resize(workload.items * workload.item_samples);
    for (std::size_t id = 0; id < workload.items; ++id) {
        const double item_offset = 0.01 * static_cast<double>(id % 101);
        for (std::size_t j = 0; j < workload.item_samples; ++j) {
            const std::size_t i = id * workload.item_samples + j;
            const double t = static_cast<double>(j);
            const double noise = (static_cast<double>(lab03::hash32(i) & 0xFFFFu) / 65535.0 - 0.5) * 0.4;
            workload.source[i] = static_cast<float>(0.8 * std::sin(0.0123 * t + 0.1 * static_cast<double>(id))
                                                    + 0.3 * std::sin(0.217 * t) + noise + item_offset);
        }
    }
}

// ---- Arrivals --------------------------------------------------------------------------------
// Item i becomes available interval_ms * i after the start. interval_ms == 0 models offline
// processing: every item is already available.
inline double arrival_ms(std::size_t item, double interval_ms) { return interval_ms * static_cast<double>(item); }

inline void wait_until(common::Clock::time_point start, double ms) {
    if (ms <= 0.0) return;
    std::this_thread::sleep_until(
        start + std::chrono::duration_cast<common::Clock::duration>(std::chrono::duration<double, std::milli>(ms)));
}

// ---- Results ---------------------------------------------------------------------------------
struct BatchRecord {
    std::size_t first_item;
    std::size_t count;
    double ready_ms;  // when the batch's last item had arrived (host clock, from the start)
    double done_ms;   // when its results were back in host memory (CUDA events, from the start event)
};

struct RunResult {
    std::vector<lab03::WindowStats> stats;  // items * windows_per_item, in item order
    std::vector<BatchRecord> batches;       // sorted by first_item
    double total_ms = 0.0;
};

// Provided baseline (src/lab10/serial.cu): one stream, pageable host memory, one batch at a time.
RunResult run_serial(const Workload& workload, std::size_t batch, double interval_ms);

// Yours (src/lab10/pipeline.cu) or solutions/lab10/pipeline.cu: two slots, each with pinned host
// buffers, device buffers, a stream, and a completion event.
RunResult run_streamed(const Workload& workload, std::size_t batch, double interval_ms);

// ---- CPU reference over the whole recording ---------------------------------------------------
inline std::vector<lab03::Reference> reference(const Workload& workload) {
    std::vector<float> calibrated(workload.source.size());
    for (std::size_t i = 0; i < calibrated.size(); ++i) {
        calibrated[i] = lab02::calibrated_value(workload.source[i], workload.gain, workload.offset);
    }
    std::vector<lab03::Reference> result;
    lab03::reference_stats(calibrated, workload.window, result);
    return result;
}

}  // namespace lab10
