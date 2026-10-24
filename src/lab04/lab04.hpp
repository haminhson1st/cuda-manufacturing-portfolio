#pragma once

// Lab 04 — sensor-screening pipeline: recording -> calibration (Lab 02) -> window statistics
// (Lab 03) -> per-window alarm -> CSV report.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "lab02/lab02.hpp"
#include "lab03/lab03.hpp"

namespace lab04 {

struct PipelineConfig {
    std::size_t window = 2048;
    float gain = 1.0f;
    float offset = 0.0f;
    int threads = 256;
};

// Device-side phase times for one GPU pipeline run, in milliseconds (CUDA events).
struct GpuPhaseTimes {
    double h2d_ms = 0.0;
    double compute_ms = 0.0;
    double d2h_ms = 0.0;
};

// Implemented in src/lab04/gpu_pipeline.cu (yours) or solutions/lab04/gpu_pipeline.cu.
// Calibrates `raw` with Lab 02's kernel (one channel), reduces the calibrated signal with Lab 03's
// fused kernel, and returns one WindowStats per window.
std::vector<lab03::WindowStats> run_gpu_pipeline(const std::vector<float>& raw, const PipelineConfig& config,
                                                 GpuPhaseTimes& times);

// CPU path with the same rules: Lab 02's fma calibration, then the double-precision Lab 03 reference.
inline std::vector<lab03::Reference> run_cpu_pipeline(const std::vector<float>& raw, const PipelineConfig& config) {
    std::vector<float> calibrated(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        calibrated[i] = lab02::calibrated_value(raw[i], config.gain, config.offset);
    }
    std::vector<lab03::Reference> stats;
    lab03::reference_stats(calibrated, config.window, stats);
    return stats;
}

// ---- Alarm rule -----------------------------------------------------------------------------
// A window alarms when any enabled feature exceeds its threshold; disabled thresholds are +inf.
// Crest factor (peak / RMS) rises with impulsive content such as bearing-defect impacts.
struct AlarmRule {
    double rms = std::numeric_limits<double>::infinity();
    double peak = std::numeric_limits<double>::infinity();
    double crest = std::numeric_limits<double>::infinity();
};

inline double crest_factor(double peak, double rms) { return rms > 0.0 ? peak / rms : 0.0; }

inline bool alarm(double rms, double peak, const AlarmRule& rule) {
    return rms > rule.rms || peak > rule.peak || crest_factor(peak, rms) > rule.crest;
}

// True when a feature lies within `relative` of an enabled threshold, so float rounding could
// legitimately flip the decision. Such windows are reported, not counted as disagreements.
inline bool near_threshold(double rms, double peak, const AlarmRule& rule, double relative) {
    const auto near = [relative](double value, double threshold) {
        return std::isfinite(threshold) && std::abs(value - threshold) <= relative * std::max(1e-30, std::abs(threshold));
    };
    return near(rms, rule.rms) || near(peak, rule.peak) || near(crest_factor(peak, rms), rule.crest);
}

// ---- Files ------------------------------------------------------------------------------------
// A recording is raw little-endian float32 samples (.f32), as written by scripts/cwru_to_f32.py
// and scripts/lab04_synthetic.py. Metadata lives in the JSON sidecar, not in this file.
inline std::vector<float> read_f32(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("Cannot open recording " + path + '.');
    const std::streamoff bytes = static_cast<std::streamoff>(in.tellg());
    if (bytes <= 0 || bytes % 4 != 0) {
        throw std::runtime_error(path + ": size must be a positive multiple of 4 bytes (float32 samples).");
    }
    std::vector<float> samples(static_cast<std::size_t>(bytes / 4));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(samples.data()), static_cast<std::streamsize>(bytes));
    if (!in) throw std::runtime_error("Failed reading " + path + '.');
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (!std::isfinite(samples[i])) throw std::runtime_error(path + ": sample " + std::to_string(i) + " is not finite.");
    }
    return samples;
}

inline void write_report(const std::string& path, const std::string& recording_id, std::size_t window,
                         const std::vector<lab03::WindowStats>& stats, const AlarmRule& rule) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open report " + path + '.');
    out << "recording_id,window_start,valid_samples,mean,rms,peak,crest,alarm\n" << std::setprecision(9);
    for (std::size_t w = 0; w < stats.size(); ++w) {
        const lab03::WindowStats& s = stats[w];
        out << recording_id << ',' << w * window << ',' << s.count << ',' << s.mean << ',' << s.rms << ',' << s.peak
            << ',' << crest_factor(s.peak, s.rms) << ',' << (alarm(s.rms, s.peak, rule) ? 1 : 0) << '\n';
    }
    out.close();
    if (!out) throw std::runtime_error("Failed writing report " + path + '.');
}

}  // namespace lab04
