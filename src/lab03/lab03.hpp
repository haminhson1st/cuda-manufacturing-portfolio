#pragma once

// Lab 03 — reduce fixed-length windows of a signal to mean, RMS, and peak.
// The contract shared by the harness (main.cu), your kernels (kernels.cu), and Lab 04.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lab03 {

// Statistics of one window. `count` is the number of valid samples: the final window of a
// signal may be partial, and its mean and RMS divide by `count`, not by the window length.
struct WindowStats {
    float mean;
    float rms;
    float peak;  // max |x|
    std::uint32_t count;
};

enum class Mode {
    fused,     // one pass computes the sum, sum of squares, and peak together
    separate,  // three launches, one statistic each (the comparison experiment)
};

inline std::size_t window_count(std::size_t n, std::size_t window) { return (n + window - 1) / window; }

// Implemented in src/lab03/kernels.cu (yours) or solutions/lab03/kernels.cu.
// x has n samples and out has window_count(n, window) entries, both on the device. threads is a
// power of two in [32, 1024]. Enqueues the work in `stream` and returns without synchronizing.
void launch_window_stats(const float* x, std::size_t n, std::size_t window, WindowStats* out, int threads,
                         Mode mode, cudaStream_t stream = nullptr);

// ---- Deterministic test signals -------------------------------------------------------------
inline std::uint32_t hash32(std::uint64_t i) {
    std::uint64_t h = i * 0x9E3779B97F4A7C15ull;
    h ^= h >> 31;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 29;
    return static_cast<std::uint32_t>(h >> 32);
}

// Patterns: zeros, constant, alternating, impulse (one large sample at n/3), ramp, and
// vibration (two tones, broadband noise, and periodic decaying bursts).
inline float signal_value(const std::string& pattern, std::size_t i, std::size_t n) {
    if (pattern == "zeros") return 0.0f;
    if (pattern == "constant") return 2.5f;
    if (pattern == "alternating") return i % 2 == 0 ? 3.0f : -3.0f;
    if (pattern == "impulse") return i == n / 3 ? 1000.0f : 0.0f;
    if (pattern == "ramp") return static_cast<float>(static_cast<double>(i % 1000) * 0.01 - 5.0);
    const double t = static_cast<double>(i);
    const double tones = 0.8 * std::sin(0.0123 * t) + 0.3 * std::sin(0.217 * t + 1.0);
    const double noise = (static_cast<double>(hash32(i) & 0xFFFFu) / 65535.0 - 0.5) * 0.4;
    const std::size_t phase = i % 9973;
    const double burst = phase < 40 ? 4.0 * std::exp(-0.1 * static_cast<double>(phase)) : 0.0;
    return static_cast<float>(tones + noise + burst);
}

inline std::vector<float> make_signal(const std::string& pattern, std::size_t n) {
    std::vector<float> x(n);
    for (std::size_t i = 0; i < n; ++i) x[i] = signal_value(pattern, i, n);
    return x;
}

// ---- Double-precision CPU reference ---------------------------------------------------------
struct Reference {
    double mean;
    double rms;
    double peak;
    double abs_mean;  // mean of |x|: the scale of the summation error bound
    std::uint32_t count;
};

inline void reference_stats(const std::vector<float>& x, std::size_t window, std::vector<Reference>& result) {
    const std::size_t windows = window_count(x.size(), window);
    result.resize(windows);
    for (std::size_t w = 0; w < windows; ++w) {
        const std::size_t begin = w * window;
        const std::size_t end = std::min(x.size(), begin + window);
        double sum = 0.0, sumsq = 0.0, abs_sum = 0.0, peak = 0.0;
        for (std::size_t i = begin; i < end; ++i) {
            const double v = x[i];
            sum += v;
            sumsq += v * v;
            abs_sum += std::abs(v);
            peak = std::max(peak, std::abs(v));
        }
        const double m = static_cast<double>(end - begin);
        result[w] = {sum / m, std::sqrt(sumsq / m), peak, abs_sum / m, static_cast<std::uint32_t>(end - begin)};
    }
}

inline WindowStats to_window_stats(const Reference& r) {
    return {static_cast<float>(r.mean), static_cast<float>(r.rms), static_cast<float>(r.peak), r.count};
}

// ---- Tolerance from a floating-point error bound --------------------------------------------
// A float sum whose longest chain of additions has k steps has error at most gamma(k) * sum|x|,
// with gamma(k) = k*u / (1 - k*u) and u = 2^-24 (Higham, Accuracy and Stability of Numerical
// Algorithms, 2nd ed., section 4.2). Strided per-thread accumulation followed by a tree gives
// k = ceil(m / threads) + log2(threads); +2 covers the divide and square root, and the factor 2
// is slack. The peak involves no rounding, so it must match exactly.
inline double gamma_bound(double k) {
    const double u = std::ldexp(1.0, -24);
    return k * u / (1.0 - k * u);
}

struct Check {
    std::size_t failures = 0;
    std::size_t peak_mismatches = 0;
    double max_mean_error = 0.0;     // |mean - ref| / mean|x|
    double max_rms_rel_error = 0.0;  // |rms - ref| / ref
    std::string first_failure;
};

inline Check validate(const std::vector<WindowStats>& got, const std::vector<Reference>& ref, std::size_t window,
                      int threads) {
    Check check;
    const double depth = std::ceil(static_cast<double>(window) / threads) + std::log2(static_cast<double>(threads)) + 2.0;
    const double tolerance = 2.0 * gamma_bound(depth);
    for (std::size_t w = 0; w < ref.size(); ++w) {
        const WindowStats& g = got[w];
        const Reference& r = ref[w];
        const double mean_error = std::abs(static_cast<double>(g.mean) - r.mean);
        const double rms_error = std::abs(static_cast<double>(g.rms) - r.rms);
        const bool finite = std::isfinite(g.mean) && std::isfinite(g.rms) && std::isfinite(g.peak);
        const bool peak_ok = g.peak == static_cast<float>(r.peak);
        const bool ok = finite && g.count == r.count && peak_ok && mean_error <= tolerance * r.abs_mean
                        && rms_error <= tolerance * r.rms;
        if (finite) {
            check.max_mean_error = std::max(check.max_mean_error, r.abs_mean > 0 ? mean_error / r.abs_mean : mean_error);
            check.max_rms_rel_error = std::max(check.max_rms_rel_error, r.rms > 0 ? rms_error / r.rms : rms_error);
        }
        check.peak_mismatches += !peak_ok;
        if (ok) continue;
        if (check.failures++ == 0) {
            check.first_failure = "window " + std::to_string(w) + ": got count=" + std::to_string(g.count)
                                  + " mean=" + std::to_string(g.mean) + " rms=" + std::to_string(g.rms)
                                  + " peak=" + std::to_string(g.peak) + "; expected count="
                                  + std::to_string(r.count) + " mean=" + std::to_string(r.mean)
                                  + " rms=" + std::to_string(r.rms) + " peak=" + std::to_string(r.peak)
                                  + " (relative tolerance " + std::to_string(tolerance) + ")";
        }
    }
    return check;
}

}  // namespace lab03
