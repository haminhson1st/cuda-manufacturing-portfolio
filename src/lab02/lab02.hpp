#pragma once

// Lab 02 — calibrate many sensor channels and flag out-of-range readings.
// The contract shared by the harness (main.cu), your kernel (kernels.cu), and Lab 04.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lab02 {

// Where element (channel, sample) lives in a flat array of channels * samples values.
enum class Layout {
    channel_major,  // index = channel * samples + sample   (each channel contiguous)
    sample_major,   // index = sample * channels + channel  (interleaved frames, as an ADC delivers them)
};

inline std::size_t element_index(Layout layout, std::size_t channel, std::size_t sample,
                                 std::size_t channels, std::size_t samples) {
    return layout == Layout::channel_major ? channel * samples + sample : sample * channels + channel;
}

// One byte per reading.
enum AlarmCode : std::uint8_t { alarm_none = 0, alarm_low = 1, alarm_high = 2 };

// Inclusive limits: a value exactly on a limit is in range.
struct Limits {
    float low;
    float high;
};

// Grid shape requested by the harness. blocks == 0 means "enough blocks for one thread per
// element"; a positive value fixes the grid, so a grid-stride loop must cover the remainder.
struct LaunchConfig {
    int threads = 256;
    int blocks = 0;
};

// Device pointers for one batch. raw, out, and alarm hold channels * samples elements in
// `layout` order; gain and offset hold one value per channel.
struct Batch {
    const float* raw;
    const float* gain;
    const float* offset;
    float* out;
    std::uint8_t* alarm;
    std::size_t channels;
    std::size_t samples;
    Layout layout;
};

// Implemented in src/lab02/kernels.cu (yours) or solutions/lab02/kernels.cu.
// Enqueues the work in `stream` and returns without synchronizing.
void launch_calibrate(const Batch& batch, Limits limits, LaunchConfig config, cudaStream_t stream = nullptr);

// ---- The scalar rule. The kernel must reproduce it bit for bit. -----------------------------
//   value = fma(gain, raw, offset)   one rounding step, identical on CPU (std::fma) and GPU (fmaf)
//   alarm = alarm_low if value < low, alarm_high if value > high, otherwise alarm_none
//   out   = value clamped to [low, high]   (the alarm uses the value BEFORE clamping)

inline float calibrated_value(float raw, float gain, float offset) { return std::fma(gain, raw, offset); }

inline std::uint8_t alarm_for(float value, Limits limits) {
    return value < limits.low ? alarm_low : (value > limits.high ? alarm_high : alarm_none);
}

inline float clamp_to(float value, Limits limits) { return std::min(std::max(value, limits.low), limits.high); }

// ---- Deterministic inputs -------------------------------------------------------------------
// Channel c has gain 2^((c % 5) - 2) and offset (c % 64)/8 - 4. Both are exact binary fractions,
// so samples can be placed exactly on a limit.
inline float channel_gain(std::size_t c) { return std::ldexp(1.0f, static_cast<int>(c % 5) - 2); }
inline float channel_offset(std::size_t c) { return 0.125f * static_cast<float>(c % 64) - 4.0f; }

// Raw reading for (channel, sample). Samples 0, 1, and 2 of every channel calibrate to exactly
// the high limit, exactly the low limit, and just above the high limit.
inline float raw_reading(std::size_t c, std::size_t s, Limits limits) {
    const float gain = channel_gain(c);
    const float offset = channel_offset(c);
    if (s == 0) return (limits.high - offset) / gain;
    if (s == 1) return (limits.low - offset) / gain;
    if (s == 2) return (limits.high - offset) / gain + 0.25f;
    const auto k = static_cast<long long>((s * 7 + c * 13) % 2048) - 1024;
    return static_cast<float>(k) * 0.25f;
}

// The acquisition system delivers interleaved frames: sample-major order.
inline std::vector<float> make_interleaved(std::size_t channels, std::size_t samples, Limits limits) {
    std::vector<float> raw(channels * samples);
    for (std::size_t s = 0; s < samples; ++s) {
        for (std::size_t c = 0; c < channels; ++c) raw[s * channels + c] = raw_reading(c, s, limits);
    }
    return raw;
}

// Host-side layout conversion, timed separately because an application pays for it too.
inline void interleaved_to_channel_major(const std::vector<float>& interleaved, std::vector<float>& out,
                                         std::size_t channels, std::size_t samples) {
    for (std::size_t s = 0; s < samples; ++s) {
        for (std::size_t c = 0; c < channels; ++c) out[c * samples + s] = interleaved[s * channels + c];
    }
}

// ---- CPU baseline: walks memory in storage order -------------------------------------------
inline void calibrate_cpu(const std::vector<float>& raw, const std::vector<float>& gain,
                          const std::vector<float>& offset, std::vector<float>& out,
                          std::vector<std::uint8_t>& alarm, std::size_t channels, std::size_t samples,
                          Layout layout, Limits limits) {
    const auto apply = [&](std::size_t i, std::size_t c) {
        const float value = calibrated_value(raw[i], gain[c], offset[c]);
        alarm[i] = alarm_for(value, limits);
        out[i] = clamp_to(value, limits);
    };
    if (layout == Layout::channel_major) {
        for (std::size_t c = 0; c < channels; ++c) {
            for (std::size_t s = 0; s < samples; ++s) apply(c * samples + s, c);
        }
    } else {
        for (std::size_t s = 0; s < samples; ++s) {
            for (std::size_t c = 0; c < channels; ++c) apply(s * channels + c, c);
        }
    }
}

// ---- Validation -----------------------------------------------------------------------------
struct Check {
    std::size_t mismatches = 0;
    std::size_t alarms_low = 0;
    std::size_t alarms_high = 0;
    std::string first_mismatch;
};

// Recomputes every reading from its logical (channel, sample) input, so a wrong layout
// mapping fails even when the arithmetic is right.
inline Check validate(const std::vector<float>& out, const std::vector<std::uint8_t>& alarm,
                      std::size_t channels, std::size_t samples, Layout layout, Limits limits) {
    Check check;
    for (std::size_t c = 0; c < channels; ++c) {
        for (std::size_t s = 0; s < samples; ++s) {
            const std::size_t i = element_index(layout, c, s, channels, samples);
            const float value = calibrated_value(raw_reading(c, s, limits), channel_gain(c), channel_offset(c));
            const float expected_out = clamp_to(value, limits);
            const std::uint8_t expected_alarm = alarm_for(value, limits);
            check.alarms_low += alarm[i] == alarm_low;
            check.alarms_high += alarm[i] == alarm_high;
            if (out[i] == expected_out && alarm[i] == expected_alarm) continue;
            if (check.mismatches++ == 0) {
                check.first_mismatch = "channel " + std::to_string(c) + ", sample " + std::to_string(s)
                                       + ": expected out=" + std::to_string(expected_out)
                                       + " alarm=" + std::to_string(expected_alarm)
                                       + "; got out=" + std::to_string(out[i])
                                       + " alarm=" + std::to_string(alarm[i]);
            }
        }
    }
    return check;
}

}  // namespace lab02
