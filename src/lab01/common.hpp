#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lab01 {
constexpr float gain = 1.25f;
constexpr float offset = -0.5f;
constexpr int repetitions = 20;
constexpr int warmups = 3;
using Clock = std::chrono::steady_clock;

inline std::size_t parse_count(int argc, char** argv) {
    if (argc > 3) {
        throw std::invalid_argument("Usage: calibrate_cpu|calibrate_cuda [positive_N [raw_timings.csv]]");
    }
    if (argc == 1) return 1000003;
    const std::string value(argv[1]);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("N must be a positive integer.");
    }
    std::size_t used = 0;
    const auto n = std::stoull(value, &used);
    if (used != value.size() || n == 0 || n > 100000000ULL) {
        throw std::invalid_argument("Use 1 <= N <= 100000000 for this introductory lab.");
    }
    return static_cast<std::size_t>(n);
}

inline std::vector<float> make_input(std::size_t n) {
    std::vector<float> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        const int centered = static_cast<int>(i % 4096) - 2048;
        x[i] = static_cast<float>(centered) * 0.125f;
    }
    return x;
}

inline void calibrate_cpu(const std::vector<float>& x, std::vector<float>& y) {
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = gain * x[i] + offset;
}

struct Check {
    bool correct = true;
    double max_abs_error = 0.0;
};

inline Check validate(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size()) throw std::logic_error("Output size differs from input.");
    Check check;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double expected = static_cast<double>(gain) * x[i] + offset;
        const double error = std::abs(static_cast<double>(y[i]) - expected);
        if (!std::isfinite(y[i]) || error > 1e-6 + 1e-5 * std::abs(expected)) {
            check.correct = false;
        }
        check.max_abs_error = std::max(check.max_abs_error,
            std::isfinite(error) ? error : std::numeric_limits<double>::infinity());
    }
    return check;
}

inline double elapsed_ms(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

inline double median(std::vector<double> values) {
    if (values.empty()) throw std::logic_error("No timing observations.");
    std::sort(values.begin(), values.end());
    const auto mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2.0;
}

inline void print_header() {
    std::cout << "backend,n,repetitions,cpu_ms,kernel_ms,transfer_compute_ms,max_abs_error,correct\n";
    std::cout << std::setprecision(9) << std::boolalpha;
}

inline void write_observations(const char* path,
    const std::vector<std::pair<std::string, std::vector<double>>>& series) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Cannot open raw timing output file.");
    output << "metric,repetition,milliseconds\n" << std::setprecision(9);
    for (const auto& item : series) {
        for (std::size_t i = 0; i < item.second.size(); ++i) {
            output << item.first << ',' << i << ',' << item.second[i] << '\n';
        }
    }
    output.close();
    if (!output) throw std::runtime_error("Failed writing raw timing output file.");
}
}  // namespace lab01
