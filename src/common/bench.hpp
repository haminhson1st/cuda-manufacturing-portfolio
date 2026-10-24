#pragma once

// Host-side timing, summaries, and raw-observation output shared by Labs 02+.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace common {

using Clock = std::chrono::steady_clock;

inline double elapsed_ms(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

inline double median(std::vector<double> values) {
    if (values.empty()) throw std::logic_error("No timing observations.");
    std::sort(values.begin(), values.end());
    const auto mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2.0;
}

// Runs work() `warmups` times, then returns the wall-clock time of each of `reps` repetitions.
template <typename Work>
std::vector<double> time_host(int warmups, int reps, Work&& work) {
    for (int i = 0; i < warmups; ++i) work();
    std::vector<double> times;
    for (int i = 0; i < reps; ++i) {
        const auto start = Clock::now();
        work();
        times.push_back(elapsed_ms(start, Clock::now()));
    }
    return times;
}

using Series = std::vector<std::pair<std::string, std::vector<double>>>;

// Long format shared with Lab 01: metric,repetition,milliseconds.
inline void write_observations(const std::string& path, const Series& series) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Cannot open raw timing file " + path + '.');
    output << "metric,repetition,milliseconds\n" << std::setprecision(9);
    for (const auto& [metric, values] : series) {
        for (std::size_t i = 0; i < values.size(); ++i) output << metric << ',' << i << ',' << values[i] << '\n';
    }
    output.close();
    if (!output) throw std::runtime_error("Failed writing raw timing file " + path + '.');
}

}  // namespace common
