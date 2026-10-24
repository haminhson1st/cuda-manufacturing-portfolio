// Lab 11 reference solution. Compare it with your src/lab11/app.cu after a working attempt.

#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <set>
#include <stdexcept>

#include "common/bench.hpp"
#include "lab09/lab09.hpp"
#include "lab11/lab11.hpp"

namespace lab11 {
namespace {

struct Input {
    std::string id;
    std::string path;
};

std::vector<Input> enumerate(const ImagesConfig& config) {
    std::vector<Input> inputs;
    if (ends_with(config.input, ".csv")) {
        for (const auto& entry : lab07::read_image_list(config.input)) {
            if (config.split == "all" || entry.split == config.split) inputs.push_back({entry.image_id, entry.path});
        }
    } else if (ends_with(config.input, ".pgm") || ends_with(config.input, ".ppm")) {
        inputs.push_back({stem(config.input), config.input});
    } else {
        for (const auto& path : list_images(config.input)) inputs.push_back({stem(path), path});
    }
    if (inputs.empty()) throw std::runtime_error("No images found in " + config.input + '.');
    std::set<std::string> ids;
    for (const auto& input : inputs) {
        if (!ids.insert(input.id).second) {
            throw std::runtime_error("Two inputs share the id '" + input.id + "'; report ids must be unique.");
        }
    }
    return inputs;
}

}  // namespace

RunSummary run_images(const ImagesConfig& config) {
    const std::vector<Input> inputs = enumerate(config);
    std::ofstream out(config.output);
    if (!out) throw std::runtime_error("Cannot open report " + config.output + '.');
    out << "image_id,source,width,height,mask_pixels,score,predicted_defect\n" << std::setprecision(9);

    const bool run_cpu = config.backend != Backend::cuda;
    const bool run_gpu = config.backend != Backend::cpu;
    const bool decide = !std::isnan(config.image_threshold);
    std::unique_ptr<lab09::Inspector> inspector;
    if (run_gpu) inspector = std::make_unique<lab09::Inspector>(config.detector);  // buffers reused across images

    RunSummary summary;
    for (const auto& input : inputs) {
        auto phase = common::Clock::now();
        const common::HostImage image = common::read_pnm(input.path);
        const double decode_ms = common::elapsed_ms(phase, common::Clock::now());

        common::HostImage mask;
        common::HostImage* mask_out = config.mask_dir.empty() ? nullptr : &mask;
        phase = common::Clock::now();
        unsigned long long count = 0;
        if (run_gpu) {
            lab07::GpuPhaseTimes times;
            count = inspector->score(image, times, mask_out);
        }
        if (run_cpu) {
            const unsigned long long cpu_count = lab07::score_cpu(image, config.detector, run_gpu ? nullptr : mask_out);
            if (run_gpu && cpu_count != count) ++summary.disagreements;
            if (!run_gpu) count = cpu_count;
        }
        const double compute_ms = common::elapsed_ms(phase, common::Clock::now());

        const double score = static_cast<double>(count) / (static_cast<double>(image.width) * image.height);
        const bool flagged = decide && score > config.image_threshold;
        out << input.id << ',' << base_name(input.path) << ',' << image.width << ',' << image.height << ',' << count
            << ',' << score << ',';
        if (decide) out << (flagged ? 1 : 0);
        out << '\n';
        if (mask_out != nullptr) common::write_pnm(config.mask_dir + "/" + input.id + "-mask.pgm", mask);

        ++summary.items;
        summary.flagged += flagged;
        summary.decode_ms += decode_ms;
        summary.compute_ms += compute_ms;
        summary.item_ms.push_back(decode_ms + compute_ms);
    }
    out.close();
    if (!out) throw std::runtime_error("Failed writing report " + config.output + '.');
    return summary;
}

RunSummary run_sensors(const SensorsConfig& config) {
    RunSummary summary;
    auto phase = common::Clock::now();
    const std::vector<float> raw = lab04::read_f32(config.input);
    summary.decode_ms = common::elapsed_ms(phase, common::Clock::now());

    phase = common::Clock::now();
    const bool run_cpu = config.backend != Backend::cuda;
    const bool run_gpu = config.backend != Backend::cpu;
    std::vector<lab03::Reference> cpu;
    std::vector<lab03::WindowStats> stats;
    if (run_cpu) cpu = lab04::run_cpu_pipeline(raw, config.pipeline);
    if (run_gpu) {
        lab04::GpuPhaseTimes times;
        stats = lab04::run_gpu_pipeline(raw, config.pipeline, times);
    } else {
        for (const auto& reference : cpu) stats.push_back(lab03::to_window_stats(reference));
    }
    if (run_cpu && run_gpu) {
        summary.disagreements = lab03::validate(stats, cpu, config.pipeline.window, config.pipeline.threads).failures;
        for (std::size_t w = 0; w < cpu.size(); ++w) {
            const bool gpu_alarm = lab04::alarm(stats[w].rms, stats[w].peak, config.rule);
            const bool cpu_alarm = lab04::alarm(cpu[w].rms, cpu[w].peak, config.rule);
            if (gpu_alarm != cpu_alarm && !lab04::near_threshold(cpu[w].rms, cpu[w].peak, config.rule, 1e-5)) {
                ++summary.disagreements;
            }
        }
    }
    summary.compute_ms = common::elapsed_ms(phase, common::Clock::now());

    lab04::write_report(config.output, config.id, config.pipeline.window, stats, config.rule);
    summary.items = stats.size();
    for (const auto& window : stats) summary.flagged += lab04::alarm(window.rms, window.peak, config.rule);
    summary.item_ms.push_back(summary.decode_ms + summary.compute_ms);
    return summary;
}

}  // namespace lab11
