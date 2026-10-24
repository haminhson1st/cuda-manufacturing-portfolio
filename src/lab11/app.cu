// Lab 11 — YOUR capstone integration: the two workflows behind `inspect`.
// Build: make lab11   Test: make test-lab11   Sanitize: make sanitize-lab11
// Read tutorials/11-capstone.md first. The reference solution is in solutions/lab11/app.cu.

#include <stdexcept>

#include "lab09/lab09.hpp"
#include "lab11/lab11.hpp"

namespace lab11 {

RunSummary run_images(const ImagesConfig& config) {
    // TODO(lab11, images): integrate Labs 05-07 and 09 into one documented workflow.
    //  1. Enumerate inputs deterministically, each with a stable, unique id:
    //       a .csv input        -> lab07::read_image_list, keeping rows of config.split ("all" keeps
    //                              every row); id = image_id
    //       a .pgm/.ppm input   -> that one image; id = stem(path)
    //       anything else       -> list_images(directory); id = stem(path)
    //     Fail with a clear message when nothing is found or two inputs share an id (a.pgm and
    //     a.ppm in one directory do).
    //  2. Open config.output and write the header
    //       image_id,source,width,height,mask_pixels,score,predicted_defect
    //  3. For each input, in order: decode it with common::read_pnm (which rejects unsupported and
    //     malformed files), then score it:
    //       cpu  -> lab07::score_cpu
    //       cuda -> ONE lab09::Inspector created before the loop, so device buffers are reused
    //       both -> both; count the image in disagreements when the mask counts differ
    //     score = mask_pixels / (width * height); predicted_defect = score > image_threshold, left
    //     empty when the threshold is NaN; source = base_name(path). Write the mask to
    //     <mask_dir>/<id>-mask.pgm when mask_dir is set.
    //  4. Return the RunSummary: items, flagged, disagreements, decode_ms and compute_ms totals,
    //     and item_ms (decode + compute of each image).
    (void)config;
    throw std::runtime_error("run_images is not implemented yet: edit src/lab11/app.cu.");
}

RunSummary run_sensors(const SensorsConfig& config) {
    // TODO(lab11, sensors): integrate Labs 02-04.
    //  1. Decode the recording with lab04::read_f32 (it rejects truncated and non-finite input).
    //  2. cpu  -> lab04::run_cpu_pipeline, converted with lab03::to_window_stats
    //     cuda -> lab04::run_gpu_pipeline
    //     both -> both; count windows whose statistics fail lab03::validate, or whose alarm
    //             decisions differ away from a threshold (see src/lab04/screen.cu), as disagreements.
    //  3. Write the report with lab04::write_report, using the GPU statistics when the GPU ran.
    //  4. Return the RunSummary: items = windows, flagged = alarmed windows, decode_ms,
    //     compute_ms, and a single item_ms value for the recording.
    (void)config;
    throw std::runtime_error("run_sensors is not implemented yet: edit src/lab11/app.cu.");
}

}  // namespace lab11
