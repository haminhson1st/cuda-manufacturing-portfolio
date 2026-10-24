# 04 — Ship the sensor-screening case study

**Time:** 6–8 hours. **Business task:** flag unusual windows in recorded bearing vibration. This is a laboratory screening exercise, not a validated predictor of remaining machine life.

Read the [project brief](../projects/01-sensor-screening/README.md) and obtain a small selection of normal/faulty recordings from [CWRU](https://engineering.case.edu/bearingdatacenter). Preserve recording names, load conditions, sampling information, and source attribution.

## Build a usable pipeline

1. Convert the selected source files once into a documented binary or CSV format using a small preprocessing script. Keep parsing on the CPU.
2. Record source hashes and conversion steps in [a dataset manifest](../templates/dataset-manifest.json).
3. Add command-line controls for file, window length, threshold, and CPU/GPU backend.
4. Reuse calibration and window statistics from Labs 02–03.
5. Export `recording_id,window_start,valid_samples,mean,rms,peak,alarm`.
6. Select an alarm rule using development recordings. Freeze it before evaluating held-out recordings.

Do not randomly split adjacent windows from the same recording across development and evaluation. Keep complete recordings together. Record differences in load/speed so the report does not silently attribute every difference to fault status.

## Evaluate two different questions

First, do CPU and GPU implementations return equivalent statistics and decisions? Second, does the chosen alarm separate the selected normal/faulty recordings usefully? Correct GPU code can implement an ineffective detector; report both outcomes.

Benchmark actual input lengths. For a scaling experiment, batching/replaying recordings is acceptable if clearly labeled; it adds compute work, not independent fault examples.

## Publish

Create a case study using [the template](../templates/case-study.md). Include one result table, one failure example, an explanation of threshold selection, and compute/transfer/full-pipeline timings. Keep raw recordings untracked. Show the commands and source commit.

**Acceptance:** one command produces the CSV, held-out evaluation is documented, CPU/GPU agreement passes, and your performance claim names its timing boundary. Update the project status only when these artifacts exist.

**Extension:** add frequency-domain features with cuFFT after the core course. Continue to [image indexing](05-image-indexing.md).
