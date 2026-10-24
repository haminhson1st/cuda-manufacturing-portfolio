# 11 — Assemble a reproducible inspection application

**Time:** 6–8 hours. **Business task:** let another engineer process an image collection or sensor recording with one documented command.

Read [the capstone brief](../projects/03-capstone/README.md). Prioritize integration of existing work. Image inspection is the primary mode; sensor screening is a second mode using an independent dataset.

## Define the interface before wiring it

Design an executable named `inspect`. The following is a target interface you must implement, not a command currently supplied by this scaffold:

```text
inspect images --input INPUT_DIRECTORY --backend cpu|cuda --output REPORT.csv
inspect sensors --input RECORDING --window WINDOW_LENGTH --backend cpu|cuda --output REPORT.csv
```

Add explicit parameters/configuration for normalization, thresholds, batch size, and selected implementation. Export the effective configuration with each run.

## Implement

1. Put reusable numerical kernels behind small interfaces and keep file handling in host code.
2. Add deterministic input enumeration and stable output IDs.
3. Validate dimensions, missing files, unsupported formats, and invalid parameter values.
4. Preserve a CPU path that can reproduce the algorithm's output.
5. Add tests for one input, partial batches, malformed input, and threshold boundaries.
6. Adapt the supplied Slurm script for the capstone and record its resource request.

Do not pair MVTec images with CWRU windows as if they describe the same machine. They are separate workflows sharing engineering techniques.

## Define a business workload

Use an explicitly hypothetical target, such as four cameras at 25 images/s each. Define resolution, acceptable latency, error tolerance, and how inputs arrive. Offline throughput alone cannot establish that a live factory deadline is met.

## Deliverable

Ask a reader to follow your instructions from a fresh clone. They should be able to identify data prerequisites, compile, run a small test, and generate a report. Log any manual steps that were missing.

**Acceptance:** reproducible CLI, CPU/GPU numerical agreement, frozen detector parameters, documented quality evaluation, and an integration test. Continue to [the portfolio release](12-portfolio-release.md).
