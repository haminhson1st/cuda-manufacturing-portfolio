# Project 3 — Manufacturing inspection application

**Status:** planned; integration exercises are Labs 08–12.

## Problem

An engineer needs a reproducible inspection tool and evidence for selecting a CPU or GPU deployment. Build one application with image-inspection and sensor-screening modes, using the existing projects.

## Proposed workload

Use a clearly hypothetical example of four cameras at 25 images/s each. Specify resolution, arrival pattern, latency requirement, and acceptable inspection error before evaluating the target. A batch benchmark does not by itself demonstrate a live camera deadline.

The image and sensor datasets are independent. Combining their implementations in one application does not create synchronized factory data.

## Acceptance criteria

- [ ] CLI validates inputs and records effective parameters.
- [ ] CPU and CUDA modes implement equivalent algorithms.
- [ ] A multicore CPU comparison records its thread count.
- [ ] Correctness and inspection-quality checks survive optimizations.
- [ ] Batching/stream experiments measure throughput and latency.
- [ ] Build, Slurm job, data manifest, and measurement protocol are reproducible.
- [ ] Final recommendation names workload sizes where GPU execution helps or loses.
- [ ] Release notes identify hardware, source revision, limitations, and authorship.

## Final presentation

Deliver a concise report and a three-to-five-minute demonstration. Include one rejected optimization and one detector failure. Discuss deployment on eventual factory hardware as future validation, not a completed claim.

Follow [Lab 11](../../tutorials/11-capstone.md) and [Lab 12](../../tutorials/12-portfolio-release.md). No final application is implemented in this scaffold yet.
