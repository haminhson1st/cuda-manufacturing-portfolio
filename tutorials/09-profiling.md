# 09 — Find the bottleneck before optimizing

**Time:** 6–8 hours. **Business task:** improve elapsed inspection time without changing the accepted output.

Read the assess/parallelize/optimize/deploy discussion in [NVIDIA's Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html). Review [the measurement protocol](../docs/benchmarking.md).

## Capture a baseline

Use a fixed input list and frozen detector parameters. Save a source commit. Record normal application timing first, outside the profiler. Then, inside a GPU allocation, profile the executable you built. For the supplied Lab 01 executable:

```bash
nsys profile --trace=cuda,nvtx,osrt --output=runs/lab01-timeline \
  ./build-gpu/calibrate_cuda 1000003
ncu --set basic --launch-count 1 ./build-gpu/calibrate_cuda 1000003
```

Create `runs/` first and use a fresh output name for each experiment. Replace the executable with your inspection target when ready. The one-launch Nsight Compute command is a small diagnostic capture, not a representative application benchmark.

If tools are absent, inspect installed modules. If counters are restricted, consult ASU RC and record the limitation; do not attempt to change shared-system permissions.

## Ask questions from the timeline

How much time is spent reading/decoding files? Are transfers repeated unnecessarily? Are allocations inside the image loop? Are CPU and GPU idle between stages? Which kernel consumes the most wall-clock time?

## Change exactly one thing

Choose a measured bottleneck: reuse buffers, keep intermediates on the GPU, combine compatible pixel operations, or improve an access pattern. Write a prediction before editing. Re-run numerical and decision-level checks, then time outside the profiler under the same conditions.

## Deliverable

Add a short before/after report containing the bottleneck evidence, change, observed improvement or regression, and next decision. Publish a compact trace screenshot only if it adds explanation and reveals no private paths/data; raw traces remain untracked.

**Acceptance:** your claim concerns measured application behavior, with no guarantee inferred merely from occupancy or another isolated counter. Continue to [streams](10-streaming.md).
