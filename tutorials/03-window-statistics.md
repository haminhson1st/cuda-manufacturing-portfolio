# 03 — Reduce readings into window statistics

**Time:** 6–8 hours. **Business task:** summarize vibration windows as mean, RMS, and maximum absolute amplitude.

## Read and specify

Study one simple implementation from NVIDIA's [reduction sample](https://github.com/NVIDIA/cuda-samples/tree/master/cpp/2_Concepts_and_Techniques/reduction). Read shared-memory synchronization in [Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html).

For `m` valid samples, define mean as `sum(x)/m`, RMS as `sqrt(sum(x*x)/m)`, and peak as `max(abs(x))`. For a final partial window, divide by the number of valid samples, not the configured full-window length. Reject empty inputs or explicitly define an empty-output behavior.

## Implement in stages

1. Write a CPU reference using double-precision accumulation.
2. Start with one block per window and one statistic: sum.
3. For a power-of-two block size, load valid input into shared memory and fill unused entries with the operation's identity value.
4. Reduce with all block threads participating in required barriers. Predicate memory accesses; do not let out-of-range threads skip a barrier reached by their peers.
5. Add sum-of-squares and peak. Keep the CPU reference independent.
6. Handle windows larger than a block by assigning multiple input elements per thread before the block reduction.

A block barrier does not synchronize separate blocks. If you later split one window over multiple blocks, use partial results and a second reduction launch as a simple implementation.

## Verification

Use all zeros, constants, alternating signs, a single impulse, partial windows, and window lengths smaller/larger than a block. Floating-point reduction order can change results; record absolute/relative tolerances and inspect non-finite output explicitly. Widen the reference precision rather than demanding exact equality for general floats.

Run memcheck, racecheck, and synccheck on representative small cases. Racecheck targets shared-memory hazards; it is not proof that every possible data race in an application is absent. [Sanitizer manual](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).

## Experiment and deliverable

Compare one statistic at a time versus computing the three statistics while reusing loaded values. Report useful windows/s, tolerances, and the workload. Save a notebook explaining synchronization and the partial-window rule.

**Done:** the CPU/GPU statistics agree, the checks pass, and one benchmark is reproducible. Continue to [the sensor case study](04-sensor-project.md).
