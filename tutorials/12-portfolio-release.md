# 12 — Publish evidence a reviewer can reproduce

**Time:** 6–8 hours. **Business task:** recommend when the GPU implementation is useful and when a CPU is sufficient.

Use [the benchmark protocol](../docs/benchmarking.md), [case-study template](../templates/case-study.md), and [portfolio playbook](../docs/portfolio-playbook.md).

## Freeze and measure

1. Freeze source, input lists, parameters, and build configuration.
2. Compare optimized CPU, straightforward CUDA, and your final CUDA/library version. Add a sensible multicore CPU baseline and record thread counts.
3. Use small inputs, realistic single items, and large batches. Keep complete application timing separate from kernel timing.
4. Re-run correctness checks and the held-out inspection evaluation.
5. Repeat measurements, retain raw observations, and report medians and variability.

If GPUs lose for small inputs or the detector misses defects, include those results. Do not search for a flattering workload and present it as the general case.

## Write the report

Begin with the observed result and its scope. Explain the manufacturing problem, your implementation contribution, the comparison, and remaining limitations. Include a compact architecture sketch if it helps readers locate CPU/GPU work. Explain one rejected optimization.

Estimate business impact only from explicit assumptions. For example, compare measured images/s with the hypothetical arrival rate. Do not translate throughput into financial savings without a stated cost and deployment model.

## Prepare a short demonstration

Show a reproducible run, one correct output, one failure case, the measurement table, and the reason for your final implementation choice. Aim for three to five minutes. Describe Sol as the development/benchmark platform; deployment hardware would require separate validation.

## Publish

Update README statuses and link the report. Create a release/tag corresponding to the evaluated commit, then pin the repository if it represents work you can explain. Record data access instructions rather than uploading restricted raw inputs. Keep AI assistance and upstream attribution visible.

**Completion checklist:** a fresh reader can build the application; test evidence exists; measurements identify hardware and timing scope; accuracy metrics use a held-out evaluation; and no performance or production claims exceed the evidence.

After this course, choose one extension based on observed needs: cuFFT sensor features, improved anomaly detection, advanced kernel optimization, or deployment on a specific edge GPU. Each should start with a concrete workload and acceptance criteria.
