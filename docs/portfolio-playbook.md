# Make the engineering visible

A portfolio reader should understand the problem, find your contribution, and inspect the evidence in a few minutes.

## What to publish at each milestone

| Milestone | Useful public evidence |
| --- | --- |
| First lab | Correctness log, kernel explanation, one meaningful modification |
| Sensor project | Recording manifest, window statistics, alarm analysis, CPU/GPU timings |
| Image project | Permitted example outputs, failure cases, accuracy table, timing boundaries |
| Capstone | Build instructions, versioned experiment, comparisons, concise recommendation |

Use the top-level README as an entry point. Keep detailed logs in reports, and put large raw data and profiler traces outside Git. Link compact measurements to the implementation commit.

## Tell a defensible story

Explain one decision per experiment: what you suspected, what you changed, what the result showed, and what you would try next. Useful examples include discovering that transfers dominate small workloads, or that a tiled kernel is slower for a small filter.

Do not describe this scaffold as completed manufacturing software. Replace planned status with implemented status only after implementation, and measured status only after a reproducible run. A chart needs measured data, units, hardware, timing scope, and a source file. Do not create an attractive chart from invented speedups.

## A case-study outline

1. Business constraint and workload.
2. CPU baseline and acceptance criteria.
3. CUDA decomposition and one meaningful design tradeoff.
4. Correctness and inspection-quality results.
5. Performance results with transfer and I/O costs visible.
6. Limitations and recommendation.

Use [the case-study template](../templates/case-study.md). Write the final README result sentence after the experiment, not before it.

## Interview preparation

Be ready to explain the mapping from one data item to a thread, why each synchronization is necessary, how bounds are handled, why you chose a library/custom kernel, and how timing was collected. Explain one failed optimization in detail.

For AI-assisted work, keep the origin of the scaffold explicit and identify what you implemented, changed, tested, and interpreted. The ability to understand and defend the work matters more than the number of generated files.

## Resume wording after completion

Use this structure only after filling it with measured evidence:

> Implemented [operation] in CUDA C++ on [GPU], achieving [measured result with timing boundary] versus [specified CPU baseline] on [workload], with [correctness/quality result].

Until then, describe the repository as an ongoing CUDA manufacturing learning project. Do not claim production deployment, commercial savings, or validated fault diagnosis based only on a laboratory benchmark.
