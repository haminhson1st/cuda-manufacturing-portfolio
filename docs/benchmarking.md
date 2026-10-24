# Benchmark protocol

## Define the experiment before running it

State the input size, data type, algorithm, output tolerance, CPU thread count, GPU, and timing boundary. Use the same mathematical operation and input data in the variants you compare. Compile optimized builds and verify correctness before collecting performance numbers.

Keep these measurements distinct:

| Metric | Included | Excluded |
| --- | --- | --- |
| CPU compute | Algorithm on host arrays | File input, initialization unless explicitly added |
| GPU kernel | Device work between CUDA events | Host-device transfer, allocation, file input |
| Transfer + compute | H2D copy, kernels, D2H copy, completion | Allocation, startup, file input in Lab 01 |
| Full application | File read/decode through final report output | Scheduler waiting time |

Lab 01 uses already allocated buffers and an initialized CUDA context for its steady-state measurements. Its `transfer_compute_ms` is **not** complete application latency. Measure cold startup separately when it matters to the deployment.

## Repeatable procedure

1. Record source commit, input identity, module list, compiler flags, CPU model, allocated CPU threads, GPU model, driver, and toolkit version.
2. Initialize the context and warm up the selected operation.
3. Collect at least 20 timed repetitions. Increase the count if measurements are unstable; keep raw observations.
4. Use CUDA events in the appropriate stream for device timing. Synchronize before stopping a host timer that is intended to include GPU completion.
5. Report median and a spread such as min/max or interquartile range. For tail latency, gather enough application requests; 20 kernel runs are not a convincing p99 estimate.
6. Run profiling and sanitizers separately from the normal timing experiment because instrumentation can change execution time.
7. Recheck numerical outputs and application decisions after each optimization.

For multiple streams, record completion of all relevant work before ending the overall timer. Do not sum overlapping per-kernel durations and call that wall-clock latency.

## Derived metrics

Define speedup as `CPU time / GPU time` with the GPU timing boundary named. Throughput is completed useful units divided by elapsed wall-clock time. Keep samples/s, windows/s, and images/s separate. If estimating effective bandwidth, document which logical bytes you counted; it is not automatically the same as measured DRAM traffic.

Use a fair optimized CPU reference. Add a multicore CPU comparison in the capstone and record its thread count. Lab 01's single-threaded CPU loop is an educational starting point.

## Quality and business interpretation

Treat defective items as positives. Defect recall is `TP / (TP + FN)`. False-reject rate is `FP / (FP + TN)` among good items. If a denominator is zero, report the metric as undefined. Keep selection of thresholds separate from final evaluation.

Repeated frames/overlapping windows from the same source are not independent new examples. Split sensor data by recording and document image partitions. Increasing batch size may improve throughput while increasing waiting time for an individual item.

Store reviewed results in a new dated subdirectory of `results/` with [an experiment record](../templates/experiment.md). Include failures and regressions. The [NVIDIA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provides the assess/parallelize/optimize/deploy methodology underlying this workflow.
