# 08 — Choose between a custom kernel and a library

**Time:** 6–8 hours. **Business task:** simplify a frequently used aggregation without losing correctness or performance.

Read the overview and examples in [NVIDIA CCCL](https://github.com/NVIDIA/cccl), which includes Thrust and CUB. Use APIs available in your loaded toolkit; record that version.

## Keep the comparison narrow

Choose one operation already implemented: summing a defect mask, summing squared sensor readings, or computing a maximum. Define its input/output types and numerical tolerance before switching implementations.

## Implement

1. Add a backend choice such as `custom` or `library` to that operation.
2. Implement the library path using the appropriate Thrust/CUB reduction example for your toolkit.
3. Manage any temporary storage outside steady-state timing, but record its size and startup cost.
4. Use the same input and output semantics in both versions.
5. Verify the result on edge sizes, zeros, and nontrivial values. Do not assume different reduction orders are bit-identical for floating point.

For a binary mask sum, use an accumulator wide enough for the number of pixels. A byte input does not mean the sum should be a byte.

## Experiment

Measure custom/library versions across small, medium, and large inputs. Include the application's transfer boundary when relevant. Compare maintenance effort as well: lines of bespoke kernel code, synchronization complexity, temporary-storage management, and numerical behavior.

State which implementation you would keep and why. A library win is a successful engineering outcome. A custom win for one narrow shape is evidence only for that tested shape and environment.

## Portfolio deliverable

Create an experiment report with both implementations, a result table, and a decision. Preserve upstream attribution when adapting examples. Include the library/toolkit version in the manifest.

**Extension:** investigate a cuFFT feature for sensor analysis later; do not add an FFT merely to increase the number of technologies listed. Continue to [profiling](09-profiling.md).
