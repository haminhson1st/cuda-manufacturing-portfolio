# 03 — Reduce readings into window statistics

**Time:** 6–8 hours. **Business task:** summarize vibration windows as mean, RMS, and peak absolute amplitude.

| You edit | Provided | Reference solution |
| --- | --- | --- |
| [src/lab03/kernels.cu](../src/lab03/kernels.cu) | Contract [lab03.hpp](../src/lab03/lab03.hpp), harness [main.cu](../src/lab03/main.cu), tests in the [Makefile](../Makefile) | [solutions/lab03/kernels.cu](../solutions/lab03/kernels.cu): open it after a time-boxed attempt |

## Concepts

**A reduction is a recurrence you restructure.** `sum = sum + x[i]` carries a dependence from one iteration to the next. H&P shows the standard transformation: expand the scalar into partial results that can be computed in parallel, then *reduce* them (6e §4.5, printed p. 344 / PDF p. 376). On a GPU, each thread accumulates a partial sum over a strided slice of the window. The block then combines the partials in a tree of `log2(threads)` steps.

**Partial results live in on-chip shared memory.** H&P calls it *local memory*: a small, low-latency scratchpad on each multithreaded SIMD processor, shared by the threads of one thread block and freed when the block exits (printed pp. 326–327 / PDF pp. 358–359, Figure 4.18). Mind the vocabulary: H&P's "local memory" is CUDA's **shared memory**, and H&P's "private memory" is roughly CUDA's per-thread **local memory**. The launch sizes the array dynamically: the third `<<<>>>` argument gives the bytes, and the kernel declares `extern __shared__ float shared[];` (CUDA Programming Guide 13.4.2 §2.3.3.2.2, printed p. 52 / PDF p. 70).

**Barriers have rules.** `__syncthreads()` makes every thread of the block wait, and shared-memory writes before it become visible to the block after it (Guide §2.3.2.1, printed p. 48 / PDF p. 66). It is allowed in conditional code only if the condition evaluates the same way for the entire block; otherwise execution may hang or produce undefined results (Guide §5.4, printed p. 597 / PDF p. 615). In a tree reduction, `if (tid < stride)` guards the **work**, never the barrier. Threads with no samples still participate and contribute the identity value (0 for the sums and for max|x|).

A block barrier does not synchronize separate blocks. This lab gives each window its own block, so no cross-block coordination is needed.

**Tolerances come from error analysis.** Floating-point addition is not associative, so a tree sum differs from a sequential one. The harness does not demand equality. It applies a standard bound: a float sum whose longest chain has `k` additions is off by at most γ(k)·Σ|x|, where γ(k) ≈ k·2⁻²⁴ (Higham, *Accuracy and Stability of Numerical Algorithms*; this bound comes from general numerical analysis, not the course corpus). With strided accumulation and a tree, `k = ceil(m / threads) + log2(threads)`. The **peak** involves no rounding, so it must match exactly. Read `validate()` in [lab03.hpp](../src/lab03/lab03.hpp).

## The contract

Window `w` covers samples `[w·W, min(n, (w+1)·W))`. For its `m` valid samples: `mean = sum/m`, `rms = sqrt(sum(x²)/m)`, `peak = max|x|`, `count = m`. The final window may be partial. Divide by `m`, never by `W`. The launcher (already written) uses one block per window and reserves `3 × threads` floats of shared memory in fused mode.

## Implement

0. On Sol, confirm the harness: `make SOLUTION=1 test-lab03`.
1. Start with `fused_kernel`, computing only the sum and writing `mean` and `count`. Run one case: `./build/window_stats --pattern constant --samples 1000 --window 256 --reps 1 --warmups 0`.
2. Add the sum of squares (`fmaf(v, v, acc)`) and the peak (`fmaxf(acc, fabsf(v))`).
3. Make sure windows larger than the block work. The strided loop gives each thread several samples (try `--window 20000`).
4. Write `single_stat_kernel` for `--mode separate`: the same structure with one shared array, writing one field per launch.
5. `make test-lab03`, then `make sanitize-lab03`, which runs memcheck, racecheck, and synccheck on both modes. Racecheck targets shared-memory hazards. Passing it does not prove every possible race is absent ([Sanitizer manual](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)).

| Symptom | Likely cause |
| --- | --- |
| `No window was written` | Kernel body still empty |
| `count` or `mean` wrong only in the last window | Divided by `W`, or `end` not clamped to `n` |
| Results change between runs; racecheck hazard | A tree level read a partner before a barrier |
| Hang, or a synccheck error | `__syncthreads()` inside `if (tid < stride)` or after an early `return` |
| `peak` mismatch | Used `fmaxf(acc, v)` instead of `fmaxf(acc, fabsf(v))`, or a wrong identity |
| Mean barely outside tolerance for huge windows | Longer addition chains than the bound assumes (for example, one thread summing everything) |

## Experiment

```bash
make bench-lab03          # fused/separate x window 256/4096/65536 x threads 128/256/512, 16 Mi samples
```

1. Separate mode reads the signal three times. Does `kernel_ms` grow by about 3×? What does that say about whether the kernel is memory-bound?
2. With `window = 256` and `threads = 512`, half the threads never load data. How does `windows_per_s` respond?
3. Report `max_mean_error` and `max_rms_rel_error` next to the analytic tolerance, with the workload that produced them.

## Done when

- [ ] `make test-lab03` and `make sanitize-lab03` pass.
- [ ] Your notebook explains every barrier, the identity values, the partial-window rule, and how the tolerance was derived.
- [ ] One benchmark is reproducible, and it reports windows/s with the timing boundary named.

**Extension:** replace the last five tree levels with warp shuffles (`__shfl_down_sync`), or use `cooperative_groups::reduce` (Guide printed p. 276 / PDF p. 294). Then compare. Splitting one huge window across several blocks needs partial results and a second pass. Continue to [the sensor case study](04-sensor-project.md).
