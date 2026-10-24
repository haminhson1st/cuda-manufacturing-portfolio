# 02 — Calibrate multiple sensor channels

**Time:** 6–8 hours. **Business task:** calibrate batches from many sensors, each with its own gain and offset, and flag out-of-range readings.

| You edit | Provided | Reference solution |
| --- | --- | --- |
| [src/lab02/kernels.cu](../src/lab02/kernels.cu) | Contract [lab02.hpp](../src/lab02/lab02.hpp), harness [main.cu](../src/lab02/main.cu), tests in the [Makefile](../Makefile) | [solutions/lab02/kernels.cu](../solutions/lab02/kernels.cu): open it after a time-boxed attempt |

## Concepts

**Map neighboring threads to neighboring addresses.** GPU address-coalescing hardware merges a warp's accesses into a few memory blocks when adjacent CUDA threads touch nearby addresses at the same time (Hennessy & Patterson 6e §4.4, printed p. 323 / PDF p. 355). In the CUDA Programming Guide's terms, consecutive threads reading consecutive 4-byte elements need about four 32-byte transactions per warp, while scattered addresses can need 32 (§2.3.4.1, printed pp. 59–60 / PDF pp. 77–78). Coalescing is a performance mechanism, not a correctness guarantee. In this lab, thread `i` touches element `i` in **both** layouts, so the large arrays stay coalesced either way. What changes is how a thread finds its channel's gain and offset.

**Guards are masks.** `if (i < n)` does not cost a separate code path per thread. A GPU disables the SIMD lanes whose condition is false, and H&P notes that the same mask mechanism handles loops whose element count does not match the hardware width (printed p. 326 / PDF p. 358). The Guide's bounds-checking example shows the pattern in CUDA C++ (§2.1.2.3.1, printed p. 24 / PDF p. 42).

**Grid-stride loops decouple the grid from the data.** If each thread handles `i, i + stride, i + 2·stride, …` with `stride = blockDim.x * gridDim.x`, any grid size covers any `n`. The tests force this with `--blocks 3`.

**Bit-identical arithmetic.** A fused multiply-add rounds once instead of twice. The Guide warns that you must know whether the host compiler contracts `a*b + c` into an FMA before comparing CPU and GPU results (§5.5.1.5, printed pp. 652–653 / PDF pp. 670–671). This lab calls `std::fma` on the CPU and `fmaf` on the GPU, so the results must be **identical**. Validation uses exact equality, not a tolerance.

## The contract

| Item | Rule (see [lab02.hpp](../src/lab02/lab02.hpp)) |
| --- | --- |
| Channel-major | `index = channel * samples + sample` |
| Sample-major | `index = sample * channels + channel` (interleaved frames, as an acquisition system delivers them) |
| Value | `value = fmaf(gain[c], raw[i], offset[c])` |
| Alarm | `alarm_low` if `value < low`, `alarm_high` if `value > high`, else `alarm_none`. Uses the value **before** clamping, so clipping is detectable |
| Output | `fminf(fmaxf(value, low), high)` |

The inputs are built so that samples 0, 1, and 2 of every channel calibrate to exactly `high`, exactly `low`, and just above `high`. An off-by-one comparison fails immediately. The harness pre-fills device outputs with `0xFF` bytes, so any element your kernel skips reads back as `NaN`/255.

## Implement

0. **Check the toolchain first.** On Sol, run `make SOLUTION=1 test-lab02`. It builds and tests the reference solution in `build-solution/`. If that fails, the problem is the environment or the course code, not your kernel. Record it in your notebook.
1. Run the starter: `make test-lab02`. The first test fails with `No element was written`, because the starter launches nothing. That is expected.
2. In `launch_calibrate`, compute `blocks` when `config.blocks == 0`: round `n / threads` up, capped at 2³¹−1.
3. Write `calibrate_kernel` as a grid-stride loop. Compute the index in `size_t`. `blockIdx.x * blockDim.x` is 32-bit unsigned arithmetic.
4. Derive the channel from the layout, then apply the scalar rule.
5. Iterate with `make test-lab02` (stops at the first failure) or `make -k test-lab02`. Then run `make sanitize-lab02`.

Run one case by hand when debugging:

```bash
./build/calibrate_channels --channels 3 --samples 257 --layout sample --reps 1 --warmups 0
```

| Symptom | Likely cause |
| --- | --- |
| `got out=nan alarm=255` | That element was never written: the grid is too small and there is no grid-stride loop, or the stride is wrong |
| Fails only for `--layout sample` | The channel was derived as `i / samples` for both layouts |
| Fails only at samples 0–2 | `<=` instead of `<`, or the alarm was computed from the clamped value |
| `invalid configuration argument` | `blocks` is 0 or too large, or more than 1024 threads per block (Guide §2.2.2.2, printed p. 42 / PDF p. 60) |
| memcheck reports an invalid global read | The loop bound or index type is wrong |

## Experiment

```bash
make bench-lab02          # writes runs/<time>/lab02.csv: both layouts x 128/256/512 threads
```

Use 64 channels × 1,000,000 samples and answer from your own measurements:

1. Compare `kernel_gbps` (logical bytes per element: 4 read + 4 written + 1 alarm) with your GPU's documented memory bandwidth. Is this kernel limited by memory or by arithmetic?
2. Does block size matter here? Explain the result rather than ranking the numbers.
3. The channel lookup is a 64-bit division (`i / samples`) or modulo (`i % channels`). Integer division is comparatively expensive on GPUs (general knowledge, not from the course sources). Does it show up in your timings?
4. Compare `convert_ms` (the host transpose a channel-major batch needs) with `kernel_ms` and `transfer_compute_ms`. Choose a layout from the **application** cost, not the kernel time alone.

Treat any difference as an observation to explain, not a guarantee from the layout's name.

## Done when

- [ ] `make test-lab02` and `make sanitize-lab02` pass with your kernel.
- [ ] You can explain the index → (channel, sample) map for both layouts and why both stay coalesced.
- [ ] The notebook includes a mapping sketch, a correctness log (`make test-lab02 | tee runs/lab02-tests.txt`), the benchmark CSV, and your layout decision, including conversion cost.

**Extension:** remove the division with a 2D grid (`blockIdx.y` selects the channel) and measure it. Alternatively, split calibration and thresholding into two kernels with an intermediate array and compare against the fused kernel. Continue to [Lab 03](03-window-statistics.md).
