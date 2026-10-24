# 01 — From a C++ loop to GPU sensor calibration

**Time:** 6–8 hours. **Business task:** convert raw sensor samples to engineering units with `y = gain*x + offset`. **Prerequisite:** [Sol setup](../docs/sol-setup.md).

## Read and predict

Follow [NVIDIA's introductory CUDA tutorial](https://developer.nvidia.com/blog/even-easier-introduction-cuda/). Your CSE520 textbook also shows C-to-CUDA DAXPY in section 4.4, printed p. 312/PDF p. 344; see [provenance](../docs/references.md).

Before running our reference, inspect [the kernel](../src/lab01/calibrate_cuda.cu) and answer: which element does thread 17 in block 2 process with 256 threads per block? What prevents the final block from accessing past the array? Which operation establishes that the host may read the result?

## Build and run

Use the commands in the [repository README](../README.md). CPU and GPU programs accept an optional positive sample count and optional raw-timing output filename. Run:

```bash
make lab01
./build/calibrate_cpu 257
./build/calibrate_cuda 257
./build/calibrate_cuda 1000003
make test-lab01
make sanitize-lab01
```

The programs print CSV, including numerical error and `correct`. No example timing values are supplied because timings must come from your hardware.

Preserve individual repetitions as well as the summary (choose fresh filenames to keep earlier experiments):

```bash
mkdir -p runs
./build/calibrate_cpu 1000003 runs/cpu-raw.csv > runs/cpu-summary.csv
./build/calibrate_cuda 1000003 runs/cuda-raw.csv > runs/cuda-summary.csv
```

## Understand the reference

1. `common.hpp` creates deterministic inputs and computes a double-precision reference for validation.
2. The CPU implementation performs calibration in a host loop.
3. The CUDA implementation allocates explicit device arrays and copies data.
4. Each thread computes one output after checking its index.
5. CUDA events time the kernel; a separate host clock measures copy–kernel–copy completion.

`cpu_ms` and `kernel_ms` are steady-state compute medians. `transfer_compute_ms` includes both transfers and kernel completion, but excludes buffer allocation, context initialization, file I/O, and input generation. All paths use 20 measured repetitions after warm-up. The generated input and calibration constants are deliberately easy to verify; expand the data before making broad numerical claims.

## Your modification

Change the calibration constants, then add an upper/lower output clamp to both implementations. Extend the independent reference calculation. Test negative inputs and values exactly at clamp boundaries. Keep the original calibration version in Git history.

## Done when

- [ ] Sizes 1, 255, 256, 257, and 1,000,003 pass; zero/invalid sizes fail clearly.
- [ ] The GPU run passes memcheck.
- [ ] You can explain every allocation, transfer, launch, and synchronization.
- [ ] Your notebook separates the three timing scopes and includes an observed small-input result.
- [ ] Your commit describes the modification you made, not simply that a tutorial was copied.

Publish the scaffold using [the GitHub tutorial](../docs/github-publishing.md), then continue to [Lab 02](02-sensor-calibration.md).
