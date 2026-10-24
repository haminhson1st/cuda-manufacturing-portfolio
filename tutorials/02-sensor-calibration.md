# 02 — Calibrate multiple sensor channels

**Time:** 6–8 hours. **Business task:** calibrate batches from many sensors with different gain/offset values and flag out-of-range readings.

## Read and define the data

Inspect NVIDIA's [vectorAdd](https://github.com/NVIDIA/cuda-samples/tree/master/cpp/0_Introduction/vectorAdd) for explicit memory management. Read coalescing in [Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html). Coalescing concerns how threads' accesses are combined into transactions; the layout that helps depends on the mapping you choose.

Use `C` channels and `S` samples per channel. Write down two layouts:

- Channel-major: `index = channel*S + sample`.
- Sample-major: `index = sample*C + channel`.

The scalar result is `clamp(gain[channel]*raw + offset[channel], low, high)`. Define whether the alarm uses the original calibrated value or the clamped value. Use the former if you want to detect clipping.

## Implement

1. Create a separate Lab 02 implementation and register it as a new CMake target.
2. Generate inputs where every channel has a different known calibration.
3. Write a CPU version and a GPU version with explicit transfers.
4. Map neighboring threads to neighboring elements; derive the channel index from the selected layout.
5. Add a grid-stride loop so a fixed-size launch can cover larger input arrays.
6. Return both calibrated values and byte-sized alarm flags. Keep allocation outside repeated steady-state measurements.

Do not add streams or shared memory yet. First establish correctness and the layout experiment.

## Verify and experiment

Test one channel, 3 channels, 32 channels, sample counts not divisible by the block size, and values exactly on each threshold. Compare every result and flag. Run memcheck on both layouts.

Measure 128, 256, and 512 threads per block for a fixed workload. Then compare layouts using a clearly documented mapping. Treat a performance difference as an observation to explain, not a guarantee from the layout's name.

## Portfolio deliverable

Add a small mapping sketch/table, a correctness log, and a timing CSV. Explain which layout you selected and why. Report separately any CPU layout-conversion cost needed by the application. Acceptance: both layouts return the same logical sensor values; your selected version has a reproducible measurement.

**Extension:** fuse calibration and thresholding, then compare against two kernels with an intermediate array. Proceed to [Lab 03](03-window-statistics.md).
