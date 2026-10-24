# 05 — Map threads to image pixels

**Time:** 6–8 hours. **Business task:** prepare inspection images and identify pixel differences from a reference image.

| You edit | Provided | Reference solution |
| --- | --- | --- |
| [src/lab05/kernels.cu](../src/lab05/kernels.cu) | Contract [lab05.hpp](../src/lab05/lab05.hpp), harness [main.cu](../src/lab05/main.cu), [image helpers](../src/common/image.hpp), [pitched device images](../src/common/device_image.cuh) | [solutions/lab05/kernels.cu](../solutions/lab05/kernels.cu) |

## Concepts

**2D grids are a convenience; warps are still linear.** Launch with `dim3` blocks and grids (CUDA Programming Guide 13.4.2 §2.1.2.3, printed p. 23 / PDF p. 41). Threads of a block are linearized with x moving fastest: `threadIdx.y` has a stride of `blockDim.x`, and that decides which threads share a warp (§2.3.2, printed p. 48 / PDF p. 66). For coalescing, consecutive `threadIdx.x` values should touch consecutive bytes of a row (§2.3.4.1, printed p. 62 / PDF p. 80).

**Rows, strides, and pitch.** A row-major 2D array is a sequence of 1D rows: skip `y` rows, then index `x` (Patterson & Hennessy, *Computer Organization and Design*, printed p. 216 / PDF p. 239). Real image buffers often pad each row, so the distance between row starts, the **pitch**, is in **bytes** and may exceed `width × channels`. The harness allocates device images with `cudaMallocPitch` and copies with `cudaMemcpy2D`, which honor both pitches. These two APIs are not in the course corpus; see the [CUDA Runtime API memory functions](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html). Your kernels must address pixel `(x, y)` as `data + y * pitch + x * channels`.

**Guards cause divergence only at the edges.** The `if (x >= width || y >= height) return;` guard disables lanes of the warps that straddle the right and bottom borders. H&P notes that conditional-code efficiency depends on how often branches diverge (6e §4.4, printed p. 326 / PDF p. 358). Interior warps pay almost nothing. The Modal GPU Glossary's [warp divergence entry](https://github.com/modal-labs/gpu-glossary/blob/9597497a3d8102c9cd17fc35ff06af5d97928275/gpu-glossary/perf/warp-divergence.md#L45-L73) shows the predicated SASS (secondary reference, Modal Labs and contributors, CC BY 4.0).

**Exact integer rules.** Every stage is defined in [lab05.hpp](../src/lab05/lab05.hpp) with integer arithmetic, so CPU and GPU must agree byte for byte:

| Stage | Rule |
| --- | --- |
| RGB → gray | `(77R + 150G + 29B + 128) >> 8`, channel order R, G, B (BT.601 weights in 8-bit fixed point) |
| Normalize | `v <= lo → 0`, `v >= hi → 255`, else `((v − lo)·255 + (hi − lo)/2) / (hi − lo)` |
| Difference | `abs(test − reference)` in `int`; unsigned bytes would wrap |
| Mask | 255 where diff > threshold, else 0 |

The test image is the reference plus a brightened rectangle at a known location. `--check-defect 1` additionally requires the mask to equal that rectangle exactly.

## Implement

0. On Sol: `make SOLUTION=1 test-lab05`.
1. Fix `make_grid`. The starter launches a single block, so only the top-left corner is processed. Round up in both dimensions.
2. Write the three kernels in order: `rgb_to_gray_kernel`, `normalize_kernel`, `absdiff_mask_kernel`. Host functions cannot be called from device code, so write the device arithmetic yourself.
3. `make test-lab05` covers 1×1, one-row and one-column images, 641×479, padded host strides, several patterns, and three block shapes. Then run `make sanitize-lab05`.
4. Save viewable outputs and look at them: `./build/image_ops --width 641 --height 479 --save runs/lab05`. That writes PGM/PPM files most image viewers open.

The harness compares every stage and names the first wrong one. Device outputs start filled with `0xA5` (165), padding included.

| Symptom | Likely cause |
| --- | --- |
| `got 165` | The pixel was never written: grid too small (the starter's one block) or a guard too strict |
| `guard_violations > 0` | Wrote past `width`, usually by indexing with `pitch` where `width` belongs, or by a missing x guard |
| Only `rgb_to_gray` fails | Forgot the factor 3 per pixel, or swapped R and B |
| Wrong values near 0 or 255 | Unsigned subtraction wrapped, or a different rounding rule |
| Correct for 64×64, wrong for 641×479 | Indexed rows with `width` instead of `pitch` (the device pitch is usually rounded up to an alignment boundary: see the `device_pitch` column) |

## Experiment

```bash
make bench-lab05          # 1920x1080 and 3840x2160 with 8x8, 16x16, 32x8, 8x32, 32x32 blocks
```

1. A warp of a 32×8 block reads 32 consecutive bytes of one row; a warp of an 8×32 block reads 8 bytes from each of four rows. Predict which is faster, then check.
2. Which costs more for a single image: `kernel_ms` or `transfer_compute_ms`? What would change if images arrived continuously?
3. Byte-sized elements make small transactions. Note this as a motivation for the vectorized-load extension.

Keep decode time outside kernel measurements and visible in full-pipeline measurements once you load real images.

## Done when

- [ ] `make test-lab05` and `make sanitize-lab05` pass.
- [ ] Your notebook shows one input/output pair, the indexing formula with pitch, edge-case results, and the block-shape timing table.
- [ ] You can explain why reference-image subtraction only works when viewpoint, alignment, and illumination make the comparison meaningful.

**Extension:** process four pixels per thread with `uchar4` loads (watch alignment and the row tail), or fuse the stages into one kernel and compare. Continue to [filtering](06-tiled-filter.md).
