# 06 — Reuse neighboring pixels with shared memory

**Time:** 6–8 hours. **Business task:** smooth image noise before computing an inspection score.

| You edit | Provided | Reference solution |
| --- | --- | --- |
| [src/lab06/kernels.cu](../src/lab06/kernels.cu) | Contract [lab06.hpp](../src/lab06/lab06.hpp), harness [main.cu](../src/lab06/main.cu), tests in the [Makefile](../Makefile) | [solutions/lab06/kernels.cu](../solutions/lab06/kernels.cu) |

## Concepts

**Each input pixel is read many times.** A (2R+1)×(2R+1) box filter reads every input pixel up to (2R+1)² times. Cache blocking addresses the same problem on CPUs: working on a B×B block cuts memory words accessed by roughly a factor of B and exploits temporal and spatial locality (H&P 6e §2.3, printed p. 109 / PDF p. 141). On a GPU you can do the blocking explicitly: copy a tile into the on-chip scratchpad that H&P calls local memory and CUDA calls **shared memory** (H&P printed pp. 326–327 / PDF pp. 358–359), then read neighbors from there.

**Tiles need halos.** To produce a block's outputs, the tile must also hold R pixels of border, the **halo**, on every side. H&P's Pixel Visual Core example makes the cost concrete: for a 5×5 stencil, 16 of the 25 inputs needed at a core come from halo elements (6e Chapter 7, printed p. 586 / PDF p. 618). The harness reports the cost as `global_loads_per_pixel`: (2R+1)² for the direct kernel, and (Bx+2R)(By+2R)/(Bx·By) for the tiled one. These are *issued* loads, not DRAM traffic. The direct kernel's repeated reads are often served by the L1 and L2 caches, which is why tiling is not a guaranteed speedup.

**The barrier is not optional, and neither is its placement.** All threads load the tile cooperatively, then `__syncthreads()`, then compute. A thread whose output pixel lies outside the image must still help load and must still reach the barrier. `__syncthreads()` is legal in conditional code only when the condition is uniform across the block (CUDA Programming Guide 13.4.2 §5.4, printed p. 597 / PDF p. 615). So: no early `return` before the barrier, and a guarded store after it.

**Shared-memory banks.** Shared memory has 32 banks, and successive 32-bit words map to successive banks. Threads of a warp accessing different words in the same bank are serialized (Guide §2.3.4.2, printed p. 62 / PDF p. 80; transpose example on printed pp. 68–69 / PDF pp. 86–87). This tile stores bytes, so four neighboring threads read the same word, which is a broadcast rather than a conflict (general CUDA knowledge, consistent with the bank model above).

## The contract

`out(x, y) = (Σ in(clamp(x+dx), clamp(y+dy)) + area/2) / area` over dx, dy ∈ [−R, R], with `area = (2R+1)²` and clamp-to-edge borders. The sum is an integer, so the CPU and GPU agree exactly. Input and output are separate buffers (out of place), so no thread overwrites a neighbor that another thread still needs. The launcher reserves `tile_bytes(R, block)` of dynamic shared memory, and the harness keeps the tile under the default 48 KiB.

## Implement

0. On Sol: `make SOLUTION=1 test-lab06`.
1. **Direct kernel first.** Read every neighbor from global memory with clamped coordinates, then `make test-lab06` and check that the direct cases pass. Keep this version: it is your comparison baseline.
2. **Tiled kernel.** Follow the TODO: a cooperative load loop over `tile_w × tile_h` entries (more entries than threads), compute the tile origin in `int`, call `__syncthreads()`, then a guarded computation from the tile.
3. `make test-lab06` (radii 1, 2, 3, 7; images from 1×1 up to 641×479; block shapes 32×8, 8×32, 32×32; padded strides).
4. `make sanitize-lab06` runs memcheck, racecheck, and synccheck on both variants.

| Symptom | Likely cause |
| --- | --- |
| Hang, or a synccheck error | Early `return` before `__syncthreads()` |
| racecheck reports a hazard | Read the tile before the barrier, or two threads wrote the same tile entry |
| Wrong only at image borders | Clamp missing, or the origin computed in unsigned arithmetic (`blockIdx.x * blockDim.x - R` wraps) |
| Wrong only along interior block edges | Halo not loaded: the load loop stops at the block size instead of the tile size |
| `too many resources requested` / `invalid argument` | Shared memory request too large for the block and radius |

Useful single cases: a constant field checks normalization, `--pattern impulse` shows the kernel's footprint, and `--pattern corner` exercises the clamp. Images smaller than the filter (`--width 2 --height 3 --radius 7`) check the borders.

## Experiment

```bash
make bench-lab06          # 1920x1080 and 3840x2160, radius 1/2/3/5/7, direct and tiled, 16x16 blocks
```

Write a hypothesis before running: how should the direct/tiled ratio change as R grows? Then compare it with `kernel_ms`, `shared_bytes`, and `global_loads_per_pixel`. Treat shared memory as a mechanism to evaluate, not a guarantee of speedup. Lab 09 will let you measure actual memory traffic with Nsight Compute.

## Deliverable

A before/after table with equivalent outputs, the halo overhead per configuration, and a short tradeoff explanation that includes a case where tiling did not help. Label algorithmic changes, such as a separable two-pass filter, separately from memory-layout changes.

**Extension:** a separable filter (row pass, then column pass) changes the work from O(R²) to O(R) per pixel. Verify that its boundary rule matches before comparing. Asynchronous copies into shared memory (Guide §4.12, printed p. 343 / PDF p. 361) are an advanced follow-up. Continue to [the image case study](07-image-project.md).
