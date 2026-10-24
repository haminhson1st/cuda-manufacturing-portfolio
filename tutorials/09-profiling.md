# 09 — Find the bottleneck before optimizing

**Time:** 6–8 hours. **Business task:** improve elapsed inspection time without changing the accepted output.

| You edit | Provided | Reference solution |
| --- | --- | --- |
| [src/lab09/inspector.cu](../src/lab09/inspector.cu), plus your before/after report | Harness [main.cu](../src/lab09/main.cu) (`profile_inspect`), contract [lab09.hpp](../src/lab09/lab09.hpp), [NVTX ranges](../src/common/nvtx.hpp), and `make profile-lab09` | [solutions/lab09/inspector.cu](../solutions/lab09/inspector.cu) |

Read the assess/parallelize/optimize/deploy discussion in [NVIDIA's Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) and review [the measurement protocol](../docs/benchmarking.md). `profile_inspect` runs **your** Lab 07 pipeline (`lab07::score_gpu`, with your Lab 05/06 kernels) as the baseline.

## Concepts

**Measure the application, then attribute the time.** Amdahl's Law (Lab 04) says to speed up the largest fraction first, and a profile tells you what that fraction is. Nsight Systems records a timeline of host threads, CUDA API calls, copies, and kernels; Nsight Compute measures hardware counters for individual kernels. The Makefile already compiles with `-lineinfo`, which lets both tools map device code back to source lines (CUDA Programming Guide 13.4.2 §2.7.4.5, printed p. 135 / PDF p. 153). NVTX ranges name host phases on the timeline: the harness marks `decode`, `cpu reference`, `baseline`, `reuse`, and each `image`, and your Inspector adds `upload`, `kernels`, and `download`.

**Read kernel counters through the roofline.** A kernel's attainable performance is bounded by the lower of peak compute and memory bandwidth × arithmetic intensity (H&P 6e §4.3, printed p. 349 / PDF p. 381, Figure 4.28; Patterson & Hennessy, PDF p. 568). The Lab 05–07 kernels do a handful of integer operations per byte, so expect them to be memory-bound. Nsight Compute's throughput section lets you test that hypothesis. Occupancy, or any other single counter, is not a performance guarantee.

**Allocation and synchronization show up as gaps.** `lab07::score_gpu` creates six pitched images, a counter, and two events for every image. On the timeline they appear as `cudaMallocPitch`, `cudaFree`, and `cudaEventCreate` calls between the kernels, and freeing device memory can wait for the GPU. The Guide notes that a device or page-locked allocation issued between two streams' commands prevents them from running concurrently (§3.1.3.3, printed p. 141 / PDF p. 159), and that stream-ordered allocation exists because `cudaMalloc`/`cudaFree` can be costly (§4.3.2, printed p. 261 / PDF p. 279).

## Capture a baseline

1. Freeze the input list, detector parameters, and source commit. Prefer your Lab 07 MVTec list; `make profile-lab09` generates a synthetic 384×384 set if you have none.
2. Time outside the profiler first: `./build/profile_inspect --list LIST --variant baseline --repeat 5`.
3. Profile inside a GPU allocation. `make profile-lab09` runs both captures below for both variants, writes `nsys stats` summaries, and runs Nsight Compute on the Lab 06 filters. By hand:

```bash
mkdir -p runs
nsys profile --trace=cuda,nvtx,osrt --output=runs/lab09-baseline \
  ./build/profile_inspect --list LIST --variant baseline --verify 0 --repeat 5
nsys stats runs/lab09-baseline.nsys-rep
ncu --set basic --kernel-name regex:box_ --launch-count 1 \
  ./build/box_filter --variant tiled --radius 3 --reps 1 --warmups 0 --cpu-reps 1
```

Use a fresh output name for each experiment; the Lab 01 executable works the same way (`./build/calibrate_cuda 1000003`). A profile is a diagnostic capture, not a representative benchmark: the tools add overhead. If the tools are missing, inspect the installed modules. If counters are restricted (`ERR_NVGPUCTRPERM`), consult ASU RC and record the limitation; do not attempt to change shared-system permissions.

## Ask questions from the timeline

How much time goes to reading and decoding files? Are transfers repeated unnecessarily? Are allocations inside the image loop, and does each image wait for a `cudaFree`? Are the CPU and GPU idle between stages? Within an `image` range, how long are the kernels compared with the gaps between them? Which kernel consumes the most time?

For the filters, compare the direct and tiled Nsight Compute reports with your Lab 06 hypothesis: memory throughput, achieved bandwidth, and where the time goes.

## Change exactly one thing

The change prepared in this lab is **buffer reuse**. `lab09::Inspector` keeps its device images, counter, and events alive across images, and reallocates only when the image shape changes. Write your prediction before editing, for example: "per-image `gpu_wall_ms` falls by about the allocation time visible on the timeline; `compute_ms` is unchanged."

0. On Sol: `make SOLUTION=1 test-lab09`.
1. Implement `ensure_buffers`: return early when the shape matches, otherwise replace the six `unique_ptr<DeviceImage>` members, remember the shape, and count the allocation.
2. Implement `score`: the Lab 07 stages using the member buffers, wrapped in NVTX ranges.
3. Run `make test-lab09`. The harness checks every count against the CPU reference and against the baseline, and checks that `allocations` equals the number of shape changes in the processing order. Then run `make sanitize-lab09`.
4. Run `make bench-lab09`, then profile again and compare the two timelines.

| Symptom | Likely cause |
| --- | --- |
| `Inspector::score is not implemented yet` | Starter still throws |
| `allocated N times; expected M` with N > M | `ensure_buffers` reallocates on every call: the shape is not compared or not remembered |
| Wrong counts after the image size changes | The buffers were not reallocated for the new shape |
| Wrong counts only on later repeats | A reused buffer or counter carries stale data; `lab07::launch_count_mask` zeroes the counter itself |
| No NVTX rows in Nsight Systems | `--trace` lacks `nvtx`, or the NVTX header was not found at build time |

Other single changes worth their own experiments: pinned host staging for the upload, fusing normalization into the difference kernel, or the Lab 06 block shape that measured best. Change one thing at a time and keep the accepted output identical; the harness checks it.

## Deliverable

A short before/after report: the bottleneck evidence (a timeline excerpt or `nsys stats` lines), the change, the observed improvement or regression with the same inputs (`gpu_wall_ms`, images/s), and the next decision. Publish a compact trace screenshot only if it adds explanation and reveals no private paths or data; raw traces stay untracked (`*.nsys-rep` and `*.ncu-rep` are ignored).

**Acceptance:** your claim concerns measured application behavior, with no guarantee inferred merely from occupancy or another isolated counter. Continue to [streams](10-streaming.md).
