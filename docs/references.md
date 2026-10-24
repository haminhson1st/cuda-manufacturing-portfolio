# References and provenance

Links were checked while preparing the course on October 22, 2026. Follow documentation compatible with the toolkit installed on Sol. Use sample tags matching the toolkit instead of assuming the latest branch will build with an older version.

## Course grounding

- Hennessy and Patterson, *Computer Architecture: A Quantitative Approach*, 6th edition, section 4.4, printed p. 312 (PDF p. 344): conventional DAXPY loop and CUDA mapping. Retrieved through the user's CSE520 KNS. The copyrighted textbook is not included here.
- NVIDIA, *CUDA Programming Guide*, version 13.4.2, section 2.3.4.1, printed p. 58 (PDF p. 76): coalesced global-memory access. Printed pp. 68–69 (PDF pp. 86–87): bank conflicts in a shared-memory transpose example. Retrieved through CSE520 KNS. [Official guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html).
- The KNS identifies GPU memory hierarchy and locality as prerequisites for memory coalescing. The weekly order here is a pedagogical plan, not an assertion that every neighboring topic has an architectural dependency.

Citations used in Labs 02–12, retrieved through the CSE520 KNS MCP tools on October 23, 2026. "Printed" is the book's page label; "PDF" is the one-based page in the indexed file.

| Source | Location | Used for |
| --- | --- | --- |
| Hennessy & Patterson, *Computer Architecture: A Quantitative Approach*, 6th ed. | §1.8, printed p. 40 (PDF p. 72) | Real applications as benchmarks; kernel, toy, and synthetic pitfalls (Lab 12) |
| same | §1.8, printed p. 45 (PDF p. 77) | Reproducibility as the reporting principle; response-time-qualified throughput (Labs 11, 12) |
| same | §1.9, printed p. 50 (PDF p. 82) | Amdahl's Law (Lab 04) |
| same | §2.3, printed p. 109 (PDF p. 141) | Cache blocking reduces memory words accessed (Lab 06) |
| same | §4.3, printed p. 349 (PDF p. 381), Figure 4.28 | Roofline model (Lab 09) |
| same | §4.4, printed p. 323 (PDF p. 355) | Address-coalescing hardware (Lab 02) |
| same | §4.4, printed pp. 325–326 (PDF pp. 357–358) | Branch masks, divergence, strip-mining guards (Labs 02, 05) |
| same | §4.4, printed pp. 326–327 (PDF pp. 358–359), Figure 4.18 | GPU memory structures: private, local (CUDA shared), GPU memory (Labs 03, 06) |
| same | §4.5, printed p. 344 (PDF p. 376) | Reductions: scalar expansion and the reduce step (Lab 03) |
| same | Chapter 4, printed pp. 352–353 (PDF pp. 384–385), Figure 4.30 | "Debunking the 100× GPU vs. CPU myth": optimized CPU baselines (Lab 12) |
| same | §5.5, printed p. 412 (PDF p. 444) | Atomic read-modify-write primitives; contention as a bottleneck (Labs 07, 08) |
| same | Chapter 7, printed p. 586 (PDF p. 618) | Stencil halos in the Pixel Visual Core (Lab 06) |
| same | Appendix C, printed p. C-8 (PDF p. 782) | Pipelining raises throughput, not individual latency (Lab 10) |
| Patterson & Hennessy, *Computer Organization and Design* | PDF p. 53 | Throughput (bandwidth): tasks completed per unit time (Lab 10) |
| same | §1.10, printed p. 49 (PDF p. 72) | Pitfall: improving one part does not speed up the whole proportionally (Lab 12) |
| same | printed p. 216 (PDF p. 239) | Row-major 2D array addressing (Lab 05) |
| same | §4.5, printed p. 275 (PDF p. 298) | Pipeline speedup and stage imbalance (Lab 10) |
| same | PDF p. 568 | Roofline model (Lab 09) |
| NVIDIA, *CUDA Programming Guide* 13.4.2 | §1.1.3, printed p. 4 (PDF p. 22) | Prefer an existing library when one fits (Lab 08) |
| same | §2.1.2.3, printed pp. 23–24 (PDF pp. 41–42) | `dim3` launches, thread intrinsics, bounds checking (Labs 02, 05) |
| same | §2.2.2.2, printed p. 42 (PDF p. 60) | At most 1024 threads per block (Lab 02) |
| same | §2.3.2, printed p. 48 (PDF p. 66) | Thread linearization (x fastest) and `__syncthreads` (Labs 03, 05) |
| same | §2.3.3.2.2, printed p. 52 (PDF p. 70) | Dynamic shared memory (`extern __shared__`, third launch argument) (Labs 03, 06) |
| same | §2.3.4.1–2.3.4.2, printed pp. 59–62 (PDF pp. 77–80) | Coalescing, 2D access patterns, shared-memory banks (Labs 02, 05, 06) |
| same | §2.5.3, printed p. 108 (PDF p. 126) | Events in streams and event timing (Lab 10) |
| same | §2.5, printed p. 110 (PDF p. 128) | Overlapping CPU work with GPU work by querying events (Lab 10) |
| same | printed p. 124 (PDF p. 142) | Page-locked (pinned) host memory, required for asynchronous copies (Labs 04, 10) |
| same | §2.5.6–2.5.8, printed p. 115 (PDF p. 133) | Non-blocking streams, implicit synchronization, concurrency guidelines (Lab 10) |
| same | §2.7.4.5, printed p. 135 (PDF p. 153) | `-lineinfo` for source correlation in Nsight tools (Lab 09) |
| same | §3.1.3.3, printed p. 141 (PDF p. 159) | Allocations between stream commands prevent concurrency (Labs 09, 10) |
| same | §4.3.2, printed p. 261 (PDF p. 279) | Stream-ordered allocation; the cost of `cudaMalloc`/`cudaFree` (Labs 07, 08) |
| same | printed p. 276 (PDF p. 294) | `cooperative_groups::reduce` (Lab 03 extension) |
| same | §4.8, printed pp. 309–310 (PDF pp. 327–328); printed p. 513 (PDF p. 531) | Lazy module loading and `CUDA_MODULE_LOADING` (Lab 08) |
| same | §4.12, printed p. 343 (PDF p. 361) | Asynchronous copies for stencil halos (Lab 06 extension) |
| same | §5.4, printed p. 597 (PDF p. 615) | `__syncthreads` in conditional code must be block-uniform (Labs 03, 06, 07) |
| same | §5.4, printed p. 598 (PDF p. 616) | `__syncthreads_count` (Lab 07 extension) |
| same | §5.4.5.1, printed pp. 605–606 (PDF pp. 623–624) | `atomicAdd`, including `unsigned long long` (Labs 07, 08) |
| same | §5.5.1.5, printed pp. 652–653 (PDF pp. 670–671) | Fused multiply-add and CPU/GPU comparison (Lab 02) |
| Modal GPU Glossary, commit `9597497a3d81` (secondary; Modal Labs and contributors, CC BY 4.0) | [warp-divergence.md L45–73](https://github.com/modal-labs/gpu-glossary/blob/9597497a3d8102c9cd17fc35ff06af5d97928275/gpu-glossary/perf/warp-divergence.md#L45-L73) | Predicated SASS for a divergent branch (Lab 05) |

Not found in the course corpus, so cited from outside it: Thrust and CUB APIs ([CCCL documentation](https://nvidia.github.io/cccl/)), `cudaMallocPitch`/`cudaMemcpy2D` ([CUDA Runtime API, memory management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)) and the floating-point summation error bound used in Lab 03 (N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002, §4.2).

## Tutorials and reference implementations

- [NVIDIA: An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
- [CUDA Samples and build instructions](https://github.com/NVIDIA/cuda-samples)
- [Vector addition](https://github.com/NVIDIA/cuda-samples/tree/master/cpp/0_Introduction/vectorAdd)
- [Reduction](https://github.com/NVIDIA/cuda-samples/tree/master/cpp/2_Concepts_and_Techniques/reduction)
- [Separable convolution](https://github.com/NVIDIA/cuda-samples/tree/master/cpp/2_Concepts_and_Techniques/convolutionSeparable)
- [Streams](https://github.com/NVIDIA/cuda-samples/tree/master/cpp/0_Introduction/simpleStreams)
- [Writing SIMT kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
- [Asynchronous execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- [CUDA Core Compute Libraries: Thrust and CUB](https://github.com/NVIDIA/cccl)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/)
- [Performance-counter permissions](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-nsightcompute)
- [GPU compute capabilities](https://developer.nvidia.com/cuda/gpus)

No upstream sample implementation is bundled. If you adapt one during a lab, record its exact commit and retain its required notices.

## Data

- [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter): normal/faulty bearing experiments, recording conditions, and source data.
- [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad): industrial anomaly benchmark; CC BY-NC-SA 4.0, noncommercial. Use the dataset's official partitions and record any additional development split.

These sources are independent datasets, not synchronized observations from the same factory.

## Sol and GitHub

- [ASU SSH access](https://docs.rc.asu.edu/terminal-clients/)
- [ASU interactive sessions](https://docs.rc.asu.edu/interactive-sessions/)
- [ASU resource requests](https://docs.rc.asu.edu/requesting-resources/)
- [GitHub repository quickstart](https://docs.github.com/en/repositories/creating-and-managing-repositories/quickstart-for-repositories)

Use NVIDIA's current CUDA terminology for code. Historical textbook terminology and device-specific limits should not be generalized into guarantees for every GPU.
