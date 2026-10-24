# References and provenance

Links were checked while preparing the course on October 22, 2026. Follow documentation compatible with the toolkit installed on Sol. Use sample tags matching the toolkit instead of assuming the latest branch will build with an older version.

## Course grounding

- Hennessy and Patterson, *Computer Architecture: A Quantitative Approach*, 6th edition, section 4.4, printed p. 312 (PDF p. 344): conventional DAXPY loop and CUDA mapping. Retrieved through the user's CSE520 KNS. The copyrighted textbook is not included here.
- NVIDIA, *CUDA Programming Guide*, version 13.4.2, section 2.3.4.1, printed p. 58 (PDF p. 76): coalesced global-memory access. Printed pp. 68–69 (PDF pp. 86–87): bank conflicts in a shared-memory transpose example. Retrieved through CSE520 KNS. [Official guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html).
- The KNS identifies GPU memory hierarchy and locality as prerequisites for memory coalescing. The weekly order here is a pedagogical plan, not an assertion that every neighboring topic has an architectural dependency.

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
