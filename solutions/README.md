# Reference solutions

Complete versions of the files you edit in Labs 02–12:

| Lab | Your file | Solution |
| --- | --- | --- |
| 02 | [src/lab02/kernels.cu](../src/lab02/kernels.cu) | [lab02/kernels.cu](lab02/kernels.cu) |
| 03 | [src/lab03/kernels.cu](../src/lab03/kernels.cu) | [lab03/kernels.cu](lab03/kernels.cu) |
| 04 | [src/lab04/gpu_pipeline.cu](../src/lab04/gpu_pipeline.cu) | [lab04/gpu_pipeline.cu](lab04/gpu_pipeline.cu) |
| 05 | [src/lab05/kernels.cu](../src/lab05/kernels.cu) | [lab05/kernels.cu](lab05/kernels.cu) |
| 06 | [src/lab06/kernels.cu](../src/lab06/kernels.cu) | [lab06/kernels.cu](lab06/kernels.cu) |
| 07 | [src/lab07/gpu_inspect.cu](../src/lab07/gpu_inspect.cu) | [lab07/gpu_inspect.cu](lab07/gpu_inspect.cu) |
| 08 | [src/lab08/reductions.cu](../src/lab08/reductions.cu) | [lab08/reductions.cu](lab08/reductions.cu) |
| 09 | [src/lab09/inspector.cu](../src/lab09/inspector.cu) | [lab09/inspector.cu](lab09/inspector.cu) |
| 10 | [src/lab10/pipeline.cu](../src/lab10/pipeline.cu) | [lab10/pipeline.cu](lab10/pipeline.cu) |
| 11 | [src/lab11/app.cu](../src/lab11/app.cu) | [lab11/app.cu](lab11/app.cu) |
| 12 | [src/lab12/cpu_parallel.cu](../src/lab12/cpu_parallel.cu) | [lab12/cpu_parallel.cu](lab12/cpu_parallel.cu) |

Labs 04, 07, and 09–12 link your earlier code: Labs 02–03 for 04 and 10, Labs 05–07 for 07 and 09, Labs 02–07 and 09 for 11, and Labs 05–07 and 09 for 12. With `SOLUTION=1` they link the reference versions of those too.

`make SOLUTION=1 <target>` compiles these instead of your `src/` versions, into a separate `build-solution/` directory. Use it for two things:

1. **Check the harness and toolchain** on Sol before you start a lab (`make SOLUTION=1 test-lab02`). A failure there is an environment or course problem, not a problem with your kernel.
2. **Compare after a time-boxed attempt.** Read the solution only after your version passes, or after you are stuck and have written down what you tried.

These solutions were prepared with AI assistance and are not your portfolio work. If the public repository should show only your own implementations, add `solutions/` to `.gitignore` before publishing, or delete the folder from the published branch.
