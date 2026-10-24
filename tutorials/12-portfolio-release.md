# 12 — Publish evidence a reviewer can reproduce

**Time:** 6–8 hours. **Business task:** recommend when the GPU implementation is useful and when a CPU is sufficient.

| You edit | Provided | Reference solution |
| --- | --- | --- |
| [src/lab12/cpu_parallel.cu](../src/lab12/cpu_parallel.cu), plus the case study, demonstration, and release | Harness [main.cu](../src/lab12/main.cu) (`release_bench`), contract [lab12.hpp](../src/lab12/lab12.hpp), [lab12_report.py](../scripts/lab12_report.py), and `make bench-lab12` | [solutions/lab12/cpu_parallel.cu](../solutions/lab12/cpu_parallel.cu) |

Use [the benchmark protocol](../docs/benchmarking.md), [the case-study template](../templates/case-study.md), and [the portfolio playbook](../docs/portfolio-playbook.md).

## Concepts

**Compare against a fair CPU.** H&P revisits a study titled "Debunking the 100× GPU vs. CPU myth": with both platforms optimized, the GPU was slower on sort (0.8×) and only 1.6–5.7× faster on several other kernels (6e Chapter 4, printed p. 352 / PDF p. 384, Figure 4.30). Repeated on successor chips, the GPU was 2–5 times faster than the CPU (printed p. 353 / PDF p. 385). Large GPU speedups usually compare against one CPU thread. `release_bench` therefore measures four variants on the same decoded images and the same frozen detector:

| Variant | Implementation |
| --- | --- |
| `cpu` | `lab07::score_cpu`, one thread |
| `cpu-parallel` | **your** multicore baseline: the headline comparison |
| `cuda-baseline` | Lab 07's straightforward GPU version (fresh buffers per image) |
| `cuda` | Lab 09's final version (buffers reused) |

**Report so others can reproduce.** Reproducibility is the guiding principle of reporting, and benchmark reports publish both the baseline and the optimized results (H&P 6e §1.8, printed p. 45 / PDF p. 77). The harness writes `raw.csv` (every repetition), `summary.csv` (median, interquartile range, min–max, images/s), and `environment.json` (GPU, driver and runtime versions, CPU model and threads, Slurm job, source commit, build, inputs, detector, and timing boundary). `lab12_report.py` turns them into `report.md`.

**Measure real work.** H&P warns that kernels, toy programs, and synthetic benchmarks have led to performance pitfalls; the best benchmarks are real applications (6e §1.8, printed p. 40 / PDF p. 72). Measure your frozen dataset. The synthetic default of `make bench-lab12` is for rehearsal only.

**Accelerating one part is not accelerating the whole.** Expecting an improvement to one aspect to raise overall performance proportionally is a classic pitfall (Patterson & Hennessy §1.10, printed p. 49 / PDF p. 72). The report keeps decoding separate from processing, and your case study must say what fraction of the application the GPU actually accelerates.

## Implement the multicore baseline

0. On Sol: `make SOLUTION=1 test-lab12`.
1. Implement `score_all_cpu_parallel` following its TODO: `std::thread` workers that claim images through an atomic index, write distinct elements of the result, catch exceptions into a `std::exception_ptr`, and are all joined before returning.
2. Run `make test-lab12`. It covers sizes 1, 5, and all; 3 threads; 64 threads clamped to the number of images; the edge images; and invalid variants and sizes. It also generates a report. Then run `make sanitize-lab12`.

| Symptom | Likely cause |
| --- | --- |
| `score_all_cpu_parallel is not implemented yet` | Starter still throws |
| `terminate called ...` | An exception escaped a worker, or a `std::thread` was destroyed without being joined |
| Counts are wrong only occasionally | A data race: a plain shared index, or two workers writing the same element |
| `cpu-parallel` is no faster than `cpu` | Fewer CPUs allocated than threads (check `SLURM_CPUS_PER_TASK` and `--cpus-per-task`), or a 1-image workload, which has nothing to parallelize across images |

Request enough CPU cores in the job for a fair baseline (the default `--cpu-threads` follows `SLURM_CPUS_PER_TASK`), and record the count. If you compile the CPU code with extra flags such as `-march=native`, record them as well.

## Freeze and measure

1. Freeze the source, input list and split, `configs/<dataset>.conf`, and build configuration (`CUDA_ARCH`, CUDA module).
2. Inside an allocation with enough CPUs:

   ```bash
   make bench-lab12 RELEASE_LIST=data/processed/mvtec-bottle/images.csv RELEASE_SPLIT=test \
        RELEASE_CONFIG=configs/mvtec-bottle.conf
   ```

   This writes `runs/<time>/release/` with `raw.csv`, `summary.csv`, `environment.json`, and `report.md`, measuring 1, 10, and all images with 10 repetitions after 2 warm-ups. The source commit is recorded automatically.
3. Re-run correctness and the held-out evaluation (Lab 07 `evaluate`) on the frozen commit. Every row of the report must say `correct = true`; the report marks any speedup against an incorrect result as invalid.
4. Keep the raw observations. Report medians with their variability, and add repetitions while the spread is large.

If GPUs lose for small inputs, or the detector misses defects, include those results. Do not search for a flattering workload and present it as the general case.

## Write the report

Begin with the observed result and its scope. Explain the manufacturing problem, your implementation contribution, the comparison (headline: `cuda` against `cpu-parallel`, at the stated timing boundary), and the remaining limitations. Paste or link the `report.md` tables rather than retyping numbers. Include a compact architecture sketch if it helps readers see where CPU and GPU work happen, and explain one rejected optimization (Lab 09 or Lab 10 provides candidates).

Estimate business impact only from explicit assumptions. For example, compare the measured images/s with the hypothetical arrival rate from Lab 11. Do not translate throughput into financial savings without a stated cost and deployment model.

## Prepare a short demonstration

Show a reproducible run, one correct output, one failure case, the measurement table, and the reason for your final implementation choice. Aim for three to five minutes. Describe Sol as the development and benchmark platform; deployment hardware would require separate validation.

## Publish

Update the README statuses and link the report. Copy the reviewed `report.md`, `summary.csv`, and `environment.json` into a dated `results/` directory; keep raw data and traces out of Git. Decide whether `solutions/` belongs in the published repository (see [its note](../solutions/README.md)). Then tag the evaluated commit and publish the release from your own terminal:

```bash
git tag -a v1.0 -m "Evaluated release: see results/<date>/report.md"
git push origin v1.0
```

Create the GitHub release from that tag and attach the report, not the raw dataset. Pin the repository if it represents work you can explain. Record data-access instructions rather than uploading restricted raw inputs, and keep AI assistance and upstream attribution visible.

**Completion checklist:** a fresh reader can build the application; test evidence exists; measurements identify hardware and timing scope; accuracy metrics use a held-out evaluation; and no performance or production claim exceeds the evidence.

After this course, choose one extension based on observed needs: cuFFT sensor features, improved anomaly detection, advanced kernel optimization, or deployment on a specific edge GPU. Each should start with a concrete workload and acceptance criteria.
