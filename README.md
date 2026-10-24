# CUDA for Manufacturing

An applied CUDA C++ learning portfolio: sensor analytics, visual inspection, and measured GPU acceleration on ASU Sol.

**Status:** course scaffold, Lab 01 reference code, and starter code with reference solutions for Labs 02–12. The manufacturing projects are planned. Nothing in Labs 02–12 has been compiled or run on a GPU yet, and no GPU performance or inspection-accuracy results have been measured. See [validation status](docs/validation.md).

The engineering question is: **when does a GPU improve a manufacturing workflow enough to justify using it?** Each project will compare correctness, application performance, and the quality of the resulting decisions.

## Start here

1. Read the [12-week learning path](tutorials/README.md).
2. Follow [Sol setup](docs/sol-setup.md), then [Tutorial 01](tutorials/01-first-gpu.md).
3. Publish the scaffold using the [GitHub walkthrough](docs/github-publishing.md).
4. Record your changes and evidence using the [lab notebook](templates/lab-notebook.md) and [benchmark protocol](docs/benchmarking.md).

**Pace:** 6–8 hours per week. **Background assumed:** C/C++, arrays and pointers, basic computer architecture. The first exercises introduce Linux/Slurm commands as needed.

## Projects

| Project | Manufacturing question | Planned evidence | Status |
| --- | --- | --- | --- |
| [Sensor screening](projects/01-sensor-screening/README.md) | Can batches of vibration windows be screened efficiently? | CPU/GPU agreement, windows/s, alarm errors by recording | Planned |
| [Visual inspection](projects/02-visual-inspection/README.md) | Can an image-processing pipeline flag visible defects? | Defect recall, false rejects, image throughput | Planned |
| [Inspection capstone](projects/03-capstone/README.md) | Which workload sizes benefit from GPU execution? | Reproducible CPU/GPU comparison and deployment recommendation | Planned |

## Included today

- Twelve tutorials with a business task, implementation sequence, correctness checks, experiment, and portfolio deliverable.
- An original, small C++/CUDA calibration reference in [src/lab01](src/lab01), including explicit transfers and timing boundaries.
- Labs 02–12 as **starter code**: a complete harness (inputs, CPU reference, validation, timing, CSV output) around files with TODOs for you to implement. Lab 09 adds Nsight Systems/Compute capture targets; Lab 10 provides a serial baseline for your stream pipeline; Lab 11 is the `inspect` capstone application with [frozen configurations](configs/README.md); Lab 12 measures the release against a multicore CPU baseline and generates the results report. Matching [reference solutions](solutions/README.md) build with `make SOLUTION=1`.
- Data tools for the case studies: synthetic recordings and images for rehearsal, converters for CWRU and MVTec AD files, and evaluators that keep threshold selection separate from held-out evaluation.
- A GNU [Makefile](Makefile) with per-lab build, test, sanitizer, and benchmark targets, plus [Slurm scripts](slurm).
- Dataset and experiment templates, project briefs, and a guide to presenting evidence on GitHub.
- A GitHub Actions workflow that checks documentation, tests the CPU reference, and compiles every CUDA target without a GPU. It does **not** validate CUDA execution.

The projects' case studies, measurements, and release are your work to produce. Nothing here is presented as completed software.

## First build

The build uses GNU make and needs no CMake. On a Linux compute session with a C++17 compiler:

```bash
make cpu
make test-cpu
./build/calibrate_cpu 1000003
```

On an allocated A100 with the CUDA module loaded:

```bash
make -j 4                      # CPU reference plus every CUDA lab
make test-lab01                # one lab's tests; `make test` runs them all
make SOLUTION=1 test-lab02     # check the harness against the reference solution
make sanitize-lab01            # compute-sanitizer memcheck
make help                      # every target and option
```

`CUDA_ARCH=80` (the default) targets A100/A30. Pass another value, for example `make CUDA_ARCH=86`, if the allocated GPU changes; the build directory rebuilds itself when flags change. Read the [setup guide](docs/sol-setup.md) before running these commands.

## Results

No measurements published yet. Add results only after running the documented workload. Keep kernel-only speedup separate from transfers-plus-compute and complete application speedup.

The eventual results table should identify the workload, CPU threads, GPU model, software versions, timing boundary, numerical error, and source commit. An unsuccessful optimization is a useful result when the experiment is reproducible.

## Learning and authorship

This initial scaffold, the first-lab reference, the Lab 02–12 harnesses, and the reference solutions were prepared with AI assistance. The portfolio owner's work will be demonstrated through implementation commits, tests, profiling interpretation, and experiment reports. Keep [the development log](docs/development-log.md) current and cite upstream examples when adapting them.

Educational manufacturing examples are not validated production inspection or predictive-maintenance systems. Public benchmark data and synthetic inputs are identified in each report.

## Repository map

```text
tutorials/   guided weekly lessons
src/         lab01 reference; labs 02-12 harnesses and the files you edit; common/ helpers
solutions/   reference solutions for the files you edit (built with make SOLUTION=1)
configs/     frozen detector and screening rules (name = value files)
scripts/     test runner, documentation checker, Lab 04 data tools
projects/    business briefs and acceptance criteria
docs/        setup, publishing, measurement, references
slurm/       single-GPU execution and validation scripts
templates/   notebook, report, experiment, dataset records
results/     reviewed measurements and reports added by the learner
data/        acquisition notes; raw datasets stay untracked
```

Read [references and attribution](docs/references.md) and [usage notes](NOTICE.md) before reusing third-party material.
