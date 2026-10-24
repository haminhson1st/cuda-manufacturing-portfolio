# CUDA for Manufacturing

An applied CUDA C++ learning portfolio: sensor analytics, visual inspection, and measured GPU acceleration on ASU Sol.

**Status:** course scaffold and first-lab reference code. The manufacturing projects are planned. No GPU performance or inspection-accuracy results have been measured for this repository yet. See [validation status](docs/validation.md).

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
- Optional CUDA builds, CPU/GPU correctness cases, and [Slurm scripts](slurm).
- Dataset and experiment templates, project briefs, and a guide to presenting evidence on GitHub.
- A GitHub Actions workflow that checks documentation and CPU code. It does **not** validate CUDA execution.

Later tutorial implementations are exercises. They are not presented as completed software.

## First build

On a Linux compute session with a C++17 compiler and CMake 3.20 or newer:

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --parallel 4
ctest --test-dir build-cpu --output-on-failure
./build-cpu/calibrate_cpu 1000003
```

On an allocated A100 with the CUDA module loaded:

```bash
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-gpu --parallel 4
ctest --test-dir build-gpu --output-on-failure
./build-gpu/calibrate_cuda 1000003
compute-sanitizer --tool memcheck ./build-gpu/calibrate_cuda 1000003
```

The architecture value `80` targets A100/A30. Select the appropriate value if the allocated GPU changes. Read the [setup guide](docs/sol-setup.md) before running these commands.

## Results

No measurements published yet. Add results only after running the documented workload. Keep kernel-only speedup separate from transfers-plus-compute and complete application speedup.

The eventual results table should identify the workload, CPU threads, GPU model, software versions, timing boundary, numerical error, and source commit. An unsuccessful optimization is a useful result when the experiment is reproducible.

## Learning and authorship

This initial scaffold and first-lab reference were prepared with AI assistance. The portfolio owner's work will be demonstrated through implementation commits, tests, profiling interpretation, and experiment reports. Keep [the development log](docs/development-log.md) current and cite upstream examples when adapting them.

Educational manufacturing examples are not validated production inspection or predictive-maintenance systems. Public benchmark data and synthetic inputs are identified in each report.

## Repository map

```text
tutorials/   guided weekly lessons
src/         implementations; lab01 reference included
projects/    business briefs and acceptance criteria
docs/        setup, publishing, measurement, references
slurm/       single-GPU execution and validation scripts
templates/   notebook, report, experiment, dataset records
results/     reviewed measurements and reports added by the learner
data/        acquisition notes; raw datasets stay untracked
```

Read [references and attribution](docs/references.md) and [usage notes](NOTICE.md) before reusing third-party material.
