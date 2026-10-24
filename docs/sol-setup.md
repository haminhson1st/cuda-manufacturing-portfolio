# Tutorial 00: prepare Sol for CUDA

**Outcome:** compile and validate a small program in a scheduled GPU allocation. Plan for 60–90 minutes plus queue time. Commands below run in Bash on Sol unless marked PowerShell.

## 1. Connect

Activate the ASU VPN, then open PowerShell on Windows:

```powershell
ssh YOUR_ASURITE@sol.asu.edu
```

Use your own ASURITE name and complete Duo authentication. The browser portal at <https://sol.asu.edu> is another entry point. ASU's [SSH guide](https://docs.rc.asu.edu/terminal-clients/) documents both access prerequisites and the terminal workflow.

## 2. Allocate one GPU

From the login shell:

```bash
interactive -p htc -q public -G a100:1 -c 4 --mem=24G -t 01:00:00
```

Wait for the allocation. Then run `hostname` and `nvidia-smi`. Run compilation, tests, and profiling in the allocated session. Read/edit documentation while waiting rather than starting GPU work on the login node.

These options follow ASU's [GPU resource requests](https://docs.rc.asu.edu/requesting-resources/) and [interactive-session guide](https://docs.rc.asu.edu/interactive-sessions/), checked October 22, 2026. Account-specific allocation restrictions and installed modules still need checking in your session.

## 3. Select installed tools

```bash
module spider cuda
module spider cmake
```

Select exact installed module names; `CUDA_MODULE_NAME` and `CMAKE_MODULE_NAME` below are placeholders. Inspect `module spider CUDA_MODULE_NAME` for required compiler modules, then load those prerequisites first.

```bash
module load CUDA_MODULE_NAME
# Only if cmake is not already available:
module load CMAKE_MODULE_NAME
module list
nvcc --version
cmake --version
g++ --version
```

Use ASU's provided CUDA installation. You do not need to install a driver or CUDA Toolkit on your laptop to execute this course on Sol. If installing any local tools later, use D:\ in accordance with your workstation policy.

## 4. Put the repository on Sol

After publishing with the [GitHub tutorial](github-publishing.md):

```bash
mkdir -p "$HOME/projects"
cd "$HOME/projects"
git clone https://github.com/haminhson1st/cuda-manufacturing-portfolio.git
cd cuda-manufacturing-portfolio
```

That URL is a proposed destination, not a claim that the repository already exists. Before publishing, you can upload the folder through the Sol portal. Keep large datasets and run artifacts in storage appropriate to your ASU allocation; source code is small enough for a normal project directory.

## 5. Build and validate

```bash
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-gpu --parallel 4
ctest --test-dir build-gpu --output-on-failure
./build-gpu/calibrate_cuda 1000003
compute-sanitizer --tool memcheck ./build-gpu/calibrate_cuda 1000003
```

NVIDIA lists A100 as compute capability 8.0, hence CMake value `80`. [GPU capability table](https://developer.nvidia.com/cuda/gpus). If you request another GPU, verify its value and use a separate build directory.

Expected behavior: successful exit, CSV header and one result row, and `correct=true`. Timing values depend on the allocation. Do not compare them with imaginary target numbers. For the exact timing boundaries, read [Lab 01](../tutorials/01-first-gpu.md).

## 6. Submit a batch job

Set the exact installed module name and any prerequisite modules in your shell. The script checks the required variable rather than guessing a version:

```bash
export CUDA_MODULE='REPLACE_WITH_INSTALLED_CUDA_MODULE'
export CMAKE_MODULE='REPLACE_WITH_INSTALLED_CMAKE_MODULE'
sbatch --export=ALL slurm/lab01.sbatch
squeue -u "$USER"
```

Omit `CMAKE_MODULE` when CMake is already available. Submit from the repository root. The job writes logs and measurements into an ignored `runs/` subdirectory. The script prints their location. After inspection, copy selected measurements and a manifest into `results/` for publication.

Run `exit` to release an interactive allocation when finished.

## Troubleshooting

| Symptom | Next check |
| --- | --- |
| `nvcc` not found | CUDA module and its compiler prerequisites |
| `nvidia-smi` fails | Whether the GPU allocation has actually begun |
| CMake cannot find a compatible host compiler | Match the host compiler to the selected CUDA module |
| Invalid partition/GPU request | Current ASU resource documentation and your account restrictions |
| Job remains pending | `squeue`; inspect the scheduler's reason rather than repeatedly submitting |
| Nsight Compute reports `ERR_NVGPUCTRPERM` | Ask ASU RC about permitted profiling; continue timing without restricted counters |

Do not benchmark a debug build or include sanitizer execution time in a performance claim.
