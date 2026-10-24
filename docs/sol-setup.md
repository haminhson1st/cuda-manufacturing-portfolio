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
```

Select the exact installed module name; `CUDA_MODULE_NAME` below is a placeholder. Inspect `module spider CUDA_MODULE_NAME` for required compiler modules, then load those prerequisites first. The build uses GNU make, which is part of the standard Linux toolchain, so no build-system module is needed.

```bash
module load CUDA_MODULE_NAME
module list
nvcc --version
g++ --version
make --version
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
make -j 4 lab01
make test-lab01
./build/calibrate_cuda 1000003
make sanitize-lab01
make SOLUTION=1 -j 4 && make SOLUTION=1 test-gpu   # optional: every lab against the reference solutions
```

NVIDIA lists A100 as compute capability 8.0, hence the default `CUDA_ARCH=80` ([GPU capability table](https://developer.nvidia.com/cuda/gpus)). If you request another GPU, verify its value and pass it, for example `make CUDA_ARCH=90`, or use a separate directory with `BUILD=build-sm90`. A changed flag rebuilds the affected objects automatically. `make help` lists every target.

Expected behavior: successful exit, CSV header and one result row, and `correct=true`. Every test line starts with `PASS`. Timing values depend on the allocation. Do not compare them with imaginary target numbers. For the exact timing boundaries, read [Lab 01](../tutorials/01-first-gpu.md).

The Labs 02–12 tests fail until you implement their kernels. That is how the starter code is meant to behave. The `SOLUTION=1` run tells you whether the toolkit, harness, and GPU work before you start. The tests need only `python3` with its standard library. The real-data converters for Labs 04 and 07 additionally need NumPy/SciPy and Pillow, which a Python module on Sol can provide.

## 6. Submit a batch job

Set the exact installed module name and any prerequisite modules in your shell. The script checks the required variable rather than guessing a version:

```bash
export CUDA_MODULE='REPLACE_WITH_INSTALLED_CUDA_MODULE'
sbatch --export=ALL slurm/lab01.sbatch
sbatch --export=ALL,LAB=03 slurm/lab.sbatch               # any lab 01-06: build, test, sanitize, benchmark
sbatch --export=ALL,LAB=03,SOLUTION=1 slurm/lab.sbatch    # the same against the reference solution
squeue -u "$USER"
```

Submit from the repository root. Each job writes logs and measurements into an ignored `runs/` subdirectory and prints its location. After inspection, copy selected measurements and a manifest into `results/` for publication.

Run `exit` to release an interactive allocation when finished.

## Troubleshooting

| Symptom | Next check |
| --- | --- |
| `nvcc` not found | CUDA module and its compiler prerequisites |
| `nvidia-smi` fails | Whether the GPU allocation has actually begun |
| `nvcc` rejects the host compiler version | Load the compiler module that `module spider CUDA_MODULE_NAME` lists, or pass `NVCC="nvcc -ccbin <g++ path>"` |
| `make` reports `missing separator` | A Makefile recipe line lost its leading tab; restore it (editors that expand tabs cause this) |
| Invalid partition/GPU request | Current ASU resource documentation and your account restrictions |
| Job remains pending | `squeue`; inspect the scheduler's reason rather than repeatedly submitting |
| Nsight Compute reports `ERR_NVGPUCTRPERM` | Ask ASU RC about permitted profiling; continue timing without restricted counters |

Do not benchmark a debug build or include sanitizer execution time in a performance claim.
