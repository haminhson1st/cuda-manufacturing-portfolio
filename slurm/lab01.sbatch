#!/bin/bash -l
#SBATCH --job-name=cuda-lab01
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=lab01-%j.out

set -euo pipefail
: "${CUDA_MODULE:?Export CUDA_MODULE with the installed module name before sbatch.}"
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root.}"
module load "$CUDA_MODULE"
if [[ -n "${CMAKE_MODULE:-}" ]]; then module load "$CMAKE_MODULE"; fi

run_dir="runs/lab01-${SLURM_JOB_ID}"
mkdir -p "$run_dir"
{
    date -u
    hostname
    git rev-parse HEAD || true
    git status --short
    module list
    nvidia-smi
    nvcc --version
    cmake --version
    g++ --version
    lscpu
} > "$run_dir/environment.txt" 2>&1

cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-gpu --parallel "${SLURM_CPUS_PER_TASK:-4}"
ctest --test-dir build-gpu --output-on-failure | tee "$run_dir/tests.txt"
./build-gpu/calibrate_cpu 1000003 "$run_dir/cpu-raw.csv" > "$run_dir/cpu.csv"
./build-gpu/calibrate_cuda 1000003 "$run_dir/cuda-raw.csv" > "$run_dir/cuda.csv"
compute-sanitizer --tool memcheck --error-exitcode 3 \
    ./build-gpu/calibrate_cuda 1000003 > "$run_dir/sanitizer.txt" 2>&1
printf 'Run artifacts: %s\n' "$run_dir"
