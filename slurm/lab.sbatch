#!/bin/bash -l
#SBATCH --job-name=cuda-lab
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=lab-%j.out

# Build, test, sanitize, and (where defined) benchmark one lab. Submit from the repository root:
#   export CUDA_MODULE='REPLACE_WITH_INSTALLED_CUDA_MODULE'
#   sbatch --export=ALL,LAB=02 slurm/lab.sbatch               # your code in src/
#   sbatch --export=ALL,LAB=02,SOLUTION=1 slurm/lab.sbatch    # the reference solution
# Labs 04, 07, and 09-12 also need python3 (standard library only) for their synthetic test data.
# For Lab 09 profiles, run `make profile-lab09` in the job or an interactive session instead. For a
# documented capstone run on real data, use slurm/capstone.sbatch. LAB=12 runs bench-lab12, which
# reads RELEASE_LIST, RELEASE_SPLIT, and RELEASE_CONFIG from the exported environment.

set -euo pipefail
: "${CUDA_MODULE:?Export CUDA_MODULE with the installed module name before sbatch.}"
: "${LAB:?Export LAB=01 ... 12.}"
SOLUTION="${SOLUTION:-0}"
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root.}"
module load "$CUDA_MODULE"

run_dir="runs/lab${LAB}-${SLURM_JOB_ID}"
mkdir -p "$run_dir"
{
    date -u
    hostname
    echo "LAB=$LAB SOLUTION=$SOLUTION"
    git rev-parse HEAD || true
    git status --short || true
    module list
    nvidia-smi
    nvcc --version
    g++ --version
    make --version | head -n 1
    lscpu
} > "$run_dir/environment.txt" 2>&1

make -j "${SLURM_CPUS_PER_TASK:-4}" SOLUTION="$SOLUTION" "lab${LAB}"
make SOLUTION="$SOLUTION" "test-lab${LAB}" 2>&1 | tee "$run_dir/tests.txt"
make SOLUTION="$SOLUTION" "sanitize-lab${LAB}" > "$run_dir/sanitizer.txt" 2>&1
if make -n "bench-lab${LAB}" > /dev/null 2>&1; then
    make SOLUTION="$SOLUTION" RUN_DIR="$run_dir" "bench-lab${LAB}"
fi
printf 'Run artifacts: %s\n' "$run_dir"
