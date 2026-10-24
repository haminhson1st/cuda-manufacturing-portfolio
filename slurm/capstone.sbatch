#!/bin/bash -l
#SBATCH --job-name=inspect
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=inspect-%j.out

# One documented capstone run (Lab 11). Record this resource request in your report.
# Submit from the repository root, for example:
#   export CUDA_MODULE='REPLACE_WITH_INSTALLED_CUDA_MODULE'
#   sbatch --export=ALL,MODE=images,INPUT=data/processed/mvtec-bottle/images.csv,CONFIG=configs/mvtec-bottle.conf,EXTRA="--split test" slurm/capstone.sbatch
#   sbatch --export=ALL,MODE=sensors,INPUT=data/processed/cwru/105-DE.f32,CONFIG=configs/cwru-inner-race.conf slurm/capstone.sbatch

set -euo pipefail
: "${CUDA_MODULE:?Export CUDA_MODULE with the installed module name before sbatch.}"
: "${MODE:?Export MODE=images or MODE=sensors.}"
: "${INPUT:?Export INPUT with the input directory, list, or recording.}"
: "${CONFIG:?Export CONFIG with the frozen configuration file.}"
EXTRA="${EXTRA:-}"
BACKEND="${BACKEND:-cuda}"
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root.}"
module load "$CUDA_MODULE"

run_dir="runs/inspect-${SLURM_JOB_ID}"
mkdir -p "$run_dir"
{
    date -u
    hostname
    git rev-parse HEAD || true
    git status --short || true
    module list
    nvidia-smi
    nvcc --version
    g++ --version
    lscpu
} > "$run_dir/environment.txt" 2>&1

make -j "${SLURM_CPUS_PER_TASK:-4}" lab11
SOURCE_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
export SOURCE_COMMIT
cp "$CONFIG" "$run_dir/"
# EXTRA is split into words on purpose, e.g. EXTRA="--split test --save-masks runs/masks".
# shellcheck disable=SC2086
./build/inspect "$MODE" --config "$CONFIG" --input "$INPUT" --backend "$BACKEND" \
    --output "$run_dir/report.csv" $EXTRA | tee "$run_dir/summary.csv"
printf 'Run artifacts: %s (report.csv, report.csv.json, summary.csv, environment.txt)\n' "$run_dir"
