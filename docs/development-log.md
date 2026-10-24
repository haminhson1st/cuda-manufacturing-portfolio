# Development log

## 2026-10-22 — learning scaffold

- Created the twelve-week tutorial sequence and manufacturing project briefs.
- Added AI-assisted calibration reference code and build/job scaffolding.
- No manufacturing implementation milestones or GPU performance claims completed.
- See [validation](validation.md) for checks actually performed.

## 2026-10-23 — make build and Labs 02–06 starter code

- Replaced CMake/CTest with a GNU Makefile: per-lab build, test, sanitizer, and benchmark targets; `SOLUTION=1` builds the reference solutions into `build-solution/`.
- Added shared helpers (`src/common`), harnesses and TODO kernel files for Labs 02–06, reference solutions, Lab 04 data tools, a generic Slurm job, and a CUDA compile-only CI job.
- Rewrote tutorials 02–06 around the starter code, with course citations retrieved through the CSE520 KNS.
- Still not compiled or executed: see [validation](validation.md). No measurements.

## 2026-10-23 — Labs 07–08 starter code

- Lab 07: `score_images` application (image lists, PGM/PPM input, CPU/GPU agreement, mask output), a TODO file for the mask-count reduction and detector composition, the synthetic image set, the MVTec AD converter, and a calibrate/evaluate script that refuses test rows during calibration.
- Lab 08: `reduce_compare` harness measuring setup, first-call, and steady-state cost for custom, Thrust, and CUB backends, with exact CPU checks and a pattern that catches byte-sum shortcuts.
- Rewrote tutorials 07–08 around the starter code, with KNS citations.
- Still not compiled or executed: see [validation](validation.md). No measurements.

## 2026-10-23 — Labs 09–10 starter code

- Lab 09: `profile_inspect` harness comparing the Lab 07 baseline with a buffer-reusing `Inspector` (TODO file), with NVTX ranges, CPU and baseline cross-checks, and an allocation-count check; `make profile-lab09` captures Nsight Systems and Nsight Compute reports.
- Lab 10: `stream_batches` harness with a provided serial baseline and a TODO file for the two-slot pinned-memory stream pipeline; checks bit-identical results and batch coverage, and reports throughput and per-item latency under offline and simulated-arrival models.
- Added `PinnedBuffer`, `Stream`, and `Event` RAII helpers and `common::NvtxRange`.
- Rewrote tutorials 09–10 around the starter code, with KNS citations.
- Still not compiled or executed: see [validation](validation.md). No measurements.

## 2026-10-23 — Labs 11–12 starter code

- Lab 11: the `inspect` application (images and sensors subcommands, configuration files, a JSON run record per report) with a TODO file for both workflows; frozen synthetic configurations in `configs/`; `scripts/compare_reports.py`; integration tests against `score_images` and `screen`; `slurm/capstone.sbatch`.
- Lab 12: `release_bench` measuring single-thread CPU, a multicore CPU baseline (TODO file), and both GPU versions with raw, summary, and environment outputs; `scripts/lab12_report.py` writes `report.md` and flags invalid comparisons.
- Rewrote tutorials 11–12 around the starter code, with KNS citations.
- Still not compiled or executed: see [validation](validation.md). No measurements.

## Add your next entry

Record date, source commit, your changes, tests actually run, experiment result, and next question. Link the associated issue or pull request after publishing.
