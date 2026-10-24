# 04 — Ship the sensor-screening case study

**Time:** 6–8 hours. **Business task:** flag unusual windows in recorded bearing vibration. This is a laboratory screening exercise, not a validated predictor of remaining machine life.

| You edit | Provided | Reference solution |
| --- | --- | --- |
| [src/lab04/gpu_pipeline.cu](../src/lab04/gpu_pipeline.cu), plus your threshold study and report | Application [screen.cu](../src/lab04/screen.cu), contract [lab04.hpp](../src/lab04/lab04.hpp), [synthetic data](../scripts/lab04_synthetic.py), [CWRU converter](../scripts/cwru_to_f32.py), [evaluator](../scripts/lab04_evaluate.py) | [solutions/lab04/gpu_pipeline.cu](../solutions/lab04/gpu_pipeline.cu) |

Read the [project brief](../projects/01-sensor-screening/README.md). `screen` links **your** Lab 02 and Lab 03 kernels, so those labs must pass first. `make SOLUTION=1 test-lab04` uses the reference versions of all three.

## Concepts

**Keep intermediates on the device.** The calibrated signal produced by Lab 02's kernel is the input of Lab 03's kernel. Copying it back to the host in between would add two transfers and no information. Device memory persists between launches, and launches in one stream run in order, so the second kernel sees the first kernel's output without host synchronization.

**Measure the whole application.** The summary row separates `read_ms`, `cpu_ms`, `h2d_ms`, `gpu_compute_ms`, `d2h_ms`, `gpu_wall_ms` (which also includes device allocation and, for the first CUDA call in the process, context setup), `report_ms`, and `total_ms`. Amdahl's Law bounds the result: if an enhancement applies to only part of the task, overall speedup cannot exceed 1/(1 − that fraction) (H&P 6e §1.9, printed p. 50 / PDF p. 82). Report how much of `total_ms` the kernels actually are. Host buffers here are pageable. Page-locked (pinned) host memory speeds up synchronous copies and is required for asynchronous ones (CUDA Programming Guide 13.4.2, printed p. 124 / PDF p. 142). That is Lab 10's topic.

**Correct code can implement a poor detector.** `--backend both` checks CPU/GPU agreement: statistics within the Lab 03 error bound, then identical alarm decisions. Windows whose feature lies within 10⁻⁵ of a threshold are counted as `borderline`, not as disagreements. Whether the alarm separates normal from faulty recordings is a separate question, answered by the evaluator.

## Implement the GPU path

Follow the TODO in [gpu_pipeline.cu](../src/lab04/gpu_pipeline.cu): allocate with `common::DeviceBuffer`, upload, launch `lab02::launch_calibrate` with one channel and infinite limits, then `lab03::launch_window_stats` on the calibrated buffer. Download the statistics and time the three phases with `common::EventTimer`. Then:

```bash
make test-lab04       # synthetic recordings, CPU/GPU agreement, edge cases, evaluator smoke test
make sanitize-lab04
```

## Rehearse on synthetic data

The synthetic generator writes four normal and four fault recordings (12 kHz, about 2 s each), a `labels.csv` that keeps whole recordings in `dev` or `test`, and two tiny edge-case files. Fault recordings contain periodic decaying impacts, a simplified stand-in for a localized defect.

```bash
python3 scripts/lab04_synthetic.py --output runs/lab04-syn
mkdir -p runs/lab04-syn/reports
for f in runs/lab04-syn/*.f32; do
  ./build/screen --input "$f" --window 2048 --backend both --header 0 \
    --output "runs/lab04-syn/reports/$(basename "$f" .f32).csv"
done
python3 scripts/lab04_evaluate.py sweep --labels runs/lab04-syn/labels.csv --feature crest runs/lab04-syn/reports/*.csv
```

`sweep` reads **only the dev split**. It lists thresholds at quantiles of the normal windows' feature values, with window alarm rates and recording-level outcomes. The synthetic faults are deliberately easy: RMS, peak, and crest factor (peak/RMS) each separate the dev classes completely. This rehearses the procedure; it is not evidence that a feature works. Expect real recordings, with their load and speed differences, to be much less tidy.

## Use real recordings

1. Download a small selection from the [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter), for example normal baseline files plus one fault type at several loads. Store them under `data/raw/cwru/` (ignored by Git) and keep the page you selected them from.
2. Check each file's **sampling rate** on the CWRU page. A 2048-sample window spans different durations at 12 kHz and at 48 kHz, so do not mix rates without resampling.
3. Convert each file. This needs NumPy and SciPy, so load a Python module that provides them on Sol. The file number and condition below are only an example; copy yours from the CWRU table:

   ```bash
   python3 scripts/cwru_to_f32.py data/raw/cwru/105.mat --output-dir data/processed/cwru \
     --channel DE --sample-rate 12000 --label fault --condition "inner race 0.007 in, 0 hp"
   ```

   The JSON sidecar records the source hash, variable, rpm, condition, and conversion error. Copy these fields into a [dataset manifest](../templates/dataset-manifest.json).
4. Write `labels.csv` by **recording**: never split adjacent windows of one recording across dev and test. Note load and speed differences so the report does not attribute every difference to fault status.
5. Run `screen` without thresholds to get feature reports, then `sweep` each feature on dev. Pick one rule and `--min-fraction`, and **freeze them** in your notebook.
6. Re-run `screen` with the frozen thresholds (for example `--crest-threshold 4.1`), then evaluate the held-out split once:

   ```bash
   python3 scripts/lab04_evaluate.py evaluate --labels data/processed/cwru/labels.csv --split test runs/lab04-cwru/reports/*.csv
   ```

## Publish

Write a case study with [the template](../templates/case-study.md). Include one result table, one failure example, the threshold-selection procedure, and the timing scopes from the summary rows. Show the commands and the source commit, and keep the raw recordings untracked. For a scaling experiment, replaying or concatenating recordings is acceptable if labeled. It adds compute work, not independent fault examples.

**Acceptance:** one command per recording produces the CSV, CPU/GPU agreement passes, the held-out evaluation is documented separately from threshold selection, and every performance claim names its timing boundary. Update the project status only when these artifacts exist.

**Extension:** frequency-domain features with cuFFT after the core course. Continue to [image indexing](05-image-indexing.md).
