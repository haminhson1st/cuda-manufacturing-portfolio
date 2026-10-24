# 11 — Assemble a reproducible inspection application

**Time:** 6–8 hours. **Business task:** let another engineer process an image collection or sensor recording with one documented command.

| You edit | Provided | Reference solution |
| --- | --- | --- |
| [src/lab11/app.cu](../src/lab11/app.cu), plus your frozen configuration, workload definition, and fresh-clone instructions | The `inspect` command line [main.cu](../src/lab11/main.cu), contract [lab11.hpp](../src/lab11/lab11.hpp), [frozen configurations](../configs/README.md), [compare_reports.py](../scripts/compare_reports.py), [capstone.sbatch](../slurm/capstone.sbatch) | [solutions/lab11/app.cu](../solutions/lab11/app.cu) |

Read [the capstone brief](../projects/03-capstone/README.md). Prioritize integrating existing work: `inspect` links **your** Labs 02–07 and 09. Image inspection is the primary mode; sensor screening is a second mode with an independent dataset. Do not pair MVTec images with CWRU windows as if they described the same machine. They are separate workflows that share engineering techniques.

## The interface

```text
inspect images  --input DIR|LIST.csv|IMAGE.pgm --output REPORT.csv [--config FILE] [--backend cpu|cuda|both] [options]
inspect sensors --input RECORDING.f32 --output REPORT.csv [--config FILE] [--backend cpu|cuda|both] [options]
```

`inspect images --help` and `inspect sensors --help` list every option. Any option can also come from a configuration file of `name = value` lines; the command line overrides the file, and a misspelled name fails the run exactly like a misspelled option. Every successful run writes `REPORT.csv.json`, a run record holding the effective configuration, the source commit (from `SOURCE_COMMIT`), the GPU and CUDA versions, and the summary.

| Output | Contents |
| --- | --- |
| Images report | `image_id,source,width,height,mask_pixels,score,predicted_defect` |
| Sensors report | The Lab 04 format: `recording_id,window_start,valid_samples,mean,rms,peak,crest,alarm` |
| Summary line (stdout) | `mode,backend,items,flagged,decode_ms,compute_ms,total_ms,items_per_s,item_p50_ms,item_p95_ms,disagreements,ok` |

## Concepts

**Reproducibility is the guiding principle.** H&P puts it directly: list everything another experimenter would need to duplicate the results (6e §1.8, printed p. 45 / PDF p. 77). For an application, that means a frozen configuration file, the run record, the source commit, and the Slurm resource request.

**Deterministic enumeration and stable IDs.** `readdir` returns directory entries in no particular order, so `list_images` sorts them. A list input uses its `image_id` column; a directory or single file uses the file stem. Two inputs with the same ID (for example `a.pgm` and `a.ppm`) must fail the run, because reports are joined by ID.

**Strict input validation.** Fail on the first missing, unsupported, or malformed input and name the file, rather than producing a report that silently skips items. `common::read_pnm` and `lab04::read_f32` already reject bad files; your job is to let their errors reach the user.

**Integration reuses decisions you already measured.** On the GPU, images go through one `lab09::Inspector` (Lab 09's buffer reuse); on the CPU, through `lab07::score_cpu`. `--backend both` must agree exactly for images, because the pipeline is integer arithmetic, and within the Lab 04 tolerance and threshold rule for sensors.

**A workload is more than a throughput number.** TPC benchmarks measure throughput only while a response-time limit is met (H&P 6e §1.8, printed p. 45 / PDF p. 77). Define an explicitly hypothetical target, for example four cameras at 25 images/s each: resolution, acceptable per-image latency, error tolerance, and how images arrive. The summary reports offline images/s and per-item p50/p95 decode-plus-compute times. Offline throughput alone cannot establish that a live deadline is met; Lab 10 showed how batching moves latency.

## Implement

0. On Sol: `make SOLUTION=1 test-lab11`.
1. `run_images`, following its TODO: enumerate the inputs, open the report, create one `Inspector` for the GPU path, then for each input decode, score, write the row, and optionally save the mask. Fill the summary.
2. `run_sensors`: decode, run the CPU and/or GPU pipeline, check agreement for `both`, write the Lab 04 report, and fill the summary.
3. Run `make test-lab11`, then `make sanitize-lab11`. The integration tests check that:
   - CPU, CUDA, and `both` reports agree;
   - `inspect images` matches `score_images` and `inspect sensors` matches `screen`, so the capstone is the labs' detector and not a new one;
   - decisions use `score > threshold` exactly at the boundary;
   - the run record holds the effective values;
   - directory, list, and single-file inputs and mask output work;
   - a malformed image, an empty directory, a missing input, a misspelled configuration key, out-of-range values, and the wrong configuration for the mode all fail.

| Symptom | Likely cause |
| --- | --- |
| `run_images is not implemented yet` | Starter still throws |
| `Keys differ` when compared with `score_images` | IDs not taken from the list or the file stem, or directory entries not filtered to `.pgm`/`.ppm` |
| Different row order from run to run | Inputs not enumerated in sorted order |
| `threshold` check fails | `>=` used instead of `>`, or the decision written when the threshold is unset |
| `disagreements` > 0 with `--backend both` | A stale buffer in your Lab 09 Inspector, or different detector parameters on the two paths |
| The malformed-image test passes when it should fail | The decoding error is caught and swallowed instead of reaching `main` |

## Freeze, run, and hand over

1. **Freeze** the rules from your Lab 04 and Lab 07 studies as `configs/<dataset>.conf` (start from [synthetic-images.conf](../configs/synthetic-images.conf)) and commit them before evaluating held-out data.
2. **Run on Sol** through the job script, which captures the environment and the effective configuration:

   ```bash
   export CUDA_MODULE='REPLACE_WITH_INSTALLED_CUDA_MODULE'
   sbatch --export=ALL,MODE=images,INPUT=data/processed/mvtec-bottle/images.csv,CONFIG=configs/mvtec-bottle.conf,EXTRA="--split test" slurm/capstone.sbatch
   ```

   Keep the frozen thresholds in the configuration file, not in `EXTRA`, so the committed file is the rule. Record the job's resource request (the `#SBATCH` lines) in your report.
3. **Compare with the workload.** For example, 100 images/s requires sustaining at most 10 ms per image on average, and the deadline constrains p95, not the mean. State which parts of a live deployment the offline measurement does not cover: arrival jitter, camera I/O, and other processes on the machine.
4. **Fresh-clone test.** Ask a reader, or yourself in a new directory, to follow only the README: data prerequisites, `make lab11`, `make test-lab11`, then a report from the documented command. Log every manual step that turned out to be missing, and fix the instructions.

## Deliverable

A short README section, or `docs/capstone.md`, with the commands, the data prerequisites, the frozen configuration, the hypothetical workload, and one example summary line with its run record.

**Acceptance:** a reproducible CLI, CPU/GPU numerical agreement, frozen detector parameters, a documented quality evaluation (from Labs 04 and 07), and a passing integration test.

**Extension:** an `--on-error skip` mode that records failures per item instead of stopping; overlapping the decoding of image *i + 1* with the GPU work on image *i* (Lab 10's two slots); or a check of the measured rate against a `--target-rate`. Continue to [the portfolio release](12-portfolio-release.md).
