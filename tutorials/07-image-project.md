# 07 — Build a visual inspection baseline

**Time:** 6–8 hours. **Business task:** assign each image a defect score, then route suspect items for review.

| You edit | Provided | Reference solution |
| --- | --- | --- |
| [src/lab07/gpu_inspect.cu](../src/lab07/gpu_inspect.cu), plus your threshold study and report | Application [main.cu](../src/lab07/main.cu) (`score_images`), contract [lab07.hpp](../src/lab07/lab07.hpp), [synthetic images](../scripts/lab07_synthetic.py), [MVTec converter](../scripts/mvtec_to_pgm.py), [evaluator](../scripts/lab07_evaluate.py) | [solutions/lab07/gpu_inspect.cu](../solutions/lab07/gpu_inspect.cu) |

Read the [visual-inspection brief](../projects/02-visual-inspection/README.md). `score_images` links **your** Lab 05 and Lab 06 kernels, so those labs must pass first. `make SOLUTION=1 test-lab07` uses the reference versions of all three.

## The detector

A deliberately small texture baseline, assembled from earlier labs:

| Stage | Implementation |
| --- | --- |
| Gray | Lab 05 `launch_rgb_to_gray` (color images only) |
| Normalize | Lab 05 level stretch, `--lo`/`--hi` (default 0/255, which leaves values unchanged) |
| Smooth | Lab 06 tiled box filter, `--radius` (default 3) |
| Residual and mask | Lab 05 `launch_absdiff_mask` on the normalized and smoothed images; a residual above `--pixel-threshold` (default 30) marks the pixel |
| Score | mask pixels / image pixels. **This is the new reduction you write** |
| Decision | `predicted_defect = score > --image-threshold`, a threshold chosen on development images and then frozen |

The residual is large wherever intensity changes faster than the smoothing window: scratches, spots, and cracks, but also edges and normal texture. Both thresholds are parameters. This is a learning baseline with known limitations, not a promise of high MVTec performance.

## Concepts

**Atomics finish a reduction across blocks.** Each block can reduce its own pixels with the Lab 03 tree, but blocks cannot synchronize with each other. An atomic read-modify-write combines their totals: `atomicAdd` reads a word in global or shared memory, adds to it, and writes it back as one indivisible operation, and it supports `unsigned long long` (CUDA Programming Guide 13.4.2 §5.4.5.1, printed pp. 605–606 / PDF pp. 623–624). H&P presents atomic read-modify-write primitives as the foundation of all synchronization, and warns that contended synchronization becomes a bottleneck (6e §5.5, printed p. 412 / PDF p. 444). Hence one atomic per **block**, not one per pixel.

**Exact by construction.** Integer addition is associative, so the count does not depend on the order in which blocks finish, and every stage before it is integer arithmetic too. CPU and GPU mask counts must therefore be identical, and `--backend both` checks exactly that. A floating-point pipeline would need Lab 04's near-threshold analysis instead.

**Convenience has a cost you will measure later.** The reference `score_gpu` allocates its device images afresh for every image. That is simple and correct, but `cudaMalloc` and `cudaFree` can be expensive and synchronizing; the Guide's stream-ordered allocator exists partly to avoid that cost (§4.3.2, printed p. 261 / PDF p. 279). `gpu_wall_ms` includes the allocations, while `gpu_compute_ms` does not. Lab 09 profiles exactly this kind of choice, so keep your first version as the baseline.

## Implement

0. On Sol: `make SOLUTION=1 test-lab07`.
1. **`count_mask_kernel`.** One thread per pixel on a 2D grid. Out-of-image threads contribute 0 and still reach every barrier. Reduce the block's 0/1 values in shared memory, then make one `atomicAdd` per block. The launcher is written for you: it zeroes the counter and checks that the thread count is a power of two.
2. **`score_gpu`.** Compose the stages on the device, as Lab 04 composed its pipeline: allocate, upload, five launches checked with `CUDA_CHECK_LAUNCH()`, then download the count (and the mask when requested). Time the three phases with `common::EventTimer`.
3. Run `make test-lab07` (synthetic set, tiny/thin/RGB edge images, three block shapes, radius 0/1/5, mask output, evaluator smoke test, malformed inputs), then `make sanitize-lab07`.

| Symptom | Likely cause |
| --- | --- |
| `score_gpu is not implemented yet` | Starter still throws |
| GPU count is 0 while the CPU count is positive | Kernel body empty, or the atomic was never reached |
| GPU count exceeds the CPU count | The counter was not zeroed, or out-of-image threads counted padding bytes |
| Hang, or a synccheck error | Early `return` before a barrier |
| Disagreement only for color images | Grayscale conversion skipped, or the wrong source passed to normalize |

## Rehearse on synthetic images

```bash
python3 scripts/lab07_synthetic.py --output runs/lab07-syn
./build/score_images --list runs/lab07-syn/images.csv --split dev --backend both --output runs/lab07-syn/dev.csv
python3 scripts/lab07_evaluate.py calibrate --scores runs/lab07-syn/dev.csv --false-reject 0.05
./build/score_images --list runs/lab07-syn/images.csv --split test --image-threshold 0 --output runs/lab07-syn/test.csv
python3 scripts/lab07_evaluate.py evaluate --scores runs/lab07-syn/test.csv --threshold 0
```

`calibrate` reads good images from the dev split only. It picks the threshold whose dev false-reject rate stays within the target and refuses rows from the test split. On the synthetic set, good images score 0 and every scratch or spot scores above 0, so calibration picks 0 and the test split separates cleanly. That rehearses the procedure; it is not evidence about real parts.

## Use one MVTec AD category

1. Download one category from [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad). It is licensed CC BY-NC-SA 4.0 (noncommercial). Record the source, license, and archive checksum, and keep the images outside Git, for example under `data/raw/mvtec/bottle`.
2. Convert it once. This needs Pillow, so load a Python module that provides it on Sol:

   ```bash
   python3 scripts/mvtec_to_pgm.py data/raw/mvtec/bottle --output data/processed/mvtec-bottle
   ```

   The converter writes PGM/PPM images, an `images.csv`, and a `manifest.json` following the [dataset template](../templates/dataset-manifest.json). The last 20% of the sorted `train/good` images become the `dev` split; the official test images become `test`.
3. Tune `--radius`, `--pixel-threshold`, and `--lo`/`--hi` on `train` and `dev` only. Save masks (`--save-masks DIR`, with `DIR` created first) to see what the detector reacts to. Then choose the image threshold with `calibrate`.
4. **Freeze** every parameter in your notebook. Then score `--split test` once with `--image-threshold` and run `evaluate`.
5. If you tune with labeled anomalies, carve your own development subset out of the test defects and evaluate only on the rest. Say so in the report: the original benchmark protocol is then no longer preserved.

## Evaluate

`evaluate` reports TP, FP, TN, and FN with defects as positives, defect recall, the false-reject rate, recall per defect type, the worst misses, the worst false rejects, and images near the threshold. CPU/GPU agreement is a separate numerical test that `--backend both` already enforces.

Examine at least three failures with their saved masks: illumination changes, normal texture mistaken for damage, small defects erased by smoothing, or defects with low contrast. Describe them even when the output looks unattractive. Report the timing scopes from the summary row: decode, CPU, the GPU phases, and `gpu_wall_ms`.

## Publish

Write a case study with data selection, threshold method, quality metrics, timing scopes, and permitted examples. Save the exact parameters and source commit. Acceptance requires reproducibility and honest limitations, not a particular accuracy percentage.

**Extension:** try `__syncthreads_count(predicate)`, which returns how many threads of the block have a nonzero predicate (Guide printed p. 598 / PDF p. 616), and compare it with your explicit tree. For aligned objects, try a reference-image detector: the Lab 05 difference against a mean good image. Continue to [library comparison](08-library-comparison.md).
