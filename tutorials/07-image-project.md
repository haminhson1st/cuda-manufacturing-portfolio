# 07 — Build a visual inspection baseline

**Time:** 6–8 hours. **Business task:** assign each image a defect score, then route suspect items for review.

Read [the visual-inspection brief](../projects/02-visual-inspection/README.md). Use one category from [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad). Record the source and noncommercial license. Keep the original dataset outside Git.

## Define a deliberately small detector

One possible texture baseline is grayscale normalization, local smoothing, absolute residual from the smoothed image, and the fraction of pixels whose residual exceeds a pixel threshold. Then compare this fraction with an image-level threshold. Both thresholds are parameters, and normal texture may also generate large residuals.

This is a learning baseline with known limitations, not a promise of high MVTec performance. For aligned objects, a reference-image approach is another experiment; document alignment assumptions.

## Implement

1. Assemble the same stages on CPU and GPU using the previous labs.
2. Reuse a reduction to convert a mask into an image score.
3. Produce `image_id,score,predicted_defect` plus optional masks for inspection.
4. Reserve a development subset of normal training images for threshold calibration. Keep the official test set untouched until the rule is frozen.
5. If you use labeled anomalies for tuning, explicitly create a separate development split and evaluate only on the remaining held-out examples; do not imply the original benchmark protocol was preserved.

## Evaluate

Record TP, FP, TN, and FN with defects as positives. Report recall and false-reject rate, plus the number of examples. CPU/GPU score agreement is a separate numerical test. If an item lies near a threshold, inspect whether numerical tolerance changes its decision.

Examine at least three failures, such as illumination changes, normal texture mistaken for damage, or small defects erased by smoothing. Describe them even when the output looks unattractive.

## Publish

Write a case study with data selection, threshold method, quality metrics, timing scopes, and permitted examples. Save the exact parameters and source commit. Acceptance requires reproducibility and honest limitations, not a particular accuracy percentage.

Continue to [library comparison](08-library-comparison.md).
