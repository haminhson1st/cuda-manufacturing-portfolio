# 06 — Reuse neighboring pixels with shared memory

**Time:** 6–8 hours. **Business task:** smooth image noise before computing an inspection score.

Inspect NVIDIA's [separable convolution sample](https://github.com/NVIDIA/cuda-samples/tree/master/cpp/2_Concepts_and_Techniques/convolutionSeparable). Your task is a small, understandable filter comparison; reproducing every sample optimization is unnecessary.

## Establish one mathematical operation

Choose a 3×3 box filter initially. Write its coefficients, output type, and boundary rule. Use zero padding or clamping consistently in the CPU and GPU versions. Preserve an out-of-place input/output arrangement so neighbors are not overwritten during the operation.

## Implement and inspect

1. Write the direct CPU reference and a direct GPU kernel that loads neighbors from global memory.
2. Verify them before optimizing.
3. Add a shared-memory tile containing the output region plus its halo. Cooperatively load all required tile elements, including halos.
4. Synchronize after loading. Every participating block thread must reach the barrier, including threads whose output coordinate is outside the image.
5. Guard output stores. Do not return early before a required block barrier.
6. Repeat for a larger filter only after the small case works.

If comparing a separable two-pass filter to a direct 2D filter, ensure that coefficients and boundary rules implement the same intended operation. Label algorithmic changes separately from memory-layout changes.

## Tests

A constant field tests normalization; an impulse tests the kernel shape; a corner impulse tests boundaries. Also test non-multiple dimensions and tiny images smaller than the filter. Run memcheck, racecheck, and synccheck.

## Experiment

Measure direct/tiled versions for several image sizes and filter radii. Record the extra halo work and shared-memory bytes per block. Hypothesize why the result changes with radius, then use measurements to evaluate that hypothesis. Shared memory is an implementation mechanism, not a guarantee of speedup.

**Deliverable:** a before/after table with equivalent outputs and a short tradeoff explanation. For the course's coalescing/bank-conflict reading, see [references](../docs/references.md). Continue to [the image case study](07-image-project.md).
