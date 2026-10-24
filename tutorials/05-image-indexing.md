# 05 — Map threads to image pixels

**Time:** 6–8 hours. **Business task:** prepare inspection images and identify pixel differences.

Read 2D indexing and memory-access examples in [Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html).

## Make indexing explicit

Specify width, height, channel order, element type, and row stride. An image's row stride may exceed `width*channels`; do not silently assume tightly packed input when your loader permits padding.

Start with generated grayscale arrays to isolate indexing. Add image loading through a small established library or OpenCV later; do not spend the lab writing a PNG decoder.

## Implement

1. Generate gradients, constant fields, checkerboards, and a rectangle whose location is known.
2. Write CPU and CUDA versions of normalization and absolute difference.
3. Use `x = blockIdx.x*blockDim.x + threadIdx.x` and the corresponding `y` expression, then guard both coordinates.
4. Add RGB-to-grayscale conversion with fixed documented coefficients and identical rounding rules on CPU/GPU.
5. Add a binary difference mask. Save a viewable output using a simple format or image library.

For RGB inputs, explicitly name RGB versus BGR order. For byte outputs, define rounding and clamping. Do the arithmetic in a suitable wider type to avoid unsigned subtraction wraparound.

## Verify

Test 1×1, one-row/one-column images, 641×479, and a padded-stride input. Compare every pixel or use a documented tolerance for floating intermediate results. Check that padding and guard regions remain unchanged. Run memcheck.

## Experiment

Compare block shapes such as 16×16 and 32×8 on the same data. Save timings and document the mapping; do not assume a shape wins universally. Keep decode time outside kernel measurements and visible in full-pipeline measurements.

**Portfolio evidence:** one input/output pair, indexing explanation, edge-case results, and a timing table. Reference-image subtraction is only a useful detector when viewpoint/alignment and illumination make the comparison meaningful. Continue to [filtering](06-tiled-filter.md).
