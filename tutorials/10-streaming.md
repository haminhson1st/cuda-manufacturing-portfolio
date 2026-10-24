# 10 — Batch inputs and coordinate streams

**Time:** 6–8 hours. **Business task:** increase sustained processing rate while respecting per-item latency.

Study [simpleStreams](https://github.com/NVIDIA/cuda-samples/tree/master/cpp/0_Introduction/simpleStreams) and [asynchronous execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html).

## Start with a serial pipeline

Keep a correct one-stream implementation as the baseline. Use a recorded sequence of inputs and outputs with stable item IDs. Allocation and decoding policies must be explicit.

## Add two reusable slots

1. Allocate separate pinned host buffers and device buffers for two slots.
2. Give each slot a stream and completion event.
3. Before refilling a slot, wait for its previous work to finish and consume its output.
4. Enqueue H2D transfer, kernels, and D2H transfer in that slot's stream.
5. Record completion after the D2H operation. Keep buffers alive until completion.
6. Drain both slots at shutdown; do not drop the final partial batch.

Operations in one stream provide an ordering mechanism for that stream. Dependencies across streams need explicit coordination. Multiple streams alone do not guarantee overlap or a performance improvement; observe the timeline and hardware behavior.

## Verify

Use unique patterns per input so accidentally reusing a buffer is visible. Compare all outputs and IDs with the serial pipeline for one item, odd item counts, partial batches, and many alternating slots. Re-run memory checks; sanitizer success is only part of the evidence.

## Experiment

Try batch sizes 1, 4, 16, and 64 while keeping total work fixed. Measure throughput and per-item completion latency from a defined arrival point. Include batch-fill waiting time when simulating a live stream. For offline processing, state that all inputs are already available.

## Portfolio deliverable

Publish a table showing batching's effect, a timeline if overlap occurs, and a recommended configuration for your workload. Explain a case where batching is inappropriate. Acceptance is correct ownership/completion plus a defensible measurement, not guaranteed overlap.

Continue to [the capstone](11-capstone.md).
