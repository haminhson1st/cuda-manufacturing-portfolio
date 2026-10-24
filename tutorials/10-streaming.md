# 10 — Batch inputs and coordinate streams

**Time:** 6–8 hours. **Business task:** increase sustained processing rate while respecting per-item latency.

| You edit | Provided | Reference solution |
| --- | --- | --- |
| [src/lab10/pipeline.cu](../src/lab10/pipeline.cu) | Serial baseline [serial.cu](../src/lab10/serial.cu), harness [main.cu](../src/lab10/main.cu) (`stream_batches`), contract [lab10.hpp](../src/lab10/lab10.hpp), `PinnedBuffer`/`Stream`/`Event` in [cuda_utils.cuh](../src/common/cuda_utils.cuh) | [solutions/lab10/pipeline.cu](../solutions/lab10/pipeline.cu) |

Study [simpleStreams](https://github.com/NVIDIA/cuda-samples/tree/master/cpp/0_Introduction/simpleStreams) and [asynchronous execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html).

## The workload

A recorded sequence of sensor items (default: 256 items of 65,536 samples). Each item is calibrated with **your** Lab 02 kernel and reduced to window statistics with **your** Lab 03 kernel. Item IDs are stable, and every item carries its own offset, so a batch that reads or writes the wrong buffer produces different results. A batch concatenates whole items; the item length is a multiple of the window, so windows never cross items.

Arrivals: with `--interval-ms 0` the run is offline and every item is already available. With `--interval-ms T`, item *i* arrives *i·T* ms after the start, and a batch can start only once its last item has arrived (batch-fill waiting). Per-item latency is the completion time of the item's batch (measured with CUDA events) minus the item's arrival time.

## Concepts

**Streams order work; events observe it.** Commands in one stream run in issue order; commands in different streams may overlap. An event recorded into a stream completes when the stream reaches it, and `cudaEventElapsedTime` turns two completed events into a duration (CUDA Programming Guide 13.4.2 §2.5.3, printed p. 108 / PDF p. 126). The host can wait for one event while other streams keep working; the Guide's example overlaps CPU work with GPU kernels and copies this way (printed p. 110 / PDF p. 128).

**What prevents overlap.** Asynchronous copies need page-locked (pinned) host memory (printed p. 124 / PDF p. 142). Any command on the legacy NULL stream between two streams' commands serializes them, unless the streams were created non-blocking (§2.5.8, printed p. 115 / PDF p. 133). So a launch that forgets its stream argument silently serializes the pipeline. A device or page-locked allocation between commands also prevents concurrency (§3.1.3.3, printed p. 141 / PDF p. 159), so allocate both slots before the clock starts. The Guide's guidelines: issue independent work before dependent work, and delay synchronization as long as possible (printed p. 115 / PDF p. 133).

**Pipelining trades latency for throughput.** Pipelining increases throughput, the number of tasks completed per unit time, but it does not reduce, and usually slightly increases, the time of each individual task (H&P 6e Appendix C, printed p. C-8 / PDF p. 782; throughput defined in Patterson & Hennessy, PDF p. 53). With two slots, filling host buffers, copying in, computing, and copying out for different batches overlap like pipeline stages, and the slowest stage limits the gain (Patterson & Hennessy §4.5, printed p. 275 / PDF p. 298). Batching adds a second trade-off: larger batches amortize per-launch and per-copy overhead, but early items wait while the batch fills.

**Ownership is the correctness problem.** A slot's buffers belong to its in-flight batch until its event completes. Refill them earlier and the copy-in may still be reading them; read the results earlier and the copy-out may still be writing them.

## Implement

0. On Sol: `make SOLUTION=1 test-lab10`.
1. Read [serial.cu](../src/lab10/serial.cu): everything is allocated before the clock starts, batches run one at a time, and copies are synchronous from pageable memory.
2. Write `run_streamed` following its TODO: two slots (pinned input and output, device buffers, a stream, an event), *retire* before refill, fill, then enqueue the copy-in, two kernels, copy-out, and event in the slot's stream. Finally drain both slots and sort the batch records.
3. Run `make test-lab10`: 37 items with batch sizes 1, 3, 4, 7, 16, and 64 (partial final batches, many alternating slots, one batch larger than the input), a single item, 1024-thread blocks, and simulated arrivals. Results must match the serial baseline **bit for bit**, and the batches must cover every item exactly once. Then run `make sanitize-lab10`.
4. Look for overlap on a timeline:

```bash
nsys profile --trace=cuda,nvtx,osrt --output=runs/lab10-streams \
  ./build/stream_batches --mode streams --batch 16
```

In the CUDA HW rows, check whether one slot's copies overlap the other slot's kernels.

| Symptom | Likely cause |
| --- | --- |
| `run_streamed is not implemented yet` | Starter still throws |
| `first difference from the serial baseline in window w (item i)` | A slot was refilled before its event completed, or results were copied from the wrong slot or offset |
| `batches cover N of M items` | The final partial batch was dropped: drain every busy slot |
| Identical results, but no speedup and no overlap on the timeline | Kernels launched without the slot's stream, pageable host buffers, or an allocation inside the loop |
| `serial baseline: ...` errors | Your Lab 02 or Lab 03 kernels fail; make those labs pass first |

## Experiment

```bash
make bench-lab10      # batch 1, 4, 16, 64 x serial/streams, offline and with 0.05 ms arrivals; 256 items
```

1. **Offline:** how does `items_per_s` change with batch size in each mode? Where does batching stop helping?
2. **Live arrivals:** report the median and p95 latency against batch size. Explain the batch-fill waiting time, and find the configuration with the highest throughput that meets a stated latency target, for example p95 < 2 ms.
3. **Timeline:** does overlap occur? If it does not, explain why from the timeline. A result without overlap is valid when explained.
4. Describe a case where batching is inappropriate, such as a single camera at a low frame rate with a tight per-item deadline.

For offline runs, state that all inputs were already available; the latency columns then measure time from the start of processing, not response time to an arriving item.

## Portfolio deliverable

Publish a table showing batching's effect, a timeline excerpt if overlap occurs, and a recommended configuration for your workload, and explain a case where batching is inappropriate. Acceptance is correct ownership/completion plus a defensible measurement, not guaranteed overlap.

**Extension:** use three slots; make one stream wait for another with `cudaStreamWaitEvent`; try non-blocking streams and explain what changes; or capture a batch as a CUDA Graph to cut launch overhead. Continue to [the capstone](11-capstone.md).
