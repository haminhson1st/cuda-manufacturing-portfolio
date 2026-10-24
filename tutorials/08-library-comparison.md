# 08 — Choose between a custom kernel and a library

**Time:** 6–8 hours. **Business task:** simplify a frequently used aggregation without losing correctness or performance.

| You edit | Provided | Reference solution |
| --- | --- | --- |
| [src/lab08/reductions.cu](../src/lab08/reductions.cu) | Harness [main.cu](../src/lab08/main.cu) (`reduce_compare`), contract [lab08.hpp](../src/lab08/lab08.hpp), tests in the [Makefile](../Makefile) | [solutions/lab08/reductions.cu](../solutions/lab08/reductions.cu) |

## The operation

Count the nonzero bytes of a packed 8-bit mask: the aggregation behind Lab 07's score. The input is `n` bytes already on the device; the output is a 64-bit count. Every backend must use the same semantics: "nonzero", not "equal to 255". The `values` test pattern holds arbitrary nonzero bytes to catch shortcuts such as summing the bytes and dividing by 255. Choose the accumulator from the number of items, not the input type: a byte input does not make the sum fit in a byte or a 16-bit integer.

| Backend | Implementation | Temporary storage |
| --- | --- | --- |
| `custom` | Your kernel: grid-stride loop, block reduction, one `atomicAdd` per block | None beyond the 8-byte result |
| `thrust` | `thrust::count_if(thrust::device, first, last, predicate)` | Managed inside Thrust, not reported |
| `cub` | `cub::DeviceReduce::Sum` over a transform iterator, in two phases | Queried and allocated by you, once |

## Concepts

**Prefer a library unless you can show why not.** The CUDA Programming Guide advises that when a library already implements an algorithm, using it is often more productive and performant than reimplementing it, because libraries are tuned for each GPU architecture (§1.1.3, printed p. 4 / PDF p. 22). Thrust and CUB are part of the [CUDA Core Compute Libraries (CCCL)](https://github.com/NVIDIA/cccl), which are not in the course corpus. Use the API documentation that matches the version the harness prints in `thrust_version`.

**Temporary storage is a design decision.** CUB's device algorithms run in two phases. A call with a null storage pointer only reports the bytes needed; a second call with real storage does the work. You therefore own the allocation and can keep it out of the timed loop: the harness reports it as `temp_bytes` and `prepare_ms`. The corollary: pass a null pointer by mistake and CUB just answers the size question, without counting anything. Thrust hides its temporary storage and may allocate on every call. The Guide's stream-ordered allocator exists because `cudaMalloc`/`cudaFree` can be costly and synchronizing (§4.3.2, printed p. 261 / PDF p. 279). Measure it rather than assume.

**The first call is different.** Since CUDA 12.2, kernels are loaded lazily, when first needed (Guide §4.8, printed pp. 309–310 / PDF pp. 327–328). So the first call of each backend pays to load its kernels. `CUDA_MODULE_LOADING=EAGER` moves that cost to program start (Guide printed p. 513 / PDF p. 531). The harness creates the CUDA context before any timing and reports `first_ms` separately from the steady-state median `steady_ms`. Every timing boundary is the same: the mask is already on the device, and the count is available on the host.

**Exactness.** All three backends add integers, so they must agree with the CPU bit for bit, whatever order each one combines partial results in. Floating-point sums would not, as Lab 03 showed; see the extension.

## Implement

0. On Sol: `make SOLUTION=1 test-lab08`.
1. **Custom.** Adapt your Lab 07 count kernel to a packed 1D array: a grid-stride loop with a per-thread count, a shared-memory block reduction, and one `atomicAdd`. In `count_custom`, zero the counter with `cudaMemsetAsync` and choose a modest grid (a few blocks per SM).
2. **Thrust.** One call to `thrust::count_if`, plus a predicate functor whose `operator()` is `__host__ __device__` and `const`.
3. **CUB.** `cub_temp_bytes` performs the size query, and `count_cub` performs the reduction. Feed CUB `thrust::make_transform_iterator(d_mask, functor)`, which yields 1 for nonzero bytes and 0 otherwise without copying anything, and pass an `unsigned long long*` output so CUB accumulates in 64 bits.
4. `make test-lab08` (every backend × six patterns × edge sizes, plus a 100 MB case), then `make sanitize-lab08`.

Run one backend at a time while you work:

```bash
./build/reduce_compare --backend thrust --pattern values --bytes 1000003 --reps 3
```

| Symptom | Likely cause |
| --- | --- |
| `... is not implemented yet` | That starter function still throws |
| Custom is correct on the first call and wrong afterwards | The counter is not zeroed on every call (the harness pre-fills it with `0xFF` bytes) |
| Wrong only for `--pattern values` | Bytes were summed instead of counting nonzero bytes |
| CUB returns quickly with a wrong count | A null storage pointer was passed in phase 2, so CUB only computed the size |
| Compile errors about calling a host function | The functor's `operator()` is missing `__host__ __device__` |

## Experiment

```bash
make bench-lab08      # 1e3, 1e5, 1e7, 1e9 bytes x three backends, one process each
```

1. For small inputs, which backend has the lowest `steady_ms`? Explain the result with per-call costs: launches, allocation, and synchronization.
2. For large inputs, compare `gbps` with your GPU's memory bandwidth and with Lab 02's `kernel_gbps`.
3. Compare `first_ms − steady_ms` across backends. When would that startup cost matter to the inspection application?
4. Compare maintenance effort: lines of bespoke code, synchronization reasoning, temporary-storage management, and numerical behavior.

State which implementation you would keep, and why. A library win is a successful engineering outcome. A custom win for one narrow shape is evidence only for that tested shape and environment.

## Deliverable

An experiment report built from the [template](../templates/experiment.md), with all three implementations, the result table, and a decision. Record the toolkit (`nvcc --version`) and the `thrust_version` the harness prints in the manifest. Preserve upstream attribution if you adapt library examples.

**Extension:** repeat the comparison for the floating-point sum of squares of a Lab 03 signal (`thrust::transform_reduce` against your kernel). The backends combine terms in different orders, so expect last-bit differences, and validate with the Lab 03 error bound instead of equality. A cuFFT feature for sensor analysis can come later; do not add an FFT merely to lengthen the list of technologies. Continue to [profiling](09-profiling.md).
