# DeepSeek V4.1 Flash: parallel uncached Engram decode reads

## Problem and diagnosis

One token reads 24 native Engram rows at each of two layers. Each row is only
264 bytes, but the current uncached reader serializes all 48 reads. A matched
microbenchmark showed that sorting/deduplicating through the prefill batch
reader alone did not improve latency (about 4.81 ms for both tables). Four
readers reduced it to about 1.37 ms whether rows were sorted or left in original
order. This change uses the smaller original-order implementation.

## Change

On macOS, `DS4_ENGRAM_PARALLEL_DECODE=1` partitions a 24-row read into four
six-row calls to the existing serial reader, using `dispatch_apply_f` to join
before return. Inputs are validated before dispatch. Each worker writes a
disjoint output range and records its own error; the caller propagates an error
after every worker has joined. It adds no row cache, sorting, heap allocation,
new descriptor, or change to the batched prefill reader. Other platforms and
row counts use the existing path. As with the serial reader, output can be
partially written after an IO/data error and callers must honor failure.

## Environment and results

Measured on 2026-09-13, Apple M2 Ultra, 192 GiB unified memory, macOS 15.7.4,
Metal, against upstream `bd66c402070042bf0a79ad6ece8242de4c93680c`.
Model: DeepSeek V4.1 Flash calibrated IQ2_XXS/Q2_K, 365,713,686,528 bytes;
SHA-256 `1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42`.
The model was reused without conversion. The machine's existing
`iogpu.wired_limit_mb=188000` was unchanged. One inference/Metal benchmark ran
at a time; ordinary desktop background services remained running.

All model benchmarks use `speed-bench/promessi_sposi.txt`, SSD streaming,
a 2,048-token prefill, 8,257 allocated context, default power and automatic
expert-cache sizing. The planner reports 135.26 GiB dynamic expert cache plus
7.12 GiB prefill headroom. Each process starts a fresh engine/cache; the OS/file
cache is not flushed. Order is control, candidate, candidate, control. Controls
use the same branch binary with the opt-in flag absent. The production path
with the flag absent is unchanged from the upstream base.

For independent model measurements, `DS4_METAL_DISABLE_STREAMING_EXPERT_SLABS=1`
is present in both arms. The queue optimization is absent.

[Raw model CSV](v41_engram_m2_ultra.csv).

| Run | Variant | Prefill t/s | Decode t/s (512 tokens) | First token ms | Steady t/s (511 tokens) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | control | 68.60 | 10.27 | 389.896 | 10.34 |
| 2 | candidate | 68.68 | 10.80 | 402.655 | 10.87 |
| 3 | candidate | 68.69 | 10.80 | 368.530 | 10.86 |
| 4 | control | 68.80 | 10.32 | 385.098 | 10.38 |

Mean 512-token decode: 10.295 → 10.800 t/s (+4.9%).
All four 512-token decoded outputs are identical. These are two observations
per variant on one host, not release medians or a cross-device estimate.

Reproduce each arm with the candidate flag absent or set to `1`:

```sh
DS4_METAL_DISABLE_STREAMING_EXPERT_SLABS=1 DS4_ENGRAM_PARALLEL_DECODE=1 \
./ds4-bench -m MODEL --ssd-streaming \
  --prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 2048 \
  --ctx-alloc 8257 --gen-tokens 512 --show-output --csv /tmp/candidate.csv
```

The real-row microbenchmark (excluding pass 0) measures 4.787 ms serial
and 1.359 ms parallel per token for both tables. [Raw CSV](v41_engram_m2_ultra_rows.csv).

```sh
./speed-bench/engram_decode_bench MODEL 162955640832 384006168 264333279232 384016682
```

Those offsets are valid only for the exact GGUF identified above.

The checked-in row benchmark uses 128 deterministic 24-row sets per table and
eight alternating passes. It reads through the real Engram table API and checks
every output bit. This workload measures scattered uncached reads, not every
possible token distribution, storage device or shared-host load.

## Validation

- Clean default Metal build, CPU compile, and restored Metal executable links.
- `make test-engram`: serial and parallel modes, row order, duplicate IDs,
  signed zero, output guards, invalid final row rejected before writing,
  EDOM in each of four partitions, EIO on truncated input, and existing
  hash/history, batched-reader and numeric checks.
- `MTL_DEBUG_LAYER=1 DS4_ENGRAM_PARALLEL_DECODE=1
  tests/test_deepseek41_graph MODEL --session-fixture`: real-model session,
  snapshot/restore, cancellation and bounds checks.
- `make test-frontends test-deepseek41-gguf test-quality-api`.

The SDK 15 build retains two upstream unused Metal 4 symbol warnings. Full
legacy `make test` was not run against mismatched older Flash vectors. Other
platforms were not executed; CPU portability was compile-checked only.
