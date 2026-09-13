# DeepSeek V4.1 Flash: owned Metal slab residency

## Problem and diagnosis

DeepSeek V4.1 Flash SSD decoding on this host fell to approximately 0.26 tokens/s
with the default owned expert-cache slabs. GPU execution itself did not explain
the multi-second token latency. Disabling slabs removed the severe cliff.
Registering allocations in an unattached residency set did not help; attaching
the set to the existing command queue did. Additional CPU `mlock` and warmup
alone did not remove the delay.

The model-free benchmark in this change allocates shared Metal buffers, writes
and locks them on the CPU, then reads small, dispersed samples through GPU
addresses with `useResource`. It performs no model-file IO. With a 104 GiB pool,
the initial queue-detached phase spent about 250 ms per large-buffer submission,
of which approximately 249.7 ms was in Metal's reported kernel/driver interval;
GPU execution was approximately 0.013 ms. Queue attachment reduced submission
wall time to approximately 0.180 ms. The 32 GiB control did not exhibit the cliff.
These per-phase means exclude the first four iterations of each phase.
These observations locate the delay in large-allocation submission/residency
handling, without claiming a particular private driver implementation or a
hardware-wide defect.

The queue-toggle run retains the residency set while detached. After its first
attachment, phase 2 remains fast: detachment alone is not an eviction command.
Do not interpret it as a reversible OFF/ON/OFF/ON performance experiment.
Fresh-process model comparisons below supply independent controls.

## Change

`DS4_METAL_STREAMING_SLAB_RESIDENCY=1` registers only owned expert-cache slab
buffers and attaches the set once to the existing queue on macOS 15+. It adds
no explicit `requestResidency`/`endResidency` pair. Cache teardown removes and
releases the set with the physical pool. Successful mlock-margin relief also
removes/releases it, preventing later slab allocation from reattaching until
the cache is rebuilt. Model mappings and disk-only Engram tables are excluded.
The feature remains opt-in pending evidence on other hosts and memory budgets.

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

[Raw model CSV](v41_slab_residency_m2_ultra.csv); model-free [104 GiB](v41_slab_residency_m2_ultra_repro_26.csv) and [32 GiB](v41_slab_residency_m2_ultra_repro_8.csv) CSVs.

| Run | Variant | Prefill t/s | Decode t/s (8 tokens) | First token ms | Steady t/s (7 tokens) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | control | 76.15 | 0.25 | 4103.550 | 0.25 |
| 2 | candidate | 70.95 | 9.57 | 329.048 | 13.83 |
| 3 | candidate | 69.16 | 9.67 | 316.315 | 13.71 |
| 4 | control | 60.11 | 0.25 | 3935.309 | 0.25 |

All four decoded outputs are identical. Eight-token decode intentionally keeps
the multi-second control bounded; its first-token cost materially affects the
aggregate. This is not a 512-token sustained-throughput claim. Prefill varies
from 60.11 to 76.15 t/s across controls and 69.16 to 70.95 t/s across candidates;
these two pairs do not establish a precise prefill speedup.

Reproduce each arm with the flag absent or set to `1`:

```sh
DS4_METAL_STREAMING_SLAB_RESIDENCY=1 ./ds4-bench -m MODEL --ssd-streaming \
  --prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 2048 \
  --ctx-alloc 8257 --gen-tokens 8 --show-output --csv /tmp/slab-on.csv
```

## Validation

- Clean default Metal build, CPU compile, and restored Metal executable links.
- `make test-metal-slab-residency` with Metal API validation: protected expert
  survives relief; an unprotected slot is unlocked/evicted; the queue releases
  the set; allocation cannot reattach after relief; rebuilding reenables it;
  weak references confirm the set and old physical buffers are destroyed.
- `make test-metal-ssd-experts test-metal-moe-prefill`, plus SSD
  `--table-admission` and MoE `--ssd-address`, under Metal API validation.
- With the residency flag enabled and Metal API validation:
  `tests/test_deepseek41_graph MODEL --session-fixture` (real weights, snapshots,
  restored logits, prefix reuse, cancellation, boundary and malformed-state checks).
- `make test-frontends test-engram test-deepseek41-gguf test-quality-api`.
- Both 104 GiB and 32 GiB model-free runs check every GPU checksum.

The macOS SDK 15 build reports the same two pre-existing unused Metal 4 symbol
warnings as the upstream base. Full legacy `make test` was not run: its default
model vectors target the older Flash checkpoint; the V4.1 checks above were
used. This is single-host Metal evidence, not a release sign-off or validation
of CUDA, ROCm, RDMA, other Macs, or all memory-pressure conditions.
