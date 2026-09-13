# DeepSeek V4.1 Flash: bounded streaming decode queue

## Problem and diagnosis

After removing the large-slab submission cliff, single-host SSD decoding still
waited at every layer, despite selected-expert ID readback already bounding
work on the same ordered Metal queue. A diagnostic profile with the companion
slab-residency fix held constant measured 81 versus 43 command buffers per token.
The mean of the last seven of eight tokens changed from 75.299 to 66.849 ms,
while GPU execution remained 41.379 versus 41.427 ms. Host wait time changed
from 58.847 to 51.324 ms. This profile explains the scheduling opportunity; it
is not the standalone performance measurement for this PR. The [per-token
profile CSV](v41_decode_queue_m2_ultra_profile.csv) records milliseconds;
`queued` denotes companion slab queue residency, not this PR's decode switch.

## Change

`DS4_METAL_ENABLE_V41_STREAM_DECODE_QUEUE=1` reuses the existing bounded queue
schedule for single-host, non-quality SSD decode. It keeps the drain after
layer 13 (before layer 14 overwrites shared Engram input), and at token completion.
Expert-ID readback and existing cache-replacement synchronization are preserved.
Resident, quality/imatrix and two-host TP eligibility remain unchanged.

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

For this independent comparison, `DS4_METAL_DISABLE_STREAMING_EXPERT_SLABS=1`
is present in both arms. Otherwise the unrelated large-slab latency dominates
on this host. Neither companion optimization is required or enabled.

[Raw model CSV](v41_decode_queue_m2_ultra.csv).

| Run | Variant | Prefill t/s | Decode t/s (512 tokens) | First token ms | Steady t/s (511 tokens) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | control | 68.30 | 10.28 | 382.664 | 10.34 |
| 2 | candidate | 66.55 | 11.36 | 362.478 | 11.43 |
| 3 | candidate | 68.63 | 11.41 | 370.835 | 11.49 |
| 4 | control | 68.96 | 10.28 | 404.379 | 10.35 |

Mean 512-token decode: 10.280 → 11.385 t/s (+10.7%).
All four 512-token decoded outputs are identical. These are two observations
per variant on one host, not release medians or a cross-device estimate.

Reproduce each arm with the candidate flag absent or set to `1`:

```sh
DS4_METAL_DISABLE_STREAMING_EXPERT_SLABS=1 DS4_METAL_ENABLE_V41_STREAM_DECODE_QUEUE=1 \
./ds4-bench -m MODEL --ssd-streaming \
  --prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 2048 \
  --ctx-alloc 8257 --gen-tokens 512 --show-output --csv /tmp/candidate.csv
```

Balanced same-engine harness result:

```text
variant=control first_split=2 second_split=32 tokens=512 seconds=42.790230 tokens_per_second=11.9653
variant=candidate first_split=2 second_split=32 tokens=512 seconds=38.415040 tokens_per_second=13.3281
exact_rows=529 exact_floats=68389120 exact_selected_ids=528 vocab=129280
```

Command: the SSD streaming example in [README](README.md#metal-decode-schedule-ab).

## Validation

- Clean default Metal build, CPU compile, and restored Metal executable links.
- `MTL_DEBUG_LAYER=1 tests/test_deepseek41_graph MODEL --stream-decode-queue-parity`:
  two sessions share a 512-expert cache, forcing eviction. After restoring a
  120-token prefix, positions 121–144 compare every full-vocabulary float,
  finite logits, Engram history and all saved cache spans bit-for-bit.
- `MTL_DEBUG_LAYER=1 make test-deepseek41-metal`.
- `make test-frontends test-engram test-deepseek41-gguf test-quality-api`.
- Balanced same-engine decode harness, alternating variant order and assignment
  to sessions, with complete-logit and selected-token equality checks.

The SDK 15 build retains two upstream unused Metal 4 symbol warnings. Full
legacy `make test` was not run against mismatched older Flash vectors. These
focused V4.1 results do not establish release, multi-host or other-backend QA.
