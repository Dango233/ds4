# Fork features

This main branch combines upstream DwarfStar with the features below. Each
feature is opt-in. Other experimental branches are maintained separately.

## Raw completions

Start `ds4-server --raw-completions` to treat the `prompt` in
`POST /v1/completions` as an already-rendered model prompt. The server continues
that prompt directly, retaining generated thinking/control tokens rather than
using them as no-thinking stop markers. The client supplies the model's prompt
format. Normal model stop tokens and requested text stops still apply.

Without the flag, completions use upstream's model-specific prompt rendering.
The flag only selects raw mode for `/v1/completions`.

## DeepSeek V4.1 Flash on Metal

```sh
DS4_METAL_STREAMING_SLAB_RESIDENCY=1 \
DS4_METAL_ENABLE_V41_STREAM_DECODE_QUEUE=1 \
DS4_ENGRAM_PARALLEL_DECODE=1 \
./ds4-server -m ds4flash.gguf --ssd-streaming --raw-completions
```

- Slab residency attaches owned expert-cache slabs to the Metal queue on macOS
  15+. Memory-pressure relief releases the set until the cache is rebuilt.
- The decode queue reduces layer waits in single-host SSD decode while keeping
  the synchronization before shared Engram input reuse and token completion.
- Engram decode reads use four joined readers on macOS.

These are presence switches: unset a variable to disable it; setting it to `0`
still enables it. All three default to off.

Each optimization includes its own tests and measurements:

- [Slab residency](../speed-bench/v41_slab_residency_m2_ultra.md)
- [Decode queue](../speed-bench/v41_decode_queue_m2_ultra.md)
- [Engram reads](../speed-bench/v41_engram_m2_ultra.md)

The independent optimization measurements use an M2 Ultra with 192 GiB,
macOS 15.7.4 and the calibrated DeepSeek V4.1 Flash Q2 GGUF. Cross-device
behavior and full release QA require separate validation.

## Consolidation checks (2026-09-13)

The consolidation preserves fork/main history and includes upstream
`bd66c402070042bf0a79ad6ece8242de4c93680c` and the three optimization PR commits.

Validation on the M2 Ultra:

- Metal and CPU compilation, followed by fresh Metal executable links.
- Frontend, Engram, V4.1 GGUF and quality-tool unit tests.
- Real-model `/v1/completions` requests in raw and default modes, each with JSON
  and SSE output, with and without `ignore_eos`. Trace checks verify that raw
  prompts are unchanged and default prompts use the model template.
- A 2,048-token prefill and 512-token decode with all three optimization flags
  enabled: 75.55 prefill t/s, 16.81 full decode t/s, 16.93 t/s after the first
  token. The generated output matches the previously tested combined branch.
  [Raw CSV](../speed-bench/fork_main_m2_ultra.csv).

The performance check uses `speed-bench/promessi_sposi.txt`, `--ctx-alloc 8257`,
SSD streaming and the same calibrated V4.1 Q2 model recorded in the individual
reports. This is one consolidation run. The SDK 15 build retains the two
upstream unused Metal 4 symbol warnings.

## Upstream alignment (2026-09-14)

This branch now includes upstream
`a04f46fa423e45712c8c7e430eff422479f314a3` (DeepSeek V4.1 CUDA support),
the updated three optimization branches, and the existing raw completion mode.
The single-host decode queue opt-in is explicitly macOS-only. Upstream's new
non-macOS batched Engram reader and its error tests are preserved.

Revalidation on the same M2 Ultra and exact Q2 model:

- Metal and CPU builds, with freshly restored Metal executable links.
- Frontend, Engram, V4.1 GGUF and quality-tool unit tests.
- Real-model raw/default `/v1/completions`: JSON and SSE, with and without
  `ignore_eos`, including trace verification of prompt rendering.
- All three optimization flags enabled: 77.49 prefill t/s, 16.92 full decode
  t/s across 512 tokens, 17.07 t/s after the first token. Output matches all
  eight independent queue/Engram runs from this revalidation.
  [Raw CSV](../speed-bench/fork_main_m2_ultra_20260914.csv).

This is one combined performance run with the same 2,048-token prompt,
8,257 allocated context, SSD streaming and automatic expert cache.
Independent ABBA comparisons and parity evidence are in each optimization report.

The complete Metal kernel suite is not green: the new router check fails at
`tests/test_deepseek41_metal.c:105`, with exactly the same values on pristine
upstream and all three PR branches. Seven targeted kernel subtests and the
optimization-specific checks pass. No kernel or tolerance was changed.
CUDA/ROCm hardware and full legacy `make test` were not run; the two existing
SDK 15 unused Metal 4 symbol warnings remain.
