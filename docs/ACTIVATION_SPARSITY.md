# Dense-FFN activation sparsity: measured negative

The question: could Muse Glimmer 30B (or any dense checkpoint here) run in
less memory through contextual activation sparsity, the Deja Vu /
PowerInfer / LLM-in-a-Flash family of techniques, where only the FFN
neurons a token actually activates are kept in memory, hot rows cached and
cold rows streamed, the dense analog of this engine's expert cache?

The answer, measured 2026-08-16 on the real install, is no: the activation
mass is spread across ~90% of the FFN, there is no hot set, and the
temporal locality a neuron cache would need does not exist. This is the
same shape of result as `docs/EXPERT_ROUTING.md`'s (routing is spread, not
domain-concentrated), one architecture class over. If you are about to
propose weight streaming or pruning keyed on what a token "actually
uses", read both pages first.

## The instrument

`MFERENCE_FFN_HIST=/path.json` on `turbospark-check` (museGlimmer family
only; the flag names the only flow that feeds it) captures the post-SiLU
FFN activation vector `silu(gate) * up` per layer per decode token, and
`scripts/ffn_sparsity.py` analyzes the capture. Design points, in
`crates/runtime/src/ffn_hist.rs`:

- **Zero extra dispatches and no math change.** `silu_mul`'s output is
  redirected into a per-layer region of a `52 x 19,968` capture buffer
  (the shared `ffn_act` scratch is overwritten 52 times per token, so
  per-layer data cannot survive to the readback otherwise) and `down_proj`
  reads from the same region. Verified: stdout md5 `98e99a1f...` identical
  with the flag on and off, and the capture itself is byte-identical
  across two processes (`d2488707...`).
- Prefill passes are counted but not read back (prefill routes
  differently; the `router_hist` lesson).
- The capture costs ~30% of decode throughput (a 2 MB readback plus a
  host top-k per token). Diagnostic-grade, never on by default.
- Sanity: per layer, `sum(hist_count) == decode_passes * inter`, asserted
  by the script before it reports anything. The per-layer rows also
  differ from each other (layer 0 n50 = 6,982 against layer 51's 4,655),
  which is what says the capture reads 52 real regions rather than one
  region 52 times.

## The numbers

Two greedy prompts (a prose question, a code task), 400 decode passes,
`~/models/museglimmer-30b.gturbo`, deterministic to the byte:

| statistic (mean over 52 layers) | value | of 19,968 |
| --- | ---: | ---: |
| neurons covering 50% of \|act\| mass | 6,986 | 35% |
| neurons covering 90% | 16,650 | 83% |
| neurons covering 95% (n95) | 18,186 | **91.1%** |
| neurons covering 99% | 19,561 | 98% |
| activation mass below \|a\| < 0.01 | 1.7% | |
| activation mass below \|a\| < 0.1 | 35.7% | |

Temporal locality, the number that decides whether a cache can work
(consecutive tokens' top-K neuron sets, overlap fraction, against the
chance level for sets that size):

| K | overlap | chance | expert cache, for contrast |
| ---: | ---: | ---: | --- |
| 512 | 0.240 | 0.026 | ~0.84 hit rate at 16 slots |
| 1024 | 0.247 | 0.051 | |
| 2048 | 0.277 | 0.103 | |
| 4096 | 0.349 | 0.205 | |

Both prompts agree to within a point (n95 90.0% prose, 90.6% code), so
this is not a domain artifact.

## The reading

1. **There is no hot set.** 95% of the activation mass needs 91% of the
   neurons. Even a perfect n95 working set would keep 9.9 of the FFN's
   10.9 GB resident, a ~1 GB saving on a 15.7 GB install, before any
   quality cost and before building a row-granular streamer.
2. **The locality is real and useless.** Overlap runs ~9x chance at
   K = 512, so consecutive tokens do activate alike -- but at 0.24 against
   the ~0.85 the expert cache lives on, a neuron cache would miss three
   quarters of its set every token. Streaming those misses is the whole
   FFN again.
3. **Thresholding (CATS-style) has no operating point.** Zeroing
   everything under \|a\| < 0.01 drops only 1.7% of mass but (per the
   coverage curve) removes few weights; under 0.1 it drops 35.7% of the
   mass, which is not a sparsification, it is a different model.
4. This is the expected SiLU result. The techniques above were built on
   ReLU models with 90-97% natural sparsity; recovering sparsity on a
   SiLU model takes ReLUfication finetuning, which is training work on the
   checkpoint, not engineering work on this engine.

## What this closes, and what remains

**Closed:** PowerInfer / LLM-in-a-Flash-style neuron streaming, neuron
caches, activation-threshold FFN skipping, and any "load only what the
token uses" scheme for this checkpoint. Do not re-derive this without a
census reading that contradicts this one; the instrument is standing and a
run costs ~a minute per prompt.

Muse Glimmer's memory and power cost is what dense 30B costs. The levers
that actually exist are recorded elsewhere: `--power-profile efficiency`
(16 W, never throttles, `docs/POWER_BASELINE.md`) and, if memory ever
binds harder than power, a sub-4-bit GGUF intake for the family
(`unsloth/Muse-Glimmer-30B-GGUF` publishes IQ2/IQ3 builds; needs an
`arch_registry.rs` row and a name table first, and note the IQ path costs
~2x J/token on Gemma).

The capture generalizes to the other dense families for the price of the
same redirect in their flows (`families/llama/dense.rs`,
`families/qwen/dense.rs`); none has been measured, and this result does
not automatically transfer. Every one of them is SiLU too, so the prior
is the same.
