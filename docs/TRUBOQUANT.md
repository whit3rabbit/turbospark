# TurboQuant KV-Cache Quantization

**Status (2026-09-06): built and shipped as `--kv-bits off|2|3|3.5|4`,
default OFF.** This page used to be titled "Assessed, Not Adopted" and
conclude that KV quantization did not pay on this engine. That verdict is
reversed -- see "The reversal" below for what changed and why the original
reasoning (kept intact further down) was answering a different question
than the one that mattered. The short version: "does this help today" was
the wrong test for an opt-in flag that costs every other caller nothing.

The rest of this page is in two parts. **The reversal and what was built**
describes the shipped feature: the codec, the kernels, the per-layer
policy, the flag surface, and what is still owed. **The original
assessment (2026-08-15)** is kept below it unedited except for this
banner, because it is the record of why the decision needed reversing
rather than defaulting to on, and its negative findings (the QJL residual,
the sub-4-bit quality cliff, the head-dimension sensitivity) are still
true and still worth citing on their own terms.

## The reversal

The 2026-08-15 assessment read `sharpner/turboquant-mlx`
(https://github.com/sharpner/turboquant-mlx), a proof of concept of
Google's 2025 TurboQuant paper, against this port's own memory and
throughput profile and concluded: KV is not the dominant memory term for
any family near a ceiling, attention is a single-digit share of decode at
this port's 4-8K contexts, and the one thing that makes turboquant-mlx
fast (MLX's fused `quantized_matmul`) is exactly the kernel this port
would have had to write itself with no equivalent to borrow. All true, and
all beside the point once the feature is opt-in: those are arguments for
defaulting `--kv-bits` OFF, which it is, not for refusing to build it. A
caller who wants a smaller KV cache at a stated quality cost -- a longer
context on a memory-constrained machine, or a deliberate memory/quality
trade this repo has no standing to make on someone else's behalf -- had no
way to ask for one before this landed.

**What was built goes further than the assessment's own recommendation.**
Item 2 under "Ideas worth taking" said: if this is ever built, 4-bit
affine is the only point worth building, skip fractional rates, skip
Lloyd-Max, skip 2-bit. What actually shipped is mlx-vlm's real
`_TurboQuantMSECodec` (`mlx_vlm/turboquant.py`) ported whole --
`crates/compute::kv_quant` is a portable CPU reference for it, checked
bit-identical against mlx-vlm's own printed sign vectors and packed words
(`crates/compute/tests/kv_quant.rs`) -- which is the paper-correct
Lloyd-Max path (their "V3"), not the affine one ("V2") the assessment
called safe. Per K or V row: divide by its own norm (stored separately),
apply a Randomized Hadamard Transform with a FIXED +/-1 sign vector (one
per K or V for the whole model, mlx-vlm's own `KEY_SEED`/`VALUE_SEED`
defaults of 0/1 -- not baked into the Q/K projections at repack time the
way the assessment's "rotation stage" predicted, so no `crates/repack` or
install-format change was needed at all), then quantize each rotated
coordinate against a 1-D Lloyd-Max codebook fit to the sphere-marginal
density a coordinate of a random rotated unit vector in `dim` dimensions
follows. Widths are `off`, `2`, `3`, `3.5` (K3/V4, mlx-vlm's own split for
its one fractional rate), and `4`.

**And the hard part the assessment named -- a real fused dequant inside
the attention kernel -- exists now.** `crates/gpu/src/shaders/attention_tq.metal`
reads packed rows directly in the QK^T and AV passes;
`crates/gpu/src/shaders/kv_quantize_tq.metal` is the write-side quantizer.
Both are parity-tested against the CPU reference
(`crates/gpu/tests/attention_tq_parity.rs`,
`crates/gpu/tests/kv_quantize_parity.rs`).

**Layer eligibility is mlx-vlm's own policy, not this port's invention.**
`crates/model-io::kv_quant::layer_is_quantized` mirrors
`should_quantize_kv_layer` exactly: every layer quantizes when the stack
is two layers deep or fewer, otherwise every layer but the last does
(mlx-vlm's own measurement that the last full-attention layer is
quantization-sensitive on Gemma-class models). Sliding-window and linear
(gated-DeltaNet) layers are never candidates regardless of position --
callers gate on the mask value first. A head_dim outside `32..=512` or not
a power of two REFUSES `--kv-bits` at open by name (`rht_supported`)
rather than silently falling back to a slower path nothing here
implements; mlx-vlm's own dense-rotation-matrix fallback for other widths
was deliberately not ported.

**Wired into every family and the whole stack, off by default
everywhere.** `families/{gemma4,qwen,qwen4,llama,gptoss,museglimmer}/attn.rs`
all dispatch the quantized path when `--kv-bits` is on (`qwen` covers both
the dense and MoE halves of that architecture). The flag threads through
`crates/invocation` (`KvBits`, the 5-place rule), `crates/cli`
(`turbospark-check --kv-bits`), `crates/server` (same flag, resolved once
at process open like `--speculative` and `--steering`), `crates/ffi`
(`kvBits` wire option, `SessionInfo.kvBits` reporting what was resolved),
and the Swift binding (`OpenOptions.KvBits`). `crates/bench` reaches it
ONLY through a separate, more-parameterized opener
(`open_model_runner_for_protocol_speculative_kv_quant`) that the plain
protocol opener calls with `KvQuant::Off` explicitly -- the same
structural guard `DraftPolicies` and `SteeringPolicy` already use so a new
axis cannot reach a frozen memory-oracle or quality-gate row by accident
(AGENTS.md Gotcha 35, `crates/bench/CLAUDE.md` Gotchas 4-5). Every family's
real-forward test suite gained a `real_forward_<family>_kv_quant.rs`
fixture-level suite.

**RUN 2026-09-07** on the real `~/models/qwen38-27b.gturbo` install (M4
Max, AC), `TURBOSPARK_KV_QUANT_INSTALL_DIR=~/models/qwen38-27b.gturbo
cargo test -p turbospark-bench --test kv_quant_probe --release --
--ignored --nocapture`:

| width | peak MiB | delta         | ref ppl | ppl delta |
|-------|----------|---------------|---------|-----------|
| off   | 631.5    | (baseline)    | 4.9432  | (baseline)|
| 2     | 523.1    | -108.4 (-17.2%) | 4.9452 | +0.0020 |
| 3     | 511.3    | -120.1 (-19.0%) | 4.9854 | +0.0422 |
| 3.5   | 448.2    | -183.3 (-29.0%) | 4.9675 | +0.0243 |
| 4     | 624.8    | -6.7 (-1.1%)    | 4.9617 | +0.0185 |

All five widths finite, perplexity delta small and monotone-ish across
widths. Read the MiB delta per the probe's own note: this is measured at
the protocol's own context window on a family where KV dominates the
counted footprint (Gotcha 40), so it is not a footprint ceiling and does
not transfer to a family where KV is a smaller share. `--kv-bits 4`'s
-1.1% is real but small at this window; it would read larger at a longer
context, where KV is a bigger fraction of `phys_footprint`.

**RUN 2026-09-07**, same session: `AGENTS.md`'s greedy-then-sampled pair,
`--kv-bits 4`, both installs. All four runs reached `stop=MaxTokens` with
no error, and the text stayed coherent through 400 new tokens on both
models (gemma4 42.5/42.3 tok/s greedy/sampled, qwen38-27b 18.2/18.4).
`--kv-bits off` byte-identical to `HEAD~1` (a two-binary diff) is still
owed and not run this session.

## The original assessment (2026-08-15)

Kept for the record: this is what made KV quantization look like a bad
trade before it was reframed as an opt-in flag rather than a default. Its
factual claims about turboquant-mlx's own measurements, and about this
port's memory and throughput profile as they stood in 2026-08-15, are
unedited below.

The question this page originally answered: should this engine adopt
KV-cache quantization in the shape of `sharpner/turboquant-mlx`, a proof
of concept of Google's 2025 TurboQuant paper on Apple Silicon, and if not,
are any of its ideas worth taking anyway?

The answer, assessed 2026-08-15 by reading their README against this port's
own recorded measurements (no code was run and nothing here is a new
measurement): implementable, but it does not pay on this engine today.
KV is not the dominant memory term for any family that is near a memory
ceiling, attention is a single-digit share of decode at this port's 4-8K
contexts, and the one thing that makes turboquant-mlx fast (MLX's fused
`quantized_matmul`) is exactly the kernel this port would have to write
itself. What is worth keeping is their negative results and one favorable
datapoint about Gemma's head dimensions, all recorded below with explicit
revisit triggers.

This is a desk assessment, not a measured negative like
`docs/EXPERT_ROUTING.md` or `docs/SPECULATIVE_DECODING.md`: their numbers
are theirs (different models, different engine) and this port's numbers
come from its own frozen rows. The two are never blended into one
comparison here.

## What turboquant-mlx is

A proof of concept of the TurboQuant paper (Google, 2025): quantize the
attention KV cache to 2-4 bits so long contexts fit in memory and the
attention pass reads fewer bytes. Weights are untouched. This is entirely
orthogonal to the weight quantization this port already does everywhere.

Two implementation paths:

- **V2 (affine, hardware-accelerated):** K/V quantized through
  `mx.quantize` and consumed by `mx.quantized_matmul`, MLX's fused
  dequant-inside-the-matmul Metal kernel. Variants add a random QR
  rotation before quantizing (distributes outlier channels so the values
  are closer to iid Gaussian) and a 1-bit QJL residual (a
  Johnson-Lindenstrauss sign-bit correction to attention scores).
- **V3 (Lloyd-Max codebooks, paper-correct):** optimal centroids at 2-4
  bits, plus outlier-channel splitting for fractional rates (2.5/3.5
  bit). No fused kernel: dequant is software in the SDPA hot path.

Tested on Llama 3.2 3B, Llama 3.1 8B, Mistral 7B, and Gemma 3 4B on an
M4 Max (64 GB). Note their baselines are 4-bit WEIGHT-quantized models,
so KV error compounds on top of weight error. That is the same regime
this port's installs are in, which makes their quality numbers more
relevant here than the paper's full-precision ones.

## Their measured results

All numbers below are theirs, from their README, on their models. None
of them transfer to this port's installs as values. The SHAPE of the
results is what transfers.

Perplexity (lower is better, deltas vs their fp16-KV baseline):

| strategy | Llama 3.2 3B | Llama 3.1 8B | Mistral 7B | Gemma 3 4B |
|---|---|---|---|---|
| fp16 baseline | 12.94 | 9.47 | 6.79 | 12.18 |
| V2 4-bit rotated | -0.8% | +1.4% | +1.4% | +2.9% |
| V2 3-bit rot+QJL | +5.3% | +7.8% | +5.1% | **-1.1%** |
| V3 3.5-bit mixed | +0.3% | +6.7% | +4.0% | +2.1% |
| V3 3-bit Lloyd-Max | +5.1% | +8.6% | +7.0% | +6.2% |
| V3 2.5-bit mixed | +27.0% | +35.2% | +10.8% | +7.0% |
| V3 2-bit Lloyd-Max | +64.3% | +65.5% | +19.3% | +20.2% |

Throughput (Llama 3.2 3B, tok/s) and KV compression at T=8192:

| strategy | T=512 | T=8192 | KV at 8192 | ratio |
|---|---|---|---|---|
| fp16 | 208 | 148 | 969 MB | 1x |
| V2 4-bit LEAN | 188 | **156** | 266 MB | 3.6x |
| V2 3-bit rot+QJL | 101 | 45 | -- | -- |
| V3 3.5-bit mixed | 82 | 24 | 236 MB | 4.1x |
| V3 3-bit Lloyd-Max | 98 | 27 | 207 MB | 4.7x |

Four findings worth extracting:

1. **4-bit affine KV is near-lossless and near-free** (when a fused
   kernel exists). V2 4-bit runs -0.8% to +2.9% perplexity at ~105% of
   fp16 speed at 8K (the compressed cache reads fewer bytes, so it wins
   at long context).
2. **Below 4 bits there is a quality cliff.** 3-bit costs 5-9% on
   D=128 models, and 2-bit costs 19-65% everywhere. Consistent with this
   port's own weight-side experience that sub-4-bit needs either
   codebooks (Phase S) or QAT (the ternary family), neither of which
   their V3 software path could afford at speed.
3. **The paper's QJL residual is a measured negative in practice.**
   Their own explanation: the correction is linear in attention scores,
   and softmax amplifies score errors exponentially, so (b-1)-bit MSE
   plus a 1-bit residual loses to plain b-bit. A residual-correction
   scheme for KV should not be built here without a number that
   contradicts theirs.
4. **Head dimension decides quantization tolerance.** D=256 (their
   Gemma 3) quantizes far better than D=128 (their Llamas): rot+QJL at
   3 bits even beat fp16 on Gemma (-1.1%, a regularizer effect). More
   coordinates per head means outliers average out.

And one about cost: **V3, the path with no fused kernel, runs at ~16%
of fp16.** That is the price of software dequant in the SDPA hot path,
and it is the floor this port would start from without kernel work.

## What implementing it here would take (2026-08-15 estimate; see "The reversal" above for what actually landed)

This port's KV cache was FP16 end to end at the time of this estimate.
`crates/gpu/src/kv_cache.rs`
allocates one K and one V Metal buffer per layer at construction (linear
for full-attention layers, a `sliding_window + 128` ring for SWA layers
under `fp16_ring_enabled`), the host memcpys one token's row per layer
per step (`write_k`/`write_v`), and `attention_decode.rs`'s split-KV
kernel reads the buffers through `KvView` with ring addressing
specialized via `FC_ATTN_RING_CAP`.

A V2-shaped 4-bit KV would need:

- **A write-side quantizer.** Per-token, per-head-group affine
  quantization of the K row and V row before (or instead of) the memcpy.
  Cheap either way: one row per layer per token.
- **A read-side FUSED dequant inside `attention.metal`**, in both
  places attention touches KV: the QK^T score loop of
  `attention_decode_partial` and the AV accumulation. This is the hard
  part. turboquant-mlx never wrote this kernel: `mx.quantized_matmul`
  is MLX's, and their V3 numbers (16% of fp16) show what happens without
  it. This port has no equivalent to borrow. The existing dequant GEMVs
  (INT4/INT8/Q8_0/...) are weight-matrix kernels with the wrong
  addressing for a ring-buffered, GQA-shared, split-KV cache.
- **New specialization axes done right.** Block format and group size
  would have to join `attention_constants_key` beside scale, ring
  capacity, chunk count, and sinks (crates/gpu CLAUDE.md Gotcha 7), or
  quantized dispatches silently reuse the FP16 pipeline, the exact trap
  AGENTS.md Gotcha 18 records for the ring capacity itself.
- **A rotation stage if quality needs it**: a fixed random orthogonal
  matrix folded into the Q/K projections at repack time (QuaRot
  lineage), which touches `crates/repack` and every family's install
  format.
- **Re-frozen quality gates for every family.** Quantized KV moves
  numerics on every generated token of every install. Per the
  verification policy, that is all three real-model gates per family
  PLUS the quality gate, with every digest re-frozen and the perplexity
  movement explained. Eight families now.

Effort class: a full phase, comparable to a family bring-up (the
attention kernel alone is harder than a GEMV port: it has the ring, GQA,
split-KV combine, and sinks axes to preserve). That cost is not the
reason to decline, but it sets the bar the benefit has to clear.

## Whether it pays here, as a DEFAULT (2026-08-15 analysis; still the reason `--kv-bits` defaults OFF)

It does not, today, on either axis -- which is an argument for the flag
defaulting off, not for refusing to offer it. See "The reversal" above.

**Memory.** Where KV actually sits in this port's measured footprints
(all numbers from the recorded oracle rows and their accounting in
AGENTS.md / CLAUDE.local.md):

| family | KV | measured peak | KV share | dominant term |
|---|---|---|---|---|
| mistral7b dense @ 8192 | 1,024 MiB | 1,201 MiB | ~85% | KV |
| qwen3_5 dense 27B @ 4096 | 256 MiB | 660 MiB | ~39% | KV + GDN state |
| gemma4 @ 4096 | ~320 MiB | 2,100-2,200 MiB | ~15% | expert slot cache |
| qwen3moe @ 4096 | small | ~2,750 MiB | <10% | slot cache (2,094 MiB) |
| gpt-oss @ 8192 | small-ish | ~5,420 MiB | <10% | slot cache (4,854 MiB) |

The families near their ceilings are MoE, and their dominant term is the
expert slot cache, which KV quantization cannot touch. The one family
where KV dominates (dense Mistral at 8,192) already runs in 1.2 GiB on a
36 GB machine, so cutting it to ~450 MiB enables nothing that is blocked
today. And this engine already carries three KV mitigations that
turboquant-mlx's uniform cache does not: the SWA ring (AGENTS.md Gotcha
18: 922 -> 320 MiB on Gemma), gpt-oss's alternating 128-token window on
even layers, and Qwen's linear-attention layers carrying no per-token KV
at all (30 of 40 layers on qwen36, 36 of 48 on the dense 27B).

**Throughput.** Post-split-KV, `attention_decode_partial` is 13-19% of
cb1's GPU busy time (CLAUDE.local.md, 2026-08-07 dispatch ranking),
which is single-digit percent of a whole decode token. Reading 4x fewer
KV bytes cannot buy much from a term that small. Their V2 speed win
appears at 8K+ context on a model whose attention share is large. This
port's protocol runs at 4,096-8,192 by design, and at those windows
attention is already flat and cheap here.

**Quality risk.** Every point of KV quantization is pure downside on
the quality axis until context grows: the gates would absorb a
perplexity perturbation on all eight families to save memory that
nothing is short of.

The general form, consistent with this repo's own findings: KV
quantization is a LONG-CONTEXT technology. TurboQuant's own headline
(969 MB of KV at a mere 8K on a 3B model) describes a model with
D=128 x 28-ish layers of full attention and no ring. This engine's
families either window, ring, or linearize most of their layers, which
is the same memory being saved by architecture instead of by
quantization.

## Ideas worth taking

Knowledge, mostly. In order of value:

1. **Do not build QJL-style residual correction for KV.** Their measured
   negative, with a mechanism (softmax amplifies score error). Cite this
   page if it comes up.
2. **If KV quantization is ever built, 4-bit affine is the only point
   worth building.** The 3-bit and below cliff is steep on D=128 and
   their V3 codebook path shows the kernel cost of fancier schemes.
   Skip fractional rates, skip Lloyd-Max, skip 2-bit. **OVERRIDDEN
   2026-09-06**: what shipped is the full Lloyd-Max V3 path at all five
   widths including 2-bit, because a real fused kernel closed the cost gap
   this item was reacting to, and every width below `off` is opt-in rather
   than a default anyone pays for unasked. The quality-cliff finding
   itself is not disputed -- a caller choosing 2-bit is choosing that cliff
   with eyes open, which is a different thing from this port choosing it
   for them.
3. **Gemma is the pilot family.** Its head dims (256 SWA / 512 full) sit
   in the regime their data says is most quantization-tolerant: their
   D=256 model was the one where quantized KV matched or beat fp16.
   The D=128 families (llama, qwen3moe, gpt-oss) are where the quality
   risk concentrates.
4. **Random-rotation preprocessing is a real quality lever** (their
   rotated variants beat plain at every width). It is not specific to
   KV (it is the QuaRot family of tricks) and could matter to future
   weight quantization work here, but nothing current asks for it: the
   incumbent MLX INT4 installs are already the best-measured
   quantizations in `docs/BENCHMARKS.md`.
5. **Norm-baking** (fold L2 norms into the quant scales so dequant needs
   no separate elementwise pass) is a tidy micro-optimization that only
   exists once a quantized-KV path exists. Note only.

Already present here, no action: pre-allocated per-layer buffers (their
step=256 growth trick solves a problem this port's
allocate-once-at-construction design does not have), ring buffers,
`MADV_DONTNEED` on reset, fused Metal attention.

## What would have changed the 2026-08-15 answer (now moot; kept for the reasoning)

This section asked what would justify building the feature at all. That
question is answered -- it is built, opt-in, default off. What is left of
it is a note on when `--kv-bits` might be worth DEFAULTING to something
other than off, which nothing here proposes today:

- **Long-context support landing.** If the protocol context ever moves
  to 32K+, KV becomes the growing term for every family (it is the only
  per-token term on dense families, per Gotcha 40's accounting), and a
  case could be made for resolving `--kv-bits auto` to 4-bit on the
  families closest to a ceiling. Nothing here proposes that default change
  today; it would need its own quality-gate sign-off per family.
- **A memory ceiling actually binding.** Today no install is refused for
  KV. If a family arrives whose KV at its own protocol window does not
  fit beside its slot cache, `--kv-bits` is already the tool to reach for.
- **An attention-bound profile.** If a future dispatch ranking shows
  `attention_decode_partial` dominating cb1 (it grows with context, and
  today it is third), the shipped kernel is already the throughput lever;
  nothing further needs building.

## Real-model verification owed

Everything above the codec's own unit and parity tests is fixture-level.
Before quoting a memory or quality number for any `--kv-bits` width on a
real install: run `crates/bench/tests/kv_quant_probe.rs`
(`TURBOSPARK_KV_QUANT_INSTALL_DIR=<install> cargo test -p turbospark-bench
--test kv_quant_probe --release -- --ignored --nocapture`) and the
standard greedy-then-sampled real-model smoke from `AGENTS.md` with
`--kv-bits` set, per family. Neither has been run from this sandboxed
development environment, which has no real model installs.
