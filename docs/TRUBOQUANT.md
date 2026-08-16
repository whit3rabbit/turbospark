# TurboQuant-MLX and KV-Cache Quantization (Assessed, Not Adopted)

The question this page answers: should this engine adopt KV-cache
quantization in the shape of `sharpner/turboquant-mlx`
(https://github.com/sharpner/turboquant-mlx), a proof of concept of
Google's 2025 TurboQuant paper on Apple Silicon, and if not, are any of
its ideas worth taking anyway?

Answer, assessed 2026-08-15 by reading their README against this port's
own recorded measurements (no code was run and nothing here is a new
measurement): **implementable, but it does not pay on this engine today.**
KV is not the dominant memory term for any family that is near a memory
ceiling, attention is a single-digit share of decode at this port's 4-8K
contexts, and the one thing that makes turboquant-mlx fast (MLX's fused
`quantized_matmul`) is exactly the kernel this port would have to write
itself. What IS worth keeping is their negative results and one favorable
datapoint about Gemma's head dimensions, all recorded below with explicit
revisit triggers.

This is a desk assessment, not a measured negative like
`docs/EXPERT_ROUTING.md` or `docs/SPECULATIVE_DECODING.md`: their numbers
are theirs (different models, different engine) and this port's numbers
come from its own frozen rows. The two are never blended into one
comparison here.

## What turboquant-mlx is

A proof of concept of the TurboQuant paper (Google, 2025): quantize the
attention KV CACHE to 2-4 bits so long contexts fit in memory and the
attention pass reads fewer bytes. Weights are untouched. This is entirely
orthogonal to the weight quantization this port already does everywhere.

Two implementation paths:

- **V2 (affine, hardware-accelerated):** K/V quantized through
  `mx.quantize` and consumed by `mx.quantized_matmul`, MLX's fused
  dequant-inside-the-matmul Metal kernel. Variants add a random QR
  ROTATION before quantizing (distributes outlier channels so the values
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
   fp16 speed at 8K (the compressed cache reads fewer bytes, so it WINS
   at long context).
2. **Below 4 bits there is a quality cliff.** 3-bit costs 5-9% on
   D=128 models, and 2-bit costs 19-65% everywhere. Consistent with this
   port's own weight-side experience that sub-4-bit needs either
   codebooks (Phase S) or QAT (the ternary family), neither of which
   their V3 software path could afford at speed.
3. **The paper's QJL residual is a measured negative in practice.**
   Their own explanation: the correction is linear in attention SCORES,
   and softmax amplifies score errors exponentially, so (b-1)-bit MSE
   plus a 1-bit residual loses to plain b-bit. A residual-correction
   scheme for KV should not be built here without a number that
   contradicts theirs.
4. **Head dimension decides quantization tolerance.** D=256 (their
   Gemma 3) quantizes far better than D=128 (their Llamas): rot+QJL at
   3 bits even BEAT fp16 on Gemma (-1.1%, a regularizer effect). More
   coordinates per head means outliers average out.

And one about cost: **V3, the path with no fused kernel, runs at ~16%
of fp16.** That is the price of software dequant in the SDPA hot path,
and it is the floor this port would start from without kernel work.

## What implementing it here would take

This port's KV cache is FP16 end to end. `crates/gpu/src/kv_cache.rs`
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
  `attention_decode_partial` and the AV accumulation. This is the whole
  ballgame. turboquant-mlx never wrote this kernel: `mx.quantized_matmul`
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

## Whether it pays here

It does not, today, on either axis.

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
KV bytes cannot buy much from a term that small. Their V2 speed WIN
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
   Skip fractional rates, skip Lloyd-Max, skip 2-bit.
3. **Gemma is the pilot family.** Its head dims (256 SWA / 512 full) sit
   in the regime their data says is MOST quantization-tolerant: their
   D=256 model was the one where quantized KV matched or beat fp16.
   The D=128 families (llama, qwen3moe, gpt-oss) are where the quality
   risk concentrates.
4. **Random-rotation preprocessing is a real quality lever** (their
   rotated variants beat plain at every width). It is not specific to
   KV (it is the QuaRot family of tricks) and could matter to future
   WEIGHT quantization work here, but nothing current asks for it: the
   incumbent MLX INT4 installs are already the best-measured
   quantizations in `docs/BENCHMARKS.md`.
5. **Norm-baking** (fold L2 norms into the quant scales so dequant needs
   no separate elementwise pass) is a tidy micro-optimization that only
   exists once a quantized-KV path exists. Note only.

Already present here, no action: pre-allocated per-layer buffers (their
step=256 growth trick solves a problem this port's
allocate-once-at-construction design does not have), ring buffers,
`MADV_DONTNEED` on reset, fused Metal attention.

## What would change the answer

Named so the next reader does not re-derive this page:

- **Long-context support landing.** If the protocol context ever moves
  to 32K+, KV becomes the growing term for every family (it is the ONLY
  per-token term on dense families, per Gotcha 40's accounting) and 4-bit
  affine KV with a fused dequant attention kernel becomes the design to
  reach for. V2-shaped: affine, 4-bit, optional baked rotation, no QJL,
  nothing below 4 bits. Pilot on Gemma per item 3 above.
- **A memory ceiling actually binding.** Today no install is refused for
  KV. If a family arrives whose KV at its own protocol window does not
  fit beside its slot cache, this trades kernel work for admission.
- **An attention-bound profile.** If a future dispatch ranking shows
  `attention_decode_partial` dominating cb1 (it grows with context, and
  today it is third), halving its bytes read becomes a throughput lever
  rather than a memory one.

None of those are on the roadmap. Until one is, this page is the
decision: read, learned from, not adopted.
