# LFM3 Phase 0

The facts behind `ModelFamily::Lfm3`, read off the published files on
2026-10-27. Scope is the text model `LiquidAI/LFM3-1.7B` (and its `-Base`
sibling, which shares the config). Everything here was read, not inferred.
`docs/NEW_MODEL.md` is the checklist this follows.

## Status

| Phase | State |
| --- | --- |
| 0. Scope | Done (this page) |
| 1. Repack | Done: family, baseline, HF/MLX parser, GGUF derivation and name table, `rope_freqs` value check, synthetic install, open-boundary test |
| 2. Kernels | Done: `short_conv_decode` / `short_conv_prefill`, CPU reference, parity tests |
| 3. Decode flow | Done: `families/lfm3/`, sequential prefill. Greedy, templated, 600 tokens coherent on the MLX 4-bit and GGUF Q8_0 installs |
| 4. Head and sampler | Partial: sampled at CLI defaults coherent for 600 tokens, a short question stops `EndOfTurn`. No sampled run on the Q8_0 install yet |
| 5-6. Memory, quality | Done for the MLX 4-bit install: `lfm3_memory_oracle` (peak 343-351 MiB, ceiling 420) and `lfm3_quality_gate` (perplexity 14.7455 and both digests, reproduced from two fresh processes). The Q8_0 install's own pair is frozen too (`lfm3_q8_memory_oracle`, `lfm3_q8_quality_gate`). Both quality rows re-froze 2026-10-28 for the GQA decode attention default ("Decode attention" below) |
| 7. CLI | Done: catalog rows `lfm3-1.7b` (MLX 4-bit, verified) and `lfm3-1.7b-q8` (GGUF Q8_0, verified), both pinned, guard byte figures pasted back |
| 8. Docs | Done: BENCHMARKS.md, MODELS.md, README, DEVIATIONS.md, CHANGELOG |

Checked against llama.cpp on the same bytes (2026-10-28, Homebrew llama.cpp
0.8.0, build 11960): the Q8_0 GGUF, the same 42 prompt ids, greedy with the
GGUF's repeat penalty turned off on llama.cpp's side. 600 greedy tokens on a
TCP prompt are identical, and a 226-token answer to a second prompt is
identical and ends at the same end of turn. There is no external reference
for the MLX bytes, since mlx-lm has no `lfm3` model.

The 4-bit MLX install's greedy answer to the TCP prompt is fluent and
factually wrong in places, and the Q8_0 install's is better. Since the Q8_0
path matches llama.cpp, that gap is the 4-bit artifact, not the port.

## Chunked prefill

`families/lfm3/prefill.rs` runs a micro-batch of up to 64 prompt tokens
through every layer in one command buffer, each projection as one M-row
GEMM (Steel from 32 rows on INT4), the full layers' attention through the
multi-row GQA kernel, the sliding layers' through the one-row kernel per
row. Only INT4-affine installs take it; `supports_chunked_prefill` is false
on the Q8_0 install. `TURBOSPARK_LFM3_CHUNKED_PREFILL=0` runs the per-token
loop behind the same entry point.

With `TURBOSPARK_PREFILL_STEEL=0` and `TURBOSPARK_PREFILL_ROWS_ATTN=0` the
driver is bit-exact against sequential prefill, decode after it included
(`the_exact_chunk_arm_matches_sequential_bit_for_bit`, on an INT4 fixture
whose sliding ring wraps inside a micro-batch).

On the MLX 4-bit install (M4 Max, AC, 2026-10-28):

| check | result |
| --- | --- |
| `tslab ab`, long-synthesis (2,918 tokens), chunk 128 | +771% prefill tok/s over the per-token loop, CI [+770%, +776%], 3 clean pairs, null floor 0.48%, peak +15.2 MiB (ledger `lfm3-c2-ttft`) |
| `tslab profile`, same prompt | 12.95-12.99 s per-token, 4.49-4.61 s exact arm, 1.47-1.48 s default; the exact arm's prefill logits digest equals the per-token arm's, and all three 128-token greedy decodes are identical (`lfm3-c1-chunk`) |
| `lfm3_prefill_arm_probe`, 64-3000 tokens | mean KL 1.80e-5 nats with the multi-row attention, 1.60e-5 with the one-row kernel; top-1 6/6, top-5 5/5 at every length |

The memory oracle and the quality gate drive prefill through `produce`
explicitly, so their frozen rows do not cover this driver; the gate still
reproduces them unchanged. The peak figure above is the A/B's own paired
reading, not an oracle row. The probe row above predates the GQA decode
default below; re-run after it, it reads 1.89e-5 and 1.72e-5 nats, top-1 6/6,
and the exact arm still matches sequential prefill bit for bit. On top of
`main`'s `TURBOSPARK_PREFILL_SMALL_N` route (which now engages on this
driver's default arm for projections too narrow for a Steel tile) it reads
2.19e-5 and 1.41e-5 nats, top-1 6/6. The A/B above predates that route.

## Decode attention

The flow's attention scratch takes the GQA-shared decode partial
(`attention_decode_gqa_partial`) since 2026-10-28: 16 Q over 4 KV at
head_dim 128 is a shape `attention_decode_gqa_supported` serves, on the full
and the sliding layers alike. `TURBOSPARK_ATTN_DECODE_KERNEL=barrier` restores
the old kernel process-wide. It is KL-bounded, not bit-exact, so both quality
rows re-froze by the playbook protocol (`SPEEDUP_PLAYBOOK.md`, "Quality gates
on both arms"):

| check | MLX 4-bit | GGUF Q8_0 |
| --- | --- | --- |
| barrier arm reproduces the frozen row | yes | yes |
| perplexity, barrier -> GQA | 14.7455 -> 14.7434 | 15.8615 -> 15.8614 |
| greedy digest | unchanged | unchanged |
| sampled text, first divergence | char 481 of ~1240, a swapped phrase | char 453 of ~1185, a swapped phrase |
| two fresh GQA processes agree | yes | yes |
| memory oracle peak | 347 MiB (ceiling 420) | 350 MiB (ceiling 420) |

`tslab ab` on the MLX 4-bit install, long-synthesis (2,918 tokens), 128
decode, chunk off, M4 Max, AC: the barrier arm reads -15.67% decode tok/s
against the GQA default, CI [-16.77%, -14.99%], 0 of 3 clean pairs better,
null floor 2.16% (ledger `lfm3-b2b-gqa-decode`, verdict
`1793229947818-85694-16`). The engagement lines confirm `attn_decode_gqa` on
the default arm with no env set. The same A/B in the other direction before
the flip read +18.64% (`lfm3-a3-gqa`). The per-token prefill rate under
chunk off moves less, -10.71% for barrier (`lfm3-b2-gqa-default`).

## Fused FFN

Decode's FFN runs gate, up and silu as one INT4 GEMV
(`gpu::encode_gate_up_act_gemv`, `families/lfm3/mlp.rs`), 3 dispatches to 1
per layer, 48 fewer per token. The kernel is `dequant_int4_gemv_simd` with a
second accumulator, rounded to half before the vendored `silu_mul_fp16`, so
it is bit-exact against the chain
(`the_fused_ffn_matches_the_three_dispatch_chain_bit_for_bit`; switching the
fused call to gelu fails it). Down stays a plain GEMV, because
`post_ffn_norm` sits between it and the residual add. INT4 installs only,
and not where `TURBOSPARK_INT4_MMA8` takes the shape.
`TURBOSPARK_LFM3_FUSED_FFN=0` keeps the chain.

Both quality rows reproduce unchanged with it on, from two fresh processes
on the MLX 4-bit install, and the memory oracle reads 349 MiB. The speed
effect is not measurable on this host: `tslab ab --metric decode_tok_s`,
chunk off, default against `lfm3_fused_ffn=0`, 3 clean pairs each, M4 Max,
AC, is NOISE at both protocol prompts. The chain reads -0.55% on
short-explanation (CI [-1.45%, +0.36%], null floor 1.13%) and -2.87% on
long-synthesis (CI [-3.16%, -2.64%], all pairs worse, under the 4.17% null
floor). Ledgers `lfm3-b1b-fused-decode-short-explanation` and
`lfm3-b1b-fused-decode-long-synthesis`.

## Measured so far

On the M4 Max, `turbospark-check --temperature 0`, sequential prefill:
MLX 4-bit decodes at about 205 tok/s on a 25-token prompt, GGUF Q8_0 at
about 97 tok/s on a 42-token prompt. These are single smoke runs, not
benchmark rows.

The memory oracle's peak (343 to 351 MiB at 8,192 context) agrees with the
static accounting below: about 100 MiB of KV and conv state plus a
~250 MiB dense-install residual. It drives two of the three protocol cases:
`medium-review` reasons past the 4,096-token budget while coherent and
stops `endOfTurn` at 7,490 tokens when given room, so it is a verbosity
limit, not a defect. `short-explanation` decode reads either about 180-198
or about 68-71 tok/s across runs, on a quiet host, while `long-synthesis`
holds at 164-172. The working hypothesis is the GPU's clock state on small
per-token passes. The floor (50 tok/s) sits under the slow state.

## Two fixes outside the family that LFM3 needed

- `{% generation %}`: transformers' assistant-mask block tag, which
  minijinja rejects, so `pull` refused the template. `rewrite_generation_tags`
  (`crates/tokenizer/src/jinja_compat.rs`) maps it to `{% if true %}` with
  the same whitespace control.
- ChatML BOS: the resolver hardcoded Qwen's `<|endoftext|>` stand-in. LFM3
  declares `<|startoftext|>` and its template prints `{{ bos_token }}`, so
  every prompt opened with an end-of-text token. A config-declared BOS now
  wins. Qwen's `bos_token: null` is unchanged.

## Pins

| Artifact | Revision | Notes |
| --- | --- | --- |
| `LiquidAI/LFM3-1.7B` | `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd` | Reference `config.json`, `modeling_lfm3.py`, tokenizer, chat template |
| `LiquidAI/LFM3-1.7B-MLX-4bit` | `d0ca73dcfa70ce18733754a492cdc1fb0123262d` | Affine 4-bit, group 64, BF16 companions, embedding quantized. Inside `validate_quant` |
| `LiquidAI/LFM3-1.7B-GGUF` | `b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf` | `Q8_0` (q8_0) and `Q4_K_M` (q4_k + q6_k) are executable block types |

`tests/lfm3_network.rs` re-reads all three and asserts that the GGUF headers
and the MLX config derive the same `ArchConfig`, equal to
`model_io::lfm3_1_7b()`.

## Architecture strings

HF `model_type` and GGUF `general.architecture` are both `lfm3`, so the
manifest family string is `lfm3` too. It is not `lfm2`: that line has no
sliding window, no sandwich norms and no output gate, and stays unregistered.

## The layer graph

24 layers, hidden 2048, 16 query heads over 4 KV heads, head_dim 128,
SwiGLU FFN 7680, vocab 128000, tied head, every norm at eps 1e-5.
`layer_types` gives 13 `conv` layers (0, 1 and every odd index), 6
`full_attention` (2, 6, 10, 14, 18, 22) and 5 `sliding_attention` (4, 8,
12, 16, 20) at window 256. The mask codes are 6, 1 and 0. Mask 6 is new.

```text
h = embed[id]                                    // no scale, no norm
per layer i:
  x = rms(h, operator_norm)
  conv layer (mask 6):
    B, C, xx = split3(x @ in_proj)               // 3 x 2048, no bias
    u = B * xx
    y = causal_depthwise_conv7(tail | u)         // no activation
    o = (C * y) @ out_proj                       // tail keeps the last 6 rows of u
  attention layer:
    q = q_norm(x @ q_proj), k = k_norm(x @ k_proj)   // per-head RMSNorm
    v = x @ v_proj
    full:    rope_proportional_neox(theta 1e6, 32 of 64 pairs)
    sliding: rope_proportional_neox(theta 1e4, all 64 pairs)
    a = attention(scale 128^-0.5, causal or window 256)
    a = a * sigmoid(x @ gate_proj)               // one scalar per head
    o = a @ o_proj
  h = h + rms(o, post_operator_norm)
  h = h + rms(w2(silu(w1 x') * w3 x'), post_ffn_norm), x' = rms(h, ffn_norm)
h = rms(h, embedding_norm)                       // the FINAL norm, despite the name
logits = h @ embed_tokens
```

## RoPE

HF's `compute_default_rope_parameters` keeps the frequency progression of
the whole head and zeroes the inverse frequencies past
`partial_rotary_factor * head_dim / 2`, with `rotate_half` pairing
`(i, head_dim/2 + i)`. That is `rope_proportional_neox` with a head_dim
divisor (Gemma's convention), not Spark's `rope_neox_subdim`. The factor is
0.5 on full layers. Sliding layers rotate the whole head.

## GGUF conventions

- `rope_freqs.weight` (`[64]`, F32) holds 32 x `1.0` then 32 x `1e30`.
  llama.cpp applies it only to non-sliding layers. It encodes the zero tail,
  it is not learned scaling. Every other family still refuses the tensor by
  name. For `lfm3` the walk reads its bytes and accepts only that exact
  pattern, with the boundary at `partial_rotary_factor`, then drops it. A
  missing vector also refuses.
- `rope.dimension_count` is 128, the whole head. The partial factor lives
  only in the vector above.
- `attention.head_count_kv` is a per-layer array with 0 on conv layers.
  `attention.sliding_window_pattern` marks sliding layers `true`.
- `shortconv.l_cache` is 7.
- `shortconv.conv.weight` is ggml `[7, 2048]`, logical `[2048, 7]`. MLX ships
  `[2048, 7, 1]` and HF `[2048, 1, 7]`. All three are the same bytes:
  channel-major with the taps contiguous. No transform is needed.
- `output_norm.weight` maps to `embedding_norm.weight`, the MLX spelling.
- The tokenizer is `tokenizer.ggml.pre = lfm2`. A GGUF catalog row needs
  `sidecar_repo = LiquidAI/LFM3-1.7B` for `tokenizer.json`.

## Chat template

ChatML (`<|im_start|>` / `<|im_end|>`), BOS `<|startoftext|>` 124894, EOS
list `[124900]`. The generation prompt opens `<think>` by default
(`enable_thinking` true), so completions run long. `protocol_parameters`
carries a provisional 4,096-token budget until real completions are
measured. The template uses HF's `{% generation %}` block tag, which
minijinja may not parse. Check before Phase 3's gate.

## Static memory accounting

To confirm in Phase 5. Full-attention KV is 6 layers x 4 heads x 128 x 2 x 2
bytes, 12 KiB per token, about 96 MiB at 8192. Five sliding layers hold a
ring of `256 + 128` rows. Conv tails are 13 x 6 x 2048 x 2 bytes, 312 KiB,
fixed.

Dense, so per AGENTS.md Gotcha 40 the resident weights should not
appear in `phys_footprint`. There is no streamed-expert advantage here.

## Negative findings and deferrals

- `LFM3-1.7B-QAD-Q4_0.gguf` refuses: q4_0 is not an executable block type.
- `LFM3-VL-2B`, `LFM3-Audio-2B`, `LFM3-Omni-2B`: deferred. Their towers are
  not the Qwen ViT this engine has.
- `d1-2B`, `d1-Omni-2B`: not assessed. Likely out of class.
- mlx-lm has no `lfm3` model file as of this date (only `lfm2`, `lfm2_moe`,
  `lfm2-vl`). The MLX conversions ship LiquidAI's own code. The HF
  `modeling_lfm3.py` is the reference, cross-checked against llama.cpp's
  `src/models/lfm3.cpp`.
- No new `ArchConfig` sub-config: one scalar, `short_conv_kernel_size`
  (manifest `shortConvKernelSize`), resolved to 0 when absent, never to the
  Gemma value.
