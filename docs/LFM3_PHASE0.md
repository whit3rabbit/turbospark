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
| 5-6. Memory, quality | Done for the MLX 4-bit install: `lfm3_memory_oracle` (peak 343-351 MiB, ceiling 420) and `lfm3_quality_gate` (perplexity 14.7455 and both digests, reproduced from two fresh processes). Not run on the Q8_0 install |
| 7. CLI | Partial: probe, pull and run work for the MLX and GGUF repos. No catalog row |

Not yet checked against another engine. Homebrew llama.cpp 0.7.0 refuses
`lfm3` as an unknown architecture, and on 2026-10-27 a Homebrew HEAD build
failed because llama.cpp master itself did not compile (`ggml_top_k_mask`
undeclared in `qwen4exp.cpp` and `glm5-next.cpp`). Release tags from
`b11938` postdate the `lfm3` commit and are the next thing to try. The
inputs are ready: a raw prompt whose tokens match ours exactly, and 200
greedy tokens from our Q8_0 install.

The 4-bit MLX install's greedy answer to a TCP prompt is fluent and
factually wrong in places, and the Q8_0 install's is better. Whether that
is quantization or a port error is that comparison's question.

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
