# Gemma 3 Phase 0 findings (`gemma3` / `gemma3_text`, GGUF `gemma3`)

Phase 0 below was fact-finding only. The 1B GGUF slice is implemented; see
[Implementation (1B GGUF)](#implementation-1b-gguf) at the end. Read 2026-11-14 off the real
published files, headers and configs, no weights downloaded.

## Witnesses

| what | where | revision |
|---|---|---|
| GGUF (1B) | `ggml-org/gemma-3-1b-it-GGUF` (`Q4_K_M`, `Q8_0`, `f16`) | `f9c28bcd85737ffc5aef028638d3341d49869c27` |
| MLX configs | `mlx-community/gemma-3-1b-it-4bit` (`model_type gemma3_text`), `mlx-community/gemma-3-4b-it-4bit` (`gemma3`, text under `text_config`) | read at HEAD |
| official BF16 | `google/gemma-3-*-it` | **gated (HTTP 401)**: no unauthenticated BF16 reference, so a byte-level HF to GGUF check needs a token or an `mlx-community` bf16 copy |

Download volume for the GGUF architecture string `gemma3` was about 0.9M in a
2026-11-14 sample of the top and trending GGUFs.

## Facts (1B Q4_K_M header: 340 tensors, 26 blocks)

- Architecture `gemma3`; `block_count 26`, `embedding_length 1152`,
  `feed_forward_length 6912`, `head_count 4`, `head_count_kv 1`,
  `key_length`/`value_length 256`, `rms_epsilon 1e-6`, `freq_base 1e6`,
  `sliding_window 512`. No `sliding_window_pattern` key: the 5-local:1-global
  schedule is the converter's constant (`sliding_window_pattern 6` in the MLX
  config), and the local rope base (10000, `rope_local_base_freq`) is not in the
  file either.
- Tensors per layer (13): `attn_q`, `attn_k`, `attn_v`, `attn_output`,
  `attn_q_norm`, `attn_k_norm`, `attn_norm`, `post_attention_norm`, `ffn_norm`,
  `post_ffw_norm`, `ffn_gate`, `ffn_up`, `ffn_down`; plus `token_embd` and
  `output_norm`. No `output.weight`: tied head.
- MLX config: `query_pre_attn_scalar 256`, no softcap, `rope_scaling` absent on
  1B and `{factor 8, linear}` on the 4B text config (global layers).

## Why this is not a row onto an existing flow

The Gemma 4 flow (`crates/runtime/src/families/gemma4/attn.rs`) is the nearest
neighbour (sliding/full schedule, sandwich norms, q/k norms, tied head, scaled
embedding) and differs in ways that would decode fluently and wrongly:

1. **V is normalized in Gemma 4** (`encode_rms_norm_no_scale_perhead`); Gemma 3
   has no V norm.
2. **Norm convention.** Gemma 3's RMSNorm applies `1 + w` and llama.cpp's
   converter bakes the `+1` into the GGUF; the Gemma 4 flow reads plain `x * w`
   (`encode_rms_norm_bf16w`, verified byte-identical HF vs MLX for Gemma 4). A
   Gemma 3 install needs the `-1` unbake (the Muse GGUF walk is the precedent:
   `gguf_checkpoint::conventions`) or a centered kernel.
3. **Linear RoPE scaling (factor 8) on the global layers of 4B and up** has no
   path in either flow (`RopeScalingConfig` is YaRN-shaped).
4. **Attention scale** comes from `query_pre_attn_scalar`, which is not a GGUF
   key; it equals `head_dim` for 1B and 4B but not for every size.
5. Gemma 4 specifics that Gemma 3 lacks: per-layer scalar, `attention_k_eq_v`,
   partial rotary on full layers, MoE and router.

## Plan, in order

1. Decide the family boundary: a new `ModelFamily::Gemma3` (cross-layer: model-io,
   registry, names, config, runtime flow, FFI, Swift, catalog, docs) versus
   switches on the Gemma 4 flow. Flow selection keys on the family, so reusing
   `Gemma4` for a Gemma 3 file is the hazard `AGENTS.md` names.
2. 1B first (no rope scaling): GGUF row, name table, norm `-1` convention,
   derived mask `[0,0,0,0,0,1]` and the 10000 local base as explicit constants
   guarded by the config, q/k un-permute check against a bf16 reference.
3. Get a reference: an `mlx-community/gemma-3-1b-it-bf16` pair (public) for byte
   checks, or a Hugging Face token for the gated official repo.
4. Real 1B install and decode, then 4B with the linear-scaling gap closed or
   refused explicitly.

Open questions: whether the global-layer rope in the GGUF carries the scaling in
`rope_freqs.weight` (the Llama 3.1 refusal precedent) for 4B+; the exact
`query_pre_attn_scalar` for 12B and 27B.

## Implementation (1B GGUF)

Done 2026-11-14 for `ggml-org/gemma-3-1b-it-GGUF` rev `f9c28bcd`.

### Decisions, each read off bytes or the HF source

- **A new `ModelFamily::Gemma3`, on the Gemma 4 flow's dense arm.** Flow
  selection keys on the family, and four Gemma 4 behaviours are wrong for
  Gemma 3, so switches keyed on `Gemma4` would have meant labelling a Gemma 3
  file `Gemma4`. The shared code is the dense arm of `families/gemma4/`; the
  family-keyed switches are in `norm.rs` (centered norms, V norm) and
  `state.rs` (layer scalar, the flow check). The tokenizer needed its own
  dialect (`ChatDialect::Gemma3`): a Gemma 3 table has `<start_of_turn>` and
  `<end_of_turn>`, no `<turn|>`, so the Gemma fallback failed to load it.
- **Norms: `x * (1 + w)` on every norm, unbaked at install.** llama.cpp adds 1
  to every `*norm.weight` (per-layer, q/k and the FINAL norm, unlike Muse
  Glimmer). Measured: GGUF minus 1, rounded to BF16, equals the HF bf16 tensor
  byte for byte on layers 0, 5, 25 and the final norm; the raw GGUF value
  does not. The install stores the centered value and the flow uses the
  centered kernels. The alternative (keep the baked value, plain kernel) was
  rejected: BF16 near 1.0 has a quantum of 2^-8, which the centered kernel's
  own comment names as destructive.
- **No q/k row permutation.** Q8_0 `attn_q`/`attn_k` rows match the HF rows in
  stored order (relative error 0.0067 and 0.0059) and not un-permuted (1.46).
- **No V norm.** `modeling_gemma3.py` (transformers v4.50.0) defines `q_norm`
  and `k_norm` only. The GGUF cannot show this (Gemma 4's V norm is also
  weightless), and a decode cannot either: applying it by mutation still
  decodes fluent text. This rests on the HF source.
- **Constants the GGUF omits**, fixed in `gemma3_1b()` and guarded by
  `gguf_config`: the 5:1 mask (`is_sliding = (i + 1) % 6 != 0`, HF source),
  local rope base 10000 (`rope_local_base_freq`; the generic `freq_base`
  handling would give the sliding layers 1e6), attention scale
  `query_pre_attn_scalar^-0.5` = 0.0625, epsilon 1e-6.
- **Refused by name:** any `rope.scaling.*` (4B and up, factor 8 linear on the
  global layers), `key_length != 256` (the 27B's scale is `hidden / heads`),
  a different `sliding_window_pattern`, MoE keys, an untied head, partial
  rotary width, a different epsilon. The flow itself refuses a Gemma 3 arch
  carrying any Gemma 4 feature.
- **Q5_0.** The 1B Q4_K_M carries 117 Q5_0 tensors, which have no kernel; they
  dequantize to BF16 at repack (the Qwen4Exp precedent). The install is
  1.3 GiB against the 0.77 GiB file. The manifest slot probing one is written
  as raw BF16.
- **Chunked prefill.** The dense driver's FFN GEMMs exist only for affine
  dtypes, so a Gemma 3 GGUF install reports `supports_chunked_prefill` false
  and prefills token by token (`gemma3_chunked_refusal`); the driver refuses
  by name if called anyway. A 1,026-token prompt prefills in about 22 s.

### Real gate (isolated home, deleted afterwards)

Q4_K_M and Q8_0 pulled with `--sidecar-repo mlx-community/gemma-3-1b-it-bf16`;
`turbospark run ... --temperature 0` through the chat template. Both answer
the capital of France, `17 * 23 = 391`, the first five primes and a French
translation coherently and stop on `<end_of_turn>`; the two differ in wording
on open-ended prompts, as quantizations do. Both recover a code word planted
1,000 tokens earlier (past the 512 window). Decode 40-65 tok/s. Mutation
checks: dropping the centered kernel gives `<unused62>` forever; applying the
V norm gives different but fluent text. Outputs are in the shared notes
folder, not the repo.

### Not verified

No logit, KL or perplexity comparison against llama.cpp or transformers; the
reference copy (`mlx-community`, a conversion) is not proven equal to the gated
original; no MLX/HF install path (it would need the `+1` bake or a centered
read of the source, and a refusal is in place); no 270M, 4B, 12B or 27B (4B+
needs linear rope scaling); the Q5_0 to BF16 transcode is lossy at BF16
rounding and its effect was not measured; no vision, tool calls, steering,
speculation or lab measurement; the long-prompt check is one needle, not a
quality gate.
