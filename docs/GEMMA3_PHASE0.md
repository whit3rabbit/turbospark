# Gemma 3 Phase 0 findings (`gemma3` / `gemma3_text`, GGUF `gemma3`)

Fact-finding only; nothing here is implemented. Read 2026-11-14 off the real
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
