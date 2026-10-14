# EmbeddingGemma

Portable f32 CPU port of `google/embeddinggemma-300m`: a 24-layer Gemma3
text backbone run as a bidirectional encoder, mean pooling, a bias-free
Dense head (768 -> 3072 -> 768), and L2 normalization. Output is a
768-dim unit vector. Matryoshka truncation to 512/256/128 is supported.

## Pins and references

| Role | Repository | Revision |
|---|---|---|
| Upstream checkpoint (gated) | [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m/tree/57c266a740f537b4dc058e1b0cda161fd15afa75) | `57c266a740f537b4dc058e1b0cda161fd15afa75` |
| MLX F16 ("bf16") | [mlx-community/embeddinggemma-300m-bf16](https://huggingface.co/mlx-community/embeddinggemma-300m-bf16/tree/2f420efb007317a1ef2aa1608f87ef24226e6807) | `2f420efb007317a1ef2aa1608f87ef24226e6807` |
| MLX 8-bit | [mlx-community/embeddinggemma-300m-8bit](https://huggingface.co/mlx-community/embeddinggemma-300m-8bit/tree/c8316c9e35cf8830541b04d315398c67faa6e497) | `c8316c9e35cf8830541b04d315398c67faa6e497` |
| MLX 4-bit | [mlx-community/embeddinggemma-300m-4bit](https://huggingface.co/mlx-community/embeddinggemma-300m-4bit/tree/5d9ef074df3957afc5c77127f208fddbc3c54187) | `5d9ef074df3957afc5c77127f208fddbc3c54187` |
| QAT q8_0, unquantized (gated) | [google/embeddinggemma-300m-qat-q8_0-unquantized](https://huggingface.co/google/embeddinggemma-300m-qat-q8_0-unquantized/tree/7b5b24595322ab0ea4d08827066860a6df8cb0aa) | `7b5b24595322ab0ea4d08827066860a6df8cb0aa` |
| QAT q4_0, unquantized (gated) | [google/embeddinggemma-300m-qat-q4_0-unquantized](https://huggingface.co/google/embeddinggemma-300m-qat-q4_0-unquantized/tree/7740c61c3ad98a9deb8423a20388ab6c3c5c45f2) | `7740c61c3ad98a9deb8423a20388ab6c3c5c45f2` |

Provenance: the MLX F16 repo is converted from upstream. Per the
mlx-community model cards, the 8-bit (and the pinned, unverified 6- and
5-bit) repos come from QAT q8_0 and the 4-bit repo from QAT q4_0. The
stored codes confirm this for 8-bit and 4-bit (see Known divergences).
The QAT checkpoints have the same `config.json` and `tokenizer.json` as
upstream.

Numeric reference: transformers 4.57.1 (`models/gemma3/modeling_gemma3.py`,
`masking_utils.py`, `configuration_gemma3.py`) and sentence-transformers
5.1.2, torch 2.14.1, f32 on CPU with the default sdpa attention. Structure
and gotchas were cross-read from the mlxrs `embeddings/embeddinggemma`
port. Numbers come from HF only. The full pin table for every family is
`src/pins.rs`.

## Architecture

- Token embedding gathered, then multiplied by `sqrt(hidden_size)`
  (f32 `27.712812`).
- 24 layers, 4-norm sandwich block:
  `h = x + post_attention_layernorm(attn(input_layernorm(x)))`,
  `out = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))`.
- Gemma RMSNorm `x * rsqrt(mean(x^2) + 1e-6) * (1 + w)`, with `1 + w`
  folded in at load.
- Attention: 3 query heads, 1 kv head, head_dim 256, per-head q/k RMSNorm
  on head_dim, NeoX RoPE, scale `query_pre_attn_scalar ** -0.5` (1/16).
  Full layers (5, 11, 17, 23) use RoPE base 1e6 and see every token.
  Sliding layers use base 1e4 and see keys with `|q - k| < 257` (see
  divergences: the configured 512 is halved by transformers).
- MLP: `down(gelu_tanh(gate(x)) * up(x))`, 768 -> 1152 -> 768.
- Final RMSNorm, mean pooling over all tokens (prompt included), Dense
  768 -> 3072, Dense 3072 -> 768, L2 normalize.
- Text: `prompt + text`, stripped like Python `str.strip()`, tokenized
  with `<bos>` (2) and `<eos>` (1) from the `tokenizer.json` template,
  cut to `max_seq_length` (2048) with the special tokens kept.

## Public API

`EmbeddingGemma::load(dir)` detects the layout from the tensor names:
`embed_tokens.weight` is the sentence-transformers layout (head from
`modules.json`, `2_Dense/`, `3_Dense/`). `model.embed_tokens.weight` is
the MLX layout, with the head as `dense.0`, `dense.1` in the main file. The
MLX repos ship `modules.json` without its directories, so pooling defaults
to mean when `1_Pooling/config.json` is absent, and the Dense count is
checked against `modules.json`. Then `embed(text)`,
`embed_with_prompt(name, text)`, `embed_batch(texts, prompt_name)`,
`embed_ids(ids)`, `tokenize(text, prompt_name)`,
`with_truncate_dim(Some(256))`, `prompt_names()`.

## Verified stages

All numbers from this checkout on an Apple M4 Max, release build.

| Stage | Evidence | Result |
|---|---|---|
| Tiny model, HF layout (default suite) | 3 layers (sliding, full, sliding), band 3, 4 sequences of 3 to 16 tokens, every hidden state and final embedding vs HF | worst hidden 4.3e-6, worst embedding 2.1e-7 abs |
| Tiny model, MLX 4-bit layout (default suite) | same model quantized 4-bit/group 64 (every linear, embedding table, Dense head), vs HF on the dequantized weights | worst hidden 6.0e-6, worst embedding 1.8e-7 abs |
| Tokenizer | ids vs `SentenceTransformer.tokenize` for 7 texts (query/document/STS prompts, padded whitespace, German/Japanese/Russian, 638 tokens, truncated to 2048) | exact |
| Upstream checkpoint | 7 embeddings vs `SentenceTransformer.encode` | worst 3.3e-7 abs, min cosine 1 - 1.4e-12 |
| Upstream trace (case 0) | scaled embedding / layer 0 / last hidden | 0 / 1.2e-4 (max ref 950) / 3.1e-5 (max ref 75) |
| Matryoshka 256 | vs `encode(truncate_dim=256, normalize_embeddings=True)` | 2.5e-7 abs |
| MLX F16, same weights | vs HF pipeline with the conversion's weights widened to f32 | 2.8e-7 abs |
| MLX 8-bit, same weights | vs HF pipeline with weights dequantized in numpy | 4.8e-7 abs |
| MLX 4-bit, same weights | as above | 2.3e-7 abs |
| QAT q8_0 checkpoint | 7 embeddings vs `SentenceTransformer.encode` on it, ids identical to upstream | worst 3.2e-7 abs, min cosine 1 - 2.7e-12 |
| QAT q4_0 checkpoint | as above | worst 2.8e-7 abs, min cosine 1 - 1.2e-12 |
| MLX F16 vs upstream (its source) | conversion error | worst 1.4e-4 abs, min cosine 0.99999944 |
| MLX 8-bit vs QAT q8_0 (its source) | conversion error | worst 3.5e-3 abs, min cosine 0.99965 (per case 0.99965 to 0.99994) |
| MLX 4-bit vs QAT q4_0 (its source) | conversion error | worst 3.8e-2 abs, min cosine 0.94744 (per case 0.9474 to 0.9840) |
| QAT q8_0 vs upstream | QAT drift | worst 2.7e-2 abs, min cosine 0.97633 (per case 0.9763 to 0.9930) |
| QAT q4_0 vs upstream | QAT drift | worst 5.1e-2 abs, min cosine 0.93687 (per case 0.9369 to 0.9844) |
| MLX 8-bit vs upstream | QAT drift + quantization | worst 2.7e-2 abs, min cosine 0.97638 (per case 0.9764 to 0.9929) |
| MLX 4-bit vs upstream | QAT drift + quantization | worst 5.0e-2 abs, min cosine 0.91663 (per case 0.9166 to 0.9754) |

Tolerances in `tests.rs`: tiny hidden 5e-5 and embedding 2e-6 (about 10x
the measured worst). Real embeddings 5e-6 abs and cosine > 0.99999 for
the upstream and both QAT checkpoints, and for every MLX checkpoint
against its same-weights golden (about 10x the measured worst, which is
f32 reduction-order noise). The remaining floors bound the conversion or
QAT, not the port, and are the measured worst with margin (max abs,
min cosine): MLX 8-bit vs QAT q8_0 1e-2 and 0.999, MLX 4-bit vs QAT q4_0
1e-1 and 0.93, and against upstream F16 5e-4 and 0.99999, 8-bit 5e-2 and
0.97, 4-bit 1e-1 and 0.90. QAT vs upstream is reported, not asserted.

Mutation checks (each made the default suite fail, then was restored):
band width set to the unhalved `sliding_window`, band removed, band made
inclusive (`<=`), and RMSNorm scale `w` instead of `1 + w`.

Timing (release, M4 Max, CPU, upstream f32): load 2.7 s, 18 tokens 65 ms,
638 tokens 2.1 s, 2048 tokens 8.6 s. Not optimized: sliding layers still
compute the full score matrix and mask it.

## MLX provenance

The check is the distance, in quantization steps, between each stored
code and `round((w - bias) / scale)` of a candidate source weight, using
the stored F16 scale and bias. For codes quantized from that weight the
mean is 0.25 (uniform rounding error). Over 6 tensors (`embed_tokens`,
`layers.0` q_proj and down_proj, `layers.11` o_proj, `layers.23`
gate_proj, `dense.0`), from `mlx_provenance.json`:

| Codes | Candidate | mean | p99 | max | share > 0.5 step |
|---|---|---|---|---|---|
| 8-bit | QAT q8_0 | 0.2488 to 0.2494 | 0.525 to 0.530 | 1.05 to 1.09 | 2.6% to 2.9% |
| 8-bit | upstream | 0.31 to 2.35 | 1.0 to 8.2 | 4.4 to 29.2 | 18% to 86% |
| 4-bit | QAT q4_0 | 0.249 to 0.267 | 0.498 to 0.499 | 1.002 to 1.004 | 0.5% to 0.7% |
| 4-bit | upstream | 0.249 to 0.327 | 0.51 to 1.03 | 1.14 to 4.11 | 1.6% to 22% |

The QAT rows match a fresh quantization (mean 0.25) and the upstream
rows do not, which confirms the model-card provenance.

The codes are not all within 0.5 step: up to 2.9% sit between 0.5 and
1.09 steps away. A likely, unconfirmed cause is MLX recomputing the
scale (`edge / q0`) and rounding scale and bias to F16 after the codes
were chosen. An
emulation of MLX's affine quantizer from the QAT weights, on 3 tensors,
matched 97.8% to 97.9% of 8-bit and 99.8% of 4-bit codes in f32
arithmetic, and every stored bias but only 89% to 90% of 8-bit codes in
f16 arithmetic, so the exact conversion arithmetic is unverified. The
port is exact against the stored weights either way.

## Known divergences and limits

- Sliding window. transformers 4.57.1 `Gemma3TextConfig.__init__`
  (`models/gemma3/configuration_gemma3.py:229`) sets
  `sliding_window = (sliding_window // 2) + 1` when
  `use_bidirectional_attention` is true, so the shipped 512 becomes a
  257 reach in each direction (`|q - k| < 257`). This port follows that.
  The mlxrs port (and its mlx-embeddings source) use `|q - k| < 512` or
  no band at all, which diverges from HF for inputs over 257 tokens.
  A future transformers release could change the rewrite, so re-check the
  config code when moving the reference pin.
- The repo named "bf16" stores F16 tensors.
- MLX checkpoints are computed in f32 over the dequantized weights, not in
  the F16/BF16 arithmetic MLX itself uses. MLX runtime output was not
  measured (mlx is not installed in the reference venv).
- Matryoshka: `with_truncate_dim` renormalizes after truncating, matching
  sentence-transformers with `normalize_embeddings=True`. Plain
  `encode(truncate_dim=...)` returns the un-renormalized prefix.
- `embed_batch` runs each text unpadded. sentence-transformers pads a
  batch, and the goldens were made with `batch_size=1`, so padded-batch
  numerics are not covered.
- Refused rather than approximated: causal attention, logit softcapping,
  RoPE scaling, activations other than `gelu_pytorch_tanh`, attention
  bias, `include_prompt = false`, `do_lower_case = true`, quantization
  other than affine 4/8-bit group 64, and per-layer quantization
  overrides. The 5- and 6-bit conversions are pinned but stay refused:
  they were not downloaded or measured, and their codes straddle 32-bit
  words, which the numpy reference dequantizer does not implement.
- `sentence-transformers` defaults `use_cache=True`. The generator checks
  that the sliding cache leaves the 638-token prefill unchanged (max diff
  0.0), so the cache-free port matches.
- Python tokenizers 0.22.2 vs Rust `tokenizers` 0.20: ids agree on the
  7 fixture texts only.

## Fixtures and regeneration

`crates/embeddings/testdata/embeddinggemma/` (about 1.2 MB):
`tiny_hf/`, `tiny_mlx_q4/` (checkpoints), `tiny_cases.json`,
`tiny_{hf,q4}_{hidden,embed}_*.npy`, `real_cases.json` (texts, prompt
names, sentence-transformers ids), `real_embed.npy`,
`real_embed_mrl256.npy`, `real_trace_case0.npy`,
`real_mlx_{bf16,8bit,4bit}_embed.npy`, `real_qat_{q8,q4}_embed.npy`,
`mlx_provenance.json`, and `manifest.json` (generator,
package versions, pins, sha256 of every file). Regeneration is
deterministic: a full rerun reproduced every existing file byte for byte.

```sh
# venv: torch 2.14.1, numpy 2.5.4, safetensors 0.8.0, transformers 4.57.1,
# sentence-transformers 5.1.2, tokenizers 0.22.2
PY="emb-venv/bin/python -I"
M=~/models/embeddings
$PY crates/embeddings/tools/gen_embeddinggemma_fixtures.py --tiny \
    --real $M/embeddinggemma-300m \
    --mlx bf16 $M/mlx-embeddinggemma-300m-bf16 \
    --mlx 8bit $M/mlx-embeddinggemma-300m-8bit \
    --mlx 4bit $M/mlx-embeddinggemma-300m-4bit \
    --qat q8 $M/embeddinggemma-300m-qat-q8_0-unquantized \
    --qat q4 $M/embeddinggemma-300m-qat-q4_0-unquantized
```

Tests:

```sh
cargo test -p turbospark-embeddings --no-default-features --features embeddinggemma
# real checkpoints (dirs default to ~/models/embeddings/<name>):
#   TURBOSPARK_EMBEDDINGGEMMA_DIR, TURBOSPARK_EMBEDDINGGEMMA_MLX_BF16_DIR,
#   TURBOSPARK_EMBEDDINGGEMMA_MLX_8BIT_DIR, TURBOSPARK_EMBEDDINGGEMMA_MLX_4BIT_DIR,
#   TURBOSPARK_EMBEDDINGGEMMA_QAT_Q8_DIR, TURBOSPARK_EMBEDDINGGEMMA_QAT_Q4_DIR
cargo test --release -p turbospark-embeddings --no-default-features \
    --features embeddinggemma embeddinggemma -- --ignored --nocapture --test-threads 1
```

Not covered by any gate here: retrieval or STS task quality, Metal
execution, catalog/FFI/Swift integration, and the 5/6-bit and QAT-bf16
MLX pins.
