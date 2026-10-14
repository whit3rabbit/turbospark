# BERT / XLM-RoBERTa (Snowflake arctic-embed)

Portable f32 CPU port of the Snowflake arctic-embed sentence encoders: a
post-LN BERT or XLM-RoBERTa encoder, sentence-transformers CLS pooling,
and L2 normalization. One encoder covers `model_type` `bert` (arctic-embed
v1 and v1.5) and `xlm-roberta` (arctic-embed-l-v2.0 and its mlx-community
conversions). Matryoshka truncation is supported.

## Pins

| Repository | Revision | Arch | Layers x hidden | Vocab | Prompts (`query`) | max_seq_length |
|---|---|---|---|---|---|---|
| [Snowflake/snowflake-arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0/tree/ac6544c8a46e00af67e330e85a9028c66b8cfd9a) | `ac6544c8a46e00af67e330e85a9028c66b8cfd9a` | XLMRobertaModel | 24 x 1024 (16 heads) | 250002 | `query: ` | 8192 (no sentence_bert_config, see below) |
| [Snowflake/snowflake-arctic-embed-m-v1.5](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5/tree/e58a8f756156a1293d763f17e3aae643474e9b8a) | `e58a8f756156a1293d763f17e3aae643474e9b8a` | BertModel | 12 x 768 | 30522 | `Represent this sentence for searching relevant passages: ` | 512 |
| [Snowflake/snowflake-arctic-embed-l](https://huggingface.co/Snowflake/snowflake-arctic-embed-l/tree/d8fb21ca8d905d2832ee8b96c894d3298964346b) | `d8fb21ca8d905d2832ee8b96c894d3298964346b` | BertModel | 24 x 1024 | 30522 | same as m-v1.5 | 512 |
| [Snowflake/snowflake-arctic-embed-m](https://huggingface.co/Snowflake/snowflake-arctic-embed-m/tree/fc74610d18462d218e312aa986ec5c8a75a98152) | `fc74610d18462d218e312aa986ec5c8a75a98152` | BertModel | 12 x 768 | 30522 | same as m-v1.5 | 512 |
| [Snowflake/snowflake-arctic-embed-s](https://huggingface.co/Snowflake/snowflake-arctic-embed-s/tree/e596f507467533e48a2e17c007f0e1dacc837b33) | `e596f507467533e48a2e17c007f0e1dacc837b33` | BertModel | 12 x 384 | 30522 | same as m-v1.5 | 512 |
| [Snowflake/snowflake-arctic-embed-xs](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs/tree/d8c86521100d3556476a063fc2342036d45c106f) | `d8c86521100d3556476a063fc2342036d45c106f` | BertModel | 6 x 384 | 30522 | same as m-v1.5 | 512 |
| [mlx-community/snowflake-arctic-embed-l-v2.0-bf16](https://huggingface.co/mlx-community/snowflake-arctic-embed-l-v2.0-bf16/tree/3da10e38e7875d6501adc14665c95b6708646121) | `3da10e38e7875d6501adc14665c95b6708646121` | l-v2.0, BF16 | | | | 8192 |
| [mlx-community/snowflake-arctic-embed-l-v2.0-8bit](https://huggingface.co/mlx-community/snowflake-arctic-embed-l-v2.0-8bit/tree/98033f618b7503c73aa857bc0656f763a1587e73) | `98033f618b7503c73aa857bc0656f763a1587e73` | l-v2.0, affine 8-bit g64 | | | | 8192 |
| [mlx-community/snowflake-arctic-embed-l-v2.0-6bit](https://huggingface.co/mlx-community/snowflake-arctic-embed-l-v2.0-6bit/tree/c8125c01b2142f0065f2494c123a7df46f9ef08e) | `c8125c01b2142f0065f2494c123a7df46f9ef08e` | l-v2.0, affine 6-bit g64 | | | | 8192 |
| [mlx-community/snowflake-arctic-embed-l-v2.0-4bit](https://huggingface.co/mlx-community/snowflake-arctic-embed-l-v2.0-4bit/tree/430b52bb3dcb9ad2404b1ac2b60f2fdc00f26033) | `430b52bb3dcb9ad2404b1ac2b60f2fdc00f26033` | l-v2.0, affine 4-bit g64 | | | | 8192 |

Every pin uses CLS pooling then Normalize. The output width is the
hidden size: 1024 (l-v2.0, l), 768 (m, m-v1.5), 384 (s, xs). m-v1.5 and
l-v2.0 advertise `matryoshka_dimensions: [256]` in `config.json`. All
prompt tables also get sentence-transformers' default empty `document`
prompt, and `default_prompt_name` is null, so `embed(text)` uses no prompt.

Provenance: the four MLX repos are conversions of l-v2.0 (model cards).
The stored codes confirm it: each code sits a mean 0.248 to 0.252
quantization steps from `round((w - bias) / scale)` of the upstream
weight (word and position tables, layer 0 query, layer 23 output dense,
from `real_cases.json` `mlx.*.code_offsets_vs_upstream`), which is what a fresh
quantization of those weights gives. The quantized repos store F16 scales,
biases, LayerNorms, and linear biases. The bf16 repo stores BF16.

Numeric reference: transformers 4.57.1 (`models/bert/modeling_bert.py`,
`models/xlm_roberta/modeling_xlm_roberta.py`, `activations.py`) and
sentence-transformers 5.1.2 (`models/Transformer.py`,
`SentenceTransformer.py`), torch, f32 on CPU, default sdpa attention. The
full pin table for every family is `src/pins.rs`.

## Architecture

- Embeddings: `LayerNorm((word[id] + token_type[0]) + position[pos])`
  (the reference's summation order). Token type row 0: the BERT tokenizer
  emits all-zero `token_type_ids` for one segment, and XLM-R uses the zero
  buffer.
- Positions: BERT `0..n`. XLM-R `create_position_ids_from_input_ids`:
  `cumsum(id != pad) * (id != pad) + pad`, so real tokens count from
  `pad + 1 = 2` and a pad id keeps position 1 without advancing. 8194
  position rows therefore hold 8192 tokens.
- Layers (post-LN): `a = LN(attn_out(mha(x)) + x)`,
  `out = LN(fc2(gelu(fc1(a))) + a)`. Biased q/k/v/out projections, scale
  `1/sqrt(head_dim)` (sdpa default), bidirectional. A padding mask is the
  additive `finfo(f32).min` key mask transformers builds for sdpa.
- GELU: `hidden_act = "gelu"` is `ACT2FN["gelu"]` = `GELUActivation` =
  `torch.nn.functional.gelu`, the exact erf form, not tanh. The port uses
  the shared `ops::gelu_erf` (Abramowitz-Stegun erf, abs error < 1.5e-7).
  The measured parity below includes that approximation.
- Pooling: CLS (`last_hidden_state[:, 0]`). `pooler.*` weights in the
  checkpoint are ignored, as sentence-transformers ignores them.
- Text: `prompt + text`, Python `str.strip()`, tokenized with the
  post-processor's specials (`[CLS]`/`[SEP]` or `<s>`/`</s>`), content cut
  from the end so the whole fits `max_seq_length` with the specials kept.

## max_seq_length

sentence-transformers 5.1.2 `Transformer.load_config` reads the first of
`sentence_bert_config.json`, `sentence_roberta_config.json`, ...,
`sentence_xlm-roberta_config.json`, `sentence_xlnet_config.json` that
exists. When none sets `max_seq_length`, `Transformer.__init__` uses
`min(config.max_position_embeddings, tokenizer.model_max_length)`.
l-v2.0 ships none of these files, and its `tokenizer_config.json` has
`model_max_length: 8192`, so the limit is `min(8194, 8192) = 8192`
(confirmed: `SentenceTransformer(...).max_seq_length == 8192`).

The `max_length: 512` key in that file is an unused init kwarg: a 14004-token
text is cut to exactly 8192 ids by sentence-transformers and by this port.
An absent `model_max_length` means transformers' effectively infinite
default, so the position table decides. The v1/v1.5 BERT checkpoints set
512 in `sentence_bert_config.json`. The MLX repos ship the same
`tokenizer_config.json` limit and no sentence config, so they also get
8192.

## MLX layout and pooling policy

The mlx-community repos use the upstream tensor names (no prefix), with
every 2-D tensor quantized: the word table (kept packed, dequantized per
gathered row), the position table (likewise), the 1-row token-type table,
all linears, and the unused pooler. They ship `modules.json` naming
`1_Pooling` and `2_Normalize`, but no `1_Pooling/config.json`.

Policy (`BertEmbedder::load`):

1. `{Pooling path}/config.json` present: use it (`Layout::SentenceTransformers`).
2. Absent (`Layout::Mlx`): use the pooling of the pinned upstream the
   conversion came from, but only when `config.json` matches that upstream
   field for field (arch, vocab, hidden, layers, heads, intermediate,
   positions, token types, pad id) and `config_sentence_transformers.json`
   has exactly its prompt table. Today that is l-v2.0 (CLS). The match is
   a fingerprint, not proof of origin. `pooling_source()` reports
   `ConvertedFrom("Snowflake/snowflake-arctic-embed-l-v2.0")`.
3. Otherwise refuse with `Unsupported`. `BertEmbedder::load_with_pooling`
   takes an explicit `PoolingConfig` instead (`PoolingSource::Caller`).

`modules.json` itself is required, and must read Transformer, Pooling,
optional Normalize: sentence-transformers would invent mean pooling for a
directory without it, which is wrong for every arctic checkpoint.

## Public API

`BertEmbedder::load(dir)`, `load_with_pooling(dir, pooling)`,
`embed(text)`, `embed_with_prompt(name, text)`, `embed_query(text)`,
`embed_document(text)`, `embed_batch(texts, prompt_name)`,
`embed_ids(ids)`, `tokenize(text, prompt_name)`,
`with_truncate_dim(Some(256))`, `dimension()`, `full_dimension()`,
`prompt_names()`, `prompt(name)`, `layout()`, `pooling()`,
`pooling_source()`, `max_seq_length()`, `matryoshka_dimensions()`,
`config()`, `encoder()`. `Encoder::forward_traced(ids, mask, trace)` runs
the encoder under an optional padding mask.

## Token tables

An F32 or BF16 table is gathered row by row from the mapped safetensors
file (`Table::Mapped`), like the SigLIP2 text tower, so l-v2.0's
250002 x 1024 word table (1.02 GB as f32) and its 8194 x 1024 position
table are never copied to the heap. MLX-quantized tables stay packed
(`EmbeddingTable::Quantized`). Only an F16 table (none of the pins) is
widened at load.

## Verified stages

All numbers from this checkout on an Apple M4 Max, release build. "abs"
is the max elementwise difference, "cos" the min cosine over cases.

| Stage | Evidence | Result |
|---|---|---|
| Tiny BERT (default suite) | 2 layers, 5 sequences of 3 to 24 tokens: max positions, pad id 0 inside with mask 1, right padding with mask 0, every hidden state and the embedding vs HF | worst hidden 1.4e-6, embedding 1.5e-7 |
| Tiny XLM-R (default suite) | 5 sequences: pads inside with mask 1 (positions 2,1,3,1,1,4,5), right padding with mask 0, 24 real tokens reaching position row 25 | worst hidden 1.7e-6, embedding 1.5e-7 |
| Tiny XLM-R MLX 4-bit | every 2-D tensor quantized (tables included), F16 rest, vs HF on numpy-dequantized weights | worst hidden 1.9e-6, embedding 1.2e-7 |
| Tiny XLM-R MLX 6-bit | as above, codes crossing 32-bit words | worst hidden 1.7e-6, embedding 1.5e-7 |
| Tiny XLM-R MLX BF16 | BF16 everywhere, mapped BF16 tables | worst hidden 1.7e-6, embedding 2.1e-7 |
| Tokenizer | ids vs `SentenceTransformer.tokenize` for 6 texts on all 6 upstreams and the 4 MLX repos (whose `tokenizer.json` is a newer serialization of the same Metaspace pre-tokenizer), plus one 14004-token text on l-v2.0 | exact, and the long text cuts to 8192 |
| l-v2.0 | 6 embeddings vs `encode` (13 to 952 tokens) | 2.5e-7 abs, 1 - cos 2.4e-12 |
| l-v2.0 trace (case 0) | embedding / layer 0 / last hidden | 3.3e-6 (max ref 6.8) / 9.5e-6 (22.6) / 7.6e-6 (2.6) |
| l-v2.0 Matryoshka 256 | vs `encode(truncate_dim=256, normalize_embeddings=True)` | 2.9e-7 abs |
| m-v1.5 | 6 embeddings (12 to 512 tokens, case 5 truncated) | 1.3e-7 abs, 1 - cos 5.1e-13 |
| m-v1.5 Matryoshka 256 | as above | 1.4e-7 abs |
| l | 6 embeddings | 2.9e-7 abs, 1 - cos 3.2e-12 |
| m | 6 embeddings | 1.5e-7 abs, 1 - cos 7.3e-13 |
| s | 6 embeddings | 1.7e-7 abs, 1 - cos 4.6e-13 |
| xs | 6 embeddings | 1.2e-7 abs, 1 - cos 2.0e-13 |
| xs trace (case 0) | embedding / layer 0 / last hidden | 1.4e-6 (max ref 6.3) / 1.5e-6 (7.7) / 1.5e-6 (3.3) |
| MLX bf16, same weights | vs HF pipeline on the BF16 weights widened to f32 | 2.6e-7 abs, 1 - cos 2.1e-12 |
| MLX 8-bit, same weights | vs HF pipeline on numpy-dequantized weights | 2.2e-7 abs, 1 - cos 1.9e-12 |
| MLX 6-bit, same weights | as above | 2.6e-7 abs, 1 - cos 2.6e-12 |
| MLX 4-bit, same weights | as above | 2.2e-7 abs, 1 - cos 2.0e-12 |
| MLX bf16 vs upstream | conversion error | 1.3e-3 abs, cos 0.999953 |
| MLX 8-bit vs upstream | conversion error | 3.3e-3 abs, cos 0.99948 (per case 0.99948 to 0.99973) |
| MLX 6-bit vs upstream | conversion error | 1.4e-2 abs, cos 0.99139 (per case 0.9914 to 0.9965) |
| MLX 4-bit vs upstream | conversion error | 4.8e-2 abs, cos 0.90167 (per case 0.9017 to 0.9478) |

Tolerances in `tests.rs`: tiny hidden 2e-5 and embedding 2e-6 (about 10x
the measured worst. The erf approximation and f32 reduction order are the
only sources). Real: 3e-6 abs and cosine > 1 - 1e-10 for every upstream
and for every MLX repo against its same-weights golden (about 10x the
worst abs, 30x the worst 1 - cos). MLX vs upstream floors bound the
conversion, not the port (max abs, min cos): bf16 5e-3 / 0.9998, 8-bit
1e-2 / 0.998, 6-bit 5e-2 / 0.97, 4-bit 1.5e-1 / 0.85. The 4-bit floor
keeps only 1.5x on 1 - cos.

Mutation checks. Each was applied, the default suite run, then the file
restored and compared byte for byte with `cmp`.

| Mutation | Failing default tests |
|---|---|
| XLM-R position offset (`count + pad` to `count - 1 + pad`) | 8: tiny XLM-R, q4, q6, bf16 parity, both position unit tests, the fixture guard, the id-bound test |
| CLS replaced by mean pooling in the head | 5: every tiny parity test (embedding) |
| LayerNorm moved before the attention residual add (pre-LN) | 5: every tiny parity test (hidden states) |

Memory: the l-v2.0 upstream test (load plus 6 embeddings up to 952
tokens) peaks at a 1.56 GB footprint (`/usr/bin/time -l`). The dense
linears alone are 1.21 GB, so the 1.02 GB word table is not on the heap.
Resident size reaches 2.83 GB because touched file-backed pages count.

Timing (release, M4 Max CPU, f32, best of 3 after a warm-up). The host
was loaded, so treat these as upper bounds:

| Checkpoint | Load | 17 / 13 tokens (short query) | 512 tokens |
|---|---|---|---|
| xs | 0.08 to 0.10 s | 9.0 to 17.1 ms | 177 to 286 ms |
| m-v1.5 | 0.77 to 0.89 s | 51 to 83 ms | 1.23 to 1.76 s |
| l-v2.0 | 3.1 to 4.9 s | 152 to 240 ms | 3.41 to 4.97 s |

Ranges span two runs at load averages of about 14 and 20.

Not optimized: one unpadded sequence per call, no Metal.

## Known divergences and limits

- `ops::gelu_erf` approximates erf (abs error < 1.5e-7) where torch uses
  the exact erf. Included in every measurement above.
- MLX checkpoints run in f32 over dequantized or widened weights, not in
  MLX's F16/BF16 arithmetic. MLX runtime output was not measured.
- `embed_batch` runs each text unpadded. sentence-transformers pads a
  batch, and the goldens use `batch_size=1`. The padded path is covered only
  by the tiny models (`Encoder::forward_traced` with a mask).
- Refused rather than approximated: `model_type` other than `bert` and
  `xlm-roberta`, non-absolute `position_embedding_type`, `is_decoder`,
  `add_cross_attention`, `hidden_act` other than `gelu` (the tanh
  variants are unverified), prefixed (`bert.` / `roberta.`) tensor names,
  module lists other than Transformer, Pooling[, Normalize] (no Dense),
  `include_prompt = false`, `do_lower_case = true`, quantization other
  than affine 4/6/8-bit group 64, and per-layer quantization overrides.
- Only the l-v2.0 Matryoshka width 256 and m-v1.5's 256 are checked.
  Other truncations are plain prefix-and-renormalize.
- Python tokenizers vs Rust `tokenizers` 0.20: ids agree on the fixture
  texts only.
- The 8192-token limit is checked by tokenization only. The longest
  forward checked is 952 tokens.

## Fixtures and regeneration

`crates/embeddings/testdata/bert/` (about 1.9 MB): `tiny_bert/`,
`tiny_xlmr/`, `tiny_xlmr_mlx_{q4,q6,bf16}/` (checkpoints),
`tiny_cases.json`, `tiny_{bert,xlmr,q4,q6,bf16}_{hidden,embed}_*.npy`,
`real_cases.json` (texts, prompt names, per-checkpoint ids, prompts,
max_seq_length, the l-v2.0 8192 cut, MLX dtypes and code offsets),
`real_{tag}_embed.npy`, `real_{m_v15,l_v2}_mrl256.npy`,
`real_{l_v2,xs}_trace_case0.npy`, `real_mlx_{bf16,8bit,6bit,4bit}_embed.npy`,
and `manifest.json` (generator, package versions, pins, sha256 of every
file).

```sh
# venv: transformers 4.57.1, sentence-transformers 5.1.2 (see manifest.json)
PY="emb-venv/bin/python -I"
M=~/models/embeddings
$PY crates/embeddings/tools/gen_bert_fixtures.py --tiny \
    --real l_v2 $M/snowflake-arctic-embed-l-v2.0 \
    --real m_v15 $M/snowflake-arctic-embed-m-v1.5 \
    --real l $M/snowflake-arctic-embed-l --real m $M/snowflake-arctic-embed-m \
    --real s $M/snowflake-arctic-embed-s --real xs $M/snowflake-arctic-embed-xs \
    --mlx bf16 $M/mlx-snowflake-arctic-embed-l-v2.0-bf16 \
    --mlx 8bit $M/mlx-snowflake-arctic-embed-l-v2.0-8bit \
    --mlx 6bit $M/mlx-snowflake-arctic-embed-l-v2.0-6bit \
    --mlx 4bit $M/mlx-snowflake-arctic-embed-l-v2.0-4bit
```

Tests:

```sh
cargo test -p turbospark-embeddings --no-default-features --features bert
# real checkpoints (dirs default to ~/models/embeddings/<name>):
#   TURBOSPARK_ARCTIC_{L_V2,M_V15,L,M,S,XS}_DIR,
#   TURBOSPARK_ARCTIC_L_V2_MLX_{BF16,8BIT,6BIT,4BIT}_DIR
cargo test --release -p turbospark-embeddings --no-default-features \
    --features bert bert:: -- --ignored --nocapture --test-threads 1
```

Not covered by any gate here: retrieval quality, Metal execution,
catalog/FFI/Swift integration.
