# Nomic BERT (snowflake-arctic-embed-m-long)

Portable f32 CPU port of `Snowflake/snowflake-arctic-embed-m-long`: the
Nomic BERT encoder, a 12-layer post-LN BERT with rotary positions
(dynamic NTK scaling past 2048 tokens), a packed bias-free qkv
projection, and a SwiGLU MLP, then sentence-transformers CLS pooling and
L2 normalization. Output is a 768-dim unit vector. Inputs run up to 8192
tokens.

## Pins and references

| Role | Repository | Revision |
|---|---|---|
| Checkpoint | [Snowflake/snowflake-arctic-embed-m-long](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long/tree/92d97331f1f4b6a366c1f161354b9f3390cc219f) | `92d97331f1f4b6a366c1f161354b9f3390cc219f` |

No MLX conversion exists. The weights are F32 safetensors, and no pickle
is read.

Numeric reference: the remote code pinned with the checkpoint, sha256 in
`testdata/nomic_bert/manifest.json`:

- `modeling_hf_nomic_bert.py` (`90570ba7...`), cited below as `M:line`
- `configuration_hf_nomic_bert.py` (`8b42dd8e...`), cited as `C:line`

run under transformers 4.57.1, sentence-transformers 5.1.2, torch 2.14.1,
einops 0.8.1, f32 on CPU, imported only from the local pinned directory
with `trust_remote_code=True` and `HF_HUB_OFFLINE=1`. The module-cache
copies were byte-identical to the pinned files.

## Architecture (from the pinned code)

- `NomicBertConfig` extends `GPT2Config` (C:4), so the widths are
  `n_embd` 768, `n_head` 12, `n_inner` 3072, `n_layer` 12,
  `n_positions` 8192.
- Embeddings: `word_embeddings[id] + token_type_embeddings[0]`, then
  `emb_ln` (eps 1e-12). There is no position table because
  `rotary_emb_fraction > 0` (M:385-391). `token_type_ids` default to
  zeros (M:1063-1064), and BERT single-segment tokenization yields zeros
  too. The vocabulary is padded up to a multiple of
  `pad_vocab_size_multiple` (64) before the table is built
  (M:1033-1035). 30528 already is one.
- Attention: `Wqkv` (768 -> 2304, `qkv_proj_bias = false`) rearranged
  `"... (three h d) -> ... three h d"` (M:774-775), so the thirds are q,
  k, v. `out_proj` has no bias (M:749). Scores are divided by
  `norm_factor = sqrt(64)` in f32 (M:719-723, 793). Exact as a multiply
  by 1/8.
- Rotary: `rotary_emb_fraction = 1.0` rotates all 64 dims (M:725).
  `rotary_emb_interleaved = false` takes `x.chunk(2)` and
  `repeat "... d -> ... 1 (2 d)"` (M:483-508): the NeoX half-split
  layout, `ops::rope_neox`, on q and k only (M:619-621). Base 1000,
  `inv_freq = 1 / base^(arange(0, 64, 2) / 64)` in f32 (M:630-633).
- Dynamic NTK (`rotary_scaling_factor = 2`, so the NTK class is chosen,
  M:727-735, with `max_trained_positions = 2048`). For a sequence of `n`
  tokens with `n > 2048` the base becomes
  `1000 * (2 * n / 2048 - 1) ^ (64 / 62)` (M:639-644), and the cos/sin
  table is built from that base. `n <= 2048` keeps base 1000. At
  `n = 2048` the formula gives exactly base 1000, so the boundary is
  continuous. Examples: n = 3000 gives 1971.04, n = 8192 gives 7453.48.
- Block, post-LN (`prenorm = false`, M:888-901):
  `x = norm1(x + attn(x))`, `x = norm2(x + mlp(x))`.
- MLP (`NomciBertGatedMLP`, M:446-480): `activation_function = "swiglu"`
  selects `F.silu` (M:818-822). `y = fc11(x)`, `gate = fc12(x)`,
  `fc2(y * silu(gate))` (M:472-478), so fc11 is the value and fc12 the
  gate. No biases (`mlp_fc1_bias`, `mlp_fc2_bias` false). The hidden
  width is `n_inner` rounded up to a multiple of 256 (M:463-464), 3072
  here.
- Pipeline (`modules.json`): Transformer, Pooling (CLS), Normalize.
  Prompts: `query` = `"Represent this sentence for searching relevant
  passages: "`, `default_prompt_name` null. `sentence_bert_config.json`
  sets `max_seq_length` 8192. Text is stripped and lowercased by the
  BERT normalizer (accents stripped), with `[CLS]` (101) and `[SEP]`
  (102) kept on truncation.

## Public API

`NomicBert::load(dir)`, `embed(text)`, `embed_with_prompt("query", text)`,
`embed_batch(texts, prompt_name)`, `embed_ids(ids)`,
`tokenize(text, prompt_name)`, `with_truncate_dim(Some(d))`,
`dimension()`, `full_dimension()`, `prompt_names()`, `prompt(name)`,
`max_seq_length()`, `config()`, `encoder()` (with `forward_traced` and
`inv_freq_for(n)`). The 94 MB word table is copied to the heap. The
checkpoint was not trained for Matryoshka truncation, so
`with_truncate_dim` exists for API parity only.

## Verified stages

Release build, Apple M4 Max, this checkout.

| Stage | Evidence | Result |
|---|---|---|
| Tiny model (default suite) | 3 layers, n_embd 64, 4 heads, n_inner 200 (rounds to 256), vocab 60 (pads to 64), max_trained_positions 16, factor 2, built from the remote classes. 6 sequences of 2, 4, 16, 17, 40, 64 tokens, fresh model per sequence, every state and the embedding | worst hidden 2.9e-6, worst embedding 2.2e-7 abs |
| Tiny rotary inv_freq | vs the reference buffer after each forward | within 2 ulps |
| Real rotary | inv_freq and cos/sin rows 0, 1, 7, 1000, 2047 at n = 2048, 2049, 3000, 8192 | inv_freq within 2 ulps, cos/sin worst 7.2e-6 abs |
| Tokenizer (default suite, pinned tokenizer.json copied into testdata) | ids vs `SentenceTransformer.tokenize` for 7 texts (query prompt, whitespace, accents and uppercase, Japanese, 522, 3000, and a text cut to 8192) | exact |
| Real checkpoint | 6 embeddings vs `SentenceTransformer.encode` (12 to 3000 tokens, one past 2048 for NTK) | worst 3.9e-7 abs (3000 tokens), min cosine 1 - 4.1e-12 |
| Real trace (case 0) | emb_ln / layer 0 / last hidden | 2.4e-6 / 7.6e-6 / 2.5e-5 |

Tolerances in `tests.rs`: tiny hidden 1e-5 and embedding 2e-6 (3.5x and
9x the measured worst). Real embeddings 3e-6 abs and cosine > 1 - 1e-9
(about 8x and 100x).

The inv_freq gap is torch's f32 `pow`, which is within 1 ulp of the
correctly rounded value. The port evaluates the power in f64 and rounds
once, so a non-representable scaled base can land 1 ulp off before the
reciprocal and 2 ulps after. Base 1000 (unscaled) matches exactly.

Mutation checks (each applied to `encoder.rs`, the tiny suite run, the
file restored and compared byte-equal):

| Mutation | Result |
|---|---|
| `rope_neox` -> `rope_interleaved` | tiny parity fails at case 0, layer 0, 0.39 |
| NTK rescale off (always base 1000) | tiny parity fails at case 3 (17 tokens), layer 0, 0.015. inv_freq test fails. On the real checkpoint the 3000-token case drops to cosine 0.549 (max abs 0.52), all shorter cases unchanged |
| NTK boundary `n > M` -> `n >= M` | survives. Equivalent mutant: the scaled base equals the plain one at `n = M` |
| SwiGLU silu on fc11 instead of fc12 | tiny parity fails at case 0, layer 0, 1.49 |

Timing (release, CPU, real checkpoint, host loaded: load average 14 to
33 on 16 cores during the runs): load 1.3 to 1.8 s, 19 tokens 53 to
217 ms, 522 tokens 1.4 to 2.3 s, 3000 tokens 15.4 s.

## Known divergences and limits

- Stateful reference rotary. The NTK class replaces its `inv_freq`
  buffer when a forward is longer than 2048 and rebuilds the cos/sin
  cache only when a forward is longer than the cached length
  (M:639-652). A shorter text after a long one then reuses the scaled
  cache. Each layer has its own instance. The port is stateless: every
  call computes what a freshly loaded reference computes for that one
  sequence, and the goldens were made that way (a deep copy of the
  pristine model per text). Measured drift of a reused reference: on
  the real checkpoint, the query case after the 3000-token case moves by
  max abs 7.9e-3, cosine 0.99775. On the tiny model, 0.23 max abs.
  sentence-transformers also sorts a batch longest-first and pads it, so
  in a mixed batch every text gets the NTK base of the padded length.
  The port does not reproduce either effect.
- `NomicBertPreTrainedModel.from_pretrained` reads `pytorch_model.bin`
  first when present (M:330-338) and loads with `strict=False` (M:342),
  logging instead of failing on missing keys. The generator refuses a
  directory with a pickle and checks every loaded tensor against the
  safetensors file. The custom loader also does not call `eval()`, so a
  direct `AutoModel` user keeps dropout on. `SentenceTransformer.encode`
  calls `eval()` (`SentenceTransformer.py:962`).
- transformers 4.57.1 ran the remote code without errors or workarounds.
- `embed_batch` runs each text unpadded. The goldens use `batch_size=1`.
- `embed_ids` refuses more than `n_positions` (8192) ids.
- Refused rather than approximated: activations other than swiglu,
  `rotary_emb_fraction != 1`, interleaved rotary, `rotary_emb_scale_base`
  (xPos), `prenorm`, any of the three bias flags, `causal`,
  `parallel_block`, `use_rms_norm`, `num_heads_kv`, `type_vocab_size = 0`,
  `include_prompt = false`, `do_lower_case = true`, a Dense module, and
  an absolute position table in the checkpoint. `causal`,
  `parallel_block`, and `use_rms_norm` are stored but never read by the
  pinned code. They are refused because a checkpoint that sets them was
  not trained for this computation. `rotary_scaling_factor` null or 0
  disables NTK, and absent means the class default 1.0, which still
  rescales past `max_trained_positions`.
- Python tokenizers 0.22.2 vs Rust `tokenizers` 0.20: ids agree on the 7
  fixture texts only.

## Fixtures and regeneration

`crates/embeddings/testdata/nomic_bert/` (about 2.1 MB, 0.7 MB of it the
pinned `tokenizer.json`): `tiny/`, `tiny_cases.json` (ids, config,
recorded reference drift), `tiny_{hidden,embed}_*.npy`,
`tiny_rotary_inv_freq_*.npy`, `real_cases.json`, `real_embed.npy`,
`real_trace_case0.npy`, `real_rotary_{inv_freq,cos_rows,sin_rows}.npy`,
`tokenizer.json`, and `manifest.json` (package versions, pin,
remote-code sha256, sha256 per file). A full rerun reproduced every file
byte for byte.

```sh
PY="emb-venv/bin/python -I"   # torch 2.14.1, transformers 4.57.1, sentence-transformers 5.1.2, einops 0.8.1
M=~/models/embeddings/snowflake-arctic-embed-m-long
$PY crates/embeddings/tools/gen_nomic_bert_fixtures.py --tiny $M --real $M
cargo test -p turbospark-embeddings --no-default-features --features nomic-bert
# real checkpoint: TURBOSPARK_NOMIC_BERT_DIR, default ~/models/embeddings/snowflake-arctic-embed-m-long
cargo test --release -p turbospark-embeddings --no-default-features --features nomic-bert \
    nomic_bert -- --ignored --nocapture --test-threads 1
```

Not covered: retrieval quality, Metal, catalog/FFI/Swift integration,
padded batches, and the reference's stateful or batch-padded NTK
behavior.
