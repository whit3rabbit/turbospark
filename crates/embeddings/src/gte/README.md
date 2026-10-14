# GTE (snowflake-arctic-embed-m-v2.0)

Portable f32 CPU port of `Snowflake/snowflake-arctic-embed-m-v2.0`: the
Alibaba GTE ("new") encoder, a 12-layer post-LN BERT with rotary
positions, a packed qkv projection, and a gated GELU MLP, then
sentence-transformers CLS pooling and L2 normalization. Output is a
768-dim unit vector. Matryoshka truncation to 256 is supported. The
vocabulary is the 250048-token XLM-R SentencePiece (multilingual).

## Pins and references

| Role | Repository | Revision |
|---|---|---|
| Checkpoint | [Snowflake/snowflake-arctic-embed-m-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0/tree/95c2741480856aa9666782eb4afe11959938017f) | `95c2741480856aa9666782eb4afe11959938017f` |

No MLX conversion exists. The weights are F32 safetensors, and no pickle is
read.

Numeric reference: the remote code pinned with the checkpoint, sha256 in
`testdata/gte/manifest.json`:

- `modeling_hf_alibaba_nlp_gte.py` (`f8dff057...`), cited below as `M:line`
- `configuration_hf_alibaba_nlp_gte.py` (`d3162291...`), cited as `C:line`

run under transformers 4.57.1, sentence-transformers 5.1.2, torch 2.14.1,
f32 on CPU, imported only from the local pinned directory with
`trust_remote_code=True` and `HF_HUB_OFFLINE=1` (the `auto_map` entries
name local modules only, which the generator asserts). The copies
transformers placed in its module cache were byte-identical to the
pinned files.

## Architecture (from the pinned code)

- Embeddings: `word_embeddings[id] + token_type_embeddings[0]`, then
  LayerNorm (eps 1e-12). No absolute position table when
  `position_embedding_type = "rope"` (M:288-300, 397-414). With
  `type_vocab_size = 1` every token-type id is zeroed (M:401-402).
- Rotary: `inv_freq = 1 / 160000^(arange(0, 64, 2) / 64)` in f32 (M:189),
  cos/sin of `cat(freqs, freqs)` (M:201-205), applied as
  `q * cos + rotate_half(q) * sin` with `rotate_half = cat(-x2, x1)`
  (M:158-179): the NeoX half-split layout, `ops::rope_neox`. Applied to q
  and k only. `rope_scaling = null` selects the plain class (M:317-318).
- Attention: one `qkv_proj` (768 -> 2304, bias) split into thirds in
  q, k, v order (M:439-440, 470, 477). 12 heads of 64, scale
  `1 / sqrt(64)` (M:540, and the sdpa default scale is the same).
  `o_proj` has a bias (M:447).
- Block, post-LN (M:667-695):
  `x = attn_ln(x + attn(x))`, `x = mlp_ln(x + mlp(x))`.
- MLP (`GteGatedMLP`, M:596-620): `up_gate_proj` (768 -> 6144, no bias)
  split as `up, gate = split(..., 3072)`, so the FIRST half is up and the
  SECOND is the gate. Then `down_proj(gelu(gate) * up)` with bias.
  `hidden_act = "gelu"` is `ACT2FN["gelu"]`, the exact erf GELU.
- No final norm. The pooler is not built (`add_pooling_layer=False`,
  M:832, 839), and the checkpoint has no pooler tensors (136 tensors).
- Pipeline (`modules.json`): Transformer, Pooling (CLS), Normalize.
  Prompts: `query` = `"query: "`, `default_prompt_name` null, so
  documents get no prompt.
- `max_seq_length`: no `sentence_bert_config.json`, so
  sentence-transformers 5.1.2 resolves
  `min(max_position_embeddings 8192, tokenizer model_max_length 32768)`
  = 8192 (`models/Transformer.py:108-117`). The text is stripped
  (`Transformer.py:312`) and cut with `<s>` (0) and `</s>` (2) kept.

Flags:

- `unpad_inputs` and `use_memory_efficient_attention` are the string
  `"true"` in the pinned config, so truthy. They choose a packed layout
  and the xformers kernel (M:358-386, 911-925, 493-506), not different
  math. The tiny generator measured `unpad_inputs=True` against the
  padded path: max abs diff 0.0. They are accepted with any value.
- `logn_attention_scale` and `logn_attention_clip1` are false. The pinned
  code only logs a TODO when set (M:927-937), so they would be no-ops in
  the reference. The port refuses them anyway because a checkpoint that
  sets them expects a behavior nobody implements.
- `matryoshka_dimensions = [256]` is informational. `with_truncate_dim`
  renormalizes like `encode(truncate_dim=256, normalize_embeddings=True)`.

## Public API

`Gte::load(dir)`, `embed(text)`, `embed_with_prompt("query", text)`,
`embed_batch(texts, prompt_name)`, `embed_ids(ids)`,
`tokenize(text, prompt_name)`, `with_truncate_dim(Some(256))`,
`dimension()`, `full_dimension()`, `prompt_names()`, `prompt(name)`,
`max_seq_length()`, `config()`, `encoder()` (with `forward_traced`).
`tokenizer.json` is optional. Without it only `embed_ids` works.

The 250048 x 768 F32 word table (768 MB) is read row by row from the
mapped safetensors file instead of copied to the heap (the SigLIP2 text
tower does the same). A non-F32 table is widened at load.

## Verified stages

Release build, Apple M4 Max, this checkout.

| Stage | Evidence | Result |
|---|---|---|
| Tiny model (default suite) | 3 layers, hidden 64, 4 heads, gated MLP 96, rope_theta 160000, built from the remote classes, 5 sequences of 2 to 40 tokens, every state and the embedding | worst hidden 2.9e-6, worst embedding 3.8e-7 abs |
| Reference cross-checks (generator) | eager vs sdpa, and `unpad_inputs=True` vs padded | 2.3e-6 and 0.0 |
| Rotary inv_freq | vs the reference buffer, tiny (dim 16) and real (dim 64) | bit-exact |
| Rotary cos/sin | positions 0, 1, 7, 511, 2047, 8191 vs the reference cache | worst 6.0e-8 abs |
| Tokenizer | ids vs `SentenceTransformer.tokenize` for 9 texts (query prompt, whitespace, German/Japanese/Russian, Spanish, Chinese/Arabic, 535, 2003, and a text cut to 8192) | exact |
| Real checkpoint | 8 embeddings vs `SentenceTransformer.encode` (14 to 2003 tokens) | worst 5.7e-7 abs (2003 tokens), min cosine 1 - 7.8e-12 |
| Real trace (case 0) | embedding LN / layer 0 / last hidden | 1.7e-6 / 8.6e-6 / 5.6e-6 |
| Matryoshka 256 | vs `encode(truncate_dim=256, normalize_embeddings=True)` | 2.9e-7 abs |

Tolerances in `tests.rs`: tiny hidden 1e-5 and embedding 2e-6 (3.4x and
5x the measured worst, f32 reduction order only). Real embeddings 3e-6
abs and cosine > 1 - 1e-9 (about 5x and 100x).

Mutation checks (each applied to `encoder.rs`, the tiny parity test
failed, the file was restored and compared byte-equal):

| Mutation | First failing state | Max abs diff |
|---|---|---|
| `rope_neox` -> `rope_interleaved` (q and k) | case 0, layer 0 | 0.41 |
| qkv split q <-> k | case 0, layer 0 | 1.18 |
| qkv split k <-> v | case 0, layer 0 | 2.71 |
| MLP halves swapped (gate first) | case 0, layer 0 | 1.56 |

Timing (release, CPU, real checkpoint, host loaded: load average 14 to
33 on 16 cores during the runs): load 0.37 to 1.1 s, 14 tokens 48 ms,
535 tokens 1.5 to 1.8 s, 2003 tokens 7.0 s.

## Known divergences and limits

- The reference cannot run on this host as configured: with
  `use_memory_efficient_attention` truthy, `GteAttention.__init__`
  asserts xformers is installed (M:449-454), and xformers is not in the
  venv (it does not build on macOS). The goldens load with
  `config_kwargs={"use_memory_efficient_attention": False,
  "unpad_inputs": False}`, the documented CPU workaround. With those
  flags transformers 4.57.1 picks sdpa attention (`GteSdpaAttention`,
  M:577-586). The eager path agrees to 2.3e-6 on the tiny model. No
  other transformers 4.57.1 incompatibility was seen: every parameter
  loaded is bit-identical to the safetensors file (checked tensor by
  tensor in the generator).
- `embed_batch` runs each text unpadded. sentence-transformers pads a
  batch. The goldens use `batch_size=1`, so padded-batch numerics are
  not covered.
- `embed_ids` refuses more than `max_position_embeddings` (8192) ids.
  The reference would extend its cos/sin cache (M:209-210). Positions
  past training are not supported here.
- Refused rather than approximated: `position_embedding_type` other than
  rope, any `rope_scaling` (the NTK class M:218-250 is unported),
  `hidden_act` other than gelu, `layer_norm_type = rms_norm`,
  `pack_qkv` false or null, logn attention flags, `type_vocab_size = 0`,
  a Dense module, `include_prompt = false`, `do_lower_case = true`. An
  absent field takes the `GteConfig` default (C:94-119), and an explicit
  null is an error.
- Python tokenizers 0.22.2 vs Rust `tokenizers` 0.20: ids agree on the 9
  fixture texts only. The 17 MB tokenizer is not in testdata, so the
  tokenizer test needs the install.

## Fixtures and regeneration

`crates/embeddings/testdata/gte/` (about 1 MB): `tiny/` (checkpoint, no
sentence_bert_config.json like the pin), `tiny_cases.json` (ids, config,
cross-checks), `tiny_{hidden,embed}_*.npy`, `tiny_rotary_inv_freq.npy`,
`real_cases.json` (texts, prompt names, ids, `config_kwargs`),
`real_embed.npy`, `real_embed_mrl256.npy`, `real_trace_case0.npy`,
`real_rotary_{inv_freq,cos_rows,sin_rows}.npy`, and `manifest.json`
(package versions, pin, remote-code sha256, sha256 per file). A full
rerun reproduced every file byte for byte.

```sh
PY="emb-venv/bin/python -I"   # torch 2.14.1, transformers 4.57.1, sentence-transformers 5.1.2
M=~/models/embeddings/snowflake-arctic-embed-m-v2.0
$PY crates/embeddings/tools/gen_gte_fixtures.py --tiny $M --real $M
cargo test -p turbospark-embeddings --no-default-features --features gte
# real checkpoint: TURBOSPARK_GTE_DIR, default ~/models/embeddings/snowflake-arctic-embed-m-v2.0
cargo test --release -p turbospark-embeddings --no-default-features --features gte \
    gte -- --ignored --nocapture --test-threads 1
```

Not covered: retrieval quality, Metal, catalog/FFI/Swift integration,
padded batches.
