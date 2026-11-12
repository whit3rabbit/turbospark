# gte (embeddings)

## Overview
The GTE rotary encoder behind Snowflake arctic-embed-m-v2.0, with the sentence-transformers pooling and normalize pipeline.

## Interface
- `Gte` (`load`, `with_truncate_dim`, `embed`, `embed_with_prompt`, `embed_batch`, `embed_ids`, `head`, `tokenize`) [gte/mod.rs:Gte]
- `Encoder` (`load`, `forward`, `forward_traced`, `inv_freq`), `inv_freq`, `rope_tables` [gte/encoder.rs:Encoder]
- `GteConfig` [gte/config.rs:GteConfig]

## Invariants
- RoPE tables derive from `inv_freq(theta, dim)` and are recomputed per sequence length [gte/encoder.rs:rope_tables]
- Keys without a default (and the `model_type` check) fail with `BadConfig` when missing or malformed; keys with a default (e.g. `max_position_embeddings`, `type_vocab_size`, `rope_theta`) fall back silently [gte/config.rs:GteConfig::from_json]

## Side Effects
- Reads checkpoint and tokenizer files at `load` [gte/mod.rs:Gte]
- None otherwise

## Verification
- Run: `cargo test -p turbospark-embeddings`
