# bert (embeddings)

## Overview
BERT and XLM-RoBERTa sentence encoders (Snowflake arctic-embed v1, v1.5, l-v2.0 and mlx-community conversions). Plain f32 CPU forward pass plus the sentence-transformers pipeline (tokenize, encoder, pooling, optional Matryoshka truncation, normalize).

## Interface
- `BertEmbedder` with `load`, `load_with_pooling`, `with_truncate_dim`, `embed`, `embed_with_prompt`, `embed_query`, `embed_document`, `embed_batch`, `embed_ids`, `head`, `tokenize` [bert/mod.rs:BertEmbedder]
- `Layout`, `PoolingSource`, `known_conversion` [bert/mod.rs:known_conversion]
- `Encoder` (`load`, `forward`, `forward_traced`) [bert/encoder.rs:Encoder]
- `BertConfig`, `Arch`, `xlmr_position_ids`, `VERIFIED_QUANT_BITS`, `VERIFIED_QUANT_GROUP` [bert/config.rs:BertConfig]

## Invariants
- XLM-R position ids are derived from the pad id and skip padding, BERT uses sequential ids [bert/config.rs:xlmr_position_ids]
- Only affine quantization at the verified bit widths (4, 6, 8) and group 64 is accepted [bert/config.rs:VERIFIED_QUANT_BITS]
- Config values the family has not implemented are refused with an error instead of ignored [bert/mod.rs:BertEmbedder]

## Side Effects
- Reads safetensors, config, tokenizer and sentence-transformers JSON from the checkpoint directory at `load` [bert/mod.rs:BertEmbedder]
- None otherwise (pure CPU math, no network, no unsafe)

## Verification
- Run: `cargo test -p turbospark-embeddings`
