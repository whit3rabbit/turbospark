# embeddinggemma (embeddings)

## Overview
google/embeddinggemma-300m: a bidirectional Gemma3 text encoder with mean pooling and a Dense head. Loads the upstream sentence-transformers layout and mlx-community bf16 or affine-quantized conversions.

## Interface
- `EmbeddingGemma` (`load`, `with_truncate_dim`, `embed`, `embed_with_prompt`, `embed_batch`, `embed_ids`, `head`, `tokenize`, `prompt`) [embeddinggemma/mod.rs:EmbeddingGemma]
- `Layout` (`SentenceTransformers`, `Mlx`) [embeddinggemma/mod.rs:Layout]
- `Backbone` (`load`, `forward`, `forward_traced`) [embeddinggemma/backbone.rs:Backbone]
- `GemmaConfig`, `LayerType`, `VERIFIED_QUANT_BITS` (4, 8), `VERIFIED_QUANT_GROUP` [embeddinggemma/config.rs:GemmaConfig]

## Invariants
- Quantized embedding tables stay packed and dequantize per gathered row [../weights.rs:EmbeddingTable]
- Truncation keeps the tokenizer's special tokens, like HF `truncation=True` [embeddinggemma/mod.rs:GemmaTokenizer]
- Only verified quantization bits and group size are accepted [embeddinggemma/config.rs:VERIFIED_QUANT_BITS]
- Configs without `use_bidirectional_attention: true` are refused, since a causal mask would embed silently wrong [embeddinggemma/config.rs:GemmaConfig::from_json]

## Side Effects
- Reads checkpoint, tokenizer and sentence-transformers JSON from disk at `load` [embeddinggemma/mod.rs:EmbeddingGemma]
- None otherwise

## Verification
- Run: `cargo test -p turbospark-embeddings`
