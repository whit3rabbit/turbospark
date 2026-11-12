# Encoder Runner

## Overview
CPU runner for bidirectional BERT and XLM-RoBERTa embedding models. It validates the family and pooling from config files before opening weights, loads safetensors (BF16/F32 or MLX 8-bit affine) into owned FP32 buffers, and runs the `compute::encoder` reference blocks to produce one L2-normalized CLS vector per sequence. No GPU is used.

## Interface
- `EncoderRunner` [mod.rs:EncoderRunner]: config, weights and optional tokenizer; `open`, `from_parts`, `encode_tokens`, `encode_batch_tokens`, `encode_text`, `encode_batch_text`
- `SUPPORTED_MODEL_TYPES` [mod.rs:SUPPORTED_MODEL_TYPES]: `["bert", "xlm-roberta"]`
- `check_model_type` [mod.rs:check_model_type]: resolves the supported `model_type` from config.json or refuses
- `check_pooling` [mod.rs:check_pooling]: refuses sentence-transformers pooling that is not CLS-only
- `check_encoder_dir` [mod.rs:check_encoder_dir]: family plus pooling gate for a model directory
- `cosine_similarity` [mod.rs:cosine_similarity]: re-export from `compute`
- `EncoderWeights` [weights.rs:EncoderWeights]: embeddings, embedding LayerNorm, per-layer weights; `load_from_safetensors`
- `EncoderLayerWeightsOwned` [weights.rs:EncoderLayerWeightsOwned]: one layer in FP32, `as_borrowed` yields compute references
- Subdirectories: None

## Invariants
- Family is chosen from `config.json` `model_type`, never tensor names; a missing `model_type` is refused [mod.rs:check_model_type]
- Pooling must be CLS-only; a missing `1_Pooling/` directory passes [mod.rs:check_pooling]
- `check_encoder_dir` runs before any weights are opened [mod.rs:EncoderRunner::open]
- Empty input, length plus position offset over `max_position_embeddings`, token id >= `vocab_size`, mismatched or out-of-range token_type_ids are errors, not clamps [mod.rs:encode_tokens]
- Output is CLS-pooled and Euclidean-normalized; GELU is the erf form (`use_tanh_gelu: false`) [mod.rs:encode_tokens, reference_config]
- 8-bit quantization requires group_size equal to `compute::quant::GROUP_SIZE` [weights.rs:check_quantization]
- Missing linear biases default to zeros; missing weights, LayerNorm tensors or embeddings are errors [weights.rs:load_from_safetensors]
- Tokenizer load failure is swallowed to `None`, so `encode_text` then errors "no tokenizer loaded" [mod.rs:EncoderRunner::open]

## Side Effects
- Reads `config.json` (bounded), `1_Pooling/config.json`, `model.safetensors`, optional `tokenizer.json` [mod.rs:EncoderRunner::open, check_pooling]
- Dequantizes the whole checkpoint into host FP32 memory [weights.rs:load_from_safetensors]
- No env vars, network or Metal

## Verification
- Run: `cargo test -p turbospark-runtime encoder`
