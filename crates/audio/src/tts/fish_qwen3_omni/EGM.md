# Fish Qwen3 Omni
## Overview
Fish Qwen3 Omni is a DualAR TTS: a slow stack predicts one semantic token per step and a fast stack fills the remaining codebooks, decoded by the Fish S1 DAC. It ports `mlx_audio/tts/models/fish_qwen3_omni/` at mlx-audio e1b19b90, batch size 1, non-streaming; text tokenization is the caller's job.

## Interface
- `Model` (`open`, `load`, `build_semantic_logit_bias`, `generate`, `generate_codes_for_batch`) [mod.rs:Model]
- `ModelConfig::from_json`, `StackConfig` [mod.rs:ModelConfig] [mod.rs:StackConfig]
- `DualARTransformer` (`load`, `forward`, `fast_forward_cached`, `make_cache`, `make_fast_cache`) [mod.rs:DualARTransformer]
- `PromptPart`, `PromptMessage`, `PromptMatrix::to_model_input`, `encode_for_inference` [mod.rs:encode_for_inference]
- `sample_logits`, `GenerateParams`, `Generated`, `SpecialIds` [mod.rs:sample_logits] [mod.rs:GenerateParams]
- `decode_codes`, `adjust_speed` [mod.rs:decode_codes] [mod.rs:adjust_speed]
- `REFERENCE_COMMIT`, `RAS_WIN_SIZE`, `RAS_HIGH_TEMP`, `RAS_HIGH_TOP_P` [mod.rs:RAS_WIN_SIZE]

## Invariants
- `num_codebooks` must be positive (decoding loops over `num_codebooks - 1`) [mod.rs:ModelConfig::from_json]
- `generate` and `generate_codes_for_batch` refuse to run until `semantic_logit_bias` is set [mod.rs:Model::generate] [mod.rs:Model::generate_codes_for_batch]
- Prompt length must not exceed `text_config.max_seq_len`, and prompt plus decode budget must fit [mod.rs:Model::generate_codes_for_batch]
- Semantic token budget is `min(max_tokens, max(32, 12 * text_token_count))` [mod.rs:Model::generate_codes_for_batch]
- The logit bias is -1e9 everywhere except the semantic range and im_end id [mod.rs:Model::build_semantic_logit_bias]
- `speed` must be positive; within 1e-6 of 1.0 is a no-op [mod.rs:adjust_speed]
- Rotary phase tables are rounded to bf16 to match the reference [mod.rs:round_bf16]
- Each batch's generated codes are appended to the conversation as an assistant voice turn for the next batch [mod.rs:Model::generate]

## Side Effects
- Reads `config.json` and safetensors weights; uses the `model.` prefix unless an `embeddings` tensor is bare [mod.rs:Model::open]
- RNG: `rng::KeySequence::new(seed)` feeding `uniform01` draws [mod.rs:Model::generate]
- No env vars, threads or logging in non-test code.

## Verification
- Run: `cargo test -p turbospark-audio tts::fish_qwen3_omni`
