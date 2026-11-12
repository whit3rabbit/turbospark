# Higgs Audio v3
## Overview
Higgs Audio v3 (Boson AI) is a qwen3 backbone (shared `crate::lm`) with one fused `multimodal_embedding` table serving as both audio input embedding and tied audio head, plus an SGLang-style delayed multi-codebook sampler. It ports `mlx_audio/tts/models/higgs_audio_v3/` at mlx-audio e1b19b90, batch size 1, non-streaming; decode goes through `crate::codec::higgs_audio`.

## Interface
- `Model` (`open`, `load`, `embed_audio_codes`, `embed_audio_codes_rows`, `audio_logits`, `build_prompt_embeddings`, `generate_frames`, `generate`) [mod.rs:Model]
- `ModelConfig::from_json`, `TextConfig::to_dense_config`, `codebook_stride` [mod.rs:ModelConfig] [mod.rs:TextConfig]
- `GenerateParams`, `Generated`, `StopReason` [mod.rs:GenerateParams] [mod.rs:Generated]
- `PromptBuilder::build_prompt`, `PromptParts`, `ReferenceCodes` [mod.rs:PromptBuilder] [mod.rs:PromptParts]
- `apply_delay_pattern`, `reverse_delay_pattern` [mod.rs:apply_delay_pattern] [mod.rs:reverse_delay_pattern]
- `SamplerState`, `step`, `sample_independent`, `greedy_sample_audio` [mod.rs:SamplerState] [mod.rs:step]
- `AUDIO_PLACEHOLDER_ID`, `DEFAULT_SAMPLE_RATE`, `DEFAULT_MAX_NEW_FRAMES` [mod.rs:AUDIO_PLACEHOLDER_ID]

## Invariants
- `audio_num_codebooks` and `audio_codebook_size` must be positive [mod.rs:ModelConfig::from_json]
- Stride equals `codebook_size`; BOC and EOC are ordinary classes (defaults `size-2`, `size-1`) [mod.rs:ModelConfig::codebook_stride] [mod.rs:ModelConfig::from_json]
- Codebook k uses rows `k * codebook_size + code` of the fused table [mod.rs:Model::embed_audio_codes]
- The first-codebook EOC starts a `K - 2` frame countdown (immediate stop when `K <= 2`) [mod.rs:step]
- Ramp-in forces codebooks beyond the delay count to BOC [mod.rs:step]
- After `generation_done`, `step` returns all -1 rows [mod.rs:step]
- Fewer than K raw frames yields an empty waveform [mod.rs:Model::generate]
- `generate_frames` rejects empty prompt embeddings or a length not a multiple of hidden size [mod.rs:Model::generate_frames]
- Four tensor layouts (bare, `model.`, `backbone.`, raw `body.`) are accepted, else a `Tensor` error [mod.rs:Model::load]

## Side Effects
- Reads `config.json` and safetensors weights from the checkpoint dir [mod.rs:Model::open]
- RNG: `rng::KeySequence::new(seed)`; sampled Rust runs diverge from mlx at the same seed [mod.rs:Model::generate_frames]
- No env vars, threads or logging in non-test code.

## Verification
- Run: `cargo test -p turbospark-audio tts::higgs_audio_v3`
