# Higgs Audio v2
## Overview
Higgs Audio v2 (Boson AI) is a llama-3.2-style backbone with dual-FFN layers (text and audio paths, routed by an audio-out mask), a separate audio codebook embedding table and a two-head projector. Frames use a delay pattern and decode through `crate::codec::higgs_audio`. It ports `mlx_audio/tts/models/higgs_audio/` at mlx-audio e1b19b90, batch size 1, non-streaming.

## Interface
- `Model` (`open`, `load`, `forward_hidden`, `forward`, `generate_frames`, `generate`, `encode_reference`, `build_prompt_smart_voice`, `build_prompt_voice_clone`, `lookup_audio_embedding`) [mod.rs:Model]
- `ModelConfig::from_json`, `TextConfig`, `codebook_stride` [mod.rs:ModelConfig] [mod.rs:TextConfig]
- `GenerateParams`, `Generated`, `StopReason`, `PromptEmbeds`, `ReferenceContext` [mod.rs:GenerateParams] [mod.rs:Generated]
- `apply_delay_pattern`, `revert_delay_pattern`, `build_delay_pattern_mask`, `clip_codes` [mod.rs:apply_delay_pattern] [mod.rs:revert_delay_pattern]
- `greedy_sample_audio`, `sample_audio`, `apply_top_p_row`, `apply_fades` [mod.rs:sample_audio] [mod.rs:apply_fades]
- `smart_voice_prompt_string`, `voice_clone_prefix_string`, `voice_clone_middle_string` [mod.rs:smart_voice_prompt_string]
- `DEFAULT_CODEC_REPO`, `DEFAULT_SAMPLE_RATE` [mod.rs:DEFAULT_CODEC_REPO]

## Invariants
- `num_attention_heads` must be non-zero and divide `hidden_size` [mod.rs:ModelConfig::from_json]
- `use_audio_out_self_attention=true` and `audio_decoder_proj_num_layers != 0` are `Unsupported` [mod.rs:ModelConfig::from_json]
- `audio_num_codebooks` and `audio_codebook_size` must be positive [mod.rs:ModelConfig::from_json]
- Audio stride is `codebook_size + 2` (stream BOS and EOS sentinels) [mod.rs:ModelConfig::codebook_stride]
- Frame 0 is a synthetic all-BOS init and prefill logits are discarded [mod.rs:Model::generate_frames]
- The first `sampling_warmup_frames` steps are greedy, later steps sampled with a fresh key each [mod.rs:Model::generate_frames]
- KV caches are fresh per call [mod.rs:Model::generate_frames]
- `encode_reference` requires exactly K reference codebooks [mod.rs:Model::encode_reference]
- Backbone tensors are found at bare or `model.` prefix, else a `Tensor` error [mod.rs:Model::load]

## Side Effects
- Reads `config.json` and safetensors weights from the checkpoint dir [mod.rs:Model::open]
- RNG: `rng::KeySequence::new(seed)` for sampled draws; Rust draws differ from mlx at the same seed [mod.rs:Model::generate_frames]
- No env vars, threads or logging in non-test code.

## Verification
- Run: `cargo test -p turbospark-audio tts::higgs_audio`
