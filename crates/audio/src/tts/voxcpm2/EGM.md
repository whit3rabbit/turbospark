# VoxCPM2 TTS

## Overview
Rust port of mlx-audio `voxcpm2` (VoxCPM2, tokenizer-free 2B TTS, 48 kHz output) at a pinned mlx-audio commit. It reuses the voxcpm MiniCPM backbone and flow solver and adds V2 conditioning concatenation, reference/continuation prompts, and an AudioVAE with per-block sample-rate conditioning [mod.rs:module doc]. Scope is single batch, non-streaming, caller-tokenized text.

## Interface
- `Model` with `open`, `sample_rate`, `encode_wav`, `generate` [mod.rs:Model]
- `ModelConfig` with `from_json` [mod.rs:ModelConfig]
- `GenerateParams` (defaults: max_tokens 2000, min_tokens 2, 10 timesteps, cfg 2.0, prefix 4) [mod.rs:GenerateParams]
- `Generated` (samples, sample_rate, token_count, latents, steps, stopped, continued) [mod.rs:Generated]
- `VoicePrompt`, `ContinuationPrompt` [mod.rs:VoicePrompt] [mod.rs:ContinuationPrompt]
- `SubConfig` [mod.rs:SubConfig]
- `AUDIO_START_TOKEN` (101), `REF_AUDIO_START_TOKEN` (103), `REF_AUDIO_END_TOKEN` (104) [mod.rs:AUDIO_START_TOKEN]
- `AudioVaeConfig` (re-exported) with `from_json`, `hop_length`, `chunk_size`, `decode_chunk_size` [audio_vae.rs:AudioVaeConfig]
- `AudioVae` with `load`, `encode`, `decode`, `out_sample_rate` [audio_vae.rs:AudioVae]
- `SampleRateConditionLayer` [audio_vae.rs:SampleRateConditionLayer]
- `UnifiedCfm` with `t_span`, `sample` (V2 Euler flow solver) [dit.rs:UnifiedCfm]
- `VoxCpmLocDiTv2` (mu-token-block denoiser) [dit.rs:VoxCpmLocDiTv2]
- Private submodules `audio_vae`, `dit`; no child EGM.md.

## Invariants
- Only the `euler` CFM solver is accepted; others return `Unsupported` [mod.rs:Model::open].
- `audio_vae_config.use_noise_block` is rejected because the reference noise is non-deterministic [mod.rs:Model::open].
- Only `cond_type == "scale_bias"` without `cond_out_layer` is supported [audio_vae.rs:SampleRateConditionLayer::load].
- `max_tokens` and `inference_timesteps` must be at least 1; `warmup_patches > 0` requires a voice or continuation prompt [mod.rs:Model::generate].
- The stop head only fires when `real_steps > min_tokens`, not a fixed step [mod.rs:Model::generate].
- The DiT `mu` must be exactly 2 x hidden wide, and `n_timesteps >= 1` [dit.rs:VoxCpmLocDiTv2::forward] [dit.rs:UnifiedCfm::sample].
- Continuation output trims `patch * decode_chunk_size * (streaming_prefix_len - 1)` samples from the front [mod.rs:Model::generate].
- Reference audio arrives at the output rate and is Fourier-resampled to the encode rate before right/left padding to a patch multiple [mod.rs:Model::encode_wav] [audio_vae.rs:fourier_resample].
- KV caches for both LMs are reset at the start of each `generate` [mod.rs:Model::generate].
- Noise is supplied by the caller via `FlowNoise`, one draw per patch [dit.rs:UnifiedCfm::sample].

## Side Effects
- Reads `config.json` and safetensors shards from the checkpoint dir [mod.rs:Model::open] [audio_vae.rs:AudioVae::load].
- Mutates `Model` KV caches during `generate` (needs `&mut self`) [mod.rs:Model::generate].
- No env vars, threads, or logging in library code; tests read `VOXCPM2_MEASURE` [tests.rs]. RNG only through the caller's `FlowNoise` [dit.rs:UnifiedCfm::sample].

## Verification
- Run: `cargo test -p turbospark-audio tts::voxcpm2`
