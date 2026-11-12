# Irodori TTS

## Overview
Japanese flow-matching TTS: a Rectified Flow DiT over continuous DACVAE latents at 48 kHz, ported from `mlx_audio/tts/models/irodori_tts/`. Includes a scratch or ModernBERT text encoder, reference-latent speaker encoder, optional duration predictor, Euler RF sampler with CFG, and the MeanFlow distilled sampler. Waveforms decode through the shared DACVAE codec. Tokenization and text normalization are caller-owned.

## Interface
- `Model` with `open`, `sample_rate`, `decode_latents`, `encode_ref_audio`, `generate_latents`, `generate` [mod.rs:Model]
- `ModelConfig`, `DitConfig`, `SamplerConfig` (each `from_json`) [mod.rs:ModelConfig] [mod.rs:DitConfig] [mod.rs:SamplerConfig]
- `ModernBertConfig`, `ModernBertEncoder` [mod.rs:ModernBertConfig] [mod.rs:ModernBertEncoder]
- `IrodoriDiT` with `encode_conditions_full`, `build_kv_cache`, `forward_with_conditions` [mod.rs:IrodoriDiT]
- `Conditions`, `KvCaches` [mod.rs:Conditions] [mod.rs:KvCaches]
- `GenerateParams`, `Generated`, `RefLatent` [mod.rs:GenerateParams] [mod.rs:Generated] [mod.rs:RefLatent]
- `FlowNoise` trait, `SeededFlowNoise::new` [mod.rs:FlowNoise] [mod.rs:SeededFlowNoise]
- `build_duration_features` [mod.rs:build_duration_features]

## Invariants
- `cfg_guidance_mode="alternating"` is refused; only independent or joint [mod.rs:generate_latents]
- The codec is optional at load, but `generate` and `decode_latents` error without `dacvae/` [mod.rs:Model::open] [mod.rs:Model::generate] [mod.rs:Model::decode_latents]
- Duration-predicted seconds are clamped to `min_seconds..max_seconds` [mod.rs:Model::generate_latents]
- Predicted-duration path requires caller-supplied duration features [mod.rs:Model::generate_latents]
- Output audio is trimmed to the predicted length and trailing silence [mod.rs:Model::generate]
- Latents are channel-last from the DiT and transposed to channel-first before decode [mod.rs:Model::generate]
- Reference audio length is capped by `ref_max_seconds` unless overridden [mod.rs:Model::encode_ref_audio]
- Single batch, non-streaming; batched CFG rows run sequentially [mod.rs:module docs]

## Side Effects
- Reads `config.json`, DiT safetensors, and optional `dacvae/` dir [mod.rs:Model::open]
- RNG: noise from a `FlowNoise` source; `SeededFlowNoise` uses `rng::KeySequence` [mod.rs:SeededFlowNoise]
- No env vars, threads, or logging: None

## Verification
- Run: `cargo test -p turbospark-audio tts::irodori_tts`
