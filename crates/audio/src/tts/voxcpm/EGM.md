# VoxCPM 1.5
## Overview
Port of mlx-audio `tts/models/voxcpm`: a tokenizer-free MiniCPM latent-flow TTS. A base LM and residual LM condition a patch-level flow-matching DiT (UnifiedCFM Euler solver, CFG zero-star, sway sampling). Patches feed back through a CLS patch encoder; a stop head ends the run; a causal-conv AudioVAE decodes to a waveform. Voice cloning splices VAE-encoded reference patches into the prefix.
## Interface
- `AUDIO_START_TOKEN` (101) [mod.rs:AUDIO_START_TOKEN]
- `FlowNoise` trait, `SeededFlowNoise::new` [mod.rs:FlowNoise]
- `SubConfig`, `ModelConfig::from_json` [mod.rs:ModelConfig]
- `transpose` [mod.rs:transpose]
- `VoicePrompt`, `GenerateParams`, `Generated` [mod.rs:GenerateParams]
- `Model` with `open`, `sample_rate`, `encode_prompt_audio`, `generate` [mod.rs:Model]
- `AudioVaeConfig`, `AudioVae` with `load`, `encode`, `decode`, `hop_length` [audio_vae.rs:AudioVae]
- `VoxCpmLocDiT`, `UnifiedCfm` with `t_span`, `sample` [dit.rs:UnifiedCfm]
- `VoxCpmLocEnc::forward` [encoder.rs:VoxCpmLocEnc]
- `Mask`, `Config`, `MiniCpmModel` with `load`, `embed`, `forward` [minicpm.rs:MiniCpmModel]
## Invariants
- Only the `euler` CFM solver is accepted [mod.rs:Model::open]
- `max_tokens >= 1` and `inference_timesteps >= 1`; `n_timesteps >= 1` in the solver [mod.rs:Model::generate] [dit.rs:UnifiedCfm::sample]
- Both LM caches are reset at the start of every generate [mod.rs:Model::generate]
- Stop head acts only after step 5 (`i > 5`) [mod.rs:Model::generate]
- Prompt audio is left-padded with zeros to a multiple of `patch_size * hop_length` [mod.rs:Model::encode_prompt_audio]
- Solver warmup: `max(1, floor(len(t_span)*0.04))` zero-derivative steps [dit.rs:UnifiedCfm::sample]
- Encoder input length must equal `rows * patch_size * dim` [encoder.rs:VoxCpmLocEnc::forward]
- Token ids must be non-negative and within vocab [minicpm.rs:MiniCpmModel::embed]
- A decoder block carrying `noise.` tensors is refused [audio_vae.rs:DecoderBlock::load]
## Side Effects
- Reads `config.json` and safetensors from the checkpoint dir [mod.rs:Model::open]
- RNG: seeded `rng::KeySequence`, one normal draw per sample [mod.rs:SeededFlowNoise]
- Env vars, threads, logging: None in library code; tests read `VOXCPM_MEASURE` [tests.rs]
## Verification
- Run: `cargo test -p turbospark-audio tts::voxcpm`
