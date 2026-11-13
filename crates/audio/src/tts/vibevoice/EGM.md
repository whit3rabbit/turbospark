# VibeVoice
## Overview
Port of mlx-audio `tts/models/vibevoice` (streaming). A split Qwen2 decoder (`language_model` lower layers, `tts_language_model` upper layers) conditions a diffusion head under a deterministic DPM-Solver++ scheduler with CFG against a zeros-input negative branch. Acoustic latents decode to a waveform through a convolutional acoustic decoder. Optional voice cache preloads KV state.
## Interface
- `TTS_TEXT_WINDOW_SIZE` (5), `TTS_SPEECH_WINDOW_SIZE` (6) [mod.rs:TTS_TEXT_WINDOW_SIZE]
- `AcousticTokenizerConfig`, `DiffusionHeadConfig` with `validate`, `ModelConfig` [mod.rs:ModelConfig]
- `DiffusionNoise` trait, `SeededNoise::new` [mod.rs:DiffusionNoise]
- `Model` with `open`, `sample_rate`, `load_voice`, `generate` [mod.rs:Model]
- `GenerateParams` (cfg_scale 1.5, max_tokens 512), `Generated` [mod.rs:GenerateParams]
- `DpmSolver` with `new`, `set_timesteps`, `reset`, `timesteps`, `step` [scheduler.rs:DpmSolver]
- `AcousticDecoder`, `CausalConv`, `CausalConvTranspose`, `Block` pub(crate) [acoustic_decoder.rs:AcousticDecoder]
- `DiffusionHead` pub(crate) [diffusion_head.rs:DiffusionHead]
## Invariants
- `ddpm_num_steps` in 1..=100000 and `ddpm_num_inference_steps` in 1..=ddpm_num_steps [mod.rs:DiffusionHeadConfig::validate]
- Checkpoint must carry `language_model.embed_tokens.weight`, bare or under `model.` [mod.rs:open_backbone_weights]
- `ddpm_steps == Some(0)` rejected [mod.rs:Model::generate]
- Loaded voice layer counts must match the model [mod.rs:Model::generate]
- Type ids must be 0 or 1 [mod.rs:Model::type_embed]
- Scheduler accepts only `cosine` and `squaredcos_cap_v2` (both build the cosine betas) and epsilon/v_prediction/sample; `scaled_linear` and other names are refused as Unsupported [scheduler.rs:DpmSolver::new]
## Side Effects
- Reads `config.json` and safetensors from the checkpoint dir [mod.rs:Model::open]
- Reads a voice safetensors file [mod.rs:Model::load_voice]
- RNG: seeded `rng::KeySequence` [mod.rs:SeededNoise]
- Env vars, threads, logging: None in library code; tests read `VIBEVOICE_MEASURE` [tests.rs]
## Verification
- Run: `cargo test -p turbospark-audio tts::vibevoice`
