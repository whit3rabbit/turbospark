# KugelAudio TTS
## Overview
Port of the KugelAudio (VibeVoice derivative) text-to-speech model from mlx-audio. A Qwen2.5 decoder runs greedily over four speech tokens; each speech_diffusion token drives a diffusion head under an SDE-DPM-Solver++ scheduler, and the latents are decoded to audio by a convolutional VAE decoder. Scope is batch 1, non-streaming, with caller-owned tokenization [mod.rs:Model].

## Interface
- `Model::open`, `Model::load`, `Model::sample_rate`, `Model::generate(prompt_tokens, params, noise)`; pub field `config` [mod.rs:Model]
- `GenerateParams` (cfg_scale 3.0, max_tokens 2048, ddpm_steps) and `Generated` (latents, latent_frames, samples, sample_rate, token_count) [mod.rs:GenerateParams] [mod.rs:Generated]
- `ModelConfig`, `AcousticTokenizerConfig`, `DiffusionHeadConfig` with `from_json` and `validate` [mod.rs:ModelConfig] [mod.rs:DiffusionHeadConfig::validate]
- `DiffusionNoise` trait and `SeededNoise::new(seed)` [mod.rs:DiffusionNoise] [mod.rs:SeededNoise]
- Prompt helpers `SYSTEM_PROMPT`, `format_speaker_text`, `text_section`, `prompt_text` [mod.rs:prompt_text]
- Token constants `SPEECH_START_ID`, `SPEECH_END_ID`, `SPEECH_DIFFUSION_ID`, `EOS_TOKEN_ID`, `VALID_SPEECH_TOKENS`, `FINAL_LATENT_LOGIT_MARGIN` [mod.rs:VALID_SPEECH_TOKENS]
- `should_append_final_latent` [mod.rs:should_append_final_latent]
- Private submodules: `SDEDpmSolver` [scheduler.rs:SDEDpmSolver], `DiffusionHead` [diffusion_head.rs:DiffusionHead], `AcousticDecoder` [acoustic_decoder.rs:AcousticDecoder]

## Invariants
- ddpm steps must be in 1..=100000 and inference steps in 1..=train steps; checked at config parse [mod.rs:DiffusionHeadConfig::validate]
- `generate` rejects an empty prompt and `ddpm_steps == Some(0)`; the config's inference step count is restored after the call, even on error [mod.rs:Model::generate]
- Prompt tokens must include the trailing speech start id (caller encodes text) [mod.rs:Model::generate]
- The argmax is constrained to the four speech tokens [mod.rs:VALID_SPEECH_TOKENS]
- One extra latent is generated when speech_end wins within the logit margin [mod.rs:should_append_final_latent]
- Latents are unscaled unless the scaling factor is NaN; output is peak-normalized to 0.95 only if peak exceeds 1.0 [mod.rs:Model::generate_impl]
- Scheduler accepts the names `cosine`, `scaled_linear` and `squaredcos_cap_v2`, but always builds the cosine betas whatever the name, and accepts only epsilon/v_prediction/sample types; noise length must match the sample [scheduler.rs:SDEDpmSolver::new] [scheduler.rs:SDEDpmSolver::step]
- Decoder upsample needs stride >= 1 and kernel >= stride; latent length must equal frames * vae_dim [acoustic_decoder.rs:check_conv_transpose_geometry] [acoustic_decoder.rs:AcousticDecoder::forward]
- Diffusion head input shapes must match the batch [diffusion_head.rs:DiffusionHead]
- The backbone weight prefix is detected as `language_model.` or `model.language_model.` [mod.rs:open_backbone_weights]

## Side Effects
- Reads `config.json` and safetensors shards from the checkpoint directory [mod.rs:Model::open] [mod.rs:open_backbone_weights]
- `SeededNoise` draws from the crate-local threefry RNG keyed by seed [mod.rs:SeededNoise]
- `generate` temporarily mutates `config.diffusion_head.ddpm_num_inference_steps` [mod.rs:Model::generate]
- No env vars, threads, or logging found.

## Verification
- Run: `cargo test -p turbospark-audio tts::kugelaudio`
