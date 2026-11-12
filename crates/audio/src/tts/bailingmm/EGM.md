# BailingMM (Ming Omni TTS)
## Overview
Port of `mlx_audio/tts/models/bailingmm` covering MoE and dense variants. An LLM backbone (bailing MoE or Ming Qwen2) is read for hidden states only, a DiT flow-matching head emits one latent patch per step, an aggregator feeds patches back into the LLM stream, and an AudioVAE decodes latents to audio. There are no generated vocabulary tokens [mod.rs:Model].

## Interface
- `Model` (open, load, generate, generate_seeded, encode_prompt_latent, prepare_input_embeds, sample_rate, latent_dim) [mod.rs:Model]
- `ModelConfig` (from_json), `MarkerIds`, `PromptSegments`, `GenerateParams`, `Generated`, `StopReason` [mod.rs:ModelConfig]
- `DEFAULT_PROMPT`, `music_style_text` [mod.rs:DEFAULT_PROMPT] [mod.rs:music_style_text]
- `dit::{DiT, Aggregator, FlowLoss, FlowParams, FlowNoise, SeededNoise, MingRotary}`, `get_epss_timesteps`, `sway_sampling`, `load_flow_loss` [dit.rs:FlowNoise] [dit.rs:DiT]
- `audio_vae::{Encoder, Decoder, IstftHead, IstftState, UpsampleState}` [audio_vae.rs:Decoder]
- `ming_qwen2::{MingQwen2, SlidingConfig}` [ming_qwen2.rs:MingQwen2]

## Invariants
- LLM tensors sit under `model.model.`; the LLM weights open with a `model.` prefix, all others load by absolute name [mod.rs:Model::open]
- `max_decode_steps == 0` and empty input embeddings are rejected [mod.rs:Model::generate]
- Prompt latent length must be a multiple of latent_dim [mod.rs:Model::generate]
- latent_dim > 0, patch_size > 0, history_patch_size >= patch_size [mod.rs:check_patch_geometry]
- hidden_size must divide by num_heads [ming_qwen2.rs:MingQwen2::load_absolute] [dit.rs:DiT::load_absolute]
- Backbone config requires `model_type` [mod.rs:backbone_config]
- Noise draw order: initial flow noise per step, then one solver draw per integration step [dit.rs:FlowNoise]
- Text tokenization is caller-owned [mod.rs:MarkerIds]

## Side Effects
- Reads config.json and safetensors from the checkpoint dir [mod.rs:Model::open]
- Seeded noise uses the crate threefry KeySequence [dit.rs:SeededNoise]
- No env vars, threads, or logging seen

## Verification
- Run: `cargo test -p turbospark-audio tts::bailingmm`
