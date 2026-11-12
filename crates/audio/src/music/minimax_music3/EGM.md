# minimax_music3

## Overview
Native port of MiniMax Music 3: a Qwen3-style autoregressive backbone with CFG sampling produces frame hidden states, a flow-matching DiT (Euler solver profiles) generates latents conditioned on them, and a vocoder renders 44.1 kHz audio. Compute is abstracted behind a backend trait so a device implementation can be injected.

## Interface
- `Model::load_converted`
- `Model::load_converted_with_precision`
- `Model::load_converted_with_backend`
- `Model::load_official_tree`
- `Model::generate_text`
- `Model::generate_text_timed`
- `Model::generate_text_with_progress`
- `Model::generate`
- `Model::generate_frame_hiddens`
- `Model::run_flow`
- `Model::set_flow_profile`
- `Model::set_flow_solver`
- `Model::set_flow_guidance`
- `ModelConfig::tiny`
- `TextGenerateRequest::new`
- `TextGenerateRequest::validate`
- `GenerateRequest`
- `Generation`
- `assemble_prompt`
- `model_quant_predicate`
- `FlowProfile`
- `FlowSolver`
- `FlowGuidance`
- `Music3Precision`
- `Progress`
- `Control`
- `StageTimings`
- `OutputStats`
- `output_stats`
- `backend::ComputeBackend`
- `backend::DeviceWeight`
- `backend::Sequence`
- `SAMPLING_RATE`
- `FRAME_RATE`
- `MAX_AUDIO_FRAMES`
- `MAX_PROMPT_TOKENS`
- Internal stages: `DepthDecoder` [depth.rs], `FlowMatchingTransformer` [dit.rs], `ConditionEncoder` [fusion.rs], `MlxConv1d` [conv.rs], plus `qwen3.rs`, `vocoder.rs`, `weights.rs`, `prompt.rs`, `rng.rs`, `stress.rs`, `timing.rs`.

## Invariants
- [mod.rs:MAX_PROMPT_TOKENS] caps prompt length at 5000 and [mod.rs:MAX_AUDIO_FRAMES] caps output at 9000 frames at [mod.rs:FRAME_RATE] 25 Hz; [mod.rs:TextGenerateRequest::validate] enforces request limits.
- Sampling constants are fixed to the reference: [mod.rs:AR_CFG_SCALE]=1.5, [mod.rs:AR_CFG_TOP_K]=50, [mod.rs:DIT_CFG_SCALE]=1.7; output rate is [mod.rs:SAMPLING_RATE]=44100.
- [euler.rs:FlowProfile] step counts are REFERENCE_STEPS=30, FAST_STEPS=21, TURBO_STEPS=8; [euler.rs:FlowGuidance::EXACT] is the reference guidance.
- Chunked decode uses [mod.rs:CHUNK_FRAMES]=200 with [mod.rs:CHUNK_HOP]=100 and crop constants [mod.rs:CROP_LEFT_LATENT], [mod.rs:CROP_RIGHT_LATENT], [mod.rs:LATENT_HOP_LENGTH]=512 and [mod.rs:OVERLAP_LATENT_LENGTH]=172; [mod.rs:AR_SAMPLING_TOP_K]=50 is the sampler's own top-k; changing them changes output.
- Errors are `SpeechError` Input/Tensor/Unsupported/BadConfig, not panics [mod.rs:Model::load_converted].
- Compute is behind the [backend.rs:ComputeBackend] trait; backend-specific state is not held in this crate.

## Side Effects
- `Model::load_official_tree*` and `load_converted*` read the checkpoint tree (config, tokenizer, safetensors); the official tree enumerates per-component shards with `std::fs::read_dir` [mod.rs:load_official_tensors].
- Reads env var `TURBOSPARK_MUSIC3_BENCH_TREES` in timing helpers used only by ignored timing tests [timing.rs:bench_dir].

## Verification
- Run: `cargo test -p turbospark-audio music::minimax_music3`
