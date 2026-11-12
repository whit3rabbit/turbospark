# TADA (Text-Audio Dual Alignment)
## Overview
Port of mlx-audio `tts/models/tada`: a Llama backbone consumes token embeddings, acoustic features, an acoustic-mask embedding, and gray-coded duration embeddings. A flow-matching head (VibeVoice layout) integrates over a logSNR time grid with per-dimension CFG. Output ends at de-normalized acoustic features and durations; the waveform codec is a separate download and is not ported.
## Interface
- `normalize_text` [mod.rs:normalize_text]
- `ModelConfig` with `from_json` [mod.rs:ModelConfig]
- `DiffusionNoise` trait, `SeededNoise::new` [mod.rs:DiffusionNoise]
- `SpecialIds`, `Prompt`, `build_prompt` [mod.rs:build_prompt]
- `Model` with `open`, `sample_rate`, `generate` [mod.rs:Model]
- `GenerateParams` (defaults acoustic_cfg 1.6, 20 flow steps, logsnr), `Generated` [mod.rs:GenerateParams]
- `decode_waveform_refusal` [mod.rs:decode_waveform_refusal]
- `int_to_gray_code`, `gray_code_to_int`, `decode_gray_code_to_time` [gray_code.rs:int_to_gray_code]
- `DiffusionHead`, `HeadLayer`, `TimestepEmbedder` are pub(crate) [diffusion_head.rs:DiffusionHead]
## Invariants
- Checkpoint must carry the backbone `embed_tokens` table [mod.rs:Model::load_weights]
- Time ids must be non-negative and below `num_time_classes`; mask ids within the 2-row table [mod.rs:Model::time_embed]
- With `num_extra_steps > 0` the prompt is truncated by `shift_acoustic`; steps = min(len + extra, max_tokens) [mod.rs:Model::generate]
- A negative-branch cache exists only when `acoustic_cfg_scale != 1.0` [mod.rs:Model::generate]
- Gray code bits: positive means 1, MSB first, +/-1 encoding [gray_code.rs:gray_code_to_int]
- Waveform decode always returns Unsupported [mod.rs:decode_waveform_refusal]
## Side Effects
- Reads `config.json` and safetensors from the checkpoint dir [mod.rs:Model::open]
- RNG: seeded threefry `KeySequence`, one draw per step [mod.rs:SeededNoise]
- Env vars, threads, logging: None in library code; tests read `TADA_MEASURE` [tests.rs]
## Verification
- Run: `cargo test -p turbospark-audio tts::tada`
