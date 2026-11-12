# ArkTTS
## Overview
Port of the Audio8 TTS preview family (`mlx_audio/tts/models/arktts`): a slow AR stack predicts one semantic token per frame, a fast AR stack predicts the residual codec codebooks, and a bundled 44.1 kHz codec decodes the code grid to audio. Two checkpoints share the `arktts` model_type: a pure-attention 0.6b stack and a `falcon_h1` 0.1b hybrid [mod.rs:ModelConfig].

## Interface
- `Model` (open, generate): text prompt segments to waveform [mod.rs:Model]
- `ArkttsModel` (load, generate_codes, generate_codes_with_uniforms) [mod.rs:ArkttsModel]
- `ModelConfig` (from_json, uses_falcon_slow) [mod.rs:ModelConfig]
- `SamplingParams` with defaults (512 tokens, T 0.7, top_p 0.9, top_k 50) [mod.rs:SamplingParams]
- `Uniforms` (new, push): recorded or crate-RNG uniform stream [mod.rs:Uniforms]
- `prepare_prompt`, `prompt_segments` [mod.rs:prepare_prompt] [mod.rs:prompt_segments]
- `Generation` (code grid, frame-major) [mod.rs:Generation]
- `codec::ArkttsCodec` (load, decode, decode_stages) [codec.rs:ArkttsCodec]
- `codec::CodecConfig` (from_tts_config) [codec.rs:CodecConfig]

## Invariants
- Prompt length must stay below `max_seq_len`; new tokens are capped to the remainder [mod.rs:generate_codes_with_uniforms]
- Empty generation is an error, not silent audio [mod.rs:Model::generate]
- Generation grid is frame-major and is transposed to book-major before the codec [mod.rs:Model::generate]
- Codec needs exactly 1 + residual_codebooks rows of equal length [codec.rs:ArkttsCodec::decode]
- Codec is decode-only; encode is a documented refusal, callers pass precomputed reference codes [codec.rs:module doc]
- Sampling is an exponential race with one uniform per vocabulary entry, first index wins ties [mod.rs:exponential_race]
- Text tokenization is caller-owned; exact prompt strings come from `prompt_segments` [mod.rs:prompt_segments]

## Side Effects
- Reads weights and config.json from the checkpoint directory [mod.rs:Model::open]
- Sampled draws use a crate-local seeded RNG (not MLX RNG) [mod.rs:Uniforms::new]
- No env vars, threads, or logging seen

## Verification
- Run: `cargo test -p turbospark-audio tts::arktts`
