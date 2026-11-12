# KittenTTS

## Overview
StyleTTS2-family TTS with an ALBERT text encoder, duration and F0/noise predictors, and an iSTFTNet decoder producing 24 kHz audio. Ports `mlx_audio/tts/models/kitten_tts/` including the text preprocessor and quantized-ONNX heritage (optional activation fake-quant). espeak G2P is not ported; callers supply a phonemizer closure.

## Interface
- `Model` with `open`, `install_voice`, `phoneme_ids`, `prepare_segment`, `generate_segment`, `generate_segment_with_durations`, `config`, `synthesize`, `synthesize_with` [mod.rs:Model]
- `PreparedSegment` [mod.rs:PreparedSegment]
- `ModelConfig`, `AlbertConfig`, `IstftnetConfig` (`from_json`) [mod.rs:ModelConfig] [mod.rs:AlbertConfig] [mod.rs:IstftnetConfig]
- `ActivationQuant`, `fake_quant_dynamic_u8`, `maybe_fake_quant` [mod.rs:ActivationQuant]
- `SAMPLE_RATE` (24000), `REFERENCE_COMMIT` [mod.rs:SAMPLE_RATE] [mod.rs:REFERENCE_COMMIT]
- Re-exports: `TextPreprocessor`, `TextCleaner`, `symbol_table`, `chunk_text`, `ensure_punctuation`, `basic_english_tokenize` [frontend.rs:TextPreprocessor] [frontend.rs:chunk_text]

## Invariants
- Speed must be finite and > 0 and style finite, checked on every generate path [mod.rs:Model::generate_with_rng]
- `ref_s` length must equal `2 * style_dim` [mod.rs:Model::generate_with_rng]
- Voice tables must be `rows x 2*style_dim` with checked multiplication [mod.rs:Model::install_voice]
- Millisecond args are validated before any work [mod.rs:Model::synthesize_with]
- Empty text yields no segments [mod.rs:Model::synthesize_with]
- Resolved speed of each chunk feeds the next chunk, so speed priors compound [mod.rs:Model::synthesize_with]
- Integer literals are range-checked after URL and email removal [frontend.rs:TextPreprocessor::process]

## Side Effects
- Reads `config.json`, model safetensors, and the voices safetensors (`voices_path` with `.npz` mapped to `.safetensors`) [mod.rs:Model::open]
- RNG: `Rng::new(self.seed)` per request (SineGen random phase and noise make audio stochastic) [mod.rs:Model::synthesize_with] [mod.rs:Model::generate_segment]
- No env vars, threads, or logging: None

## Verification
- Run: `cargo test -p turbospark-audio tts::kitten_tts`
