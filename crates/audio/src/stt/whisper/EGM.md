# Whisper ASR

## Overview
Portable CPU f32 implementation of OpenAI Whisper and Distil-Whisper: conv audio encoder, causal transformer decoder with cross-attention, and greedy autoregressive decode. Weights are supplied in memory by the caller; this module does not load files except `config.json`.

## Interface
- `WhisperSpecialTokens` (Default) [mod.rs:WhisperSpecialTokens]
- `WhisperConfig` with `tiny`, `base`, `small`, `large_v3`, `distil_large_v3`, `from_json`, `from_file`, `validate` [mod.rs:WhisperConfig]
- `EncoderLayerWeights`, `DecoderLayerWeights`, `WhisperWeights` [mod.rs:EncoderLayerWeights] [mod.rs:DecoderLayerWeights] [mod.rs:WhisperWeights]
- `WhisperModel` with `new`, `encode`, `decode_step`, `transcribe` [mod.rs:WhisperModel]

## Invariants
- `validate` requires non-zero `d_model`, head counts dividing `d_model`, and `num_mel_bins` of 80 or 128; `from_json` always validates [mod.rs:WhisperConfig].
- `encode` requires exactly 3000 mel frames and emits 1500 steps (hardcoded stride-2 output) [mod.rs:WhisperModel::encode].
- `decode_step` rejects empty or over-`max_target_positions` sequences and token ids at or above `vocab_size`; returns logits for the last position only [mod.rs:WhisperModel::decode_step].
- `proj_out: None` means logits tie to `decoder_embed_tokens` [mod.rs:WhisperModel::decode_step].
- Default special token ids (eot 50257, sot 50258, transcribe 50359, no_timestamps 50363) assume the multilingual tokenizer layout [mod.rs:WhisperSpecialTokens].
- `transcribe` pads or truncates audio to `WHISPER_WINDOW_SAMPLES` (480000, defined in [crates/audio/src/whisper.rs:WHISPER_WINDOW_SAMPLES]), hardcodes language token 50259 (English), and stops at `eot` or `max_tokens`; the prompt tokens are excluded from the result [mod.rs:WhisperModel::transcribe].
- Decode recomputes the full sequence and cross-attention K/V each step (no KV cache) [mod.rs:WhisperModel::decode_step].
- All LayerNorm calls use eps 1e-5 [mod.rs:run_layernorm].

## Side Effects
- `from_file` reads a config file from disk [mod.rs:WhisperConfig::from_file].
- No env vars, threads, statics, caches, or GPU use.

## Verification
- Run: `cargo test -p turbospark-audio stt::whisper`
