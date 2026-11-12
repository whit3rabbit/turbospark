# MMS English ASR (Wav2Vec2 CTC)

## Overview
English Massively Multilingual Speech recognition using the Wav2Vec2 CTC backbone with a per-layer adapter, loaded from a pinned local Hugging Face snapshot. Runs on CPU in f32 and decodes greedily with CTC. Shared backbone pieces (attention, positional conv, vocab parsing, CTC decode) come from `stt::wav2vec::backbone`.

## Interface
- `MmsProfile` [mod.rs:MmsProfile]
- `MMS_1B_FL102_ENGLISH` [mod.rs:MMS_1B_FL102_ENGLISH]
- `MmsConfig` with `from_json` [mod.rs:MmsConfig]
- `Mms` with `load`, `profile`, `config`, `transcribe` [mod.rs:Mms]

## Invariants
- The pin is repository `facebook/mms-1b-fl102` at revision d483345545bea550895b1aa0c6ba40236b9f1e22, language "eng" [mod.rs:MMS_1B_FL102_ENGLISH].
- `from_json` requires model_type wav2vec2 with a `Wav2Vec2ForCTC` architecture, and layer feature norm, stable layer norm, and gelu; otherwise `BadConfig` or `Unsupported` [mod.rs:MmsConfig::from_json].
- Conv arrays must be non-empty and equal length; hidden size must divide by heads and pos-conv groups [mod.rs:MmsConfig::from_json].
- `Mms::load` needs `config.json`, `model.safetensors`, `adapter.eng.safetensors`, and `vocab.json` in the folder; the encoder adapters and `lm_head` come from the adapter file [mod.rs:Mms::load] [mod.rs:EncoderLayer::load].
- Input audio must be non-empty and finite; it is normalized to zero mean and unit variance as (x - mean) / (std + 1e-7), the epsilon being added to the standard deviation, before the encoder [mod.rs:Mms::logits].
- Audio shorter than a conv kernel at any stage is rejected [mod.rs:FeatureConv::forward].
- The encoder is pre-norm; each layer adds attention, feed-forward, then the ReLU adapter residual [mod.rs:EncoderLayer::forward].
- Stage tracing is internal (private `StageTensor`) and used by the ignored reference test only [mod.rs:record_stage].

## Side Effects
- Reads and opens (SafetensorsFile) files from the supplied model directory [mod.rs:Mms::load].
- No env vars (outside ignored tests), threads, statics, or GPU use.

## Verification
- Run: `cargo test -p turbospark-audio stt::mms`
