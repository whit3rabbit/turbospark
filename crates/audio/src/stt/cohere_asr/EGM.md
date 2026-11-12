# Cohere Transcribe Detection and Refusal

## Overview
Detects the Cohere Transcribe (`cohere_asr`) checkpoint distribution from `config.json` and always refuses to load it. No runnable checkpoint passed reference verification (official repo access-gated, public MLX mirror decodes incoherent text), so the module preserves that negative finding as a typed error instead of mis-decoding.

## Interface
- `COHERE_ASR_MODEL_TYPE` [mod.rs:COHERE_ASR_MODEL_TYPE]
- `OFFICIAL_REPOSITORY`, `OFFICIAL_REVISION` [mod.rs:OFFICIAL_REPOSITORY] [mod.rs:OFFICIAL_REVISION]
- `MIRROR_REPOSITORY`, `MIRROR_REVISION` [mod.rs:MIRROR_REPOSITORY] [mod.rs:MIRROR_REVISION]
- `detect(model_dir) -> Result<bool>` [mod.rs:detect]
- `load(model_dir) -> Result<()>` [mod.rs:load]

## Invariants
- `COHERE_ASR_MODEL_TYPE` is "cohere_asr" and detection compares `model_type` exactly [mod.rs:detect].
- `detect` returns `SpeechError::Input` for an unreadable `config.json` and `SpeechError::BadConfig` for invalid JSON [mod.rs:detect].
- `load` never succeeds: a non-matching config yields `BadConfig` on field `model_type`, a matching one yields `SpeechError::Unsupported` [mod.rs:load].
- The pinned official and mirror revisions are recorded constants cited in the refusal message; changing them requires re-verification [mod.rs:OFFICIAL_REVISION] [mod.rs:MIRROR_REVISION].

## Side Effects
- Reads `<model_dir>/config.json` from disk [mod.rs:detect].
- No env vars, threads, statics, or GPU use.

## Verification
- Run: `cargo test -p turbospark-audio stt::cohere_asr`
