# models

## Overview
Stable import-path facade that re-exports the crate's model families by role so callers can use `models::<role>::<model>` paths.

## Interface
- Role re-exports: `codec`, `lid`, `music`, `sts`, `stt`, `tts`, `vad` [mod.rs:pub use crate::codec etc.].
- Model aliases: `deepfilternet`, `dialogue_sidon`, `mossformer2_se` (from `sts`), `moonshine`, `whisper` (from `stt`), `kokoro` (from `tts`), `silero_vad`, `sortformer` (from `vad`) [mod.rs].

## Invariants
- Holds only `pub use` re-exports; no logic. Moving a model requires updating the alias here [mod.rs].

## Side Effects
- None [mod.rs].

## Verification
- Run: `cargo test -p turbospark-audio`
