# sts

## Overview
Speech-to-speech models: enhancement, restoration, and source separation, one subdirectory each.

## Interface
- Re-exports [mod.rs]: `DeepFilterNet`, `DialogueSidon`, `SidonConfig`, `MelRoFormer`, `MelRoFormerConfig`, `MossFormer2Se`, `MossFormer2SeConfig`, `SamAudio`, `SamAudioConfig`, `VoiceIsolator`, `VoiceIsolatorConfig`, `VoiceIsolatorStream`.
- [deepfilternet](./deepfilternet/EGM.md), [dialogue_sidon](./dialogue_sidon/EGM.md), [mel_roformer](./mel_roformer/EGM.md), [mossformer2_se](./mossformer2_se/EGM.md), [sam_audio](./sam_audio/EGM.md), [voice_isolator](./voice_isolator/EGM.md)

## Invariants
- Only module declarations and re-exports; no logic [mod.rs].

## Side Effects
- None [mod.rs].

## Verification
- Run: `cargo test -p turbospark-audio sts::`
