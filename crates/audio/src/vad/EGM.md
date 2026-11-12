# vad

## Overview
Voice activity detection and speaker diarization models.

## Interface
- Re-exports [mod.rs]: `NemotronDiarization`, `SileroVad`, `VadStreamState`, `VadTiming`, and a subset of the sortformer API (`Sortformer`, `DiarizationOutput`, `DiarizationSegment`, `GenerateOptions`, `StreamingOptions`, `StreamingProfile`). Other items, such as `SortformerConfig` and the Nemotron config types, are reached through their submodule paths.
- [nemotron_diarization](./nemotron_diarization/EGM.md), [silero_vad](./silero_vad/EGM.md), [sortformer](./sortformer/EGM.md)

## Invariants
- Only module declarations and re-exports; no logic [mod.rs].

## Side Effects
- None [mod.rs].

## Verification
- Run: `cargo test -p turbospark-audio vad::`
