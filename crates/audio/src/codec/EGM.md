# codec

## Overview
Neural audio codecs, vocoders, and speaker/semantic tokenizers, one subdirectory per model, plus shared building blocks used by several of them. Each model loads from a model directory via a `SafetensorsFile` and runs on CPU f32 buffers.

## Interface
- Shared blocks: `Conv1d`, `ConvTranspose1d` [conv.rs]; `snake1d`, `WnConv1d`, `WnConvTranspose1d` [wnconv.rs]; `CodebookIndex` (crate-private, `nearest`, `raw_row`) [vq.rs]; DAC building blocks `ResidualUnit`, `EncoderBlock`, `DecoderBlock` (crate-private) [dac.rs].
- Module list: [mod.rs] declares `pub mod` for every codec below.
- [bigvgan](./bigvgan/EGM.md), [dacvae](./dacvae/EGM.md), [descript](./descript/EGM.md), [ecapa_tdnn](./ecapa_tdnn/EGM.md), [encodec](./encodec/EGM.md), [fish_s1_dac](./fish_s1_dac/EGM.md), [higgs_audio](./higgs_audio/EGM.md), [mimi](./mimi/EGM.md), [mimo_audio_tokenizer](./mimo_audio_tokenizer/EGM.md), [moss_audio_tokenizer](./moss_audio_tokenizer/EGM.md), [nemotron_voicechat](./nemotron_voicechat/EGM.md), [s3](./s3/EGM.md), [snac](./snac/EGM.md), [spark_bicodec](./spark_bicodec/EGM.md), [stepaudio2](./stepaudio2/EGM.md), [vocos](./vocos/EGM.md), [wavvae](./wavvae/EGM.md)

## Invariants
- Codec modules pin their reference via a `REFERENCE_COMMIT` constant (for example [mimi/mod.rs:REFERENCE_COMMIT]); shared blocks must stay bit-compatible with every codec's golden fixtures.
- `Conv1d::forward` and `ConvTranspose1d::forward` take flat `&[f32]` channel-major buffers and return a new `Vec<f32>` [conv.rs:Conv1d].

## Side Effects
- None in the shared blocks; they are pure functions over caller buffers [conv.rs:Conv1d::forward, wnconv.rs:snake1d]. Weight loaders read safetensors through `WnConv1d::load` [wnconv.rs:load].

## Verification
- Run: `cargo test -p turbospark-audio codec::`
