# clap (embeddings)

## Overview
CLAP dual tower (laion/clap-htsat-unfused): HTSAT audio encoder, RoBERTa text encoder, a Slaney-mel feature extractor, and zero-shot audio classification.

## Interface
- `Clap` (`load`, `embed_audio`, `embed_audio_with`, `embed_audio_file`, `embed_text`, `tokenize`, `classify`) [clap/mod.rs:Clap]
- `ClapModel` (`load`, `from_weights`, `audio_embedding`, `text_embedding`, `logits_per_audio`) [clap/mod.rs:ClapModel]
- `SAMPLE_RATE`, `PIPELINE_TEMPLATE` [clap/mod.rs:SAMPLE_RATE]
- `ClapFeatureExtractor`, `MelFeatures`, `LongAudioCrop`, `load_wav_mono` [clap/features.rs:ClapFeatureExtractor]
- `HtsatEncoder` [clap/audio.rs:HtsatEncoder], `RobertaEncoder`, `position_ids` [clap/text.rs:RobertaEncoder]
- `ClapConfig`, `AudioConfig`, `TextConfig`, `FeatureConfig`, `HtsatGeometry` [clap/config.rs:ClapConfig]

## Invariants
- Audio input rate is 48 kHz; longer clips are cropped via `LongAudioCrop` (default `Head`), shorter ones tiled then zero-padded [clap/features.rs:ClapFeatureExtractor]
- Spectrum is computed in f64 and rounded through complex64 to match the reference [clap/features.rs:ClapFeatureExtractor]
- Config and feature config are validated before use [clap/config.rs:validate]

## Side Effects
- Reads checkpoint files and optional WAV files from disk [clap/mod.rs:Clap] [clap/features.rs:load_wav_mono]
- None otherwise

## Verification
- Run: `cargo test -p turbospark-embeddings`
