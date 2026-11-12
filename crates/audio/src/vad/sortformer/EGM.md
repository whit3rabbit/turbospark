# Sortformer diarization

## Overview
NVIDIA Sortformer speaker diarization with offline and streaming profiles, mel features, and post-processing to segments.

## Interface
- `SortformerConfig::from_json`
- `Sortformer::load`
- `Sortformer::forward`
- `Sortformer::generate`
- `Sortformer::init_streaming_state`
- `Sortformer::streaming_step`
- `Sortformer::feed`
- `Sortformer::generate_stream`
- `Sortformer::set_streaming_profile`
- `Sortformer::frame_duration`
- `Sortformer::streaming_latency`
- `Sortformer::config`
- `StreamingProfile::offline`
- `StreamingProfile::low_latency`
- `StreamingProfile::very_low_latency`
- `StreamingProfile::ultra_low_latency`
- `StreamingOptions`
- `StreamingChunk`
- `GenerateOptions`
- `DiarizationSegment`
- `DiarizationOutput`
- `extract_mel_features`
- `preds_to_segments`
- `preds_to_segments_with_options`
- `median_filter_probs`
- `trim_silence`

## Invariants
- Bad config or tensor shapes surface as `SpeechError` (Audio, BadConfig, Input, Tensor, Unsupported) at load or call time, not as panics [mod.rs:load].
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.

## Side Effects
- Reads `config.json` and the safetensors weights from the model directory in `load` [mod.rs:load]; no writes, network, or env access.

## Verification
- Run: `cargo test -p turbospark-audio vad::sortformer`
