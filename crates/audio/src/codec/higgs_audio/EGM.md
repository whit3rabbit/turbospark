# Higgs Audio tokenizer

## Overview
Higgs Audio v2 tokenizer: acoustic encoder/decoder, semantic encoder, and residual VQ.

## Interface
- `HiggsAudioConfig::from_json`
- `HiggsAudioConfig::from_pytorch_json`
- `HiggsAudioConfig::defaults`
- `HiggsAudioTokenizer::open`
- `HiggsAudioTokenizer::load`
- `HiggsAudioTokenizer::encode_tokens`
- `HiggsAudioTokenizer::decode_tokens`
- `HiggsAudioTokenizer::encode_acoustic`
- `HiggsAudioTokenizer::rvq_encode`
- `HiggsAudioTokenizer::rvq_decode_rows`
- `AcousticEncoder`
- `AcousticDecoder`
- `HiggsRvq`

## Invariants
- [mod.rs:REFERENCE_COMMIT] pins the mlx-audio commit the port and its parity fixtures are normative against; re-pin only with regenerated fixtures.
- [semantic.rs:sinc_resample] is the resampler feeding the semantic branch; it is crate-private.
- `encode_tokens` returns `SpeechError::Unsupported` unless the config carries `semantic_model_config` (released-format configs omit it, so only decode works) [mod.rs:encode_tokens].
- Bad config or tensor shapes surface as `SpeechError` (BadConfig, Input, Unsupported) at load or call time, not as panics [mod.rs:load].
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.

## Side Effects
- Reads `config.json` and the safetensors weights from the model directory in `open` [mod.rs:open]; no writes, network, or env access.

## Verification
- Run: `cargo test -p turbospark-audio codec::higgs_audio`
