# S3 speech tokenizer

## Overview
S3 speech tokenizers (V1 25 Hz and V2) mapping log-mel features to discrete speech tokens.

## Interface
- `S3Config::v1`
- `S3Config::v2`
- `S3Config::from_json_with_defaults`
- `S3TokenizerV1::new`
- `S3TokenizerV1::open`
- `S3TokenizerV1::load`
- `S3TokenizerV1::quantize`
- `S3TokenizerV1::dequantize`
- `S3TokenizerV2::open`
- `S3TokenizerV2::load`
- `S3TokenizerV2::quantize`
- `log_mel_spectrogram`
- `V1_25HZ`

## Invariants
- [mod.rs:REFERENCE_COMMIT] pins the mlx-audio commit the port and its parity fixtures are normative against; re-pin only with regenerated fixtures.
- Bad config or tensor shapes surface as `SpeechError` (BadConfig, Input, Tensor) at load or call time, not as panics [mod.rs:load].
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.

## Side Effects
- Reads `config.json` and enumerates the directory with `std::fs::read_dir` in `open` [mod.rs:open]; no writes.

## Verification
- Run: `cargo test -p turbospark-audio codec::s3`
