# Parakeet TDT

## Overview
Greedy TDT (token-and-duration transducer) speech recognition on the NeMo FastConformer encoder shared with Sortformer. `mod.rs` parses the NeMo-style config, loads the encoder and an LSTM predictor plus joint head, and runs the frame-skipping decode loop. `redux.rs` adapts Moondream's ternary "thrush-ternary-v2" export into the same config and tensor layout by expanding it to a temporary F32 safetensors file at open time.

## Interface
- `ParakeetProfile`, `PARAKEET_TDT_V2`, `PARAKEET_TDT_V3`, `PARAKEET_REDUX`: pinned repo and 40-char revision [mod.rs:ParakeetProfile]
- `ParakeetConfig` with `from_json` and public geometry fields [mod.rs:ParakeetConfig]
- `ParakeetTdt` with `open`, `config`, `transcribe`, `decode` returning `AlignedResult` [mod.rs:ParakeetTdt]
- `is_special_piece` (public so the Core ML host filters the same set) [mod.rs:is_special_piece]
- `adjust_sentencepiece_digit_spacing` [mod.rs:adjust_sentencepiece_digit_spacing]
- Parent-module-only (`pub(super)`) Redux hooks `normalized_config`, `convert_checkpoint` [redux.rs:normalized_config] [redux.rs:convert_checkpoint]

## Invariants
- Only 16 kHz, hann, per_feature, log-mel, frame_splicing=1 configs parse [mod.rs:ParakeetConfig::parse]
- Only non-causal dw_striding rel_pos batch_norm encoders with subsampling 8, odd conv kernel, 80 or 128 mels, and no attention bias are accepted, others return `Unsupported` [mod.rs:ParakeetConfig::parse]
- Joint/decoder dims must agree, durations are strictly increasing and start at 0, activation is relu [mod.rs:ParakeetConfig::parse]
- Blank id equals `vocab_size`, embedding has `vocab_size + 1` rows and the joint output has `vocab_size + 1 + num_extra_outputs` rows [mod.rs:TdtDecoder::load]
- Tensor shapes are checked before loading [mod.rs:load_checked]
- `decode` rejects mismatched sample rate, empty, or non-finite input [mod.rs:ParakeetTdt::decode]
- Standard loop forces progress after `max_symbols` consecutive zero-duration steps, Redux mode bounds iterations to `max_symbols * frame_count` and clamps blank duration to at least 1 [mod.rs:ParakeetTdt::greedy_decode]
- The predictor proposal is cached and committed only when a non-blank token is emitted [mod.rs:TdtDecoder::logits]
- Redux requires format `thrush-ternary-v2`, group size 128, 5 trits per byte, offset 1, bias-free modules, bytes above 242 and nonzero padding trits are refused [redux.rs:convert_checkpoint] [redux.rs:dequantize_ternary]
- Redux blank id must immediately follow the text vocabulary, every ternary module must appear in both config and manifest, and every `.qweight` needs a `.scales` [redux.rs:normalized_config] [redux.rs:convert_checkpoint]
- Redux LSTM bias is `bias_ih + bias_hh` [redux.rs:convert_checkpoint]
- The temp file is created with `create_new` and removed on success and failure after load [mod.rs:ParakeetTdt::open] [redux.rs:write_safetensors]

## Side Effects
- Reads `config.json`, `model.safetensors`, and for Redux `tokenizer.json`, `ternary.json` [mod.rs:ParakeetTdt::open] [redux.rs:convert_checkpoint]
- Redux writes a full F32 safetensors file in `std::env::temp_dir()` named by pid and nanosecond nonce, then deletes it [redux.rs:temporary_path] [mod.rs:ParakeetTdt::open]
- Memory-maps the checkpoint via `SafetensorsFile::open` [mod.rs:ParakeetTdt::open_with_config]
- CPU-only, no GPU, threads, env vars, or global statics (the TURBOSPEECH_PARAKEET_MODEL env var is read only in tests) [mod.rs:tests::model_dir]

## Verification
- Run: `cargo test -p turbospark-audio stt::parakeet`
