# Qwen3-ASR (CPU speech-to-text and shared audio tower)

## Overview
Portable CPU implementation of Qwen3-ASR: a Whisper-style 128-band log-mel frontend, a chunked convolutional plus transformer audio encoder, and a Qwen3 GQA text decoder with affine-quantized weights and greedy generation over a per-layer KV cache. The decoder and greedy loop are shared by other Qwen3-backbone families (Higgs Audio v3, MOSS-Music, Granite via `TokenDecoder`), and the encoder is also reused by Qwen3-ForcedAligner. `checkpoint.rs` is a descriptor-only validator used before an install receipt is written.

## Interface
- `Qwen3Asr` with `load`, `transcribe` (fixed `max_tokens` 512), `transcribe_with_options`, `transcribe_with_details` [mod.rs:Qwen3Asr]
- `Qwen3AsrTranscription`, `Qwen3AsrProfile`, `QWEN3_ASR_06B_8BIT` (pinned repo and revision) [mod.rs:QWEN3_ASR_06B_8BIT]
- `extract_language`, `reported_language`, `logprob_summary`, `prompt_token_ids`, `transcript_from_tokens`, `load_tokenizer` [mod.rs:prompt_token_ids]
- `pub use` of `Qwen3Config` and `compute_selected_logprob` [mod.rs:Qwen3Config] [decoder.rs:compute_selected_logprob]
- `Qwen3Config`, `AudioEncoderConfig`, `TextConfig`, `from_json`, `from_root`, `estimated_resident_bytes` [config.rs:Qwen3Config]
- `AudioEncoder::load`, `AudioEncoder::forward` [encoder.rs:AudioEncoder]
- `compute_features`, `AudioFeatures`, `SAMPLE_RATE`, `MEL_BINS` [frontend.rs:compute_features]
- `validate_checkpoint` [checkpoint.rs:validate_checkpoint]
- Crate-internal (`pub(super)`/`pub(crate)`): `Decoder`, `TextAttention`, `Mlp`, `DecoderLayer`, `DecodeCache`, `TokenDecoder`, `greedy_generate`, `greedy_generate_with_logprobs`, `chat_stop_ids`, `is_chat_stop`, `load_linear_sharded` [decoder.rs:Decoder]

## Invariants
- Config requires `model_type == "qwen3_asr"`, a `quantization_config` with affine mode and bits in {2,3,4,5,6,8}, 128 mel bins, `output_dim == hidden_size`, and non-negative token ids [config.rs:Qwen3Config::from_json] [config.rs:AudioEncoderConfig::from_root]
- Text config forces `rotary_dim = head_dim` and `qk_norm = true`, attention bias and non-SiLU activation are rejected [config.rs:TextConfig::from_root]
- `load` refuses untied embeddings and requires `<|audio_pad|>`, `<|audio_start|>`, `<|audio_end|>` ids in the tokenizer to equal config ids [mod.rs:Qwen3Asr::load]
- Input must be non-empty, finite, `max_tokens > 0`, a requested language must case-insensitively match `supported_languages` [mod.rs:Qwen3Asr::transcribe_with_details]
- Frontend input is capped at 1200 s of 16 kHz samples, output is band-major `[128, frames]` [frontend.rs:compute_features]
- Prompt audio-placeholder count must equal the encoder row count, else `SpeechError::Input` [mod.rs:prompt_token_ids]
- Encoder chunk is `2 * n_window` frames, rows per full 100-frame chunk are 13 (reference `_get_feat_extract_output_lengths`), chunks run in parallel but are consumed in chunk order [encoder.rs:AudioEncoder::forward] [encoder.rs:subsampled_length]
- Encoder attention is block-local with window `max_valid * (n_window_infer / chunk_size)` [encoder.rs:AudioEncoder::forward]
- Conv weights accepted in MLX `[out,kh,kw,in]` or PyTorch `[out,in,kh,kw]` order, transposed on load [encoder.rs:Conv2d::load]
- Tensor name resolution order is fixed: `base`, `thinker.`, `llm.`, then Higgs/MOSS/Voxtral fallbacks that only match when earlier ones miss [decoder.rs:resolve_weight]
- Producers return logits, argmax happens in the greedy loop, stop on `<|im_end|>`/`<|endoftext|>`, falling back to ids 151645/151643 only when neither exists [decoder.rs:greedy_generate_with_logprobs] [decoder.rs:is_chat_stop]
- Validator expects packed linears as U32 `[rows, cols/4]` with `scales`/`biases` `[rows, cols/group]`, no bias tensor, and BF16 RMS norms, keep in step with encoder.rs and decoder.rs [checkpoint.rs:expect_packed_linear] [checkpoint.rs:expect_norm]

## Side Effects
- Reads `config.json`, `model.safetensors`, and `tokenizer.json` (or `tokenizer_config.json`, `vocab.json`, `merges.txt` fallback) from the model dir, loading dequantizes all weights to f32 [mod.rs:Qwen3Asr::load] [mod.rs:load_tokenizer]
- Env var `TURBOSPARK_QWEN3_ASR_PROFILE=1` prints per-phase timings to stderr [mod.rs:Qwen3Asr::transcribe_with_details]
- Encoder spawns worker threads per batch of chunks via `ops::par_map` [encoder.rs:AudioEncoder::forward]
- `checkpoint.rs` reads only safetensors descriptors, no tensor values [checkpoint.rs:validate_checkpoint]
- No Metal/GPU use and no global statics in this directory.

## Verification
- Run: `cargo test -p turbospark-audio stt::qwen3_asr`
- Pinned-checkpoint tests are `#[ignore]` and need `TURBOSPARK_QWEN3_ASR_DIR` and `TURBOSPARK_QWEN3_ASR_WAV` [mod.rs:tests]
