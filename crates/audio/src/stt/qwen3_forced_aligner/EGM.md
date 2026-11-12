# Qwen3 Forced Aligner

## Overview
Word-level forced alignment of a known transcript against 16 kHz mono audio, ported from mlx-audio 0.5.7. It reuses the Qwen3-ASR audio encoder, decoder and tokenizer loader, replacing the LM head with a 5000-class timestamp classifier [mod.rs:Qwen3ForcedAligner]. Each word becomes `word<timestamp><timestamp>`, and argmax classes at those positions are repaired into monotone start/end times [mod.rs:Qwen3ForcedAligner::align_with_language].

## Interface
- `Qwen3ForcedAligner` with `load`, `config`, `align`, `align_with_language` [mod.rs:Qwen3ForcedAligner]
- `Qwen3ForcedAlignerConfig` with `from_json` [mod.rs:Qwen3ForcedAlignerConfig]
- `Qwen3ForcedAlignerProfile`, `QWEN3_FORCED_ALIGNER_06B_8BIT` [mod.rs:QWEN3_FORCED_ALIGNER_06B_8BIT]
- `ForcedAlignItem` (text, start_time, end_time) [mod.rs:ForcedAlignItem]
- `ForcedAlignResult` holding `items: Vec<ForcedAlignItem>`, with a `text()` method that joins item text with spaces [mod.rs:ForcedAlignResult, mod.rs:ForcedAlignResult::text]

## Invariants
- Pinned checkpoint is `mlx-community/Qwen3-ForcedAligner-0.6B-8bit` at revision 0e1a68e9... [mod.rs:QWEN3_FORCED_ALIGNER_06B_8BIT]
- Config requires `model_type == "qwen3_forced_aligner"` (in `thinker_config` or root), a nonnegative i32 `timestamp_token_id`, classify_num 5000 and timestamp_segment_time 80 ms, else `Unsupported` [mod.rs:Qwen3ForcedAlignerConfig::from_json]
- Timestamp id is a signed 32-bit value, token ids above i32 range are rejected [mod.rs:Qwen3ForcedAlignerConfig, mod.rs:Qwen3ForcedAligner::align_with_language]
- `load` checks that tokenizer ids for `<|audio_pad|>`, `<|audio_start|>`, `<|audio_end|>`, `<timestamp>` match config.json [mod.rs:Qwen3ForcedAligner::load]
- The head is loaded from `lm_head` with `classify_num` outputs and the base config's quant scheme [mod.rs:Qwen3ForcedAligner::load]
- Japanese and Korean return `Unsupported` before any model work (no external segmenter) [mod.rs:Qwen3ForcedAligner::align_with_language, mod.rs:tokenize_alignment_text]
- Words split on whitespace, CJK ideographs are one word each, apostrophes and alphanumerics stay in words, other punctuation splits, no words is an `Input` error [mod.rs:tokenize_alignment_text, mod.rs:is_cjk_char]
- Prompt must contain exactly one audio placeholder per encoder row and 2 timestamp markers per word, else `Input` error, encoder output must be a whole multiple of hidden_size [mod.rs:Qwen3ForcedAligner::align_with_language]
- Time = class * 80 ms, repaired by longest nondecreasing subsequence (runs of <= 2 snap to nearest anchor, longer runs interpolate linearly, NaN excluded), then rounded to ms [mod.rs:fix_timestamps, mod.rs:round_millis]
- Language names match `supported_languages` case-insensitively, default is "English" [mod.rs:Qwen3ForcedAligner::align]

## Side Effects
- Reads `config.json`, `model.safetensors` and tokenizer files (via `load_tokenizer`) from the model dir [mod.rs:Qwen3ForcedAligner::load]
- Weights are loaded through `SafetensorsFile`, no threads, GPU/Metal use, global statics or env vars in production code (the ignored test reads `TURBOSPARK_QWEN3_FORCED_ALIGNER_DIR`) [mod.rs:Qwen3ForcedAligner::load, mod.rs:tests]

## Verification
- Run: `cargo test -p turbospark-audio stt::qwen3_forced_aligner`
- The pinned-checkpoint fixture test is `#[ignore]` and needs `TURBOSPARK_QWEN3_FORCED_ALIGNER_DIR` [mod.rs:pinned_checkpoint_matches_mlx_word_alignment_fixture]
