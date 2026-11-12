# MOSS-Transcribe-Diarize

## Overview
Timestamped transcription with speaker labels for the pinned 4-bit MOSS-Transcribe-Diarize checkpoint. A whisper encoder (`encoder.rs`) and VQ adaptor (`adaptor.rs`) feed audio embeddings, merged four frames at a time, into the shared `qwen3_asr` decoder. `config.rs` parses strictly, `mod.rs` owns load, prompt construction with time markers, greedy decode, and `[start][Sxx] text [end]` segment parsing.

## Interface
- `MossTranscribeDiarize::{load, profile, config, processor_config, transcribe, transcribe_with_options}` [mod.rs:MossTranscribeDiarize]
- `MossTranscribeDiarizeProfile`, `MOSS_TRANSCRIBE_DIARIZE_4BIT` [mod.rs:MOSS_TRANSCRIBE_DIARIZE_4BIT]
- `DEFAULT_PROMPT`, `AUDIO_PAD_TOKEN` [mod.rs:DEFAULT_PROMPT] [mod.rs:AUDIO_PAD_TOKEN]
- `TranscriptSegment`, `MossTranscription`, `StageWitness`, `FirstLogits`, `TranscribeStages` [mod.rs:TranscriptSegment]
- `parse_segments(text, fallback_end)` [mod.rs:parse_segments]
- `MossConfig`, `WhisperAudioConfig`, `ProcessorConfig`, `PINNED_CHAT_TEMPLATE_SHA256` [config.rs:MossConfig] [config.rs:PINNED_CHAT_TEMPLATE_SHA256]
- `MossWhisperEncoder::{load, forward}` [encoder.rs:MossWhisperEncoder]
- `VqAdaptor::{load, forward}` [adaptor.rs:VqAdaptor]

## Invariants
- `chat_template.jinja` SHA-256 must equal the pinned digest, else `Unsupported` [mod.rs:MossTranscribeDiarize::load] [config.rs:PINNED_CHAT_TEMPLATE_SHA256]
- Revision pinned to `d42a296ee807e933ddd7588e2041dbbc84aff85d` [mod.rs:MOSS_TRANSCRIBE_DIARIZE_4BIT]
- Config accepts only `model_type` moss_transcribe_diarize, 80 mel bins, GELU, `audio_merge_size` 4, tied embeddings, 16 kHz, affine 4-bit group-64, scope `text_backbone_only`, encoder and adaptor prefixes excluded from quantization [config.rs:MossConfig::from_json] [config.rs:WhisperAudioConfig::from_value]
- `adaptor_input_dim` must equal `d_model * audio_merge_size` [config.rs:MossConfig::from_json]
- Tokenizer `<|audio_pad|>` id must equal `audio_token_id`, every digit must be a single token [mod.rs:MossTranscribeDiarize::load]
- Encoder and adaptor tensors must be unquantized, a `.scales` sibling is refused [encoder.rs:plain_linear] [adaptor.rs:VqAdaptor::load]
- Plain tensors must be F32, F16, or BF16 with exact shapes, conv weights accept `[out,k,in]` or `[out,in,k]` [encoder.rs:plain_tensor] [encoder.rs:Conv1d::load]
- Encoder output rows may not exceed `max_source_positions`, mel length must match `rows*80` [encoder.rs:MossWhisperEncoder::forward]
- Input must be non-empty, finite, at most `WHISPER_WINDOW_SAMPLES` (single 30 s chunk), `max_tokens > 0` [mod.rs:transcribe_impl]
- Audio placeholder count must equal `audio_token_length` before embedding injection, prompt may not embed `<|audio_pad|>` [mod.rs:transcribe_impl] [mod.rs:render_prompt]
- Audio is zero-padded to 30 s before the 80-band log-mel [mod.rs:compute_mel_features]
- Stop ids are 151643, 151645, and `<|im_end|>`, greedy loop stops before emission [mod.rs:MossTranscribeDiarize::load] [mod.rs:transcribe_impl]
- Segments with `end < start` or empty text are dropped, no match falls back to one unsegmented entry [mod.rs:parse_segments]

## Side Effects
- Reads `config.json`, optional `processor_config.json`, `chat_template.jinja` (hashed), `model.safetensors`, and tokenizer files from the model dir [mod.rs:MossTranscribeDiarize::load]
- Converts weights to f32 in memory [encoder.rs:plain_tensor] [adaptor.rs:load_exact]
- No env vars, threads, GPU use, or statics in non-test code, the gated test reads `TURBOSPARK_MOSS_TRANSCRIBE_MODEL_DIR` [mod.rs:tests]

## Verification
- Run: `cargo test -p turbospark-audio stt::moss_transcribe_diarize`
