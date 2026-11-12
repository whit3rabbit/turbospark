# Higgs Audio v3 STT

## Overview
Speech-to-text pipeline: a 128-band whisper-style audio encoder, an MLP projector with stride-2 temporal downsample, and the shared `qwen3_asr` decoder (untied LM head) driven by greedy decoding. Audio is chunked by Silero VAD ranges (or uniformly), each chunk is encoded and projected to one block of embeddings, spliced between prompt token embeddings, prefilled once, then decoded token by token. All math runs in f32 on the CPU, loaded from a pinned unquantized bfloat16 checkpoint.

## Interface
- `HiggsAudioV3Stt` with `load`, `load_with_vad`, `transcribe`, `transcribe_with_options`, `profile`, `config` [mod.rs:HiggsAudioV3Stt]
- `HiggsAudioV3Profile`, `HIGGS_AUDIO_V3_STT` pinned profile [mod.rs:HIGGS_AUDIO_V3_STT]
- `HiggsTranscription` (text, prompt_tokens, generation_tokens, audio_rows) [mod.rs:HiggsTranscription]
- `TranscribeStages`, `StageWitness`, `FirstLogits` parity-capture types [mod.rs:TranscribeStages]
- `DEFAULT_PROMPT`, `parse_output` [mod.rs:DEFAULT_PROMPT] [mod.rs:parse_output]
- `HiggsConfig` re-export, `AudioEncoderConfig`, `HiggsConfig::from_json`, `chunk_samples` [config.rs:HiggsConfig] [config.rs:AudioEncoderConfig]
- `HiggsAudioEncoder::forward`, `FeatureProjector::forward` [encoder.rs:HiggsAudioEncoder] [encoder.rs:FeatureProjector]
- `SpeechRanges` trait, `SileroRanges`, `split_long`, `chunk_ranges` [vad.rs:SpeechRanges] [vad.rs:SileroRanges] [vad.rs:chunk_ranges]

## Invariants
- Pinned checkpoint is `bosonai/higgs-audio-v3-stt` at revision `2ffd1aa39f5a1266931e405cba12e404a9f994b2` [mod.rs:HIGGS_AUDIO_V3_STT]
- Quantization is refused twice: a `quantization_config` key in config and any `.scales` tensor in a shard return `Unsupported` [config.rs:HiggsConfig] [mod.rs:HiggsAudioV3Stt]
- Config must have `model_type == "higgs_audio_3"`, 128 mel bins, GELU encoder, SiLU bias-free decoder without RoPE scaling, `projector_temporal_downsample == 2`, 16 kHz, `d_model` divisible by head count [config.rs:HiggsConfig] [config.rs:AudioEncoderConfig] [config.rs:text_config]
- The text decoder is always untied (`tie_word_embeddings: false`) with `qk_norm: true`, regardless of the HF flag [config.rs:text_config]
- Load verifies tokenizer ids: `<|AUDIO|>` equals `audio_in_token_idx`, `<|audio_eos|>` equals `audio_eos_token_id`, and `<|im_start|>`, `<|im_end|>`, `<|audio_bos|>`, `<|endoftext|>` exist [mod.rs:HiggsAudioV3Stt]
- Input must be non-empty, finite, 16 kHz mono, `max_tokens` must be positive [mod.rs:transcribe_impl]
- Decoding stops on Qwen EOS ids 151645 and 151643 before emission [mod.rs:STOP_IDS]
- Prompt layout is one user turn: prompt text, `<|audio_bos|>`, one placeholder per chunk, `<|audio_eos|>`, assistant header, placeholder id must be found or an `Input` error returns [mod.rs:prompt_token_ids] [mod.rs:transcribe_impl]
- Chunks are zero-padded to the longest chunk before the frontend, VAD failure or a missing VAD degrades to uniform chunking [mod.rs:transcribe_impl] [vad.rs:chunk_ranges]
- Encoder asserts `features.len() == frames * bins`, errors when conv frames exceed `max_source_positions`, and drops an odd trailing frame in pooling [encoder.rs:HiggsAudioEncoder]
- `k_proj` must be bias-free, every tensor length is checked on load [encoder.rs:AudioAttention] [encoder.rs:load_vector]
- Log-mel drops the final STFT frame before the global peak, clamps at peak minus 8, then `(x + 4) / 4` [encoder.rs:log_mel_128]
- `parse_output` mirrors the reference regexes: no DOTALL for `<|...|>` spans, unclosed `<think>` keeps the tail [mod.rs:parse_output] [mod.rs:remove_special_spans]

## Side Effects
- Reads `config.json`, safetensors shards, and tokenizer from the model directory [mod.rs:load_inner]
- Optionally loads the Silero VAD checkpoint directory (`model.safetensors`) [vad.rs:SileroRanges]
- Allocates the full f32 weight set in memory, no GPU use, threads, env vars, or global statics in non-test code [encoder.rs:HiggsAudioEncoder]
- Ignored parity test reads `TURBOSPARK_HIGGS_AUDIO_3_MODEL_DIR` [mod.rs:pinned_checkpoint_matches_reference_stages_and_transcript]

## Verification
- Run: `cargo test -p turbospark-audio stt::higgs_audio_3`
