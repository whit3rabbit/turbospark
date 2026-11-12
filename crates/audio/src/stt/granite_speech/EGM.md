# Granite Speech STT

## Overview
Local-only speech-to-text and speech-translation pipeline for IBM Granite Speech checkpoints (4.0 1B and 4.1 2B Plus). `GraniteSpeech` chains a paired-mel frontend, a context-blocked Conformer encoder, a windowed QFormer projector, and a Granite llama-style LLM that reuses the shared Qwen3 ASR decoder stack. All weights are upcast to f32 at load and decoding is greedy with a KV cache. Rich task output (speaker attribution, word timestamps) is parsed from the decoded text.

## Interface
- `GraniteSpeech` with `load`, `profile`, `config`, `transcribe`, `transcribe_with_prompt`, `transcribe_with_task`, `transcribe_options` [mod.rs:GraniteSpeech]
- `GraniteSpeechOptions`, `GraniteSpeechTask`, `GraniteSpeechTranscription`, `GraniteSpeechSegment`, `WordTimestamp` [mod.rs:GraniteSpeechOptions]
- `GraniteSpeechProfile`, `GRANITE_4_0_1B_SPEECH`, `GRANITE_SPEECH_4_1_2B_PLUS` [mod.rs:GRANITE_4_0_1B_SPEECH]
- Prompt consts `DEFAULT_ASR_PROMPT`, `ASR_TASK_PROMPT`, `SAA_TASK_PROMPT`, `TIMESTAMPS_TASK_PROMPT`, `PLUS_SYSTEM_PROMPT`, `LANGUAGE_CODES` [mod.rs:LANGUAGE_CODES]
- `resolve_prompt`, `append_keywords`, `translation_prompt`, `expand_chat_template`, `parse_saa`, `parse_timestamps` [mod.rs:resolve_prompt]
- Parity witnesses `StageWitness`, `StepLogits`, `TranscribeStages` [mod.rs:TranscribeStages]
- `GraniteSpeechConfig` (re-export), `EncoderConfig`, `ProjectorConfig`, `TextConfig`, `from_json`, `is_plus`, `encoded_feature_dim`, `num_queries`, `audio_token_count` [config.rs:GraniteSpeechConfig]
- `GraniteSpeechEncoder::forward` [encoder.rs:GraniteSpeechEncoder]
- `EncoderProjector::forward` [projector.rs:EncoderProjector]
- `GraniteLlm` (pub struct, its methods are `pub(crate)`) [llm.rs:GraniteLlm]
- `frontend` constants and `features` are `pub(crate)` only [frontend.rs:features]

## Invariants
- Pinned checkpoints: 1B `ibm-granite/granite-4.0-1b-speech` rev `bd87ab86...`, Plus `ibm-granite/granite-speech-4.1-2b-plus` rev `a7593c7d...` [mod.rs:GRANITE_4_0_1B_SPEECH] [mod.rs:GRANITE_SPEECH_4_1_2B_PLUS]
- Quantized checkpoints are refused: any `.scales` or `.bias_packed` tensor, `quantization_config`, or `has_lora_adapter == true` returns `Unsupported` [mod.rs:GraniteSpeech::load] [config.rs:GraniteSpeechConfig::from_json]
- Config rejects non-GELU projector, non-SiLU text MLP, attention/MLP bias, `rope_scaling`, non-default `rope_type`, even `conv_kernel_size`, `hidden_dim != num_heads * dim_head`, `input_dim` not a multiple of 160, and `window_size % downsample_rate != 0` [config.rs:EncoderConfig::from_value] [config.rs:TextConfig::from_value] [config.rs:GraniteSpeechConfig::from_json]
- `projector.encoder_hidden_size` must equal `(1 + cat_hidden_layers.len()) * hidden_dim` [config.rs:GraniteSpeechConfig::from_json]
- Load fails if tokenizer `<|audio|>` id differs from `audio_token_index` or `<|end_of_text|>` is missing [mod.rs:GraniteSpeech::load]
- Frontend: rejects empty or non-finite samples, global log10 peak is taken before the odd trailing frame is dropped, floor is `peak - 8`, normalized `/4 + 1` [frontend.rs:features]
- Encoder input must be exactly `rows * input_dim`, midpoint self-conditioning fires after layer `num_layers / 2`, exports for `cat_hidden_layers` are taken after that injection [encoder.rs:GraniteSpeechEncoder::forward]
- Only the trailing attention block is masked to `f32::MIN` past the unpadded remainder [encoder.rs:ConformerAttention::forward]
- Attention and q/k/v projections must be bias-free, a present bias tensor returns `Unsupported` [encoder.rs:load_bias_free_linear] [llm.rs:load_linear]
- Granite scalars: `embedding_multiplier` is applied once to the full prompt embedding (audio included) in `transcribe_impl`, and per token in `next_hidden`, `logits_scaling` divides logits, `attention_multiplier` replaces the softmax scale [mod.rs:GraniteSpeech::transcribe_impl] [llm.rs:GraniteLlm::next_hidden] [llm.rs:GraniteLlm::logits] [llm.rs:load_attention]
- `max_tokens == 0` errors in `transcribe_impl`, `transcribe_options` maps 0 to 256, EOS stops decoding before emission [mod.rs:transcribe_impl] [mod.rs:transcribe_options]
- A custom `prompt` is only valid for `Asr`, `word_timestamps` upgrades Asr to Timestamps [mod.rs:resolve_prompt] [mod.rs:transcribe_options]
- Plus uses the role-token prompt with a default system prompt, 1B uses `USER: ...\n ASSISTANT:` [mod.rs:prompt_token_ids]
- `parse_saa` and `parse_timestamps` treat model output as untrusted (match_indices only, no byte stepping) and require speakers introduced in order, no unattributed text, one word per `[T:N]` tag [mod.rs:parse_saa] [mod.rs:parse_timestamps]

## Side Effects
- Reads `config.json` from disk and opens safetensors shards via `open_shards`, converting to f32 [mod.rs:GraniteSpeech::load] [encoder.rs:load_vector]
- Loads `tokenizer.json` through `load_tokenizer` [mod.rs:GraniteSpeech::load]
- No env vars, threads, GPU use, global statics, or caches in non-test code (CPU f32 math only). The checkpoint test reads `TURBOSPARK_GRANITE_SPEECH_MODEL_DIR` [mod.rs:tests]

## Verification
- Run: `cargo test -p turbospark-audio stt::granite_speech`
