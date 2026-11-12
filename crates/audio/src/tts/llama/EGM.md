# Llama SNAC TTS (Orpheus-style)
## Overview
Port of mlx-audio's `llama` TTS family: a dense llama backbone (shared `crate::lm`) that interleaves text with discrete SNAC audio codes. This module owns the prompt layout, the segment generate loop with reference logits processors, flat-code grouping and parsing, and SNAC decode. Scope is batch 1, non-streaming, with caller-owned tokenization [mod.rs:Model].

## Interface
- `Model::open`, `Model::load`, `Model::generate`, `generate_from_prompt`, `generate_tokens_from_prompt`, `prepare_input_ids`, `prepare_zeroprompt`, `parse_output`; pub fields `config`, `backbone` [mod.rs:Model]
- `ModelConfig::from_json` [mod.rs:ModelConfig]
- `GenerateParams` (defaults: temperature 0.6, top_p 0.8, top_k -1, repetition_penalty 1.3, context 20, max_tokens 1200), `Generated`, `StopReason` [mod.rs:GenerateParams] [mod.rs:Generated] [mod.rs:StopReason]
- `codes_to_layers`, `decode_audio_from_codes`, `encode_audio_to_codes` [mod.rs:codes_to_layers] [mod.rs:decode_audio_from_codes] [mod.rs:encode_audio_to_codes]
- Token constants `START_OF_HUMAN`, `END_OF_TEXT`, `END_OF_HUMAN`, `START_OF_SPEECH`, `END_OF_SPEECH`, `START_OF_AUDIO`, `END_OF_AUDIO`, `AUDIO_OFFSET`, `CODES_PER_GROUP`, `DEFAULT_TOKENIZER_NAME`, `DEFAULT_SAMPLE_RATE` [mod.rs:AUDIO_OFFSET]

## Invariants
- Empty prompts are rejected; all but the last prompt token are prefilled, the last is stepped through the processors [mod.rs:Model::generate_tokens_from_prompt]
- Logits length must equal `vocab_size` [mod.rs:Model::generate_tokens_from_prompt]
- Temperature 0.0 takes the exact greedy path and never draws random numbers [mod.rs:GenerateParams] [mod.rs:Model::generate_tokens_from_prompt]
- The loop stops on `END_OF_SPEECH` or after `max_tokens` [mod.rs:Model::generate_tokens_from_prompt] [mod.rs:StopReason]
- `parse_output` crops after the last `START_OF_SPEECH`, drops `END_OF_SPEECH`, trims to whole 7-code groups, then subtracts `AUDIO_OFFSET` (non-audio ids may go negative) [mod.rs:Model::parse_output]
- `codes_to_layers` refuses a code list one short of a complete group rather than overrunning [mod.rs:codes_to_layers]
- `decode_audio_from_codes` rejects an empty list; `generate` decodes with zero noise [mod.rs:decode_audio_from_codes] [mod.rs:Model::generate_from_prompt]
- `encode_audio_to_codes` requires 3 codec levels sized 1:2:4 [mod.rs:encode_audio_to_codes]
- Sampled token ids must fit `i32` [mod.rs:Model::generate_tokens_from_prompt]

## Side Effects
- Reads `config.json` and safetensors weights from the checkpoint directory [mod.rs:Model::open]
- Allocates and mutates a per-call prompt KV cache [mod.rs:Model::generate_tokens_from_prompt]
- Categorical draws use the crate-local threefry `KeySequence` seeded by the caller [mod.rs:Model::generate_tokens_from_prompt]
- `generate*` take `&mut self` (backbone forward mutates state) [mod.rs:Model::generate]
- No env vars, threads, or logging found.

## Verification
- Run: `cargo test -p turbospark-audio tts::llama`
