# Qwen3 TTS

## Overview
Qwen3 dense backbone that generates SNAC audio code tokens after a human-turn text prompt, with optional zeroprompt voice conditioning. Tokens are regrouped into 7-code frames and decoded by the SNAC codec at 24 kHz.

## Interface
- `Model` with `open`, `load`, `prepare_input_ids`, `prepare_zeroprompt`, `parse_output`, `generate`, `generate_from_prompt`, `generate_tokens_from_prompt` [mod.rs:Model]
- `ModelConfig`, `GenerateParams`, `Generated`, `StopReason` [mod.rs:ModelConfig]
- `codes_to_layers`, `decode_audio_from_codes`, `encode_audio_to_codes` [mod.rs:codes_to_layers]
- Token consts `TOKENIZER_LENGTH`, `START_OF_SPEECH`, `END_OF_SPEECH`, `AUDIO_TOKENS_START`, `CODES_PER_GROUP`, `DEFAULT_SAMPLE_RATE`, etc. [mod.rs:AUDIO_TOKENS_START]

## Invariants
- Special token ids are offsets from `TOKENIZER_LENGTH` (151669) [mod.rs:START_OF_SPEECH]
- Empty prompt refused [mod.rs:Model::generate_tokens_from_prompt]
- Logits must be one position of `vocab_size` [mod.rs:Model::generate_tokens_from_prompt]
- A flat code list that is not whole 7-code groups is refused [mod.rs:codes_to_layers]
- Last prompt token is stepped through processors; earlier tokens prefill unconditionally [mod.rs:Model::generate_tokens_from_prompt]

## Side Effects
- Reads `config.json` and weights from the checkpoint dir [mod.rs:Model::open]
- RNG: threefry `KeySequence` from caller seed [mod.rs:Model::generate_tokens_from_prompt]
- Env vars, threads, logging: None

## Verification
- Run: `cargo test -p turbospark-audio tts::qwen3`
