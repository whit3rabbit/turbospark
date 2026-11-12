# OuteTTS

## Overview
Port of OuteTTS: a dense LM backbone generates word-aligned c1/c2 audio code tokens from a text prompt (optionally with a speaker profile), decoded by the DAC codec and loudness-normalized (EBU R128 style K-weighting).

## Interface
- `Model` with `open`, `load`, `get_completion_prompt`, `generate`, `generate_tokens_from_prompt` [mod.rs:Model]
- `ModelConfig`, `GenerateParams`, `Generated`, `StopReason`, `DEFAULT_SAMPLE_RATE` [mod.rs:ModelConfig]
- `SpecialTokens` with `c1`, `c2`, `time` [mod.rs:SpecialTokens]
- `AudioTokenMaps` (`from_lookup`, `from_ids`, `extract_audio_from_tokens`), `CODEBOOK_TOKENS` [mod.rs:AudioTokenMaps]
- `Speaker`, `SpeakerWord` with `Speaker::from_json` [mod.rs:Speaker]
- Text helpers `text_normalizations`, `chunk_text`, `merge_speaker_text`, `create_codes`, `get_global_features`, `decode_codes_to_audio`, `REFERENCE_COMMIT` [mod.rs:chunk_text]
- `integrated_loudness`, `process_audio_array` [loudness.rs:process_audio_array]

## Invariants
- Empty prompt refused [mod.rs:Model::generate_tokens_from_prompt]
- Logits must be exactly one position of `vocab_size` [mod.rs:Model::generate_tokens_from_prompt]
- Codebook 1 and 2 lengths must match [mod.rs:decode_codes_to_audio]
- Decode runs in 4096-code chunks, then normalizes to -18 LUFS with -1 dB peak limit [mod.rs:decode_codes_to_audio]
- Silence measures negative infinity LUFS; empty input stays empty [loudness.rs:integrated_loudness]

## Side Effects
- Reads `config.json` and weights from the checkpoint dir [mod.rs:Model::open]
- RNG: threefry `KeySequence` from caller seed [mod.rs:Model::generate_tokens_from_prompt]
- `get_completion_prompt` mutates the speaker's last word separator, as the reference does [mod.rs:Model::get_completion_prompt]
- Env vars, threads, logging: None

## Verification
- Run: `cargo test -p turbospark-audio tts::outetts`
