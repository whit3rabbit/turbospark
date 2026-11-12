# Fish Drama 3
## Overview
Fish Drama 3 is a multi-speaker, multi-codebook speech generator: an external trunk `Backbone` plus a local residual-code tower, with text chunking, reference prompts and a pause guard. It ports `fish_drama3/drama3.py` at mlx-audio commit dad2087f (the `REFERENCE_COMMIT`) with additions marked as not in the reference.

## Interface
- `REFERENCE_COMMIT` [mod.rs:REFERENCE_COMMIT]
- `Drama3` (`open`, `open_with`, `generate`), `Backbone`, `BackboneRow`, `GenerateOptions`, `GenerateStats`, `FrameEvent`, `Flow`, `ChunkStop` [generate.rs:Drama3] [generate.rs:Backbone]
- `TextCap::MEASURED`, `PauseGuard::MEASURED`, `PauseGuard::is_voiced` [generate.rs:TextCap] [generate.rs:PauseGuard]
- File names `TOKENIZER_FILE`, `SIDE_FILE`, `CODEC_FILE`, `TRUNK_DIR` [generate.rs:SIDE_FILE]
- `Drama3Config::from_json`, `semantic_band`, `check_tokenizer`, `MODEL_TYPE`, `SPEECH_TEMPLATE` [config.rs:Drama3Config]
- `split_text`, `split_text_with`, `validate_text`, `with_speaker_tag`, `spoken_text`, `CHUNK_BYTES`, `CHUNK_SPEAKERS`, `MAX_SPEAKER_ID` [chunking.rs:split_text] [chunking.rs:CHUNK_BYTES]
- `Utterance`, `Prompt`, `build_frames` [prompt.rs:build_frames]
- `SamplingParams` (`REFERENCE`, `FISH_SERVING`, `validate`), `candidates`, `sample`, `main_token`, `RAS` [sampling.rs:SamplingParams] [sampling.rs:sample]
- `Tower`, `CpuTower`, `TowerModel`, `TowerCache`, `TowerConfig`, `TowerEmbeddings` [tower.rs:Tower] [tower.rs:TowerModel]
- `VOICED`, `VOICED_CODES` [voiced.rs:VOICED]

## Invariants
- Only `model_type == "fish_drama3"` and template `reference-bracket-v1` load [config.rs:Drama3Config::from_json]
- Semantic band size must equal tower vocab, EOS must lie outside the band, and trunk hidden size must equal tower dim [config.rs:Drama3Config::from_json]
- Chunks are at most 300 bytes with at most 3 speakers; speaker ids up to 15 [chunking.rs:CHUNK_BYTES] [chunking.rs:CHUNK_SPEAKERS] [chunking.rs:MAX_SPEAKER_ID]
- Prompt codes must be `[num_codebooks, frames]`, non-empty, equal length and within `[0, vocab)` [prompt.rs:check_codes]
- `generate` rejects `max_frames == 0`, `text_cap.base_frames == 0`, and a backbone hidden size differing from config [generate.rs:Drama3::generate]
- A fixed `seed` fixes the codes (`KeySequence::new(options.seed)`) [generate.rs:Drama3::generate]
- Residual heads are cut to `acoustic_codebook_size()` entries (the Fish S1 DAC `build_ae()` codebook size) before top-k and top-p, so only decodable codes are drawn; book 0 keeps the full vocab [config.rs:Drama3Config::acoustic_codebook_size] [generate.rs:Drama3::generate]
- Q8 tower linears require 8-bit group-64 U32 packing, else `Unsupported` [tower.rs:Q8Linear::load]

## Side Effects
- Reads `config.json`, `tokenizer.json` and `tts_side.safetensors` from the install dir [generate.rs:Drama3::open_with]
- Threads: scoped threads for the tower tail overlap and for large Q8 matvecs (`available_parallelism`) [generate.rs:Drama3::generate] [tower.rs:Q8Linear::forward]
- RNG: threefry `KeySequence` seeded per call [generate.rs:Drama3::generate]
- No env vars or logging in non-test code.

## Verification
- Run: `cargo test -p turbospark-audio tts::fish_drama3`
