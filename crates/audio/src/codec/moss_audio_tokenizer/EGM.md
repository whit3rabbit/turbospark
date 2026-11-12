# MOSS audio tokenizer

## Overview
MOSS audio tokenizer: configurable module stack with RLFQ quantizer, whole-signal and per-frame (stateless) encode/decode. The streaming step path (AttentionStepCache) is not ported [mod.rs:encode_frame].

## Interface
- `MossConfig::from_json`
- `MossModuleConfig`
- `MossRlfqConfig`
- `MossAudioTokenizer::open`
- `MossAudioTokenizer::load`
- `MossAudioTokenizer::encode`
- `MossAudioTokenizer::encode_frame`
- `MossAudioTokenizer::decode`
- `MossAudioTokenizer::decode_frame`

## Invariants
- [mod.rs:REFERENCE_COMMIT] pins the mlx-audio commit the port and its parity fixtures are normative against; re-pin only with regenerated fixtures.
- Bad config or tensor shapes surface as `SpeechError` (BadConfig, Input, Tensor) at load or call time, not as panics [mod.rs:load].
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.

## Side Effects
- Reads `config.json` and enumerates the directory with `std::fs::read_dir` to find weight shards in `open` [mod.rs:open]; no writes.

## Verification
- Run: `cargo test -p turbospark-audio codec::moss_audio_tokenizer`
