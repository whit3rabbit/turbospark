# Soprano TTS
## Overview
Port of mlx-audio `tts/models/soprano`: a dense qwen3 backbone whose per-step post-norm hidden states (not audio tokens) feed a ConvNeXt + iSTFT decoder. Includes the text front end and sentence preprocessing. Batch 1, non-streaming, caller owns tokenization and stop ids.
## Interface
- `DecoderConfig`, `SopranoDecoder` (re-exported) [mod.rs:SopranoDecoder]
- `clean_text` (re-exported) [text.rs:clean_text]
- `REFERENCE_COMMIT`, `DEFAULT_SAMPLE_RATE` (32000) [mod.rs:DEFAULT_SAMPLE_RATE]
- `ModelConfig` with `from_json` [mod.rs:ModelConfig]
- `GenerateParams`, `StopReason`, `Generated` [mod.rs:GenerateParams]
- `Model` with `open`, `load`, `preprocess_text`, `generate_from_prompt`, `generate_hidden`, `decode_hidden` [mod.rs:Model]
- `SopranoDecoder::load`, `SopranoDecoder::forward` [decoder.rs:SopranoDecoder]
## Invariants
- A `model_path` not containing "soprano-1.1" forces decoder_dim 512, intermediate 1536, input_kernel 3 [mod.rs:ModelConfig::from_json]
- Empty prompt rejected; logits length must equal prompt_len * vocab [mod.rs:Model::generate_hidden]
- The stop token is dropped: no push, no hidden row, no forward step [mod.rs:Model::generate_hidden]
- `temperature == 0.0` is argmax; top_p applies only in (0,1) [mod.rs:Model::generate_hidden]
- Decode keeps the last `tokens * token_size` samples [mod.rs:Model::decode_hidden]
- Decoder input must be exactly `positions * input_channels` values; upsample is aligned-corners to `upscale*(positions-1)+1` [decoder.rs:SopranoDecoder::forward]
- `clean_text` drops non-ASCII outright (no NFKD) [text.rs:clean_text]
## Side Effects
- Reads `config.json` and safetensors from the checkpoint dir; backbone and decoder use different prefixes [mod.rs:Model::open]
- RNG: `minimax_music3::rng::KeySequence` seeded per call [mod.rs:Model::generate_hidden]
- Env vars, threads, logging: None
## Verification
- Run: `cargo test -p turbospark-audio tts::soprano`
