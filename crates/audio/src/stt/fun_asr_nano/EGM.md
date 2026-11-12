# Fun-ASR-Nano

## Overview
Speech recognition for the pinned Fun-ASR-Nano-2512 profile: Kaldi-style FBANK with LFR stacking, a SANM encoder (reused from `stt::sensevoice`), an adaptor into the 1024-wide space, and a 4-bit quantized Qwen3 decoder (reused from `qwen3_asr`) with greedy decoding. Loads only an already-downloaded local snapshot.

## Interface
- `FunAsrNanoProfile` [mod.rs:FunAsrNanoProfile]
- `FUN_ASR_NANO_2512` [mod.rs:FUN_ASR_NANO_2512]
- `FunAsrNano` with `load`, `profile`, `transcribe`, `transcribe_with_max_tokens` [mod.rs:FunAsrNano]

## Invariants
- Pin: `mlx-community/Fun-ASR-Nano-2512` at revision a7bc96fceaafce39ed6748e0c0fa9a9508b67f86 [mod.rs:FUN_ASR_NANO_2512].
- `Config::from_json` requires model_type fun_asr_nano, a Hamming frontend, and exact geometry (16 kHz, 80 mels, LFR 7/6, 50 blocks, 20 tp blocks, 512 width, tokenizer path "Qwen3-0.6B"); anything else is `Unsupported` [mod.rs:Config::from_json].
- `load` requires a tied-embedding Qwen3 decoder with hidden size 1024, decoder quantization fixed at 4 bits and group size 128, and tokenizer tokens `<|im_end|>` and `<|endoftext|>` [mod.rs:FunAsrNano::load].
- Audio must be non-empty and finite; `max_tokens` must be positive [mod.rs:FunAsrNano::encode_audio] [mod.rs:FunAsrNano::transcribe_with_max_tokens].
- Audio token count is ceil-divided by 2 three times (min 1) and must not exceed feature rows [mod.rs:fake_token_length] [mod.rs:FunAsrNano::decode_audio].
- Prompt audio positions are placeholder id 0, overwritten with adaptor rows before prefill; token ids are checked to fit i32 [mod.rs:FunAsrNano::decode_audio].
- Decoding stops on `<|im_end|>` or `<|endoftext|>` and the text is trimmed [mod.rs:FunAsrNano::decode_audio].
- LFR stacking replicates edge frames at boundaries [mod.rs:stack_lfr].

## Side Effects
- Reads `config.json`, `model.safetensors`, and `<tokenizer_path>/tokenizer.json` from the model directory [mod.rs:FunAsrNano::load].
- No env vars (outside ignored tests), threads, statics, or GPU use.

## Verification
- Run: `cargo test -p turbospark-audio stt::fun_asr_nano`
