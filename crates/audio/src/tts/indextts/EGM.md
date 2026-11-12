# IndexTTS

## Overview
IndexTTS is a GPT-style text-to-speech family. It ports `mlx_audio/tts/models/indextts/` (gpt2 backbone, Conformer, perceiver resampler, ECAPA-TDNN speaker encoder, conditioned BigVGAN vocoder) to Rust. Generation prefills a [conditioning | text] embedding block, samples mel codes, and vocodes the per-step hidden rows into 24 kHz audio.

## Interface
- `Model` with `open`, `load`, `get_conditioning`, `log_mel_spectrogram`, `prepare_input_embedding`, `generate` [mod.rs:Model]
- `ModelConfig`, `GptConfig` (both `from_json`) [mod.rs:ModelConfig] [mod.rs:GptConfig]
- `GenerateParams` (default temp 0.8, top_k 30, max_tokens 5000) [mod.rs:GenerateParams]
- `Generated`, `StopReason` [mod.rs:Generated] [mod.rs:StopReason]
- `DEFAULT_SAMPLE_RATE` (24000), `REF_MEL_BINS` (100) [mod.rs:DEFAULT_SAMPLE_RATE] [mod.rs:REF_MEL_BINS]
- `ConformerArgs`, `Conformer` (`load`, `forward`) [conformer.rs:Conformer]
- `PerceiverResampler` (`load`, `forward`) [perceiver.rs:PerceiverResampler]
- `EcapaTdnnArgs`, `SpeakerEncoder` (`load`, `forward`) [speaker.rs:SpeakerEncoder]
- `BigvganConditioningConfig`, `BigvganConditioning` (`load`, `forward`) [vocoder.rs:BigvganConditioning]

## Invariants
- Temperature must be finite and >= 0; 0 selects greedy and never draws [mod.rs:Model::generate]
- Reference mel must be non-empty with every row `REF_MEL_BINS` wide [mod.rs:Model::generate] [mod.rs:Model::get_conditioning]
- Text prompt length may not exceed `max_text_tokens + 2` [mod.rs:Model::prepare_input_embedding]
- Position past `max_mel_tokens + 2 + max_conditioning_inputs` is an error [mod.rs:Model::generate]
- top_k >= vocab is refused [mod.rs:Model::generate]
- Only `rel_pos` Conformer attention is ported; `(cnn_module_kernel - 1)` must be even [conformer.rs:ConformerArgs::from_json]
- Every step's post-norm row, including the stop-token step, feeds the vocoder [mod.rs:Model::generate]
- Learned position lookups are bounds-checked [mod.rs:position_row]

## Side Effects
- Reads `config.json` and `model.safetensors` from the checkpoint dir [mod.rs:Model::open]
- RNG: seeded `rng::KeySequence` from the caller seed, only used when temperature > 0 [mod.rs:Model::generate]
- `generate` takes `&mut self` (gpt KV state mutates) [mod.rs:Model::generate]
- No env vars, threads, or logging: None

## Verification
- Run: `cargo test -p turbospark-audio tts::indextts`
