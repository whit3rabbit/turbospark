# StepAudio2 token-to-wav

## Overview
StepAudio2 token2wav: flow-matching model, CAMPPlus speaker embedding, and HiFT vocoder.

## Interface
- `StepAudio2Config::reference`
- `StepAudio2Config::from_json`
- `StepAudio2Config::from_json_with_defaults`
- `StepAudio2Token2Wav::open`
- `StepAudio2Token2Wav::load`
- `StepAudio2Token2Wav::prepare_prompt`
- `StepAudio2Token2Wav::decode`
- `StepAudio2Flow::infer`
- `StepAudio2HiFT::infer`
- `StepAudio2HiFT::predict_f0`
- `SineDraws`
- `kaldi_fbank`
- `StepAudio2CampPlus`
- `SAMPLE_RATE`

## Invariants
- [mod.rs:REFERENCE_COMMIT] pins the mlx-audio commit the port and its parity fixtures are normative against; re-pin only with regenerated fixtures.
- [mod.rs:SAMPLE_RATE] fixes output at 24 kHz.
- Bad config or tensor shapes surface as `SpeechError` (BadConfig, Input, Tensor) at load or call time, not as panics [mod.rs:load].
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.

## Side Effects
- `open` reads `tiny_weights.safetensors` and `tiny_config.json` from the directory; without the tiny weights it returns `BadConfig` (converted-checkpoint loading is not implemented there) [mod.rs:StepAudio2Token2Wav::open]; no writes.

## Verification
- Run: `cargo test -p turbospark-audio codec::stepaudio2`
