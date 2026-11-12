# Nemotron VoiceChat codec

## Overview
NVIDIA Nemotron VoiceChat audio codec (STFT-domain encoder/decoder with discrete codes).

## Interface
- `NemotronVoiceChatConfig::from_json`
- `NemotronVoiceChatConfig::frame_rate`
- `NemotronVoiceChatConfig::stft_channels`
- `NemotronVoiceChatConfig::waveform_to_token_ratio`
- `NemotronVoiceChatCodec::open`
- `NemotronVoiceChatCodec::load`
- `NemotronVoiceChatCodec::encode`
- `NemotronVoiceChatCodec::encode_latents`
- `NemotronVoiceChatCodec::decode`

## Invariants
- [mod.rs:REFERENCE_COMMIT] pins the mlx-audio commit the port and its parity fixtures are normative against; re-pin only with regenerated fixtures.
- Bad config or tensor shapes surface as `SpeechError` (BadConfig, Input) at load or call time, not as panics [mod.rs:load].
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.

## Side Effects
- Reads `config.json` and the safetensors weights from the model directory in `open` [mod.rs:open]; no writes, network, or env access.

## Verification
- Run: `cargo test -p turbospark-audio codec::nemotron_voicechat`
