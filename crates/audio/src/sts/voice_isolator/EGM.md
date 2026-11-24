# Voice Isolator

## Overview
Inworld Voice Isolator v1 port: 48 kHz mono voice isolation with a band-split dual-path network (time and band attention, Mamba1 along time and across bands), offline and causal streaming checkpoints. Upstream, parity, and speed: [README.md](./README.md). Core ML streaming step contract: [COREML.md](./COREML.md).

## Interface
- `VoiceIsolatorConfig::from_json`, `band_edges`, `window_samples` [mod.rs]
- `WEIGHT_FILES` (`denoiser.safetensors`, `denoiser_realtime.safetensors`, `model.safetensors`, lookup order) [mod.rs]
- `VoiceIsolator::open`, `open_file`, `load` [mod.rs]
- `VoiceIsolator::forward`, `isolate`, `isolate_controlled`, `passes` [mod.rs]
- `VoiceIsolator::set_backend`, `on_device`, `device_fallbacks`, `op_seconds`, `set_lookahead`, `lookahead_frames` [mod.rs]
- `VoiceIsolatorStream::new`, `push`, `flush`, `latency_samples` [stream.rs]
- `StreamStep` trait plus `VoiceIsolatorStream::external`, `push_external`, `flush_external`: an external network (a Core ML program holding its own caches) computes the stream's blocks [stream.rs]

## Files
- `mod.rs`: config from safetensors metadata, loading, STFT, band encoders, decoder heads, synthesis, offline chunked render.
- `layers.rs`: time and band attention blocks, KV cache, Mamba and BiMamba, the `Exec` dispatcher that sends projections, attention, and the selective scan to an optional `ComputeBackend`.
- `stream.rs`: the plain causal stream (`streaming.Stream`, no windowed lanes) and the `StreamStep` hook.
- `tests.rs`: fixture parity against `testdata/voice_isolator/{offline,causal}` and the opt-in real-checkpoint gate.

## Invariants
- The architecture comes from the safetensors `config` metadata entry, never guessed; unimplemented fields (dual mode, mode FiLM, other decoders) and checkpoints carrying weights the network never reads are refused [mod.rs:load].
- The stream keeps the full trained 1200-frame attention window, so its output does not depend on push sizes [stream.rs].
- `isolate_controlled` calls the progress hook only between model passes; output equals `isolate` [mod.rs:isolate_controlled].
- A device op that fails falls back to the host and is counted in `device_fallbacks` [layers.rs:Exec].

## Side Effects
- Reads the checkpoint file in `open` / `open_file`; no writes or network.
- Reads `TURBOSPARK_VOICE_ISOLATOR_DEVICE_MIN_MACS` (smallest projection sent to the device, for measurement) [layers.rs].

## Verification
- Run: `cargo test -p turbospark-audio sts::voice_isolator`
- Real checkpoints: set `TURBOSPARK_VOICE_ISOLATOR_MODEL` to a directory holding the released files; tests never download.
