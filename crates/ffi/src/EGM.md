# turbospark-ffi (src)

## Overview
Rust side of the C ABI consumed by the Swift app: status codes and panic guard, opaque session handles, JSON wire shapes, the in-process HTTP server, STT, image and vision plumbing. The `extern "C"` functions live in the `api` child.

## Interface
- ABI core: [abi.rs:ABI_VERSION], [abi.rs:TS_OK] through `TS_ERR_BUSY`, [abi.rs:fail], [abi.rs:clear_error], [abi.rs:guard], [abi.rs:guard_value], [abi.rs:guard_result], [abi.rs:install_error], [abi.rs:audio_error].
- Strings: [strings.rs] (borrow C strings in, hand owned ones out via `ts_string_free`).
- Sessions: [session.rs:Session] (move-only wrapper over `Arc<SessionCore>`, no `Clone`), [session.rs:SessionCore], [session.rs:Engine], [session.rs:shaping_config], [session.rs:borrow]; [open.rs] (macOS only: opens an install, mirrors the CLI `open_session`); [telemetry.rs:phases_json], [telemetry.rs:peak_footprint_bytes], [telemetry.rs:system_info_json].
- Wire shapes: [wire.rs:sized], [wire.rs:expert_cache_slots], [wire.rs:expert_residency], [wire.rs:expert_slots_wiring], [wire.rs:load_guard], [wire.rs:kv_bits], plus option structs.
- Server: [server.rs:Server], [server.rs:ServerLoadPolicy], [server_model.rs:FfiChatModel], [server_registry.rs:LiveRegistry], [server_registry.rs:ModelResidency], [server_registry.rs:DEFAULT_MAX_ATTACHED_MODELS], [server_idle.rs:IdlePolicy], [server_idle.rs:IdleSweep], [server_image.rs:ImageBridge], [server_speech.rs:SpeechBridge], [server_transport.rs:Traffic], [server_transport.rs:TrafficSnapshot].
- Media: [image_session.rs:ImageSession], [stt_session.rs:SttModel], [stt_session.rs:SttStream], [stt_session.rs:MAX_APPEND_BYTES], [vision.rs:prepare], [vision.rs:attach_minicpm]; [embedding_probe.rs:probe]; [heavy.rs:HeavyWorkGuard] (re-export of `runtime::heavy::HeavyWorkGuard`).
- Children: [api](./api/EGM.md), [generate](./generate/EGM.md), [models](./models/EGM.md); header: [../include](../include/EGM.md).

## Invariants
- Panics must never unwind across `extern "C"`; every entry point goes through `guard` except `ts_last_error` and `ts_peak_footprint_bytes` (no code to return, cannot panic) [api/core.rs:ts_last_error], [api/core.rs:ts_peak_footprint_bytes]; guarding exists because the workspace cannot use `panic = "abort"` [abi.rs:guard].
- `ServerOptions.guardrails` and `prefixReuse` are process-wide (one KV cache and posture for every client, not per API key); `prefixReuse` is applied before the startup session attaches [api/server.rs:ts_server_start].
- `ABI_VERSION` must equal `TS_ABI_VERSION` in the header [abi.rs:ABI_VERSION].
- The cancel flag lives outside the engine mutex so Stop never blocks on a running generation [session.rs:SessionCore].
- `Session` is a thin handle so the server shares an open model without opening a second one [session.rs:Session], [server_model.rs:FfiChatModel].
- The resolved context window and slot count are carried on the session, not re-read from the request [open.rs] (module doc).
- STT audio appends are bounded to two seconds of 16 kHz f32 PCM [stt_session.rs:MAX_APPEND_BYTES].
- Server image and speech bridges bound queued prompt/text to 1 MiB [server_image.rs:MAX_QUEUED_PROMPT_BYTES], [server_speech.rs:MAX_QUEUED_TEXT_BYTES].
- Attached models cap at 8 by default [server_registry.rs:DEFAULT_MAX_ATTACHED_MODELS].
- `open`, `stt_session` and `vision` compile only on macOS [lib.rs].

## Side Effects
- `Server` runs on a background OS thread with its own tokio runtime and binds a TCP listener (default host 127.0.0.1) [server.rs:Server].
- Idle sweep runs every 30 s. The default policy is disabled and releases only request-loaded (on-demand) models after the 30 min default TTL. Host-attached models stay loaded until the host opts in via `ts_server_set_idle_unload` (nonzero seconds) [server_idle.rs:SWEEP_INTERVAL], [server_idle.rs:DEFAULT_IDLE_TTL], [server_idle.rs:IdlePolicy], [api/server.rs:ts_server_set_idle_unload].
- Opening a session loads weights and allocates the KV cache on Metal [open.rs].
- Thread-local last-error storage [abi.rs:fail].

## Verification
- Run: `cargo test -p turbospark-ffi`
