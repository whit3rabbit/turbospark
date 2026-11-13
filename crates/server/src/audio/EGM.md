# audio (server routes)

## Overview
HTTP and WebSocket routes for speech-to-text, text-to-speech and music generation. The module owns route registration, request validation and wire formats; inference is delegated to a host-supplied `AudioProvider`. When no provider is attached, no audio route is registered at all.

## Interface
- [audio/mod.rs:AUDIO_ROUTE_PATHS] route list registered only when a provider exists.
- [audio/mod.rs:AudioProvider] trait the host implements; [audio/mod.rs:AudioError], [audio/mod.rs:AudioTask], [audio/mod.rs:AudioModelInfo], [audio/mod.rs:TranscribeRequest], [audio/mod.rs:Transcription], [audio/mod.rs:SpeechRequest], [audio/mod.rs:SpeechStream], [audio/mod.rs:GenerateRequest], [audio/mod.rs:GeneratedAudio].
- [audio/stt.rs:transcriptions] `POST /v1/audio/transcriptions`; [audio/stt.rs:translations_unsupported] refuses translations.
- [audio/ws.rs:realtime] `GET /v1/audio/transcriptions/realtime` (utterance-granular WebSocket).
- [audio/speech.rs:speech] `POST /v1/audio/speech`.
- [audio/generate.rs:generate] `POST /v1/audio/generate` (music).
- [audio/jobs.rs:JobStore], [audio/jobs.rs:get_job], [audio/jobs.rs:get_job_result], [audio/jobs.rs:delete_job] async job store and routes.
- [audio/models.rs:list_models], [audio/models.rs:load_model], [audio/models.rs:delete_model].
- [audio/metrics.rs:metrics_endpoint] `GET /v1/metrics` (audio series only), [audio/metrics.rs:AudioMetrics].

## Invariants
- Upload size is capped by [audio/mod.rs:MAX_AUDIO_UPLOAD_BYTES] and transcribed duration by [audio/mod.rs:MAX_TRANSCRIBE_SECONDS].
- Only WAV decodes, resampled to [audio/stt.rs:STT_SAMPLE_RATE] via [audio/stt.rs:decode_wav_16k]; other containers are refused.
- The job store is in memory and bounded by [audio/jobs.rs:MAX_ACTIVE_JOBS], [audio/jobs.rs:FINISHED_TTL] and [audio/jobs.rs:MAX_RESULT_BYTES]; a running job already on a worker cannot be interrupted.
- Unsupported request fields are refused by name, not ignored (`deny_unknown_fields` on [audio/generate.rs:GenerateInput]).
- Voice cloning is refused for families that cannot clone ([audio/mod.rs:clones_voices]); MegaTTS3, and OmniVoice, Qwen3-TTS (`qwen3_tts`) and Drama 3 (`fish_drama3`) when a reference is given, need `consent: true`, and Drama 3 also needs `reference_text` ([audio/speech.rs:speech]). Qwen3-TTS takes `reference_text` as optional: without it the bridged engine clones from the speaker embedding alone. Neither the runtime nor the Swift bridge checks consent for Qwen3-TTS, so this route is its only gate. Realtime caps: 300 s buffered without a commit, 1 MiB per frame, 60 s idle close ([audio/ws.rs:MAX_BUFFER_SECONDS], [audio/ws.rs:MAX_FRAME_BYTES], [audio/ws.rs:IDLE_TIMEOUT]).
- The route trims `model_id` and passes it to the provider ([audio/models.rs:load_model]). Only the real provider limits it to installed catalog aliases, never paths ([audio_real.rs:valid_alias]); another `AudioProvider` carries no such guarantee from the route.
- `DELETE /v1/audio/models/{id}` is 404 for an unattached id and 409 while that model has running jobs ([audio/models.rs:delete_model]).
- Realtime auth uses headers only, no query-string key ([audio/ws.rs:RealtimeQuery]). A browser `Origin` that is not loopback is refused ([audio/ws.rs:origin_allowed]).

## Side Effects
- Spawns tasks for async jobs and aborts them on delete ([audio/jobs.rs:JobStore]).
- Loads and unloads audio models through the provider ([audio/models.rs:load_model], [audio/models.rs:delete_model]).
- Updates process-wide audio counters ([audio/metrics.rs:AudioMetrics]).

## Verification
- Run: `cargo test -p turbospark-server`
