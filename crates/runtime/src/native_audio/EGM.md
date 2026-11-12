# Native Audio Streaming Sessions

## Overview
Live streaming transcription sessions that sit beside the batch `AudioSession` in `native_audio.rs`. They deliberately avoid the process-wide heavy-work guard, because a recording can last hours and would otherwise block chat. Memory is admitted once at open, work runs on the CPU and Neural Engine only, and a concurrent call is refused rather than queued.

## Interface
- `AudioStreamSession` [stream.rs:AudioStreamSession]: Parakeet Unified (Core ML) live session; `open`, `family`, `latency_seconds`, `append`, `finish`, `cancel`
- `MultitalkerStreamSession` [multitalker_stream.rs:MultitalkerStreamSession]: Frozen E session coupling Nemotron-3 diarization to Multitalker Parakeet on one worker thread; `open(asr_dir, diar_dir)`, `append`, `finish`, `cancel`
- `SpeakerSegment` [multitalker_stream.rs:SpeakerSegment]: speaker label, index, start/end seconds, text
- `SpeakerStreamUpdate` [multitalker_stream.rs:SpeakerStreamUpdate]: segments, active speakers, received/processed seconds, `is_final`
- Subdirectories: None

## Invariants
- `AudioStreamSession::open` refuses any directory whose `detect_layout` is not `ParakeetUnified` before memory admission or load [stream.rs:AudioStreamSession::open]
- One call at a time: `try_lock` on the stream, compare-exchange on `busy`; a second caller gets "stream is busy" [stream.rs:lock, multitalker_stream.rs:call]
- `cancel` is terminal; later calls return an error because a cancelled step leaves the model mid-window [stream.rs:cancel, multitalker_stream.rs:cancel]
- Multitalker `append` accepts 1 to `MAX_PCM_CHUNK` samples [multitalker_stream.rs:append]
- Multitalker `finish` closes the session; Unified `finish` restarts the stream at time zero [multitalker_stream.rs:finish, stream.rs:finish]
- The exact error string "cancelled" is the ABI's cancelled-status marker [multitalker_stream.rs:work]
- Models and `LiveSession` live on one worker; a panic ends the worker, and `Drop` cancels, closes the channel and joins it [multitalker_stream.rs:work, Drop]
- Admission estimates: Unified is 3x compiled weights plus 256 MiB; Multitalker is safetensors bytes plus a quarter plus 512 MiB, an unmeasured estimate [stream.rs:open, multitalker_stream.rs:open]

## Side Effects
- Calls `admit_memory` at open [stream.rs:open, multitalker_stream.rs:open]
- Reads `UnifiedOptions::from_env()` options and model files [stream.rs:open]
- Spawns the named thread `multitalker-stream` [multitalker_stream.rs:open]
- Uses Core ML and the Neural Engine for Unified; Multitalker is CPU only; no Metal [stream.rs, multitalker_stream.rs]

## Verification
- Run: `cargo test -p turbospark-runtime native_audio`
