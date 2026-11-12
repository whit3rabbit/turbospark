---
description: Transcribe speech from WAV files or live microphone streams with Whisper, Qwen3-ASR and Parakeet on Apple Silicon, with speaker diarization and meeting notes in the app.
icon: microphone
---

# Speech to Text

TurboSpark provides fast local speech transcription with segment timestamps, punctuation, language detection, and, in the desktop app, speaker labels and meeting notes.

## Supported Models

Run `turbospark-model list-speech` for the Whisper rows and `turbospark-model list-audio` for everything else. The `RUNS IN` column of `list-audio` says whether a row runs in the Rust engine (CLI, server, app) or only in the app.

| Model | Catalog alias | Runs in | Notes |
| --- | --- | --- | --- |
| **Whisper** | `whisper-base` (279 MiB) | Rust (Metal) | Pinned OpenAI Whisper on native Metal kernels. `turbospark-model list-speech` names the other recognized layouts (`whisper-tiny-en`, `whisper-base-en`, `whisper-small-en` and the `-mlx-8bit` / `-mlx-4bit` conversions) with their pinned repositories; a directory holding one of those layouts can be passed to `--stt-model` as a path. |
| **Qwen3-ASR 0.6B** | `qwen3-asr-06b-8bit` | Rust (Metal decoder) | Compact multilingual recognition with low latency. |
| **Parakeet Redux** | `parakeet-redux-coreml` (210 MiB) | Rust (Core ML, macOS 15+) | Batch transcription. Token-identical to the CPU Redux decoder on the gate clips. |
| **Parakeet Ultra** | `parakeet-ultra-coreml` (603 MiB) | Rust (Core ML, macOS 15+) | Batch transcription, larger encoder. |
| **Parakeet Unified (streaming)** | `parakeet-unified-en-coreml` (581 MiB) | Rust (Core ML, macOS 15+) | Live transcription only: the int8 encoder runs on the Neural Engine with the log-mel and RNNT steps on the CPU. English only. A batch open refuses it. |
| **Speaker diarization** | `speaker-diarization-coreml` (21 MiB) | Rust (Core ML, macOS 15+) | pyannote community-1 export: who-spoke-when turns on a 16.875 ms grid. |
| **Parakeet TDT Japanese** | `parakeet-ja-coreml` (590 MiB) | App only (FluidAudio) | 30 s overlapping windows. |
| **Paraformer Mandarin** | `paraformer-zh-coreml` (212 MiB) | App only (FluidAudio) | 15 s bounded windows. Installs behind the FunASR Model License acknowledgment (`--accept funasr_model_license`). |
| **Silero VAD** | `silero-vad-coreml` (1 MiB) | Installs only | Speech detection. Pinned and verified, but no adapter runs it yet. |

The two FluidAudio rows load inside the app through the FluidAudio package (pinned 0.15.5, held offline with a blocked model registry), are validated against their recorded export signature, and answer the Transcribe page and the refinement of finished recordings. They are app-only: the server refuses those aliases in load and attach.

Also in the library layer, not yet reachable from the app or the FFI: Nemotron-3 diarization coupled to Multitalker Parakeet (`crates/audio`'s `stt/multitalker_asr` and `vad/nemotron_diarization`) for meeting transcription, matching NeMo's output on the reference clip on CPU.

### Installing

```sh
turbospark-model pull-audio whisper-base
turbospark-model pull-audio parakeet-redux-coreml
turbospark-model pull-audio speaker-diarization-coreml
turbospark-model pull-audio paraformer-zh-coreml --accept funasr_model_license
```

---

## Audio Format Requirements

* **Supported Input**: WAV files only (PCM signed 8/16/24/32-bit integer or float32). Sample rates from 8 kHz to 48 kHz (automatically resampled to 16 kHz mono).
* **Unsupported Formats**: Compressed files (MP3, OGG, FLAC, AAC) are refused by the server with `415 Unsupported Media Type`. Convert them first with `ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav`.
* **Limits**: Up to 128 MiB or 30 minutes per request.

---

## Transcribing in the Desktop App

Speech-to-Text is one pane with three modes, picked from a pill switcher. Each mode lists its own takes, files them in its own collection, and keeps one model per job (Live Text, Final Transcript, Speakers), shown in a header popover with its state. **Installed** has an Audio view for the rows.

| Mode | Captures | Typical use |
| --- | --- | --- |
| **Voice Notes** | The microphone | Dictation and memos. |
| **Meeting** | The microphone and the app playing the call | Calls and meetings, with speaker labels. |
| **Any Audio** | Another app's audio, or an imported file | Podcasts, videos, recordings. |

1. Pick a mode and, for the microphone modes, your input device.
2. Click **Record**. Live text appears as you speak when a live model is attached (Parakeet Unified streaming on Core ML, or Whisper).
3. Click **Stop**. The final pass produces the transcript with timestamps, which can be exported as Text, SRT, WebVTT, or JSON.

System audio on macOS 14.2 and later is captured with a CoreAudio process tap, which needs no Screen Recording permission and shows no purple recording indicator; older systems fall back to ScreenCaptureKit. Recording access is asked for only after you choose **Allow** in the access sheet, and declining leaves import, transcription and editing usable.

### Name the Speakers

After a Meeting take's final pass labels its speakers, a sheet lists each voice with its talk time and longest line, plays a sample, and lets you name it, merge it into another voice, or separate a merged voice. The transcript's **Name Speakers** button opens the same sheet later. One name per voice reaches every line, revision and export (JSON carries both the name and the raw id), and clearing a name restores the default label.

* A take longer than the diarizer's 30 minutes is labeled in parts that share 5 minutes. A later part's voice takes an earlier label only when at least 8 s and 70% of the shared time agree; otherwise it starts a new speaker.
* The local speaker ("You") is isolated by microphone energy.
* Cross-meeting recognition: a 64-band log-mel voiceprint per voice lets the naming sheet suggest a name you gave the same voice in an earlier meeting, in one click.

### Notes, summary and custom vocabulary

A finished take has a segmented detail switcher between the transcript and its notes. **Summarize meeting** writes a summary with a local model, and a recommendation badge marks a model suited to it. Timestamp jump chips lead from the notes back into the recording. Summaries only ever write the take's summary field; transcript segments are never changed.

A persistent **custom vocabulary** (names, product terms, jargon) is applied to every transcript with word-boundary-safe normalization and is managed from its own sheet.

---

## Transcribing via HTTP API

Start the server with one or more `--stt-model` rows: Whisper, Qwen3-ASR, or Parakeet Redux or Ultra on Core ML (macOS 15 or newer). Diarization and the streaming Parakeet Unified row are reached through the native audio surface (the C ABI and the app), not the HTTP routes.

### File Transcription

```sh
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer $KEY" \
  -F file=@recording.wav \
  -F model=whisper-base \
  -F response_format=verbose_json
```

Output includes overall text and segment breakdown:

```json
{
  "task": "transcribe",
  "language": "english",
  "duration": 4.82,
  "text": "TurboSpark runs entirely on Apple Silicon.",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.82,
      "text": " TurboSpark runs entirely on Apple Silicon."
    }
  ]
}
```

### Realtime Streaming via WebSocket

Connect to the WebSocket endpoint for continuous live audio streaming:

```sh
websocat -H="Authorization: Bearer $KEY" \
  "ws://127.0.0.1:8080/v1/audio/transcriptions/realtime?model=whisper-base"
```

* Send raw 16 kHz mono PCM frames (`s16le`).
* Send `{"type":"commit"}` to transcribe buffered audio.
* Server returns `{"type":"complete","text":"...","segments":[...]}`.
* Send `{"type":"reset"}` to discard uncommitted frames.
