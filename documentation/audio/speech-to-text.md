---
description: Transcribe speech from WAV files or live microphone streams using Whisper and Qwen3-ASR on Apple Silicon.
icon: microphone
---

# Speech to Text

TurboSpark provides fast local speech transcription with segment timestamps, punctuation, and language detection.

## Supported Models

* **Whisper** (`whisper-tiny-en`, `whisper-base-en`, `whisper-small-en`): Pinned OpenAI Whisper architecture running on native Metal GPU kernels. Also available in 4-bit and 8-bit MLX quantizations.
* **Qwen3-ASR 0.6B** (`qwen3-asr-06b-8bit`): Compact streaming speech recognition model with low latency.

---

## Audio Format Requirements

* **Supported Input**: WAV files only (PCM signed 8/16/24/32-bit integer or float32). Sample rates from 8 kHz to 48 kHz (automatically resampled to 16 kHz mono).
* **Unsupported Formats**: Compressed files (MP3, OGG, FLAC, AAC) are refused by the server with `415 Unsupported Media Type`. Convert them first with `ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav`.
* **Limits**: Up to 128 MiB or 30 minutes per request.

---

## Transcribing in the Desktop App

1. Open the **Audio Studio** workspace.
2. Select your microphone in the audio input dropdown.
3. Click **Record**. A live transcription preview appears in real time as speech segments are committed.
4. Click **Stop**. The complete transcript is saved with timestamps and can be exported as Plain Text, VTT, or SRT.

---

## Transcribing via HTTP API

### File Transcription

```sh
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer $KEY" \
  -F file=@recording.wav \
  -F model=whisper-base-en \
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
  "ws://127.0.0.1:8080/v1/audio/transcriptions/realtime?model=whisper-base-en"
```

* Send raw 16 kHz mono PCM frames (`s16le`).
* Send `{"type":"commit"}` to transcribe buffered audio.
* Server returns `{"type":"complete","text":"...","segments":[...]}`.
* Send `{"type":"reset"}` to discard uncommitted frames.
