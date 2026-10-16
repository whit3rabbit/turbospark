---
description: Audio HTTP endpoints for speech-to-text, text-to-speech, voice cloning, and music generation.
icon: waveform
---

# Audio API

`turbospark-server` provides dedicated audio endpoints when started with `--stt-model`, `--tts-model`, or `--music-model`.

Dedicated OpenAPI specification: [`audio.openapi.yaml`](../../docs/openapi/audio.openapi.yaml)

## Audio Constraints and Formats

* **Input audio**: WAV only (PCM 8/16/24/32-bit or float32). Compressed formats (MP3, OGG, FLAC) return `415 Unsupported Media Type`.
* **Output audio**: 24 kHz or 44.1 kHz mono/stereo WAV or raw PCM.
* **Concurrency**: One audio request runs at a time per loaded model. Use `async: true` for long jobs.

---

## Speech to Text (Transcription)

### POST /v1/audio/transcriptions

Transcribes audio from a WAV file using Whisper or Qwen3-ASR models.

```sh
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer $KEY" \
  -F file=@audio.wav \
  -F model=whisper-base-en \
  -F response_format=verbose_json
```

#### Parameters

* `file`: Multipart WAV file or binary body with `Content-Type: audio/wav`.
* `model`: Model alias (`whisper-tiny-en`, `whisper-base-en`, `whisper-small-en`, `qwen3-asr-06b-8bit`).
* `response_format`: `json` (default), `verbose_json`, `text`, `srt`, `vtt`.
* `stream`: When `true`, returns NDJSON stream of transcribed segments.
* `async`: When `true`, accepts as an asynchronous background job (`202 Accepted`).

---

## Realtime Transcription (WebSocket)

### GET /v1/audio/transcriptions/realtime

Low-latency utterance streaming over WebSockets:

```sh
websocat -H="Authorization: Bearer $KEY" \
  "ws://127.0.0.1:8080/v1/audio/transcriptions/realtime?model=whisper-base-en"
```

1. Send binary frames containing 16 kHz mono signed 16-bit PCM (`s16le`).
2. Send a text frame `{"type":"commit"}` to transcribe buffered frames.
3. The server responds with `{"type":"complete","text":"...","segments":[...]}`.
4. Send `{"type":"reset"}` to discard uncommitted audio.

---

## Text to Speech & Voice Cloning

### POST /v1/audio/speech

Synthesizes speech from input text. Supports Kokoro-82M, Qwen3-TTS, and MegaTTS3 zero-shot voice cloning.

#### Standard Speech Synthesis (Kokoro-82M)

```sh
curl http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "kokoro-82m-bf16",
    "input": "TurboSpark delivers high-performance local AI on macOS.",
    "voice": "af_heart",
    "speed": 1.0,
    "response_format": "wav"
  }' \
  -o speech.wav
```

#### Zero-Shot Voice Cloning (MegaTTS3)

Clones a target voice from a short reference recording (1 to 28 seconds):

```sh
# Convert reference audio to base64
REF_B64=$(base64 -i reference.wav)

curl http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "megatts3-f32",
    "input": "This speech was synthesized using a local cloned voice model.",
    "reference_audio": "'"$REF_B64"'",
    "consent": true,
    "steps": 16,
    "p_w": 1.4,
    "t_w": 3.0,
    "seed": 42
  }' \
  -o cloned.wav
```

* **`consent: true`**: Required for voice cloning requests. The caller confirms authorization to use the reference voice.
* **`steps`**: Diffusion steps: Fast (8), Balanced (16, default), Best (32).
* **`p_w`**: Intelligibility weight (default 1.4).
* **`t_w`**: Similarity weight (default 3.0).

---

## Music Generation

### POST /v1/audio/generate

Generates music tracks from text descriptions using MiniMax Music 3 on Apple Silicon GPU.

```sh
curl http://127.0.0.1:8080/v1/audio/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "minimax-music3-4bit",
    "prompt": "upbeat electronic synthwave with driving drums and retro bass",
    "lyrics": "[instrumental]",
    "audio_length": 30,
    "async": true
  }'
```

Returns `202 Accepted` with a `job_id`:

```json
{
  "job_id": "job_01h7x8...",
  "status": "running"
}
```

---

## Asynchronous Audio Jobs

For long-running transcriptions or music generations:

* **Check Status**: `GET /v1/audio/jobs/{id}` returns `running`, `succeeded`, or `failed`.
* **Download Result**: `GET /v1/audio/jobs/{id}/result` downloads the generated WAV file.
* **Cancel / Purge**: `DELETE /v1/audio/jobs/{id}` cancels active work or purges completed results.

---

## Audio Model Management

* `GET /v1/audio/models`: Lists attached audio models and supported tasks (`stt`, `tts`, `music`).
* `POST /v1/audio/models`: Dynamically attaches an audio model (`{"model": "whisper-small-en"}`).
* `DELETE /v1/audio/models/{id}`: Unloads an attached audio model from memory.
