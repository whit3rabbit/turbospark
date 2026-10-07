---
description: "HTTP REST and WebSocket API wire contract for turbospark-server."
icon: network-wired
---

# HTTP REST API Reference

`turbospark-server` (`turbospark serve`) exposes local Apple Silicon inference
over HTTP and WebSockets. It implements standardized OpenAI, Anthropic, Ollama,
and dedicated audio endpoints.

For machine-readable OpenAPI 3.1.0 specifications, see:
- [Unified Server OpenAPI Spec](../docs/openapi/turbospark.openapi.yaml) (and alias `openapi.yaml`)
- [Audio OpenAPI Spec](../docs/openapi/audio.openapi.yaml)

## Server Configuration and Addressing

```sh
# Start server with an LLM and audio models
turbospark serve \
  --model ~/models/gemma4.gturbo \
  --stt-model whisper-base-en \
  --tts-model ~/models/kokoro-82m \
  --music-model minimax-music3-4bit \
  --port 8080 \
  --api-key "$KEY"
```

- Default address: `http://127.0.0.1:8080` (loopback).
- `--bind tailnet`: Binds Tailscale IPv4 interface; requires `--api-key`.
- Authentication: Opt-in via `--api-key KEY`. When enabled, every route except
  `GET /health` requires `x-api-key: KEY` or `Authorization: Bearer KEY`.
  Missing or invalid keys return 401.

## Endpoint Index

| Method | Path | Protocol / Purpose |
|---|---|---|
| GET | `/health` | Liveness and readiness probe (unauthenticated) |
| POST | `/v1/chat/completions` | OpenAI chat completion (text, reasoning, tools, vision) |
| POST | `/v1/completions` | OpenAI legacy raw prompt completion |
| POST | `/v1/responses` | OpenAI Responses API |
| POST | `/v1/messages` | Anthropic messages completion |
| POST | `/v1/messages/count_tokens` | Anthropic prompt token counting |
| GET | `/v1/models` | OpenAI-compatible model listing |
| GET | `/v1/models/{model}` | Model detail inspection |
| POST | `/v1/embeddings` | OpenAI-compatible text embeddings |
| POST | `/v1/images/generations` | Prompt-to-PNG diffusion generation |
| POST | `/v1/images/edits` | Unsupported (returns 501) |
| GET | `/api/tags` | Ollama model tags list |
| GET | `/api/version` | Ollama server version probe |
| POST | `/api/show` | Ollama model metadata |
| POST | `/api/chat` | Ollama chat (NDJSON streaming) |
| POST | `/api/generate` | Ollama prompt generation (NDJSON streaming) |
| POST | `/api/embeddings` | Ollama legacy single embedding |
| POST | `/api/embed` | Ollama batch embeddings |
| POST | `/v1/audio/transcriptions` | Speech-to-text (Whisper, Qwen3-ASR) |
| POST | `/v1/audio/translations` | Unsupported (returns 501) |
| GET | `/v1/audio/transcriptions/realtime` | WebSocket realtime 16 kHz audio streaming |
| POST | `/v1/audio/speech` | Text-to-speech synthesis (Kokoro) |
| POST | `/v1/audio/generate` | Music generation (MiniMax Music 0.3) |
| GET | `/v1/audio/jobs/{id}` | Status of async transcription/generation job |
| DELETE | `/v1/audio/jobs/{id}` | Cancel or delete audio job |
| GET | `/v1/audio/jobs/{id}/result` | Result payload of async audio job |
| GET | `/v1/audio/models` | List loaded audio models |
| POST | `/v1/audio/models` | Dynamically attach installed audio model |
| DELETE | `/v1/audio/models/{id}` | Unload audio model |
| GET | `/v1/metrics` | Prometheus metrics text exposition |

## Health Check

### GET /health

Returns server liveness. Always exempt from authentication.

```sh
curl http://127.0.0.1:8080/health
```

```json
{
  "status": "ok",
  "state": "ready"
}
```

`state` is `"ready"` if at least one model (chat, image, or audio) is attached;
otherwise `"empty"`.

## Text Generation and Chat

### POST /v1/chat/completions

OpenAI-compatible chat completions.

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "gemma4",
    "messages": [
      {"role": "user", "content": "Explain KV caching in two sentences."}
    ],
    "max_tokens": 128
  }'
```

Set `"stream": true` for Server-Sent Events (SSE) streaming.

### POST /v1/messages

Anthropic-compatible messages API.

```sh
curl http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "gemma4",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

## Embeddings

### POST /v1/embeddings

Generates dense embeddings for strings or batches.

```sh
curl http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "bge-small-en-v1.5",
    "input": ["First sentence", "Second sentence"],
    "encoding_format": "float"
  }'
```

Limits: Up to 2048 inputs or 1 MiB total text per request. `encoding_format`
supports `"float"` and `"base64"`.

## Image Generation

### POST /v1/images/generations

Prompt-to-PNG generation using attached diffusion models (e.g. Z-Image).

```sh
curl http://127.0.0.1:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "z-image-turbo",
    "prompt": "a snowy mountain peak at sunrise",
    "size": "1024x1024",
    "n": 1
  }'
```

- Returns `data[].b64_json` PNG data.
- `size`: `WIDTHxHEIGHT` between 512 and 1024 in multiples of 16.
- Header `x-turbospark-seed` returns the resolved generation seed.

## Audio Endpoints

### POST /v1/audio/transcriptions

Transcribes audio from WAV files using Whisper or Qwen3-ASR models.

```sh
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer $KEY" \
  -F file=@recording.wav \
  -F model=whisper-base-en \
  -F response_format=verbose_json
```

- Input: WAV only (PCM 8/16/24/32-bit or float32), up to 128 MiB and 30 minutes.
- `response_format`: `json` (default), `verbose_json`, `text`, `srt`, `vtt`.
- `stream=true`: Returns NDJSON segment stream.
- `async=true`: Enqueues an asynchronous job and returns 202 with `job_id`.

### GET /v1/audio/transcriptions/realtime (WebSocket)

Utterance-level streaming over WebSocket. Connect with:

```sh
websocat -H="Authorization: Bearer $KEY" \
  "ws://127.0.0.1:8080/v1/audio/transcriptions/realtime?model=whisper-base-en"
```

Send raw binary 16 kHz mono PCM frames (`s16le`), then send text frame
`{"type":"commit"}` to transcribe buffered audio.

### POST /v1/audio/speech

Synthesizes speech using Kokoro-82M (English, voice `af_heart`).

```sh
curl http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "kokoro-82m",
    "input": "TurboSpark delivers high performance local inference.",
    "response_format": "wav"
  }' \
  -o speech.wav
```

Output: 24 kHz mono audio (`wav` or raw `pcm`).

### POST /v1/audio/generate

Music generation with MiniMax Music 0.3.

```sh
curl http://127.0.0.1:8080/v1/audio/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "minimax-music3-4bit",
    "prompt": "acoustic guitar melody",
    "audio_length": 15,
    "async": true
  }'
```

Full generation runs take significant GPU time; prefer `async: true` and poll
`/v1/audio/jobs/{id}/result`.

### Asynchronous Audio Jobs

- `GET /v1/audio/jobs/{id}`: Reports job status (`running`, `succeeded`, `failed`, `cancelled`).
- `GET /v1/audio/jobs/{id}/result`: Retrieves the output once succeeded (409 while running, 410 if cancelled).
- `DELETE /v1/audio/jobs/{id}`: Cancels running job or purges result.

## Metrics

### GET /v1/metrics

Prometheus text exposition format. Exposes latency, request counters, error
rates, in-flight work, and queue state for server workers.
