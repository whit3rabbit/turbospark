---
description: Running and configuring the TurboSpark local HTTP inference server.
icon: server
---

# Server Configuration and Deployment

`turbospark-server` (`turbospark serve`) provides a local HTTP and WebSocket service on Apple Silicon, exposing standardized endpoints compatible with OpenAI, Anthropic, Ollama, and custom audio clients.

## Starting the Server

```sh
# Basic startup with a language model
turbospark-server --model gemma4

# Complete multi-service startup (Text, STT, TTS, Music, Embeddings)
turbospark-server \
  --model gemma4 \
  --stt-model whisper-base-en \
  --tts-model kokoro-82m \
  --music-model minimax-music3-4bit \
  --coreml-embedding-model ~/models/embedding-gemma-2.mlmodelc \
  --port 8080 \
  --api-key "$TURBOSPARK_API_KEY"
```

---

## Server Flags Reference

| Flag | Default | Description |
| --- | --- | --- |
| `--model <path or alias>` | None | Path to `.gturbo` model directory or installed catalog alias. |
| `--port <N>` | `8080` | TCP port to listen on. |
| `--bind <address>` | `127.0.0.1` | Network interface to bind. Pass `127.0.0.1` (loopback) or `tailnet` (binds Tailscale IPv4 interface; requires `--api-key`). |
| `--api-key <string>` | None | Secret key required for authentication. Can also be set via `$TURBOSPARK_API_KEY`. |
| `--stt-model <alias>` | None | Enables speech-to-text endpoints (`/v1/audio/transcriptions`). |
| `--tts-model <alias>` | None | Enables text-to-speech and voice cloning endpoints (`/v1/audio/speech`). |
| `--music-model <alias>`| None | Enables music generation endpoints (`/v1/audio/generate`). |
| `--coreml-embedding-model <path>` | None | Offloads embeddings to Apple Core ML on the Neural Engine. |
| `--expert-slots <N>` | `16` | Bounded slot-cache capacity per layer for MoE models. |
| `--pool-size <N>` | `1` | Number of independent runners opened for the `--model` install. Each runner handles one generation at a time, so raise this to serve concurrent generations. Every runner pays its own memory, and a runner that does not fit refuses at startup. |

---

## Authentication and Security

* **Loopback Mode**: When running without `--api-key`, the server strictly binds to loopback (`127.0.0.1`).
* **Authenticated Mode**: Supplying `--api-key KEY` (or `$TURBOSPARK_API_KEY`) enforces authentication on every route except `GET /health`.
* **Accepted Headers**:
  * `Authorization: Bearer <KEY>`
  * `x-api-key: <KEY>`
* Missing or mismatched keys return `401 Unauthorized`.

---

## Tailnet Binding

To securely access your local Apple Silicon server from other devices (e.g. iPad, laptop, phone) over your private Tailscale network without exposing ports to the public internet:

```sh
turbospark-server \
  --model gemma4 \
  --bind tailnet \
  --api-key "$MY_SECRET_KEY"
```

The server automatically identifies your Mac's Tailscale IPv4 address (`100.x.y.z`) and requires the configured API key on all remote requests.

---

## Health Checks & Prometheus Metrics

### GET /health

Lock-free liveness and readiness probe (always unauthenticated):

```sh
curl http://127.0.0.1:8080/health
```

```json
{
  "status": "ok",
  "state": "ready"
}
```

* `state: "ready"` indicates at least one model is attached and ready to serve requests.
* `state: "empty"` indicates the server is running without any attached models.

### GET /v1/metrics

Exposes Prometheus text-format metrics covering:
* Total requests, success counts, and error status codes.
* Generation token throughput and time-to-first-token (TTFT) latency histograms.
* Current queue occupancy and worker concurrency.
