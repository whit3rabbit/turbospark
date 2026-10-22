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
  --tts-model kokoro-82m-bf16 \
  --music-model minimax-music3-4bit \
  --embedding-model embeddinggemma-2-coreml \
  --port 8080 \
  --api-key "$TURBOSPARK_API_KEY"
```

---

## Server Flags Reference

| Flag | Default | Description |
| --- | --- | --- |
| `--model <path or alias>` | None | Path to `.gturbo` model directory or installed catalog alias. Repeat to attach distinct generation models. |
| `--port <N>` | `8080` | TCP port to listen on. |
| `--bind <address>` | `127.0.0.1` | Network interface to bind. Pass `127.0.0.1` (loopback) or `tailnet` (binds Tailscale IPv4 interface; requires `--api-key`). |
| `--api-key <string>` | None | Secret key required for authentication. Can also be set via `$TURBOSPARK_API_KEY`. |
| `--stt-model <alias>` | None | Enables speech-to-text endpoints (`/v1/audio/transcriptions`). Takes a speech install directory or alias: Whisper (`whisper-base`), Qwen3-ASR, or Parakeet Redux/Ultra on Core ML (macOS 15+). Repeat for several. With no `--model`, starts an audio-only server. |
| `--tts-model <alias>` | None | Enables speech synthesis and voice cloning (`/v1/audio/speech`). Takes Kokoro (`kokoro-82m-bf16`, portable CPU), MegaTTS3, OmniVoice, or Fish Audio Drama 3. Repeat for several. |
| `--music-model <alias>`| None | Enables music generation (`/v1/audio/generate`). Takes a MiniMax Music 3 row. macOS and Metal only. |
| `--embedding-model <path or alias>` | None | Serves `/v1/embeddings` alongside generation. Takes a `.safetensors` directory or a catalog alias. A Core ML EmbeddingGemma 2 install such as `embeddinggemma-2-coreml` is detected from its directory layout and runs through Core ML. Its first load compiles for the Neural Engine and can take minutes. Embedding models are checked by family before they load: only `bert` or `xlm-roberta` checkpoints with CLS pooling are accepted, and a missing `model_type` is refused rather than assumed. A mean-pooled directory returns an explicit refusal instead of wrong vectors. |
| `--expert-cache-slots <auto or N>` | `auto` | Routed-expert cache slots per layer for MoE models. `N` is one of 8, 16, 24, 32, 48, 64, 96, 128, 192, or 256. |
| `--kv-bits <off or N>` | `off` | TurboQuant KV-cache quantization: `off`, `2`, `3`, `3.5`, or `4`. An unsupported head dimension or family refuses the flag at open rather than falling back to FP16. |
| `--speculative <auto, off, or N>` | `auto` | Speculative decoding: `auto`, `off`, or a block size 1-15. Applies to temperature-0 requests only; other requests decode sequentially. (`turbospark-check` additionally accepts `adaptive[:MAX]`.) |
| `--speculative-drafter <auto, mtp, or dflash>` | `auto` | Which drafter `--speculative` drives. `auto` reports a DFlash drafter but does not enable it. |
| `--pool-size <N>` | `1` | Number of independent runners opened for the `--model` install. Each runner handles one generation at a time, so raise this to serve concurrent generations. Every runner pays its own memory, and a runner that does not fit refuses at startup. |

A model that fails to open refuses the whole server start, so a server that came up has every endpoint it was asked for.

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

---

## Audio Model Management

Audio models can also be attached and detached at runtime instead of only at startup:

| Route | Description |
| --- | --- |
| `GET /v1/audio/models` | List the attached audio models with their `id`, `task`, and `capabilities`. Music models also carry their context and frame limits. Audio models are appended to `GET /v1/models` with a `capabilities` field as well. |
| `POST /v1/audio/models` | Attach an installed audio model: `{"model": "<installed alias>"}`. A filesystem path is refused, so an API caller cannot probe directories. |
| `DELETE /v1/audio/models/{id}` | Detach one audio model and free its worker. |

```sh
# Attach a TTS row at runtime
curl -X POST http://127.0.0.1:8080/v1/audio/models \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{"model": "omnivoice-8bit"}'
```
