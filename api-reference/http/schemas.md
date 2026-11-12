---
description: Downloadable and modular OpenAPI 3.1.0 specifications for TurboSpark services.
icon: file-code
---

# OpenAPI Specifications

`turbospark-server` provides complete machine-readable OpenAPI 3.1.0 specifications for its HTTP endpoints.

Specifications are organized into dedicated schemas per capability as well as a unified schema covering the complete router.

## Available Specifications

| Specification | Scope | Download |
| --- | --- | --- |
| **Unified Server Spec** | All endpoints: Text, Chat, Ollama, Embeddings, Images, Audio, System | [`turbospark.openapi.yaml`](../openapi/turbospark.openapi.yaml) ([GitHub](https://github.com/whit3rabbit/turbospark/blob/main/docs/openapi/turbospark.openapi.yaml)) |
| **Text and Chat Spec** | Chat completions, raw completions, responses, Anthropic messages, and Ollama | [`text.openapi.yaml`](../openapi/text.openapi.yaml) ([GitHub](https://github.com/whit3rabbit/turbospark/blob/main/docs/openapi/text.openapi.yaml)) |
| **Audio Spec** | Speech-to-text (WAV/WebSocket), Kokoro TTS, MegaTTS3/OmniVoice/Drama 3 voice cloning, MiniMax music, async jobs | [`audio.openapi.yaml`](../openapi/audio.openapi.yaml) ([GitHub](https://github.com/whit3rabbit/turbospark/blob/main/docs/openapi/audio.openapi.yaml)) |
| **Images Spec** | Prompt-to-PNG diffusion generation (`/v1/images/generations`) | [`images.openapi.yaml`](../openapi/images.openapi.yaml) ([GitHub](https://github.com/whit3rabbit/turbospark/blob/main/docs/openapi/images.openapi.yaml)) |
| **Embeddings Spec** | OpenAI-compatible and Ollama-compatible vector embeddings | [`embeddings.openapi.yaml`](../openapi/embeddings.openapi.yaml) ([GitHub](https://github.com/whit3rabbit/turbospark/blob/main/docs/openapi/embeddings.openapi.yaml)) |

## Using the Specifications

### Previewing with Swagger Editor
Upload any specification file directly to [Swagger Editor](https://editor.swagger.io/) or [Scalar](https://scalar.com/).

### Previewing Locally with Redocly
```sh
# Unified API
npx @redocly/cli preview-docs docs/openapi/turbospark.openapi.yaml

# Audio API only
npx @redocly/cli preview-docs docs/openapi/audio.openapi.yaml

# Text & Chat API only
npx @redocly/cli preview-docs docs/openapi/text.openapi.yaml
```

### Generating Client SDKs
Use `openapi-generator` or `fern` to generate type-safe client libraries for your stack:

```sh
npx @openapitools/openapi-generator-cli generate \
  -i docs/openapi/turbospark.openapi.yaml \
  -g typescript-axios \
  -o ./clients/typescript
```
