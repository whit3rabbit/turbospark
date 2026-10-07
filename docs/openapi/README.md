# TurboSpark OpenAPI Specifications

This directory contains the machine-readable OpenAPI 3.1.0 specifications for
the `turbospark-server` HTTP service (`turbospark serve`).

## Specification Files

- `turbospark.openapi.yaml` (and alias `openapi.yaml`): The unified OpenAPI
  3.1.0 specification covering all endpoints registered by the server:
  - Text & Chat: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`
  - Anthropic: `/v1/messages`, `/v1/messages/count_tokens`
  - Embeddings: `/v1/embeddings`, `/api/embeddings`, `/api/embed`
  - Images: `/v1/images/generations`, `/v1/images/edits`
  - Models: `/v1/models`, `/v1/models/{model}`
  - Ollama Compatibility: `/api/tags`, `/api/version`, `/api/show`, `/api/chat`, `/api/generate`
  - Audio: `/v1/audio/transcriptions`, `/v1/audio/translations`, `/v1/audio/transcriptions/realtime`, `/v1/audio/speech`, `/v1/audio/generate`, `/v1/audio/jobs/{id}`, `/v1/audio/jobs/{id}/result`, `/v1/audio/models`, `/v1/audio/models/{id}`
  - System: `/health`, `/v1/metrics`
- `audio.openapi.yaml`: Dedicated specification for audio-only hosts and audio
  routes.

## Contract Verification

Both specifications are tested against the server router in Rust integration
tests (`crates/server/tests/`):
- `openapi_file_lists_exactly_the_audio_routes`: Asserts that `audio.openapi.yaml`
  matches `turbospark_server::audio::AUDIO_ROUTE_PATHS`.
- `openapi_file_lists_all_server_routes`: Asserts that `turbospark.openapi.yaml`
  lists every route registered in `turbospark_server::SERVER_ROUTE_PATHS`.

To run contract verification:

```sh
cargo test -p turbospark-server --test audio_api -- openapi
cargo test -p turbospark-server --test server_api_contract
```

## Using the Specification

You can import `turbospark.openapi.yaml` into:
- Swagger Editor: https://editor.swagger.io/
- Redocly / Redoc CLI: `npx @redocly/cli preview-docs docs/openapi/turbospark.openapi.yaml`
- Postman / Insomnia: Import as OpenAPI 3.1 collection
- Client SDK generators (openapi-generator, fern, or stainless)
