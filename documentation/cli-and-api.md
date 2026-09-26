---
description: "Use the TurboSpark command-line tools and local HTTP API."
icon: sliders
---

# CLI and local API

## Model commands

```sh
turbospark-model list
turbospark-model recommend
turbospark-model info <alias>
turbospark-model pull <alias>
```

Run an interactive chat from the command line:

```sh
turbospark-check --model <alias> --chat
```

## Local server

Start the server with a catalog model:

```sh
turbospark-server --model gemma4
```

By default, the server listens on `127.0.0.1:8080` and binds to loopback. Common routes include:

- `POST /v1/chat/completions` for OpenAI-compatible chat requests
- `POST /v1/messages` for Anthropic-compatible messages requests
- `POST /v1/completions` for raw prompt completions
- `POST /v1/responses` for the OpenAI Responses API
- `POST /v1/messages/count_tokens` to count prompt tokens without generating
- `GET /v1/models` and `GET /v1/models/{model}` for model information
- `GET /health` for liveness

The server also has embeddings and Ollama-compatible routes. Which model capabilities are available depends on the loaded install. Each runner handles one generation at a time; `--pool-size` opens multiple runners when concurrent generation is needed.

If you bind to a Tailscale address with `--bind tailnet`, configure API-key authentication. The server does not provide TLS. See [local API examples](call-the-local-api.md) for request bodies and headers.

For full command options, see the [CLI and server reference in the source repository](https://github.com/whit3rabbit/turbospark/blob/main/docs/CLI.md).
