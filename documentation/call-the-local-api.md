---
description: "Send chat requests to the TurboSpark local HTTP server."
---

# Call the local API

Start the server with a model that is installed on this machine:

```sh
turbospark-server --model gemma4
```

The default address is `http://127.0.0.1:8080`. First, list the model IDs available from the running server:

```sh
curl http://127.0.0.1:8080/v1/models
```

Use an ID from that response in a request.

## OpenAI-compatible chat

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \\
  -d '{
    "model": "gemma4",
    "messages": [{"role": "user", "content": "Write a short greeting."}],
    "max_tokens": 64
  }'
```

## Anthropic-compatible messages

```sh
curl http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \\
  -H "anthropic-version: 2023-06-01" \\
  -d '{
    "model": "gemma4",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "Write a short greeting."}]
  }'
```

Replace `gemma4` with a model ID returned by `GET /v1/models`. The server also supports the OpenAI Responses API, raw prompt completions, Anthropic token counting, embeddings, and Ollama-compatible routes.

If the server was started with API-key authentication, send the same key as `Authorization: Bearer <key>` or `x-api-key: <key>`. `GET /health` does not require authentication.

For memory, storage, and context guidance, read [memory and capacity](memory-and-capacity.md). For additional server flags and client behavior, see the [CLI and server reference](https://github.com/whit3rabbit/turbospark/blob/main/docs/CLI.md).
