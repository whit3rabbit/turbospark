---
description: "Send chat requests to the TurboSpark local HTTP server."
---

# Call the local API

Start the server with an installed model:

```sh
turbospark-server --model gemma4
```

The default address is `http://127.0.0.1:8080`. List the models available from the running server:

```sh
curl http://127.0.0.1:8080/v1/models
```

## OpenAI-compatible chat

Use a model ID returned by `/v1/models`:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4",
    "messages": [{"role": "user", "content": "Write a short greeting."}],
    "max_tokens": 64
  }'
```

## Anthropic-compatible messages

```sh
curl http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "gemma4",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "Write a short greeting."}]
  }'
```

If the server was started with an API key, send it as `Authorization: Bearer <key>` or `x-api-key: <key>`. The server also supports the OpenAI Responses API, raw prompt completions, token counting, embeddings, and Ollama-compatible routes.

For model sizing, see [memory and capacity](memory-and-capacity.md). For full flags and routes, see the [CLI and server reference](https://github.com/whit3rabbit/turbospark/blob/main/docs/CLI.md).
