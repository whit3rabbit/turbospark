---
description: OpenAI, Anthropic, and Ollama compatible text generation and chat endpoints.
icon: message-lines
---

# Text and Chat API

`turbospark-server` provides standardized text and chat completions. It supports streaming (Server-Sent Events), reasoning tokens, tool calling, and legacy raw prompt completions.

Dedicated OpenAPI specification: [`text.openapi.yaml`](../openapi/text.openapi.yaml)

## OpenAI Chat Completions

### POST /v1/chat/completions

Standard OpenAI chat completion endpoint.

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "gemma4",
    "messages": [
      {"role": "system", "content": "You are a concise engineering assistant."},
      {"role": "user", "content": "Explain how speculative decoding works."}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

#### Streaming (SSE)

Set `"stream": true` to receive `text/event-stream` chunks:

```sh
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "gemma4",
    "messages": [{"role": "user", "content": "Write a short poem about Apple Silicon."}],
    "stream": true
  }'
```

Each event payload is formatted as `data: {"id":"...","choices":[{"delta":{"content":"..."}}]}`, ending with `data: [DONE]`.

#### Structured Tool Calling

Pass tools in standard OpenAI function calling format:

```json
{
  "model": "gemma4",
  "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather in a given city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"}
          },
          "required": ["city"]
        }
      }
    }
  ]
}
```

The server parses native tool calls and applies tool-call guardrails with rescue formatting when necessary.

---

## OpenAI Legacy Completions

### POST /v1/completions

Raw prompt completion without chat templating. Useful for base models and specialized rewriters such as `humanizer-12b`.

```sh
curl http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "humanizer-12b",
    "prompt": "The automated framework facilitates enhanced outcomes.",
    "max_tokens": 128
  }'
```

---

## OpenAI Responses API

### POST /v1/responses

Item-based conversational interface following the OpenAI Responses specification.

```sh
curl http://127.0.0.1:8080/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "gemma4",
    "input": "Summarize this release in one sentence."
  }'
```

---

## Anthropic Messages API

### POST /v1/messages

Anthropic-compatible endpoint. Automatically bridges Anthropic wire messages into TurboSpark generation kernels.

```sh
curl http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "gemma4",
    "max_tokens": 256,
    "system": "Respond only with bullet points.",
    "messages": [
      {"role": "user", "content": "List three advantages of fine-grained MoE streaming."}
    ]
  }'
```

### POST /v1/messages/count_tokens

Counts the tokens of a message payload without generating a reply:

```sh
curl http://127.0.0.1:8080/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "gemma4",
    "messages": [{"role": "user", "content": "Sample prompt"}]
  }'
```

---

## Ollama Compatibility API

For tools and agents configured to use Ollama:

* `GET /api/tags`: Returns available models in Ollama JSON format.
* `GET /api/version`: Returns server version string.
* `POST /api/show`: Inspects model metadata (`{"name": "gemma4"}`).
* `POST /api/chat`: Ollama chat endpoint with NDJSON streaming.
* `POST /api/generate`: Ollama prompt generation with NDJSON streaming.
