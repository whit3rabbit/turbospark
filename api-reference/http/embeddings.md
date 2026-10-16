---
description: Text embedding endpoints for dense vector representations.
icon: cubes
---

# Embeddings API

`turbospark-server` provides OpenAI-compatible and Ollama-compatible vector embedding endpoints. Embeddings are computed locally using Metal GPU, Apple Core ML (`EmbeddingGemma 2`), or MLX.

Dedicated OpenAPI specification: [`embeddings.openapi.yaml`](../../docs/openapi/embeddings.openapi.yaml)

## OpenAI Embeddings

### POST /v1/embeddings

Generates dense embedding vectors for single strings or batches of text.

```sh
curl http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "bge-small-en-v1.5",
    "input": [
      "TurboSpark runs MoE models locally on Apple Silicon.",
      "Fine-grained expert streaming limits memory footprint."
    ],
    "encoding_format": "float"
  }'
```

#### Request Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | string | required | Pinned embedding model alias. |
| `input` | string or array | required | Single text string or array of strings. Maximum 2,048 inputs or 1 MiB total payload. |
| `encoding_format`| string | `float` | Format of embedding vectors: `float` (JSON number array) or `base64` (packed float32 bytes). |
| `dimensions` | integer | optional | Dimensionality truncation for models supporting Matryoshka representations. |

#### Response Format

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0142, -0.0521, 0.0831, "..."]
    },
    {
      "object": "embedding",
      "index": 1,
      "embedding": [0.0215, -0.0489, 0.0912, "..."]
    }
  ],
  "model": "bge-small-en-v1.5",
  "usage": {
    "prompt_tokens": 28,
    "total_tokens": 28
  }
}
```

---

## Ollama Embeddings

### POST /api/embeddings

Legacy single-string Ollama embedding route.

```sh
curl http://127.0.0.1:8080/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-small-en-v1.5",
    "prompt": "Local vector search embeddings."
  }'
```

### POST /api/embed

Standard Ollama batch embedding endpoint.

```sh
curl http://127.0.0.1:8080/api/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-small-en-v1.5",
    "input": ["First sentence", "Second sentence"]
  }'
```
