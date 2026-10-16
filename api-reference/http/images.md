---
description: Image generation HTTP endpoints for local diffusion models.
icon: image
---

# Images API

`turbospark-server` provides OpenAI-compatible image generation endpoints for local diffusion models running on Apple Silicon (e.g. Z-Image Turbo and Qwen-Image 2.1 Turbo).

Dedicated OpenAPI specification: [`images.openapi.yaml`](../openapi/images.openapi.yaml)

## Generate Image

### POST /v1/images/generations

Generates an image from a descriptive text prompt.

```sh
curl http://127.0.0.1:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "z-image-turbo",
    "prompt": "cinematic shot of a robotic hummingbird resting on a cherry blossom branch, sharp focus, 8k",
    "size": "1024x1024",
    "n": 1,
    "seed": 100
  }'
```

#### Request Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | string | required | Image model alias (e.g., `z-image-turbo`, `qwen-image-2.1-turbo`). |
| `prompt` | string | required | Detailed text description of the image to generate. |
| `size` | string | `1024x1024` | Resolution format `WIDTHxHEIGHT`. Supported dimensions range from 512 to 1024 (multiples of 16). |
| `n` | integer | `1` | Number of images to generate (1 to 4). |
| `seed` | integer | random | Deterministic seed for repeatable generations. |
| `response_format`| string | `b64_json`| Output format. Returns base64-encoded PNG image data. |

#### Response Format

```json
{
  "created": 1729080000,
  "data": [
    {
      "b64_json": "iVBORw0KGgoAAAANSUhEUgAABAAAAAQACA..."
    }
  ]
}
```

The response includes the header `x-turbospark-seed` containing the seed used.

#### Saving Output with jq and base64

```sh
curl -s http://127.0.0.1:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{"model": "z-image-turbo", "prompt": "mountain landscape"}' \
  | jq -r '.data[0].b64_json' | base64 -d > output.png
```

---

## Image Edits

### POST /v1/images/edits

Image editing is currently unsupported and returns `501 Not Implemented`.
