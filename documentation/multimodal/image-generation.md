---
description: Local prompt-to-image diffusion generation using Z-Image Turbo and Qwen-Image 2.1 Turbo on Apple Silicon.
icon: image
---

# Image Generation

TurboSpark generates high-fidelity images directly on your Mac with zero cloud dependencies. It supports two state-of-the-art diffusion models optimized for Apple Silicon unified memory: **Z-Image Turbo** and **Qwen-Image 2.1 Turbo**.

## Supported Image Models

| Model | Alias | Precision | Steps | 1024 x 1024 Time (M4 Max) | Peak RAM | License |
| --- | --- | --- | ---: | --- | ---: | --- |
| **Z-Image Turbo** | `z-image-turbo-mlx-8bit` | 8-bit | 9 | ~1 to 1.5 minutes | 12.8 GiB | Apache-2.0 |
| **Qwen-Image 2.1 Turbo** | `qwen-image-2.1-turbo-mlx-mixed-4bit` | Mixed 4-bit / 8-bit | 8 | ~1.5 minutes | 11.2 GiB | Qwen Research (Non-commercial) |

*Smaller Macs*: Z-Image Turbo also provides 4-bit, 2-bit, and `q4` exports to run on Macs with 16 GB or 18 GB of memory. Qwen-Image 2.1 Turbo generates a 512 x 512 image in approximately 20 to 26 seconds on an M4 Max.

---

## Generating Images in the Desktop App

1. Click **Images** in the app sidebar.
2. Select **Z-Image Turbo** or **Qwen-Image 2.1 Turbo** from the Model menu.
3. Enter your prompt in the prompt box (e.g. "A secluded wooden cabin by a misty mountain lake at twilight, 8k resolution").
4. Configure options:
   * **Size**: 512x512, 768x768, or 1024x1024 (multiples of 16).
   * **Steps**: 8 to 9 steps (Turbo models require few diffusion steps).
   * **Seed**: Set a specific integer for reproducible outputs, or leave blank for a random seed.
5. Click **Generate**. The generated PNG image displays in the workspace with one-click export and copy buttons.

### Mounting External Model Folders

If you store diffusion checkpoints on an external Thunderbolt/USB-C SSD:
1. Open **Settings > Models > Additional Model Folders**.
2. Click **Add Folder** and select your directory.
3. The app scans and makes external image models immediately available in the selector.

---

## Generating Images via CLI

The CLI offers two execution routes:

### 1. `turbospark image` (Recommended - Swift MLX engine)

Uses the same fast MLX engine as the desktop app:

```sh
turbospark image z-image-turbo-mlx-8bit \
  --prompt "A retrofuturistic cybernetic falcon soaring over neon skyscrapers" \
  --output falcon.png \
  --size 1024x1024 \
  --seed 42
```

### 2. `turbospark-image` (Native Metal engine)

Uses the low-level native Metal runtime (`crates/gpu`):

```sh
turbospark-image \
  --model ~/.turbospark/models/image/z-image-turbo \
  --prompt "mountain sunrise" \
  --output out.png
```

*Note: The native Metal image path is currently experimental and significantly slower than the Swift MLX pipeline. Use the Swift/app pipeline for production workflows.*

---

## Generating Images via Local HTTP API

Images can be generated via the local server using standard OpenAI API conventions:

```sh
curl http://127.0.0.1:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "z-image-turbo",
    "prompt": "An astronaut exploring ancient ruins on Mars, dramatic lighting",
    "size": "1024x1024",
    "n": 1,
    "seed": 77
  }' \
  | jq -r '.data[0].b64_json' | base64 -d > mars.png
```

See the [Images API Reference](../../api-reference/http/images.md) for full parameter specifications.
