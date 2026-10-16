---
description: Running vision-language models for image comprehension and OCR on Apple Silicon.
icon: eye
---

# Vision and Multimodal Models

TurboSpark supports multimodal vision-language models (VLMs), enabling local image understanding, visual question answering, and OCR without sending sensitive imagery to third-party cloud services.

## Supported Vision Checkpoints

1. **Qwen3-VL 4B Instruct (`qwen3vl-4b-vision`)**:
   * **Family**: `qwen3_vl`
   * **Vision Architecture**: Depth-24 SigLIP2 vision tower operating on visual patches, combined with 3 **DeepStack mergers**.
   * **Mechanism**: Intermediate vision features from block 8, 16, and 24 are merged and injected directly into trunk residual layers 0, 1, and 2 via raw additions.
   * **Intake**: Pull the combined multimodal install:
     ```sh
     turbospark-model pull --repo "mlx-community/Qwen3-VL-4B-Instruct-4bit@2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b" --alias qwen3vl-4b-vision
     ```

2. **Qwen3.8 27B Vision (`qwen38-27b-vision`)**:
   * **Family**: `qwen35`
   * **Vision Architecture**: 27B dense text trunk coupled with a BF16 vision tower sidecar (~1 GB).
   * **Intake**:
     ```sh
     turbospark-model pull qwen38-27b-vision
     ```

---

## Using Vision in the Desktop App

1. Select a vision-capable model (such as `qwen3vl-4b-vision` or `qwen38-27b-vision`) from the Model menu.
2. Click the **Paperclip / Image** button in the chat input bar, or drag and drop an image file (PNG, JPEG, WebP) directly into the message window.
3. Type your question (e.g. "Transcribe the table in this image to markdown" or "Describe the chart trends").
4. Press Enter. The app pre-processes image patches via `crates/vision-io`, encodes vision latents on the Metal GPU, and streams the conversational text response.

---

## Using Vision over the Local HTTP API

Pass base64-encoded image data or image URLs in standard OpenAI format:

```sh
# Encode image to base64
IMG_B64=$(base64 -i screenshot.png)

curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "qwen3vl-4b-vision",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What does this image show?"},
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/png;base64,'"$IMG_B64"'"
            }
          }
        ]
      }
    ],
    "max_tokens": 256
  }'
```
