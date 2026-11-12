---
description: Running models on Apple Silicon Neural Engine and GPU using Apple Core ML.
icon: microchip-ai
---

# Core ML Models

TurboSpark includes native Apple Core ML runtime bindings through `crates/coreml`. On macOS, Core ML provides dedicated acceleration by offloading neural tensor workloads directly to the Apple Neural Engine (ANE) and GPU without competing for CPU resources. Core ML pipelines require macOS 15 or newer unless a page says otherwise.

## Core ML Integration Architecture

* **Rust to Core ML Bridge**: `crates/coreml` (`turbospark-coreml`) provides an Objective-C runtime boundary (via `objc2-core-ml`) to load, compile, and execute `.mlmodelc` / `.mlpackage` artifacts, with a compile cache, stateful models, and multifunction models.
* **Non-macOS Stubs**: For portable cross-target compilation (Linux, Windows), `crates/coreml` exports no-op stubs to preserve workspace integrity while restricting actual execution to macOS.
* **Compute Plan**: `turbospark_coreml::compute_plan` (and the `coreml_plan` example) reads the device Core ML plan for each op (`MLComputePlan`, macOS 14.4+), so you can check whether an export really reaches the Neural Engine before trusting a placement claim.

---

## Supported Core ML Models

All pinned rows install from the CLI and resolve by alias anywhere a model path is accepted, but only to an install of the current pin: the catalog lists every file by size and SHA-256, the installer takes the revision the catalog showed and refuses any other, and a new pin's install replaces an older one. Run `turbospark-model list-audio` for the audio rows; the pinned embedding and OCR installs live in `models/embeddings/` and `models/ocr/`.

| Pipeline | Catalog alias | Surface |
| --- | --- | --- |
| **Parakeet Redux speech-to-text** | `parakeet-redux-coreml` | Native audio sessions, `--stt-model`, and STT sessions over the C ABI. Token-identical to the CPU Redux decoder on the gate clips. |
| **Parakeet Ultra streaming** | `parakeet-ultra-coreml` | The int8 encoder on the Neural Engine with the log-mel and RNNT steps on the CPU. Batch transcription through `--stt-model`. |
| **Parakeet Unified (live)** | `parakeet-unified-en-coreml` | The `ts_audio_stream_open`/`append`/`finish`/`cancel`/`close` C ABI with Swift `AudioStream` bindings; live text while recording in the app's Record page. Streaming only. |
| **Speaker diarization** | `speaker-diarization-coreml` | pyannote community-1 export: who-spoke-when turns. Transcripts carry per-segment speakers, and the app renders localized speaker labels and stitches segment boundaries across transcription windows. |
| **EmbeddingGemma 2** | `embeddinggemma-2-coreml` | Text, image and audio embeddings with task prompts and Matryoshka sizes: `/v1/embeddings`, `ts_embedding_encode_json`, `ts_embedding_encode_request_json`, and Swift `TurboSparkEmbedding.encode(request:)`. |
| **Unlimited OCR** | `unlimited-ocr-coreml` (nine Core ML files, 4.09 GB) | `turbospark ocr` on page images, the `ts_ocr_*` C API, and Swift `TurboSparkOcr`. |
| **ANE+Steel prefill split** (dense Qwen) | programs you generate | Splits dense projections between the Neural Engine and Steel at prefill. Needs `crates/coreml/tools/gen_ane_prefill_programs.py`; see [Hardware Acceleration](hardware-acceleration.md). |

The app-only FluidAudio rows (`parakeet-ja-coreml`, `paraformer-zh-coreml`) also install through the same catalog machinery and run inside the app through the pinned FluidAudio package. See [Speech to Text](../audio/speech-to-text.md).

### Launching the server with Core ML pipelines

```sh
# Embeddings on the Neural Engine (first load compiles; can take minutes)
turbospark-server \
  --model gemma4 \
  --embedding-model embeddinggemma-2-coreml

# Speech-to-text on Core ML (macOS 15+)
turbospark-server \
  --model gemma4 \
  --stt-model parakeet-redux-coreml
```

---

## Compiling Models for Core ML

To prepare custom models for Core ML execution in TurboSpark:

1. Use Apple's `coremltools` Python package to convert PyTorch or Safetensors weights into a `.mlpackage`.
2. Compile the package into a `.mlmodelc` directory using `xcrun coremlcompiler`:

```sh
xcrun coremlcompiler compile Model.mlpackage ./output/
```

3. For the ANE+Steel prefill split, write the programs with the repository's tool instead:

```sh
python3 crates/coreml/tools/gen_ane_prefill_programs.py
```

The runner verifies programs against the install and turns the split off with one warning if they do not match.
