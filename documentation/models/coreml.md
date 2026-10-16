---
description: Running models on Apple Silicon Neural Engine and GPU using Apple Core ML.
icon: microchip-ai
---

# Core ML Models

TurboSpark includes native Apple Core ML runtime bindings through `crates/coreml`. On macOS, Core ML provides dedicated acceleration by offloading neural tensor workloads directly to the Apple Neural Engine (ANE) and GPU without competing for CPU resources.

## Core ML Integration Architecture

* **Rust to Core ML Bridge**: `crates/coreml` provides an Objective-C / Swift runtime boundary to load, compile, and execute `.mlmodelc` / `.mlpackage` artifacts.
* **Non-macOS Stubs**: For portable cross-target compilation (Linux, Windows), `crates/coreml` exports no-op stubs to preserve workspace integrity while restricting actual execution to macOS.
* **Zero Host Contention**: Core ML models targeting the Neural Engine run asynchronously, allowing background embeddings or re-ranking while the primary GPU runs LLM chat generation.

---

## Supported Core ML Models

### EmbeddingGemma 2 (`coreml_embedding_model`)

TurboSpark integrates `EmbeddingGemma 2` compiled for Apple Core ML as an embedding provider for `turbospark-server` and the desktop app:

* **Engine**: Apple Neural Engine (ANE) + GPU.
* **Input**: Batched token sequences up to 2,048 tokens.
* **Output**: Dense L2-normalized vector embeddings.
* **Memory**: Resident weights are pinned in Neural Engine memory structures, consuming minimal unified system RAM.

To launch the server with Core ML embeddings:

```sh
turbospark-server \
  --model gemma4 \
  --embedding-model embeddinggemma-2-coreml
```

---

## Compiling Models for Core ML

To prepare custom models for Core ML execution in TurboSpark:

1. Use Apple's `coremltools` Python package to convert PyTorch or Safetensors weights into a `.mlpackage`.
2. Compile the package into a `.mlmodelc` directory using `xcrun coremlcompiler`:

```sh
xcrun coremlcompiler compile Model.mlpackage ./output/
```

3. Place the resulting `Model.mlmodelc` into `~/.turbospark/models/embeddings/`.
