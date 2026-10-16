---
description: Run supported language models locally on Apple Silicon with TurboSpark.
---

# TurboSpark

TurboSpark runs language models locally on Apple Silicon. Use the macOS app for chat, or integrate through the Rust library, Swift bindings, or local HTTP API.

Start with [installation](install.md), then follow the [desktop quickstart](quickstart.md). See the [desktop guide](desktop.md) for model selection and common workflows.

For integrations, open the [Rust and Swift API reference](https://app.gitbook.com/o/CAl9qS1d6FhFWiriQE3t/s/gJ02Ufhn7qpqBrO3Frio/). For scripts and other clients, see the [local API examples](call-the-local-api.md).

Model support and memory needs depend on the exact checkpoint and context. Check [supported models](models.md) and [memory and capacity](memory-and-capacity.md) before installing a large model.

TurboSpark is a local inference runtime and macOS app for Apple Silicon. It includes a Rust runtime, Metal-backed execution paths, command-line tools, and a local HTTP server.

Start with [installation](<install (1).md>), then follow the [quickstart](quickstart.md). Check [supported models](models-1.md) before downloading a checkpoint.

For planning, see [memory and capacity](<memory-and-capacity (1).md>). Use [model install troubleshooting](<model-install-troubleshooting (1).md>) for catalog, download, or access problems. To connect an app or script, see [local API examples](<call-the-local-api (1).md>).

Model compatibility and memory needs depend on the exact checkpoint, format, quantization, and context. The model catalog is the source of truth for promoted models and their evidence status.
