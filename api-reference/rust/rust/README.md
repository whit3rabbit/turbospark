---
description: Find the TurboSpark Rust crate that fits your task, then read its guide.
icon: terminal
---

# Rust library

TurboSpark is a set of Rust crates. Pick the crate for the job you have, not the one that sounds most like inference. The catalog, model-file, tokenizer, and sampling crates work without a GPU. Running a real model needs macOS on Apple Silicon.

## Start with the job

| If you need to...                           | Start with                             |
| ------------------------------------------- | -------------------------------------- |
| Find, probe, or install a model             | [turbospark-catalog](catalog/)         |
| Read or validate an installed model         | [turbospark-model-io](model-io/)       |
| Build a `.gturbo` install from a checkpoint | [turbospark-repack](repack/)           |
| Render chat prompts and decode output       | [turbospark-tokenizer](tokenizer/)     |
| Pick the next token from logits             | [turbospark-selection](selection/)     |
| Run a generation loop                       | [turbospark-runtime](runtime/)         |
| Embed the HTTP routes in an Axum service    | [turbospark-server](server/)           |
| Keep chat history inside a token budget     | [turbospark-window-fit](window-fit.md) |
| Generate images                             | [turbospark-image](image/)             |
| Run speech, music, or audio DSP             | [turbospark-audio](audio/)             |
| Preprocess images for a vision tower        | [turbospark-vision-io](vision-io.md)   |
| Embed text, images, or audio for search     | [turbospark-embeddings](embeddings.md) |
| Read shared types and errors                | [turbospark-core](core/)               |

## All crates

| Crate                   | What it gives you                                                                                | Guide                  |
| ----------------------- | ------------------------------------------------------------------------------------------------ | ---------------------- |
| `turbospark-catalog`    | Model catalog, Hugging Face probe, hardware-aware recommendations, installs.                     | [Guide](catalog/)      |
| `turbospark-model-io`   | Install layout, `manifest.json`, architecture config, resident weight buffers.                   | [Guide](model-io/)     |
| `turbospark-repack`     | Checkpoint intake (safetensors, GGUF, Hugging Face) and `.gturbo` writing.                       | [Guide](repack/)       |
| `turbospark-tokenizer`  | Tokenizer wrapper, chat dialects and templates, streaming detokenizer, tool-call parsers.        | [Guide](tokenizer/)    |
| `turbospark-selection`  | Token selection under temperature, top-k, top-p, min-p, and penalties.                           | [Guide](selection/)    |
| `turbospark-runtime`    | Prefill and decode loops, the `LogitProducer` trait, the macOS forward runner.                   | [Guide](runtime/)      |
| `turbospark-server`     | Axum router for the local HTTP API, with options for auth and observers.                         | [Guide](server/)       |
| `turbospark-window-fit` | Trims conversation history to a token budget and keeps pinned turns.                             | [Guide](window-fit.md) |
| `turbospark-image`      | Z-Image-Turbo generation pipeline with Metal and CPU reference backends.                         | [Guide](image/)        |
| `turbospark-audio`      | DSP primitives plus speech, music, and VAD models organized by role.                             | [Guide](audio/)        |
| `turbospark-vision-io`  | Image bytes to patch rows and position tables for the `qwen3_5` vision tower.                    | [Guide](vision-io.md)  |
| `turbospark-embeddings` | EmbeddingGemma, SigLIP2 NaFlex, and CLAP embeddings, pooling, and similarity scoring on the CPU. | [Guide](embeddings.md) |
| `turbospark-core`       | Shared types, `TokenId`, errors, configuration, and chunking helpers.                            | [Guide](core/)         |
| `turbospark-invocation` | Pure CLI argument parsing with no I/O.                                                           | [Guide](invocation/)   |
| `turbospark-streaming`  | Expert weight streaming for MoE models. Advanced.                                                | [Guide](streaming/)    |
| `turbospark-compute`    | CPU reference kernels used to check the Metal kernels.                                           | [Guide](compute.md)    |
| `turbospark-gpu`        | Metal device context, pass encoders, and KV cache. Low level.                                    | [Guide](gpu/)          |
| `turbospark-cli`        | Command-line binaries: `turbospark`, `turbospark-check`, `turbospark-model`, and more.           | [Guide](cli/)          |
| `turbospark-ffi`        | C ABI for native hosts such as the Swift app. Opaque handles, streaming callbacks.               | [Guide](ffi/)          |
| `turbospark-bench`      | Throughput benchmark harness and startup probe for contributors. Not a public API.               | [Guide](bench/)        |

The Swift app consumes `turbospark-ffi` through the [Swift bindings](../../swift/swift.md). Read the [Swift page](../../swift/swift.md) if you build a Swift app; read the [FFI guide](ffi/) if you write the C side.

## Add a crate to your project

The workspace crates are not all on crates.io yet. Until a version you need is published, depend on a checkout by path:

```toml
[dependencies]
turbospark-catalog = { path = "../turbospark/crates/catalog" }
```

Each crate guide repeats the dependency lines it needs. In Rust code, the hyphens become underscores, so `turbospark-catalog` is `turbospark_catalog`.

## Where the runtime fits

`turbospark-runtime` is a low-level generation loop. It takes token IDs, a tokenizer for the same install, a generation config, and a `LogitProducer`. It does not offer a `generate(prompt) -> String` helper. If you want a finished application surface, use the [Swift session](../../swift/swift.md) or the [HTTP API](../../http/http.md). Start with the [runtime guide](runtime/) when you need the loop itself.
