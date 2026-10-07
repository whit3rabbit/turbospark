---
description: "Integrate TurboSpark through its Rust crates or Swift bindings."
icon: code
---

# API reference

Use Rust to work with the catalog, tokenizer, runtime, and model files. Use Swift to open a model, stream a reply, manage installs, or start an in-process server.

- [HTTP REST API](http.md): endpoints, auth, curl examples, audio, and OpenAPI specs.
- [Rust library](rust.md): crate map, catalog example, and low-level generation API.
- [Swift bindings](swift.md): build setup, chat example, model management, embeddings, and server APIs.

The Rust runtime is a low-level inference API. It expects token IDs, a matching tokenizer, a generation config, and a producer. For an application-facing API, use the Swift session or the [HTTP REST API](http.md).
