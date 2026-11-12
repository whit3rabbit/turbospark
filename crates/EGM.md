# crates

## Overview
The Rust workspace members of TurboSpark. Portable primitives sit at the bottom (`core`, `compute`, `selection`, `tokenizer`, `window-fit`, `invocation`, `streaming`, `model-io`, `vision-io`), the Metal and Core ML backends and the family runtime sit above them (`gpu`, `coreml`, `runtime`), and the user-facing surfaces (`cli`, `server`, `catalog`, `ffi`) sit on top. Existing per-crate `AGENTS.md` and `README.md` files are human-maintained and are not modified by these contracts.

## Subdirectories & Boundaries
- audio: [./audio/EGM.md](./audio/EGM.md) - portable audio crate: DSP, neural ops, STT, TTS, music, VAD, codecs.
- bench: [./bench/EGM.md](./bench/EGM.md) - frozen-protocol throughput benchmark, startup probe, tree replay, logit compare, meeting scoring.
- catalog: [./catalog/EGM.md](./catalog/EGM.md) - curated catalog, Hugging Face client, install driver, probe, recommend, external libraries.
- cli: [./cli/EGM.md](./cli/EGM.md) - user-facing binaries: front door, model, steer, image, music, clone, speak, ocr.
- compute: [./compute/EGM.md](./compute/EGM.md) - FP32 CPU reference kernels used as ground truth for the Metal backend.
- core: [./core/EGM.md](./core/EGM.md) - shared primitives: token types, runtime knob sets, prefill planning, steering, humanizer.
- coreml: [./coreml/EGM.md](./coreml/EGM.md) - Apple Core ML runner with a non-macOS stub, plus ANE program tools.
- embeddings: [./embeddings/EGM.md](./embeddings/EGM.md) - portable CPU embedding models (Gemma, BERT, GTE, Nomic, SigLIP2, CLAP).
- ffi: [./ffi/EGM.md](./ffi/EGM.md) - C ABI consumed by Swift: header, sessions, in-process server, model management.
- gpu: [./gpu/EGM.md](./gpu/EGM.md) - Metal kernels, dispatch, resident weights and KV cache.
- image: [./image/EGM.md](./image/EGM.md) - native text-to-image pipeline with CPU reference and Metal backends.
- invocation: [./invocation/EGM.md](./invocation/EGM.md) - command-line grammar and exit-status mapping.
- lab: [./lab/EGM.md](./lab/EGM.md) - `tslab` evaluator for kernel and method experiments.
- model-io: [./model-io/EGM.md](./model-io/EGM.md) - checkpoint formats, manifests, architecture configs, resident buffers.
- repack: [./repack/EGM.md](./repack/EGM.md) - model intake: HF safetensors and GGUF to `.gturbo` installs.
- runtime: [./runtime/EGM.md](./runtime/EGM.md) - family dispatch, Metal forward passes, sessions, speculative decode.
- selection: [./selection/EGM.md](./selection/EGM.md) - penalties, shaping, truncation and seeded token draws.
- server: [./server/EGM.md](./server/EGM.md) - OpenAI-style HTTP server and audio routes.
- streaming: [./streaming/EGM.md](./streaming/EGM.md) - MoE expert streaming: layout, mmap and pread backends, cache.
- tokenizer: [./tokenizer/EGM.md](./tokenizer/EGM.md) - tokenizer wrapper, chat templates, detokenizer, tool-call parsers.
- vision-io: [./vision-io/EGM.md](./vision-io/EGM.md) - portable vision preprocessing.
- window-fit: [./window-fit/EGM.md](./window-fit/EGM.md) - deterministic conversation-window fitting.

## Global Invariants
- Decode flow is chosen by `ArchConfig.family`, never by tensor names [crates/model-io/src/arch_config, crates/runtime/src/families].
- Producers return logits and sampling owns normalization [crates/selection/src].
- Token ids crossing crate boundaries are signed 32-bit (`TokenId`) [crates/core/src].
- FFI entry points run inside the ABI guard and never unwind across `extern "C"` [crates/ffi/src].
- Repeated Metal encoding loops use `autorelease_pool` [crates/gpu/src/context].

## Global Side Effects
- Metal and Core ML device use (gpu, coreml, runtime, image, audio).
- Hugging Face network traffic and disk writes under the model store (catalog, repack, cli).
- Local TCP listener (server, ffi in-process server).

## System Verification
- Run: `cargo test --workspace`
