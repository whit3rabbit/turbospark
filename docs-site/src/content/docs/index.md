---
title: TurboSpark: Low-Memory LLM Inference for Apple Silicon
description: A behavior-compatible Rust port of the Mference Swift inference engine for Apple Silicon Metal. Fine-grained MoE models run in ~1.6-2.2 GiB of peak RAM because routed experts stream from SSD through a bounded slot cache.
template: splash
# Sidebar group: Start Here (assigned in the sidebar config)
hero:
  title: TurboSpark: Low-Memory LLM Inference for Apple Silicon
  tagline: A behavior-compatible Rust port of the Mference Swift engine for Apple Silicon Metal. 26B-35B MoE models run in ~1.6-2.2 GiB of peak RAM because routed experts stream from SSD through a bounded slot cache.
  actions:
    - text: CLI reference
      link: /reference/cli/
      icon: right-arrow
      variant: primary
    - text: HTTP API
      link: /reference/http-api/
      icon: external
      variant: secondary
---

## What this is

`turbospark` is a behavior-compatible Rust port of [Mference](https://github.com/NeelM0906/Mference/tree/main), a Swift LLM inference engine for Apple Silicon Metal. Instead of holding full model parameters in unified RAM, it streams routed expert weights on demand from SSD storage into a lean working-memory slot cache, so fine-grained Mixture-of-Experts models run on Macs with 8, 16, 24, or 36 GB of unified memory without exhausting system memory: Gemma 4 26B-A4B (13 GB on disk) generates in ~2.1 GB of RAM, Qwen 3.6 35B-A3B (18 GB on disk) in ~1.6 GB. It ships three binaries (`turbospark-check` to generate, `turbospark-model` to find and install models, `turbospark-server` to serve the OpenAI and Anthropic APIs) plus a C ABI and SwiftPM package for embedding the engine in a native macOS app. Generation is macOS arm64 only; on other platforms the crates compile CPU stubs. The workspace MSRV is Rust 1.82 (`rust-version` in the root `Cargo.toml`).

## Who it is for

### CLI users

Pull a model and talk to it from the terminal. `turbospark-check` runs a model three ways (interactive `--chat`, raw `--prompt`, or `--messages-file` rendered through the checkpoint's own chat template), and `turbospark-model` finds, inspects, and installs models (`list`, `info`, `probe`, `recommend`, `pull`). Start with the [CLI reference](/reference/cli/).

### API integrators

Point an existing client at a local model. `turbospark-server` serves OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`, and `/v1/models` on loopback, in both non-streaming JSON and SSE. One runner per process, so requests are answered one at a time. See the [HTTP API reference](/reference/http-api/) and the [`turbospark-server` flags](/reference/cli-turbospark-server/).

### Native and Swift embedders

Run the engine in-process rather than over HTTP. `crates/ffi` exposes a C ABI (staticlib plus `turbospark.h`) with async token streaming, non-blocking cancellation, reasoning channel extraction, and model catalog management, wrapped by a SwiftPM package. See the [Swift session API](/reference/swift-session/) and the [C ABI contract](/reference/cpp-abi-contract/).

### Contributors

Change the engine itself. The workspace is 17 Rust crates spanning Metal kernels, expert streaming, model intake, tokenization, and the three front ends. Read [Verify a change](/guides/verify-a-change/) for the required gates (build, test, fmt, clippy, plus real-model smoke and per-family quality gates for decode-path changes), and [Architecture](/concepts/architecture/) for the crate map.

## What is measured, what is not

- **Parity with the Swift original.** Decode throughput lands within 1% of the Swift engine on the same install. Every family carries a memory oracle asserting a peak-footprint ceiling, every family but the dense `llama` one carries a frozen quality gate (teacher-forced perplexity plus output digests), and numerics are cross-checked against `mlx-lm`, `llama.cpp`, and MLX on identical bytes.
- **One machine.** Every published number was measured on one Apple M4 Max with 36 GB of unified memory on mains power. Throughput and footprint figures do not transfer to other chips.
- **Dense models get no streaming benefit.** Dense checkpoints (Mistral, TinyLlama, Qwen3.8-27B) run correctly but a dense token touches every weight once, so the RAM requirement is roughly the install's full size on disk.
- **Server authentication is optional and TLS is not provided.** Configure an API key with `--api-key` or `TURBOSPARK_API_KEY`. Loopback limits network exposure but does not isolate the server from other local processes. When using `--bind tailnet`, configure an API key rather than relying only on Tailnet ACLs.
- **Platform floor.** Metal GPU acceleration requires macOS on Apple Silicon; the MSRV is Rust 1.82.

## Where to go next

- CLI users: [CLI reference](/reference/cli/)
- API integrators: [HTTP API reference](/reference/http-api/), [`turbospark-server` flags](/reference/cli-turbospark-server/)
- Native and Swift embedders: [Swift session API](/reference/swift-session/), [C ABI contract](/reference/cpp-abi-contract/)
- Contributors: [Verify a change](/guides/verify-a-change/), [Architecture](/concepts/architecture/)
