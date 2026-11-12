---
description: "Map of the TurboSpark Rust workspace: what each of the 21 crates owns, how they layer together, and where each crate page lives."
icon: cubes
---

# Crate Reference

TurboSpark is a local inference engine for Apple Silicon Macs: Rust crates for
model intake, decode, and serving, Metal kernels for acceleration, and a native
Swift desktop app on top. The engine is organized as 21 crates under `crates/`.

This section documents each crate for developers who want to build on the
crates or contribute to the engine: what the crate owns, its public API
surface, and working code examples. End-user guides live in the other sections
of this space. To compile everything first, start with
[Building from Source](../developer/building-from-source.md).

{% hint style="info" %}
Package names follow the pattern `turbospark-<directory>`, so
`crates/model-io` is the `turbospark-model-io` package. Throughout these pages
a crate is named after its directory.
{% endhint %}

## The stack at a glance

Dependencies only point downward. Higher layers compose lower ones, and the
surface crates never reach past the pipeline layer into kernels directly.

```text
surfaces        cli    server    ffi    bench
                  \      |       /
pipeline             runtime    image        catalog
                       \      |      |      /
model engines      gpu   coreml   embeddings   audio (models)
                       \      |            /
intake         repack   model-io   tokenizer   streaming
                       \        |        /
primitives       core   compute   selection   invocation   vision-io   window-fit
```

Three crates need a note. `audio` contains both portable DSP primitives (WAV,
resampling, FFT, STFT, mel) and neural audio models, so it appears in both the
engine and primitive roles. `streaming` is about streaming routed-expert
weights from disk (a `pread` slot cache, mmap, eviction policy), not about
token streaming; incremental output lives in the runtime and server layers.
`coreml` is a standalone Core ML runner with no internal dependencies; the
Core ML model pipelines that use it live inside `runtime` under
`crates/runtime/src/coreml/`.

## Crate index

| Crate | Owns | Page |
| --- | --- | --- |
| `core` | Shared primitives: token IDs, errors, config basics | [core](core.md) |
| `compute` | Portable CPU compute: tensors and quantized math | [compute](compute.md) |
| `selection` | Sampling: logits to tokens | [selection](selection.md) |
| `streaming` | Routed-expert streaming from disk: `pread` cache, mmap layers, eviction | [streaming](streaming.md) |
| `invocation` | Command-line argument parsing into validated requests | [invocation](invocation.md) |
| `tokenizer` | Tokenizer loading and encoding | [tokenizer](tokenizer.md) |
| `model-io` | Checkpoint formats, manifests, mappings, resident buffers | [model-io](model-io.md) |
| `repack` | Model intake and `.gturbo` writing | [repack](repack.md) |
| `runtime` | Family dispatch, forward passes, state, sessions | [runtime](runtime.md) |
| `gpu` | Metal kernels and dispatch | [gpu](gpu.md) |
| `audio` | Audio DSP primitives and audio models by role | [audio](audio.md) |
| `embeddings` | Embedding models with pinned checkpoints | [embeddings](embeddings.md) |
| `image` | Image generation pipelines | [image](image.md) |
| `coreml` | Core ML runner (macOS; stub elsewhere) | [coreml](coreml.md) |
| `cli` | User-facing install and command surfaces | [cli](cli.md) |
| `server` | Local HTTP API service | [server](server.md) |
| `catalog` | Model catalog: known models, aliases, downloads | [catalog](catalog.md) |
| `ffi` | C ABI consumed by Swift | [ffi](ffi.md) |
| `bench` | Measurement harnesses | [bench](bench.md) |
| `vision-io` | Image decoding and preprocessing for vision models | [vision-io](vision-io.md) |
| `window-fit` | Context window fitting helpers | [window-fit](window-fit.md) |

## Reading order

If you are new to the codebase, read in this order:

1. [core](core.md) and [tokenizer](tokenizer.md) for the shared vocabulary:
   token IDs are signed 32-bit everywhere, and fixture IDs always come from a
   loaded tokenizer.
2. [model-io](model-io.md) then [repack](repack.md) to see how a checkpoint
   becomes a `.gturbo` resident buffer.
3. [compute](compute.md) and [gpu](gpu.md) for the two execution backends.
4. [runtime](runtime.md) for how a family gets dispatched and decoded, and
   [selection](selection.md) for sampling.
5. The surface you care about: [cli](cli.md), [server](server.md),
   [ffi](ffi.md), or [catalog](catalog.md).

## Rules that span crates

A few invariants are enforced across layer boundaries. The full list lives in
the repository `AGENTS.md`; the ones you are most likely to meet while reading
crate code:

* Decode flow is selected by `ArchConfig.family`, never by tensor names.
* Producers return logits; sampling owns normalization. Never softmax in both
  the producer and the sampler.
* Token IDs crossing crate boundaries are signed 32-bit
  (`turbospark_core::TokenId`).
* Family support is cross-layer: a new family threads through model-io,
  runtime, FFI, Swift, catalog, CLI, server, docs, and tests.
* Repeated Metal encoding loops use `gpu::autorelease_pool`, or autoreleased
  command objects accumulate for the life of the process.

For how changes get verified before they land, see
[Testing and Verification](../developer/testing-and-verification.md).

## Related pages

* [Building from Source](../developer/building-from-source.md)
* [Wiring a New Model Family](../developer/wiring-new-models.md)
* [Testing and Verification](../developer/testing-and-verification.md)
* [.gturbo Format Specification](../models/gturbo-format.md)
