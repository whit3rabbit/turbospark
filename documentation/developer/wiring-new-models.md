---
description: Architectural checklist for bringing up and integrating a new model family in TurboSpark.
icon: plug
---

# Wiring a New Model Family

Adding support for a new model family requires cross-layer integration across the entire TurboSpark stack: model intake, architecture representation, runtime forward passes, Metal kernels, FFI, Swift bindings, and catalog tests.

## Architecture Checklist

### 1. Model Family Discriminator (`crates/model-io`)
* Add a new enum variant to `ModelFamily` in `crates/model-io/src/arch_config/family.rs`.
* Implement `as_str()` and `parse()`. Remember: on-disk format strings are frozen compatibility contracts; do not alter existing strings.
* Add the variant to `ModelFamily::ALL`.

### 2. Checkpoint Intake & Repack (`crates/repack`)
* Add architecture detection rules for Hugging Face `config.json` (`model_type`) and GGUF header keys.
* Map upstream tensor names to standardized TurboSpark tensor names (`manifest.json`).
* Separate dense layers into `resident.bin` and routed experts into `streamed.bin`.
* Ensure required tokenizer sidecars (`tokenizer.json`, `chat_template.jinja`) are verified.

### 3. Runtime Forward Pass (`crates/runtime`)
* Create a dedicated family state struct under `crates/runtime/src/families/<name>/`.
* Implement the per-token decode loop.
* Follow the reduction-order rule: routed expert slots must be dispatched in router ranking order.
* Implement KV-cache ring sizing derived from layer masks and attention parameters.

### 4. Metal Compute Kernels (`crates/gpu`)
* If the model introduces novel tensor operations (e.g. MLA absorbed attention, centered layer norms, or new rotary variants), write MSL compute kernels.
* Ensure all repeated Metal encoding loops use `gpu::autorelease_pool` to avoid leaking command buffers.
* Bake kernel pipeline constants into pipeline cache keys to prevent wrong-pipeline reuse.

### 5. C ABI and Swift Bindings (`crates/ffi` & `swift/`)
* Trace the family enum through `crates/ffi/src/model.rs` and Swift's `ModelFamily.swift`.
* Ensure error handling stays inside the ABI guard and never unwinds across `extern "C"`.

### 6. Catalog and Verification Gate (`crates/catalog` & `crates/bench`)
* Add the pinned checkpoint to `crates/catalog/src/models.json` with revision sha, download bytes, and evidence status.
* Create a frozen quality gate test asserting perplexity and digest stability.
* Create a memory oracle gate asserting physical footprint bounds on real hardware.
