# turbospark-image

Native Z-Image-Turbo image generation pipeline: prompt conditioning, Qwen3 text encoder, dense diffusion transformer, FlowMatch Euler scheduler, VAE decoder, packed component install format, macOS Metal backend, and pure-Rust CPU reference backend.

Detailed architectural specifications and benchmark numbers are documented in [`docs/ZIMAGE_TURBO.md`](../../docs/ZIMAGE_TURBO.md) and design records in [`docs/IMAGE_GENERATION.md`](../../docs/IMAGE_GENERATION.md).

Downstream workspace crates import this package via the `turbospark_image` library name:

```toml
[dependencies]
turbospark-image = { path = "../image", version = "0.1.0" }
```

## Purpose & Role

`turbospark-image` provides an end-to-end, zero-dependency native diffusion pipeline for image generation on Apple Silicon, with a portable CPU reference implementation for non-macOS targets.

This crate is NOT a text-model family and does not plug into `RealForwardRunner`. Image installs carry their own manifest schema, install receipts, and resource envelopes validated via `install.rs`.

## Platform Requirements

- **macOS (Metal Backend)**: Requires macOS with Apple Silicon GPU for hardware-accelerated inference (`metal.rs`, `metal_ops.rs`).
- **Portable Core (CPU Backend)**: The text encoder, diffusion transformer, scheduler, and VAE math are fully implemented in portable Rust (`reference.rs`, `transformer.rs`, `vae.rs`) and compile for non-macOS targets (such as Linux x86_64).

## Key Modules

- `install.rs`: Image model manifest schema, receipt verification, and fixed resource envelope constants.
- `builder.rs` / `builder_files.rs` / `builder_manifest.rs`: Offline install builder for assembling packed `.gturbo` image directories.
- `packed.rs`: Packed component store reader (`index.json` + `tensors.bin`) supporting affine INT4 group-64 quantization over FP32 stragglers.
- `conditioning.rs`: Prompt framing, tokenizer loading, and padded token ID preparation.
- `text_encoder.rs`: Qwen3 text encoder forward pass and sharded safetensors reader.
- `transformer.rs`: Dense diffusion transformer blocks and attention calculations.
- `pipeline.rs`: Timestep embedding generation, transformer driver loop, and diffusion constants.
- `scheduler.rs`: `FlowMatchEulerScheduler` implementing flow-matching Euler noise schedules.
- `vae.rs`: Autoencoder (VAE) decoder converting latents into RGB8 image pixels.
- `rope.rs`: Multimodal 3D rotary embedding (text + spatial image axes).
- `patchify.rs`: Patchify/unpatchify routines, spatial coordinate grids, and token sequence flattening.
- `runtime.rs`: `ImageBackend` trait, `ImageRequest`/`ImageResult`, generation cancellation token, and stage lifecycle management.
- `reference.rs`: `CpuReferenceBackend`, pure CPU image generation backend serving as ground truth and portable fallback.
- `metal.rs`: `MetalImageBackend` driving GPU execution through Metal command passes.
- `metal_ops.rs`: Image-specific Metal compute kernel wrappers and function constant bindings.
- `shaders/image.metal`: MSL source code for diffusion transformer and VAE kernels.

## Development & Test Commands

```sh
# Run portable unit tests and math parity against checked-in fixtures
cargo test -p turbospark-image

# Verify cross-target portability (verifies Metal modules are cleanly gated)
cargo check --target x86_64-unknown-linux-gnu -p turbospark-image

# Run full Metal pipeline parity tests against a packed install (macOS, ignored by default)
TURBOSPARK_IMAGE_INSTALL_DIR=~/models/z-image-turbo \
  cargo test -p turbospark-image --test metal_parity -- --ignored --nocapture
```

## Tests

- `tests/conditioning_parity.rs`: Validates text tokenization and conditioning tensor assembly.
- `tests/scheduler_parity.rs`: Validates FlowMatch Euler scheduler noise curves.
- `tests/text_encoder_parity.rs`: Validates Qwen3 text encoder embeddings against reference fixtures.
- `tests/transformer_math_parity.rs`: Validates diffusion transformer blocks and attention.
- `tests/pipeline_parity.rs`: Validates timestep injection and end-to-end diffusion steps.
- `tests/vae_parity.rs`: Validates latent decoding and RGB8 reconstruction.
- `tests/metal_parity.rs`: Opt-in end-to-end hardware parity test against a real packed installation.

## Crate Gotchas

1. **Platform Dependency Gating**: Metal and GPU dependencies live under `[target.'cfg(target_os = "macos")'.dependencies]`. All Metal calls must remain gated behind `#[cfg(target_os = "macos")]` to maintain cross-target buildability.
2. **Autorelease Pool Encapsulation**: A full image generation pass executes hundreds of Metal command passes across text encoding, denoising steps, and VAE decoding. Inner stages must manage autorelease pools to prevent unbounded memory growth.
3. **Distinct Install Architecture**: Image checkpoints must never be loaded into `turbospark-runtime` or `turbospark-model` text pipelines; they have distinct manifest structures and execution requirements.
