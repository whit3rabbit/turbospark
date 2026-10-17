---
description: "Native image generation pipelines: install packing, the staged generation lifecycle, CPU reference math, and the macOS Metal backend."
icon: image
---

# image

`crates/image` (package `turbospark-image`) owns the native image generation
stack: the install format and packer, the diffusion scheduler and conditioning,
the FP32 CPU reference for the transformer and VAE, and the macOS Metal backend
that executes the packed install. The whole crate is `#![forbid(unsafe_code)]`;
the Metal code lives behind `cfg(target_os = "macos")` and everything else is
portable. The `turbospark-image` command users run is the binary in
`crates/cli/src/bin/image.rs`, not here; this crate is the library that binary
and the FFI image sessions call.

## What it does

| Area | Modules | Contents |
| --- | --- | --- |
| Install contract | `install` | The image manifest (GTURBO magic, schema 1), component metadata validation, and the per-family generation envelopes |
| Install packing | `builder` | Packs a Diffusers-style export into a checked install through a staging directory and one atomic rename |
| Packed components | `packed` | `tensors.bin` payload plus index with checked byte spans; affine INT4 group-64 and MLX affine U32 planes for eligible linear weights |
| Scheduling | `scheduler` | `FlowMatchEulerScheduler`, the pinned FlowMatchEuler discrete contract |
| Conditioning | `conditioning`, `text_encoder` | Qwen chat-template prompt framing, tokenization and padding, and the text encoder forward pass over sharded safetensors |
| DiT reference math | `pipeline`, `transformer`, `patchify`, `rope` | Streamed FP32 Z-Image transformer blocks, AdaLN modulation, 3-axis adjacent-pair RoPE, latent patchify |
| VAE | `vae` | Latent rescaling, conv2d, GroupNorm, upsampling, and the decode path to RGB8 and PNG |
| Lifecycle | `runtime` | Stage ownership (text encoder, transformer, VAE, PNG encode), progress, cancellation, and memory plan admission |
| Backends | `reference`, `metal` | `CpuReferenceBackend` (diagnostic) and `MetalImageBackend` (macOS, no CPU fallback) |
| Test fixtures | `fixtures` | `npy`/`npz` readers used by the parity gates |

## Where it sits

Depends on `core`, `compute`, `tokenizer`, and `model-io`, plus `ndarray`,
`png`, `rayon`, and `serde`. On macOS it additionally depends on `gpu` and the
`metal` crate; `repack` is a dev-dependency used by the packing tests only. It
does not depend on `runtime`. Its consumers are `cli` (the `turbospark-image`
binary) and `ffi` (the image session behind `ts_generate`).

## Public API tour

Key entry points, all reached from the crate root or their module:

| Item | Path | Meaning |
| --- | --- | --- |
| `ImageRequest` + `validate()` | `runtime` | One generation request; validation pins the envelope and dimensions |
| `generate`, `generate_with_memory_budget` | `runtime` | Run the staged lifecycle over an `ImageBackend`, returning `ImageResult { png, metadata }` |
| `ImageBackend` | `runtime` | The three-stage backend trait: `encode_conditioning`, `denoise`, `decode` |
| `CancellationToken`, `ImageProgress`, `ImageStage` | `runtime` | Cooperative cancellation and per-stage progress callbacks |
| `ImageMemoryPlan`, `ImageMemoryBudget`, `admit_image_budget` | `runtime` | Accounting-only resource admission before execution |
| `MetalImageBackend::open`, `open_with_residency` | `metal` (macOS) | Open a packed install resident or `ImageResidency::Streamed`; also `memory_plan` and `stream_metrics` |
| `CpuReferenceBackend::open` | `reference` | Diagnostic CPU execution of the packed component path |
| `build_image_install`, `ImageInstallSpec`, `ImageInstallReport` | `builder` | Pack a source export into a complete install |
| `ImageManifest::load`, `validate`, `verify_files` | `install` | Load and check an installed image model |
| `ImageFamily`, `image_family_for_model_id` | `install` | Route a model id to its envelope |
| `FlowMatchEulerScheduler` | `scheduler` | `set_timesteps`, `set_timesteps_from_sigmas`, `normalized_time` |
| `ZImageTransformer::open`, `forward_with_progress` | `pipeline` | The FP32 CPU DiT reference over a checkpoint directory |
| `frame_prompt`, `tokenize_prompt`, `encode_tokens`, `ShardedSafetensors::open` | `conditioning`, `text_encoder` | Prompt framing and text-encoder conditioning |
| `VaeDecoder::from_tensor_loader`, `decoded_to_rgb8`, `encode_rgb8_png` | `vae` | Latents to RGB8 bytes and PNG |
| `pack_component`, `PackedTensorStore`, `PackedQuantization` | `packed` | Component packing and reading |

### Families

`ImageFamily` routes on the model id prefix and each row carries its own
step count, guidance policy, prompt bound, and latent geometry:

| Family | Model id prefix | Steps | Guidance | Prompt tokens | Latent shape |
| --- | --- | --- | --- | --- | --- |
| `ZImage` ("z-image") | any id that matches no other row (default) | 9 | 0.0 | 512 | `[1, 16, 128, 128]` |
| `QwenImage21` ("qwen-image-2.1") | `mlx-community/Qwen-Image-2.1` | 40 | 1.0 | 1024 | `[1, 64, 64, 64]` |
| `QwenImage21Turbo` ("qwen-image-2.1-turbo") | `Sawfwair/Qwen-Image-2.1-Turbo` | 8 | 1.0 | 1024 | `[1, 64, 64, 64]` |
| `Krea2Turbo` ("krea-2-turbo") | `baa-ai/Krea-2-Turbo` | 8 | 1.0 | 1024 | `[1, 16, 128, 128]` |

The packer records each family's envelope and license in the manifest. The
staged `generate` lifecycle and the CLI currently enforce the Z-Image IG2
contract (batch 1, 9 steps, guidance 0.0), so the other families' envelopes
are install-side records; Krea-2-Turbo and Qwen-Image-2.1-Turbo installs
additionally carry explicit pending-native-runtime and pending-device-measurement
receipts. A checked install on disk is not by itself a runnable native target.

## Examples

### Pack and generate from the command line

From the usage text in `crates/cli/src/bin/image.rs`. The source is a
Diffusers-style export with `tokenizer/`, `scheduler/`, `text_encoder/`,
`transformer/`, and `vae/`; both commands refuse to overwrite an existing
path.

```sh
turbospark-image pack \
  --source ~/models/z-image-turbo-diffusers \
  --output ~/installs/z-image-turbo \
  --model-id Tongyi-MAI/Z-Image-Turbo \
  --model-revision f332072aa78be7aecdf3ee76d5c247082da564a6

turbospark-image generate \
  --model ~/installs/z-image-turbo \
  --prompt "A lighthouse in winter at dusk, warm windows" \
  --output lighthouse.png \
  --seed 42
```

`generate` runs native Metal on macOS by default; `--backend reference` selects
the diagnostic CPU path (1024x1024 only). Sizes run from 512 to 1024 pixels
per side in multiples of 16.

### Pack an install through the builder API

Adapted from the `pack` subcommand in `crates/cli/src/bin/image.rs` and
`crates/image/src/builder.rs`. Packing stages into a sibling directory and
publishes with one rename, so a failed run never leaves a partial install.

```rust
use turbospark_image::{build_image_install, ImageInstallSpec};

let spec = ImageInstallSpec {
    source_root: "/models/z-image-turbo-diffusers".into(),
    output_root: "/installs/z-image-turbo".into(),
    model_id: "Tongyi-MAI/Z-Image-Turbo".to_string(),
    model_revision: "f332072aa78be7aecdf3ee76d5c247082da564a6".to_string(),
};
let report = build_image_install(&spec)?;
println!(
    "packed {} files ({} bytes) into {}",
    report.file_count,
    report.total_bytes,
    report.output_root.display()
);
```

### Drive the generation lifecycle directly

Adapted from `crates/image/tests/metal_parity.rs` (the `request()` helper and
the PNG metadata gate). This needs a complete packed install on disk; build
one with `pack` first. `generate` validates the request, then runs text
encoder, transformer, VAE, and PNG encode in that order, reporting progress
per stage.

```rust
use std::collections::BTreeMap;
use std::path::Path;
use turbospark_image::{
    generate, CancellationToken, ImageBackend, ImageRequest, ImageStage,
    MetalImageBackend, IMAGE_BATCH, IMAGE_GUIDANCE, IMAGE_HEIGHT,
    IMAGE_QUANTIZATION, IMAGE_STEPS, IMAGE_WIDTH,
};

// macOS only. Open validates the install manifest before any GPU work.
let mut backend = MetalImageBackend::open(Path::new("/installs/z-image-turbo"))?;

let request = ImageRequest {
    model_id: "Tongyi-MAI/Z-Image-Turbo".to_string(),
    model_revision: "f332072aa78be7aecdf3ee76d5c247082da564a6".to_string(),
    component_revisions: BTreeMap::new(),
    prompt: "A lighthouse in winter at dusk, warm windows".to_string(),
    width: IMAGE_WIDTH,
    height: IMAGE_HEIGHT,
    batch: IMAGE_BATCH,
    scheduler_steps: IMAGE_STEPS,
    guidance_scale: IMAGE_GUIDANCE,
    seed: 42,
    quantization: IMAGE_QUANTIZATION.to_string(),
    // Non-empty provenance tag for the initial-noise implementation.
    noise_provenance: "zimage_metal_xorshift_box_muller_v1".to_string(),
};
request.validate()?;

let cancellation = CancellationToken::new();
let result = generate(&mut backend, &request, &cancellation, |event| {
    if event.stage == ImageStage::Transformer {
        println!("denoise {}/{}", event.completed, event.total);
    }
})?;

std::fs::write("lighthouse.png", &result.png)?;
// result.metadata carries prompt, seed, scheduler timesteps and sigmas,
// revisions, and the engine revision; it is also embedded in the PNG as an
// iTXt chunk keyed "turbospark".
```

The same `generate` call accepts `CpuReferenceBackend::open(root)` wherever
`MetalImageBackend` does, because both implement `ImageBackend`. Dropping
`request.validate()` from your own path is fine; `generate` validates again.

## Gotchas

{% hint style="warning" %}
`MetalImageBackend` is macOS only and has no CPU fallback inside the type.
`CpuReferenceBackend` exists to prove install layout, staging, cancellation,
and metadata; it does not claim the Metal performance or memory envelope, and
the CLI documents it as diagnostic-only. Anything built off macOS sees neither
backend: the whole `metal` module is behind `cfg(target_os = "macos")`.
{% endhint %}

{% hint style="warning" %}
The IG2 request envelope is frozen, not configurable. `ImageRequest::validate`
rejects any batch, step count, or guidance scale outside the family envelope,
any dimension outside 512 to 1024 per side, any side not a multiple of 16, and
any pixel count over 1024x1024. The quantization label must match the manifest
(the legacy INT4 label, `unquantized`, or an MLX affine label carrying its
observed bit widths), and an empty `noise_provenance` is refused.
{% endhint %}

## Related pages

- [Image generation](../multimodal/image-generation.md)
- [gpu](gpu.md): the Metal layer the image backend dispatches through
- [model-io](model-io.md): safetensors reading under the packed components
- [cli](cli.md): where the `turbospark-image` binary lives
- [Testing and verification](../developer/testing-and-verification.md)
