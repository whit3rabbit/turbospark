---
description: Generate PNG images with the native Z-Image-Turbo pipeline from Rust.
icon: box
---

# Image generation

`turbospark-image` turns a prompt into a PNG from an installed Z-Image-Turbo package. Use it when your Rust program needs local image generation and can open an install that TurboSpark has packed and verified. The Metal backend is the production path on macOS. The CPU reference backend runs on other hosts, but it is a diagnostic path that renders only 1024 x 1024 images.

Add the package from crates.io once the version you need is published, or use a path or git dependency from a checkout. The package is `turbospark-image`, and the import path is `turbospark_image`. Workspace crates in this repository import it under the alias `image`. That alias is internal, so use `turbospark_image` in your own code. The example below also reads manifest fields as `serde_json::Value`, so your `Cargo.toml` needs `serde_json` too.

## Pipeline

One call to `generate` runs four stages in order and returns a complete PNG. The backend supplies the device work for the text encoder, the transformer, and the VAE decoder. The crate owns the order, the scheduler, the cancellation rules, and the PNG encoding.

1. **Prompt conditioning.** `frame_prompt` wraps the prompt in the Qwen chat template with thinking enabled. `tokenize_prompt` truncates or pads the result to 512 tokens and returns token IDs with an attention mask.
2. **Text encoding.** The Qwen3 text encoder turns the active tokens into conditioning vectors. Each token has 2560 values (`Z_IMAGE_CAP_DIM`).
3. **Denoising.** `FlowMatchEulerScheduler` builds 9 timesteps with a static shift of 3.0. The dense diffusion transformer runs once per step. Z-Image-Turbo uses no classifier-free guidance, so `guidance_scale` must be `0.0`. The transformer has width 3840, 30 attention heads, two noise refiner blocks, two context refiner blocks, and 30 main blocks.
4. **VAE decode and PNG encode.** The 16-channel latent decodes to RGB, `decoded_to_rgb8` converts it to 8-bit samples, and the PNG encoder writes the image. The PNG carries the serialized `ImageMetadata` in an iTXt chunk with the keyword `turbospark`.

The library returns bytes only after PNG encoding succeeds, so a cancelled or failed run never yields a partial image. The TurboSpark CLI follows the same rule for its output file. For the model design and measured behavior, read [ZIMAGE\_TURBO.md](https://github.com/whit3rabbit/turbospark/blob/main/docs/ZIMAGE_TURBO.md) and [IMAGE\_GENERATION.md](https://github.com/whit3rabbit/turbospark/blob/main/docs/IMAGE_GENERATION.md).

## Choose a backend

| Backend                                                                                                           | Platform                  | Sizes                                  | Use it for                    |
| ----------------------------------------------------------------------------------------------------------------- | ------------------------- | -------------------------------------- | ----------------------------- |
| [`MetalImageBackend`](https://docs.rs/turbospark-image/latest/turbospark_image/struct.MetalImageBackend.html)     | macOS with a Metal device | Any size the request validator accepts | Production generation         |
| [`CpuReferenceBackend`](https://docs.rs/turbospark-image/latest/turbospark_image/struct.CpuReferenceBackend.html) | Any host                  | 1024 x 1024 only                       | Diagnostics and parity checks |

`MetalImageBackend::open` fails when no Metal device is present. It has no CPU fallback, so choose `CpuReferenceBackend` explicitly if you want the reference path. The Metal module compiles only for `target_os = "macos"`. Gate any code that names `MetalImageBackend` with the same `cfg`.

Both backends implement the [`ImageBackend`](https://docs.rs/turbospark-image/latest/turbospark_image/trait.ImageBackend.html) trait. You can pass either one to `generate` as `&mut dyn ImageBackend`.

## Generate an image

The example opens an install, builds a request that matches the TurboSpark CLI, and writes the PNG. Replace the path with a real install root. The root must contain `manifest.json` and a `components/` directory. See [install format and components](install-and-components.md).

```rust
use std::collections::BTreeMap;
use std::path::Path;

use turbospark_image::{
    generate, image_quantization_label, CancellationToken, ImageBackend, ImageManifest,
    ImageRequest,
};

#[cfg(target_os = "macos")]
use turbospark_image::MetalImageBackend;

#[cfg(not(target_os = "macos"))]
use turbospark_image::CpuReferenceBackend;

fn main() -> Result<(), String> {
    let root = Path::new("/path/to/installed/z-image-turbo");
    let manifest = ImageManifest::load(root)?;
    manifest.verify_files(root)?;

    let source_field = |key: &str| {
        manifest
            .source
            .get(key)
            .and_then(serde_json::Value::as_str)
            .unwrap_or("installed")
            .to_string()
    };
    let component_revisions = manifest
        .components
        .iter()
        .filter_map(|(name, component)| {
            component
                .metadata
                .get("canonical_revision")
                .and_then(serde_json::Value::as_str)
                .map(|revision| (name.clone(), revision.to_string()))
        })
        .collect::<BTreeMap<_, _>>();

    let (mut backend, noise_provenance): (Box<dyn ImageBackend>, &str) = {
        #[cfg(target_os = "macos")]
        {
            (
                Box::new(MetalImageBackend::open(root)?),
                "zimage_metal_xorshift_box_muller_v1",
            )
        }
        #[cfg(not(target_os = "macos"))]
        {
            (
                Box::new(CpuReferenceBackend::open(root)?),
                "zimage_cpu_xorshift_box_muller_v1",
            )
        }
    };

    let request = ImageRequest {
        model_id: source_field("model_id"),
        model_revision: source_field("model_revision"),
        component_revisions,
        prompt: "A lighthouse in winter".to_string(),
        width: 1024,
        height: 1024,
        batch: 1,
        scheduler_steps: 9,
        guidance_scale: 0.0,
        seed: 42,
        quantization: image_quantization_label(&manifest)?,
        noise_provenance: noise_provenance.to_string(),
    };
    request.validate()?;

    let cancellation = CancellationToken::new();
    let result = generate(backend.as_mut(), &request, &cancellation, |progress| {
        eprintln!("{:?} {}/{}", progress.stage, progress.completed, progress.total);
    })?;

    std::fs::write("lighthouse.png", &result.png).map_err(|e| e.to_string())?;
    println!("wrote lighthouse.png ({} forwards)", result.metadata.transformer_forwards);
    Ok(())
}
```

`generate` validates the request before it touches the backend. It then reports progress for each stage through `ImageProgress`, which has a stage, a completed count, and a total. The `metadata` field of `ImageResult` records the request values, the scheduler timesteps, the sigmas, and the engine revision. Store it directly, and you do not need to parse the PNG to recover it.

The example compiles on macOS. It has not run against a model, because no image install was present on the machine used to write it. Point it at a real install before you rely on it.

## Cancel a run

`CancellationToken` is cheap to clone. Each clone shares one flag, so you can keep a copy on another thread and call `cancel()` to stop a run. `generate` checks the flag between stages. When a run is cancelled, it also calls `wait_for_idle` on the backend so GPU work is fenced before the function returns.

A cancelled run returns `Err` with the text in [`IMAGE_CANCELLED`](https://docs.rs/turbospark-image/latest/turbospark_image/constant.IMAGE_CANCELLED.html). Compare against that constant rather than a copied string.

## Limits and errors

Most public fallible functions return `Result<_, String>`. A few return typed errors instead: `frame_prompt` and `load_tokenizer` return the tokenizer's `TokenizerError`, `admit_image_budget` returns `ImageBudgetError`, and `SequentialImageSlots::acquire` and `release` return `&'static str`. `generate_with_memory_budget` converts its budget refusal to `String`, so call `admit_image_budget` directly if you need the variant.

The request envelope is fixed. [`ImageRequest::validate`](https://docs.rs/turbospark-image/latest/turbospark_image/struct.ImageRequest.html#method.validate) enforces these rules:

* `batch` must be 1, `scheduler_steps` must be 9, and `guidance_scale` must be exactly `0.0`.
* Width and height must each fall between 512 and 1024 pixels, be multiples of 16, and have at most 1,048,576 total pixels. `is_supported_image_dimensions` applies the same rule.
* `prompt` cannot be blank, and `model_id` and `model_revision` cannot be empty.
* `quantization` must be a label that `image_quantization_label` can produce. Read it from the manifest instead of typing it.
* `noise_provenance` cannot be empty. The PNG metadata records whatever string you pass.

Because the envelope is the Z-Image-Turbo envelope, a manifest from another family cannot run through `generate` with its own step count or guidance. Treat `generate` as the Z-Image-Turbo entry point.

The backends open an install that already exists on disk. They do not download weights. Get the install from the TurboSpark CLI or from your own packing step before you call any backend.

## Next

* [Install format and components](install-and-components.md) covers `ImageManifest`, verification, and building an install from a source tree.
* [Rust library overview](../) lists the other TurboSpark crates.
* [Image generation notes](https://github.com/whit3rabbit/turbospark/blob/main/docs/IMAGE_GENERATION.md) describes the CLI, install sources, and verification gates.
