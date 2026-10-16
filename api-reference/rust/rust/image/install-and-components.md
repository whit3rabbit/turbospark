---
description: "Load, verify, and build the packed Z-Image-Turbo install that turbospark-image opens."
icon: box
---

# Install format and components

An image install is a directory with a `manifest.json` file and five component directories under `components/`. The weight components use the packed format described below.

Call `ImageManifest::load` and `verify_files` when your application needs to check an install before it opens a backend. Call `build_image_install` only when you pack an install from a source tree yourself. Most applications should open an install that the TurboSpark CLI already packed and verified.

```text
<install root>/
  manifest.json              # envelope, source metadata, and per-file hashes
  verified-install.json      # receipt written by the builder
  components/
    tokenizer/
    text_encoder/            # packed Qwen3 text encoder
    transformer/             # packed diffusion transformer
    scheduler/
    vae_decoder/
```

Each packed component directory holds an `index.json` and a `tensors.bin` file. The index describes each tensor, and the data file holds the packed bytes.

## Load and verify

`ImageManifest::load` reads `manifest.json` from the install root. It refuses a manifest larger than 4 MiB.

`validate` checks the manifest against the frozen image envelope without reading the tensor payloads. `verify_files` runs `validate`, then checks the size and SHA-256 of every file the manifest declares. Run the steps in that order. Each one fails before the next reads more data.

```rust
use std::path::Path;

use turbospark_image::ImageManifest;

fn check_install(root: &Path) -> Result<(), String> {
    let manifest = ImageManifest::load(root)?;
    manifest.validate()?;
    let validation = manifest.verify_files(root)?;
    println!("{} files verified", validation.file_count);
    Ok(())
}

fn main() -> Result<(), String> {
    check_install(Path::new("/path/to/installed/z-image-turbo"))
}
```

`verify_files` reads every declared file. A full install takes a while to hash, so run it once at startup or at install time rather than on every request. `MetalImageBackend::open` and `CpuReferenceBackend::open` run these checks themselves. Call them yourself when you want to report a bad install before you load a backend.

The manifest stores `components` with an owner and metadata for each component. The `transformer` entry carries `quantization_scheme`, and [`image_quantization_label`](https://docs.rs/turbospark-image/latest/turbospark_image/fn.image_quantization_label.html) reads that field. Pass its result to `ImageRequest::quantization`. The `source` field carries the model ID and revision. The example above copies them into each request.

{% hint style="warning" %}
Do not edit an install by hand. `verify_files` compares each file's size and hash with the manifest, so any change to a payload makes the install fail verification. Pack a new install instead.
{% endhint %}

## Build an install from a source tree

`build_image_install` packs a Diffusers-style source tree into the install layout. The source needs a tokenizer directory, `scheduler/scheduler_config.json`, and safetensors weights for the text encoder, transformer, and VAE. The builder also accepts the mflux index name, and it accepts the `vae_decoder/` spelling that local fixtures use. It does not download anything. The source tree stays as it was.

```rust
use std::path::PathBuf;

use turbospark_image::{build_image_install_with_progress, ImageInstallSpec};

fn main() -> Result<(), String> {
    let spec = ImageInstallSpec {
        source_root: PathBuf::from("/path/to/Z-Image-Turbo"),
        output_root: PathBuf::from("/path/to/store/z-image-turbo"),
        model_id: "Tongyi-MAI/Z-Image-Turbo".to_string(),
        model_revision: "f332072aa78be7aecdf3ee76d5c247082da564a6".to_string(),
    };
    let report = build_image_install_with_progress(&spec, |stage| eprintln!("{stage}"))?;
    println!("{} files, {} bytes", report.file_count, report.total_bytes);
    Ok(())
}
```

The example compiles against the crate on macOS. It was not run, because no source checkpoint was present on the machine used to write it.

Read these rules before you start a long build. A refused spec fails fast, but a failed pack wastes the time already spent.

- `output_root` must not exist, and its parent directory must exist. The builder refuses to overwrite an install.
- `model_id` and `model_revision` must not be blank. They become the manifest's source record.
- The builder packs into a hidden staging directory beside `output_root`, writes the receipt, and then renames the directory into place. A failed build removes the staging directory, so `output_root` never holds a partial install.

`build_image_install` is the same call without the progress closure. The progress closure receives coarse stage names as `&str`. `ImageInstallReport` returns the output path, the file count, the total bytes, and a `PackedTensorReport` for each component. Use [`pack_component`](https://docs.rs/turbospark-image/latest/turbospark_image/fn.pack_component.html) only when you need to pack one safetensors directory and skip the rest of the install.

## Model families

The manifest records a model family. [`image_family_for_model_id`](https://docs.rs/turbospark-image/latest/turbospark_image/install/fn.image_family_for_model_id.html) maps a model ID to that family, and `ImageFamily::license` returns the source license for that family. Read the license first. Then decide whether you can distribute the install. The `generate` path in this crate only runs the Z-Image-Turbo envelope, as described in the [overview](README.md#limits-and-errors).

## Next

- [turbospark-image overview](README.md) shows the pipeline and a complete generation example.
- [Z-Image-Turbo artifact and runtime boundary](https://github.com/whit3rabbit/turbospark/blob/main/docs/ZIMAGE_TURBO.md#artifact-and-runtime-boundary) explains the packed format and the MLX source it comes from.
- [Native image generation](https://github.com/whit3rabbit/turbospark/blob/main/docs/IMAGE_GENERATION.md) covers the CLI commands that pack and pull installs.
