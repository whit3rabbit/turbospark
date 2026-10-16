---
description: "Open and verify TurboSpark .gturbo model installs in Rust: manifest and architecture checks, the resident tensor index, mapped weights, SHA-256, and install receipts."
icon: box
---
# turbospark-model-io

`turbospark-model-io` checks that a TurboSpark model directory is complete and matches the architecture you expect, before any weights are used. Reach for it when your code loads `.gturbo` installs and needs to refuse a bad one with a named reason. It does not run inference. It parses `manifest.json`, validates that manifest against an `ArchConfig`, reads the tensor index in `model_weights.bin`, maps the resident weight bytes, hashes files, and checks the install receipt.

Inference and install tooling in this workspace build on this crate. You only need it directly if you want to inspect or verify an install yourself.

## Install

Add the package from crates.io once the version you need is published. Until then, point Cargo at a checkout with a path dependency, or use a git dependency:

```toml
[dependencies]
turbospark-model-io = { path = "../turbospark/crates/model-io" }
```

Import it as `turbospark_model_io`. Other crates in the TurboSpark workspace import this code through the short alias `model_io`. That alias exists only inside the workspace, so your code should use the package name and the Rust path shown above.

The crate has no GPU dependency. It depends on `memmap2` and `libc` for the mapped weight buffer. CI runs `cargo check -p turbospark-model-io` on Linux, so the crate builds there without macOS. Confirm the behavior you need on your own target before you ship.

## What the crate covers

| Item | Use it to |
| --- | --- |
| `peek_family`, `load_manifest` | Read `manifest.json` and check it against an expected architecture |
| `ArchConfig`, `ModelFamily`, `known_architecture` | Describe the shape a family expects |
| `load_resident_index`, `ResidentIndex` | Read the tensor table at the head of `model_weights.bin` |
| `ResidentBuffer` | Map the resident tensor bytes read-only |
| `hash_file`, `hash_data`, `verify_file` | Compute and check SHA-256 digests |
| `load_install_receipt`, `validate_install_receipt` | Check the `verified-install.json` receipt against the manifest |
| `load_packed_experts_layout` | Read `packed_experts/layout.json`, which gives each routed expert's byte offset in its layer file |
| `ModelError` | Report every failure above |

The crate also exports types for other model kinds, such as Whisper configs, speech receipts, vision sidecars, and n-gram tables. This reference does not cover them.

## Quick start

Open an install by reading its family, resolving the baseline architecture for that family, and loading the manifest against it:

```rust
use std::path::Path;

use turbospark_model_io::{
    known_architecture, load_manifest, peek_family, ModelError, DEFAULT_MAX_BYTES,
};

fn open_install(dir: &Path) -> Result<(), ModelError> {
    let family = peek_family(dir, DEFAULT_MAX_BYTES)?;
    let expected = known_architecture(family);
    let manifest = load_manifest(dir, &expected, DEFAULT_MAX_BYTES)?;
    println!(
        "{} ({:?}, {} layers)",
        manifest.model_id, family, manifest.num_layers
    );
    Ok(())
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: quickstart <model.gturbo directory>");
    if let Err(error) = open_install(Path::new(&dir)) {
        eprintln!("refused: {error}");
        std::process::exit(1);
    }
}
```

Against a directory with no `manifest.json`, the example prints a `refused:` line that names the missing file, then exits with status 1. A successful load needs a real install. The crate's [manifest tests](https://github.com/whit3rabbit/turbospark/blob/main/crates/model-io/tests/manifest.rs) show the accepted shapes.

## Errors

Every fallible function returns `Result<T, ModelError>`. `ModelError` implements `Display` and `std::error::Error`, so `?` converts it into `Box<dyn std::error::Error>`. The enum is not marked `#[non_exhaustive]`, so a future release can add a variant and break an exhaustive `match`. Keep a wildcard arm if your code matches on it. The full variant list is in [Validating an install](validating-an-install.md#errors).

## Where to go next

* [Validating an install](validating-an-install.md): the manifest, `ArchConfig`, hashing, and the install receipt.
* [Reading resident weights](resident-weights.md): the tensor index and the mapped weight buffer.
* [API docs on docs.rs](https://docs.rs/turbospark-model-io/latest/turbospark_model_io/)
* [GTurbo format](https://github.com/whit3rabbit/turbospark/blob/main/docs/GTURBO.md): the on-disk layout this crate reads.
* [Model family table](https://github.com/whit3rabbit/turbospark/blob/main/docs/MODEL_FAMILY.md): which families the baselines cover.
