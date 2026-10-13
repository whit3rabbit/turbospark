---
description: "How to read and validate a .gturbo install with turbospark-model-io: family detection, manifest checks against ArchConfig, install receipts, SHA-256, and ModelError."
icon: box
---
# Validating an install

Validation runs in three steps. Detect the family, load the manifest against that family's expected architecture, then check the receipt. The manifest step matters most. `load_manifest` runs every structural check and returns a `Manifest` only when they all pass. The receipt step is optional, and it checks records rather than file contents.

## The order

Start with `peek_family`. It reads `manifest.json` and returns the `ModelFamily` named in `arch.family`. A manifest with no `family` field is treated as `Gemma4`, which is the format's original architecture. An unknown family string returns `ModelError::IndexCorrupt`.

Next, resolve the expected shape with `known_architecture(family)`. This returns the baseline `ArchConfig` the family ships with. Then call `load_manifest(dir, &expected, max_bytes)`, which parses the manifest and validates it.

```rust
use std::path::Path;

use turbospark_model_io::{known_architecture, load_manifest, peek_family, ModelError, DEFAULT_MAX_BYTES};

fn load(dir: &Path) -> Result<turbospark_model_io::Manifest, ModelError> {
    let family = peek_family(dir, DEFAULT_MAX_BYTES)?;
    let expected = known_architecture(family);
    load_manifest(dir, &expected, DEFAULT_MAX_BYTES)
}
```

`max_bytes` caps the size of `manifest.json` before it is read. Use `DEFAULT_MAX_BYTES`, which is 4 MiB, unless you have a reason to change it.

## What the manifest check covers

`load_manifest` runs these checks in order. The first failure returns its error.

1. `manifest.json` exists, is under `max_bytes`, and parses as JSON.
2. The manifest does not declare the `image-generation` capability.
3. `magic` is `GTURBO`, and `version_major` is `1`.
4. Every key in `flags` is one of `streamingPresent`, `turboQuantKV`, or `aneSharedExpert`. A `turboQuantKV` flag set to `true` is rejected, because KV quantization is chosen at open time.
5. Each `arch` field matches the expected `ArchConfig`. A mismatch names the field.
6. A `quant` block, when present, is structurally valid. A manifest whose layer count and hidden size match a shipped baseline must carry one.
7. `expertStride` is a multiple of 4096 bytes.
8. `files` lists `model_weights.bin` and `packed_experts/layout.json`, plus one expert file per layer.
9. If the expected architecture has a vision tower, `files` also lists `packed_vision/layout.json` and `packed_vision/blobs.bin`.
10. If a `hadamard` section is present, its sign ranges fit inside `hadamard.bin` and do not overlap.

Passing this check means the manifest agrees with the shape you expect. That is all it proves. It does not show that the runtime can execute that shape.

### Custom shapes

Every field of `ArchConfig` is public. You can start from a baseline with struct update syntax, such as `ArchConfig { hidden_size: 2048, ..known_architecture(family) }`. The manifest must then agree with every field you set. A wrong field fails with `ArchMismatch`, which is the behavior you want. Keep in mind that the check only confirms the two sides match.

## Install receipts

A receipt is a `verified-install.json` file that records the size and SHA-256 of each file in an install. Reading it lets a loader skip re-hashing large weight files. Use it with `load_install_receipt` and `validate_install_receipt`.

```rust
use std::path::Path;

use turbospark_model_io::{
    hash_file, known_architecture, load_install_receipt, load_manifest, peek_family,
    validate_install_receipt, DEFAULT_MAX_BYTES, INSTALL_RECEIPT_DEFAULT_MAX_BYTES,
};

fn verify_receipt(dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let family = peek_family(dir, DEFAULT_MAX_BYTES)?;
    let expected = known_architecture(family);
    let manifest = load_manifest(dir, &expected, DEFAULT_MAX_BYTES)?;

    let manifest_path = dir.join("manifest.json");
    let manifest_sha256 = hash_file(&manifest_path, 1 << 20)?;
    let manifest_size = std::fs::metadata(&manifest_path)?.len();

    let receipt = load_install_receipt(dir, INSTALL_RECEIPT_DEFAULT_MAX_BYTES)?;
    validate_install_receipt(&receipt, dir, &manifest, &manifest_sha256, manifest_size)?;
    println!("receipt ok: {} files", receipt.files.len());
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: receipt <model.gturbo directory>");
    verify_receipt(Path::new(&dir))
}
```

The example compiles against the crate. It needs a real install to succeed, so on any other directory it exits with an error. For a worked success case, see the crate's [receipt tests](https://github.com/whit3rabbit/turbospark/blob/main/crates/model-io/tests/install_receipt.rs).

`validate_install_receipt` checks these things, in this order:

* `schemaVersion` is `1`. The field defaults to `1` when absent.
* `manifestSha256` matches the SHA-256 you pass in, ignoring case.
* `modelDirectoryPath` equals the canonical path of the install directory. Moving the directory invalidates the receipt.
* The receipt lists exactly the files in `manifest.files` plus `manifest.json`.
* The receipt size and SHA-256 for `manifest.json` match the values you pass in.
* The receipt size and SHA-256 for every other file match the manifest.

The check reads only the receipt and the manifest. It never opens the weight files, so it stays fast on large installs.

{% hint style="warning" %}
A receipt is a record, not a signature. The crate does not sign or authenticate `verified-install.json`, so anyone who can write to the install directory can write a receipt that passes. Use the receipt for fast integrity checks on your own installs. Do not treat it as proof that an install came from a trusted source.
{% endhint %}

## Hashing files

`hash_file(path, chunk_bytes)` returns the lowercase hex SHA-256 of a file. It streams the file in `chunk_bytes` pieces. The crate uses `1 << 20` (1 MiB). A chunk size of zero is raised to one byte internally. That is correct but very slow, so pass a real chunk size.

```rust
use std::path::Path;

use turbospark_model_io::{hash_data, hash_file, verify_file, ModelError};

fn check(path: &Path, expected: &str) -> Result<(), ModelError> {
    let digest = hash_file(path, 1 << 20)?;
    println!("sha256 = {digest}");
    println!("empty = {}", hash_data(&[]));
    verify_file(path, "model_weights.bin", expected)
}
```

`verify_file` compares case-insensitively and returns `ChecksumMismatch` with the name you pass. `hash_file_observed` takes a closure that receives the bytes hashed so far. Return `false` from it to stop early, and the call returns `Ok(None)`. Use it for progress bars or cancellation.

## Errors

Every failure in this guide is a `ModelError`. The install path raises these variants:

| Variant | Raised when |
| --- | --- |
| `PartialInstall { path }` | The directory has no `manifest.json` |
| `NotAGTurboDirectory` | `magic` is not `GTURBO` |
| `UnsupportedVersion { major, minor }` | `version_major` is not `1` |
| `UnsupportedCapability { capability }` | The manifest declares `image-generation` |
| `UnknownFlag { name }` | `flags` contains an unknown key |
| `ArchMismatch { field, expected, actual }` | An `arch` field differs from the expected `ArchConfig` |
| `ExpertStrideNotPageAligned { stride, page_size }` | `expertStride` is not a multiple of 4096 |
| `MissingFile { name }` | A required file is absent from `manifest.files` |
| `ChecksumMismatch { file }` | `verify_file` found a different digest |
| `TensorSizeMismatch { name, expected, actual }` | A mapped region runs past the end of the file |
| `IndexCorrupt { detail }` | Malformed JSON, a metadata file over its size cap, or a structural check failed |
| `IoFailed { call, detail }` | An OS call such as `open`, `read`, `stat`, or `mmap` failed |
| `TrustedReceiptInvalid { detail }` | The receipt is missing, oversized, malformed, or does not match the manifest |

Two more variants come from other parts of the crate. `TensorNotFound { name }` comes from name lookups in `safetensors`. `BadConfig { field, why }` comes from the Whisper config parser. Neither appears on the install path above.

`ModelError` is not `#[non_exhaustive]`, so keep a wildcard arm when you match on it.

```rust
use turbospark_model_io::ModelError;

fn describe(error: &ModelError) -> &'static str {
    match error {
        ModelError::PartialInstall { .. } => "download did not finish",
        ModelError::ArchMismatch { .. } => "checkpoint shape does not match the family",
        ModelError::TrustedReceiptInvalid { .. } => "receipt does not match the manifest",
        _ => "install refused",
    }
}
```

## Related

* [Overview](README.md)
* [Reading resident weights](resident-weights.md)
* [docs.rs: ModelError](https://docs.rs/turbospark-model-io/latest/turbospark_model_io/enum.ModelError.html)
