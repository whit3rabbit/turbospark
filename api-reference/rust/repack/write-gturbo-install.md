---
description: "How to write a .gturbo install directory from resident tensors or streamed expert layers, then verify it with turbospark-repack."
icon: box
---
# Write a .gturbo install

Pick one writer, give it the tensors you prepared, and then verify the directory. Use `write_gturbo_install_with_resident_index` when every weight fits in the resident index. Use `StreamingGturboWriter` when expert layers are too large to hold in memory together. The writers assemble files. They do not choose tensor names, quantize for you, or prove that the model runs.

## Write an install from resident tensors

This example quantizes one 64 by 64 matrix, concatenates its rows into a resident tensor, writes the install, and verifies it. It compiles and runs as written. The output is a one-tensor install built on a tiny fixture architecture, so it is not a runnable model.

```rust
use std::path::Path;

use turbospark_repack::{
    build_resident_weights_bin, quantize_matrix_int4, tiny_gemma4_arch,
    verify_install_full_sha256, write_gturbo_install_with_resident_index, ResidentTensorSpec,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = Path::new("out/tiny-install");
    let arch = tiny_gemma4_arch(64, 1);

    // One 64x64 matrix, row-major FP32. Columns must be a multiple of 64.
    let data: Vec<f32> = (0..64 * 64).map(|i| (i % 17) as f32 * 0.01).collect();
    let rows = quantize_matrix_int4(&data, 64, 64)?;

    // Concatenate per-row packed bytes, scales, and biases in row order.
    let mut spec = ResidentTensorSpec {
        name: "embed_lm_head".into(),
        packed: Vec::new(),
        scales: Vec::new(),
        biases: Vec::new(),
        rows: 64,
        cols: 64,
    };
    for row in &rows {
        spec.packed.extend_from_slice(&row.packed);
        spec.scales.extend_from_slice(&row.scales);
        spec.biases.extend_from_slice(&row.biases);
    }

    let resident = build_resident_weights_bin(&[spec]);
    write_gturbo_install_with_resident_index(dir, &arch, "example/tiny-install", &resident)?;
    verify_install_full_sha256(dir, &arch)?;
    println!("wrote and verified {}", dir.display());
    Ok(())
}
```

For a real checkpoint, get the `ArchConfig` from the family's config parser, such as `parse_gemma4_config(&json)`, and not from `tiny_gemma4_arch`. The resident tensors must use the names that family's loader expects. The Gemma 4 writer in `crates/repack/src/gemma4_checkpoint/` is the reference for a full set. Its `write_gemma4_install` and `write_gemma4_install_streamed` functions are public.

The crate has no public function that turns a `Vec` of quantized rows into a resident tensor. The concatenation above does that work. The same pattern works for INT8 with `quantize_matrix_int8` and `Int8AffineRow`.

## Mix quantized and raw tensors

Real checkpoints keep some tensors unquantized. Norms and scalars are common examples. For those, build a `ResidentEntrySpec` list. Pass that list to `build_resident_weights_bin_mixed`.

The enum has `Int4`, `Int8`, `Int1`, `Int2`, and `Raw` variants. `Raw` takes a `RawTensorSpec`, which holds a `dtype` tag, the bytes as they should appear on disk, and a rank-4 shape.

This example compiles and runs, and it prints the size of the resident region.

```rust
use turbospark_repack::{
    build_resident_weights_bin_mixed, RawTensorSpec, ResidentEntrySpec, DTYPE_FP32,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Four FP32 values, so the byte length matches shape (4, 0, 0, 0) and dtype FP32.
    let norm = [1.0f32, 1.0, 1.0, 1.0];
    let bytes: Vec<u8> = norm.iter().flat_map(|v| v.to_le_bytes()).collect();
    let raw = ResidentEntrySpec::Raw(RawTensorSpec {
        name: "layer0.input_norm".into(),
        dtype: DTYPE_FP32,
        bytes,
        shape: (4, 0, 0, 0),
    });
    let resident = build_resident_weights_bin_mixed(&[raw]);
    println!("resident region is {} bytes", resident.len());
    Ok(())
}
```

The raw dtype tags are `DTYPE_BF16`, `DTYPE_FP16`, and `DTYPE_FP32`. The module also defines tags for quantized and GGUF block types. The tag you choose must match the bytes you pass, and the writer does not reinterpret them. Check each raw tensor's byte length against its shape yourself.

`write_gturbo_install` also exists. It takes layers and an expert stride, and it wraps a raw resident region in an empty index. For resident weights, prefer `write_gturbo_install_with_resident_index`. For a complete `model_weights.bin` plus expert layers, use `write_gturbo_install_with_resident_index_and_experts`.

## Stream expert layers to disk

`StreamingGturboWriter` writes one layer file as you hand it over, so you can drop each layer's bytes after the call. The example below writes one layer and then finishes the install. It compiles and runs as written. The architecture has no routed experts, so this layer only shows the calls. A real MoE install takes its experts from the checkpoint.

```rust
use std::path::Path;

use turbospark_repack::{
    build_resident_weights_bin, tiny_gemma4_arch, verify_install_full_sha256, ExpertBlob,
    LayerBlobs, StreamingGturboWriter, SubTensor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = Path::new("out/streamed-install");
    let arch = tiny_gemma4_arch(64, 1);

    // The loader rejects a stride that is not a multiple of 4096 bytes, but
    // the writer does not check it. Each expert must fit inside the stride.
    let expert_stride = 4096;
    let experts_per_layer = 1;
    let mut writer = StreamingGturboWriter::new(dir, expert_stride, experts_per_layer)?;

    let layer = LayerBlobs {
        layer: 0,
        experts: vec![ExpertBlob {
            expert: 0,
            sub_tensors: vec![SubTensor {
                role: "gate".into(),
                bytes: vec![0u8; 16],
                dtype: "F32".into(),
                shape: vec![4],
            }],
        }],
    };
    writer.write_layer(&layer)?;

    // The resident index is written last, when finish runs.
    let resident = build_resident_weights_bin(&[]);
    writer.finish(&arch, "example/streamed-install", &resident)?;
    verify_install_full_sha256(dir, &arch)?;
    println!("streamed and verified {}", dir.display());
    Ok(())
}
```

The call order matters:

1. `StreamingGturboWriter::new(dir, expert_stride, experts_per_layer)` creates `packed_experts/`.
2. `write_layer` writes `packed_experts/layer_NN.bin` and records its layout entry. Call it once per layer.
3. `set_quant` and `set_hadamard` store optional manifest sections. Call them before `finish`. A Hadamard section needs `hadamard.bin` on disk before `finish_streaming` runs, because the manifest hashes every listed file.
4. `finish` writes `layout.json`, `model_weights.bin`, and `manifest.json`. Use `finish_streaming` instead when the resident region is large. It takes a closure that writes the resident bytes straight into a buffered file.

Two checks come from the writer's own error type. An expert whose bytes exceed the stride returns `WriterError::ExpertOversized`. A layer with the wrong number of experts returns `WriterError::WrongExpertCount`.

The stride has a second rule. The writer accepts any stride, but `verify_install_full_sha256` loads the manifest through `turbospark-model-io`, and that loader rejects a stride that is not a multiple of 4096. The rejection is `ModelError::ExpertStrideNotPageAligned`, and it appears at verification, after the files are written. The example's 4096 stride passes the check.

The Gemma 4 writer rounds each expert blob up to `GTURBO_PAGE_BYTES`, which is 16 KiB, before it picks a stride. `StreamingGturboWriter` uses the stride you pass, so check it against the format spec before you copy a stride.

A write that fails partway can leave files behind. The manifest is written before verification runs, so a failed verification can leave a complete-looking directory. Delete the target directory before you retry.

## Write a vision component

`write_packed_vision(dir, blocks, block_stride)` writes the vision tower's blocks under `packed_vision/`. The function takes a `LayerBlobs` and a block stride. Its doc comment says to call it before the manifest is written, because the manifest hashes every file it lists. A text-only install of a multimodal checkpoint is not the whole model. The vision layout is in [GTURBO section 3.5](https://github.com/whit3rabbit/turbospark/blob/main/docs/GTURBO.md), and the runtime side is in [VISION.md](https://github.com/whit3rabbit/turbospark/blob/main/docs/VISION.md).

## Verify the install

`verify_install_full_sha256(dir, &arch)` loads `manifest.json` against the `ArchConfig` you expect, then re-hashes every file the manifest lists. It returns `ModelError::ChecksumMismatch` on the first file whose hash differs. It reads every byte, so it is slower than the faster path in `turbospark-model-io` that trusts a prior receipt.

This check confirms consistency between the files and the manifest. It does not confirm who published the install. A party that can replace the files and the manifest together passes it. The trust boundary is in [GTURBO section 5](https://github.com/whit3rabbit/turbospark/blob/main/docs/GTURBO.md).

A passing verification is not a quality result. Quality and parity need the gates in [NEW_MODEL.md](https://github.com/whit3rabbit/turbospark/blob/main/docs/NEW_MODEL.md) and [MODELS.md](https://github.com/whit3rabbit/turbospark/blob/main/docs/MODELS.md).

## Errors

- `WriterError::Io` carries the path and the OS message.
- `WriterError::ExpertOversized` and `WriterError::WrongExpertCount` report a layer that does not match the stride or the expert count.
- `ModelError` from `verify_install_full_sha256` reports a checksum mismatch or a manifest the loader rejects.

For a larger example, read `crates/repack/tests/gturbo_writer.rs` in the repo. It writes a synthetic install and reads it back through every loader and the full verifier.
