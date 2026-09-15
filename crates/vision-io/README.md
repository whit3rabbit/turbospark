# turbospark-vision-io

Portable vision preprocessing for the `qwen3_5` vision tower. Takes encoded image bytes and produces the patch-row matrix the tower's patch-embedding GEMM consumes, plus the three position tables the tower and the text trunk need.

Plain arithmetic on `Vec<f32>`: no Metal, no macOS, no GPU, no model install. It builds and tests cleanly for non-macOS targets.

Downstream workspace crates depend on it under its real package name, no alias:

```toml
[dependencies]
turbospark-vision-io = { path = "../vision-io", version = "0.1.0" }
```

## Purpose & Role

`turbospark-vision-io` implements the complete image preprocessing and coordinate grid math required for multimodal inference. It runs on the host CPU prior to Metal dispatch, preparing patch embeddings and positional metadata so the GPU forward runner can ingest multimodal tokens with zero device-side preprocessing.

## Safety

- `#![forbid(unsafe_code)]` is enforced in `lib.rs`.
- Contains no platform-specific intrinsics or native dependencies beyond standard pure Rust image decoders.

## What It Does

```rust
use turbospark_vision_io::{decode_image_bytes, preprocess, PreprocessParams};

let params = PreprocessParams::from_preprocessor_config_json(&config_json)?;
let image = decode_image_bytes(&bytes)?;
let out = preprocess(&image, &params)?;
// out.patch_rows   -> grid.patches() * params.patch_dim() floats
// out.grid         -> (t, h, w) in patches
// out.merged_tokens -> what this image costs the trunk in tokens
```

The pipeline executes smart resize (rounding both edges to `patch_size * merge_size`, then clamping the pixel count into the checkpoint's budget), PIL bicubic resample in fixed-point arithmetic, rescale and per-channel normalization, then patchifies in merge-window order.

Beside it, three tables of pure index and weight arithmetic:
- `pos_embed_weights`: The four indices and four bilinear weights per patch for interpolating the tower's fixed 48x48 position-embedding grid onto this image's grid.
- `vision_rope_freq_rows`: The tower's 2D rotary frequency row per patch, height half then width half.
- `mrope_position_triples`: The `(t, h, w)` position every prompt token occupies once images are spliced in, plus the `rope_delta` decoding needs and the placeholder spans the injection step writes into.

## Key Modules

- `decode.rs`: Image format decoding (PNG, JPEG) into raw RGB pixel buffers.
- `params.rs`: `PreprocessParams` parser decoding `preprocessor_config.json`.
- `smart_resize.rs`: Aspect-ratio-preserving dimension rounding (`resized_dims`) adhering to checkpoint pixel bounds.
- `resize.rs`: Deterministic PIL-compatible bicubic image resampling implemented in fixed-point arithmetic.
- `normalize.rs`: Per-channel mean/standard deviation normalization and float scaling.
- `patchify.rs`: Extraction of spatial patch rows into the merge-window ordering (`patch_rows`, `GridThw`).
- `pos_embed.rs`: 2D bilinear interpolation weight and index calculator for position embeddings.
- `rope.rs`: 2D spatial RoPE frequency table generator for vision attention.
- `mrope.rs`: Multimodal 3D RoPE coordinate triple assignment for prompt/image token merging.
- `rounding.rs`: Fixed-point numerical rounding helpers.
- `error.rs`: Typed preprocessing and dimension validation errors.

## Development & Test Commands

```sh
# Run portable unit and parity tests
cargo test -p turbospark-vision-io

# Run cross-target compilation check
cargo check --target x86_64-unknown-linux-gnu -p turbospark-vision-io
```

## Tests

- `tests/pipeline_units.rs`: Unit tests for parameter extraction, image decoding, and dimension validation.
- `tests/smart_resize_parity.rs`: Validates smart resize dimensions against reference Python implementations.
- `tests/preprocess_parity.rs`: Validates end-to-end preprocessing float output against golden mlx-vlm fixtures.
- `tests/position_tables_parity.rs`: Verifies 2D RoPE, mRoPE triples, and position interpolation tables.

Every golden fixture under `tests/generated/` is probed from the vendored mlx-vlm reference by `scripts/qwen3vl_vision_oracle.py`, never transcribed. Each file's header carries its own regeneration command.

## Crate Gotchas

1. **Patch Row Memory Layout**: Patch rows use `(T, P_h, P_w, C)` inside each row, whereas the reference uses `(C, T, P_h, P_w)`. This is deliberate: it matches the tower's stored `patch_embed.proj.weight` layout, eliminating transpose overhead during repack or decode.
2. **PIL Bicubic Compatibility**: Uses PIL bicubic resampling rather than torchvision. The two differ significantly in pixel levels; fidelity requires exact PIL matching.
3. **JPEG Non-Determinism**: Parity test fixtures are generated from PNG rather than JPEG sources, as JPEG decoding implementations differ across platforms and libraries.
4. **Mandatory Pixel Budget**: The pixel budget has no default value and is rejected if the checkpoint declares neither spelling of it in `preprocessor_config.json`.
