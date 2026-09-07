# turbospark-vision-io

Portable vision preprocessing for the `qwen3_5` vision tower. Takes encoded
image bytes and produces the patch-row matrix the tower's patch-embedding
GEMM consumes, plus the three position tables the tower and the text trunk
need.

Plain arithmetic on `Vec<f32>`: no Metal, no macOS, no GPU, no model
install. It builds and tests for a non-macOS target.

Downstream workspace crates depend on it under its real package name, no
alias (unlike `foundation`, `compute`, etc.):

```toml
[dependencies]
turbospark-vision-io = { path = "../vision-io", version = "0.1.0" }
```

## What it does

```rust
use turbospark_vision_io::{decode_image_bytes, preprocess, PreprocessParams};

let params = PreprocessParams::from_preprocessor_config_json(&config_json)?;
let image = decode_image_bytes(&bytes)?;
let out = preprocess(&image, &params)?;
// out.patch_rows  -> grid.patches() * params.patch_dim() floats
// out.grid        -> (t, h, w) in patches
// out.merged_tokens -> what this image costs the trunk in tokens
```

The pipeline is smart resize (round both edges to `patch_size * merge_size`,
then clamp the pixel count into the checkpoint's budget), PIL bicubic
resample, rescale and per-channel normalize, then patchify in merge-window
order.

Beside it, three tables of pure index and weight arithmetic:

- `pos_embed_weights` -- the four indices and four bilinear weights per
  patch for interpolating the tower's fixed 48x48 position-embedding grid
  onto this image's grid.
- `vision_rope_freq_rows` -- the tower's 2-D rotary frequency row per patch,
  height half then width half.
- `mrope_position_triples` -- the `(t, h, w)` position every prompt token
  occupies once images are spliced in, plus the `rope_delta` decoding needs
  and the placeholder spans the injection step writes into.

None of the three gathers anything: the values they index live in GPU
buffers this crate cannot see.

## Key Modules

- `params.rs`: `PreprocessParams` and the `preprocessor_config.json` reader.
- `smart_resize.rs`: `resized_dims`.
- `resize.rs`: PIL bicubic, reproduced in fixed point.
- `patchify.rs`: `patch_rows`, `GridThw`.
- `pos_embed.rs` / `rope.rs` / `mrope.rs`: the three position tables.

## Development & Test Commands

```sh
cargo test -p turbospark-vision-io
cargo check --target x86_64-unknown-linux-gnu -p turbospark-vision-io
```

Every golden fixture under `tests/generated/` is probed from the vendored
mlx-vlm reference by `scripts/qwen3vl_vision_oracle.py`, never transcribed.
Each file's header carries its own regeneration command.

## Crate Gotchas

1. **Patch rows use `(T, P_h, P_w, C)` inside each row; the reference uses
   `(C, T, P_h, P_w)`.** Deliberate: it matches the tower's stored
   `patch_embed.proj.weight` layout, so no permutation is needed at repack
   or at decode. The parity test bridges the two orders explicitly.
2. **PIL bicubic, not torchvision.** The two differ by whole 8-bit levels,
   and which one is correct is a property of the calling checkpoint.
3. **Fixtures are never sourced from JPEGs**, because JPEG decode is not
   bit-compatible across decoders and a failure could not be attributed.
4. **The pixel budget has no default** and is refused if the checkpoint
   declares neither spelling of it.

`crates/vision-io/CLAUDE.md` has the full list with the reasoning.
