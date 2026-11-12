---
description: "Portable vision preprocessing for vision-language towers: image decode, PIL-bicubic smart resize, patchify, and the three position tables."
icon: eye
---

# vision-io

`crates/vision-io` (package `turbospark-vision-io`) is the small, portable
front half of the vision pipeline: decoded RGB8 pixels go in, and the patch
matrix plus the position metadata the tower and the language trunk need come
out. Nothing here touches Metal, macOS, or a model install; it is plain
arithmetic on `Vec<f32>` and `Vec<u8>` behind `#![forbid(unsafe_code)]`, and
staying buildable for a non-macOS target is an explicit crate contract. The
arithmetic is ported to match the reference processor exactly, so parity is
bit-level, not visual.

## What it does

| Area | Modules | Contents |
| --- | --- | --- |
| Decode | `decode` | JPEG/PNG bytes to `Rgb8Image` via the pure-Rust `image` decoders, `image_dimensions`, `MAX_IMAGE_DIM` |
| Smart resize | `smart_resize`, `rounding` | `resized_dims`, the HF `smart_resize` port, using Python's round-half-to-even |
| Resize | `resize` | `resize_bicubic_pil` and `resample_pil`, PIL's bicubic reproduced bit for bit on RGB8 |
| Normalize | `normalize` | `rescale_and_normalize`: rescale, then per-channel mean and std in that fixed order |
| Patchify | `patchify` | `patch_rows` and `GridThw`: the patch matrix in merge-window order |
| Pipeline | `preprocess` | `preprocess`, the composed smart resize, resize, normalize, patchify step |
| Position tables | `pos_embed`, `rope` | `pos_embed_weights` (bilinear indices and weights) and `vision_rope_freq_rows` |
| Prompt splicing | `mrope` | `splice_and_walk`, `mrope_position_triples`: one `(t, h, w)` position per token, plus `rope_delta` |
| Parameters | `params` | `PreprocessParams` and the `preprocessor_config.json` reader |

The three position tables are index and weight arithmetic only: the crate
computes which table rows each patch needs and with what weights, while the
gather itself happens in the tower, where the tables live in GPU buffers this
crate cannot see.

## Where it sits

Depends on `core` (the `TokenId` type), `serde_json`, and the external `image`
crate pinned to its pure-Rust JPEG and PNG decoders with default features off,
which is what keeps the cross-target check green. Its consumers are `runtime`
(the `qwen3_5` vision tower stages), `embeddings` (SigLIP2 processing), `cli`
(image prompts for `turbospark generate`), `server`, `ffi`, and `bench`. The
image generation crate `turbospark-image` is a separate pipeline and does not
depend on this crate.

## Public API tour

| Item | Path | Meaning |
| --- | --- | --- |
| `decode_image_bytes`, `decode_image_file` | `decode` | Bytes or file to `Rgb8Image { width, height, data }` |
| `resized_dims(orig_h, orig_w, params)` | `smart_resize` | The resize target the reference would pick, honoring the pixel budget |
| `resize_bicubic_pil`, `resample_pil` | `resize` | PIL-bicubic resize on interleaved RGB8 |
| `rescale_and_normalize(src, params)` | `normalize` | `(sample * rescale - mean[c]) / std[c]` per channel |
| `preprocess(image, params)` | `preprocess` | The whole pipeline; returns `PreprocessedImage` |
| `PreprocessedImage` | `preprocess` | `patch_rows`, `grid`, `merged_tokens`, and the chosen `resized` dimensions |
| `GridThw::new(t, h, w)`, `.patches()`, `.merged_tokens(merge)` | `patchify` | The patch grid every downstream count derives from |
| `pos_embed_weights(grid, n_grid_per_side, params)` | `pos_embed` | Four indices and weights per patch, in merge-window order |
| `vision_rope_freq_rows(grid, head_dim, theta, params)` | `rope` | The tower's 2-D rotary frequency rows (`VISION_ROPE_THETA` is 10,000, not the trunk's) |
| `splice_and_walk(ids, grids, special, merge_size)` | `mrope` | Expand image placeholders and assign every token its `(t, h, w)` triple |
| `MropePositions { triples, rope_delta, spans }` | `mrope` | The walk's answer; `rope_delta` fixes up decode-time positions |
| `PreprocessParams::from_preprocessor_config_json` | `params` | Resolve patch size, merge size, pixel budget, and normalization from the checkpoint |

## Examples

### Decode an image and run the preprocessing pipeline

Adapted from `crates/vision-io/tests/preprocess_parity.rs` and the composed
pipeline in `src/preprocess.rs`. The parameters are read from the
checkpoint's `preprocessor_config.json` rather than recalled; a surprising
token count is almost always a surprising resize, so `resized` is reported
alongside the grid.

```rust
use turbospark_vision_io::{decode_image_file, preprocess, PreprocessParams};

let config = std::fs::read_to_string(
    "/models/checkpoint/preprocessor_config.json",
)?;
let params = PreprocessParams::from_preprocessor_config_json(&config)?;

let image = decode_image_file(std::path::Path::new("photo.jpg"))?;
let prep = preprocess(&image, &params)?;

println!(
    "{}x{} -> {}h x {}w, {} patch rows, {} trunk tokens",
    image.width, image.height,
    prep.resized.0, prep.resized.1,
    prep.grid.patches(), prep.merged_tokens
);
// prep.patch_rows is the matrix the tower's patch-embedding GEMM consumes:
// grid.patches() rows of params.patch_dim() floats each.
```

### Splice an image into a prompt and get positions

Adapted from `crates/cli/src/generate/vision.rs` and
`tests/position_tables_parity.rs`. The rendered template carries one
placeholder per image; the model must instead see the expanded sequence, and
the trunk needs one `(t, h, w)` triple per token. `splice_and_walk` derives
each image's merged-token count from its own grid, so the counts cannot drift
from the grids. The two special ids come from the checkpoint's vision config.

```rust
use turbospark_vision_io::{splice_and_walk, GridThw, VisionSpecialIds};

// ids: the rendered prompt with one image_pad placeholder per image.
// grids: one GridThw per image, from `preprocess`.
let special = VisionSpecialIds {
    vision_start: vision_config.vision_start_token_id as i32,
    image_pad: vision_config.image_token_id as i32,
};
let spliced = splice_and_walk(&ids, &grids, special, params.merge_size)?;

// spliced.ids is what gets prefilled. spliced.positions.triples has one
// (t, h, w) entry per id, and spliced.positions.rope_delta is the offset a
// decode step adds to the cache position, because an image block advances
// the position clock by max(grid side), not by its token count.
```

An image costs fewer positions than tokens: a 4,096-token image block spends
only `max(llm_grid_h, llm_grid_w)` positions of context, which is the whole
point of the walk.

### Build the two tower tables for a grid

Signatures from `src/pos_embed.rs` and `src/rope.rs`. The table side comes
from the checkpoint (this family ships a 48 x 48 learned grid); the tower's
rotary base is 10,000 even though the trunk uses a different theta.

```rust
use turbospark_vision_io::{
    pos_embed_weights, rope::VISION_ROPE_THETA, vision_rope_freq_rows,
};

let table = pos_embed_weights(prep.grid, 48, &params)?;
// table.indices and table.weights: four bilinear corners per patch, weights
// summing to one, in the same merge-window order as prep.patch_rows.

// head_dim comes from the tower config; it must be a positive multiple of 4.
let freq_rows = vision_rope_freq_rows(prep.grid, head_dim, VISION_ROPE_THETA, &params)?;
```

Neither call gathers embeddings; the tower reads `table.indices` and
`table.weights` against its own resident copy of the learned table.

## Gotchas

{% hint style="warning" %}
Two orders are load-bearing and silently wrong if swapped. Every sequence this
crate emits (patch rows, pos-embed entries, RoPE rows) is in merge-window
order: the grid is cut into merge-size windows and one whole window is emitted
before the next. Inside one patch row the feature order is
`(T, P_h, P_w, C)`, which deliberately differs from the reference's
`(C, T, P_h, P_w)` so the tower's patch-embedding weight can be copied
verbatim at repack. A mismatch here is fluent wrong numerics, never a crash.
{% endhint %}

{% hint style="warning" %}
JPEG decode is not bit-compatible across decoders: Pillow and the Rust `image`
crate disagree on individual samples of the same file (different IDCT and
chroma upsampling). The crate's own parity fixtures are PNG for exactly this
reason. Do not source a tensor-parity fixture from a JPEG, or you will be
comparing decoder differences. Also note `smart_resize` needs Python's
round-half-to-even; `f64::round` does not match, which is why
`round_half_to_even` exists.
{% endhint %}

## Related pages

- [Vision models](../multimodal/vision-models.md)
- [runtime](runtime.md): the vision tower stages that consume this crate
- [embeddings](embeddings.md): SigLIP2 preprocessing on the same primitives
- [cli](cli.md): the `generate` image-prompt path
- [core](core.md): the `TokenId` type shared across the boundary
