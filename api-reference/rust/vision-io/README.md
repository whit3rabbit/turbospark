---
description: "Portable vision preprocessing for the qwen3_5 vision tower: image bytes in, patch rows and position tables out."
icon: box
---

# turbospark-vision-io

Use this crate when you need the qwen3_5 vision tower's input in plain Rust. It takes encoded image bytes and returns the patch-row matrix the tower's patch-embedding GEMM reads, plus the position tables the tower and the text trunk need. Apart from decoding, it is arithmetic on `Vec<f32>`, with no Metal, no macOS, and no model install, so it builds on non-macOS targets.

The crate stops before the GPU work. It does not run the GEMM, gather rows from the learned position table, or apply the rotary frequencies. Whatever loads the tower weights does those steps. The in-tree consumer is `turbospark-runtime`.

## Add the crate

The package name is `turbospark-vision-io`, and the import path is `turbospark_vision_io`. Inside this workspace, the crate imports `turbospark-core` under a local alias in its manifest. External users never see that alias, so your code uses the package and import names above.

Add the package from crates.io once the version you need is published. Until then, use a path or git dependency from a checkout:

```toml
[dependencies]
turbospark-vision-io = { git = "https://github.com/whit3rabbit/turbospark" }
```

The crate targets Rust 1.82 and edition 2021, and it has no Cargo features. Decoding uses the pure-Rust JPEG and PNG codecs from the `image` crate, so no C codec libraries are linked.

## What you get back

`preprocess` returns a [`PreprocessedImage`](https://docs.rs/turbospark-vision-io/latest/turbospark_vision_io/struct.PreprocessedImage.html). The three position tables come from separate functions, because the caller may need only some of them.

| Output | Produced by | What you do with it |
| --- | --- | --- |
| `patch_rows`: `Vec<f32>` with `grid.patches() * params.patch_dim()` values | `preprocess` | Feed it to the patch-embedding GEMM. Rows are in merge-window order. |
| `grid`: `GridThw` (`t`, `h`, `w`, in patches) | `preprocess` | Size the position tables and the token expansion. |
| `merged_tokens`: `usize` | `preprocess` | Count the trunk tokens this image costs, which is `patches / merge_size^2`. |
| `resized`: `(height, width)` in pixels | `preprocess` | Check this first when a token count looks wrong. |
| `PosEmbedTable` | `pos_embed_weights` | Four index arrays and four weight arrays, one entry per patch. The tower forms the bilinear blend from its learned grid. |
| `Vec<f32>` rotary rows | `vision_rope_freq_rows` | Per-patch rotary frequencies, `head_dim / 2` values each. |
| `SplicedPrompt` | `splice_and_walk` | `ids`: the expanded token ids. `positions`: a `(t, h, w)` triple per id, the image `spans`, and `rope_delta`. |

Every per-patch output uses the same merge-window order, so entry `i` in each one describes the same patch. The reason for that order is in the next section.

## Quick start

This example reads a checkpoint's config and one image file, then runs the pipeline. It compiles against the crate. The file paths are placeholders you replace with your own.

```rust
use std::path::Path;

use turbospark_vision_io::{decode_image_file, preprocess, PreprocessParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = std::fs::read_to_string("preprocessor_config.json")?;
    let params = PreprocessParams::from_preprocessor_config_json(&config)?;

    let image = decode_image_file(Path::new("photo.jpg"))?;
    let out = preprocess(&image, &params)?;

    println!("resized to {}x{}", out.resized.1, out.resized.0);
    println!("grid (t, h, w): {:?}", out.grid);
    println!("trunk tokens: {}", out.merged_tokens);
    println!("patch floats: {}", out.patch_rows.len());
    Ok(())
}
```

The pipeline runs in a fixed order. It chooses the resized size first, resizes the decoded RGB8 pixels with a PIL-style bicubic filter, normalizes the resized samples, and then cuts the result into patch rows. Normalizing before the resize would clamp negative samples, so the order is part of the contract.

## Two orders to keep straight

{% hint style="warning" %}
The inner feature order of each patch row is `(T, P_h, P_w, C)`. The reference processor emits `(C, T, P_h, P_w)`. The crate does this on purpose, so the checkpoint's `patch_embed.proj.weight` (stored as `[out, T, P, P, C]`) loads without a permutation. If you feed these rows to weights laid out the reference's way, the GEMM still runs, and the image is read as a different image without any error.
{% endhint %}

Row order is merge-window order. The patch grid is cut into `merge_size x merge_size` windows, and each whole window is emitted before the next one. Inside a window, patches run row-major. The spatial-merge step reads the rows in this order, so keep it.

## Position tables

**Learned position embedding.** `pos_embed_weights` maps each patch onto the tower's learned grid. The mapping keeps the first and last patch on the first and last table row. That differs from the half-pixel convention the image resize uses, so interior patches land slightly differently than a resampling library would place them.

The crate computes indices and weights only. An axis with one patch reads row zero of the table. The tower gathers the four rows and forms the weighted sum.

**Rotary frequencies.** `vision_rope_freq_rows` gives each patch a row of `head_dim / 2` values. The first half is the height frequencies, and the second half is the width frequencies. Swapping the halves keeps the shape and the value range, so it fails silently.

The coordinates are full-resolution patch positions, and the tower applies the rotation before the spatial merge. The vision base is `rope::VISION_ROPE_THETA`, which is 10,000. It is not the text trunk's base. `rope::vision_rope_freq_rows_default` uses that constant for you.

**Prompt positions.** `splice_and_walk` expands each image placeholder into that image's merged-token count, then assigns every id its `(t, h, w)` triple. An image block advances the position clock by its largest axis, not by its token count. Add `rope_delta` to a cache position to get the next token's position during decode. Prefer `splice_and_walk` over calling the splice and the walk separately. Separate calls can disagree about the counts, and the combined call derives both from the grids.

```rust
use turbospark_vision_io::{
    pos_embed_weights, rope::VISION_ROPE_THETA, splice_and_walk, vision_rope_freq_rows,
    PreprocessParams, PreprocessedImage, VisionSpecialIds,
};

fn position_tables(
    image: &PreprocessedImage,
    params: &PreprocessParams,
    prompt_ids: &[i32],
    vision_start: i32,
    image_pad: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    // 48 is the side of the tower's learned position grid for this family.
    let pos = pos_embed_weights(image.grid, 48, params)?;
    // head_dim is 72 for this family.
    let rope = vision_rope_freq_rows(image.grid, 72, VISION_ROPE_THETA, params)?;
    let prompt = splice_and_walk(
        prompt_ids,
        &[image.grid],
        VisionSpecialIds {
            vision_start,
            image_pad,
        },
        params.merge_size,
    )?;

    println!("pos embed patches: {}", pos.len());
    println!("rope floats: {}", rope.len());
    println!("expanded ids: {}", prompt.ids.len());
    println!("rope_delta: {}", prompt.positions.rope_delta);
    Ok(())
}
```

## Configuration

`PreprocessParams::from_preprocessor_config_json` reads a checkpoint's `preprocessor_config.json`. The reader refuses where the format gives no default, and it uses the format's own default where one exists. Those decisions are the reason to read the config instead of hardcoding values.

- **Pixel budget:** `min_pixels` and `max_pixels` win when both are present. Otherwise the reader uses `size.shortest_edge` and `size.longest_edge`, which are pixel counts despite the edge names. If neither form exists, the reader refuses. It does not fall back to the library default, which is much smaller for this family (see [the vision checkpoint notes](https://github.com/whit3rabbit/turbospark/blob/main/docs/VISION_PHASE0.md)).
- **Normalization:** `image_mean` and `image_std` are required, and each must hold three numbers. A wrong pair does not change the output shape, so the crate refuses to guess.
- **Rescale:** `rescale_factor` defaults to `1/255` when absent. That matches the reference processor's signature default.
- **Channels:** The crate fixes `in_channels` at 3, because the format's silence on channels means RGB.

`PreprocessParams` has public fields, so you can also build it by hand. The result for this family is `patch_dim()` of 1536, which is two frames times 16 by 16 pixels times three channels. Use `spatial_factor()` to get the resize granularity, which is `patch_size * merge_size`.

## Types you touch

| Type | Module | Use |
| --- | --- | --- |
| [`Rgb8Image`](https://docs.rs/turbospark-vision-io/latest/turbospark_vision_io/struct.Rgb8Image.html) | `decode` | Input to `preprocess`. `Rgb8Image::new` checks the buffer length against the dimensions. |
| [`PreprocessParams`](https://docs.rs/turbospark-vision-io/latest/turbospark_vision_io/struct.PreprocessParams.html) | `params` | Read from a config with `from_preprocessor_config_json`, or build directly. |
| [`PreprocessedImage`](https://docs.rs/turbospark-vision-io/latest/turbospark_vision_io/struct.PreprocessedImage.html) | `preprocess` | The patch rows, grid, token count, and resized size. |
| [`GridThw`](https://docs.rs/turbospark-vision-io/latest/turbospark_vision_io/struct.GridThw.html) | `patchify` | Grid in patches. `merged_tokens(merge_size)` gives the trunk count. |
| [`PosEmbedTable`](https://docs.rs/turbospark-vision-io/latest/turbospark_vision_io/struct.PosEmbedTable.html) | `pos_embed` | Four index and four weight arrays. |
| [`SplicedPrompt`](https://docs.rs/turbospark-vision-io/latest/turbospark_vision_io/struct.SplicedPrompt.html) | `mrope` | Expanded ids and their positions. |
| [`VisionSpecialIds`](https://docs.rs/turbospark-vision-io/latest/turbospark_vision_io/struct.VisionSpecialIds.html) | `mrope` | The `vision_start` and `image_pad` ids the walk keys on. |
| [`VisionIoError`](https://docs.rs/turbospark-vision-io/latest/turbospark_vision_io/enum.VisionIoError.html) | `error` | Every fallible call in the crate returns it. |

Token ids use `turbospark_core::TokenId`, which is a type alias for `i32`. Plain `i32` values work without importing core. Resolve the special ids from a loaded tokenizer rather than copying numbers out of a fixture.

## Decoding

`decode_image_bytes` takes encoded bytes and returns an `Rgb8Image`. `decode_image_file` does the same for a path. `image_dimensions` reads the size from the header without decoding pixels. The decoder reads the header first, so a huge image is refused before its pixel buffer is allocated.

Grayscale input is broadcast to three channels. An alpha channel is dropped, not composited onto a background. Decoders disagree on individual JPEG samples, so output from a JPEG will not match Pillow bit for bit. The repo's parity tests use synthetic pixels for this reason.

## Errors

`VisionIoError` implements `std::error::Error`, so `?` converts it into `Box<dyn Error>`. Each message names the value that caused it.

| Variant | When it happens |
| --- | --- |
| `Decode { detail }` | Bytes are not a supported image, the file cannot be read, or a side exceeds `MAX_IMAGE_DIM` (4,096 pixels). An oversized image reports the decoder's own text. |
| `ImageTooLarge { width, height, max_side }` | Defined for callers, but the decode path in this crate does not return it. Oversized images arrive as `Decode`. |
| `AspectRatioTooExtreme { ratio, limit }` | The long side over the short side is above `MAX_ASPECT_RATIO` (200). The check runs before any rounding. |
| `InvalidDimensions { detail }` | A zero edge, a buffer whose length does not match its dimensions, a resized edge that is not a multiple of `patch_size * merge_size`, or a `min_pixels` budget that rounds an edge to zero. |
| `BadConfig { field, why }` | The config is malformed, a required field is missing, or the pixel budget is inconsistent. |
| `PlaceholderMismatch { placeholders, grids }` | The prompt's placeholder count does not match the number of grids supplied. |

## Platform and limits

- The crate is pure Rust and sets `#![forbid(unsafe_code)]`. Its AGENTS.md names `cargo check --target x86_64-unknown-linux-gnu -p turbospark-vision-io` as a standing check.
- `preprocess` takes one image. Video and multi-frame input are not part of this API. A still image fills a temporal patch by repeating its one frame.
- The crate's tests check the arithmetic and the shapes. They do not prove that your tower's output matches the reference. Real parity needs the pinned reference and the vision gates in [the vision pipeline notes](https://github.com/whit3rabbit/turbospark/blob/main/docs/VISION.md).

## Where to go next

Start with the [crate on docs.rs](https://docs.rs/turbospark-vision-io/latest/turbospark_vision_io/). For the checkpoint facts behind the config readings, read [the vision checkpoint notes](https://github.com/whit3rabbit/turbospark/blob/main/docs/VISION_PHASE0.md). For the end-to-end pipeline, read [the vision pipeline notes](https://github.com/whit3rabbit/turbospark/blob/main/docs/VISION.md).
