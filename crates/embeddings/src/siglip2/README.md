# SigLIP2 NaFlex

Dual-tower image and text embedding model with variable-resolution
(NaFlex) image input. Images keep their aspect ratio: each is resized so
its 16x16 patch grid holds at most 256 patches, and the learned 16x16
position table is resized per image to that grid. Image and text
embeddings are 768-wide, L2-normalized, and scored with a sigmoid
(each image/text pair independently), not a softmax.

## Checkpoint and reference

| Item | Value |
|---|---|
| Checkpoint | [`google/siglip2-base-patch16-naflex`](https://huggingface.co/google/siglip2-base-patch16-naflex/tree/b53b807d3a2d5e2b3911292f2d69e5341cdc064c) |
| Revision | `b53b807d3a2d5e2b3911292f2d69e5341cdc064c` (also in `src/pins.rs`) |
| Weights | `model.safetensors`, F32, 375M parameters (197M of them the 256000-row token table) |
| Numeric reference | HF transformers 4.57.1 `Siglip2Model` + `Siglip2Processor`, torch 2.14.1, torchvision 0.29.1 |
| Port guide | mlxrs `src/embeddings/siglip2_naflex/` (structure and gotchas only, numbers come from HF) |
| MLX conversion | None found on the Hub, none pinned |

## Architecture

- Config: the checkpoint's `config.json` sets only `text_config.vocab_size`.
  Everything else is the HF class default (hidden 768, 12 layers, 12
  heads, MLP 3072, `gelu_pytorch_tanh`, LayerNorm eps 1e-6, text length
  64, `num_patches` 256, patch 16). `config.rs` fills the same defaults and
  refuses values it has not verified: another activation,
  `vision_use_head = false`, a non-square position table, non-RGB input,
  `projection_size != vision hidden_size`, a quantized checkpoint, a
  processor other than `Siglip2ImageProcessorFast`, resample other than 2,
  and disabled resize/rescale/normalize.
- Vision: `Linear(768 -> 768)` patch embedding over `(row, col, channel)`
  flattened patches, plus the position table resized to `(h, w)` with
  `F.interpolate(bilinear, align_corners=False, antialias=True)` in f32.
  Then 12 pre-norm encoder layers, `post_layernorm`, and the MAP head (learned probe,
  `nn.MultiheadAttention` with packed `in_proj`, `x + mlp(layernorm(x))`).
  The head output is the image embedding before normalization. There is no
  further projection.
- Text: token + absolute position embedding, 12 pre-norm encoder layers
  with NO attention mask, `final_layer_norm`, then `head` (biased
  `Linear(768 -> 768)`) on the LAST position.
- Output: `x / ||x||` for both towers, `logits_per_image = exp(logit_scale)
  * <img, txt> + logit_bias` (`logit_scale = 4.7475543`,
  `logit_bias = -16.776989`), probability = sigmoid(logit).

Padded image patches are not materialized. The reference gives them
`resized_pos[0]` and masks them as keys in every attention, which makes
their softmax weight exactly zero. The generator asserts batched (padded)
and per-image runs agree (max abs 1.19e-7 on the tiny model).

## Determinations

### Which image processor, which resize

`preprocessor_config.json` names `Siglip2ImageProcessorFast` with
`resample: 2`. In transformers 4.57.1, `AutoProcessor` picks the Fast class
because the saved type ends in `Fast`, but only when torchvision is
installed. Without it, it silently falls back to the slow
`Siglip2ImageProcessor` (PIL). The generator asserts the Fast class.

The Fast processor resizes a uint8 CHW tensor with torchvision
`F.resize(BILINEAR, antialias=True)`. On arm64 (and AVX2 x86) that runs
PyTorch's native uint8 antialiased kernel: f64 triangle weights widened by
the downscale factor, renormalized after clipping, quantized to int16 at
the largest precision that keeps the biggest weight under 2^15, integer
accumulation with a half-unit seed, `>> precision`, clamp, horizontal pass
first, an unchanged axis skipped. `processing::resize_bilinear_aa_u8`
reproduces it bit-exactly on all five real-config fixture images (up- and
down-scaling, including 50x37 -> 288x224 and 1000x120 -> 672x96). This was
first checked against torchvision on 60 random size pairs with 0
mismatched bytes before porting.

The slow (PIL) processor is NOT reproduced and is refused by config. It
differs from the Fast one by up to 0.0157 in pixel values (two 8-bit
levels after normalization). On the real checkpoint that moves image
embeddings by max abs 6.7e-4, min cosine 0.9999852 (measured by the
generator, recorded in `manifest.json`).

Other processor stages, verified exact against the fixtures:

- Target size: port of `get_image_size_for_max_num_patches` including its
  f64 binary search (`eps = 1e-5`), ceil to a patch multiple, at least one
  patch.
- Normalize: fused `(x - mean/r) / (std/r)` in f32 with `r =
  rescale_factor`, i.e. `(x - 127.5) / 127.5`.
- Patchify: patches row-major over the grid, each `(patch_row, patch_col,
  channel)`. `spatial_shapes = (h/16, w/16)`, and `pixel_attention_mask` is 1
  for the first `h*w` of 256 rows.

### Text: what the model receives

- `Siglip2Processor` text defaults are `padding="max_length",
  truncation=True, max_length=64`. Ids are the content, `<eos>` (1), then
  `<pad>` (0) to 64. An overlong text keeps 63 content tokens plus `<eos>`
  (fixture text 3 pins this).
- The tokenizer's `model_input_names` is `["input_ids"]`, so the processor
  returns no `attention_mask` and `Siglip2Model` runs the text encoder
  unmasked: pads attend and are attended to. The generator asserts no mask
  is returned.
- Pooling is `last_hidden_state[:, -1]`, which for any text under 63 tokens
  is a `<pad>` slot. mlxrs calls this "sticky EOS": SigLIP1's tokenizer
  padded with EOS, so the last slot was always EOS. SigLIP2's Gemma
  tokenizer pads with 0, but the model was trained with this layout, so the
  last slot is pooled regardless. Pooling the last real token instead
  drops cosine to 0.601 on fixture text 0 (mutation check below).
- Lowercasing: not applied by the reference. `tokenizer_config.json` has
  `do_lower_case: true`, but `GemmaTokenizerFast` ignores it ("A Photo of a
  Cat" and "a photo of a cat" tokenize differently), the processor does not
  lowercase, and the HF zero-shot pipeline uses the capitalized template
  "This is a photo of {}.". The pinned model card says nothing about case.
  `Siglip2::embed_text` therefore passes text through unchanged. Any casing
  policy belongs to the caller. Whether big_vision lowercased at training
  time is unverified here.

## Public API

```rust
let model = Siglip2::load(dir)?;              // config, processor, weights, tokenizer.json if present
let img = model.embed_image(&rgb8)?;          // or embed_image_file(path), embed_pixels(&NaflexInput)
let txt = model.embed_text("a photo of a cat")?; // or tokenize() + embed_ids()
let l = model.logit(&img, &txt)?;             // exp(scale) * dot + bias
let p = siglip2::sigmoid(l);                  // or model.probabilities(&images, &texts)
```

`vision().forward_trace` and `text().forward_trace` expose every stage. The
F32 token table is read row by row from the mapped safetensors file rather
than copied (786 MB), so a loaded model holds about 0.8 GB of heap.

## Verified stages

Default suite (`cargo test -p turbospark-embeddings --no-default-features
--features siglip2`), no downloads:

| Stage | Fixture | Result |
|---|---|---|
| NaFlex target size | 5 real-config + 3 tiny images | exact |
| uint8 AA bilinear resize | 5 real-config images vs torchvision PNG | 0 differing bytes |
| normalize + patchify | sampled rows + f64 sum (real), full rows (tiny) | exact (0.0) |
| position-grid resize | tiny 4x4 -> 4x4, 6x4, 1x24 | max abs 3.6e-7 |
| vision embeddings / last hidden / pooled | tiny, 3 images | max abs 5.4e-7 / 1.5e-6 / 8.3e-7 |
| text embeddings / last hidden / pooled | tiny, 3 id rows | max abs 0 / 1.2e-6 / 8.3e-7 |
| normalized image / text embeds | tiny | max abs 1.5e-7 / 1.5e-7 |
| logits | tiny 3x3 | max abs 2.4e-7 |
| config defaults and refusals | inline JSON | 9 model + 4 processor refusals |

Tiny-model tolerance is 1e-5 absolute on activations up to 3.8 in
magnitude: the residual is f32 reduction order (sequential sums here,
blocked kernels in torch) plus f32-vs-f64 resize weights, measured at
1.55e-6 at worst, so the bound has about 6x headroom.

Real checkpoint (`#[ignore]`, `TURBOSPARK_SIGLIP2_DIR` or
`~/models/embeddings/siglip2-base-patch16-naflex`, release, Apple M4 Max):

| Output | Max abs diff | Min cosine |
|---|---|---|
| image_embeds, 5 images | 1.13e-6 (small50x37) | 0.99999988 (wide1000x120) |
| text_embeds, 4 texts | 6.85e-7 | 1.00000000 |
| logits_per_image, 5x4 | 6.68e-5 (logits range -18.2 to -1.8) | n/a |
| token ids, 4 texts x 64 | exact | n/a |

Bounds: embeddings 1e-5 abs and cosine > 0.99999, logits 1e-3 (the dot
error is multiplied by `exp(4.7476) = 115.3`).

Mutation checks (each applied, the named test failed, then reverted):

- Position grid resized with `(w, h)` swapped: `tiny_model_vision_parity`
  fails, tall pos_resized max abs 1.907.
- Position grid height pass without antialiasing (plain half-pixel
  bilinear): fails on wide (4 -> 1 rows) pos_resized, max abs 0.509. The
  tall image (4 -> 6, upsampling) still passes, as expected: antialiasing
  only changes downscaling.
- Text pooled at the last non-pad token instead of the last position:
  `tiny_model_text_parity_and_logits` fails (text 0 pooled max abs 3.20)
  and `real_checkpoint_parity` fails (text 0 cosine 0.601).

Timing (release, one thread per test, matmuls threaded, host shared with
other builds at load average 14 to 16, so these are upper bounds): load
0.58 to 2.45 s with a warm page cache. A 224x224 PNG decode + preprocess +
vision tower takes 0.57 to 1.10 s, and one text 0.15 to 0.52 s.

Not verified: quality on natural images (only synthetic patterns),
batched inference (one image per forward), the slow PIL processor, the
`x86_64-unknown-linux-gnu` cross check (blocked locally by the `onig_sys`
C build needing a cross compiler), and JPEG inputs (decoder differences
from Pillow are outside this port).

## Regenerating fixtures

```sh
<venv>/bin/python -I crates/embeddings/tools/gen_siglip2_fixtures.py \
    --model-dir ~/models/embeddings/siglip2-base-patch16-naflex
```

The venv needs transformers 4.57.1, torch 2.14.1, torchvision 0.29.1,
numpy, Pillow, and safetensors (versions recorded in
`testdata/siglip2/manifest.json`). The script is deterministic (seeded,
no network). A rerun reproduced all 49 files byte for byte. Outputs:
`testdata/siglip2/tiny/` (seeded random 2+2-layer model, patch 4, max 24
patches, goldens per stage), `testdata/siglip2/real/` (synthetic source
PNGs, torchvision resize output PNGs, sampled pixel rows, real embeddings,
ids, logits), and `manifest.json` (versions, revision, Fast-vs-slow
divergence, SHA-256 per file). Total 1.2 MB.

Run the real-checkpoint tests:

```sh
CARGO_TARGET_DIR=target/emb-siglip cargo test --release -p turbospark-embeddings \
    --no-default-features --features siglip2 real_ -- --ignored --nocapture
```
