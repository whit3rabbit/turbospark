# LLaVA-OneVision vision, phase 0 and what shipped

`llava-hf/llava-onevision-qwen2-7b-ov-hf`: a Qwen2-7B-Instruct trunk (the
`qwen2` family, `ModelFamily::Qwen2Dense`) with a SigLIP-so400m tower, a
two-linear projector and an anyres image layout. Images only; the checkpoint's
video path is not implemented.

The reference is mlx-vlm's `models/llava_onevision/` and Hugging Face's
`LlavaOnevisionImageProcessor` and `pack_image_features` (read, never
executed). The numpy file `scripts/llava_onevision_tower_reference.py` is a
transcription of them, not the libraries run on the same bytes. The pixel
pipeline is not compared against any reference run (no PIL here); see
"Verification".

## Route: GGUF trunk plus `pull-vision` tower sidecar

The same route as LLaVA-1.5 (`docs/LLAVA_VISION_PHASE0.md`).

- Trunk: the community GGUF `paazmaya/llava-onevision-qwen2-7b-ov-hf-gguf`
  (sha `978914bbeebf0943745c8136554e126f6ccd1ec2`), file
  `llava-onevision-qwen2-7b-ov-hf_Q4_K_M.gguf` (4.68 GB). It declares `qwen2`,
  28 layers, hidden 3584, vocab 152128, untied head. Q4_K, Q6_K and F32 blocks
  all have kernels, so the probe says RUNNABLE. Tokenizer files and the chat
  template come from `--sidecar-repo llava-hf/llava-onevision-qwen2-7b-ov-hf`.
- Tower: shard 1 of the llava-hf repo (revision
  `0d50680527681998e456c7b78950205bedd8a068`) holds every non-language tensor,
  426 of them, 829,464,640 bytes, all F16. `pull-vision` fetches only those by
  ranged reads (791 MiB installed). Shards 2 to 4 are language model only.

The pairing is `(family, hidden_size)` only, so any 3584-wide `qwen2` trunk
(Qwen2.5-VL-7B's included) pairs with this sidecar by the manifest check, and
`--vision-sidecar auto` would attach it without comparing the tower kind to the
trunk. The revision pin in the sidecar record is the only statement of which
checkpoint it was trained against.

## Tensor inventory (shard 1, all F16)

| name | shape |
| --- | --- |
| `image_newline` | [3584] |
| `multi_modal_projector.linear_1.{weight,bias}` | [3584,1152], [3584] |
| `multi_modal_projector.linear_2.{weight,bias}` | [3584,3584], [3584] |
| `vision_tower.vision_model.embeddings.patch_embedding.{weight,bias}` | [1152,3,14,14], [1152] |
| `...embeddings.position_embedding.weight` | [729,1152] |
| `...encoder.layers.N.{layer_norm1,layer_norm2}.{weight,bias}` | [1152] |
| `...encoder.layers.N.self_attn.{q,k,v,out}_proj.{weight,bias}` | [1152,1152], [1152] |
| `...encoder.layers.N.mlp.fc1.{weight,bias}` | [4304,1152], [4304] |
| `...encoder.layers.N.mlp.fc2.{weight,bias}` | [1152,4304], [1152] |
| `...post_layernorm.{weight,bias}` | [1152], never read |

26 layers (N = 0..25). `scripts/ov_header_probe.py` lists a sharded repo's
non-language tensors from ranged header reads.

Config facts that the reader refuses to differ from
(`parse_llava_onevision_vision_config`): `model_type llava_onevision`, vision
`siglip_vision_model` with `vision_use_head false`, patch 14, image 384,
`vision_feature_layer -1`, `vision_feature_select_strategy full`,
`projector_hidden_act gelu`, `vision_aspect_ratio anyres_max_9`,
`use_image_newline_parameter true`. The vision config omits the activation and
epsilon; the HF defaults (`gelu_pytorch_tanh`, 1e-6) are what the tower runs,
and a file that states anything else is refused.

## Forward pass

```text
x   = patch_embed(rows) + pos            # per tile [729, 1152], conv with bias
for blk in 0..26:                        # hidden_states[-1] is after the LAST block
    x += attn(ln1(x))                    # LayerNorm eps 1e-6, 16 heads of 72, no rope
    x += fc2(gelu_tanh(fc1(ln2(x))))
out = linear_2(gelu_erf(linear_1(x)))    # every row; no post_layernorm, no unfold
```

The blocks are DeepSeek-VL2's SigLIP blocks (`runtime/src/vision/siglip.rs`
`encode_siglip_block`, shared). The head differs: no final norm, no 2x2 unfold,
the first projector layer reads one 1152-wide row. The tower kind is therefore
its own `VisionTowerKind::SigLipOv` (`siglip_ov`): the `SigLip` kind's manifest
meaning (merge 2, final norm, view separator) cannot carry it. The conv is a
valid-padding stride-14 conv over 384 pixels, so each tile reads 27 x 27 patches
and the 6 trailing pixels never reach it.

## Anyres rule (`crates/vision-io/src/onevision.rs`)

One image becomes a BASE tile plus a GRID of tiles:

1. Base: the whole image squashed to 384 x 384 (PIL bicubic).
2. `select_best_resolution(orig_h, orig_w, pinpoints)`: over the 36
   `image_grid_pinpoints` (`[h, w]`, 384 to 2304 in steps of 384), take the one
   with the largest `min(downscaled_area, orig_area)`, ties to the least wasted
   canvas.
3. Resize to fit that canvas with the aspect kept (`_get_patch_output_size`,
   `ceil`, clamped to the target), centre it on a ZERO canvas of raw pixels
   (before normalization, so the pad reads `-mean/std = -1`), floor of the pad
   before and the remainder after.
4. Cut 384 tiles row-major (rows outer). Each tile is 27 x 27 patches, one
   patch row is `(C, 14, 14)`, base tile first in `ClipPatches::pixels`.

Packing (`ov_pack_features`, after the tower):

- the base tile's 729 rows, raster order;
- the grid's rows as one `(tiles_h*27, tiles_w*27)` map (tile `(a,b)` row
  `(i,j)` is map `(a*27+i, b*27+j)`), cropped by `unpad_image` against the
  ORIGINAL size back to the image's aspect;
- if `sqrt(h*w / (9*27*27)) > 1.1`, bilinear-resized (`align_corners=False`, no
  antialias) to `(int(h // ratio), int(w // ratio))`;
- the learned `image_newline` row after every map row, rows flattened row-major.

Tokens per image = 729 + out_h * out_w + out_h, computed by the same plan the
packing uses, so the prompt and the rows cannot disagree. The independent
python model (written from the HF text, not the Rust) gives:

| image (h x w) | grid | map after unpad | tokens |
| --- | --- | --- | --- |
| 300 x 300 | 1 x 1 | 27 x 27 | 1485 |
| 336 x 448 (shapes.png) | 1 x 2 | 27 x 36 | 1728 |
| 448 x 448 (page-small.png) | 2 x 2 | 54 x 54 | 3699 |
| 480 x 640 | 2 x 2 | 40 x 54 | 2929 |
| 1280 x 1024 (page.png) | 4 x 3 | 102 x 81, pooled to 90 x 72 | 7299 |

## Prompt layout

`<image>` is id 151646 (in range of the 152128-row GGUF table, so the fill id
is the marker itself). The template renders a bare `<image>` per image, no start
or end token, so the run is a bare block of `tokens` placeholders
(`clip_expand_image_placeholders`) and the tower's rows replace them. Positions
are the plain token index (Qwen2 has no mRoPE). The Qwen2-7B tokenizer has no
`<tool_call>` pair, which the ChatML resolver used to require; it is optional
now (both or neither).

The front end rides the CLIP path: `ClipLayout::OneVision`, `ClipPatches`
gains `anyres`, and `grid` / `grid_w` carry the token count as `(tokens, 1)`
(the prompt layer only multiplies them). `ClipPreprocessParams::tokens()` is
the worst case, base plus the largest pinpoint grid, 37 tiles.

## Verification

All on this Apple M4 Max, release build, a private store.

- **Tower and packing against the numpy transcription**
  (`runtime/tests/onevision_vision_parity.rs`, ignored, real installs): cosines,
  port vs numpy, same patch rows.

  | image | embed | block 0 | block 25 | packed run |
  | --- | --- | --- | --- | --- |
  | shapes.png (3 tiles, crop path) | 0.99999997 | 0.99999983 | 0.99971851 | 0.99981311 |
  | page.png (13 tiles, pooled path) | 0.99999996 | 0.99999984 | 0.99987069 | 0.99995481 |

  The worst single row of the last block is 0.97 (shapes) and 0.78 (page),
  with a worst packed row of 0.91 and 0.62; SigLIP's late blocks carry a few
  high-norm rows where F16 storage loses the most (the CLIP tower's pattern).
  The whole-tensor cosines are the claim.
- **Mutation check:** the packing's tile order was changed to column-major
  (`(tx * tiles_h + ty)` for `(ty * tiles_w + tx)`; asserted applied) and the
  page.png parity test FAILED, then passed again after restoring the file. The
  1 x 2 shapes grid cannot see that mutation (the two orders coincide), which
  is why page.png is the gate for it. The failing cosine was not recorded.
- **What the numpy path does not cover:** the patch rows come from the port, so
  the resize/pad/tile pixels are pinned only by the unit tests and the python
  token model. A resize that differs from PIL by a rounding step would not be
  seen here.
- **Real runs**, `turbospark-check --messages-file`, greedy, the sidecar
  attached (the numbers include a second Metal job on the host, not a timing):
  - shapes.png, "List every shape you see in this image and its color.":
    `The image contains three shapes: a red square, a blue circle, and a green
    triangle.` (1748 prompt tokens, 1728 of them image).
  - The same question with NO image: `The image contains a circle with a red
    outline and a blue center.` (the control: the correct answer needs the
    image).
  - page-small.png (3699 image tokens, 2 x 2): the model transcribes
    `000012 | parity streaming streaming oracle fixture`; the image's first line
    is `00012 | parity streaming streaming oracle fixture ...`. The words are
    exact and only visible in the image; the line number gained a zero.
  - page.png (7299 image tokens, pooled): the answer starts `parity streaming
    streaming` and then degenerates into repeating `attention streaming`. The
    first words are image-only content, the rest is wrong, so the high-token,
    pooled case works mechanically and is a poor reader.
  - `turbospark-server` on `/v1/chat/completions` (data-URL `image_url`) and
    `/v1/messages` (base64 `image`): both returned the same shapes sentence as
    the CLI, 1748 prompt tokens.

## Not done

- No video input, no multi-image stress, no quality or memory gate, no
  benchmark row.
- FFI (`attach_clip` and the in-process server model) compile and ride the CLIP
  arms; nothing ran them, and the Swift app is untouched.
- No catalog row (the installs are `pull` / `pull-vision` commands).
- The pixel pipeline has no reference run behind it.
- Token counts use Python's `int(round(x, 7))` as a round-then-truncate; a tie
  at the seventh decimal is not handled the way Python's half-even would.
