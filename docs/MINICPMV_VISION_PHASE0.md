# MiniCPM-V 4.7 vision: Phase 0 and bring-up record

Facts about the MiniCPM-V 4.7 vision tower and how its Qwen 3.6 MoE trunk
consumes image rows, read off the real checkpoint and the reference source,
with the gates that were run against them. The mechanics shared with the other
towers are in [VISION.md](VISION.md); this page is what is different.

## Sources

| What | Where | Revision |
| --- | --- | --- |
| Checkpoint | `mlx-community/MiniCPM-V-4.7-35B-A3B-mxfp4` (catalog `minicpm-v47-35b-mxfp4`, text only) | `810419dd09762d63aad6da5b6a418c4af8946e60` |
| Reference source (read only) | mlx-vlm `models/minicpmv4_6/{vision,minicpmv4_6,config,processing_minicpmv4_6}.py` and `models/minicpmv4_7/{minicpmv4_7,rope,config,processing_minicpmv4_7}.py` | `main` as cloned 2026-11-01 |

The reference source was read and transcribed. It was **not executed** (the
auto-mode classifier denies running code imported from the cloned repository),
so the parity reference below is a numpy transcription
(`scripts/minicpm_v_tower_reference.py`) and no number on this page is a
comparison against mlx-vlm run on the same bytes. See "What is and is not
verified".

## Verdict

**Same trunk family, new tower, combined install only.** The wrapper's
`text_config` is `qwen3_5_moe_text`: 40 layers, hidden 2048, 256 experts top-8
with a shared expert, the existing `QwenGdnMoe` flow, text-installable today
as `minicpm-v47-35b-mxfp4`. Its `vision_tower.` is a **SigLIP encoder with two
merge stages**, not the Qwen tower the same prefix names on Qwen 3.6, so the
repack used to drop it as a "foreign" tower
(`arch_registry::FOREIGN_VISION_WRAPPERS`, now empty). It now has its own kind,
`VisionTowerKind::MiniCpmV` (persisted `minicpm_v`), reader and runtime tower.

**A sidecar is refused by name.** The pairing record can only say
"`qwen35moe`, hidden 2048", which a Qwen 3.6 trunk also says; attaching this
tower to it would run silently and wrongly. `pull-vision` and
`attach_minicpm_images` both refuse; the whole checkpoint installs with
`turbospark-model pull --repo`.

## Inventory

459 vision tensors, all BF16 and **unquantized** (the checkpoint's quantization
block names only language-model gates): 437 under `vision_tower.`
(27 blocks x 16, plus patch embedding weight and bias, the position table, and
`post_layernorm` weight and bias), 16 under `vit_merger.`, 6 under `merger.`.
Shard 1 holds the tower, shard 5 the two adapters. All are converted to FP16 at
repack (the converter refuses a value FP16 cannot hold).

- `patch_embedding.weight` is `[1152, 14, 14, 3]` = `(out, P_h, P_w, C)`, the
  MLX layout, so its flattened rows are in the `(P_h, P_w, C)` order the
  patch rows use and it is copied without a permutation.
- `position_embedding.weight` is `[4900, 1152]`, a 70 x 70 table
  (`image_size` 980 / patch 14).
- Per block: `layer_norm1/2` (weight and bias), `self_attn.{q,k,v,out}_proj`
  (weight and bias), `mlp.fc1/fc2` (weight and bias).
- `vit_merger`: `layer_norm1`, `self_attn.{q,k,v,out}_proj`, `pre_norm`
  (4608), `linear_1` `[17216, 4608]`, `linear_2` `[1152, 17216]`.
- `merger.mlp.0`: `pre_norm` (4608), `linear_1` `[4608, 4608]`, `linear_2`
  `[2048, 4608]`.

## The forward pass (default `16x` mode)

```text
x = patch_embed(rows) + position_embedding[bucket_h * 70 + bucket_w]
blocks 0..=6                       # pre-LN SigLIP, full bidirectional attention
x = vit_merger(x)                  # 2x2 windows, then a quarter of the tokens
blocks 7..=26
x = post_layernorm(x)
out = merger(x)                    # 2x2 groups -> LN(4608) -> 4608 -> gelu -> 2048
```

- **Blocks.** `x += attn(LN1(x))`, `x += fc2(gelu_tanh(fc1(LN2(x))))`;
  LayerNorm eps 1e-6, 16 heads of 72, scale 72^-0.5, no rope, no mask.
- **Positions are BUCKETS, not an interpolation.** Per axis of `n` patches
  `frac = clip(arange(n) / n, 0, 1 - 1e-6)` in f32 and the bucket is the count
  of boundaries `np.arange(1/70, 1.0, 1/70)` (f64, cast to f32) that are
  `<= frac`; the id is `bucket_h * 70 + bucket_w`. A tie such as `2/28 = 5/70`
  counts, so `n = 28` maps index 2 to bucket 5. The port reproduces the f32/f64
  arithmetic and the values are pinned to numpy's
  (`position_buckets_cover_the_table_monotonically`).
- **`vit_merger` runs after block 6** (`insert_layer_id`), on 2x2 windows of
  tokens: `LN1` per token, a 4-token self-attention (q/k/v/out with bias,
  16 heads), a residual add, then `linear_2(gelu(linear_1(pre_norm(window as
  4608))))` plus the MEAN of the four window tokens. Blocks 7 to 26 therefore
  run on a quarter of the tokens.
- **`merger`** groups the post-LN rows 2x2 again, concatenated `(gh, gw, dim)`.
- **Every GELU is the tanh form.** MLX spells it `"precise"` in the mergers and
  `"tanh"` in the encoder and the 4.7 `vit_merger`; mlx's own
  `nn.GELU` source maps both to `gelu_approx`.
- **4x mode** (no `vit_merger`) is not built; the parser refuses a config whose
  `downsample_mode` is not `16x`.

### Token order is a HOST permutation

Blocks 0 to 6 and 7 to 26 contain no position-dependent op after the
embedding, so the tokens are permuted before block 0: 4x4 blocks in raster
order, 2x2 windows in raster order inside a block, raster inside a window. A
`vit_merger` window is then four consecutive rows, the merged tokens of a 4x4
block are four consecutive rows already in `(a, b)` order for the final
merger, and the output rows come out in the page's raster order with no
un-permutation. The numpy reference uses the reference's own
`reshape/transpose` formulation, so it does not share the permutation.

## Preprocessing

`MiniCPMVImageProcessor`, thumbnail only: `find_best_resize(allow_upscale)`
fits the image to about 448 x 448 pixels (`height = int(448 / sqrt(w/h))`,
`width = int(height * w/h)`), each side rounded to a multiple of 56
(`patch * 4`) with Python's half-to-even `round`, PIL BICUBIC (the resize
`crates/vision-io` already reproduces), `(px / 255 - 0.5) / 0.5`, and
**raster** patch rows with `(P_h, P_w, C)` inside (588 values, no temporal
duplication). A 1024x1280 page resizes to 392x504: 28 x 36 patches, 63 tokens.

**Slicing is not built.** The default processor (`max_slice_nums` 9) would add
up to nine detail crops, a `<slice>` grid and newline rows with canvas
positions; this port always takes the reference's `max_slice_nums = 1` arm, so
a large page is one 448-area view. A prompt carrying `<slice>` markers is
refused by name. The consequence for use: dense small text is NOT legible at
about 63 to 64 tokens, so the content gates use large shapes and large type.

## The trunk

- **Template and ids.** The checkpoint's template renders one
  `<|image_pad|>` (248056) per image. The processor expands it to
  `<image_id>` (248090), the ordinal as text tokens, `</image_id>` (248091),
  `<image>` (248078), the merged-token run of the SAME id 248056, and
  `</image>` (248079). The fill and the template marker share an id, so the
  splice counts markers in the rendered prompt only.
- **Canvas mRoPE.** Interleaved sections `[11, 11, 10]`, theta 1e7, partial
  rotary 0.25: the existing qwen3.5 trunk dispatch. Positions come from
  `CanvasRoPE.get_rope_index`, ported for thumbnails
  (`vision_io::canvas_positions`): the `<image>` marker takes
  `(origin, origin - 1, origin - 1)`, the fill `(origin, origin + row,
  origin + col)`, `</image>` `(origin, origin + rows, origin + cols)`, and the
  clock resumes at `origin + max(rows, cols) + 1`. **Marker tokens that follow
  `</image>`, a `\n` (198) or the next `<image>`, are absorbed into the frame
  and take `(origin, origin, origin)`.** That is the reference's behavior and
  the unit tests pin it with a contrasting text token.
- **Embedding.** The fill rows REPLACE the token embedding (the reference's
  `_apply_vision_delta` adds `feature - current`), with no further norm.
- **Chunked prefill.** The qwen MoE chunked driver refuses a live image map,
  so image turns take the sequential prefill; speculation is refused with an
  image as for every vision family.

## What was built

| Layer | Change |
| --- | --- |
| `model-io` | `VisionTowerKind::MiniCpmV` |
| `repack` | `minicpmv4_7` branch of `parse_vision_config` (checks the kind's constants), `vit_merger.`/`merger.` join the vision bucket for the Qwen MoE trunk, `vision_minicpm.rs` reader (16-role blocks, shape-checked residents), the wrapper leaves `FOREIGN_VISION_WRAPPERS` |
| `catalog` | `pull-vision` refuses this tower by name |
| `gpu` | `vision_window_mean4_add_fp16` |
| `vision-io` | `minicpm.rs`: parameters, best-fit resize, raster patching, placeholder expansion, canvas positions |
| `runtime` | `vision::minicpm::MiniCpmVisionTower`; `RealForwardRunner::attach_minicpm_images` (preprocess, splice, encode, inject in one call) |
| front ends | CLI, FFI (generate and in-process server) and the server each branch on `is_minicpm_vision()` and call that one method (server: `VisionInfo::MiniCpm`, `RequestImages::MiniCpm`) |

## Gates

### Tower against the numpy transcription

`vision::minicpm::tests::the_real_tower_matches_the_numpy_reference` (ignored,
real checkpoint) runs the tower on a tower-only fixture built from the real
shards by `examples/minicpm_tower_local.rs`, then compares each stage with
`scripts/minicpm_v_tower_reference.py` on the same pixels. Cosine per stage is
**>= 0.99999341** on both test pages. Two mutations fail the test as intended:
the strided (not own-four-rows) window mean, and a position bucket off by one.
The reference is a numpy transcription of the source, not mlx-vlm run on the
same bytes.

### End to end (real 21.5 GiB combined install, M4 Max, greedy)

Install: `TURBOSPARK_HOME=/Volumes/T7/tshome-minicpm turbospark-model pull
--repo mlx-community/MiniCPM-V-4.7-35B-A3B-mxfp4@810419dd... --alias
minicpm-v47-vision`. `turbospark-check --messages-file ... --image ...
--temperature 0 --top-k 1`, chat template applied:

| Run | Result |
| --- | --- |
| `bigtext.png` ("HELLO" black, "4217" red), "Read the text" | `HELLO` / `4217` (91 prompt tokens, 8 new) |
| `shapes.png`, "List each shape with its color" | `Square - Red`, `Circle - Blue`, `Triangle - Green` |
| same shapes prompt, NO image | "I can't see the image you're referring to." |
| `turbospark-server`, OpenAI `/v1/chat/completions` and Anthropic `/v1/messages`, `shapes.png` | the same three shapes and colors on both wires |
| the same server, `bigtext.png` | `HELLO` / `4217` on both wires |
| `page.png` (dense small text) | **not legible**: a run of `0` characters. Expected: one 448-area view is about 63 tokens |

The content that is only in the image (shape colors, `4217`) is what is
asserted. Not run: speculative decoding with an image (refused by design), the
`<slice>` detail crops (not built), a second image in one turn, 4x mode.
