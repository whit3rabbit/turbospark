# Muse-Glimmer vision: Phase 0 and bring-up record

Facts about the Muse-Glimmer vision tower and how its trunk consumes image
rows, read off the real checkpoint and the reference source, with the gates
that were run against them. The mechanics shared with the other towers are in
[VISION.md](VISION.md) and [QWEN25_VL_VISION_PHASE0.md](QWEN25_VL_VISION_PHASE0.md);
this page is what is different.

## Sources

| What | Where | Revision |
| --- | --- | --- |
| Checkpoint | `mlx-community/Muse-Glimmer-30B-4bit` (catalog `museglimmer`) | `3e7677d7a40d348a3daba263a2b1c0aa41910710` |
| Reference source (read only) | mlx-vlm `models/muse_glimmer/{vision,muse_glimmer,config,processing_muse_glimmer}.py` | `main` as cloned 2026-11-01 |

The reference source was read and transcribed. It was **not executed**: the
parity reference is a numpy transcription
(`scripts/muse_glimmer_tower_reference.py`) and no number on this page is a
comparison against mlx-vlm run on the same bytes. See "What is and is not
verified".

## Verdict

**Same family, new tower, small trunk seam.** The text trunk was already
supported (the seventh family); until now its 815 vision tensors were dropped
at repack. The tower is a `VisionTowerKind::MuseGlimmer` (persisted
`muse_glimmer`). The trunk needs no mask or position change: image rows replace
the embedding at their positions and positions stay plain and sequential, so
the injection is one blit in each of the two embedding sites.

Compared with the verified towers:

| | Qwen3-VL tower | Qwen2.5-VL tower | Muse-Glimmer tower |
| --- | --- | --- | --- |
| Norm | LayerNorm eps 1e-6 | RMSNorm | LayerNorm eps **1e-5**, plus `ln_pre` and `ln_post` |
| qkv | fused in the checkpoint | fused | **three** `Linear`s with bias (fused at ingest) |
| MLP | tanh GELU | gated SiLU | **exact** erf GELU |
| Position | learned table, align-corners | none | learned 32x32 table, **half-pixel** centres with validity masks |
| Rope | `[h \| w]`, 0-indexed, merge-window order | same | **`[w \| h]`, 1-indexed**, raster order, theta 10000 |
| Attention | whole page | 112 px windows of merged tokens | **32-patch windows** on 38 of 50 blocks |
| Merge | LayerNorm then patch-major concat | RMSNorm then patch-major concat | `ln_post`, then a **dim-major** pixel shuffle |
| Tail | `fc1`, GELU, `fc2` | same | `gelu(fc2(gelu(fc1)))`, projection into the trunk, **no-scale RMSNorm** |
| Patch row | `(T, P, P, C)` merge-window order | same | **`(T, C, P, P)` raster order** |

## Inventory

815 tensors on the 4-bit checkpoint. Bases (excluding the packed planes'
`.scales` and `.biases`): 806 under `vision_tower.`, 2 under `vision_adapter.`
and 1 under `vision_projection.`.

- `vision_tower.layers.N.*` (50 blocks, 16 tensors each): `norm1`/`norm2`
  `{weight,bias}`, `attn.{q,k,v}_proj.{weight,bias}`, `attn.proj.{weight,bias}`,
  `mlp.fc1.{weight,bias}` `[8960, 1536]`, `mlp.fc2.{weight,bias}`. **BF16,
  unquantized.**
- `vision_tower.patch_embedder.patch_embedding.weight` `[1536, 1176]` (bias-free
  `Linear`, `1176 = 2 * 3 * 14 * 14`) and
  `patch_embedder.position_embedding_table.weight` `[1024, 1536]`.
- `vision_tower.ln_pre.*` and `ln_post.*` `[1536]`.
- `vision_adapter.fc1.weight` `[4096, 768]` U32 (= 6144 columns, 4-bit group 64),
  `fc2.weight` `[4096, 512]` U32, and `vision_projection.weight` `[6656, 512]`
  U32. **4-bit affine** with BF16 `.scales`/`.biases`; dequantized to FP16 at
  ingest.

Config: `vision_config` depth 50 (`num_hidden_layers`), hidden 1536, 16 heads
of 96, intermediate 8960, patch 14, `patch_temporal` 2, `merge_size` 2,
`pos_emb_height`/`width` 32. `out_hidden_size` at the ROOT is 6144 (hidden times
merge squared, the adapter's input); the trunk width the projection writes
into is `text_config.hidden_size` 6656, and that is what the port stores as
`out_hidden_size`.

**Constants of the kind, checked at intake and not stored:** the 32-patch
window (equal to `pos_emb_height`), the 32x32 table, LayerNorm eps 1e-5, rope
theta 10000, exact GELU, 14 x 2 x 2, and the block-by-block full/window
pattern, which the config lists and the parser compares against the rule
"every fourth block, and the last" (indices 3, 7, ..., 47 and 49).

## Token layout

`config.json` names `image_token_id` 200092 (`<|patch|>`) and `video_token_id`
200091 only. The chat template renders a **bare** `<|patch|>` per image part.
The processor then replaces each with `<|image_start|>` + N x `<|patch|>` +
`<|image_end|>`, ids **200080** and **200081** in the pinned `tokenizer.json`
(constants of the kind in `repack::qwen36_config`, because the config does not
carry them). The front ends therefore wrap the rendered ids
(`wrap_bare_image_placeholders`, run before `splice_and_walk`) and the
sidecar-attach marker check has a bare-placeholder sibling
(`verify_bare_image_placeholder`). N is `grid.h * grid.w / 4`.

The model reasons before answering (`<|start|>assistant to=self`), so a gate
needs a generous `--max-new` and reads the final channel.

## The forward pass

```text
x     = patch_embed(rows) + pos_interp(grid)        # raster order
x     = ln_pre(x)
order = window order of the PATCHES                 # 32x32 windows, ragged edges
x, rope = permute both by `order`
for blk in 0..50:
    x += attn(ln1(x))      # qkv with bias, q and k rotated, v not; window
                           # segments except blocks 3, 7, ..., 47 and 49
    x += fc2(gelu_erf(fc1(ln2(x))))
x     = ln_post(x[inverse of order])
y     = pixel_shuffle(x)                            # dim-major, merged raster
y     = gelu(fc2(gelu(fc1(y))))                     # 6144 -> 4096 -> 4096, no bias
y     = proj(y)                                     # 4096 -> 6656, no bias
out   = y * rsqrt(mean(y^2) + 1e-5)                 # no weight
```

- **Position interpolation** is the reference's `_position_embeddings`: per axis
  the sample centre is `(i + 0.5) * (32 / n) - 0.5`, the two taps are its floor
  and floor plus one, a tap off the table has weight zero (and is clamped for
  the gather), and the four corner weights multiply. This is NOT the
  align-corners rule the Qwen3 tower uses: at the bottom-right of a 4x4 grid
  over a 2x2 table the half-pixel rule yields 0.5625 of the edge value where an
  edge-replicating rule yields all of it
  (`position_rows_use_half_pixel_centres_and_zero_the_off_table_tap`).
- **Rope.** `freqs = cat(freq_w, freq_h, freq_w, freq_h)` over positions
  `(col + 1, row + 1)`; with the half-split pairing that is the existing 2-D
  rope kernel fed rows of `[w angles | h angles]` (24 + 24 values at head_dim 96).
- **Windows** are built over raster PATCHES with `pad = (-n) % 32`, so no empty
  window is added when an edge divides evenly. The port reuses
  `qwen25::window_layout` with a merge of 1 (units are patches). A grid inside
  one window is the identity and degenerates to full attention.
- **The pixel shuffle** is the reference's
  `reshape(t, h/m, m, w/m, m, d).transpose(0, 1, 3, 5, 2, 4)`: a merged row is
  `[d0 of its 4 patches, d1 of its 4 patches, ...]`. Qwen's mergers
  concatenate whole patch rows instead. It is a host step after `ln_post`
  (a gather no kernel here does).
- The `perception_emb_norm` has no weight, so it is in no checkpoint; it is
  `encode_vision_rms_norm` with no weight at eps 1e-5.

## Preprocessing

`MuseGlimmerImageProcessor`, which shares only its normalize step with Qwen's:

- **Size.** `smart_resize` takes a cap on MERGED TOKENS (`max_image_tokens`
  4096) in units of `patch * merge = 28` px, not a pixel budget. If the ideal
  grid is over the cap it is scaled down keeping the aspect; then of the four
  floor/ceil grids with `h, w >= 1` and `h * w <= cap`, the one whose aspect
  is closest to the image's wins. The 1024x1280 test page becomes 36x45 merged
  tokens, 1008x1260 px, 1,620 tokens (a 36/45 aspect equal to the page's).
  On an exact tie the port takes the first candidate in the reference's
  `itertools.product` order where the reference iterates a Python `set`; the
  test page and ordinary photographs have no tie.
- **Resample** is PIL **Lanczos** (support 3, `sinc(x) * sinc(x / 3)`), run
  through the same fixed-point two-pass path as the bicubic filter
  (`resize::PilFilter`).
- **Normalize** 1/255, mean and std 0.5 from `processor_config.json`.
- **Patchify** in raster order. A row is `(T, C, P_h, P_w)`: the still repeated
  across the 2 temporal frames outermost, then the channel, then the patch's
  pixels. Because the patch embedding is a plain `Linear`, the weight is copied
  verbatim and the rows follow whatever order the processor emits.
- The config is `processor_config.json` with the processor nested under
  `image_processor`; install writes the nested object out as the flat
  `preprocessor_config.json` every reader looks for. Its `max_image_tokens`
  selects `ResizeRule::MuseTokens`, and the derived pixel pair
  (`max_pixels = cap * 784`) lets the session's memory-budget clamp lower the
  cap.

## The trunk seam

Two sites embed a token in this flow: the per-token produce path and the
chunked prefill driver. In both, an image-pad position host-writes the tower's
row INSTEAD of the lookup AND of the following no-scale `embed_norm`: the
reference scatters image features into the already-normed text embeddings, and
the projection's own RMSNorm is the image row's norm. The host write lands
before the pass commits, which makes it visible to every dispatch after it.
Positions are the plain token index, so the injection map carries mRoPE
triples the flow ignores (the walk needs a `start`-then-`pad` pair, which the
wrap supplies).

## What was built

| Layer | Change |
| --- | --- |
| `vision-io` | `PilFilter::Lanczos`; `ResizeRule` on `PreprocessParams`; `muse` (token-cap resize, raster `(T,C,P,P)` patchify); `wrap_bare_image_placeholders` |
| `model-io` | `VisionTowerKind::MuseGlimmer` |
| `repack` | Muse's three prefixes bucketed as the tower; `vision_muse` reader (q/k/v fusion, BF16 to FP16, adapter and projection dequantized); `parse_vision_config` for `muse_glimmer` with the kind's constants |
| `catalog` | Muse admitted to combined-vision and `pull-vision`; sidecar fetch of the three prefixes; `processor_config.json` flattened into `preprocessor_config.json` |
| `runtime` | `vision::muse::MuseVisionTower`, routed from `encode_image`; the two embedding seams |
| front ends | CLI, FFI and server wrap the rendered ids; `verify_bare_image_placeholder` at sidecar attach |

## Gates

### Tower against the numpy transcription (real sidecar, 3.6 GiB)

`the_real_tower_matches_the_numpy_reference` (ignored; setup in its doc), patch
rows from `crates/vision-io`, cosine per stage:

| Page | Patches | Windows | patch embed | block 0 | block 49 | final rows |
| --- | --- | --- | --- | --- | --- | --- |
| 560x700 crop of the test page | 2,000 | 4, ragged | 0.99999994 | 0.99999987 | 0.99999579 | 0.99999879 |
| 448x448 page | 1,024 | 1 (identity) | 0.99999994 | 0.99999987 | 0.99966633 | 0.99998394 |

**Two bars.** The residual-stream stages are dominated by a few outlier
channels (block 49's peak is about 6.2k against an rms of about 108, eight
channels over 1,000, against FP16's 65.5k), so their raw cosine is sensitive to
FP16 rounding of those channels: 0.9997 at block 49 on the 448 page, whose
normalized output is 0.99998. The test holds the residual stages to 0.999 and
the final rows, which are what the trunk reads, to 0.9999. The 448 page is the
weaker reading and is recorded rather than averaged away.

The reference reads the checkpoint with q, k and v as three projections, so the
fused-qkv ingest is checked independently of the port's own fusion.

**Mutation checks** (on the 560x700 crop). Forcing every block to full
attention: block 0 0.99962, block 49 **0.97183**, final rows 0.98602, and the
test fails. A patch-major pixel shuffle in place of the dim-major one: every
block stage unchanged and the final rows **0.24890**, which localizes it to the
merge.

The 448x448 page fits one 32-patch window, so it cannot test the window logic;
the 560x700 crop (a 50x40 patch grid) exists for that.

### End to end (combined install, 18.2 GiB, on an external drive)

`turbospark-check --messages-file ... --image ... --temperature 0 --top-k 1
--max-context 8192`. The model reasons first (`to=self`), so the output holds
its scratch text and then the answer.

- **The 1024x1280 text page** (resized to 1008x1260, 1,620 image tokens, 1,691
  prompt tokens, prefill 87 s, decode 16 tok/s), asked for its first three
  lines. The output begins
  `00012 | parity streaming streaming oracle fixture tensor kernel oracle manifest gradient kernel expert | 8316.48`
  then `00030 | prefill parity decode gradient embedding streaming gradient kernel gradient throughput | 9656.18`
  and `00048 | decode tensor quantize gradient throughput streaming attention streaming fixture | 4200.76`.
  The first line is the page's own (the figure is the one
  [VISION.md](VISION.md) records for this page); none of these appears in the
  prompt.
- **No-image control**, same prompt: the model's scratch text says it sees no
  page or image and considers asking for one.
- **A shapes image** (a red square, a blue circle, a green triangle):
  "Red square ... in the upper left; Blue circle ... in the upper right; Green
  triangle ... oriented point-up, in the lower center". Positions included.
- **The server**, `POST /v1/chat/completions` (an `image_url` data URL) and
  `POST /v1/messages` (a base64 `image` block), same shapes image, greedy:
  both return the same correct description. The Anthropic response carries a
  `thinking` block and a `text` block.
- **The tokenizer.** Against the loaded tokenizer, `<|image_start|>` resolves
  to 200080, `<|patch|>` to 200092 and `<|image_end|>` to 200081, the template
  passes the bare-placeholder check, and the Qwen-style marker check refuses it
  (`crates/tokenizer/tests/vision_markers_muse.rs`, ignored).

## What is and is not verified

Verified on real bytes: the tower's four stages on three page shapes (above),
ingest of the whole tower both as a 3.6 GiB sidecar and inside the combined
install (manifest `visionTower: muse_glimmer`, depth 50, out 6656), the CLI and
both server endpoints, the two mutations, the kind's token ids.

**Not verified:**

- **Against mlx-vlm executed on the same bytes.** The reference here is a numpy
  transcription of its source, so a misreading shared by the transcription and
  the Rust tower would pass. Running mlx-vlm was not authorized in this
  session.
- **Preprocessing against the HF-side processor bit for bit.** The Lanczos
  filter is checked against a hand-computed downscale and the resize rule
  against the reference's search by construction, not against a PIL-produced
  fixture.
- **A text-only trunk plus `--vision-sidecar`.** The sidecar's tower is
  parity-checked and its marker check is checked against the real tokenizer,
  but the attach path itself was not run on a second, text-only install.
- **`ts_generate` and the Swift app**, which share the CLI's and server's
  functions but were not driven.
- **Logit parity of the trunk with an image**, a text-quality or memory oracle
  for the row, chunked image prefill beyond what the 1,691-token run exercised,
  speculation with an image, and video.
- **The sliding-window trunk past 2,048 tokens of image.** A 4,096-token image
  cap makes the prompt larger than the trunk's 2,048-token sliding window; the
  gates here stayed under it.
