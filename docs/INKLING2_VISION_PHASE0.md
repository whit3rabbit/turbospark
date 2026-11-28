# Inkling 2 Mini vision: Phase 0 and bring-up record

Facts about the Inkling 2 Mini vision tower and how its trunk consumes image
rows, read off the real checkpoint header and the reference source. The
mechanics shared with the other towers are in [VISION.md](VISION.md); the text
trunk is [INKLING2_PHASE0.md](INKLING2_PHASE0.md). This page is what is
different.

## Sources

| What | Where | Revision |
| --- | --- | --- |
| Checkpoint | `thinkingmachines/Inkling-2-Mini` (catalog `inkling2-mini`) | `ed40e7da625f594fc9c4990f5f66be24873c22d7` |
| Reference source (read only) | transformers `models/inkling2/{modeling,configuration,image_processing,processing}_inkling2.py`, `vision_utils.py` | `b62646bbda836d467060b7e3d64028202079b77d` |
| Second reading (read only) | mlx-vlm `models/inkling2/{vision,inkling2,config,processing_inkling2}.py` (Neywa-Labs fork) | `main` as read 2026-11-27 |
| Cross-engine reference | llama.cpp `tools/mtmd/models/inkling2v.cpp`, `mtmd-image.cpp`, `mtmd-model.cpp`; `ggml-org/Inkling-2-Mini-GGUF` `mmproj-Inkling-2-Mini-BF16.gguf` | support since `5c008e0008`; read at `4c734444` |

The reference source was read and transcribed. It was **not executed**: the
parity reference is a numpy transcription
(`scripts/inkling2_vision_reference.py`). The llama.cpp build is a separate
engine run on its own converted mmproj, so agreement with it is a
cross-engine check, not a same-bytes comparison.

HF `main` was `c9d02830` on 2026-11-27, three commits past the pin; its
`config.json` is byte-identical, only the README and eval results changed.

## Verdict

**Same family, new tower, small trunk seam.** The text trunk is supported on
four verified installs; until now every intake dropped the 422
`model.visual.*` tensors. The tower is a `VisionTowerKind::Inkling2`
(persisted `inkling2`). The trunk needs no mask or position change: image rows
replace the embedding at their positions, AFTER `embed_norm` (the norm is a
text-only op), and positions stay plain and sequential. The trunk mask is the
ordinary causal (and sliding-window) mask: `Inkling2TextModel` builds it with
`create_causal_mask` / `create_sliding_window_causal_mask` from the standard
kwargs, with no `token_type_ids` or image-span bidirectionality.

The tower ships as a vision sidecar range-read from the HF original, so one
636 MB download serves every Inkling install (HF, MLX4, LAMQ54, GGUF): pairing
is family plus hidden 2304.

| | Muse-Glimmer tower | Inkling 2 tower |
| --- | --- | --- |
| Norm | LayerNorm eps 1e-5, `ln_pre` and `ln_post` | LayerNorm eps 1e-5, `ln_pre` only, **no post norm** |
| Tokens | patches | **CLS** (`class_embedding + cls_position_embedding`) then patches, per image |
| qkv | three `Linear`s with bias | same (fused at ingest) |
| Layer scale | none | **`ls_1.gamma`, `ls_2.gamma`** on both residual branches |
| MLP | erf GELU | erf GELU, 1024 -> 4096 -> 1024 |
| Position | learned 32x32, half-pixel, zero-weight taps | learned 32x32, half-pixel, **border-clamped** taps (`F.interpolate` semantics) |
| Rope | `[w \| h]`, 1-indexed, half-split pairs | `[w \| h]`, 1-indexed, CLS at `(0, 0)`, **interleaved pairs** (GPT-J) |
| Attention | 32-patch windows on most blocks | **whole image** (CLS included), every block |
| Order | raster, then dim-major pixel shuffle | **merge-block order** throughout; the merger is a plain concat |
| Tail | adapter MLP, projection, RMSNorm | `proj_in` 4096 -> GELU -> `proj_out` 2304, **no biases, no norm** |
| Resize | Lanczos to a token budget | **none** (pad to 40 px); we downscale only above 4,096 tokens |

## Inventory

Single shard `model-00001-of-00001.safetensors` (header 148,408 bytes), 1,244
tensors: 822 `model.llm.*` and 422 `model.visual.*`, all vision tensors BF16.
318,256,128 vision parameters (292,041,728 backbone, 26,214,400 projector).

- `model.visual.backbone.embeddings.patch_embedding.weight` `[1024, 1200]`
  (bias-free `Linear`; columns are one 20x20 patch flattened `(row, col, C)`).
- `...embeddings.position_embedding` `[1024, 1024]` (32x32 grid x hidden; no
  `.weight` suffix in this spelling).
- `...embeddings.class_embedding`, `...cls_position_embedding` `[1024]`.
- `...backbone.ln_pre.{weight,bias}` `[1024]`.
- `...layers.{0..22}`: `ln_1`, `ln_2` `{weight,bias}` `[1024]`;
  `ls_1.gamma`, `ls_2.gamma` `[1024]`;
  `self_attn.{q,k,v,out}_proj.{weight [1024,1024], bias [1024]}`;
  `mlp.c_fc.{weight [4096,1024], bias [4096]}`;
  `mlp.c_proj.{weight [1024,4096], bias [1024]}`.
- `model.visual.proj_in.weight` `[4096, 4096]`,
  `model.visual.proj_out.weight` `[2304, 4096]`, no biases.

Other spellings of the same tensors, all accepted at ingest:

| Source | Prefix | Renames |
| --- | --- | --- |
| HF original | `model.visual.backbone.`, `model.visual.proj_{in,out}` | as above |
| transformers / mlx-community (`Inkling-2-Mini-4bit`, BF16, unquantized) | `model.vision_tower.` | `pre_layernorm`, `layernorm_{before,after}`, `lambda_{1,2}`, `mlp.fc{1,2}`, `merger.linear_{1,2}`, `position_embedding.weight` |
| mlx-vlm sanitized | `vision_tower.` | `pre_layernorm`, `layernorm_{before,after}`, `lambda_{1,2}`, `mlp.fc{1,2}`, `proj_{in,out}` |

`config.json -> vision_config` (pinned): `model_type inkling2_vision`,
`hidden_size 1024`, `num_hidden_layers 23`, `num_attention_heads 16`,
`mlp_ratio 4.0`, `native_patch_size 20`, `adapter_merge_size 2`,
`temporal_patch_size 2` (unused for still images: the patch embedding has no
temporal axis), `posemb_grid_size 32`, `projection_hidden_dim 4096`,
`projector_bias false`, `layer_norm_eps 1e-05`, `text_hidden_size 2304`.
From the transformers defaults (not in the file): `hidden_act gelu` (erf),
rope type `axial` with theta 10000, `image_token_id 200054`.

## Token layout

The template renders each image part as
`<|message_user|><|content_image|><|unused_200054|><|end_message|>`
(ids 200000, 200005, 200054, 200010); images are allowed only in user and
tool messages. The processor expands the single `<|unused_200054|>` to
`rows * cols` copies. There are no row separators and no image end token
beyond the closing `<|end_message|>`.

## The forward pass

Per image of `R x C` tiles (each tile is one trunk token), patch grid
`2R x 2C`, sequence `1 + 4RC`:

1. Patch embed: each 40x40 tile is 2x2 patches of 20x20, each flattened
   `(row, col, C)` into 1200 values, through the bias-free `Linear`. The patch
   sequence is in merge-block order: block row, block col, row in block, col
   in block.
2. Position: the 32x32 table is resampled to the `2R x 2C` patch grid,
   bilinear, `align_corners=False`, border padding, no antialias:
   `src = (i + 0.5) * 32 / n - 0.5`, taps `floor` and `floor + 1` clamped to
   `[0, 31]`, weights `1 - |src - tap|`. Added to the patch embedding.
3. CLS row `class_embedding + cls_position_embedding` in front of each image.
4. `ln_pre` (LayerNorm eps 1e-5) over all rows, CLS included.
5. 23 blocks: `x = x + ls_1 * out_proj(Attn(ln_1(x)))`;
   `x = x + ls_2 * c_proj(gelu_erf(c_fc(ln_2(x))))`. Attention is full
   bidirectional within one image, scale `64^-0.5`, rope applied in fp32.
6. Rope: positions `(h + 1, w + 1)` per patch, `(0, 0)` for CLS.
   `inv = 1 / 10000^(arange(0, 32, 2) / 32)` (16 values);
   angles `cat(w * inv, h * inv)` (32), each repeated for one interleaved
   pair, so channel pair `(2i, 2i + 1)` rotates by angle `i`:
   `(x1, x2) -> (x1 cos - x2 sin, x2 cos + x1 sin)`.
7. Merger: drop CLS; consecutive groups of 4 rows (the 2x2 block, in order
   (0,0), (0,1), (1,0), (1,1)) concatenate to 4096, then `proj_in`, erf GELU,
   `proj_out` to 2304. One row per tile, tiles in raster order.
8. Trunk: `inputs_embeds.masked_scatter(input_ids == 200054, rows)`, where
   `inputs_embeds` already went through `embed_norm`. Image rows never pass
   that norm.

## Preprocessing

No resize in the reference: `size {40, 40}` is one tile, not a target.

- PIL images with transparency are composited onto an 8 px checkerboard
  (204 where `(row // 8 + col // 8)` is even, else 153):
  `(rgb * a + bg * (255 - a) + 127) // 255`. Others convert to RGB.
- Pad bottom and right to a multiple of 40 with -1 on the 0-255 scale.
- `x / 255`, then `(x - 0.5) / 0.5`; the padding becomes -1.00784.
- `pixel_values` `[R * C, 40, 40, 3]`, tiles in raster order;
  `image_grid_thw [1, R, C]`.

**Cap (our policy, not the reference's).** An image whose padded size exceeds
4,096 tiles is downscaled with its aspect ratio kept so the padded size fits,
then the session's `LoadGuard` budget (`crates/runtime/src/vision/budget.rs`)
may clamp further. llama.cpp uses the same 4,096-token cap
(`set_limit_image_tokens(1, 4096)`). Images under the cap are byte-exact to
the reference's preprocessing; over-cap images are compared to llama.cpp only
qualitatively because the resize filters differ.

## The trunk seam

- Decode (`families/inkling2/mod.rs`): at an image position the row goes
  straight into the post-norm hidden state; the embedding lookup and
  `embed_norm` are skipped.
- Multi-row prefill and verify (`families/inkling2/verify.rs`): the lookups
  skip image rows and the rows overwrite the normed buffer after
  `norm_rows`.
- The looped core, the attention-residual blocks, the conv state and the
  relative bias need nothing: they consume the initial hidden state and plain
  positions.
- Speculation (DSpark, DFlash2, LoopCD, prompt-copy drafting) is refused by
  name while an image is in the prompt, and only then: attaching a sidecar
  must not disable it for text turns.

## What was built

| Layer | Where |
| --- | --- |
| Preprocessing | `crates/vision-io/src/inkling2.rs`: checkerboard composite, cap, pad, normalize, merge-block patch rows, placeholder expansion |
| Tower kind | `VisionTowerKind::Inkling2` (persisted `inkling2`) |
| Config and ingest | `crates/repack/src/gemma4_checkpoint/vision_inkling2.rs`: `parse_inkling2_vision_config`, `read_inkling2_vision_entries` (fuse q/k/v, permute q/k to half-split pairs, fold the layer scale, sum the class row; HF and mlx-vlm spellings) |
| Sidecar fetch | `stream_vision_sidecar` resolves `inkling2_loop` itself (the family is not in the generic HF table); catalog row `inkling2-mini-vision-tower` |
| Tower | `crates/runtime/src/vision/inkling2.rs`, on the Muse block (`muse::encode_block`) with one whole-image segment |
| Trunk seam | `families/inkling2/mod.rs` (decode) and `verify.rs` (multi-row): image rows replace the post-norm state; `inkling2_speculation_blocker` refuses only image turns |
| Front ends | `RealForwardRunner::attach_inkling2_images` (CLI, FFI), and the server's `VisionInfo::Inkling2` / `RequestImages::Inkling2` |
| References | `scripts/inkling2_vision_reference.py` (numpy), `scripts/inkling2_vision_fetch.py` (ranged fetch of the 422 BF16 tensors), `scripts/inkling2_preprocess_oracle.py` |

## Gates

All on an M4 Max, 2026-11-27, sidecar pulled with
`turbospark-model pull-vision --repo thinkingmachines/Inkling-2-Mini@ed40e7da... --alias inkling2-mini-vision-tower`
(607.1 MiB, 23 blocks, stride 25,198,592 bytes).

**Preprocessing against the numpy transcription**
(`crates/vision-io/tests/inkling2_preprocess_parity.rs`): exact f32 on an
83x121 RGB page (padding on both edges), a 45x63 RGBA page and an exact
40x40 tile. Mutants caught: padding value 0, the in-tile patch order swapped.

**Ingest of the real tower** (`crates/repack/tests/inkling2_vision_reader.rs`,
ignored): 7 residents and 23 blocks; `proj_b == ls_1 * out_proj.bias` and the
permuted q/k rows checked against the source bytes. 53 tensors carry 30,644
values in FP16's subnormal range (the folded tensors' product rounding is not
counted as loss).

**Tower against the numpy transcription**
(`vision::inkling2::tests::inkling2_real_tower_matches_the_numpy_reference`,
ignored; both sides preprocess the same raw pixels):

| Page | Tiles | embed | block 0 | block 22 | merger |
| --- | --- | --- | --- | --- | --- |
| 83x121 gradient | 3x4 | 0.99999995 | 0.99999975 | 0.99999908 | 0.99999810 |
| 410x563 shapes | 11x15 | 0.99999994 | 0.99999961 | 0.99999348 | 0.99999450 |
| 701x903 (upsampled table) | 18x23 | 0.99999993 | 0.99999958 | 0.99998708 | 0.99999141 |

Bars: 0.999 per stage, 0.9999 on the merger. Mutants caught: 0-indexed rope
positions (merger 0.9916), the class row kept in the merger (0.7316), Muse's
zero-weight edge taps on the 701x903 page (0.99944). The edge-tap mutant
SURVIVES on pages under 640 px per side, where the patch grid downsamples the
32x32 table and no tap leaves it; only an upsampling page tells them apart.

**End to end** on `inkling2-mini-mlx4` with the sidecar, greedy, a 480x320
PNG (red circle, blue square; 12x8 tiles, 127 prompt tokens):

| Path | Answer |
| --- | --- |
| `turbospark-check`, chunked prefill (128) | "A red circle on the left and a blue square on the right." |
| same, `TURBOSPARK_INKLING2_CHUNKED_PREFILL=0` (per token) | identical, 63 tokens both |
| server `/v1/chat/completions` (data URL) | identical, 127 + 63 tokens |
| server `/v1/messages` (base64 block) | identical |
| llama.cpp `llama-mtmd-cli` `171527522`, ggml-org BF16 mmproj, IQ3_S trunk | identical final text |

A text-only turn is byte-identical with and without the sidecar attached
(200 tokens, reasoning included), and a text turn with the sidecar attached
still runs DSpark (`--speculative 4 --speculative-drafter dflash`).

## What is and is not verified

- Verified by reading: every config value, every shape and name in the
  header, the template, the processor, the forward pass above, the causal
  trunk mask, and llama.cpp's preprocessing and 4,096-token cap.
- Verified by running: the gates above.
- Inferred from shapes only: `temporal_patch_size` is unused for images.
- NOT run: transformers or mlx-vlm on the same bytes (the numpy transcription
  stands in); an embedding-level comparison with llama.cpp (the Homebrew build
  has no `mtmd-debug`, and its trunk differs, so the llama.cpp row is
  behavioral only; its image token count was not observed); trunk logits with
  an image against a reference; the HF, LAMQ and GGUF installs (only
  `inkling2-mini-mlx4` is installed here); `ts_generate` and the macOS app
  (the app attaches vision sidecars since 2026-11-28, but no run of this tower
  through it is recorded); multi-image prompts; an image over the 4,096-tile cap end to
  end; speculation on an image turn (refused by name); a frozen memory or
  quality row.
- Not done: the session `LoadGuard` clamp on the tile cap. The cap is the
  fixed 4,096 tiles, and a prompt that does not fit the context is refused by
  the ordinary context check.
