# InternVL vision (InternViT): Phase 0 and bring-up record

Facts about the InternVL checkpoints, what this port does with their IMAGE
half, and what was gated. One tower covers BOTH wrapper directories: the
`internvl_chat` wrapper (root `model_type`, InternVL2.5 / InternVL3
generation) and the `internvl` wrapper (InternVL3.5 and later). The tower is
the same InternViT; the wrappers differ in trunk and tensor spelling, and the
reader keys on the spelling it sees.

## Sources

| What | Where | Pin |
| --- | --- | --- |
| Checkpoint installed and run | `mlx-community/InternVL3-1B-4bit` | `7927bb6b86987f19398a14eee20a73c45fdb1867` |
| Second build of the same flavor (config + index read) | `mlx-community/InternVL2_5-1B-4bit` | `2f49cb7e2a06e34ae8f1880735cd1fdc1bf21359` |
| Plain-flavor build (config + index read; refuses at a kernel gate) | `mlx-community/InternVL3_5-1B-4bit` | `f9d179a8be8ac53e96c6ee5cce8493856d4b8f09` |
| Reference code read | mlx-vlm `mlx_vlm/models/internvl/{config,vision,internvl,processing_internvl}.py` and `internvl_chat/{config,vision,language,processor,internvl_chat}.py` | commit `e8035ae`, shallow clone |

Every fact below came from the real `config.json`, `tokenizer.json`,
`preprocessor_config.json`, `chat_template.jinja` (where present) and the
published `model.safetensors.index.json` at the pinned revisions. The GELU
semantics (`nn.GELU(approx=...)`) came from mlx's own `nn/layers/activations.py`.

## What the checkpoint is

- Root `model_type` `internvl_chat`, `text_config.model_type` `qwen2` (the
  0.5B trunk: hidden 896, 24 layers, 14 heads over 2 KV, vocab 151674). The
  same pair on InternVL2_5-1B. Root `internvl` with
  `text_config.model_type` `qwen3` on InternVL3_5-1B (1024 wide, 28 layers,
  16 heads over 8 KV). **The trunk decides the family**, through
  `arch_registry::config_json_resolution`'s root-then-text_config lookup:
  there are deliberately NO `internvl` / `internvl_chat` rows in
  `SUPPORTED_HF`, because a wrapper row would resolve a config whose trunk
  the registry does not know (`internlm2` and friends) into Qwen and parse a
  foreign trunk. The wrapper strings live in the vision reader
  (`INTERNVL_MODEL_TYPE` / `INTERNVL_CHAT_MODEL_TYPE`), which refuses any
  other root `model_type` by name.
- `vision_config` (`intern_vit_6b` / `internvl_vision`): 24 blocks, 1024
  wide, 4096 MLP, 16 heads, patch 14, image 448 (the `internvl` wrapper
  spells the size pair `[448, 448]`), LayerNorm eps 1e-6,
  `qkv_bias`/`attention_bias` true, `qk_normalization`/`use_qk_norm` FALSE on
  all three builds. `norm_type` `layer_norm` (an `rms_norm` config is refused
  by name; no kernel here runs it).
- Root `downsample_ratio` 0.5. The pixel-shuffle factor is
  `int(1 / downsample_ratio)` = 2, stored as `VisionConfig.spatial_merge_size`
  and checked at parse (`1 / downsample_ratio` must be a whole number; 0.3
  and 0.75 are refused). The ratio itself is a non-binary float and never a
  manifest field.
- Position table: `(448 / 14)^2 + 1` = 1025 rows, class row included. The
  runtime refuses any other patch count rather than interpolating (the CLIP
  rule); the dynamic tiling always feeds tiles of the tower's own size.
- Span ids: `<IMG_CONTEXT>` is the config's own key
  (`image_token_index` 151667 on the chat builds, `image_token_id` 151671 on
  the plain one; both match the published tokenizers). `<img>` / `</img>` are
  NOT config keys -- they are resolved from the tokenizer by text at attach
  time, and the manifest records them as zero. The two published vocabularies
  spell them at different ids (151665/151666/151667 on Qwen2.5,
  151669/151670/151671 on Qwen3), so hardcoding would be wrong twice.

## Tensor inventory (both flavors, from the indexes)

`internvl_chat` (981 tensors on both 1B builds): tower under
`vision_model.` -- `embeddings.{class_embedding,patch_embedding,position_embedding}`,
`encoder.layers.N.{norm1,norm2,attn.qkv,attn.proj,mlp.fc1,mlp.fc2,ls1,ls2}`
(fused `attn.qkv` already; the tower bodies are UNQUANTIZED FP16/BF16 on
every published build); projector under `mlp1.` (the mlx-vlm module list:
`mlp1.0` LayerNorm, `mlp1.1` Linear, `mlp1.3` Linear -- slot 2 is the GELU),
with the two Linears 4-bit affine; trunk under `language_model.model.*` plus
`language_model.lm_head.*` (already canonical).

`internvl` (1153 tensors on InternVL3_5-1B): tower under `vision_tower.` --
`embeddings.{cls_token,patch_embeddings.projection,position_embeddings}`,
`encoder.layer.N.{layernorm_before,layernorm_after,attention.{q,k,v}_proj,attention.projection_layer,mlp.fc1,mlp.fc2,lambda_1,lambda_2}`
(SEPARATE q/k/v, fused at repack into one `qkv` blob in q, k, v row order);
projector under `multi_modal_projector.` (the LFM3 projector's prefix and
layout), quantized; trunk DOUBLED under `language_model.model.model.*` plus
`language_model.model.lm_head.*`, renamed by `canonicalize_internvl_header`
onto the walk's namespace, exactly as mlx-vlm's `Model.sanitize` does.

**No final norm in either flavor**: neither index carries a
`vision_model.layernorm` / `vision_tower.layernorm` tensor, matching
mlx-vlm's `VisionModel` (the HF original has one; the conversions and the
reference drop it). The runtime runs none.

## What intake does

- `crates/repack/src/gemma4_checkpoint/vision_internvl.rs`: the config
  parser accepts only the two wrapper spellings (refusing foreign and
  absent `model_type` by name), derives the `VisionConfig`, and refuses the
  shapes it does not run (RMSNorm blocks, qk normalization, a downsample
  ratio without a whole shuffle factor, a non-square or ragged size pair).
  The reader buckets every tensor into the fourteen-role block (the qwen
  tower's twelve plus `ls1`/`ls2`), three residents, the position table and
  the six projector tensors; a missing, duplicated or unknown name is
  refused by name, per the shared reader rule. The plain flavor's q/k/v are
  fused in q, k, v row order (pinned byte-for-byte by a synthetic test
  whose tensors carry distinct seeds -- the mutation check below is why).
- `classify.rs`: `vision_tower.` claims for Qwen3Dense (the plain wrapper),
  `vision_model.` + `mlp1.` for Qwen2Dense (the chat wrapper), and
  `multi_modal_projector.` for Qwen3Dense beside the LFM3 arm. Whether the
  bucket is INGESTED is still `vision_should_ingest`'s call, so a text-only
  install drops them as before.
- `catalog/stream.rs`: `parse_vision_for_family` dispatches on the root
  `model_type` (the family resolves through the trunk, which the same file
  names), the bytes-detection check names the two tower prefixes under an
  InternVL wrapper, and an install whose repo ships no usable
  `preprocessor_config.json` (InternVL3_5's processor config names only
  `image_seq_length`) gets a stub carrying the reference processor's own
  defaults instead of a refusal.
- `qwen3_config.rs`: the trunk parser accepts a `text_config` wrapper ONLY
  when the root `model_type` is `internvl` (whose trunk fields are
  plain-qwen3 verbatim); any other wrapper is refused with the same message
  as before. `parse_qwen2_config` already accepted wrappers.

## The forward pass

```text
x   = [cls + pos[0] ; patch_embed(rows) + pos[1..]]   # [1025, 1024]
for blk in 0..24:
    x += attn(ln1(x)).proj * ls1                      # scaled residual add
    x += fc2(tanh_gelu(fc1(ln2(x)))) * ls2
rows = x[1..]                                         # class row dropped
rows = pixel_shuffle(rows, merge 2)                   # host permutation
out  = linear_2(gelu(layer_norm(rows)))               # the projector
```

- The layer scale (`ls1`/`ls2` on the chat flavor, `lambda_1`/`lambda_2` on
  the plain one) is a per-block learned `[hidden]` vector applied by the
  port-local `vision_residual_scaled_add_fp16` kernel (`crates/gpu`), fused
  with the residual add.
- The block MLP's activation is `nn.GELU(approx="precise")` in mlx-vlm,
  which is the TANH form (mlx spells the exact erf one `approx="none"`), for
  BOTH wrappers -- the `internvl` wrapper imports the same MLP class. The
  projector's activation is per flavor: exact erf in `internvl_chat`'s
  `mlp1`, tanh in `internvl`'s `MultiModalProjector`. The published wrappers
  pair flavor to trunk one to one (`internvl_chat` over `qwen2`, `internvl`
  over `qwen3`), so the projector arm keys on the trunk FAMILY as the
  flavor proxy; a future wrapper that breaks the pairing takes the tanh arm
  plus a deviation note rather than a silent wrong activation on a gated
  path.
- Tiles are separate runs: one image's dynamic tiles never attend across
  each other, and the rows concatenate in tile order, which is the order the
  single `<img> ... </img>` span consumes.
- The pixel shuffle is a pure permutation (derived:
  `out[b, h2, w2, mc*C + c] = x[b, h2*f + mc/f, w2*f + mc%f, c]`), done on
  the host over FP16 bits; no arithmetic, no rounding.

## Preprocessing and prompt

`crates/vision-io/src/internvl.rs`: the reference's dynamic tiling
(`dynamic_preprocess`) -- aspect-ratio pair `(cols, rows)` with `cols * rows`
in `[min_dynamic, max_dynamic]` (1..12), ties broken toward the closest
target AREA (`tile^2 * i * j`), the resized image cropped row-major into
exact tiles, and a THUMBNAIL of the original appended when the image split.
Each tile is scaled 1/255 and normalized with the ImageNet constants
(0.485/0.456/0.406 over 0.229/0.224/0.225; the published preprocessor config
states the same pair and the constants are the defaults), patches emitted
`(P_h, P_w, C)` per row -- the layout the MLX conv weight
`[out, kh, kw, C]` flattens to, the transpose of CLIP's and Pixtral's.
Tokens per tile: `(32 / 2)^2` = 256; the whole image sits in ONE span.

The prompt expansion (`internvl_expand_image_placeholders`) replaces each
rendered marker with `[<img>] [<IMG_CONTEXT>] * count [</img>]` and returns
the spans over the context run. The marker is whichever id the checkpoint's
own template rendered: the `internvl` wrapper's template emits bare
`<IMG_CONTEXT>`, the `internvl_chat` one emits `<img>` -- a rendered prompt
carrying NEITHER is refused, one carrying BOTH is refused as expanded.
The published `internvl_chat` template is TEXT-ONLY (a plain Qwen2 template
whose string concatenation cannot render a content-part list), so an image
turn's parts are flattened into `<img>\n` markers before rendering
(`flatten_internvl_parts`, the DeepSeekV2 flatten's shape).

## Gates

**Probe.** `turbospark-model probe mlx-community/InternVL3-1B-4bit`: RUNNABLE,
`declares internvl_chat`, `family qwen2` (resolved through
`text_config.model_type`), MLX affine 4-bit group 64, 24 layers 896 hidden,
2026-11-27. `mlx-community/InternVL3_5-1B-4bit`: `declares internvl`,
`family qwen3`, then REFUSED by the existing quantization gate (4-bit at
group 32; this port's kernels implement group 64) -- a kernel-capacity
refusal, not a wiring one, and the only published `internvl`-wrapper MLX
conversion found (no 8-bit rebuild exists as of 2026-11-27).

**Install.** `turbospark-model pull --repo mlx-community/InternVL3-1B-4bit
--alias internvl3-1b-4bit`: 936.4 MiB install, `packed_vision/` 577 MB (24
blocks), and the manifest declaring the full tower (`visionTower internvit`,
depth 24, 1024 wide, 1025 positions, merge 2, outHidden 896,
`visionImageTokenId 151667`). The three reader refusals hit during bring-up
(a wrong projector tail table, the missing `encoder.` container component, a
shape squeeze that kept one leading batch row) were each fixed and the walk
re-run; the names above are what the final install carries.

**Text smoke.** `turbospark-check --model internvl3-1b-4bit --messages-file`
with "What is 17 plus 26?": answers `43`, EndOfTurn, prefill 45 tokens
(M-series, greedy, 2026-11-27).

**Image smoke, re-run on the fixed projector** (2026-11-27, same revision,
release CLI, greedy). A large-text page (896x1120, Helvetica 96 pt:
`MARIGOLD` / `HARBOR 4821` / `GATE PAGE` / `magic word:` / `zephyr`) is
transcribed exactly at EndOfTurn, prefill 3369 tokens; the same question with
no image asks for the image (prefill 38). On the dense page below the first
row's leading seven words read back, then the output drifts into a repetition
loop and stops at the token cap. No reference run says whether that is the
0.5B 4-bit trunk or the port. The install was removed afterwards.

**Dense-page smoke (OLD BUILD, kept for the record).** It ran before a projector bug was
fixed: the LayerNorm over the shuffled rows (4096 wide) wrote into a buffer
sized for the trunk width (896), so about 78% of each tile's norm output
landed outside its allocation. The transcript below describes that build,
and its drift cannot be attributed to the model. The fix is covered by a
synthetic projector parity test only; no install run has repeated this smoke.

The deterministic page (`scripts/make_vision_test_page.py
--size 896 1152`, seed 11), prompt "Transcribe the first three lines of the
page exactly":

```text
00030 | prefetch parity decode gradient embedding streaming gradient throughput | 93374
00048 | decode tensor quantize gradient throughput streaming fixture | 420076
00066 | checkpoint manifest throughput prefix tokenizer parity prefix torch tokenizer parity prefetch tensor embedding quant
```

Prefill 3371 tokens (13 tower runs x 256 + text), 7.0 s. The page's own rows
(viewed independently) are `00012 | parity streaming streaming oracle ...`,
`00030 | prefetch parity decode gradient embedding streaming gradient
through...`, `00048 | decode tensor quantize gradient throughput streaming
attention ... | 6200.76`, `00066 | checkpoint manifest throughput prefill
streaming footprint tokenizer parity prefill ...`. Three consecutive row IDs
in page order with their leading body words read back; the drift past the
first words and the skipped first row are readback-quality limits of a 0.5B
4-bit trunk, the same class the LFM3 gates record -- a model that never saw
the picture cannot produce page rows at all.

**Negative control.** The same prompt with no image: prefill 40 tokens (no
tower run), output rambles about "pages" with no row IDs and no body words.

**Mutation check.** The plain flavor's qkv fusion was reversed (k, v, q)
with the reader tests rebuilt: `the_plain_flavors_separate_projections_fuse_
in_qkv_order` FAILED only after the fixture's same-shape siblings were given
distinct seeds (the first run SURVIVED the mutation because q/k/v were
byte-identical -- Gotcha 51's self-relative trap, caught and fixed).
Restored, all tests green.

## Not done, and why

- **No tower-stage parity probe against an independent implementation.**
  The gate is the end-to-end page read plus the negative control; no numpy
  reference captured the tower's intermediates. The tower is structurally
  the CLIP one (cls, learned positions, biased attention, LayerNorm blocks)
  with a layer scale and a different activation; its residuals are the two
  places the reference's exact arithmetic is most likely to differ (the
  host-composed class+position row 0, and the erf-vs-tanh GELU choices,
  both pinned by comments and tests rather than by a numeric oracle).
- **No `internvl`-wrapper real install.** The only published MLX
  conversion of the plain flavor quantizes at group 32, which the kernels
  refuse before any of this port's wiring runs; its config path (registry
  resolution, wrapper-aware trunk parse, plain-flavor reader) is covered by
  the synthetic tests and the probe, and its install path is UNVERIFIED.
- **No position interpolation.** The reference interpolates the position
  table bicubically for grids other than the training one; this port
  refuses such inputs. The dynamic tiling always feeds the tower's own
  size, so nothing published reaches the refusal.
- **No quality gate or memory oracle, and no server or app support.** The
  smokes ran through the CLI only. `crates/ffi/src/vision.rs::attach_internvl`
  exists but has never run, and the FFI path is incomplete: it does not
  flatten content parts for the text-only `internvl_chat` template, and its
  session vision report reads `preprocessor_config.json` with the Qwen
  reader, which requires `patch_size` keys. The server has no InternViT arm.
- **No video.** The wrappers' processors have video paths; this port's
  InternViT path is stills only.
- **`mini`/`blob` variants, 6B towers** (`intern_vit_6b` is the model_type
  STRING the configs carry, not a size gate): anything whose config keys
  disagree with the published trio's values is refused at parse by the
  checks above, not silently run.
