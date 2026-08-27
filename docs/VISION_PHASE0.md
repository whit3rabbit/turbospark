# Vision Phase 0 findings (qwen3_5 vision tower)

Fact-finding for the vision bring-up (see the approved plan at milestone
M-V0). Everything below is read off real checkpoint headers, real config
files, and the vendored `mlx-vlm` reference source at `../mlx-v/mlx-vlm`
(no model weights downloaded, no forward pass run). Items 3 and 4 need an
actual forward pass and are left open pending a download decision.

Checkpoints probed: `prism-ml/Bonsai-27B-mlx-1bit` (1-bit) and
`mlx-community/Qwen3.8-27B-4bit` (INT4). Their `vision_config`,
`chat_template.jinja`, `tokenizer_config.json` and `preprocessor_config.json`
are identical; the two differ only in the trunk's quantization, consistent
with `crates/model-io/CLAUDE.md`'s existing note that both share one
baseline.

## 1. Vision tensor inventory

- Prefix: `vision_tower.` (2,180 total tensors, 333 under this prefix, one
  shard: `model.safetensors`). Confirms the classify.rs prefix list is
  already correct for this family; no new spelling needed.
- 27 blocks (`vision_tower.blocks.0..26`), each: `norm1.{weight,bias}`,
  `attn.qkv.{weight,bias}` [3456,1152]/[3456], `attn.proj.{weight,bias}`
  [1152,1152]/[1152], `norm2.{weight,bias}`, `mlp.linear_fc1.{weight,bias}`
  [4304,1152]/[4304], `mlp.linear_fc2.{weight,bias}` [1152,4304]/[1152].
  **`intermediate_size` is 4304, not 4608** -- the plan's kernel table used
  4608 by analogy with the merger's hidden width; fix at implementation.
- Non-block tensors: `patch_embed.proj.{weight,bias}` [1152,2,16,16,3]/[1152],
  `pos_embed.weight` [2304,1152] (48x48 grid, `sqrt(2304)=48`),
  `merger.norm.{weight,bias}` [1152], `merger.linear_fc1.{weight,bias}`
  [4608,4608]/[4608], `merger.linear_fc2.{weight,bias}` [5120,4608]/[5120].
  **`out_hidden_size` is 5120** (matches the qwen3_5 text trunk's
  `hidden_size`, as expected -- the merger outputs directly into the trunk's
  residual width).
- Dtype: **F16 in the 1-bit checkpoint** (0.858 GiB total), **BF16 in the
  4-bit checkpoint** (same 333 tensors, same shapes). Neither checkpoint
  quantizes its vision tower -- both ship it at full precision. This is the
  input to the M-V0.4 INT4-transcode-quality question: we are the ones
  choosing to quantize it, not continuing an existing quantization.
- `vision_config` (identical in both checkpoints): `depth=27, hidden_size=1152,
  intermediate_size=4304, num_heads=16 (head_dim=72), patch_size=16,
  temporal_patch_size=2, in_channels=3, spatial_merge_size=2,
  num_position_embeddings=2304, out_hidden_size=5120,
  hidden_act="gelu_pytorch_tanh", deepstack_visual_indexes=[]` (confirmed
  empty/disabled, as the plan assumed).

## 2. mRoPE semantics -- RESOLVED, and simpler than the plan assumed

The `[11,11,0]` vs `[11,11,10]` "discrepancy" flagged during planning is not
a discrepancy: `[11,11,0]` is an unused default in
`qwen3_5/language.py`'s `Qwen3_5RotaryEmbedding.__init__` signature. The
actual construction site (`language.py:1466-1470`) passes
`args.rope_parameters["mrope_section"]`, i.e. the checkpoint's real value.
Both checkpoints' `config.json` declare (under `text_config.rope_parameters`):

```
mrope_interleaved: true
mrope_section: [11, 11, 10]   (sums to 32 = rotary_dim/2)
partial_rotary_factor: 0.25
rope_theta: 10000000
rope_type: "default"
```

`text_config`: `hidden_size=5120, num_attention_heads=24,
num_key_value_heads=4, head_dim=256`. `rotary_dim =
int(head_dim * partial_rotary_factor) = 64`, matching the port's existing
`ropeNeoxSubdim`/`partial_rotary_factor=0.25` baseline field exactly
(`crates/model-io/src/arch_config/family.rs:68`, `qwen_gdn_dense_27b()`).

**`crates/model-io/CLAUDE.md`'s `arch_baselines/qwen.rs` entry already
states this is settled**: `rope_type: "default"` makes `mlx_lm` (the
text-only reference) apply plain, non-mrope RoPE, ignoring
`mrope_section` entirely -- confirmed here by reading the mlx-vlm
`get_rope_index` (`qwen3_vl/language.py:282-362`): for a prompt with **no**
`image_grid_thw`/`video_grid_thw`, it returns a single sequential
`position_ids` array (`mx.arange`), never the 3-row mrope form. So a
pure-text prompt was never exercising mrope in the reference either --
nothing to "degenerate" to, it simply never engages.

**The degenerate-equivalence invariant, made precise**: `get_rope_index`
only builds true 3-row `(t,h,w)` position ids for an image's own token span.
Every text span -- including the text *between* two images, and the text
*after* the last image -- gets `t=h=w=<sequential counter>`, where the
counter resets to `max(previous span) + 1` after each image
(`language.py:359-362` in the snippet read). Since `mrope_section`'s
"interleaved" channel selection only matters when `t`, `h`, `w` diverge for
a token, and they are identical for every text token, **the existing
`rope_neox_subdim` kernel is exact for every text token in a mixed prompt,
not merely for text-only prompts**. The new `rope_mrope_interleaved` kernel
from M-V2/M-V5 therefore only needs to fire at image-pad token positions.
This narrows M-V5's dispatch condition (a text-token position never touches
the new kernel) without changing its structure.

Still open: the exact bit layout of "interleaved" section selection
(`_interleaved_position_selector`, `rope_utils.py:352-360`) for computing
which of the three per-token positions feeds which frequency pair -- needed
to write `rope_mrope_interleaved`'s CPU reference in M-V2, but not needed to
prove the degenerate case above (that proof only requires t=h=w, not the
selection formula). Defer full derivation to M-V2 implementation.

## 3. Activation magnitude probe -- RESOLVED, real forward pass

Ran the actual `VisionModel` from `../mlx-v/mlx-vlm/mlx_vlm/models/qwen3_vl/`
(the real reference code, not a reimplementation) against a real 333-tensor
F16 vision tower downloaded from `prism-ml/Bonsai-27B-mlx-1bit` (only the
vision-tower byte range, ~879 MiB, fetched by ranged HTTP -- the 27B text
trunk was not downloaded), on a synthetic 1024x1280 rendered OCR page (smart
resize gave `grid_thw=[1,80,64]`, 5,120 patches, 1,280 merged tokens).

Full per-block trace (`scripts/vision_tower_probe.py --mode activation`),
absmax / rms:

| block | absmax | rms | | block | absmax | rms |
|---|---|---|---|---|---|---|
| 0 | 10.8 | 0.59 | | 14 | 407.8 | 0.89 |
| 1 | 6.6 | 0.38 | | 15 | 410.8 | 0.95 |
| 2 | 14.9 | 0.36 | | 16 | 412.0 | 1.03 |
| 3 | 20.6 | 0.33 | | 17 | 422.0 | 1.15 |
| 4 | 22.8 | 0.33 | | 18 | 426.3 | 1.27 |
| 5 | 22.1 | 0.30 | | 19 | 432.3 | 1.88 |
| 6 | 17.6 | 0.32 | | 20 | 450.0 | 2.46 |
| 7 | 13.9 | 0.35 | | 21 | 458.5 | 2.44 |
| 8 | 12.9 | 0.35 | | 22 | 473.3 | 3.24 |
| **9** | **404.8** | 0.76 | | 23 | 478.0 | 2.88 |
| 10 | 405.8 | 0.77 | | 24 | 487.5 | 2.93 |
| 11 | 406.0 | 0.79 | | 25 | 466.3 | 2.13 |
| 12 | 406.3 | 0.80 | | **26** | **8,384.0** | 110.5 |
| 13 | 406.5 | 0.83 | | merger out | 69.6 | 0.36 |

**Not a smooth climb -- two sharp step jumps with a plateau between.**
Block 8->9 jumps absmax 31x (12.9 -> 404.8), then blocks 9-25 plateau
around 400-490 with rms climbing gradually, then block 25->26 jumps another
18x (466 -> 8,384) right before the merger's own LayerNorm resets the
scale. This is the classic "outlier feature" / attention-sink pattern seen
in ViT and SigLIP towers -- a small number of channels or tokens carry a
disproportionate share of the norm, concentrated at specific layers rather
than growing uniformly with depth. At this page size, **8,384 is ~13% of
FP16's 65,504 ceiling -- real headroom, not a near-miss, but not generous
either**, and the fact that it arrives as a step rather than a slope means
a larger page is not guaranteed to extrapolate linearly (a good reason to
test the 4096x4096 extreme, not skip it because the trend "looks safe").

**FP16-vs-FP32 accumulation comparison** (same weights, same input, one run
computed end-to-end in FP16 and one in FP32): merger output cosine
similarity **0.999993**, max abs diff 1.03, mean abs diff 0.0004 over a
(1280, 5120) output. **FP16 accumulation is numerically safe at this page
size** -- accumulating in FP32 changes almost nothing.

**Not yet tested: the largest legal page.** `preprocessor_config.json`
allows up to 16,777,216 pixels (4096x4096), i.e. up to 16,384 patches --
3.2x this test's 5,120. The block-26 blowup is plausibly closer to a
fixed outlier-feature magnitude than something that scales with sequence
length, but this was not verified at the extreme end. **Action item for
M-V2's own gate**: rerun this probe (script below) at a near-4096x4096 page
before finalizing FP16 as the vision-tower compute dtype; if the extreme
case pushes past FP16's ceiling, the fix is scoping the final block(s)'
accumulator to FP32 rather than widening the whole tower.

## 4. INT4 transcode quality -- RESOLVED (against the plan's default)

Fake-quantized every 2D matmul weight in the tower (qkv, proj, fc1 --
84 tensors total; excludes norms, biases, `fc2`, and `patch_embed`, see
below) to MLX's own affine INT4 at group_size=64 via `mx.quantize` /
`mx.dequantize` (the same scheme this port already uses for its resident
INT4 weights), then re-ran the identical forward pass and compared merger
output against the unquantized FP16 reference:

- Overall cosine similarity: **0.994557**
- Mean per-token cosine: **0.997085**
- **Worst per-token cosine: 0.868732** (1st percentile: 0.963504)
- Mean abs diff 0.0177, max abs diff 15.125 (on values with rms ~0.36)
- Relative error: mean 0.48, p99 6.15 (dominated by near-zero reference
  values in the denominator; the tail is real, not purely an artifact of
  the epsilon)

**This is a real, non-trivial degradation, not a clean pass.** The overall
cosine looks fine in isolation (0.995), but the per-token distribution has
a genuine tail: roughly 1% of image tokens see cosine similarity drop into
the 0.86-0.96 range. For a general vision-language task that tail might be
tolerable; for character-level OCR -- where a single corrupted token can be
the difference between two similar characters -- it is a real risk that
**this probe cannot rule out, because it has no way to measure actual OCR
text accuracy**: doing that needs the full text trunk (tokenizer, LM head,
~15-27 GiB depending on checkpoint), which was not part of the approved
download and is deferred to M-V5's real end-to-end gate (which already
budgets a full checkpoint).

**A second, independent finding narrows the INT4 plan further**: `mlp.fc2`
(the down-projection, weight shape `[1152, 4304]`) could not be quantized
at group_size=64 at all in this probe, because **`4304 % 64 == 16 != 0`**
-- `intermediate_size` (item 1's corrected 4304, not 4608) does not divide
evenly into groups of 64. 27 `fc2` tensors were silently skipped by the
probe's `shape[-1] % 64 == 0` guard. A real M-V3 implementation would need
to either pad the down-projection's input width to 4352 (68*64, adding 48
zero columns and rows) or accept an irregular last group -- a genuinely new
implementation cost the plan did not anticipate (this is a different tensor
than the patch-embed padding concern the plan raised and item 5 above
retracted; that one turned out fine, this one is real).

**Recommendation: change the M-V3 default from INT4 to FP16 (unquantized)
for the vision tower's matmul weights.** The reasoning:

- **The memory case for INT4 is weak here.** The tower is 0.858 GiB total
  at FP16, ~32 MiB per block (27 blocks), so 2-slot double buffering costs
  **~65 MiB peak** either way. INT4 would cut that to ~17 MiB -- a savings
  of well under 50 MiB, which is noise next to the multi-GiB text trunk
  already resident for any qwen3_5 install, and next to the KV cache at any
  realistic context length. The plan's original "~4x fewer streamed bytes"
  framing is correct in relative terms but the absolute savings do not
  matter at this component's size.
- **The accuracy cost is real and only partially measured.** A demonstrated
  1% tail of meaningfully degraded tokens, on a component feeding an OCR
  task specifically chosen for its sensitivity to per-character precision,
  is a bad trade for a sub-50-MiB memory savings.
- FP16 also removes the `fc2` padding problem entirely (no group-size
  divisibility constraint) and removes the need for a bit-exact INT4 GEMM
  kernel dependency in M-V2 for the tower path specifically (the batched
  GEMM work is still needed elsewhere, but the vision tower itself would
  use a plain FP16 batched matmul rather than the quantized one).
- This does NOT change the streaming design at all: `StreamLayout` and
  `PreadExpertStreamer` are dtype-agnostic. A block's stride simply becomes
  ~32 MiB instead of ~8 MiB, and 2-slot double buffering stays the
  mechanism. The "stream so peak memory stays constant" property is
  unaffected; only the per-page byte count read from disk goes up as
  described above.

**This reverses the plan's stated INT4-by-default decision.** Flagging for
explicit confirmation before M-V2/M-V3 lock in the dtype, since it changes
which GEMM kernel path the tower needs (plain FP16 batched matmul, not the
INT4 kernel M-V2's table assumed) and removes one risk item while
introducing none.

## 5. Resize semantics -- RESOLVED

Read directly from `../mlx-v/mlx-vlm/mlx_vlm/models/qwen3_vl/processing_qwen3_vl.py`
(a torch-free numpy port of HF's `Qwen2VLImageProcessorFast`, used
unmodified by `qwen3_5`/`qwen3_5_moe`):

- **Resize is PIL `Image.BICUBIC`** (`processing_qwen3_vl.py:89`), not
  torchvision antialias. This is the side of sconce's known PIL-vs-torchvision
  trap that we want -- match PIL bicubic exactly, no antialias pre-filter.
- **`factor = patch_size * merge_size = 16 * 2 = 32`**, not 28 as the plan
  guessed (the plan's 28 assumed `patch_size=14`, the generic Qwen2-VL value;
  this checkpoint's `vision_config.patch_size` is 16). `smart_resize` rounds
  height/width to the nearest multiple of 32, then rescales by
  `sqrt(target_pixels / actual_pixels)` and re-floors/re-ceils to a multiple
  of 32 if outside `[min_pixels, max_pixels]` (`_smart_resize_image`,
  `processing_qwen3_vl.py:94-116`, a direct port of HF's `smart_resize`).
- **Pixel budget is checkpoint-specific, not the generic library default.**
  `preprocessor_config.json` on both checkpoints declares
  `size: {shortest_edge: 65536, longest_edge: 16777216}`, i.e.
  **min_pixels = 65,536 (256x256), max_pixels = 16,777,216 (4096x4096)** --
  far above the generic Qwen2-VL default (`56*56` .. `14*14*4*1280` =
  3,136 .. 1,003,520) that the vendored code falls back to when no
  `preprocessor_config.json` override is given. At the max, a single image
  can produce `(4096/32)^2 = 16,384` patches -> `4,096` merged tokens after
  the 2x2 merge. This is a real "unlimited-ocr"-relevant number: a single
  full-resolution page can cost thousands of prompt tokens, which is exactly
  why the tower must stream (constant memory) rather than resident-load.
- **Normalization**: `image_mean = image_std = [0.5, 0.5, 0.5]`, i.e.
  `2*(px/255) - 1` (matches the plan's DeepSeek-OCR-derived guess; same
  formula, confirmed for this checkpoint's own `preprocessor_config.json`).
- **Patch flatten order (load-bearing for the repack weight layout and the
  vision-io patchify function)**: after resize, the processor builds patch
  rows via `reshape` + a fixed `transpose(0,1,4,7,5,8,3,2,6,9)`
  (`processing_qwen3_vl.py:238-244`). Working through the axis permutation:
  the flattened row order is `[grid_h/merge, grid_w/merge, merge_h, merge_w]`
  as the leading (patch-index) axes, and within one row the trailing
  `C*T*P*P` axes are ordered **`(C, T, P_h, P_w)`** -- channel-major. `PatchEmbed`
  (`qwen3_vl/vision.py:88`) then does
  `reshape(-1, C, T, P, P).moveaxis(1, 4)` before the Conv3d, i.e. it
  converts to **`(T, P_h, P_w, C)`** order for the actual matmul against the
  Conv3d weight. **This is exactly the layout `patch_embed.proj.weight`
  ships in**: safetensors shape `[1152, 2, 16, 16, 3]` = `[out, T, P, P, C]`,
  contiguous, so its 1536 trailing values per output row are already in
  `(T,P,P,C)` order. Consequence for `crates/vision-io` and the repack
  weight ingest: **`vision-io::preprocess` should emit patch rows directly
  in `(T,P,P,C)` order** (skip producing the intermediate `(C,T,P,P)`
  ordering the Python processor uses internally -- there is no reason to
  reproduce that intermediate step, only its final numeric result), and the
  repack walk can copy `patch_embed.proj.weight` into the packed GEMM weight
  **verbatim, with no permutation**, reshaped as a plain `[1152, 1536]`
  matrix.
- **Patch column count is `C*T*P*P = 3*2*16*16 = 1536`, not 1176** (the
  plan's figure assumed `patch_size=14`). **1536 is already a multiple of
  64** (`1536 = 24*64`), so **M-V2 risk item 4 (padding the patch-embed GEMM
  input to satisfy a `cols % 64 == 0` assert) does not apply** -- no padding
  needed at repack or at dispatch.

## 6. Vision token ids and template behavior -- RESOLVED

From `config.json` (both checkpoints):
`vision_start_token_id=248053, vision_end_token_id=248054,
image_token_id=248056, video_token_id=248057` (and an unused
`audio_*` pair in the tokenizer's special-token table, per
`tokenizer_config.json`: `image_token="<|image_pad|>"`,
`video_token="<|video_pad|>"`, `vision_bos_token="<|vision_start|>"`,
`vision_eos_token="<|vision_end|>"`).

From the checkpoint's own `chat_template.jinja`: an image content part
renders as the literal three-token run
`<|vision_start|><|image_pad|><|vision_end|>` -- **one single `<|image_pad|>`
token per image at the template level**, not N copies. `add_vision_id` (a
template-level flag) only controls an optional `"Picture N: "` text prefix
before the run and **defaults to falsy** when the caller does not set it, so
**the port's existing hardcoded `add_vision_id=false`
(`crates/tokenizer/src/jinja_chat_template.rs:141`) needs no change** -- it
already matches this template's default behavior for the common (unlabeled)
case. Do not thread a new context variable for this.

**Consequence for M-V6**: the N-token expansion (one `<|image_pad|>` ->
`merged_tokens` copies of that same token id) is **not** a text-splicing
problem before tokenization. It happens as a **token-id post-process step
after the template is rendered and tokenized**: locate each `image_token_id`
occurrence in the encoded id sequence (in order) and replace it with
`merged_tokens_for_that_image` repetitions of the same id, using the
per-image `merged_tokens` count already known from `vision-io::preprocess`
(which must run before tokenization only insofar as we need to know each
image's `merged_tokens` count in image order -- the actual splice is on
token ids, not on prompt text). This is simpler than the plan's original
"splice N pad tokens into message content before render" framing; update
M-V6 to splice at the token-id level instead.

## Corrections to the approved plan (M-V2/M-V3 kernel table and risk register)

1. `intermediate_size` is **4304**, not 4608 (fc1/fc2 shapes in the kernel
   table). 4608 is only the merger's hidden width (`hidden_size *
   spatial_merge_size^2 = 1152*4`).
2. `out_hidden_size` (merger output / trunk injection width) is **5120**,
   matching the qwen3_5 trunk hidden size exactly (expected, but now
   confirmed rather than assumed).
3. Patch-embed GEMM: **1536 input columns, not 1176**; already
   `%64==0`. **Risk register item 4 (GEMM column padding) is retracted.**
4. Patch-embed weight requires **no permutation at repack** -- the
   safetensors layout already matches the GEMM's required row-major
   `(T,P,P,C)` column order. `vision-io::preprocess` must emit patch rows in
   that same `(T,P,P,C)` order (a straightforward but exact requirement --
   get the axis order right once here, since it is silent-wrong-numerics on
   a mismatch, not a crash).
5. `factor` (resize rounding granularity) is **32** (`patch_size(16) *
   merge_size(2)`), not 28. Pixel budget is checkpoint-specific
   (`preprocessor_config.json`'s `size.{shortest_edge,longest_edge}`,
   65,536..16,777,216), not the generic library default -- `vision-io` must
   read this from the install's config rather than hardcoding the generic
   Qwen2-VL numbers.
6. M-V5's mRoPE dispatch condition is narrower than planned: **only
   image-pad token positions ever need `rope_mrope_interleaved`**; every
   text token (before, between, or after images) provably uses the existing
   `rope_neox_subdim` path unchanged, not merely "when the prompt happens to
   be text-only."
7. M-V6's token expansion is a **post-tokenization id-sequence splice**, not
   a pre-render text splice.

## Reproducing items 3 and 4

Both ran against the real `prism-ml/Bonsai-27B-mlx-1bit` vision tower
(879 MiB, fetched by ranged HTTP -- the 27B text trunk was never
downloaded) and a synthetic rendered OCR page, via:

```sh
uv run --python 3.12 --with requests -- \
  python scripts/fetch_vision_tower.py prism-ml/Bonsai-27B-mlx-1bit /tmp/vision-probe

uv run --python 3.12 --with mlx --with numpy --with pillow --with transformers -- \
  python scripts/vision_tower_probe.py \
    --vendor-root ../mlx-v/mlx-vlm \
    --config /tmp/vision-probe/config.json \
    --tower-dir /tmp/vision-probe \
    --image /tmp/vision-probe/imgs/ocr_page.png \
    --mode activation   # or --mode int4
```

**Not yet run: the 4096x4096 extreme-page-size activation check** flagged
in item 3, and the full end-to-end OCR-text-diff measurement flagged in
item 4 (needs the text trunk, deferred to M-V5).
