# Qwen2-VL vision: Phase 0 and bring-up record

Facts about the Qwen2-VL vision tower and how its trunk consumes image rows,
read off the real checkpoint and the reference source, with the gates that were
run against them. The mechanics shared with the other towers are in
[VISION.md](VISION.md); the 2.5 sibling's page is
[QWEN25_VL_VISION_PHASE0.md](QWEN25_VL_VISION_PHASE0.md). This page is what is
different.

## Sources

| What | Where | Revision |
| --- | --- | --- |
| 2B checkpoint | `mlx-community/Qwen2-VL-2B-Instruct-4bit` | `01af461cdb9574acc09084a0ef94e216e142b085` |
| Reference source (read only) | mlx-vlm `models/qwen2_vl/{vision,language,config,processing_qwen2_vl}.py` | `e8035ae`, shallow clone at `/tmp/mlx-vlm` |
| Sibling comparison | mlx-vlm `models/qwen2_5_vl/` against `models/qwen2_vl/` | same clone |

The reference source was read and transcribed. It was **not executed**: no
number on this page is a comparison against mlx-vlm run on the same bytes. See
"What is and is not verified".

## Verdict

**Same family as the 2.5 sibling, a third tower.** The trunk is a plain Qwen2
dense model (28 layers, hidden 1536, kv 2, `rms_norm_eps` 1e-6, flat config at
the root beside `vision_config`), so `config.json` resolves `qwen2_vl` to
`ModelFamily::Qwen2Dense` through its own registry row (exact equality;
`qwen2`, `qwen2_vl` and `qwen2_5_vl` are three rows, never one), and the tower
rides on `VisionConfig` with `VisionTowerKind::Qwen2Vl` (persisted `qwen2_vl`).
The tower keys every tower-specific path; nothing keys on a shape.

mlx-vlm's own `qwen2_vl/config.py` `VisionConfig` defaults are junk (`embed_dim`
1280 against `hidden_size` 1536, `image_size` 384, `vocab_size` 32000, a
`mlp_ratio` 4.0 that only happens to match). Every value here comes from the
published checkpoint's `config.json`, fetched with `curl -L` at the revision in
the table above; nothing was taken from a default.

## The tower, against the 2.5 sibling

Both towers share the conv patch embed (`[out, T, P, P, C]`, no bias, copied
verbatim), the 2-D rope (theta 10000, `[h angles | w angles]`, NeoX pairing),
and no absolute position table. The blocks differ:

| | Qwen2.5-VL (`Qwen25Vl`) | Qwen2-VL (`Qwen2Vl`) |
| --- | --- | --- |
| Block norms | RMSNorm, weight only | LayerNorm, weight AND bias, eps 1e-6 |
| MLP | gated: `down(silu(gate(x)) * up(x))`, bias each | plain: `fc2(gelu(fc1(x)))`, bias each |
| Block MLP activation | `nn.silu` | `nn.GELU(approx="fast")` |
| Attention | windows of 4x4 merged tokens on 28 of 32 blocks | FULL on every block; the reference has no window machinery at all |
| Merger `ln_q` | RMSNorm, weight only | LayerNorm, weight AND bias |
| Merger MLP | `mlp.0`, GELU erf, `mlp.2` | `mlp.0`, GELU erf, `mlp.2` (the same) |

Three details worth their own lines:

- **The block MLP's activation is the FAST SIGMOID GELU.** `nn.GELU(approx="fast")`
  in mlx is `x * sigmoid(1.702 x)` (`activations.py::gelu_fast_approx`, max
  error about 0.015 against the erf form), which is this port's
  `GeluKind::Quick` kernel -- the same function CLIP's quick-GELU uses. The
  reference reads the activation from NO config key (it is hardcoded in the
  module). transformers spells the same function `quick_gelu`
  (`Qwen2VLVisionConfig.hidden_act`, read in 5.18.0), so the two agree and
  intake accepts only that spelling. The merger's own GELU is a bare `nn.GELU()`, the exact erf
  form, same as the 2.5 tower.
- **Full attention on every block** means no window permutation, no
  `cu_window_seqlens`, and page order from patch rows to returned merger rows.
  The `qwen2_5_vl` window keys (`window_size`, `fullatt_block_indexes`) do not
  exist in the reference config, and a `qwen2_vl` file that carries either is
  refused by name rather than run full-attention anyway.
- **`vision_config.hidden_size` names the TRUNK width on this family** (1536
  on the 2B), because it is `PatchMerger`'s output dim
  (`PatchMerger(dim=config.hidden_size, context_dim=config.embed_dim)`); the
  block residual width is `embed_dim` (1280). The parser refuses a
  disagreement, and maps them onto the shared `VisionConfig` as
  `out_hidden_size` (trunk) and `hidden_size` (tower) respectively.

The MLP width is the float relation `mlp_ratio * embed_dim` (4 x 1280 = 5120 on
the 2B), not a stored key. The parser computes the product, requires it to be an
exact positive integer, and stores only the integer: the manifest stays
float-free, and `mlp_ratio` itself is never copied (a non-binary float cannot
round-trip a manifest, AGENTS.md Gotcha 24).

## Inventory (2B)

391 tensors under `vision_tower.`: 12 per block times 32 blocks, plus the patch
embedding and six merger tensors. All F16 and **unquantized** on the 4-bit
checkpoint (the tower bodies sit outside the `quantization` block, as on the
2.5 sibling). Per block: `norm1.{weight,bias}`, `attn.qkv.{weight,bias}`,
`attn.proj.{weight,bias}`, `norm2.{weight,bias}`, `mlp.fc1.{weight,bias}`
`[5120, 1280]`, `mlp.fc2.{weight,bias}` `[1280, 5120]`.
Non-block: `patch_embed.proj.weight` `[1280, 2, 14, 14, 3]`,
`merger.ln_q.{weight,bias}`, `merger.mlp.0.{weight,bias}` `[5120, 5120]`
(square: the shuffle makes fc1's input and output the same width),
`merger.mlp.2.{weight,bias}` `[1536, 5120]`.

Config (`vision_config`): depth 32, `embed_dim` 1280, 16 heads of 80,
`mlp_ratio` 4, patch 14, temporal 2, merge 2, `in_chans` 3,
`hidden_size` 1536 (the trunk width, see above). No `layer_norm_eps` and no
`hidden_act` key: absent means the reference's own default (eps 1e-6; the
hardcoded fast GELU), and when a checkpoint declares either it is CHECKED
against those values and refused on a mismatch, never stored -- the eps and the
activation are properties of the kind (`VisionTowerKind`'s float-free rule, the
way SigLIP's final-norm eps and Gemma 4's rope theta are handled).

Token ids, all four declared at the root of the published file:
`vision_start_token_id` 151652, `vision_end_token_id` 151653,
`image_token_id` 151655, `video_token_id` 151656 (no video path exists here).

**Refusals at intake** (`crates/repack/src/qwen36_config.rs::parse_qwen2_vl_vision_config`,
`crates/repack/tests/qwen2_vl_config.rs`): a `deepstack_visual_indexes` key
(Qwen2-VL has no deepstack; that is Qwen3-VL, and silently dropping the key
would drop the injections it describes), a window key, `mlp_ratio` whose
product is not a whole positive width, a tower width that disagrees with the
trunk, a foreign `hidden_act` or `layer_norm_eps`, `rope_scaling.type` neither
`mrope` nor `default`, a depth other than 32, and the 14 x 2 x 2 x 3 geometry
mismatches.

## The trunk's mRoPE

`rope_scaling` sits at the flat root: `{"type": "mrope", "mrope_section":
[16, 24, 24]}`, summing to `head_dim / 2` (128 / 2). The sections are three
CONTIGUOUS runs of rotary pairs, exactly the 2.5 scheme: the reference's
`Qwen2RotaryEmbedding` passes `style="chunked"` and
`apply_multimodal_rotary_pos_emb(..., style="chunked")`, and the two families'
`get_rope_index` agree on images (their diff is video-only:
`second_per_grid_ts` time intervals, which no image path reaches). The llama
flow's chunked-mRoPE dispatch therefore takes BOTH tower kinds
(`families/llama/attn.rs`), and the shared `crates/vision-io::mrope` triples
and the install seam (`set_prompt_vision`) are reused unchanged.

The reader accepts the `mrope_section` under the flat root `rope_scaling` (the
published conversions), `text_config.rope_scaling`, or
`text_config.rope_parameters` -- two spellings, one meaning -- and refuses a
file that declares none of them by name.

## Preprocessing

None needed. The published `preprocessor_config.json` is the same
`Qwen2VLImageProcessor` file the 2.5 sibling ships: smart resize to a multiple
of 28 (`patch * merge`) between `min_pixels` 3136 and `max_pixels` 12,845,056,
PIL bicubic, rescale 1/255, CLIP mean and std (0.4815, 0.4578, 0.4082 /
0.2686, 0.2613, 0.2758), the still repeated across the 2-frame temporal patch.
`crates/vision-io` reads the geometry and the pixel budget from the file and is
generic over patch size, so patch 14 needed no code. The 896x1120 gate page
below resizes to 896x1120 exactly (32x40 patches, 320 merged tokens).

## What was built

| Layer | Change |
| --- | --- |
| `model-io` | `VisionTowerKind::Qwen2Vl` |
| `repack` | `qwen2_vl` in the HF registry; `parse_vision_config` reads the `qwen2_vl` key vocabulary (`embed_dim`, `mlp_ratio`, trunk-width `hidden_size`) and checks the kind's constants; the shared tower reader takes the Qwen2-VL role tables (`vision.rs`) |
| `runtime` | `vision::qwen2::Qwen2VisionTower` (LayerNorm blocks, quick-GELU MLP, full attention on every page), routed from `encode_image`, so `set_prompt_vision`, the prompt builders and every front end are unchanged; `families/llama/attn.rs` takes the chunked mRoPE for this kind too |

`vision-io`, `cli`, `ffi` and `server` needed no change: the preprocessing is
parameter-driven, and the default `attach` path reaches `encode_image`, which
dispatches on the tower kind.

## Vision pixel budget

`resolve_vision_pixel_budget` covers this tower and the 2.5 one under the same
policy as the others (`vision::budget`): clamp `max_pixels` until the page's
scratch fits `physical - guard reserve - weights/expert cache - KV cache`,
refuse if `min_pixels` does not fit, and do nothing under `--load-guard off`.
Having no position table, they skip `VisionShape` and use their own
`scratch_bytes` (`vision/qwen2.rs`, `vision/qwen25.rs`), which a unit test
holds equal to the summed lengths of the allocated buffers. Full attention adds
no `patches x patches` buffer, so scratch is linear in the page on both towers;
windowing on 2.5 changes the work, not the memory.

Computed (not measured) scratch at the declared 12,845,056 pixels (65,536
patches, 16,384 merged tokens):

| Tower | Bytes |
| --- | --- |
| Qwen2-VL 2B | 1,567,621,120 (1.46 GiB) |
| Qwen2-VL 7B | 1,634,729,984 (1.52 GiB) |
| Qwen2.5-VL 3B | 1,587,920,896 (1.48 GiB) |
| Qwen2.5-VL 7B | 1,638,252,544 (1.53 GiB) |

Under the default `relaxed` guard (4 GiB reserve) the 3B and 2B ceilings stand
unclamped while weights plus KV cache stay under about 30.5 GiB on a 36 GiB
machine and 10.5 GiB on a 16 GiB one. So on either, an ordinary session still
serves `max_pixels` 12,845,056. The clamp acts on small or heavily committed
machines. Host-side copies of the patch rows are not counted, as for the other
towers.

## Gates

### Real install, end to end (2B, combined)

`mlx-community/Qwen2-VL-2B-Instruct-4bit` @ `01af461d` through
`turbospark-model pull` with the release CLI built from this tree: probe
RUNNABLE (declares `qwen2_vl`, family `qwen2`, MLX affine 4-bit at group 64),
2.1 GiB install. The manifest carries `visionTower qwen2_vl`, depth 32, hidden
1280, out 1536, `visionMropeSection [16, 24, 24]`, and `packed_vision/` holds
32 block blobs. From the install:

- **Text smoke.** `--messages-file`, banner `chat template applied`: "the
  largest planet in the solar system" answers `Jupiter` at EndOfTurn.
- **Image smoke.** A deterministic PIL-rendered text page (896x1120, Helvetica:
  `TURBOSPARK` / `QWEN2VL` / `GATE PAGE` / `magic word:` / `zephyr`) is
  transcribed exactly, all five lines, at EndOfTurn; prefill 1314 tokens (320
  merged image tokens plus the prompt).
- **No-image control.** The same transcription question with no image part:
  the model declines ("I'm sorry, but I cannot assist with that.") and none of
  the page's strings appear; prefill 32 tokens. The transcription is
  image-driven.
- **Re-run after the audit fixes** (2026-11-27, same revision, release CLI,
  greedy, `--max-new 120`). Text answers Jupiter. A large-text page
  (`MARIGOLD` / `HARBOR 4821` / `GATE PAGE` / `magic word:` / `zephyr`) reads
  back all five lines, in title case instead of the page's capitals, prefill
  1310 tokens. The no-image control says there is no image (prefill 28). A
  dense-text page reads back recognizable row text with errors until the
  token cap. The install was removed afterwards.
- **Cleanup.** The install was removed after the runs; no Hugging Face cache
  entries were added by the ranged download.

**An honest note on the mRoPE seam.** The first gate run predates the
`families/llama/attn.rs` fix above: the trunk dispatched the INTERLEAVED
mRoPE selector for the new kind, and the image transcription still passed in
full. An end-to-end transcription gate is therefore NOT a sensitive check for
that seam -- the chunked dispatch is pinned by the source read (`style=
"chunked"` in `qwen2_vl/language.py`, and the image-identical `get_rope_index`)
and by the kernel's own parity gate on the 2.5 family
(`crates/gpu/tests/qwen25_vision_kernels.rs`), not by this page's smoke.

The text and image smokes were re-run to completion after the fix, same
outcomes (Jupiter; all five lines, this time at the rendered capitalization
exactly; the control declines).

### Offline tests

- `crates/repack/tests/qwen2_vl_config.rs`: the published config parses to the
  new tower with every field pinned; both `mrope_section` spellings parse; a
  foreign root `model_type` reaches the foreign reader and is refused; deepstack,
  window keys, fractional MLP width, trunk-width disagreement, foreign
  eps/act, depth and geometry are refused by name; the role tables cover the
  published tower (12 x 32 + 7 = 391).
- `crates/repack/tests/arch_registry.rs`: the `qwen2` / `qwen2_vl` /
  `qwen2_5_vl` trio resolves to the same family through three distinct rows,
  in both directions, with the near-misses unrecognized.
- `crates/runtime/src/vision/qwen2.rs` unit tests: the published shape
  resolves; foreign kinds and non-divisible/zero shapes are refused; the role
  table and per-role sizes are pinned (the mutation guard for the block blob).
- **Mutation check.** Disabling the deepstack refusal in
  `parse_qwen2_vl_vision_config` makes `a_deepstack_key_is_refused_by_name`
  fail; the guard was restored and the suite re-run green. The role-table
  mutation guard is the same test the InternViT tower carries: same-shaped
  slots (the two norm weights, the two fc biases) can only be told apart by
  the exact table content.

## What is and is not verified

Verified on real bytes: the end-to-end install (probe, walk, manifest fields,
32 packed blobs), the text and image smokes with the no-image control, the
config parser's refusals, and the registry exact-equality.

**Not verified:**

- **Tower-stage parity against a reference implementation.** No numpy
  transcription exists for this tower (`scripts/qwen25_vl_tower_reference.py`
  covers the 2.5 sibling only), so the LayerNorm/quick-GELU/full-attention
  transcription above has no per-stage cosine evidence. One block is checked
  against a CPU transcription at synthetic tiny widths
  (`vision::qwen2::tests::a_block_matches_a_cpu_transcription_of_the_reference`);
  the patch embed and the merger have no such test.
- **Against mlx-vlm executed on the same bytes.** As on every tower page so
  far.
- **The mRoPE seam is pinned synthetically**, not on the real checkpoint:
  `families/llama/attn.rs::mrope_seam_tests`.
- **A large page's GPU time and real peak memory.** The pixel budget below
  bounds scratch memory only. Every block attends over the whole page, so work
  grows with the square of the patch count and nothing caps it. No page near
  `max_pixels` has been run.
- **Logit parity of the trunk with an image**, the 7B checkpoint, video, and
  chunked prefill with an image on this family (the text smoke took the
  chunked path unchanged; an image prompt was not driven through chunked
  prefill specifically).
- Preprocessing against the HF processor bit for bit (no golden fixture; the
  2.5 page's note carries over unchanged).
