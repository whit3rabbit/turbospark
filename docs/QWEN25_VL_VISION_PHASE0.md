# Qwen2.5-VL vision: Phase 0 and bring-up record

Facts about the Qwen2.5-VL vision tower and how its trunk consumes image rows,
read off the real checkpoints and the reference source, with the gates that
were run against them. The mechanics shared with the other towers are in
[VISION.md](VISION.md); this page is what is different.

## Sources

| What | Where | Revision |
| --- | --- | --- |
| 3B checkpoint | `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` | `46d4cf06a06ffc1a766c214174f9cbed2f45bcab` |
| 7B checkpoint | `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` | `fdcc572e8b05ba9daeaf71be8c9e4267c826ff9b` |
| Reference source (read only) | mlx-vlm `models/qwen2_5_vl/{vision,language,config}.py` | `main` as cloned 2026-11-01 |

The reference source was read and transcribed. It was **not executed**: the
parity reference below is a numpy transcription (`scripts/qwen25_vl_tower_reference.py`),
and no number on this page is a comparison against mlx-vlm run on the same
bytes. See "What is and is not verified".

## Verdict

**Same family, new tower.** The trunk is a plain Qwen2 dense model (28 layers,
hidden 3584 on the 7B, which is the registered `Qwen2Dense` shape), so no new
`ModelFamily` was added: `config.json` resolves `qwen2_5_vl` to `qwen2`, and
the tower rides on `VisionConfig` with `VisionTowerKind::Qwen25Vl` (persisted
`qwen25_vl`). The 3B (hidden 2048, 36 layers) is not a registered `Qwen2Dense`
shape, so it is used for tower checks through a sidecar only.

The tower is a cousin of the Qwen3-VL one and not a mode of it:

| | Qwen3-VL / qwen3_5 tower | Qwen2.5-VL tower |
| --- | --- | --- |
| Block norms | LayerNorm with bias | RMSNorm, weight only (eps 1e-6) |
| MLP | `fc1`, GELU tanh, `fc2` | gated: `down(silu(gate(x)) * up(x))`, all with bias |
| Position | learned table, bilinear | none; 2-D rope only |
| Attention | whole page, every block | windows of 4x4 merged tokens on 28 of 32 blocks |
| Merger | LayerNorm, reshape, fc1, GELU erf, fc2 | RMSNorm (`ln_q`), reshape, `mlp.0`, GELU erf, `mlp.2` |
| Patch | 16, `(out, T, P, P, C)` | 14, `(out, T, P, P, C)` |

## Inventory (3B)

390 tensors under `vision_tower.`: 12 per block times 32 blocks, plus the patch
embedding and five merger tensors. All F16 and **unquantized** on both
checkpoints. Per block: `norm1.weight`, `attn.qkv.{weight,bias}`,
`attn.proj.{weight,bias}`, `norm2.weight`, `mlp.{gate,up,down}_proj.{weight,bias}`.
Non-block: `patch_embed.proj.weight` `[1280, 2, 14, 14, 3]`, `merger.ln_q.weight`,
`merger.mlp.0.{weight,bias}` `[5120, 5120]`, `merger.mlp.2.{weight,bias}`
`[2048, 5120]`.

`patch_embed.proj.weight` is already in the `(T, P_h, P_w, C)` row order
`crates/vision-io` emits, so it is copied without a permutation, as for the
other towers.

Config (`vision_config`): depth 32, hidden 1280, 16 heads of 80, intermediate
3420, patch 14, temporal 2, merge 2, `out_hidden_size` 2048 (the 3B) or 3584
(the 7B), `window_size` 112, `fullatt_block_indexes` `[7, 15, 23, 31]`.
`in_chans` is the spelling of `in_channels`.

**Window size and the full-attention pattern are properties of the tower kind,
checked at intake and not stored**, the way Gemma 4's rope theta is. Every
published size declares 112 and every eighth block; the parser refuses a
checkpoint that says otherwise (`crates/repack/tests/qwen25_vl_config.rs`)
instead of running it with the wrong windows.

## The forward pass

```text
x     = patch_embed(rows)                       # linear, no bias, no pos table
order = window order of the page's merged tokens
x, rope = permute both by `order`               # in units of 2x2 patch groups
for blk in 0..32:
    x += attn(rms(x, n1))     # q,k rotated by the 2-D rope; v not
                              # segments = windows, except blocks 7, 15, 23, 31
    x += down(silu(gate(rms(x, n2))) * up(rms(x, n2)))
out = mlp2(gelu_erf(mlp0(rms(x, ln_q).reshape(-1, 5120))))
out = out un-permuted by `order`
```

- **Windows.** Merged tokens (each a 2x2 patch group) are regrouped into
  windows of 4x4 merged tokens, row-major inside a window and row-major over
  windows. The reference pads each edge by `window - edge % window`, which adds
  an empty window row when the edge divides evenly; it drops zero-length
  segments, so the padding changes nothing and the port does not build it.
  Edge windows are ragged.
- **Rope.** The angle row is `[h angles | w angles]`, `head_dim / 4` each,
  theta 10000, repeated twice over the head and applied with `rotate_half`:
  the same layout `crates/vision-io` already emits for Qwen3-VL, so the 2-D rope
  kernel is shared. The rows are permuted with the patches.
- **The merger runs in window order and the output rows are put back** with
  `argsort(window_index)` on the host.

## Preprocessing

`Qwen2VLImageProcessor`: smart resize to a multiple of 28 (`patch * merge`)
between `min_pixels` 3136 and `max_pixels` 12,845,056, PIL bicubic, rescale
1/255, **CLIP mean and std** (0.4815, 0.4578, 0.4082 / 0.2686, 0.2613, 0.2758;
Qwen3-VL uses 0.5 / 0.5), the still repeated across the 2-frame temporal patch.
`crates/vision-io` already reproduces PIL's bicubic bit for bit and reads the
mean, std and pixel budget from `preprocessor_config.json`, so patch 14 needed
no preprocessing change. The 1024x1280 test page resizes to 1036x1288: 92x74
patches, 1,702 merged tokens.

## The trunk

- **Chunked mRoPE.** `rope_scaling.mrope_section` is `[16, 24, 24]`, summing to
  `head_dim / 2`. The sections are three **contiguous** runs of rotary pairs
  (pairs 0..16 take `t`, 16..40 take `h`, 40..64 take `w`), where Qwen3-VL
  interleaves them by residue mod 3. `rope_mrope_chunked` is the new kernel; it
  ends in the same `apply_neox_pair` as the interleaved one, so at
  `t == h == w` it is bit-identical to the pre-existing rope
  (`crates/gpu/tests/qwen25_vision_kernels.rs`). The llama flow picks it when
  the tower kind is `qwen25_vl`.
- **Positions.** Text tokens carry one position in all three slots. An image
  block's `t` is constant, its `h` and `w` run over the merged grid, offset by
  the text before it, and the clock advances by the largest axis: the same
  `get_rope_index` walk `crates/vision-io::mrope` already ports for images.
- **Template.** Per image the checkpoint's template renders
  `<|vision_start|><|image_pad|><|vision_end|>`; the placeholder expands to the
  merged-token count at the token-id level. The template lives in
  `tokenizer_config.json` (`chat_template.json` is a duplicate).
  A first message that is not a system turn gets the template's default
  "You are a helpful assistant." system turn.
- **Ids.** `image_token_id` 151655, `vision_start` 151652, `vision_end` 151653,
  `video_token_id` 151656 (unused: no video path).

## What was built

| Layer | Change |
| --- | --- |
| `model-io` | `VisionTowerKind::Qwen25Vl` |
| `repack` | `qwen2_5_vl` in the HF registry; `Qwen2Dense` buckets `vision_tower.*` as `VisionTower`; `parse_vision_config` reads the Qwen2.5 spelling and checks the kind's constants; the shared tower reader takes the Qwen2.5 role tables |
| `catalog` | `Qwen2Dense` admitted to combined-vision (catalog intent and bytes-detected) and to `pull-vision` |
| `gpu` | `vision_silu_fp16`; `rope_mrope_chunked` |
| `runtime` | `vision::qwen25::Qwen25VisionTower` (host window permutation, per-window attention dispatch), routed from `encode_image`, so `set_prompt_vision` and every front end are unchanged; the llama flow's one mRoPE dispatch site |

## Gates

### Tower against the numpy transcription (3B tower, sidecar)

`the_real_tower_matches_the_numpy_reference` (ignored; setup in its doc).
Cosine per stage, patch rows from `crates/vision-io`:

| Page | Patches | patch embed | block 0 | block 31 | merger |
| --- | --- | --- | --- | --- | --- |
| 448x448 | 1,024 (16 windows) | 0.99999998 | 0.99999993 | 0.99999657 | 0.99999660 |
| 1024x1280 | 6,808 (ragged edges) | 0.99999998 | 0.99999993 | 0.99998120 | 0.99998036 |

**Mutation check.** Forcing every block to full-page attention drops block 31
to 0.680 and the merger to 0.699 on the 448x448 page and fails the test, so the
gate sees the window logic.

Block 31's activation peak is 27,823 on the small page and 34,181 on the large
one against FP16's 65,504, so FP16 has a factor of 1.9 in hand on the page
tried; a larger or stranger page could close that and the overflow capture
(`TURBOSPARK_VISION_OVERFLOW`) is not wired to this tower.

### Kernels

`crates/gpu/tests/qwen25_vision_kernels.rs`: SiLU against its definition and
against GELU (the fixture separates them); the chunked mRoPE bit-identical to
`rope_neox_subdim` at equal positions, equal to a CPU selector when they
diverge, and distinguishable from the interleaved selector; a section split
that does not cover the rotary pairs is refused.

## What is and is not verified

Verified on real bytes: the tower's four stages on two page sizes (above), the
ingest of both checkpoints' towers, the kernels, the config parser's refusals.

**Not verified:**

- **Against mlx-vlm executed on the same bytes.** The reference here is a numpy
  transcription of its source, so a misreading shared by the transcription and
  the Rust tower would pass. Running mlx-vlm needs a decision about executing
  third-party code that was not made here.
- **Preprocessing against the HF processor bit for bit.** `crates/vision-io`
  is PIL-exact and generic over patch size, and the tower parity above
  replays its rows, but no Qwen2.5 golden fixture was generated.
- **Logit parity of the trunk with an image** (the stage 4 of
  [VISION.md](VISION.md)).
- **Chunked prefill with an image on this family**, speculative decoding with
  an image, and video.
