# Mage-VL vision (`mage_vl`), still images

Status: wired, tested on synthetic weights only. No real weights have run, so
nothing here claims parity or answer quality.

Reference read (not executed): `mlx_vlm/models/mage_vl` (`vision.py`,
`mage_vl.py`, `processing_mage_vl.py`, `config.py`).

## Published checkpoints (anonymous metadata reads, 2026-11-27)

| Repository | Revision | Layout | Verdict |
| --- | --- | --- | --- |
| `mlx-community/Mage-VL-8bit` | `40dc28d825dfaf0dfdea374f39539475d5ee35f2` | affine 8-bit group 64, BF16 tower, `model.safetensors` 5,348,214,363 B | intake accepts the shapes |
| `netprojects/Mage-VL-Cowork-MLX-4bit` | `0fa390b98d72fa7860b0979ef32987ecad25e2de` | affine 4-bit group 64, BF16 tower, 3,142,599,841 B | intake accepts the shapes |
| `sr29/Mage-VL-mlx-4bit`, `pratikgx/Mage-VL-Cowork-MLX-4bit` | `c39d8024`, `97e4f26e` | 4-bit group 64 | configs read only |
| `mlx-community/Mage-VL-OptiQ-4bit`, `nativ-community/Mage-VL-OptiQ-4bit` | `c98dad5f`, `4f0a4243` | mixed 4/8-bit, tower in `optiq/optiq_vision.safetensors` as `[out, C, P, P]` | refused |
| `microsoft/Mage-VL` | `d88b153285f1633a61b2f693c59c8576693af185` | BF16 `model.*` names | refused |
| `microsoft/Mage-VL-Cowork` | `6fbe7f91da477c1c73790dbb1f08d001b8626be1` | BF16, `llava_onevision2` | refused |

Headers (shapes and dtypes) were range-read for the first two and for the
OptiQ pair. No weight data was downloaded.

## Scope

- Still images do not depend on the codec path. A still is one frame
  (`temporal_patch_size` 1), rope time index 0, full attention over its own
  patches. Several images attend separately, as the reference's block-diagonal
  mask does.
- Video and codec-stream input are not ported. No front end has a video content
  part (the server's part type has no `video_url` variant; that request was not
  exercised here), and a grid of more than one frame is refused by name in
  `mage_vit_rope_rows`.
- `pull-vision` sidecars are refused: install the whole checkpoint.
- No catalog row. Install with `turbospark-model pull --repo`.
- The macOS app was not touched or tested.

## What differs from the towers already here

The tower runs in `crates/runtime/src/vision/qwen2.rs` as a variant of the
Qwen2-VL tower (same twelve block roles, full attention, same merger):

- LayerNorm (`layernorm_pre`) between the patch embedding and block 0.
- Exact erf GELU in the block MLP and the merger. LayerNorm eps 1e-6.
- Rope: head split 4:6:6 over `(t, h, w)` at half width, exponent `j / size`,
  theta 10,000, table repeated twice across the head, and an INTERLEAVED
  rotation over adjacent lanes (`vision_rope_interleaved_fp16`).
- No post-encoder norm. The last block's rows reach the merger.
- Trunk: plain `qwen3`, 1-D positions. `set_prompt_vision` keeps the spans of
  the front ends' mRoPE walk and injects at sequential positions.

Preprocessing is the existing Qwen path (factor 32, PIL bicubic, CLIP
statistics) with the checkpoint's budget, 3,136 to 4,000,000 pixels.

## Intake

`config.json` root `mage_vl`, trunk from `text_config` (`qwen3`), rope theta
from `rope_theta` or `rope_parameters`. The tower is detected from
`vision_tower.*` bytes; every tensor name and shape is checked. A `mage_vl`
repository with no such tensors is refused rather than installed text-only.

## Tests (synthetic)

- `crates/repack/tests/mage_vl_vision.rs`: refusals and the walk.
- `crates/catalog` `a_mage_vl_pull_installs_the_tower_or_refuses_by_name`.
- `crates/runtime/tests/mage_vl_vision_synthetic.rs`: GPU tower against a CPU
  forward pass written in the test (worst difference about 0.0016 of the peak,
  bound 0.003), and injection into the qwen3 trunk. Mutations that failed it:
  split-half rope, rope exponent, quick GELU in the block or the merger,
  skipped `layernorm_pre`, removed position flattening.
- `crates/ffi/tests/mage_vl_vision_synthetic.rs`: `ts_session_open`,
  `ts_generate` and the in-process server, image and plain turns.

The CLI's `--image` takes the same generic path but was not run.

## Real gate still owed

Pull `mlx-community/Mage-VL-8bit` @ `40dc28d8` (about 5.4 GB), or the 4-bit
Cowork conversion (about 3.1 GB). Compare tower stages with mlx-vlm on one
image, then run a text smoke and an image smoke through `--messages-file`.
