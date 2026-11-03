# LFM3-VL vision: Phase 0 and bring-up record

Facts about the `lfm3_vl` checkpoint and its tower, what this port does with
them, and what was measured. The tower is a NEW kind
(`VisionTowerKind::Lfm3Vl`, persisted `lfm3_vl`), so nothing on
[`VISION.md`](VISION.md) describes it. The trunk is the existing `lfm3` family.

## Sources

| What | Where | Pin |
| --- | --- | --- |
| Checkpoint under test | `LiquidAI/LFM3-VL-2B-MLX-8bit` | `343038aa6c2c16df91b55d114cf374eaccea4b84` |
| Reference code read | mlx-vlm `mlx_vlm/models/lfm3_vl/{vision,lfm3_vl,config}.py` and `lfm3/language.py::MLP`, plus the repo's own `image_processing_lfm3_vl.py` | `4693cd0e`-era checkout, cloned 2026-11-01 |
| Other builds seen | `-MLX-4bit` (`4146d7fa`), `-MLX-6bit`, `-MLX-5bit`, `-MLX-bf16` | headers and configs only, except 4-bit |

Every fact below came from `config.json`, `processor_config.json`, the
safetensors headers, the HF image processor and the mlx-vlm source.

## Verdict

**This is not SigLIP2, and not the Qwen tower.** `docs/LFM3_PHASE0.md` filed
`LFM3-VL-2B` as "towers are not the Qwen ViT this engine has", which was right
and underspecified. The tower is 14 RMSNorm blocks that alternate a gated
depthwise 3x3 convolution with grouped-query attention, behind a linear
patchifier and ahead of a pixel-unshuffle MLP projector. Almost every kernel
it needs already existed (matmul, RMSNorm, LayerNorm, GELU, 2-D rope,
bidirectional attention, residual add, mul). Two did not: SiLU and the gated
depthwise conv.

**The trunk needed no work.** The text config equals the registered
`lfm3_1_7b` baseline field for field (hidden 2048, 24 layers, 16 heads over 4
KV heads, head_dim 128, FFN 7680, vocab 128000, the same 13/6/5 layer mask),
so the VL checkpoint's trunk installs and runs through the existing lfm3
flow. Positions are plain sequential: LFM3-VL has no mRoPE, and the template
puts `<|vision_start|><|image|><|vision_end|>` around each image, the marker
run `splice_and_walk` already expands by token id.

## Tensor inventory (`LFM3-VL-2B-MLX-8bit`)

All BF16 and unquantized (the quantization block covers the trunk only).

| Tensor | Shape |
| --- | --- |
| `vision_tower.patchifier.weight` | [1024, 1536] |
| `vision_tower.embedding_norm.weight` | [1024] |
| `vision_tower.layers.N.operator_norm.weight`, `.ffn_norm.weight` | [1024] |
| `vision_tower.layers.N.feed_forward.w1/w3.weight`, `.w2.weight` | [2560, 1024], [1024, 2560] |
| conv layers (0, 1, 3, 5, 7, 9, 11, 13): `conv.in_proj.weight` | [3072, 1024] |
| `conv.spatial_conv.weight` | [1024, 3, 3, 1] (depthwise, MLX layout) |
| `conv.out_proj.weight` | [1024, 1024] |
| attention layers (2, 4, 6, 8, 10, 12): `self_attn.q_proj`, `out_proj` | [1024, 1024] |
| `self_attn.k_proj`, `v_proj` | [512, 1024] (8 KV heads of 64) |
| `self_attn.q_layernorm.weight`, `k_layernorm.weight` | [64] |
| `multi_modal_projector.layer_norm.weight/bias` | [1024] |
| `multi_modal_projector.linear_1.weight/bias` | [4096, 4096], [4096] |
| `multi_modal_projector.linear_2.weight/bias` | [2048, 4096], [2048] |

138 tensors in all. No biases anywhere in the blocks.

## The forward pass

```text
h = embedding_norm(patchifier(rows))                  # RMSNorm eps 1e-5, rows are 1536 wide
for block in 14:
    x = operator_norm(h)
    conv block:   b, c, v = split(in_proj(x), 3)
                  h += out_proj(c * depthwise_conv3x3(b * v))     # zero padded, no bias
    attn block:   q, k, v = projections; q, k = per-head RMSNorm(q), RMSNorm(k)
                  rotate q, k with the 2-D rope; GQA attention, scale 64^-0.5, no mask
                  h += out_proj(attn)
    h += w2(silu(w1(ffn_norm(h))) * w3(ffn_norm(h)))
y = layer_norm(h)                                     # eps 1e-6, per token row
y = y.reshape(seq / 4, 4096)                          # four consecutive tokens
out = linear_2(gelu_erf(linear_1(y)))                 # [seq / 4, 2048]
```

- **Tokens are in merge-window order throughout**: `2 x 2` patch blocks
  row-major over the block grid, row-major inside a block. The projector's
  reshape reads four consecutive rows as one window, and the conv has to
  un-permute to the grid, convolve, and permute back. The kernel gathers
  neighbours straight from the window-ordered rows by computing each token's
  grid cell.
- **The rope is `vision_rope_freq_rows` at theta 1e6**, head_dim 64: row
  angles then column angles, with the kernel's half-split pairing standing for
  the reference's `[angles, angles]` concatenation. The qwen tower's function
  already took theta as an argument.
- **The conv's tap is `w[ch][dr + 1][dc + 1]`**, cross-correlation, no flip.
  `b * v` is formed per neighbour.
- **There is no position table**, no class token, no temporal loop for a
  still image (`grid_t = 1`, with the temporal pair baked into each patch row).
- **GELU is the exact (erf) form** in the projector. The blocks use SiLU.

## Preprocessing

`processor_config.json` carries it as an `image_processor` object:
patch 16, temporal 2, merge 2, mean and std 0.5, rescale 1/255, and bicubic resize,
`size = {shortest_edge: 65536, longest_edge: 16777216}` (pixel COUNTS). That is
the Qwen3-VL processor's contract, and `crates/vision-io` runs it unchanged.
The repo ships no `preprocessor_config.json`, so the installer writes the
`image_processor` object under that name (`catalog::stream::lfm3_preprocessor_json`).

**The patch row order differs.** The reference flattens each patch
`(C, T, P_h, P_w)`. `vision-io` emits `(T, P_h, P_w, C)` for every tower. The
patchifier's columns are permuted once at repack
(`permute_patchifier_columns`) instead of carrying a second layout.

The declared ceiling is exactly 65,536 patches. The qwen pixel-budget
resolver cannot shape this tower (no position table), so it declines to clamp.
The tower refuses a larger page by name (`MAX_PATCHES`).

## What intake does

`crates/repack/src/gemma4_checkpoint/vision_lfm3.rs`:

- `parse_lfm3_vl_vision_config` refuses any value the tower kind implies and
  the runtime does not store: rope theta 1e6 and `axial`, `norm_eps` 1e-5, a 3x3
  conv, the projector's GELU, bias, and eps 1e-6, hidden 4096, an unshuffle
  factor equal to the merge size, patch 16 / temporal 2 / merge 2, depth 14.
- Blocks are written with a role set per kind. A layer carrying both kinds'
  tensors, or neither, or part of one, is refused by name.
- **K and V are expanded to full width** (each KV head's 64 rows repeated
  twice, consecutively: head `h` reads KV head `h / 2`, as `repeat_kv` does).
  The vision attention kernel is multi-head, so the expansion costs 2 MiB per
  attention block and changes no arithmetic.
- Everything is narrowed BF16 to FP16 (exact in range, and the overflow refusal
  applies). Residents land under `vision.` (`patchifier.weight`,
  `embedding_norm.weight`, `projector.*`).

`crates/model-io::packed_experts_layout` now waives the "every blob has one
sub-tensor layout" check for `packed_vision/` only. The routed-expert check
stays, and so does the equal-blob-size check (each blob must equal the layer
stride).

## The trunk seam

The lfm3 flow gained the embed blit at both sites (`produce` and the chunked
prefill driver), the shape the llama and qwen flows already use: an image-pad
position's embedding is the tower's FP16 row, written to the shared
`scratch.x` before the pass commits. Positions stay sequential, so there is no
angle half to the seam. `lfm3_vision_inject_synthetic.rs` holds the two sites
to each other bit for bit across chunk splits that cut the image block.

## Gates

**Install.** `turbospark-model pull --repo LiquidAI/LFM3-VL-2B-MLX-8bit --alias lfm3-vl-2b-8bit
--out <dir>` detects the tower from the shard bytes, writes the 14 blocks
(338 MB of `packed_vision/blobs.bin`) and the preprocessor config. 2.1 GiB.

**Stage 2, tower against an independent reference.**
`crates/runtime/tests/lfm3_vl_vision_real.rs` dumps the Metal tower's stages,
and `scripts/lfm3_vl_tower_probe.py` runs a self-contained numpy implementation
written from the reference formulas over the checkpoint's ORIGINAL tensors
(no permutation, no expanded KV heads, no narrowing), in F32, in an
emulated-BF16 arm, and in an FP16-storage twin. It executes nothing from the
reference checkout.

| Input | worst stage cosine vs F32 | output rows vs F32 | emulated-BF16 reference's own agreement with F32 (floor) |
| --- | --- | --- | --- |
| 1024x1280 text page, 5,120 patches | 0.99999650 | 0.99999650 | 0.99972139 |
| 448x448 color image, 784 patches | 0.99989506 (block 13) | 0.99996600 | 0.99785429 |
| 448x448 text page, 784 patches | 0.99999731 | 0.99999731 | 0.99955365 |

The port agrees with F32 better than a BF16 reference agrees with itself, and
sits within 1e-5 of its FP16-storage twin (0.99999543 worst on the page). The
color image exists because the text pages are gray: every channel and
temporal-pair permutation gives the same answer on gray input, so only a color
image can tell the patchifier's column permutation from a wrong one.

**Mutation check.** Setting the tower's rope theta to 1e4 leaves the stages
before the first attention block untouched and drops the output rows to
0.7002 (worst stage 0.6207).

**Stage 4, the model reads the page.** Greedy (`--temperature 0 --top-k 1`),
`turbospark-check --messages-file ... --image page.png`, prompt "Transcribe the
first three lines of text in this image exactly." Ground truth from
`make_vision_test_page.py --size 1024 1280`:

```text
00012 | parity streaming streaming oracle fixture tensor kernel oracle manifest gradient kernel expert | 8316.48
00030 | prefill parity decode gradient embedding streaming gradient kernel gradient throughput | 9656.18
00048 | decode tensor quantize gradient throughput streaming attention streaming fixture | 4200.76
```

Output (64 new tokens, 1,306-token prompt):

```text
| 0012 | parity streaming streaming oracle fixture tensor kernel oracle manifest gradient kernel expert | 8316.4 |
| 0003 | prefill parity decode gradient embedding streaming gradient kernel gradient throughput | 9656.18 |
| 0008 | decode tensor quantize gradient throughput streaming attention streaming
```

Every word of all three lines is right, in order. The errors are the left
column's ids (`0012`, `0003`, `0008` for `00012`, `00030`, `00048`) and the
last digit of line 1's number. A model that never saw the picture cannot
produce those word sequences.

- **Both embed sites agree**: the same run with
  `TURBOSPARK_LFM3_CHUNKED_PREFILL=0` (the per-token walk) prints the same
  three lines.
- **Negative control**: the same prompt with no `--image` answers "I don't have
  access to images in this conversation."
- **Server**: `/v1/chat/completions` (an `image_url` data URL) and
  `/v1/messages` (a base64 `image` block) return the same transcription,
  with no `x-anyllm-degradation` header.
- Timing, not a benchmark: the tower for the full page runs in about 5.9 s of
  test wall time including open. The 1,306-token prefill (tower included) took
  14.0 s chunked and 18.9 s per token.

## Not done, and why

- **No greedy comparison against mlx-vlm.** Running the cloned reference code
  was declined by the session's execution policy, so the cross-engine claim
  rests on the independent numpy implementation above and on the page
  transcription against the generator's ground truth, not on a second
  engine's tokens.
- **The 4-bit build does not install.** `LiquidAI/LFM3-VL-2B-MLX-4bit` is
  affine group 32, and this port's kernels implement only 4- and 8-bit at group 64.
  The probe refuses it by name. The 8-bit build is the catalog row.
- **The builds' towers differ.** The 4-bit and 8-bit repos carry different
  tower bytes (up to 1% per weight on spot checks), so a reference must read
  the same repo as the install. The first probe run compared the 8-bit install
  to the 4-bit tower and read 0.9973, and the matching repo reads 0.9999965.
- **No sidecar.** A sidecar-attached trunk refuses an `lfm3_vl` tower by name,
  because `pairsWith` has no LFM3-VL record.
- **Still images only**, with no video. There is no quality gate (perplexity) or memory
  oracle for the VL install, and no speculation path exists for this family.
- **No pixel-budget clamp** for this tower (see Preprocessing).
- **Server and FFI memory admission** use the shared per-image limits. The
  FFI and Swift paths were not exercised here (they take the same
  `encode_image` / `set_prompt_vision` entry points the CLI and server used).
