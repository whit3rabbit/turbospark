# Gemma 4 vision: Phase 0

Facts about the Gemma 4 vision tower and how the trunk consumes it, read off
the checkpoint and the reference code before any bring-up. No code changed
for this page. The implementation plan lives with the work, not here.

## Sources

| Source | Pin |
| --- | --- |
| Checkpoint | `mlx-community/gemma-4-26b-a4b-it-4bit` @ `0d77464eeb233a2da68ebf9d7dc4edaac7db956d` (the same revision as the gated Gemma 4 text install) |
| Reference model | transformers `models/gemma4/modeling_gemma4.py`, `configuration_gemma4.py`, `processing_gemma4.py`, `image_processing_pil_gemma4.py` @ `28034545c35acac05793695bd5c878f66e6b544e` (the revision `docs/COREML.md` already pins for the EmbeddingGemma 2 image path) |
| Cross-engine reference | mlx-vlm `mlx_vlm/models/gemma4/vision.py` @ `4693cd0eaf022ae7ef61c6fdb55e17e6399879a7` |

Every fact below came from `config.json`, `processor_config.json`,
`model.safetensors.index.json`, the safetensors header of
`model-00003-of-00003.safetensors` (range read, no weights downloaded), or the
reference source at the pins above.

## Verdict

**This is a new tower, not the Qwen one at a different width.** The shapes
match the Qwen SigLIP tower almost exactly (depth 27, hidden 1152, 16 heads of
72, intermediate 4304), which makes it easy to misread as a reuse. The block
is Gemma-shaped instead: RMSNorm sandwich norms, per-head q/k norms, a
scale-free v norm, attention scale 1.0, a gated GELU MLP, no biases, and an
axial 2-D rope at theta 100. None of the Qwen tower's LayerNorm, fused qkv,
bias, or merger code applies.

**The trunk needs a mask change, not only an embedding blit.** With
`use_bidirectional_attention: "vision"`, sliding layers attend
bidirectionally inside each image span while full layers stay causal. A
sequential (one row per pass) prefill cannot express that, so an image span
must be prefilled in one batched pass with its K/V written before any of its
rows attend.

## Inventory

358 non-trunk tensors on this checkpoint: 355 under `vision_tower.` and 3
under `embed_vision.`. There is no audio tower on the 26B (`audio_config` is
null). `docs/MODEL_FAMILY.md` quotes "roughly 815 vision tensors" for Gemma 4.
That number does not match this checkpoint and should be corrected when that
table is next edited.

Per block, 13 tensors, all BF16, unquantized:

| Role | Shape |
| --- | --- |
| `input_layernorm.weight` | [1152] |
| `self_attn.q_proj.linear.weight` | [1152, 1152] |
| `self_attn.k_proj.linear.weight` | [1152, 1152] |
| `self_attn.v_proj.linear.weight` | [1152, 1152] |
| `self_attn.q_norm.weight` | [72] |
| `self_attn.k_norm.weight` | [72] |
| `self_attn.o_proj.linear.weight` | [1152, 1152] |
| `post_attention_layernorm.weight` | [1152] |
| `pre_feedforward_layernorm.weight` | [1152] |
| `mlp.gate_proj.linear.weight` | [4304, 1152] |
| `mlp.up_proj.linear.weight` | [4304, 1152] |
| `mlp.down_proj.linear.weight` | [1152, 4304] |
| `post_feedforward_layernorm.weight` | [1152] |

27 blocks x 13 = 351, plus 4 non-block tower tensors:

| Tensor | dtype | Shape |
| --- | --- | --- |
| `vision_tower.patch_embedder.input_proj.weight` | BF16 | [1152, 768] |
| `vision_tower.patch_embedder.position_embedding_table` | BF16 | [2, 10240, 1152] |
| `vision_tower.std_bias` | BF16 | [1152] |
| `vision_tower.std_scale` | BF16 | [1152] |
| `embed_vision.embedding_projection.{weight,scales,biases}` | U32 / BF16 / BF16 | [2816, 144] / [2816, 18] / [2816, 18] |

The projector is MLX affine 4-bit, group 64 (1152 / 18). The trunk's
quantized GEMV already reads that layout.

Sizes at FP16: a block is about 20.2 M parameters (38.5 MiB), the 27 blocks
about 1.01 GiB. The position table alone is 23.6 M parameters (45 MiB). The
Qwen block is about 15.2 M parameters, so the gated MLP makes each Gemma
block about a third larger.

## The tower, in order

```text
x    = pixels / 255                         # processor: rescale only, do_normalize false
x    = 2 * (x - 0.5)                        # in model code, not the processor
h    = x @ input_proj^T                     # [patches, 768] -> [patches, 1152], no bias
h   += table[0][pos_x] + table[1][pos_y]    # padding patches get 0
for blk in 0..27:
    a  = rmsnorm(h, input_layernorm)
    q  = rmsnorm_head(a @ Wq^T, q_norm)     # per head over 72
    k  = rmsnorm_head(a @ Wk^T, k_norm)
    v  = rmsnorm_head(a @ Wv^T)             # NO scale weight
    q, k = axial_rope(q, k, pos)            # theta 100
    o  = attention(q, k, v, scale = 1.0)    # bidirectional, padding masked
    h += rmsnorm(o @ Wo^T, post_attention_layernorm)
    m  = rmsnorm(h, pre_feedforward_layernorm)
    m  = (gelu_tanh(m @ Wg^T) * (m @ Wu^T)) @ Wd^T
    h += rmsnorm(m, post_feedforward_layernorm)
p    = avg_pool_3x3_by_position(h) * sqrt(1152)   # FP32
p    = (p - std_bias) * std_scale                  # FP32, then cast
t    = rmsnorm_noscale(p) @ embedding_projection^T  # [soft, 2816]
```

`rmsnorm` is `x * (mean(x^2) + eps)^-0.5 * w` with `eps` 1e-6, computed in
FP32. The weight multiplies directly. There is no `1 + w` offset.

## Traps, each one mutation from fluent wrong output

1. **Attention scale is 1.0.** The q/k norms carry the scale. A default
   `head_dim^-0.5` scale produces attention that is merely too flat.
2. **v is normed, without a weight.** `v_norm` is `Gemma4RMSNorm(with_scale=False)`
   and appears in no checkpoint, so nothing in the inventory reminds you it
   exists. mlx-vlm keeps it as `VisionRMSNormNoScale`.
3. **The rope type in `config.json` reads `"default"`.** The vision rotary
   class raises unless the type is `"axial"`, and `configuration_gemma4.py`
   sets `default_rope_type = "axial"` for the vision config. Treat the vision
   rope as axial at theta 100 regardless of the string. The text rope types
   (`default` on sliding, `proportional` 0.25 on full) are unrelated.
4. **Axial rope layout is H-H-W-W.** The head's 72 channels split into two
   halves of 36. Each half is rotated with `rotate_half` by one coordinate.
   The inverse frequencies use `head_dim // 2` as the spatial dimension.
   The first half turns by x (the patch's column), the second by y. Pinned
   by `crates/compute/tests/gemma4_vision_reference.rs` against mlx-vlm on a
   9 x 6 grid, where swapping them fails.
5. **The pooler's `sqrt(1152)` scale overflows FP16.** The reference keeps
   the pooled rows in FP32 through the `std_bias` subtraction and only casts
   after `std_scale`. An FP16 pooler is a silent saturation, not an error.
6. **Image soft tokens are not embed-scaled.** Text rows go through
   `Gemma4TextScaledWordEmbedding`, which multiplies by `sqrt(2816)`. Image
   rows are written by `masked_scatter` after that lookup, from the
   projector, with no scale. The blit must replace the scaled lookup, never
   feed the scaling.
7. **Padding patches are real rows.** The processor pads every image to
   `max_soft_tokens * 9` patches with position (-1, -1). The patch embedder
   zeroes their position term, the encoder masks them as keys, and the pooler
   drops them. A port that runs only the real patches must produce the same
   pooled rows, which the parity gate should check on a non-square image.

## The trunk side

- **Placeholders.** `<|image|>` (`image_token_id` 258880) in the chat
  template expands to `boi` (255999), N copies of the image token, then `eoi`
  (258882). N is the image's soft-token count.
- **Positions.** Plain 1-D positions. No mRoPE, no rope delta after the image.
- **Per-layer inputs.** `hidden_size_per_layer_input` is 0 on the 26B, so no
  PLE lookup happens at image positions. Smaller Gemma 4 checkpoints with
  PLE substitute the pad embedding at multimodal positions before the PLE
  lookup, which matters only if one of those is brought up.
- **The mask.** `text_config.use_bidirectional_attention` is `"vision"`.
  Image tokens are grouped into blocks of contiguous vision positions. On
  sliding layers the mask is `sliding_window AND (causal OR same_block)`. On
  full layers it is causal only. Gemma 3 made every layer bidirectional, so a
  Gemma 3 port's mask is wrong here.
- **What that means for this engine.**
  `crates/runtime/src/families/gemma4/attn_batch.rs` attends one row at a
  time with `seq_len = position + 1`, which is causal by construction. A
  sliding layer at an image row needs its span extended to the end of the
  row's image block, with every row's K/V for that block written first. The
  sliding window is 1024 and an image block can be 1120 tokens at the
  largest budget, so ring capacity has to cover the look-ahead as well as the
  window. Ring capacity is part of correctness here.
- **Speculation and decode.** The bidirectional overlay only touches rows
  inside an image block, which all sit in the prompt, so decode is
  unchanged. Speculative verify past an image must still be refused until
  tested, as it is for Qwen vision.

## Preprocessing

`Gemma4ImageProcessor`: RGB, resize with aspect ratio kept so the patch count
fits `max_soft_tokens * 9` and both sides are multiples of 48 px, PIL bicubic
(`resample` 3), rescale by 1/255, no normalize, 16 px patches row-major. The
default budget is 280 soft tokens (2520 patches). Supported budgets are 70,
140, 280, 560, and 1120.

`crates/runtime/src/coreml/embedding_gemma2_media.rs` already ports this for
EmbeddingGemma 2, with `gemma4_resize_dims` and the test
`resize_matches_transformers_examples`. It lives behind the Core ML module,
so a portable copy belongs in `crates/vision-io` for the LM path.

## Cost at the default budget

2520 patch rows per image, one bidirectional segment. Attention is
2520 x 2520 per head at the default budget and 10080 x 10080 at 1120 soft
tokens. The tower is streamed or resident at about 1.01 GiB FP16. Neither
number has been measured on this engine.

## Implementation status (2026-11-01)

Built and held by synthetic tests, and run end to end on the real checkpoint (below):

- Intake: `visionTower: "gemma4"` in the manifest, the tower and the
  dequantized projector ingested (`repack/src/gemma4_checkpoint/vision_gemma4.rs`).
  Intake refuses any `use_bidirectional_attention` other than `"vision"`,
  because the runtime implements that one mask.
- Tower: `runtime/src/vision/gemma4.rs`, reached through
  `RealForwardRunner::encode_image_gemma4`.
- Trunk: `runtime/src/families/gemma4/vision_prefill.rs`, reached through
  `set_prompt_vision_gemma4` and `prefill_chunk`. Each span runs whole,
  with all rows' K/V written before any row attends; sliding layers attend
  to the span's end, full layers stay causal. The sequential path,
  `--kv-bits`, `TURBOSPARK_BATCHED_GEMV` and `TURBOSPARK_ROUTED_BATCH`
  refuse an image span by name.
- The sliding ring on an install with a Gemma 4 tower holds
  `window + 280` (the default budget); a larger span is refused up front.

- Front ends: the server (`/v1/chat/completions` and `/v1/messages`) and
  the FFI (`ts_generate` and the in-process server) expand each rendered
  `<|image|>` to `boi`, one placeholder per soft token, `eoi`
  (`turbospark_vision_io::gemma4_expand_image_placeholders`), encode, inject
  and prefill through the chunked driver, at the default 280-token budget.
  Both read the "TURBOSPARK 42" image on the real install
  (`real_backend_reads_a_gemma4_image_sent_over_both_endpoints`,
  `crates/ffi/tests/gemma4_vision_real.rs`). The CLI (`turbospark-check
  --image PATH --messages-file FILE`) takes the same path
  (`crates/cli/src/generate/vision.rs::attach_gemma4`) and read it on the
  real install (2026-11-01: "TURBOSPARK 42", stop at end of turn, 295
  prompt tokens). No automated test holds the CLI path; it shares the
  expansion and injection helpers the server and FFI tests cover.
- Swift: the package's `RealModelVisionTests` passes on the install
  (`TURBOSPARK_TEST_MODEL` and `TURBOSPARK_TEST_IMAGE`): 290 prompt tokens
  with the image against 22 without. The app gates image input on
  `vision.active`, which the FFI open now reports for this tower.

Not built: sidecars for this tower, and soft-token budgets above the default (the
install's KV ring is sized for 280).

## FP16 sensitivity of the attention (measured 2026-11-01)

The real checkpoint's q and k norm weights are constant within each block,
about 1.0 for q and 0.45 for k (range-read from blocks 0, 13 and 26). With
attention scale 1.0 on RMS-normed heads of 72, that puts the attention logits
near +-32. At that scale a one-ulp FP16 difference in q or k moves the
softmax by a percent or two.

Measured on one block at the real widths with those norm weights
(`vision::gemma4` tests): the Metal block is 0.109 from an F32 reference at a
residual magnitude of 17.9, and a CPU block that rounds to FP16 wherever the
Metal path stores a buffer is 0.092 from F32 and 0.094 from the GPU. Each
kernel matches the CPU to about one ulp on identical inputs. So the tower is
as accurate as FP16 storage allows, and FP16 storage is what limits it.

The reference itself runs in BF16, which has fewer mantissa bits than FP16.

## Real-checkpoint tower parity (2026-11-01)

Measured on the real tower weights (range-read, no install) against mlx-vlm,
one deterministic 480 x 640 image resized to 912 x 672: 2,394 patches, 266
soft tokens. Cosine to mlx-vlm's FLOAT32 run, with mlx-vlm's own BF16 run
beside it:

| Stage | Metal FP16 vs F32 | mlx-vlm BF16 vs F32 |
| --- | --- | --- |
| patch embed | 0.99999996 | 0.99999562 |
| block 0 | 0.99999990 | 0.99999142 |
| block 13 | 0.99999982 | 0.99998779 |
| block 26 | 0.99997633 | 0.99806299 |
| pooled, standardized | 0.99998262 | 0.99849413 |
| projector output | 0.99998145 | 0.99837135 |

So the FP16 tower is about 100 times closer to the F32 truth than the
reference's normal BF16 execution, at every stage. The sensitivity measured
above on random weights does not compound on the real ones. Block 26's
activations reach an absmax of 2,367, well inside FP16. Swapping the rope's
x and y drops the projector output to cosine 0.55, so the gate
discriminates.

191 of the tower's BF16 tensors lose bits converting to FP16, all in FP16's
subnormal range (no overflow); the parity above is measured with those
conversions in place.

One image took 7.0 s through the tower and projector on the scalar vision
matmul, and 0.7 s with the matrix-unit GEMM and flash attention (PR 242,
`docs/VISION.md`), at the same parity. `TURBOSPARK_VISION_MMA=0` and
`TURBOSPARK_VISION_ATTN_MMA=0` restore the scalar kernels.

Reproduce with the commands in `docs/TESTING.md` (Gemma 4 tower against
mlx-vlm).

## Open before bring-up

- Whether other Gemma 4 checkpoints set `use_clipped_linears` (this one does
  not, and carries no clip buffers).

## Real-checkpoint image prompt (2026-11-01)

The full install (`mlx-community/gemma-4-26b-a4b-it-4bit@0d77464e`, repacked
with its tower through the streamed writer) against mlx-vlm on one prompt:
a rendered 640 x 480 image reading "TURBOSPARK 42" and the question "What
text is written in this image?". 295 prompt tokens, 266 of them image
placeholders. The runtime answers `TURBOSPARK 42` then end-of-turn, as
mlx-vlm does.

**The two references disagree on the mask.** transformers makes only
SLIDING layers bidirectional inside an image span ("Gemma 4 explicitly
disables bidirectional attention on global attention layers"); mlx-vlm
(vendored) overlays every layer. The runtime implements the transformers
rule. The dump script runs mlx-vlm under four rules (sliding only, all
layers, none, full only) by patching its `_make_masks`, and a probe of the
masks each patch produces confirmed them.

Which rule the runtime matches, measured two ways:

- Image-row hidden states per layer (one minus cosine). Against the
  transformers rule the runtime tracks the noise floor (the runtime forced
  causal against mlx-vlm causal) at every layer: 1.4e-3 against 2.2e-3 at
  layer 0, 0.221 against 0.222 at layer 26. Against "no overlay" it is two
  to five times further at every layer.
- Next-token logits at the last prompt position and 24 decode steps,
  teacher-forced on one shared token stream. Median KL: transformers rule
  3.1e-7, all layers 1.2e-6, full only 2.2e-6, none 2.2e-6.

The MEAN KL is not usable here and the gate does not assert it: one
near-tie position (where the rules' own greedy paths part) reaches KL 1 to
2.5 against every reference, and mlx-vlm's cached decode agrees with its
own one-pass forward only to a mean KL of 5e-3 to 2e-2 on this 4-bit model.
Forcing the runtime's sliding layers causal makes "no overlay" the closest
reference and fails the gate.

The repack of the real checkpoint found one bug the synthetic tests could
not: the walk hands a quantized tensor's reader only the `.weight`, so the
projector's `.scales` and `.biases` must be fetched from the shards. The
synthetic ingest test now hands the reader exactly what the walk does.

Timing: the tower took 6.9 s for 2,394 patches on the scalar vision
matmul. With the matrix-unit GEMM and flash vision attention (PR 242) it
took 0.7 s from in-memory weights and 1.1 s on the install, which streams
its blocks. Both gates above pass unchanged on those kernels. The 295-token
prefill took 3.8 to 4.7 s.

A rerun of `gemma4_vision_real_trunk` on the merged kernels (one run per
arm, host under swap, so the times are indicative) passed in both arms with
identical greedy tokens: tower 1.6 s with the default kernels and 5.2 s
with `TURBOSPARK_VISION_MMA=0` (GEMM off, attention still on the matrix
units). Median KL against the transformers rule was 2.4e-7 and 5.0e-7.

If transformers is the reference of record, mlx-vlm's all-layer overlay is
an upstream difference worth reporting.
