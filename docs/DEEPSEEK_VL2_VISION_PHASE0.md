# DeepSeek-VL2 vision: Phase 0

Facts about DeepSeek-VL2-tiny, read off the real checkpoints and the
reference code, and a feasibility verdict for the `Deepseek2` family. No
engine code changed for this page, and no weights were downloaded or run:
the evidence is `config.json`, `processor_config.json`, `chat_template.json`,
`tokenizer.json`, `model.safetensors.index.json`, a ranged safetensors header
read of the MLX shard, and the mlx-vlm source. Read on 2026-11-01.

| what | where | revision |
| --- | --- | --- |
| MLX checkpoint | `mlx-community/deepseek-vl2-tiny-4bit` | `4772de7c` (prefix of the sha the Hub API returned) |
| original checkpoint | `deepseek-ai/deepseek-vl2-tiny` | `66c54660` (same) |
| reference code | mlx-vlm `mlx_vlm/models/deepseek_vl_v2/` | shallow clone at `/Volumes/T7/refs/mlx-vlm`, 2026-11-01 |

Larger siblings (`deepseek-vl2-small`, `deepseek-vl2`) were not read.

## Status: DeepSeek-VL2-tiny takes images end to end

Everything below the "Verdict" heading is the Phase 0 reading and is kept as
written; the Verdict's blockers were then cleared. Branch
`vision/deepseek-vl2-trunk`.

**Trunk (cheapest route, no new family).** The `deepseek2` flow gained a
plain-attention arm, selected by an inactive `ArchConfig::mla`
(`RealDeepseek2State::mha`, `encode_mha_attention_block`: q/k/v GEMV, NeoX
half-split rope over the full 128-wide head at theta 10000, the split-KV
attention kernel with scale `head_dim^-0.5`). The flow's MoE half already
matched the model (softmax over all 64 experts, top-6, no renormalization,
`routed_scaling_factor` 1, INT8-forced router, fused shared expert, dense
lead layer), so nothing there changed. Intake reads the MLX 4-bit checkpoint
directly: `parse_deepseek_vl2_config`, a header canonicalization
(`shared_experts` to `shared_expert`), `switch_mlp.*` routed stacks, F16
scales and biases narrowed to BF16, the F16 router quantized to INT8. The
template does not write the BOS the tokenizer's `add_bos_token` implies;
`MfTokenizer::apply_chat_template` now opens a V2-dialect prompt with it
(`"The capital of France is Paris."` without it, `"Paris."` with it).

**Trunk gate.** Teacher-forced argmax over 17 ids agrees 17/17 with an
independent numpy forward (`scripts/deepseek_vl2_trunk_reference.py`); the
last-position logits have cosine 0.999747, KL 0.00020 nats, top-5 overlap 5/5.
A 802-token text prompt ("secret word" needle) answers correctly, so the
sequential path holds at length.

**Tower.** `VisionTowerKind::SigLip` (persisted `siglip`). The SigLIP-so400m
tower and the 2x2-unfold projector run in `runtime/src/vision/siglip.rs`; all
tiles of an image run in one pass per block (attention dispatched per tile at
a byte offset), so each 30 MiB block is read once per image. The final norm
uses eps 1e-5 and the blocks 1e-6. The 4-bit planes are dequantized to FP16 at
intake (`mlp.fc2` is F16: its 4304-wide input has no valid group size), the
sidecar is 802.5 MiB, `attn_pool` is dropped. `pull-vision` accepts the
repo's `processor_config.json` in place of `preprocessor_config.json` and
writes it beside the tower.

**Preprocessing** (`crates/vision-io/src/deepseek_vl2.rs`): `select_best_resolution`,
a transcription of PIL `ImageOps.pad` (bicubic `contain` with Python's
half-to-even `round`, centered paste, fill `int(mean * 255)`), a global view
plus row-major local tiles, patch rows in `(14, 14, 3)` order. A 384 px tile
reads 27 x 27 patches: 384 is not a multiple of 14, the stride-14 conv drops
the last 6 pixels. Tiling is off past two images. The `Pillow` rounding
convention is the 10.x one (`round`, not `int`); an older Pillow can differ by
one padded row.

**Injection.** Positions are plain, so `PromptVision::flat` applies; the
`deepseek2` sequential `produce` path gained the embedding blit it lacked
(without it the 617 image positions carried the untrained `<image>` row and
the model produced fluent garbage, which is how the gap was found). The V2
template writes `{{message['content']}}`, so an image part is flattened to the
reference's `<image>\n` before rendering.

**Tower gate.** `scripts/deepseek_vl2_tower_reference.py` is a numpy
transcription of the tower and projector written from the mlx-vlm source. It
was **read, never executed**: no mlx-vlm code ran. On the real shapes image
(3 tiles, first 2 compared, identical pixels both sides):

| stage | cosine | relative L2 |
| --- | --- | --- |
| embed | 0.999999 | 0.00029 |
| block 0 | 0.999999 | 0.00054 |
| block 26 | 0.999997 | 0.00345 |
| final norm | 0.999978 | 0.00672 |
| projector | 0.999986 | 0.00532 |

Cosine alone is blunt here (the residual carries outlier channels near 6300),
so a mutation check matters: making attention span all tiles instead of one
drops the final-norm cosine to 0.557 and the projector to 0.647. A tanh-GELU
to erf-GELU swap in the blocks is NOT separable at this bound (cosine
0.99995), as expected for two functions that differ by under 5e-4; the unit
tests and the reference's own formula pin it.

**Image gate** (real checkpoint, release build, `turbospark-check`):

- `shapes.png` + "List each shape in this image with its color." -> "Red, Blue,
  Green". With a free-form question: "a red square, a blue circle, and a green
  triangle." The same questions with NO image: "Circle - Red, Triangle - Blue,
  Square - Green, Pentagon - Purple..." (a hallucination, and different).
- `page-small.png` + "Read the text in this image, exactly." -> first line
  `00012 | parity streaming streaming oracle fixture`, then the following lines
  in the same format. Without the image the model writes a letter.
- `turbospark-server` on both `/v1/chat/completions` and `/v1/messages`, same
  answers for both images, and the no-image control hallucinates.

**Not verified.** No benchmark was run (the directive said to defer it). The
tower is compared to a numpy transcription of the reference, not to mlx-vlm
output, so a misreading of the reference's tiling would be shared. The page
text after the first line was not diffed against the source. Multi-image
(more than two) requests, the FFI paths and the macOS app were compiled and
reasoned about but not driven. `deepseek-vl2-small` and `deepseek-vl2` were
not read.

## Verdict (Phase 0, historical)

**BLOCKED ON THE TRUNK, NOT ON THE TOWER. Stop after Phase 0.** The vision
half is small: the tower is a plain SigLIP-so400m that the existing FP16
kernels cover, the projector is two matmuls, and the tiling is a host-side
layout. The trunk is the problem. `deepseek2` cannot run this model today,
and making it run is a new-family-sized job (`docs/NEW_MODEL.md`) that has to
land and pass a text quality gate before an image can mean anything.

1. **The trunk is not MLA.** VL2-tiny declares `use_mla: false`,
   `kv_lora_rank: null`, `qk_nope_head_dim: 0`, `qk_rope_head_dim: 0`,
   `v_head_dim: 0`: plain multi-head attention (10 heads, 10 KV heads, head
   dim 128) with MoE feed-forwards. The reference loader switches its
   attention class on `qk_nope_head_dim == 0` (`TextConfig.__post_init__`
   sets `attn_type = "LlamaAttention"`). The `deepseek2` flow is MLA end to
   end: `RealDeepseek2State::build` refuses an install whose `arch.mla` is
   inactive, requires `kv_lora_rank == 512` and `q_lora_rank == 0`, sizes its
   KV cache in MLA latent rows, and every layer must carry mask 5.
2. **The checkpoint is MLX affine, not GGUF.** `deepseek2` has only a GGUF
   intake (V2-Lite Q8_0). `crates/catalog/src/stream.rs` has no `Deepseek2`
   arm in either the config parse or the writer match, and
   `crates/catalog/src/probe/` has none, so the probe would not call this
   repo runnable. The routed experts are stacked MLX tensors
   (`switch_mlp.*_proj`, `[64, 896, 160]` U32 plus scales and biases). The
   routed-expert kernels do include the vendored INT4-affine pair
   (`RoutedBlobLayout::Affine`, `encode_moe_phase1_any`), so the compute
   exists; the deepseek2 state and dispatch have only been exercised with the
   GGUF layouts (`gpu::PHASE2_FIXED_SLOTS` is a GGUF constraint it checks).
3. **The registry knows one shape per family.** `arch_baselines` maps
   `Deepseek2` to `deepseek_v2_lite_16b()` (27 layers, hidden 2048), and
   `manifest::is_production_arch` accepts an install only if its
   `(num_layers, hidden_size)` matches a registered baseline. VL2-tiny is 12
   layers at hidden 1280, so it needs its own baseline and the validation
   fields for a non-MLA trunk.
4. **The prompt format has no template renderer here.** The checkpoint ships
   no Jinja template that the repo's chat path could use as is (see
   "Chat template"); the MLX repo carries a short `chat_template.json`.

### What the trunk work would be

All of this precedes any image work, and each item needs its own gate:

- MLX safetensors intake for the trunk: config parser (`language_config`
  nesting, `language_model.` prefix), tensor classification, a streamed
  writer packing the stacked affine experts into `packed_experts`, the
  dense layer 0 FFN, the 2-expert fused shared FFN (1792 rows), the F16
  router (`mlp.gate.weight`, `[64, 1280]`), and the untied 4-bit
  `lm_head`/embedding. Closest precedents: the Qwen GDN MoE and Qwen4Exp
  writers.
- A baseline and `ArchConfig` for the 12-layer shape, with the MLA block
  empty, and an MHA attention path in the `deepseek2` flow (or the llama
  flow's dense attention with this flow's MoE tail). Plain neox RoPE over the
  full 128-wide head, theta read off the config (the tiny config carries no
  `rope_scaling`; the dense reference applies `nn.RoPE(head_dim,
  traditional=false, base=rope_theta)`).
- The router semantics are already the flow's: softmax over ALL 64 experts,
  top 6, no renormalization (`norm_topk_prob` is absent; the reference
  gate returns the raw softmax values times `routed_scaling_factor`, 1.0).
- A DeepSeek chat dialect renderer and a text-only quality and KL gate
  against mlx-vlm on the same MLX bytes.

Rough size: comparable to a full family bring-up (the Flint and LFM3 pages
record what that costs), before the vision side. The vision side is
roughly the Qwen4Exp reuse commit plus a new tower kind and a tiling
preprocessor, so between that commit (24 files) and the Gemma 4 commit
(66 files, 7.3k lines).

## Trunk configuration (against the flow's assumptions)

From `language_config`, identical in both repos:

| key | value | V2-Lite (what the flow was built for) |
| --- | --- | --- |
| `hidden_size` | 1280 | 2048 |
| `num_hidden_layers` | 12 | 27 |
| `first_k_dense_replace` | 1 | 1 |
| `num_attention_heads` / `num_key_value_heads` | 10 / 10 | 16 / 16 |
| attention | MHA, head dim 128, no biases | MLA, nope 128 + rope 64, kv_lora 512 |
| `intermediate_size` (dense layer 0) | 6848 | 10944 |
| `moe_intermediate_size` | 896 | 1408 |
| `n_routed_experts` / `num_experts_per_tok` | 64 / 6 | 64 / 6 |
| `n_shared_experts` | 2 (fused width 1792) | 2 (fused width 2816) |
| `topk_method` / `n_group` / `topk_group` | greedy / 1 / 1 | greedy |
| `vocab_size` | 129280 | 102400 |
| `max_position_embeddings` | 4096 | 163840 (YaRN) |
| `tie_word_embeddings` | untied (`lm_head` present) | untied |
| `bos` / `eos` token ids | 0 / 1 | |

The router, the shared-expert shape and the 64-expert/top-6 geometry match
what `deepseek2/moe.rs` already does. The attention and the dimensions do
not.

## Tensor inventory (MLX 4-bit, 915 tensors, one shard)

Affine 4-bit, group size 64. Names after the reference's `sanitize`:

- `language_model.model.*`: embedding (4-bit, `[129280, 160]` U32 +
  `[129280, 20]` scales), 12 layers of `self_attn.{q,k,v,o}_proj` (4-bit,
  `[1280, 160]`), norms, layer 0 dense `mlp.{gate,up,down}_proj`, layers
  1..11 `mlp.gate.weight` (F16 `[64, 1280]`, not quantized),
  `mlp.switch_mlp.{gate,up,down}_proj` (`[64, 896, 160]` U32),
  `mlp.shared_experts.*` (`[1792, 160]`); untied 4-bit `lm_head`.
- `vision.vision_tower.*`: `patch_embed.proj.weight` `[1152, 14, 14, 3]` F16
  (MLX conv layout), `patch_embed.proj.bias`, `pos_embed` `[1, 729, 1152]` F16,
  27 blocks, final `norm`, and an `attn_pool` head the loader ignores.
- `projector.layers.0` (`[1280, 576]` U32 = 4608 inputs, 4-bit) and
  `projector.layers.2` (`[1280, 160]` U32); the GELU between them has no
  parameters.
- `image_newline` and `view_separator`, both F16 `[1280]` (the original
  repo spells the second `view_seperator`).

**The tower is partly quantized in the MLX build**, the Qwen4Exp situation
(`docs/QWEN4EXP_VISION_PHASE0.md`): per block, `attn.qkv`, `attn.proj` and
`mlp.fc1` carry `scales`/`biases` (the qkv weight reads `[3456, 144]` U32),
while `mlp.fc2` is F16 `[1152, 4304]` because 4304 does not divide by 64.
The projector's two linears are quantized too. The original BF16 repo has
none of that, so a sidecar built from `deepseek-ai/deepseek-vl2-tiny` would
avoid the dequantize step, and repack would need to accept the
`vision.`-prefixed (not `vision_tower.`) spelling of that repo. Its
`language.*` tensors are per-expert (`experts.N.gate_proj`, 2582 tensors),
not stacked.

## The tower

`vision_config`: `layers 27`, `width 1152`, `siglip_so400m_patch14_384`,
`mlp_ratio 3.7362` (MLP width 4304), `patch_size 14`, 16 heads of 72,
`layer_norm_eps 1e-6` (mlx-vlm `VisionConfig` defaults).

Forward pass, read off `vision.py` (`ignore_head=True`, so the attention-pool
head is never run):

```text
h = conv14x14(image 384x384x3) + bias       # 27x27 = 729 positions
h = h + pos_embed                           # absolute, no interpolation, no CLS
for blk in 0..27:
    h = h + attn(norm1(h))                  # fused qkv with bias, full bidirectional
    h = h + mlp(norm2(h))                   # fc1, GELU tanh ("precise"), fc2
out = LayerNorm(h)                          # the final tower norm IS applied
```

Every operation has an FP16 kernel in `crates/gpu/src/vision.rs`
(`encode_vision_layer_norm`, `encode_vision_matmul`,
`encode_vision_attention`, `encode_vision_gelu` with `GeluKind::Tanh`,
`encode_vision_residual_add`). There is no rope, so `encode_vision_rope_2d`
is not used. The patch embedding is the same reshape trick the Qwen tower
uses (a conv with stride equal to kernel is a matmul over flattened
patches). The row order must be checked against the MLX layout
`(kh, kw, C)`, which differs from Qwen's `(T, P, P, C)`: a mismatch keeps
every shape and reads a different image. 729 position rows come from a
27x27 conv output of a 384 input at patch 14 (384 / 14 floors to 27).

This needs a new `VisionTowerKind` (persisted e.g. `siglip`), because the
shared fields change meaning exactly as they did for Gemma 4: no merge, no
rope, an absolute table of 729 rows, and a projector instead of a merger.

## The projector (`downsample_mlp_gelu`, depth 2, ratio 2)

```text
x: [729, 1152] -> [27, 27, 1152] -> zero-pad to 28x28
   -> 2x2 unfold, windows row-major, channel-major inside each window -> [196, 4608]
   -> Linear(4608 -> 1280) -> GELU(erf) -> Linear(1280 -> 1280)   # layers.0, layers.2
```

The reference's manual unfold concatenates each 2x2 window as
`reshape(patch, (bs, -1))` of a `(C, 2, 2)` slice, so channels are the slow
axis inside a window. The pad is one zero row and one zero column (27 is odd).
The GELU is `nn.GELU()` (exact erf). One tile produces 14 x 14 = 196 rows of
width 1280.

## Tiling and the token layout

`processor_config.json`: `candidate_resolutions` (23 entries, multiples of
384 up to 3456 on one side, e.g. `[384,384]`, `[768,384]`, `[1152,1152]`),
`image_mean`/`image_std` 0.5/0.5, `patch_size 14`, `downsample_ratio 2`,
`image_token "<image>"`.

1. **Choose the grid**: `select_best_resolution((w, h), candidates)` picks
   the candidate maximizing the downscaled-to-fit pixel count (capped at the
   source's), tie-broken by least waste. This is `int()` truncation in
   Python; port it exactly.
2. **Views**: a GLOBAL view (`ImageOps.pad` to 384x384, fill color
   `int(0.5 * 255) = 127`) and a LOCAL canvas (`ImageOps.pad` to the chosen
   `(W, H)`, same fill), cropped into `(H/384) * (W/384)` tiles in row-major
   order. `ImageOps.pad` resizes to fit (bicubic in PIL's default) and
   centers. Cropping is used when at most two images are in the request.
3. **Normalize** `(x/255 - 0.5) / 0.5`, channel-last into the conv.
4. **Tower and projector per tile**: `1 + th * tw` tiles in, 196 rows each.
5. **Row layout per image** (`tile_tag "2D"`, `global_view_pos "head"`):
   the global view as a 14 x 14 grid with an `image_newline` row appended to
   each grid row (14 x 15 = 210 rows), then ONE `view_separator` row, then
   the local tiles stitched into a `(th * 14) x (tw * 14)` grid (tiles are
   transposed from `[th, tw, h, w]` to `[th, h, tw, w]`) with an
   `image_newline` row after each grid row: `(th * 14) * (tw * 14 + 1)` rows.
   Total rows: `210 + 1 + th * 14 * (tw * 14 + 1)`, for example 1,023 for a
   2 x 2 grid.
6. **Tokens**: the prompt carries that many `<image>` ids in one run. The
   real id is **128815** (read from `tokenizer.json`). The mlx-vlm config
   default `image_token_index: 100015` is not this model's id and must not be
   copied. The rows are written over the embedding table lookup at those
   positions (`input_embeds[idx, image_indices] = features`); positions are
   plain 1-D, so there is no mRoPE and the existing degenerate-position path
   applies.

The 14 x 14 grid is `ceil((384 // 14) / 2)`, the processor's own expression.

## Chat template

The repo ships no Jinja file. The MLX build has `chat_template.json`:

```text
{% for m in messages %}{% if user %}<|User|>:{% elif assistant %}<|Assistant|>{% endif %} {{content}}\n\n{% endfor %}
{% if add_generation_prompt %}<|Assistant|>:{% endif %}
```

Special ids: `<|User|>` 128821, `<|Assistant|>` 128822, `<image>` 128815,
`<|ref|>` 128816, `<|det|>` 128818, `<|grounding|>` 128820. The processor
also has its own `conversation.py` dialect (`sft_format: "deepseek"`), which
is the authority if the two differ; this page did not diff them. The
tokenizer is BPE with a byte-level pre-tokenizer sequence and 128,000 base
entries plus added tokens (129,280 rows in the embedding). Grounded outputs
use the `<|ref|>`/`<|det|>` markers; no box parsing is in scope.

## Reuse map, once the trunk exists

| piece | reuse |
| --- | --- |
| tower kernels | `gpu::encode_vision_*`, unchanged |
| streamed block loop, slots, scratch, budget | `runtime/src/vision/` (the Gemma 4 tower is the precedent for a second kind) |
| sidecar format, `pull-vision`, `--vision-sidecar auto` | unchanged, plus a pairing record for the new family and hidden 1280 |
| embedding blit | the `prompt_vision` map already used by the llama and qwen flows; the deepseek2 flow has none |
| preprocessing | new in `crates/vision-io`: candidate selection, pad, tile, normalize (Gemma 4's `gemma4.rs` is the shape) |
| gates | stage-2 parity against mlx-vlm on the same bytes, then the stage-4 logit gate and a content assertion on `scripts/make_vision_test_page.py` output |

## Not done, and not claimed

- No tower, projector or trunk number was measured. Nothing here has run.
- The original repo's header was not read (its shard file names differ from
  `model.safetensors`); its inventory above is from the index file only.
- `deepseek-vl2-small` and `deepseek-vl2` are untouched. Their language
  configs may use MLA, which would change the trunk verdict for them (the
  reference's attention switch is per config), and that is worth reading
  before choosing which size to bring up first.
- The two prompt dialects were not diffed.
