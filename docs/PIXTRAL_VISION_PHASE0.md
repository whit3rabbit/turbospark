# Pixtral vision: Phase 0 and bring-up record

Facts about the Pixtral-12B vision tower and how the `llama` trunk consumes it,
read off the checkpoints and the reference source, plus the gates that were run.
The mechanics shared with the other towers are in [VISION.md](VISION.md); this
page is what is specific to `VisionTowerKind::Pixtral`. The route and the
traps shared with LLaVA are in [LLAVA_VISION_PHASE0.md](LLAVA_VISION_PHASE0.md).

## Sources

| What | Where | Revision |
| --- | --- | --- |
| Tower and projector | `mlx-community/pixtral-12b-bf16` | `378cf65efd79647fc324a079c141fd92cb29916e` |
| Trunk | `ggml-org/pixtral-12b-GGUF`, `pixtral-12b-Q4_K_M.gguf` (7.0 GiB) | `cba1ea4420bc2b4f15f50fdec59e30769880a63c` |
| Tokenizer, template, processor config | the bf16 repo above (the GGUF repo ships none) | same |
| Reference source (read only) | mlx-vlm `models/pixtral/{vision,pixtral,processing_pixtral,image_processing_pixtral}.py` and `models/mistral3/` | `main` as cloned 2026-11-01 |

The reference source was read and transcribed. It was **not executed**: the
parity reference is a numpy transcription (`scripts/pixtral_tower_reference.py`)
and no number on this page is a comparison against mlx-vlm run on the same
bytes. See "What is and is not verified".

`mlx-community/pixtral-12b-4bit` (the obvious small download) **quantizes the
tower**: every projection in it is a packed 4-bit plane. The reader refuses a
packed tensor by name and points at the bf16 conversion, which is the source
this port ingests.

## The route: GGUF trunk plus a sidecar tower

Same as LLaVA, for the same reason. The `llama` family has no safetensors
intake; its installs come from GGUF. Pixtral's language model is a fine-tune of
Mistral-Nemo-12B (40 layers, hidden 5120, 32 heads of 128 over 8 KV heads,
vocab 131,072, rope theta 1e9), so the GGUF is Pixtral's own trunk and the tower
is a `pull-vision` sidecar of the bf16 repo.

The trunk shape is **not a registered baseline**, and nothing needed one:
`is_production_arch` only decides whether quantization metadata is mandatory,
and `arch_from_gguf` reads the shape off the file. The probe reports it as a
dense `llama` of 40 layers and 5120 hidden, all block types executable (Q4_K,
Q6_K, F32).

The trunk's tokenizer comes from the bf16 repo (`--sidecar-repo`), which also
supplies the chat template as `chat_template.json`.

**The sidecar pairs on `(family, hidden_size)` only**, as LLaVA's does: a
5120-wide `llama` sidecar attaches to any other 5120-wide `llama` trunk, and
`--vision-sidecar auto` does it silently. Name the sidecar explicitly.

## Inventory

222 tensors outside the language model in the bf16 repo:

| Group | Tensors | Notes |
| --- | --- | --- |
| `vision_tower.vision_model.patch_conv.weight` | 1 | `[1024, 16, 16, 3]`, MLX layout `(out, kh, kw, C)`, no bias |
| `vision_tower.vision_model.ln_pre.weight` | 1 | RMSNorm weight |
| `vision_tower.vision_model.transformer.layers.N.*` | 24 x 9 | `attention_norm`, `attention.{q,k,v,o}_proj`, `ffn_norm`, `feed_forward.{gate,up,down}_proj`, **no biases** |
| `multi_modal_projector.linear_{1,2}.{weight,bias}` | 4 | `[5120, 1024]`, `[5120, 5120]`, with biases |

The tower is in shard 1 and the projector in shard 5. The original
`mistral-community` HF export spells the tower `vision_tower.*` without the
`vision_model.` level, and the reader takes both. The tower is 0.4B parameters, 830
MiB at FP16.

`vision_config` in the published `config.json` states only `head_dim` (64),
`intermediate_size` (4096), `image_size` (1024), `patch_size` (16),
`rope_theta` (10000) and `hidden_act`. Width, depth and head count are silent
and mean `PixtralVisionConfig`'s defaults (1024, 24, 16). The reader checks
them against every tensor shape instead of trusting them.

## The forward pass

```text
x  = conv16x16(pixels)                  # [patches, 1024], no bias
x  = rms_norm(x, ln_pre)                # eps 1e-5
for blk in 0..23:                       # ALL 24; hidden_states[-1], no final norm
    x = x + o_proj(attn(rms(x, attention_norm)))     # 16 heads of 64, scale 0.125,
                                                     # 2-D rope on q and k, no biases
    x = x + down(silu(gate(rms(x, ffn_norm))) * up(rms(x, ffn_norm)))
out = linear_2(gelu_erf(linear_1(x)))   # [patches, 5120]
```

- **No class token, no position table.** Position enters only through the rope.
- **Bidirectional attention over the whole image.** The reference builds a
  block-diagonal mask per image; with one image that is one all-zero block.
- **All 24 blocks run.** `vision_feature_layer = -1` and the strategy is
  `full`: the last block's output, every patch (LLaVA's second-to-last block and
  dropped class row do not apply).
- **`hidden_act` says `gelu` and the blocks run SiLU.** The published config's
  `hidden_act` is `gelu`, a class default that the reference ignores: mlx-vlm's
  `MLP` calls `nn.silu` unconditionally, as Mistral's own implementation does.
  The reader does not read the field as a switch. The projector uses the exact
  erf GELU.
- **Epsilon 1e-5** for `ln_pre` (MLX's `nn.RMSNorm` default) and every block
  norm (`rms_norm_eps`).

### The rope

`PixtralRotaryEmbedding` builds `freqs[j] = theta^(-2j / 64)` for `j in 0..32`,
gives the **height the even entries and the width the odd entries**
(`freqs[::2]`, `freqs[1::2]`), lays the two halves side by side, and duplicates
them over the head; `position_ids_in_meshgrid` indexes that table with
`row * 64 + col`. The kernel's NeoX half-split pairs element `i` with `i + 32`,
which is the duplication, so a row of angles is
`[row * freqs[0], row * freqs[2], ... (16) | col * freqs[1], col * freqs[3], ... (16)]`
(`vision_io::pixtral_rope_freq_rows`, F32). This is neither Qwen's contiguous
ladder per axis nor Gemma's axial rope, and swapping the even and odd
assignment keeps every shape and gives a plausible embedding, which is what
the mutation check below exercises.

## Preprocessing

`PixtralImageProcessor` (`longest_edge` 1024, patch 16, CLIP mean and std, PIL
bicubic), which is a third pipeline, not Qwen's smart resize and not CLIP's
crop:

1. if the longer side is over 1024, scale down by
   `ratio = max(h / 1024, w / 1024)` with each side floored.
2. Round **each side up** to a multiple of 16 by **resizing** to that size (a
   slight stretch: a 1000x1000 image becomes 1008x1008, never cropped or
   padded).
3. Rescale `1/255`, normalize.
4. Patches row-major over a `grid_h x grid_w` grid, each row `(C, 16, 16)`, 768
   values. That is `patch_conv.weight`'s own flattening, so the repack permutes
   the MLX `(out, kh, kw, C)` weight to `(out, C, kh, kw)` once and the GEMM
   reads rows directly.

The 1024x1280 test page becomes 1024 x 832 (width scaled by 0.8, rounded up),
a 64x52 grid, 3,328 patches. The 448x448 page is a 28x28 grid, 784 patches.

## The prompt layout

Pixtral is not a bare run. Per image the processor emits, for a grid of
`h x w`: `[IMG] * w` for each row, `[IMG_BREAK]` between rows (not after the
last), and one `[IMG_END]`. Only the `[IMG]` positions are replaced by the
tower's rows, so **each patch row is its own injection span** and the break and
end tokens keep their ordinary embeddings
(`vision_io::pixtral_expand_image_placeholders`). The ids come from the
tokenizer's `added_tokens` at install time and live in the manifest's tower
start and end id fields (`[IMG_BREAK]` and `[IMG_END]`); `[IMG]` is id 10.

The template renders one `[IMG]` per image part, `[INST]` and `[/INST]` around
the user turn, and emits `bos_token` itself, so the BOS is not doubled
(`clip_prepend_bos` leaves a prompt that already opens with it alone).

**The template reads a text chunk's string from `chunk["content"]`**, where
the OpenAI and Qwen spelling is `chunk["text"]`. An undefined key renders
empty, so an image turn would drop the user's question and ask the model
nothing about the picture. Text parts therefore carry both keys
(`crates/tokenizer/src/jinja_chat_template.rs`), which no other template
iterates over.

## What was built

| Layer | Addition |
| --- | --- |
| `model-io` | `VisionTowerKind::Pixtral`, persisted `pixtral` |
| `vision-io` | `pixtral.rs`: resize rule, preprocess, rope rows, row-structured expander; `ClipLayout`, `ClipPatches::grid_w`, `ClipPreprocessParams::{grid_for, check_against_tower}`, `flat_expand_image_placeholders` |
| `repack` | `vision_pixtral.rs`: `parse_pixtral_vision_config`, `pixtral_special_token_ids`, `read_pixtral_vision_entries` (fuses q/k/v, refuses packed planes, both source spellings) |
| `catalog` | `stream_pixtral_vision_sidecar`, reached from `stream_vision_sidecar` for `model_type = pixtral` |
| `runtime` | `vision/pixtral.rs` tower; `encode_image_clip*` routes to it; `set_prompt_vision_flat` deals an image's rows across its per-row spans; `pixtral_break_end_ids` |
| `tokenizer` | text parts carry `content` as well as `text` |
| `cli`, `ffi`, `server` | the CLIP arms now call the shared expander with the break and end ids and per-image grids |

The blocks pack into a seven-role blob (`n1_w qkv_w proj_w n2_w gate_w up_w
down_w`), q/k/v fused by row concatenation, no biases.

## Tower parity (numpy transcription, not mlx-vlm)

`vision::pixtral::tests::the_real_tower_matches_the_numpy_reference` (ignored,
macOS, env-driven) runs the Metal tower on a real image and compares four
stages against `scripts/pixtral_tower_reference.py`, cosine per stage over the
whole tensor:

| Input | ln_pre | block 0 | block 23 | projector (worst row) |
| --- | --- | --- | --- | --- |
| 448x448, 28x28 grid, 784 patches | 0.99999996 | 0.99999991 | 0.99999001 | 0.99999698 (0.9956) |
| 1024x1280, 64x52 grid, 3328 patches | 0.99999996 | 0.99999991 | 0.99997311 | 0.99996303 (0.9801) |

**Mutation check.** Swapping the even and odd frequency assignment of the rope
(`vision_io::pixtral_rope_freq_rows`) leaves ln_pre at 0.99999996 and block 0
at 0.99993688, because block 0's attention barely moves with position, then
drops block 23 to 0.9402 and the projector to 0.9633 (worst row -0.19), and the
test fails at its 0.999 bar. The guard measures the rope. The mutation was
restored and the pass re-confirmed.

## Real-model gates

Install: `turbospark-model pull --repo ggml-org/pixtral-12b-GGUF --file
pixtral-12b-Q4_K_M.gguf --sidecar-repo mlx-community/pixtral-12b-bf16 --alias
pixtral12b` under a private `TURBOSPARK_HOME`, plus
`pull-vision mlx-community/pixtral-12b-bf16` for the sidecar. Run with
`--vision-sidecar` named explicitly, temperature 0, top-k 1.

Expanded prompt for `shapes.png` (448x336, a 28x21 grid), from
`TURBOSPARK_DEBUG_PROMPT_IDS=1`: 622 ids, `[1, 3, ...]` (BOS, `[INST]`), 588
`[IMG]` (21 x 28), 20 `[IMG_BREAK]` (id 12), one `[IMG_END]` (id 13), then the
question and `[/INST]` (id 4). BOS is not doubled.

| Run | Output |
| --- | --- |
| `shapes.png`, "List each shape in this image and its color." | `The image contains three shapes: a red square, a blue circle, and a green triangle.` |
| `page-small.png`, "Transcribe the first three lines ..." | `00012 \| parity streaming streaming oracle fixture`, `00030 \| prefill parity decode gradient embed`, `00048 \| decode tensor quantize gradient throughput` |
| same shapes question, no image | `I'm unable to view or analyze images directly. ...` |
| `/v1/chat/completions` (OpenAI `image_url` data URI) | the same shapes sentence, 622 prompt tokens |
| `/v1/messages` (Anthropic base64 `image` block) | the same shapes sentence, 622 input tokens |

The shapes sentence names three shape and color pairs only the image supplies,
and the control shows the text trunk cannot produce them. The page
transcription starts with the known first line `00012 | parity streaming`.
Prefill was 110 s for 622 tokens and 69 s for 828 tokens (a cold device, a
single-slot expert-less dense trunk streamed from a shared external drive);
decode was about 19 tok/s. Neither is a benchmark.

## What is and is not verified

Verified: the tower against an independent numpy transcription at four stages on
two real images, with a mutation that fails the guard; end-to-end content on the
CLI and both server endpoints; the prompt layout; the control.

Not verified:

- **mlx-vlm was never executed.** A convention the numpy transcription and the
  Rust tower share (both written from the same source reading) would pass. The
  correct descriptions are the independent evidence, since a wrong rope, a wrong
  patch order or a wrong MLP activation would not describe the colors and shapes
  of a rendered page correctly.
- Only a 448x336 and a 1024x832 grid were run end to end. A portrait image, a
  grid whose longest edge is capped, and several images in one request were not.
- A text-only multi-turn chat after an image turn was not exercised.
- One quantization (Q4_K_M trunk). The bf16 tower is converted to FP16.
- No speed or memory gate. Speculative decoding is disabled on an image runner
  by existing code.

## Mistral3 (Mistral-Small-3.1-24B)

Not attempted: the budget went to Pixtral, and a 24B trunk plus its tower would
not fit alongside the other installs on the shared drive. The delta from this
page, read from the reference source and unimplemented:

- patch 14, `spatial_merge_size` 2, image longest edge 1540 (a different
  preprocessing target);
- an RMSNorm on the tower's output, then a 2x2 patch merger
  (`merging_layer`, Linear 4096 to 1024, no bias, with a specific unfold order)
  before the projector;
- the projector's two linears have no biases;
- one `[IMG]` per merged patch row element, same break and end layout;
- the 24B trunk is a different `llama` shape (40 layers, hidden 5120 as well,
  so the sidecar pairing trap applies again).

A Mistral3 tower reuses `VisionTowerKind::Pixtral`'s block code and adds a
merger plus a norm, so it is a small delta once a trunk fits.
