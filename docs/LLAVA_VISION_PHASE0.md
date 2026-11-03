# LLaVA-1.5 vision: Phase 0 and bring-up

Facts about the LLaVA-1.5 CLIP tower and how the `llama` trunk consumes it,
read off the checkpoint and the references, plus what the bring-up measured.
The mechanics shared with the other towers are in [VISION.md](VISION.md); this
page is only what is specific to `VisionTowerKind::Clip`.

Status as of 2026-11-01: the `llama` family has a real image gate on
LLaVA-1.5-7B through the CLI (`--image`) and through both server endpoints
(`/v1/chat/completions` and `/v1/messages`), via a vision sidecar on a GGUF
trunk. The FFI path (`ts_generate`, the in-process server model, the open-time
vision info) compiles and mirrors the CLI but has not been run.

## Sources

| Source | Pin |
| --- | --- |
| Tower and projector | `llava-hf/llava-1.5-7b-hf` @ `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` (FP16 safetensors, 3 shards) |
| Trunk | `second-state/Llava-v1.5-7B-GGUF`, `llava-v1.5-7b-Q4_K_M.gguf` (4,081,004,320 bytes), `main` on 2026-11-01 |
| Tokenizer, template, processor config | the `llava-hf` repo above (the GGUF repo ships none) |
| Reference | `transformers` (`CLIPVisionModel`, `LlavaMultiModalProjector`, `CLIPImageProcessor`, `LlavaForConditionalGeneration`), installed by `uv run --with transformers` on 2026-11-01 |

Every fact below came from `config.json`, `preprocessor_config.json`,
`processor_config.json`, `chat_template.jinja`, `model.safetensors.index.json`
and the GGUF header.

## The route: GGUF trunk plus a sidecar tower

The `llama` family has **no safetensors intake**. `catalog::stream::stream_mlx`
refuses it (`other => no safetensors intake here`), and its installs
(`mistral7b`, `tinyllama`, `mixtral`) all come from GGUF. LLaVA's language
model is a fine-tune of Vicuna-1.5, so "use the stock text model" is wrong
too: the trunk has to be LLaVA's own weights.

| Route | Cost |
| --- | --- |
| A. Add a safetensors trunk writer for `llama` | New config parser, tensor classifier and writer for a family that has none; the quantization story (FP16 source, MLX affine) is a separate project. |
| B. GGUF trunk, tower from the safetensors repo as a **vision sidecar** | One reader (`read_clip_vision_entries`) and one branch in `stream_vision_sidecar`. Reuses the sidecar format, `pull-vision`, `attach_vision_sidecar`, the packed-block streamer and the llama flow's embedding blit unchanged. |

Route B. Two consequences are worth stating:

- **The sidecar pairs on `(family, hidden_size)` only.** A 4096-wide `llama`
  sidecar also attaches to Mistral 7B. The record's pinned revision says which
  checkpoint the projector was trained against, and nothing enforces it. The
  practical edge is `--vision-sidecar auto`: with this tower the only 4096-wide
  `llama` sidecar installed, `auto` attaches it to ANY 4096-wide `llama` trunk,
  Mistral 7B included, where it can only mislead (its `<image>` marker is not
  in that tokenizer). Name the sidecar explicitly until the pairing record
  carries more than family and width.
- **The trunk's tokenizer comes from the other repo.** The GGUF carries one
  (32,000 tokens), but this port loads an HF `tokenizer.json`, so the install
  takes `--sidecar-repo llava-hf/llava-1.5-7b-hf`, which also supplies the
  chat template.

## Inventory

686 tensors in the HF checkpoint. Outside the language model:

| Group | Tensors | Notes |
| --- | --- | --- |
| `vision_tower.vision_model.embeddings.*` | 3 | `class_embedding` [1024], `patch_embedding.weight` [1024, 3, 14, 14] (no bias), `position_embedding.weight` [577, 1024] |
| `vision_tower.vision_model.pre_layrnorm.*` | 2 | sic, `pre_layrnorm`, not `pre_layernorm` |
| `vision_tower.vision_model.encoder.layers.N.*` | 24 x 16 | `layer_norm1/2` w+b, `self_attn.{q,k,v,out}_proj` w+b, `mlp.fc1/fc2` w+b |
| `vision_tower.vision_model.post_layernorm.*` | 2 | never read, see below |
| `multi_modal_projector.linear_{1,2}.*` | 4 | [4096, 1024], [4096, 4096], each with a bias |

All FP16, none quantized. The tower is 303M parameters.

## The forward pass

Read off `transformers` and cross-checked against mlx-vlm's `llava` (read, not
run):

```text
x   = [class_embedding ; conv14x14(pixels)] + position_embedding     # [1 + 576, 1024]
x   = pre_layrnorm(x)                                                 # eps 1e-5
for blk in 0..23:                                                     # NOT 24
    x = x + attn(layer_norm1(x))      # 16 heads of 64, scale 0.125, bidirectional, no rope
    x = x + fc2(quick_gelu(fc1(layer_norm2(x))))
feat = linear_2(gelu_erf(linear_1(x[1:])))                            # class row dropped
```

- **Twenty-three blocks.** `vision_feature_layer = -2` selects
  `hidden_states[-2]`, the output of block index 22 (`hidden_states[0]` is the
  embeddings after `pre_layrnorm`). Block 23 and `post_layernorm` are never
  read, so the repack drops them (25 MiB less per image) and the manifest
  declares `depth = 23`. A port that ran all 24 would produce a plausible
  embedding.
- **`quick_gelu`**, `x * sigmoid(1.702 x)`, in the blocks. The projector uses
  the exact erf GELU (`nn.GELU()`). Different kernels:
  `GeluKind::Quick`, `GeluKind::Erf`.
- **No rope, no mask.** Positions are the learned absolute table, added once.
- **The class row is dropped after the last block and before the projector**
  (`vision_feature_select_strategy = default`).
- All LayerNorms use epsilon 1e-5 (a kind constant, not a manifest float).

## Preprocessing

`CLIPImageProcessor`, which is **not** what the original LLaVA repo does (that
pads to a square with the mean color; the Hugging Face processor center-crops):

1. resize so the shortest edge is 336, PIL bicubic, long edge
   `int(336 * long / short)`;
2. center crop to 336x336, `top = (h - 336) / 2`, `left = (w - 336) / 2`;
3. `x * (1/255)`, then `(x - mean) / std` with the OpenAI CLIP constants;
4. patches row-major over a 24x24 grid, each row **`(C, 14, 14)`**, 588 values.

That row order is `patch_embedding.weight`'s own flattening, so the repack
copies the conv weight verbatim and the GEMM reads it row-major. An MLX-layout
source (`[out, kh, kw, C]`) is permuted at repack. The opposite choice from the
qwen tower's `(T, P, P, C)`, for the same reason.

Measured against `CLIPImageProcessor` on the 640x480 shapes image: worst
element difference **4.8e-7** (`clip_vision_parity`), so resize, crop and
normalize are the processor's.

## Prompt building

The template renders **one bare `<image>`** per image, then a newline, then
the text, and the processor repeats it
`(336/14)^2 + 1 - 1 = 576` times (`num_additional_image_tokens = 1` for the
class row, minus 1 for the `default` strategy). No start or end token.

```text
USER: <image>
Describe the shapes in this image. ASSISTANT:
```

Five traps, each a fluent wrong answer or an opaque refusal:

**The llava-hf template emits no `<s>`, and this port's convention is that the
template carries the BOS.** Every front end encodes the rendered prompt with
`add_bos = false`, so the first prefill of this work ran BOS-less (ids began at
`USER`, 3148). The Hugging Face processor tokenizes with special tokens on and
prefills `<s>` first, and Vicuna was trained with it. `clip_prepend_bos`
prepends the tokenizer's BOS (`encode("", true)`) when the rendered prompt does
not already open with it, at the CLI, FFI and server expansion sites, and a
template that does emit `<s>` is never doubled. Prompt length 602 became 603.

**`<image>` is id 32000 and the GGUF trunk has 32,000 embedding rows.** The HF
checkpoint pads its table to 32,064; the GGUF does not. The llama flow
range-checks every prompt id BEFORE the embedding blit that would have replaced
the lookup (`families/llama/mod.rs`, `prefill.rs`), so an expanded run of 32000
is refused as out of vocabulary. The expander therefore takes a separate
**fill id**: `clip_expand_image_placeholders(rendered, marker, fill, counts)`
finds the marker by the tokenizer's id and writes `fill` (the marker where it is
in range, id 0 otherwise) into the run. The run's content is never read as a
token. `RealForwardRunner::clip_image_ids()` returns `(marker, fill)`.

**The llava-hf template only renders list-typed content.** It iterates
`message['content'] | selectattr('type', ...)`, so a plain-string user turn
renders `USER: ASSISTANT:`. A text-only `--messages-file` therefore asks the
model an empty question (8 prompt tokens, a confused answer). An image turn
carries typed parts and renders correctly. Use `--prompt "USER: ... ASSISTANT:"`
for text-only checks of the trunk.

**`verify_image_markers` assumes a start token.** It requires a distinct
`vision_start` and exactly one of each. LLaVA has none, so
`MfTokenizer::verify_bare_image_marker` checks the marker alone, and the three
sidecar attach sites (CLI, FFI, server) pick it for a CLIP tower.

**The position table is fixed.** 577 rows, a 24x24 grid plus the class row; the
processor always crops to one size, so there is nothing to interpolate. A tower
asked for a different patch count refuses by name.

## Injection

`PromptVision::flat` (the Gemma 4 constructor): rows replace the placeholders'
embeddings, positions are the plain token index, rope delta zero. The llama
flow already hosts the embedding blit and the `(t, h, w)` rope dispatch for
`qwen3_vl`; with every triple equal it takes the ordinary rope kernel, so the
whole injection is the blit. Attention over the image run is causal like any
text (LLaVA does not make it bidirectional).

## What was built

| Layer | Addition |
| --- | --- |
| `model-io` | `VisionTowerKind::Clip`, persisted `clip` |
| `vision-io` | `clip.rs`: params, `clip_preprocess`, `clip_expand_image_placeholders` |
| `gpu` | `vision_gelu_quick_fp16`, `GeluKind::Quick` |
| `repack` | `vision_clip.rs`: `parse_llava_vision_config`, `read_clip_vision_entries` (fuses q/k/v, drops the unread tail) |
| `catalog` | `stream_clip_vision_sidecar`, reached from `stream_vision_sidecar` for `model_type = llava` |
| `runtime` | `vision/clip.rs` tower, `encode_image_clip[_with_stages]`, `set_prompt_vision_flat`, `clip_image_ids` |
| `tokenizer` | `verify_bare_image_marker` |
| `cli`, `ffi`, `server` | `attach_clip` (CLI, FFI); the marker check at all three attach sites; `VisionInfo::Clip` / `RequestImages::Clip` with a planner arm and an encode arm (server and the FFI's in-process server model); the FFI open-time `VisionInfo` (`active`, `max_pixels`) |

The blocks pack into the qwen tower's twelve-role blob (q/k/v fused by row
concatenation), so the streamer, the role resolver and the FP16 scoping are the
qwen tower's.

## Gates

**Offline** (no model): config parse and refusals, the reader on a synthetic
checkpoint (fusion order, MLX-layout permutation, dropped tail, missing and
unknown names, projector width), preprocessing (resize dims, crop, row order),
the expander, the quick-GELU kernel against its closed form and against tanh
(`vision_parity`), the bare-marker check, the `clip` spelling.

**Stage 2, `crates/runtime/tests/clip_vision_parity.rs`** (`#[ignore]`d), on
the shapes image, cosine against `transformers` in float32:

| stage | cosine | worst row |
| --- | --- | --- |
| `hidden_states[0]` (after `pre_layrnorm`) | 0.99999961 | 0.999999 |
| `hidden_states[1]` (after block 0) | 0.99999920 | 0.999998 |
| `hidden_states[-2]` (after block 22) | 0.99958579 | 0.8108 |
| projector output | 0.99983399 | 0.6877 |

**The last two rows are the reference's own FP16 floor, and the test says so.**
Two of 577 rows sit below 0.99 (rows 517 and 382), and neither is a high-norm
row. The same reference run in FP16 against its own float32 run reads cosine
0.99947686 with worst row 0.7827, on the same rows (row 517: 0.7827, row 382:
0.9625). The port against that FP16 reference reads 0.99993927, worst row
0.9945. The port is therefore closer to float32 (0.99959) than FP16 storage
alone predicts, and the gap is precision on a few sensitive positions, not a
function difference. The assertions are against the FP16 reference
(cosine > 0.9999, worst row > 0.99) plus a floor-relative check on the float32
comparison.

Mutation-checked, both red and localizing:

| mutation | embed | block 0 | block 22 | projector |
| --- | --- | --- | --- | --- |
| none | 0.99999961 | 0.99999920 | 0.99958579 | 0.99983399 |
| blocks use tanh-GELU | 0.99999961 | **0.99131816** | **0.93487320** | **0.95641557** |
| projector reads from row 0 (class row kept) | unchanged | unchanged | unchanged | **0.34212241** |

**Real runs** (`turbospark-check --messages-file --image`, greedy, the sidecar
on a Q4_K_M trunk unless stated, M4 Max with other sessions on the host, not a
benchmark): 603-token prompts, prefill about 20 to 30 s, decode 28 to 45 tok/s.

*shapes.png* (a red circle left, a blue square right, a green triangle below,
on white), "Describe the shapes in this image and say what color each one is.":

- port, Q4_K_M, with BOS: "The image features a red square, a blue circle, and
  a green triangle, all placed on a white background. The red square is located
  on the left side of the image, the blue circle is in the middle, and the green
  triangle is on the right side."
- port, **Q8_0** trunk, same prompt: the same text for the first 55 tokens and
  beyond. Two quantizations agreeing word for word says the answer is set by
  the image rows and the trunk's behavior, not by trunk quantization noise.
- `transformers`, FP16, same image and prompt: "In the image, there are three
  shapes: a red circle, a blue square, and a green triangle. The red circle is
  located on the left side ..."

All three get the colors, the count and the left-to-right order right. The
port swaps which shape carries which color (circle and square). **That is the
reference's own weakness, measured:** asked four direct questions about this
image, the FP16 reference answers "What shape is the red object on the left?"
with *Square*, the blue one on the right with *Circle*, and "What color is the
circle?" with *Blue*, all wrong, while the port answers the first with
*Circle*, right. LLaVA-1.5-7B binds attributes unreliably, and the reference
and the port are unreliable in the same way.

What was ruled out as the cause of the swap: the prompt (the port's 603 ids are
**identical** to `AutoProcessor`'s: head `[1, 3148, 1001, 29901, 29871]`, a
576-long run, the same 22-id tail; `scripts/llava_prompt_ids.py`), the tower
(inside its FP16 floor), trunk quantization (Q4_K_M and Q8_0 agree), and the
trunk's long prefill (a 611-token text prompt retrieves a fact planted at its
start, "PINEAPPLE").

*shapes2.png* (a yellow circle left, a magenta square right, on black), the
negative control with the colors, count and background changed: CLI, "The image
features two distinct shapes, one being a yellow circle and the other a pink
square. The yellow circle is located on the left side ... the pink square is on
the right side." That matches these pixels and none of the previous image's,
which is what a model that never saw the picture cannot produce.

**Server**, `turbospark-server --model llava15-7b --vision-sidecar ...`, a
base64 data URL on both wire formats with the same prompt: `/v1/chat/completions`
and `/v1/messages` return identical text, no `x-anyllm-degradation` header. On
`shapes2.png`: "The image features two shapes, one is a yellow circle and the
other is a pink square. The yellow circle is located on the left side of the
image, while the pink square is on the right side." A second request on
`shapes.png` to the same server names red, blue and green and mentions nothing
from the previous image, so the injection map does not leak across requests.

## Reproducing

```sh
export TURBOSPARK_HOME=/Volumes/T7/turbospark-llava   # keep it off the real store

turbospark-model pull-vision --repo \
  llava-hf/llava-1.5-7b-hf@b234b804b114d9e37bb655e11cbbb5f5e971b7a9 \
  --alias llava15-vision-tower
turbospark-model pull --repo second-state/Llava-v1.5-7B-GGUF \
  --file llava-v1.5-7b-Q4_K_M.gguf \
  --sidecar-repo llava-hf/llava-1.5-7b-hf@b234b804b114d9e37bb655e11cbbb5f5e971b7a9 \
  --alias llava15-7b

turbospark-check --model llava15-7b \
  --vision-sidecar $TURBOSPARK_HOME/models/text/llava15-vision-tower.gturbo-vision \
  --messages-file msgs.json --image shapes.png --temperature 0 --top-k 1 --max-new 80
```

`scripts/llava_vision_probe.py` dumps the reference stages the parity test
reads; `scripts/llava_reference_generate.py` is the reference's greedy answer.

## Not done

- **The FFI paths compile and mirror the CLI and server but have not been run**
  against an install (`attach_clip`, the in-process server model, the open-time
  `VisionInfo`), and neither has the Swift app.
- **No catalog rows.** The two installs are `pull`/`pull-vision` commands, not
  `models.json` entries, and no row carries the pinned revisions.
- **No quality or memory gate.** No frozen quality digest, no memory oracle, and
  the 20 tok/s prefill is unprofiled.
- **One image per prompt is the reference's limit** (`llava.py` refuses more);
  the port expands any count but only one has been run.
- **LLaVA-1.6 / NeXT** (anyres tiling, image-newline tokens) is a different
  pipeline and is not attempted.

## Pixtral and Mistral3 (assessed, not implemented)

Neither reuses this tower, so neither was attempted.

- **Pixtral-12B** (`mistral-community/pixtral-12b`): a Mistral-Nemo-class trunk
  (40 layers, hidden 5120, 32 heads of 128 with 8 KV heads, vocab 131,072,
  rope theta 1e9) and a Pixtral tower: patch 16, RMSNorm, a gated SiLU MLP, 2-D
  rope, variable image size up to 1024, and `[IMG_BREAK]`/`[IMG_END]` tokens
  between patch rows.
- **Mistral3** (Mistral-Small-3.1-24B): the same Pixtral tower at patch 14 with
  a 2x2 patch merger, on a 40-layer, 5120-wide, 32,768-intermediate trunk.

Both need a new tower kind (the RMSNorm, gated MLP and 2-D rope tower is
the same shape as Qwen2.5-VL's), a trunk baseline for a shape the `llama`
family does not register, and a prompt layout that is not a bare run. That is a
tower-sized project, not a reuse.
