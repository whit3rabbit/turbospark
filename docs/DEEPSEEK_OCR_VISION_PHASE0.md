# DeepSeek-OCR vision, Phase 0

Status: Phase 0 facts, read off the real checkpoint header and the
reference sources before any code. Scope decision at the end.

Checkpoints: `mlx-community/DeepSeek-OCR-4bit` (first target, 2.3 GiB) and
`baidu/Unlimited-OCR-2` (second target, same trunk, see the last section).

Reference discipline: the conventions below come from READING mlx-vlm's
`deepseekocr/{sam,vision,deepseekocr,processing_deepseekocr}.py` and the
checkpoint's own `deepencoder.py`. No code from either was imported or run.
`scripts/deepseek_ocr_tower_reference.py` is a numpy transcription written
from that reading, and it is the parity oracle.

## 1. The trunk is DeepSeek-VL2-tiny's

`config.json` (`model_type` `deepseekocr`, `language_config` with
`use_mla: false`, `qk_nope_head_dim` 0): 12 layers, hidden 1280, 10 heads (no
GQA), vocab 129280, 64 routed experts top-6 plus 2 shared (fused as one
SwiGLU of 2 x 896), dense intermediate 6848, `first_k_dense_replace` 1,
greedy top-k, softmax router, rope theta 10000 (absent, the format default),
rms eps 1e-6. That is the shape `parse_deepseek_vl2_config` already accepts
(`docs/DEEPSEEK_VL2_VISION_PHASE0.md`), so no family was added. Only two
registry rows were: `deepseekocr` and `unlimited-ocr` resolve to
`ModelFamily::Deepseek2` in `arch_registry.rs`.

The one intake change: `classify_for_family` claimed `vision.`, `projector.`,
`image_newline`, `view_separator` for `Deepseek2` (VL2's names). The OCR
checkpoint's towers are `sam_model.` and `vision_model.`, so the first pull
failed with `unclassifiable tensor: sam_model.blocks.0.attn.proj.bias`. Both
prefixes are now claimed, family-gated, as the vision bucket.

## 1b. Text gate (measured, trunk only, before any image work)

`turbospark-model pull --repo mlx-community/DeepSeek-OCR-4bit --alias dsocr`
installs the text path (1.6 GiB). The template is `{{content}} ` per message
(no roles); the engine's prompt starts with BOS id 0 (the V2 dialect rule).

- `What is the capital of France? Answer in one short sentence.` answers
  `The capital of France is Paris.` (15 prompt ids, 34 tok/s decode).
- Teacher-forced against `scripts/deepseek_vl2_trunk_reference.py` (an
  independent numpy forward; it gained BF16 companion support for this
  checkpoint, VL2 ships F16): 15 ids agree 15/15, last-position logits
  cosine 0.999975, KL 0.00003 nats, top-5 5/5. A 1048-id prompt agrees
  1044/1048, last-position cosine 0.99734, KL 0.0167, top-5 4/5. The four
  disagreeing positions (18, 85, 102, 1045) are interior; at the answer
  position both pick the same id.
- A needle question over that 1048-id prompt ("The secret word for the vault
  is MARIGOLD ... What is the secret word") is answered `0` by the engine AND
  has the same argmax in the numpy reference. This model is OCR-tuned and does
  not do retrieval QA, so no behavioural retrieval claim is made for the text
  path; the numerical agreement at length is the gate.

## 2. Header inventory (871 tensors, read from the real file)

Trunk: `language_model.model.embed_tokens`, `language_model.lm_head` (both
4-bit, `[129280, 160]` U32 with `[129280, 20]` scales and biases),
`language_model.model.norm.weight`, per-layer attention and MoE tensors.

Vision side, all BF16 except the projector:

| tensor | shape |
|---|---|
| `sam_model.patch_embed.proj.weight` / `.bias` | `[768,16,16,3]` / `[768]` |
| `sam_model.pos_embed` | `[1,64,64,768]` |
| `sam_model.blocks.N.attn.qkv.{weight,bias}` | `[2304,768]` / `[2304]` |
| `sam_model.blocks.N.attn.proj.{weight,bias}` | `[768,768]` / `[768]` |
| `sam_model.blocks.N.attn.rel_pos_{h,w}` | `[27,64]` (window blocks), `[127,64]` (blocks 2,5,8,11) |
| `sam_model.blocks.N.mlp.lin1` / `lin2` | `[3072,768]` / `[768,3072]` with biases |
| `sam_model.blocks.N.norm1` / `norm2` | `[768]` weight and bias |
| `sam_model.neck.0.weight` | `[256,1,1,768]` (1x1, no bias) |
| `sam_model.neck.1.{weight,bias}`, `neck.3.{weight,bias}` | `[256]` (LayerNorm2d) |
| `sam_model.neck.2.weight` | `[256,3,3,256]` (3x3, no bias) |
| `sam_model.net_2.weight` / `net_3.weight` | `[512,3,3,256]` / `[1024,3,3,512]` |
| `vision_model.embeddings.class_embedding` | `[1024]` |
| `vision_model.embeddings.position_embedding.weight` | `[257,1024]` |
| `vision_model.embeddings.patch_embedding.weight` | `[1024,14,14,3]` (UNUSED, see 3) |
| `vision_model.pre_layrnorm.{weight,bias}` | `[1024]` |
| `vision_model.transformer.layers.N.{layer_norm1,layer_norm2}` | `[1024]` weight and bias |
| `...self_attn.qkv_proj.{weight,bias}` | `[3072,1024]` / `[3072]` (fused) |
| `...self_attn.out_proj.{weight,bias}` | `[1024,1024]` / `[1024]` |
| `...mlp.fc1` / `fc2` | `[4096,1024]` / `[1024,4096]` with biases |
| `projector.layers.{weight,scales,biases,bias}` | U32 `[1280,256]`, BF16 `[1280,32]` x2, BF16 `[1280]` |
| `image_newline`, `view_separator` | `[1280]` each |

The convs are already in MLX layout (`[O,kh,kw,I]`, channels last). Only the
projector is quantized (4-bit, group 64, so 2048 inputs = 32 groups of 64).
Everything else is BF16 and the sidecar dequantizes the projector to FP16.

## 3. Forward pass

Input: one 1024 x 1024 RGB view, `(pixel/255 - 0.5)/0.5`.

SAM-ViT-B:

1. 16x16 stride-16 conv with bias gives 64 x 64 x 768; add `pos_embed`
   (64x64, no interpolation at 1024).
2. 12 blocks, each `x = x + attn(norm1(x))`, `x = x + mlp(norm2(x))`, both
   LayerNorms eps 1e-6, MLP `lin1`, exact (erf) GELU, `lin2`. Blocks 2, 5, 8,
   11 attend globally over all 4096 tokens. The others use 14 x 14 windows:
   the 64 x 64 grid is zero-padded to 70 x 70 AFTER `norm1` (5 x 5 windows),
   and the padded tokens are real keys (they carry the qkv bias), NOT masked.
   The result is unpartitioned and cropped back to 64 x 64.
3. Attention is 12 heads of 64. `softmax(scale * q.k + rel_h + rel_w) v`
   where `rel_h[h,w,kh] = q[h,w] . Rh[h,kh]`, `Rh[h,k] = rel_pos_h[h - k +
   (n-1)]`, and the same for width. The bias is built from the UNSCALED query
   and added after the 1/8 scaling. Window blocks use `n = 14`, global `n = 64`.
4. Neck: 1x1 conv 768 -> 256, LayerNorm2d (over channels, eps 1e-6), 3x3
   pad-1 conv 256 -> 256, LayerNorm2d. Then `net_2` (3x3 stride 2 pad 1,
   256 -> 512) and `net_3` (512 -> 1024): output 16 x 16 x 1024.

CLIP-L (24 layers, 1024 wide, 16 heads, MLP 4096, quick-GELU
`x * sigmoid(1.702 x)`):

- It does NOT run its own 14 x 14 patch conv in this model. The SAM grid
  (256 x 1024) is its patch embedding: prepend `class_embedding`, add the
  257-row position table (exactly the 16 x 16 + 1 rows, so no interpolation
  in base mode), then `pre_layrnorm`.
- Layer LayerNorms and `pre_layrnorm` use eps 1e-5 (the official
  `deepencoder.py`: `layernorm_epsilon` and `pre_layernorm_epsilon` 1e-5).
  mlx-vlm's `VisionConfig.layer_norm_eps` defaults to 1e-6 for the layer
  norms. The checkpoint's config does not set it, so the two sources
  disagree. The engine follows the official code (1e-5). The difference is
  below fp16 noise on these activations and the parity test uses one value.
- No final LayerNorm: the encoder output is the last layer's residual stream.

Projection and layout: `concat(CLIP[1:], SAM)` along channels (order
matters) gives 256 x 2048, one `Linear(2048, 1280)` with bias gives
256 x 1280. Base-mode rows are 16 lines of (16 projected + `image_newline`)
then one `view_separator`: 16 x 17 + 1 = 273 rows, in that order.

## 4. Prompt layout

There is no chat role framing. The processor emits `BOS (id 0)`, the prompt
text with each `<image>` (id 128815) replaced by 273 copies of the image id
in base mode, then the rest. The tokens' embeddings are replaced by the 273
tower rows, in order. The model's own prompts: `<image>\nFree OCR.` and
`<image>\n<|grounding|>Convert the document to markdown.`.

The shipped `chat_template.jinja` is not that layout, so the OCR path must
build the layout directly (BOS, then text parts with an image expanded to
273 ids). Judged through `--messages-file` the template still decides the
text parts, and the engine puts the image rows where `<image>` renders.

## 5. Image preprocessing (base mode)

`ImageOps.pad(image, (1024, 1024), color=(127,127,127))`: scale to fit
preserving aspect ratio (PIL BICUBIC), centre on a 127 grey canvas. Then
`/255`, mean 0.5, std 0.5. No crops. Gundam (dynamic 640 crops plus the
1024 global view) additionally needs bicubic-antialias interpolation of the
CLIP position table to 10 x 10 and the crop-grid layout.

## 6. Scope decision

Built: BASE mode (one 1024 x 1024 global view, 273 rows). NOT built:
Gundam (640 px local crops plus the global view: needs a bicubic-antialias
resample of the CLIP position table to 10 x 10 and the crop-grid layout), and
the Tiny/Small/Large resolutions (64/100/256 `<image>` tokens at 512/640/1280).

## 7. Results (real checkpoint, `mlx-community/DeepSeek-OCR-4bit`)

Intake: `turbospark-model pull-vision --repo ... --alias dsocr-vision` writes a
764.6 MiB sidecar (24 CLIP blocks at a 25.2 MB stride streamed, 197 MB of SAM
and projector residents). Trunk and tower are separate installs.

Tower gate, `crates/runtime/tests/deepseek_ocr_tower_real.rs` against
`scripts/deepseek_ocr_tower_reference.py` (numpy, written from reading, no
mlx-vlm code run) on a synthetic structured input: cosine >= 0.99998 at every
stage (SAM patch, blocks 0/2/5/11, neck, net_2, SAM out, CLIP pre, layers
0/11, CLIP out) and 0.999992 on the final 273 x 1280 rows. Mutations, both
failing that test: dropping the relative-position bias (`sam_b0` 0.9986,
`rows` 0.634) and padding SAM windows with token 0 instead of zeros (`rows`
0.889). The reference and the engine share one reading of the sources, so
this proves the implementation matches the reading, not that the reading is
right; the OCR gate below is the check on the reading.

Image gate, `turbospark-check --messages-file ask-free-ocr.json --image`, the
model's own prompt `<image>\nFree OCR. ` (prompt ids: BOS, 273 image ids, 5
text ids = 279):

| input | result |
|---|---|
| `page-small.png` (448 px crop) | 24/24 lines: line number and first three words exact, numbers 00012..00426 in order, 107 to 239 tok/s decode |
| `page.png` (1024 x 1280, ~9 px glyphs after the 0.8 pad scale) | 21/24 of the transcribed lines exact; the right-edge decimals of lines 1 to 4 correct; digit drift on tiny text (00120 read as 00130) and no stop at the page end (135 rows for a 70-line page, MaxTokens) |
| no image, same prompt | fluent generic text about OCR tools, none of the page's numbers |

Server gate, `scripts/ocr_server_gate.py` (`turbospark-server --model dsocr
--vision-sidecar ...`), `page-small.png`: `/v1/chat/completions` and
`/v1/messages` both 24/24 (279 prompt tokens, finish `stop` / `end_turn`); the
no-image control carries none of the numbers. `scripts/check_ocr_page_output.py`
scores them against 24 lines transcribed by eye from the page image (the
generator needs Pillow and the exact font).

Not verified: Gundam, any page cleaner than a synthetic monospace one, FFI and
the Swift app (the FFI arms compile and mirror the SigLIP ones, nothing ran
them), power or throughput numbers, an A/B against mlx-vlm (never run).

## 8. Unlimited-OCR-2: refused at the trunk, not attempted

`turbospark-model probe baidu/Unlimited-OCR-2` says `REFUSED: config.json
invalid: rope_scaling is not wired for the DeepSeek-VL2 trunk` (6.2 GiB, no
chat template, `model_type` `unlimited-ocr`). Its trunk sets `rope_scaling`,
which the plain-attention `deepseek2` arm does not implement, so it needs a
flow change (scaled rope in the MHA arm, with its own numpy oracle) before any
tower question. The registry row and the tower reader's `unlimited-ocr`
spelling are ready; nothing was downloaded.
