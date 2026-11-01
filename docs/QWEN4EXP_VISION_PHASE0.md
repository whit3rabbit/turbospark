# Qwen4Exp vision: Phase 0

Facts about the `qwen4_exp` vision tower read off the real checkpoint and the
two references, before any code. Implementation notes live in
[VISION.md](VISION.md). This page only records what the checkpoint says and
what that decides.

Read on 2026-11-01. No weights were downloaded: the evidence is `config.json`,
`preprocessor_config.json`, `tokenizer_config.json`, `chat_template.jinja`,
`model.safetensors.index.json`, and the safetensors header of shard 1 (an
8-byte length read plus a 60,945-byte header range read).

| what | where | revision |
| --- | --- | --- |
| checkpoint | `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit` (catalog `qwen4-reap288`) | `668f31bcc56bf9400e64c9463445eee47597c2d9` |
| its source | `Qwen/Qwen3.8-Flash-Next` (from `config.json` `mererun_conversion`) | `f5d08274bafd880402bd16f5e3e6c514136ec06c` |
| trunk reference | `transformers` `models/qwen4_exp/modular_qwen4_exp.py` (local uv cache) | the copy `docs/QWEN4_PHASE0.md` read |
| flow reference | `mlx-vlm` `models/qwen4_exp/` (local uv cache) | same |

The Swift GGUF tiers ship their tower as a separate 0.91 GB BF16 mmproj
(`docs/QWEN4_EXP.md`). That is a GGUF intake question and is out of scope
here.

## 1. The tower is the Qwen3.8 dense tower, but QUANTIZED

`vision_config` is field for field the `qwen3_5` tower except the output
width: depth 27, hidden 1152, 16 heads (head_dim 72), intermediate 4304,
patch 16, temporal 2, merge 2, 2304 position rows (48x48),
`gelu_pytorch_tanh`, `deepstack_visual_indexes: []`, and
`out_hidden_size: 2560`, which equals the trunk's `hidden_size`. The merger
input is `1152 * 4 = 4608`.

The tensors live under `vision_tower.` (MLX spelling), all 501 of them in
shard 1, 448,092,512 bytes. That is the 333-name inventory of the Qwen3.8
tower plus 168 `.scales`/`.biases` companions, because **this conversion
quantized the tower**. The config's `quantization` block (4 bits, group 64)
has no vision override, and mlx-vlm's `quant_predicate` returns `True` for
every path outside the PLE shards and router gates, so every vision module
whose input width divides by 64 was quantized:

| tensor | dtype | stored shape | logical shape |
| --- | --- | --- | --- |
| `blocks.N.attn.qkv.weight` | U32 + BF16 scales/biases `[3456, 18]` | `[3456, 144]` | `[3456, 1152]` |
| `blocks.N.attn.proj.weight` | U32 + `[1152, 18]` | `[1152, 144]` | `[1152, 1152]` |
| `blocks.N.mlp.linear_fc1.weight` | U32 + `[4304, 18]` | `[4304, 144]` | `[4304, 1152]` |
| `blocks.N.mlp.linear_fc2.weight` | BF16 | `[1152, 4304]` | same (4304 is not a multiple of 64) |
| `merger.linear_fc1.weight` | U32 + `[4608, 72]` | `[4608, 576]` | `[4608, 4608]` |
| `merger.linear_fc2.weight` | U32 + `[2560, 72]` | `[2560, 576]` | `[2560, 4608]` |
| `pos_embed.weight` | U32 + `[2304, 18]` | `[2304, 144]` | `[2304, 1152]` |
| `patch_embed.proj.weight` | BF16 | `[1152, 2, 16, 16, 3]` | same, rank 5 as on Qwen3.8 |
| every bias and norm | BF16 | rank 1 | same |

Consequences:

- The current vision writer cannot ingest this. It refuses `U32` and would
  record the packed shape. The repack must dequantize the affine planes to
  FP16 (`w = q * scale + bias`, nibble `i` at bit `4 * (i % 8)` of word
  `i / 8`, the layout `compute::dequantize_int4_affine` already reads). The
  runtime tower is FP16 end to end and gains no kernel.
- Bits and group come from the checkpoint's `quantization` block, not from
  the shapes: `[N, 144]` words with 18 groups is both 4 bits at group 64
  (1152 wide) and 8 bits at group 32 (576 wide).
- The reference this install must match is a 4-bit tower. mlx-vlm runs the
  quantized planes, so a dequantized FP16 tower is the same function up to
  FP16 rounding. The Qwen3.8 parity numbers in `VISION.md` were measured on
  a BF16 tower and do not transfer.
- At FP16 the dequantized tower is about 0.86 GiB, 27 blocks of about
  30.5 MiB streamed two at a time, the same residency as Qwen3.8.

## 2. Image tokens and markers

| id | token |
| --- | --- |
| 248053 | `<\|vision_start\|>` |
| 248054 | `<\|vision_end\|>` |
| 248055 | `<\|vision_pad\|>` |
| 248056 | `<\|image_pad\|>` (`image_token_id`) |
| 248057 | `<\|video_pad\|>` (`video_token_id`) |

The template emits `<|vision_start|><|image_pad|><|vision_end|>` per image,
the `qwen3_5` form. The processor is `Qwen3VLProcessor` with
`Qwen2VLImageProcessorFast`, `image_mean = image_std = [0.5, 0.5, 0.5]`, and
`size {shortest_edge: 65536, longest_edge: 16777216}`, identical to
`docs/VISION_PHASE0.md` item 4. `crates/vision-io` needs no change.

## 3. mRoPE

`text_config.rope_parameters` is `mrope_interleaved: true`,
`mrope_section: [11, 11, 10]`, partial factor 0.25 at head_dim 256, so
rotary_dim 64, theta 1e7. That is the Qwen3.8 dense layout, so
`encode_rope_mrope_interleaved` applies unchanged. The positions come from
the inherited `compute_3d_position_ids` (the `get_rope_index` walk), so
`turbospark_vision_io::mrope_position_triples` and `rope_delta` apply as on
Qwen3.8.

GDN layers take no position. Only the 12 QSA layers rotate, and in two
places:

1. **Attention q and k**, at the triple. The same seam as
   `families/qwen/attn.rs`.
2. **The QSA indexer.** `Qwen4ExpTextModel.forward` keeps the FULL 3-D
   position history (`indexer_cache.indexer_position_ids`) and every
   indexer rotates its pooled block keys at `full_cos[group_starts]`, the
   block's FIRST token, and its query at the current token's triple. So
   above the budget a pooled block that starts on an image token must be
   rotated by that token's triple, not by its index.

The port's indexer rotates pooled keys inside
`gpu::encode_qsa_advance_blocks` at a scalar position. Those pooled keys are
read only when selection runs (`complete_blocks > index_top_k`, i.e.
`visible > 2051` on this checkpoint). Below that the dense kernel runs and
the indexer's ropes are never consumed.

Decision: an image prompt is refused
on a Qwen4Exp runner whose context can cross the budget, unless
`TURBOSPARK_QSA_FORCE_DENSE=1` keeps selection off. The catalog row already
caps the window at 2048. A triple-aware pooled rope is the follow-up that
would lift this.

## 4. PLE at image positions

Both references hash the RAW `input_ids`, not the spliced embeddings:

- transformers: `Qwen4ExpModel.forward` sets `ple_input_ids = input_ids`
  BEFORE `masked_scatter` splices the image rows into `inputs_embeds`, then
  calls the language model with `input_ids=None` and that `ple_input_ids`.
- mlx-vlm: the language model passes its `inputs` ids to
  `layer.ple.ple_embedding` alongside `inputs_embeds`.

So at an image position the n-gram hash sees `image_token_id` (248056)
repeated, the PLE table rows are looked up for those placeholder n-grams,
and the PLE query reads the residual, which now carries the tower row. The
port's PLE path already hashes the token id the producer was handed, so the
blit must replace the embedding lookup ONLY and leave the id flowing to PLE.
Feeding the tower row's argmax or skipping PLE at image positions would both
be fluent and wrong.

## 5. The embedding blit

`hidden = embed_tokens(ids).repeat(1, 1, hc_count)`, and the image splice
happens on `inputs_embeds` before that repeat. So the tower row (2560 wide)
is tiled `hc_count` (4) times into the 10240-wide residual, replacing the
tiled lookup. The trunk's `hidden_size` is the row width, which is what
`set_prompt_vision` already validates against.

## 6. What is refused

- Speculation with an image. Qwen4Exp has no rollback (`checkpoint()`
  asserts), the MTP head is not ingested for this family, and the DFlash2
  drafter is gated to `QwenGdnDense`. No speculative path can attach today.
- Context past the QSA budget with an image (item 3).
- Video. Same boundary as every other tower here.

## 7. Open after this page

- A real install. The combined pull streams the whole 131-shard checkpoint,
  and this page did not download it.
- Stage 2 tower parity against mlx-vlm on the 4-bit tower, and stage 4 logit
  parity. Both need the install.
- Triple-aware pooled block rope in the QSA indexer, to lift item 3's
  refusal.
