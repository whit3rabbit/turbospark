# Missing vision support in TurboSpark

Snapshot 2026-11-27, statuses updated 2026-11-28. Reference: mlx-vlm at commit `e8035ae`
(https://github.com/Neywa-Labs/mlx-vlm, shallow clone at `/tmp/mlx-vlm`).

This page lists the vision-language models in mlx-vlm and says which half
TurboSpark lacks for each: the text family (the trunk), the vision tower, or
both. Use it to pick the next vision port. Each gap names the mlx-vlm file to
port from and the TurboSpark files to extend.

## Status legend

One status per model directory:

- `supported`: the text family and the vision tower are both wired, with a
  real-model gate or parity doc cited.
- `supported-unverified`: wired, but no real-weights gate was found in the docs
  read.
- `tower-only`: a vision tower exists for this model, but its text family is
  not wired.
- `family-only`: the text family runs, but there is no vision tower for this
  model.
- `missing`: neither half is wired.
- `out-of-scope`: not an image-to-text model (encoder, detector, depth,
  segmentation, generator, or retrieval embedding).

Read `supported` with care. In most rows it means a GGUF trunk plus a
`pull-vision` sidecar, not a combined HF or MLX install. Each supported row
lists its limits.

## Method

- **Scope.** The audit covers 127 mlx-vlm model directories: vision-language
  models plus the vision-only and generation models classified out of scope. A
  heuristic found them: a vision-named file, or a main model file that imports
  a vision module (`mistral3`, `unlimited_ocr`). A few VLMs may be missing from
  the list.
- **Grouping.** One read-only audit agent per family group. Each compared its
  mlx-vlm directories with `crates/model-io` (`VisionTowerKind`),
  `crates/repack` (`arch_registry.rs`, checkpoint readers), `crates/vision-io`,
  `crates/runtime` (`vision/`, `families/`, `real_forward_vision_api.rs`) and
  `docs/`. Nothing was executed.
- **Spot checks.** Before the first merge, the key claims were checked against
  source. Several gaps found then have since been closed; the tables show the
  current status.
- **Depth.** Some rows rest on docs or on source reads only. Those rows say so,
  and they are collected under [Unverified](#unverified-and-open-questions).

## Summary

| # | Group | Dirs | supported | tower-only | family-only | missing | out-of-scope |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Qwen-VL lineage | 9 | 7 | 0 | 0 | 1 | 1 |
| 2 | Gemma and PaliGemma | 5 | 1 (+1) | 0 | 1 | 2 | 0 |
| 3 | LLaVA, Idefics, SmolVLM, Llama vision | 11 | 3 (+2) | 0 | 2 | 2 | 2 |
| 4 | InternVL, Phi, Granite, Aya, Nemotron | 12 | 1 | 1 | 0 | 10 | 0 |
| 5 | Mistral and Pixtral | 6 | 1 | 1 | 0 | 4 | 0 |
| 6 | DeepSeek, Janus, Unlimited OCR | 7 | 2 (+1) | 0 | 2 | 2 | 0 |
| 7 | OCR and document parsers | 11 | 0 | 1 | 4 | 4 | 2 |
| 8 | MiniCPM, MoonDream, Molmo, Jina, FastVLM | 10 | 2 | 0 | 1 | 7 | 0 |
| 9 | Chinese and other lab VLMs | 16 | 0 (+1) | 0 | 3 | 12 | 0 |
| 10 | LFM, Inkling, Muse, D1 | 9 | 4 | 0 | 0 | 5 | 0 |
| 11 | Vision-only and image generation | 31 | 0 | 1 | 0 | 1 | 29 |
| | **Total** | **127** | **21 (+5)** | **4** | **13** | **50** | **34** |

A `(+n)` counts `supported-unverified` rows. Of the 127 directories, 93 are
image-to-text models in scope (127 minus 34 out-of-scope): 21 supported, 5
supported-unverified, 4 tower-only, 13 family-only and 50 missing. The
`supported` count includes the sidecar-only rows.

The `internvl` and `internvl_chat` wrapper strings are deliberately absent from
`SUPPORTED_HF`, because an InternVL checkpoint's trunk is whatever
`text_config.model_type` says
([INTERNVL_VISION_PHASE0.md](INTERNVL_VISION_PHASE0.md)).

## Where to start

**Wired, real-weights gate owed.** Close these before porting more:

- `gemma4_unified`, `llava_next`, `mage_vl` (still images): synthetic tests
  only. Each Phase 0 page lists what its gate needs.
- `idefics3`: the port is smoked on SmolVLM-256M; the 8B has not run.
- `internvl_chat`: the CLI is real-gated; the server and FFI image paths have
  run on a synthetic install only.
- `unlimited_ocr`: no real-weights gate beyond oracles.

**Next ports**, ordered by the effort estimates in the group sections:

- Small or medium, reuses a tower we have: `mistral3` (the Pixtral tower plus
  the 2x2 merger), `multi_modality` (Janus), `deepseekocr_2`, the `minicpm_ocr`
  gate, the `indic_ocr` block-OCR checkpoint.
- Large or blocked here: `mistral_large3`, `mistral_large4`, `mistral4`,
  `kimi_k25`, `kimi_k3`, `deepseek_v4` and `deepseek_v41` (284B-class),
  `muse_spark` (816B), `glm4v_moe`, `llama4` (expert slot cache too large), and
  `inkling` v1 (confirm public weights before starting).

## Unverified and open questions

- **Shallow pass, group 4.** The Phi, Granite, Aya and RADIO "no matching
  tower" verdicts rest on the `VisionTowerKind` doc comments and repo-wide
  searches. The tower files were not read.
- **`llava_bunny`.** The HF `model_type` spelling and text backbone are not
  verified.
- **`minicpmv4_6`.** One real-weights smoke; no tower parity, no server run.
- **`fastvlm`.** Source reads only. The `llava_qwen2` registry
  fallthrough for `fastvlm` was not tested.
- **`dots_ocr`, `oxr`.** The Qwen2 and Qwen3 trunk fit was judged from mlx-vlm
  imports, not from a config comparison.
- **`deepseek_v41`.** Text side (`engram.py`, `fakequant.py`) not checked.
  `unlimited_ocr` is supported-unverified: no real-weights gate beyond oracles.
- **`kimi_vl`, `kimi_k25`.** The text path (Deepseek2) is GGUF-only. HF
  `deepseek_v3` is not in `SUPPORTED_HF`, so HF intake is unverified.
- **`sensenova_loop`.** `model_type` inferred from its Qwen3.5 inheritance.
  `cohere_compass`: reuse of the Qwen3-VL tower unverified.
- **`qwen3_vl_moe`, `qwen4_exp`.** Sidecar-only. The `qwen3_vl_moe` tower was
  not re-measured on this checkpoint.
- **`gemma4_unified`.** No real weights run.
- **`mistral3`.** Whether the `llama` GGUF string needs a new registry entry was
  not settled.


## 1. Qwen-VL lineage

Summary: 7 supported (qwen2_vl, qwen2_5_vl, qwen3_vl, qwen3_vl_moe with caveat, qwen3_5, qwen3_5_moe, qwen4_exp with caveat), 1 missing (qwen3_omni_moe), 1 out-of-scope (qwen3_vl_embedding). No row is tower-only or family-only. Evidence is from docs (MODEL_FAMILY.md vision table, per-tower Phase 0 pages) and registry/enum source; no real-weight gate was re-run by this audit.

| mlx-vlm dir | HF model_type | mlx-vlm vision file | TurboSpark text family | TurboSpark vision tower | Status | Files to read |
|---|---|---|---|---|---|---|
| qwen2_vl | qwen2_vl | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/qwen2_vl/vision.py) | Qwen2Dense (registry row, exact equality beside `qwen2` and `qwen2_5_vl`) | `Qwen2Vl` (runtime/vision/qwen2.rs: LayerNorm blocks, plain quick-GELU MLP, full attention on every block, LayerNorm merger `ln_q`) | supported (2026-11-27, 2B combined install; text and image smokes; no tower-stage parity probe) | ../docs/QWEN2VL_VISION_PHASE0.md, ../crates/repack/src/qwen36_config.rs, ../crates/runtime/src/vision/qwen2.rs |
| qwen2_5_vl | qwen2_5_vl | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/qwen2_5_vl/vision.py) | Qwen2Dense (registry row line 300) | `Qwen25Vl` (runtime/vision/qwen25.rs) | supported | ../docs/QWEN25_VL_VISION_PHASE0.md, ../docs/MODEL_FAMILY.md (7B 4-bit combined install gated 2026-11-01; tower numpy parity, mlx-vlm not executed) |
| qwen3_vl | qwen3_vl | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/qwen3_vl/vision.py) | Qwen3Vl (rows `qwen3_vl`, `qwen3_vl_text`) | `QwenSiglip` + 3 deepstack mergers | supported | ../docs/QWEN3VL_PHASE0.md (4B real install, 7 tower stages cosine 0.99999+ vs mlx-vlm), ../crates/runtime/tests/qwen3vl_vision.rs |
| qwen3_vl_moe | qwen3_vl_moe | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/qwen3_vl_moe/vision.py) | Qwen3Moe (rows exist, but HF safetensors trunk has no writer; runs from GGUF `qwen3vlmoe`) | `QwenSiglip` + deepstack (depth 27, blocks 8/16/24), sidecar via `pull-vision` | supported (GGUF trunk + sidecar only; tower not re-measured on this checkpoint) | ../docs/MODEL_FAMILY.md (line 404, real-gated 2026-11-01), ../crates/repack/src/arch_registry.rs |
| qwen3_vl_embedding | qwen3_vl (embedding head) | [qwen3_vl_embedding.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/qwen3_vl_embedding/qwen3_vl_embedding.py) | Qwen3Vl trunk exists | `QwenSiglip` exists | out-of-scope (embedding model: no lm_head, last-token pooling and L2 normalize; subclasses qwen3_vl) | ../crates/embeddings/src/lib.rs, ../crates/embeddings/MODELS.md |
| qwen3_5 | qwen3_5 / qwen3_5_text | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/qwen3_5/vision.py) | QwenGdnDense | `QwenSiglip` (default kind) | supported | ../docs/VISION.md, ../docs/VISION_PHASE0.md, ../docs/MODEL_FAMILY.md (real-gated, CLI/server/FFI) |
| qwen3_5_moe | qwen3_5_moe / qwen3_5_moe_text | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/qwen3_5_moe/vision.py) | QwenGdnMoe | `QwenSiglip` (depth 27) | supported | ../docs/MODEL_FAMILY.md (verified 2026-09-19, 4-stage mlx-vlm cosine 0.99999+, Qwen3.6-35B-A3B); chunked prefill with image still refused |
| qwen3_omni_moe | qwen3_omni_moe | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/qwen3_omni_moe/vision.py) | none (thinker is a Qwen3 MoE with audio encoder, talker, code2wav; no `qwen3_omni_moe` string anywhere in crates) | none registered (tower is the Qwen3-VL-MoE kind with deepstack, so `QwenSiglip` is a likely fit) | missing | ../crates/repack/src/arch_registry.rs, ../crates/runtime/src/families/qwen |
| qwen4_exp | qwen4_exp / qwen4_exp_text | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/qwen4_exp/vision.py) | Qwen4Exp | `QwenSiglip` (qwen3_5 tower, 4-bit quantized, out_hidden 2560; repack dequantizes to FP16) | supported (sidecar only; stage-2 parity only) | ../docs/QWEN4EXP_VISION_PHASE0.md, ../docs/MODEL_FAMILY.md (line 394) |

### Gaps

- qwen2_vl: DONE 2026-11-27 (was: both halves partly missing). The tower landed as `VisionTowerKind::Qwen2Vl` over the Qwen2Dense trunk (`docs/QWEN2VL_VISION_PHASE0.md`), reusing the 2.5 pipeline's patchify, 2-D rope and chunked mRoPE; the block deltas are LayerNorm norms with biases, a plain two-layer quick-GELU MLP, and full attention on every block. Remaining gap: no tower-stage parity probe against a transcription reference (the end-to-end image gate passed even under a wrong mRoPE selector, so it does not pin that seam; see the honest note in the Phase 0 page).
- qwen3_vl_moe: the vision half is gated, but the trunk only runs from the official GGUF plus a `pull-vision` sidecar. An MLX/HF safetensors `qwen3_vl_moe` checkpoint cannot be installed combined (no safetensors writer for routed experts on the llama MoE flow), and the tower was not re-measured against a reference here. Source: https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/qwen3_vl_moe/vision.py. Extend ../crates/repack/src/arch_registry.rs and the Qwen3Moe repack path. Effort M: expert-weight safetensors writer, chunked-prefill image injection.
- qwen3_vl_embedding: not a generation VLM, so out of scope for this audit. If wanted, add a multimodal embedding head (last-token pooling, normalize, no lm_head) over the existing Qwen3Vl trunk and tower. Source: https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/qwen3_vl_embedding/qwen3_vl_embedding.py. Files: ../crates/embeddings/src/lib.rs, ../crates/runtime/src/real_forward_vision_api.rs. Effort M: pooling head and prompt template, but needs a vision-conditioned hidden-state export.
- qwen3_omni_moe: neither half. Needs a Qwen3 MoE thinker family row (qwen3_omni_moe thinker config nesting), the audio encoder, and optionally the talker/code2wav speech path; vision itself is the Qwen3-VL-MoE tower with deepstack. Source: https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/qwen3_omni_moe/vision.py (plus thinker.py, audio.py). Extend ../crates/repack/src/arch_registry.rs, ../crates/repack/src/qwen3vl_config.rs, ../crates/runtime/src/vision/mod.rs. Effort L: thinker-only image support is M, but audio/talker make the full model L.
- qwen4_exp: supported only as a `pull-vision` sidecar on the text-only install. No combined install, no stage 4 logit parity, no server run; image prompts refused where QSA block selection could run (context above the indexer budget). Source: https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/qwen4_exp/vision.py. Extend ../crates/runtime/src/vision/siglip.rs and ../crates/runtime/src/families/qwen4. Effort M: QSA-safe image prefill and combined intake.

## 2. Gemma and PaliGemma lineage

Summary: 5 dirs: 1 supported (gemma4), 1 supported-unverified (gemma4_unified), 1 family-only (gemma3), 2 missing (gemma3n, paligemma).

| mlx-vlm dir | HF model_type | mlx-vlm vision file | TurboSpark text family | TurboSpark vision tower | Status | Files to read |
|---|---|---|---|---|---|---|
| gemma3 | gemma3 (text_config gemma3_text) | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/gemma3/vision.py) | ModelFamily::Gemma3, GGUF only, text-only (no HF row; `gemma3_text` safetensors path unimplemented) | none (SigLIP so400m 896px + avg-pool projector; existing `SigLip` kind is not wired to Gemma3) | family-only | ../crates/repack/src/arch_registry.rs, ../docs/GEMMA3_PHASE0.md |
| gemma3n | gemma3n | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/gemma3n/vision.py) | none (named only as "GGUF-witnessed, unregistered", [docs/MODEL_FAMILY.md](../docs/MODEL_FAMILY.md):334) | none | missing | ../docs/MODEL_FAMILY.md |
| gemma4 | gemma4 / gemma4_text | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/gemma4/vision.py) | ModelFamily::Gemma4 | VisionTowerKind::Gemma4, ../crates/runtime/src/vision/gemma4.rs, trunk in runtime/src/families/gemma4/vision_prefill.rs | supported (real-model gate: "TURBOSPARK 42" read on gemma-4-26b-a4b-it-4bit via FFI/server tests, [docs/GEMMA4_VISION_PHASE0.md](../docs/GEMMA4_VISION_PHASE0.md) "Implementation status") | ../docs/GEMMA4_VISION_PHASE0.md, ../crates/runtime/tests/gemma4_vision_real_trunk.rs |
| gemma4_unified | gemma4_unified (text gemma4_unified_text) | none separate; encoder-free embedder inside [gemma4_unified.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/gemma4_unified/gemma4_unified.py) (`VisionEmbedder`) | ModelFamily::Gemma4 (HF rows `gemma4_unified`, `gemma4_unified_text`) | `VisionTowerKind::Gemma4Unified`, ../crates/runtime/src/vision/gemma4_unified.rs, trunk shared with gemma4 | supported-unverified (wired 2026-11-27, synthetic tests only; combined MLX installs, 280-token budget, no sidecar, no audio or video, BF16 `google/*` repos refused) | [docs/GEMMA4_UNIFIED_VISION_PHASE0.md](../docs/GEMMA4_UNIFIED_VISION_PHASE0.md) |
| paligemma | paligemma (text gemma / gemma2) | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/paligemma/vision.py) | none (no `paligemma` or Gemma-1 `gemma` HF/GGUF row; Gemma2 exists GGUF-only) | none (SigLIP kind exists but is not wired to a PaliGemma family) | missing | ../crates/model-io/src/arch_config/family.rs |

### Gaps

- gemma3: vision tower missing, plus the text path is GGUF-only (no safetensors/mlx-community intake, which is what VLM checkpoints ship as). Port from https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/gemma3/vision.py (and gemma3.py for the projector, language.py for the image-token bidirectional mask). Extend ../crates/runtime/src/vision/siglip.rs (check reuse vs. the Qwen SigLIP block: Gemma3 is a plain SigLIP with learned pos-embed, 896px, 256 soft tokens via 4x4 avg pool + RMSNorm projector), ../crates/model-io/src/arch_config/sub_configs.rs, ../crates/repack/src/arch_registry.rs (add `gemma3`, `gemma3_text` HF rows with the +1 norm bake), ../crates/runtime/src/real_forward_vision_api.rs, and a new vision-io preprocessor under ../crates/vision-io/src/. Effort L: needs HF intake, norm bake, tower wiring, and the Gemma 4-style span-bidirectional prefill mask; the Gemma 4 vision path is the template.
- gemma3n: both halves missing. Port from https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/gemma3n/vision.py (MobileNetV5-style conv tower, not SigLIP) and language.py (AltUp, LAUREL, per-layer embeddings, KV sharing). Extend ../crates/model-io/src/arch_config/family.rs and sub_configs.rs, ../crates/repack/src/arch_registry.rs, ../crates/runtime/src/families/ (new family), ../crates/runtime/src/vision/ (new tower), ../crates/vision-io/src/. Effort L: new text architecture and a new conv vision tower, nothing reusable. Audio tower (audio.py) is extra scope.
- gemma4_unified: WIRED (2026-11-27), see [docs/GEMMA4_UNIFIED_VISION_PHASE0.md](../docs/GEMMA4_UNIFIED_VISION_PHASE0.md). Still owed: the real-weights install, tower parity, an image prompt and the memory oracle.
- paligemma: both halves missing (Gemma 1/2 text with a prefix-LM mask over image + prompt, `gemma` model_type). Port from https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/paligemma/vision.py (SigLIP so400m, 224/448/896) and paligemma.py (linear projector, no pooling), language.py. Extend ../crates/repack/src/arch_registry.rs (new HF row; Gemma2 family exists GGUF-only per ../docs/MODEL_FAMILY.md), ../crates/runtime/src/vision/siglip.rs, ../crates/model-io/src/arch_config/family.rs and sub_configs.rs, ../crates/runtime/src/real_forward_vision_api.rs. Effort L: new HF intake for Gemma 1/2 and the prefix-bidirectional mask, though the SigLIP tower may be partly reusable.

## 3. LLaVA, Idefics, SmolVLM, Llama vision

Summary: 11 dirs: supported 3 (llava, llava_onevision, both only as GGUF trunk plus vision sidecar, and smolvlm), supported-unverified 2 (idefics3: same port, the 8B not run; llava_next: synthetic tests only), family-only 2 (llava_bunny, idefics2), missing 2 (mllama, llama4), out-of-scope 2 (colidefics3, llama_nemotron_vl_embedding).

| mlx-vlm dir | HF model_type | mlx-vlm vision file | TurboSpark text family | TurboSpark vision tower | Status | Files to read |
|---|---|---|---|---|---|---|
| llava | llava (text llama) | llava/vision.py (CLIP) | Llama, GGUF trunk only (arch_registry comment: the HF `llama` parser refuses the llava wrapper) | VisionTowerKind::Clip, wired in real_forward_vision_api.rs | supported (sidecar route; real-gated 2026-11-01, [docs/MODEL_FAMILY.md](../docs/MODEL_FAMILY.md) Llama row, [docs/LLAVA_VISION_PHASE0.md](../docs/LLAVA_VISION_PHASE0.md); HF safetensors intake of the trunk is not supported) | ../crates/runtime/src/vision/clip.rs, ../crates/vision-io/src/clip.rs, ../crates/repack/src/gemma4_checkpoint/vision_clip.rs |
| llava_next | llava_next | llava_next/vision.py (CLIP) | Llama, GGUF trunk only | `VisionTowerKind::ClipNext`, runtime/vision/clip.rs plus vision-io/llava_next.rs | supported-unverified (sidecar route; synthetic tests only, no real-weights run; [LLAVA_NEXT_VISION_PHASE0.md](LLAVA_NEXT_VISION_PHASE0.md)) | ../crates/repack/src/gemma4_checkpoint/vision_clip.rs, ../crates/vision-io/src/llava_next.rs |
| llava_bunny | bunny-llava (unverified spelling) | llava_bunny/vision.py (SigLIP) | Llama/Qwen2 text possible, no bunny intake | none for Bunny (SigLipOv and SigLip exist but are shaped for other checkpoints) | family-only (unverified) | ../crates/runtime/src/vision/siglip.rs |
| llava_onevision | llava_onevision | llava_onevision/vision.py (SigLIP) | Qwen2Dense, GGUF trunk | SigLipOv, anyres + image_newline | supported (sidecar route; real-gated 2026-11-02 but partial: stray digit and dense 13-tile page degenerates, [docs/LLAVA_ONEVISION_VISION_PHASE0.md](../docs/LLAVA_ONEVISION_VISION_PHASE0.md); images only) | ../crates/runtime/src/vision/onevision.rs, ../crates/vision-io/src/onevision.rs |
| mllama | mllama | mllama/vision.py | Llama family exists, but no gated cross-attention layers | none | missing | ../crates/runtime/src/families/llama |
| llama4 | llama4 | llama4/vision.py | none runnable (PLANNED_GGUF llama4, refused: expert slot cache 58.4 GiB) | none | missing | ../crates/repack/src/arch_registry.rs |
| idefics2 | idefics2 (text mistral) | idefics2/vision.py (NaViT SigLIP + perceiver) | Llama (mistral row, GGUF only) | none (SigLip tower is fixed 384 so400m; no perceiver resampler) | family-only | ../crates/runtime/src/vision/siglip.rs |
| idefics3 | idefics3 (text llama) | idefics3/vision.py (SigLIP, variable size) | Llama (MLX safetensors, wrapper only) | `VisionTowerKind::SigLipNaViT`, runtime/vision/idefics3.rs | supported-unverified (same port as smolvlm; the 8B not run; [IDEFICS3_VISION_PHASE0.md](IDEFICS3_VISION_PHASE0.md)) | ../crates/runtime/src/vision/idefics3.rs |
| smolvlm | smolvlm (idefics3 subclass) | reuses idefics3/vision.py | Llama (SmolLM2, MLX safetensors) | shared with idefics3 | supported (one smoke on SmolVLM-256M 4-bit, 2026-11-27; same record) | same as idefics3 |
| colidefics3 | colidefics3 | reuses idefics3 | n/a | n/a | out-of-scope: ColBERT-style multi-vector retrieval, no generation | n/a |
| llama_nemotron_vl_embedding | llama_nemotron_vl_embedding | granite_vision/vision.py | n/a | n/a | out-of-scope: pooled embedding model, no generation | n/a |

### Gaps

- llava_next: wired 2026-11-27. Owed: a real-weights gate (files and sizes in [LLAVA_NEXT_VISION_PHASE0.md](LLAVA_NEXT_VISION_PHASE0.md)).
- llava_bunny: SigLIP tower variant and its own projector. Port from https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/llava_bunny/vision.py and llava_bunny.py. Extend ../crates/runtime/src/vision/siglip.rs and ../crates/repack/src/gemma4_checkpoint/vision.rs. Effort M. The model_type spelling and text backbone were not verified.
- idefics2: needs a variable-size SigLIP (NaViT bucketed positions) plus perceiver resampler. Port from https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/idefics2/vision.py. Extend ../crates/runtime/src/vision/ (new module), ../crates/vision-io/src/ (new preprocessing), ../crates/model-io/src/arch_config/sub_configs.rs (new VisionTowerKind). Effort L: new tower plus resampler, and the trunk is GGUF only.
- idefics3 and smolvlm: IMPLEMENTED (2026-11-27), one port for both ([IDEFICS3_VISION_PHASE0.md](IDEFICS3_VISION_PHASE0.md)). Smoked on SmolVLM-256M only. Still owed: an Idefics3-8B run, processor pixel parity, a tower-stage comparison, the server splice and a vision sidecar. SmolVLM2 video is not covered.
- mllama: needs gated cross-attention decoder layers (new text flow) and a two-stage tiled vision encoder. Port from https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/mllama/language.py and vision.py. Extend ../crates/runtime/src/families/ and ../crates/runtime/src/vision/. Effort L: cross-attention KV is a new cache concept.
- llama4: refused text (large MoE streaming, chunked attention, rope scaling) and no tower. Port from https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/llama4/language.py and vision.py. Extend ../crates/repack/src/arch_registry.rs, ../crates/runtime/src/families/. Effort L (both halves, plus slot-cache memory blocker).
- Cross-cutting note: llava and llava_onevision work only as GGUF trunk plus pull-vision sidecar. An HF safetensors/MLX install of either is not supported. idefics2 still has only the GGUF text path; idefics3 and smolvlm install their MLX conversions.
- Not checked: whether PLANNED/SUPPORTED_GGUF has per-arch rows for the idefics text backbones beyond llama/mistral; llava_bunny backbone.

## 4. InternVL, Phi, Granite, Aya, Nemotron

Summary: 12 dirs checked: 1 supported (internvl_chat), 1 tower-only (internvl), 0 family-only, 10 missing, 0 out-of-scope. Text-family evidence is from `SUPPORTED_HF` / `SUPPORTED_GGUF` in `crates/repack/src/arch_registry.rs`; no vision tower kind in `VisionTowerKind` (`crates/model-io/src/arch_config/sub_configs.rs`) matches any Phi, Granite, Aya or RADIO tower, and `st` finds no docs or runtime code naming any of these (the only hits are audio models and `docs/MODEL_COVERAGE.md` listing `nemotron_h` as planned). The InternViT gap closed 2026-11-27 ([docs/INTERNVL_VISION_PHASE0.md](INTERNVL_VISION_PHASE0.md)).

| mlx-vlm dir | HF model_type | mlx-vlm vision file | TurboSpark text family | TurboSpark vision tower | Status | Files to read |
|---|---|---|---|---|---|---|
| internvl | internvl (text: qwen3, per `from ..qwen3.language import LanguageModel`) | internvl/vision.py (InternViT: patch embed, cls, learned pos, layer scale) | Qwen3Dense via HF `qwen3` (resolved through `text_config.model_type`; the wrapper has deliberately NO SUPPORTED_HF row, and `parse_qwen3_config` accepts the wrapper's `text_config` only under root `internvl`) | `VisionTowerKind::InternViT`, runtime/vision/internvl.rs | tower-only (config path verified: probe of `mlx-community/InternVL3_5-1B-4bit@f9d179a8` declares internvl, family qwen3, then refuses at the existing group-32 kernel gate; the only published plain-flavor MLX conversion, so no install gate) | [docs/INTERNVL_VISION_PHASE0.md](../docs/INTERNVL_VISION_PHASE0.md), [crates/repack/src/gemma4_checkpoint/vision_internvl.rs](../crates/repack/src/gemma4_checkpoint/vision_internvl.rs) |
| internvl_chat | internvl_chat (text: qwen2 per language.py) | internvl_chat/vision.py | Qwen2Dense via HF `qwen2` (same wrapper-row policy) | `VisionTowerKind::InternViT`, runtime/vision/internvl.rs | supported (real-gated 2026-11-27 on `mlx-community/InternVL3-1B-4bit@7927bb6b`: probe, streamed install with the 24-block tower, greedy text smoke, page transcription with a no-image control, re-run 2026-11-27 after a projector buffer-overflow fix: a large-text page reads exactly, a dense page drifts into repetition; [docs/INTERNVL_VISION_PHASE0.md](../docs/INTERNVL_VISION_PHASE0.md)). Limits: combined install only, no tower-stage parity probe, no quality gate or memory oracle, server/FFI wired but tested on a synthetic install only, the chat template is text-only so image turns flatten parts to `<img>` markers | ../docs/INTERNVL_VISION_PHASE0.md, ../crates/repack/src/gemma4_checkpoint/vision_internvl.rs, ../crates/runtime/src/vision/internvl.rs, ../crates/vision-io/src/internvl.rs |
| phi3_v | phi3_v | phi3_v/vision.py (CLIP ViT-L/336 + HD transform) | GGUF `phi3` -> Llama only; no HF `phi3` row | `Clip` kind is LLaVA-1.5 specific (penultimate layer, 577 pos); no HD/sub-image glue | missing | [crates/repack/src/arch_registry.rs](../crates/repack/src/arch_registry.rs), [docs/LLAVA_VISION_PHASE0.md](../docs/LLAVA_VISION_PHASE0.md) |
| phi4mm | phi4mm | phi4mm/vision.py (SigLIP2) | no HF `phi4mm`; Phi3-style trunk with LoRA adapters | none (SigLIP2 NaFlex absent) | missing | [crates/repack/src/arch_registry.rs](../crates/repack/src/arch_registry.rs) |
| phi4_siglip | phi4-siglip | phi4_siglip/vision.py (SigLIP2) | no HF row (Phi3 class in language.py) | none | missing | [crates/repack/src/arch_registry.rs](../crates/repack/src/arch_registry.rs) |
| granite_vision | granite_vision (text granite) | granite_vision/vision.py (SigLIP, llava-next anyres) | GGUF `granite` -> Llama only; no HF `granite` row | none (`SigLip` kind is DeepSeek-VL2 specific) | missing | [crates/repack/src/arch_registry.rs](../crates/repack/src/arch_registry.rs), [docs/LLAVA_ONEVISION_VISION_PHASE0.md](../docs/LLAVA_ONEVISION_VISION_PHASE0.md) |
| granite4_vision | granite4_vision (text granitemoehybrid) | granite4_vision/vision.py + qformer.py + downsampling.py | none (granitemoehybrid not registered) | none | missing | [crates/repack/src/arch_registry.rs](../crates/repack/src/arch_registry.rs) |
| granite_docling | granite_docling (text granite_docling_text, Idefics3-style) | granite_docling/vision.py | none registered | none | missing | [crates/repack/src/arch_registry.rs](../crates/repack/src/arch_registry.rs) |
| aya_vision | aya_vision (text cohere2) | aya_vision/vision.py (SigLIP) | none (cohere2 only exists as an audio LM in [crates/audio/src/lm/cohere2.rs](../crates/audio/src/lm/cohere2.rs)) | none | missing | [crates/audio/src/lm/cohere2.rs](../crates/audio/src/lm/cohere2.rs) |
| nemotron_h_nano_omni | nemotron_h_nano_omni (vision radio, audio parakeet) | nemotron_h_nano_omni/vision.py (RADIO ViT) | none; `nemotron_h` is listed unregistered in [docs/MODEL_COVERAGE.md](../docs/MODEL_COVERAGE.md) and [docs/MODEL_FAMILY.md](../docs/MODEL_FAMILY.md) | none | missing | [docs/MODEL_COVERAGE.md](../docs/MODEL_COVERAGE.md), [docs/MODEL_FAMILY.md](../docs/MODEL_FAMILY.md) |
| nemotron_parse | nemotron_parse | nemotron_parse/vision.py (RADIO + neck) | none (mBART-style encoder-decoder, no family) | none | missing | [crates/runtime/src/families](../crates/runtime/src/families)/ |
| nemotron_parse_v3_5 | nemotron_parse_v3_5 | nemotron_parse_v3_5/vision.py (subclass of RadioVisionModel) | none | none | missing | same as nemotron_parse |

Counts (authoritative): family-only 0, missing 10, supported 1, supported-unverified 0, tower-only 1, out-of-scope 0. nemotron_parse (both) is an OCR parser with a document-token decoder; it is still image-to-text, so it is classed missing, not out-of-scope.

### Gaps

- internvl: TOWER AND TEXT PATH DONE (2026-11-27), shared with internvl_chat; see [docs/INTERNVL_VISION_PHASE0.md](../docs/INTERNVL_VISION_PHASE0.md). The wrapper has NO SUPPORTED_HF row on purpose: an InternVL checkpoint's trunk is whatever `text_config.model_type` says, and a wrapper row would resolve a config whose trunk the registry does not know (`internlm2` and friends) into Qwen. Still missing: a real install gate (the only published plain-flavor MLX conversion quantizes 4-bit at group 32, which the kernels refuse) and a tower-stage parity probe.
- internvl_chat: DONE (2026-11-27). One InternViT tower shared with internvl, wired end to end (two tensor flavors in one reader, fused qkv at repack for the plain one, layer-scale kernel, dynamic tiling, pixel shuffle, flat injection through `set_prompt_vision_flat`) and real-gated on `InternVL3-1B-4bit`. Still owed: a quality gate, a memory oracle, and a real-weights server/FFI run (both paths are wired and synthetic-tested).
- phi3_v: both missing. Source https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/phi3_v/vision.py (CLIP with HD transform and sub-image newline embeddings). Text: add HF `phi3` to ../crates/repack/src/arch_registry.rs (GGUF path exists with LongRoPE per `derive_phi3_extensions`). Tower: reuse ../crates/runtime/src/vision/clip.rs only after a feature-layer and HD-glue change. Effort L: HD tiling plus a new projector and token glue.
- phi4mm: both missing. Source https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/phi4mm/vision.py (SigLIP2 NaFlex) and language.py (LoRA-bearing Phi3). Needs LoRA merge at repack. Effort L: new tower, LoRA, plus audio branch to refuse.
- phi4_siglip: both missing. Source https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/phi4_siglip/vision.py. Phi3 trunk via HF `phi3` row, SigLIP2 tower new in ../crates/runtime/src/vision/siglip.rs. Effort M.
- granite_vision: both missing. Source https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/granite_vision/vision.py. Text: add HF `granite` (needs embedding/residual/logit scales, which the Llama flow cannot express for 3.x per the note at arch_registry.rs line 209). Tower: SigLIP variant of ../crates/runtime/src/vision/siglip.rs with llava-next anyres from ../crates/vision-io/src/onevision.rs. Effort L.
- granite4_vision: both missing. Source https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/granite4_vision/vision.py plus qformer.py and downsampling.py. `granitemoehybrid` is a Mamba/MoE hybrid with no family. Effort L.
- granite_docling: both missing. Source https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/granite_docling/vision.py (SigLIP2 plus DeepStackMerger connector). Effort L: new text mapping and tower.
- aya_vision: both missing. Source https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/aya_vision/vision.py. Needs a Cohere2 text family in ../crates/runtime/src/families/ and `ModelFamily` in ../crates/model-io/src/arch_config/. Effort L.
- nemotron_h_nano_omni: both missing. Source https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/nemotron_h_nano_omni/vision.py (RADIO). `nemotron_h` is the biggest blocker (Mamba2 hybrid, see ../docs/MODEL_COVERAGE.md). Effort L.
- nemotron_parse and nemotron_parse_v3_5: both missing. Sources https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/nemotron_parse/vision.py and https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/nemotron_parse_v3_5/vision.py (RADIO plus neck), decoder in language.py (encoder-decoder, cross-attention, not a decoder-only flow). Effort L: needs a new cross-attention runtime flow.

Unverified: I did not read ../crates/runtime/src/real_forward_vision_api.rs dispatch or each tower file in depth; "no matching tower" rests on the `VisionTowerKind` doc comments and a repo-wide `st` for the model names.

## 5. Mistral and Pixtral

Summary: 1 supported (pixtral, via a GGUF `llama` trunk plus a `pull-vision` sidecar, real-gated on Q4_K_M only), 0 supported-unverified, 1 tower-only (mistral3), 0 family-only, 4 missing (mistral_large3, mistral_large4, mistral4, ministral3). Note the TurboSpark HF registry has no `pixtral` or `mistral3` string; Pixtral works only because ggml's GGUF is `llama`-arch.

| mlx-vlm dir | HF model_type | mlx-vlm vision file | TurboSpark text family | TurboSpark vision tower | Status | Files to read |
|---|---|---|---|---|---|---|
| pixtral | pixtral (text_config: mistral) | pixtral/vision.py | Llama (GGUF intake only; no HF `pixtral` row in SUPPORTED_HF) | VisionTowerKind::Pixtral, wired in real_forward_vision_api.rs | supported (GGUF Q4_K_M trunk + bf16 sidecar; real gate in [docs/PIXTRAL_VISION_PHASE0.md](../docs/PIXTRAL_VISION_PHASE0.md), mlx-vlm never executed, numpy reference) | ../docs/PIXTRAL_VISION_PHASE0.md, ../crates/runtime/src/vision/pixtral.rs, ../crates/vision-io/src/pixtral.rs, ../crates/repack/src/gemma4_checkpoint/vision_pixtral.rs |
| mistral3 | mistral3 (text_config mistral/ministral3/mistral4) | pixtral/vision.py (imported; mistral3 has no vision.py) | Llama would cover the 24B dense trunk via GGUF only; HF `mistral3` unregistered | Pixtral tower exists, but the Mistral3 merger and norm are not built | tower-only | ../docs/PIXTRAL_VISION_PHASE0.md (section "Mistral3", "Not attempted"), ../docs/MODEL_FAMILY.md |
| mistral_large3 | mistral_large3 (vision model_type pixtral) | pixtral/vision.py | Deepseek2 (MLA MoE) per arch_registry comment, no `mistral_large3` string and not run | Pixtral tower only | missing | ../crates/repack/src/arch_registry.rs, ../docs/DEEPSEEK2_PHASE0.md |
| mistral_large4 | mistral_large4 / mistral_large4_text | pixtral/vision.py | none (MLA, DSA, shared KV indexers, SSMax, LeRoPE, softcapping) | Pixtral tower only | missing | ../crates/runtime/src/families/ |
| mistral4 | mistral4 | none (language.py only, used under mistral3 wrapper) | none (MoE 128 experts top-4, MLA-style attention) | Pixtral via mistral3 wrapper | missing | ../crates/runtime/src/families/deepseek2 |
| ministral3 | ministral3 (text config only) | none (text-only wrapper; vision comes from mistral3) | none; GGUF string `mistral3` is unregistered (MODEL_FAMILY.md line 334) | Pixtral via mistral3 wrapper | missing | ../docs/MODEL_FAMILY.md, ../crates/repack/src/arch_registry.rs |

### Gaps

- mistral3: text family via HF is unregistered and the vision merger is unbuilt. The tower block code is reusable (Pixtral), but PIXTRAL_VISION_PHASE0 lists the delta: patch 14, spatial_merge_size 2, longest edge 1540, RMSNorm on tower output, 2x2 patch merger (`merging_layer`, Linear 4096 to 1024, no bias), bias-free projector. Port from https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/mistral3/mistral3.py and https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/mistral3/processing_mistral3.py (tower blocks: https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/pixtral/vision.py). Extend ../crates/runtime/src/vision/pixtral.rs, ../crates/vision-io/src/pixtral.rs, ../crates/repack/src/gemma4_checkpoint/vision_pixtral.rs, ../crates/model-io/src/arch_config/sub_configs.rs, and register the llama string in ../crates/repack/src/arch_registry.rs. Effort M: the doc calls it "a small delta", but needs a 24B trunk gate (40 layers, hidden 5120, so the sidecar pairing trap applies).
- mistral_large3: text needs the Deepseek2 MLA family on a 128-expert, 7168-wide, 61-layer shape (the registry notes it as unlocked by `deepseek2`, but no checkpoint is gated and there is no `mistral_large3` model_type string; Mistral-native params.json has no config.json). Vision is the Pixtral tower plus merger (see mistral3). Port from https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/mistral_large3/mistral_large3.py and https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/mistral_large3/config.py. Extend ../crates/repack/src/arch_registry.rs, ../crates/runtime/src/families/deepseek2, ../crates/runtime/src/vision/pixtral.rs. Effort L: a huge model that will not fit 36 GB, a params.json intake, and a new vision merger.
- mistral_large4: neither half. Text needs latent attention plus DSA, shared KV and indexers, NoPE, LeRoPE, SSMax and softcapping, which no TurboSpark family has. Vision is Pixtral encoder plus Mistral projector. Port from https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/mistral_large4/language.py and https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/mistral_large4/mistral_large4.py. Extend ../crates/runtime/src/families/ (new family), ../crates/model-io/src/arch_config/family.rs, ../crates/repack/src/arch_registry.rs. Effort L: a new attention family with custom kernels, not testable locally.
- mistral4: no text family (MoE with 128 experts top-4 and MLA-style attention, per https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/mistral4/language.py, ~70 GB at 4-bit, Mistral-Small-4-119B). It is only reachable through the mistral3 wrapper, so the vision half is the mistral3 delta. Extend ../crates/runtime/src/families/deepseek2 or a new family, ../crates/repack/src/arch_registry.rs. Effort L: new MoE and attention graph, too large to gate on this machine.
- ministral3: dense text-only config class (hidden, layers, tied embeddings, layer_types, sliding_window), with no vision code of its own; vision comes from the mistral3 wrapper. Port from https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/ministral3/language.py and https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/ministral3/config.py. The `mistral3` GGUF string and HF `ministral3`/`mistral3` are unregistered (../docs/MODEL_FAMILY.md line 334). Extend ../crates/repack/src/arch_registry.rs and ../crates/runtime/src/families/llama (sliding window and tied head, which the Llama parser refuses today for live windows). Effort M: dense, but needs layer_types and sliding-window handling plus the mistral3 vision delta.

Caveats: pixtral is marked supported on the strength of [docs/PIXTRAL_VISION_PHASE0.md](../docs/PIXTRAL_VISION_PHASE0.md) only (one quantization, two image grids, no portrait, multi-image, or multi-turn run). I did not verify that the code paths still pass at HEAD.

## 6. DeepSeek, Janus, Unlimited OCR

Summary: 7 dirs: 2 supported (deepseek_vl_v2, deepseekocr), 1 supported-unverified (unlimited_ocr), 2 family-only (deepseek_v4, deepseek_v41), 2 missing (deepseekocr_2, multi_modality).

| mlx-vlm dir | HF model_type | mlx-vlm vision file | TurboSpark text family | TurboSpark vision tower | Status | Files to read |
|---|---|---|---|---|---|---|
| deepseek_vl_v2 | deepseek_vl_v2 | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/deepseek_vl_v2/vision.py) | Deepseek2 (plain-attention arm), registry row `deepseek_vl_v2` | VisionTowerKind::SigLip (SigLIP-so400m + 2x2 unfold projector) | supported (tiny only; small and full untouched) | ../docs/DEEPSEEK_VL2_VISION_PHASE0.md, ../crates/runtime/src/vision/siglip.rs, ../crates/vision-io/src/deepseek_vl2.rs |
| deepseekocr | deepseekocr | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/deepseekocr/vision.py), [sam.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/deepseekocr/sam.py) | Deepseek2 (same trunk as VL2-tiny), registry row `deepseekocr` | VisionTowerKind::DeepseekOcr (SAM-ViT-B + CLIP-L + conv compress + projector) | supported (base mode only; Gundam not in this path) | ../docs/DEEPSEEK_OCR_VISION_PHASE0.md, ../docs/MODEL_FAMILY.md, ../crates/runtime/src/vision/deepseek_ocr.rs, ../crates/vision-io/src/deepseek_ocr.rs |
| deepseekocr_2 | deepseekocr_2 | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/deepseekocr_2/vision.py) | none: no `deepseekocr_2` row in SUPPORTED_HF (trunk would likely be Deepseek2, unconfirmed) | none: SAM then a Qwen2 encoder (visual causal flow) is a different tower from DeepseekOcr's CLIP-L | missing | ../crates/repack/src/arch_registry.rs, ../crates/model-io/src/arch_config/sub_configs.rs |
| deepseek_v4 | deepseek_v4 | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/deepseek_v4/vision.py) | DeepseekV4Flash (284B-A13B baseline; reached via GGUF path, no `deepseek_v4` HF row in arch_registry.rs) | none (`VisionConfig::NONE` in arch_baselines/deepseek.rs) | family-only | ../crates/model-io/src/arch_baselines/deepseek.rs, ../crates/repack/src/gguf_checkpoint/mod.rs |
| deepseek_v41 | deepseek_v41 | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/deepseek_v41/vision.py) | not verified: V4 Flash family exists but V4.1 adds engram/fakequant (engram.py); no V4.1 row found | none (reuses deepseek_v4 vision) | family-only (unverified: may be missing if V4.1 text differs) | ../crates/model-io/src/arch_baselines/deepseek.rs |
| unlimited_ocr | unlimited-ocr (config.py default says deepseek_v2) | none of its own: reuses deepseekocr SAM, CLIP vision ([unlimitedocr.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/unlimited_ocr/unlimitedocr.py)) | UnlimitedOcr2 (DASH, page2d rope; routed in arch_registry.rs) | own file in [crates/runtime/src/vision/unlimited_ocr2.rs](../crates/runtime/src/vision/unlimited_ocr2.rs), sidecar uocr2_tower.bin | supported-unverified for Unlimited-OCR-2 (synthetic and dump oracles cited, README flags OCR-only); v1 refused (R-SWA ring) | ../docs/UNLIMITED_OCR2_PHASE0.md, ../crates/runtime/src/families/unlimited_ocr2/, ../crates/runtime/src/vision/unlimited_ocr2.rs, ../crates/vision-io/src/unlimited_ocr2.rs |
| multi_modality | multi_modality (Janus / Janus-Pro) | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/multi_modality/vision.py), [sam.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/multi_modality/sam.py) | none: Llama family exists but no `multi_modality` row; no Janus references in crates or docs | none: SigLIP-L (+ SAM high-res for Janus-Pro/DeepSeek-VL) with MlpProjector aligner; existing SigLip kind is the so400m VL2 variant | missing | ../crates/repack/src/arch_registry.rs, ../crates/runtime/src/vision/siglip.rs |

### Gaps

- deepseekocr_2: both halves missing. Port from https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/deepseekocr_2/vision.py (and deepseekocr_2.py, processing_deepseekocr.py). Extend ../crates/repack/src/arch_registry.rs (add row, likely Deepseek2), ../crates/model-io/src/arch_config/sub_configs.rs (new VisionTowerKind), ../crates/runtime/src/vision/deepseek_ocr.rs (reuse SAM, replace CLIP with a Qwen2 encoder using learned queries), ../crates/vision-io/src/deepseek_ocr.rs (12x12 crops plus 16x16 global view). Effort M: SAM and trunk exist, the Qwen2 encoder with causal-flow queries is new.
- deepseek_v4: text family exists (DeepseekV4Flash, GGUF path only; no HF `deepseek_v4` registry row), vision missing. Port from https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/deepseek_v4/vision.py (2D-RoPE ViT, patch 14, dim 1024, downsample 3). Extend ../crates/model-io/src/arch_baselines/deepseek.rs (VisionConfig), a new tower in ../crates/runtime/src/vision/, ../crates/vision-io/src/. Effort L: new tower plus a 284B-scale model that cannot be gated on this 36 GB Mac.
- deepseek_v41: same vision as v4 (imports deepseek_v4.vision). Source https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/deepseek_v41/vision.py. Text side (engram.py, fakequant.py in the same dir) was not checked against TurboSpark and may need a family variant. Files: ../crates/model-io/src/arch_baselines/deepseek.rs, ../crates/repack/src/arch_registry.rs. Effort L: depends on deepseek_v4 tower plus unverified text differences.
- multi_modality (Janus): both halves missing. Port from https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/multi_modality/vision.py and sam.py. Extend ../crates/repack/src/arch_registry.rs (row to Llama), ../crates/model-io/src/arch_config/sub_configs.rs, ../crates/runtime/src/vision/siglip.rs (SigLIP-L variant, AttentionPoolLatent, SAM high-res fusion), a new preprocessor in ../crates/vision-io/src/. Effort M: Llama trunk exists and SigLIP/SAM pieces can be reused; generation (VQ image output) is out of scope for understanding-only support.
- unlimited_ocr (supported-unverified): no real-weights gate beyond oracles was found in the docs read; ../docs/UNLIMITED_OCR2_PHASE0.md lists cross-page block stitching as open. Unlimited-OCR v1 (R-SWA ring) is refused at the door. Source https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/unlimited_ocr/dash.py. Effort S to close the verification gap, M for v1.
- deepseek_vl_v2 / deepseekocr caveats: VL2 small and full are untouched (../docs/DEEPSEEK_VL2_VISION_PHASE0.md), Gundam crops for deepseekocr are not built (../docs/DEEPSEEK_OCR_VISION_PHASE0.md section 7). Effort S to M each.

## 7. OCR and document parsers

Summary: 11 dirs audited (granite_docling is another group). supported 0, tower-only 1 (minicpm_ocr), family-only 4 (got, dots_ocr, indic_ocr, oxr), missing 4 (glm_ocr, paddleocr_vl, florence2, falcon_ocr), out-of-scope 2 (minicpm_ocr_jev, falcon_perception). No mlx-vlm model_type in this group is registered in `arch_registry.rs`, and no doc or test names any of them (searches for florence, paddleocr, dots_ocr, glm_ocr, falcon, got_ocr, oxr returned nothing relevant).

| mlx-vlm dir | HF model_type | mlx-vlm vision file | TurboSpark text family | TurboSpark vision tower | Status | Files to read |
|---|---|---|---|---|---|---|
| got | got (Qwen2 0.5B text via `qwen2.language`) | got/vision.py | Qwen2Dense exists (`qwen2` row) but `got` is not registered | none (SAM-style ViTDet plus conv neck, no kind) | family-only | ../crates/repack/src/arch_registry.rs |
| dots_ocr | dots_ocr (text_config reuses `llava_bunny.language`, Qwen2-style; vit is `dots_vit`) | dots_ocr/vision.py | Qwen2Dense likely fits, `dots_ocr` not registered (unverified fit) | none (dots_vit NaViT) | family-only | ../crates/repack/src/arch_registry.rs |
| glm_ocr | glm_ocr (text `glm_ocr_text`, vision `glm_ocr_vision`) | glm_ocr/vision.py | none (GLM-4 style trunk with M-RoPE, no ModelFamily) | none | missing | ../crates/runtime/src/families/ |
| paddleocr_vl | paddleocr_vl | paddleocr_vl/vision.py | none (ERNIE-4.5 style trunk, no ModelFamily) | none (NaViT SigLIP variant; `SigLip` kind is the DeepSeek-VL2 so400m fixed-grid tower, not this) | missing | ../crates/runtime/src/vision/siglip.rs |
| florence2 | florence2 (vision `davit`) | florence2/vision.py | none (BART-style encoder-decoder, no flow) | none (DaViT) | missing | ../crates/runtime/src/families/ |
| indic_ocr | indic_ocr (Qwen3.5-0.8B block OCR plus PP-DocLayoutV3 detector) | none (no vision.py; pipeline.py, blocks.py) | QwenGdnDense (`qwen3_5` row) covers the trunk, `indic_ocr` not registered | QwenSiglip would cover the Qwen3.5 tower (unverified for this checkpoint); layout detector has nothing | family-only | ../crates/repack/src/qwen36_config.rs |
| minicpm_ocr | minicpm_ocr (subclass of minicpmv4_6 plus Qwen3.5 backbone) | none of its own; inherits minicpmv4_6/vision.py | QwenGdnDense exists | MiniCpmV exists (`docs/MINICPMV_VISION_PHASE0.md`), but intake accepts only `minicpmv4_6` and `minicpmv4_7` | tower-only | ../docs/MINICPMV_VISION_PHASE0.md |
| minicpm_ocr_jev | minicpm_ocr_jev | none | n/a | n/a | out-of-scope (2B Qwen3.5 classifier with a `score` head, scores OCR/render match, not a generator) | n/a |
| falcon_ocr | falcon_ocr | none (early-fusion, patches fed to the shared trunk; language.py, layout.py) | none (custom Falcon early-fusion trunk with hybrid mask) | none | missing | ../crates/runtime/src/families/ |
| falcon_perception | falcon_perception | none (anyup.py upsampler) | n/a | n/a | out-of-scope (detection and segmentation output, boxes and masks) | n/a |
| oxr | oxr (Qwen3-0.6B text, vision `siglip2_navit`) | oxr/vision.py | Qwen3Dense exists (`qwen3` row), `oxr` not registered | none (SigLIP2 NaViT, variable grid; no kind) | family-only | ../crates/repack/src/arch_registry.rs |

### Gaps

- got: vision tower missing (SAM ViTDet 1024 px encoder plus conv neck, linear projector to Qwen2). Port from https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/got/vision.py (config: https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/got/got.py). Extend ../crates/model-io/src/arch_config/sub_configs.rs (new VisionTowerKind), ../crates/vision-io/src/lib.rs, ../crates/runtime/src/vision/mod.rs, ../crates/runtime/src/real_forward_vision_api.rs. Effort L: windowed SAM attention with rel-pos and a new kind across all layers.
- dots_ocr: vision tower missing (dots_vit, 1.2B NaViT with variable grid) and `dots_ocr` model_type not registered. https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/dots_ocr/vision.py. Extend ../crates/repack/src/arch_registry.rs, ../crates/runtime/src/vision/qwen25.rs (closest windowless NaViT pattern), ../crates/vision-io/src/smart_resize.rs. Effort L: new tower plus registration.
- glm_ocr: both missing. https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/glm_ocr/vision.py (text: .../glm_ocr/language.py). Needs a ModelFamily and flow (../crates/runtime/src/families/, ../crates/model-io/src/arch_config/) plus a tower. Effort L: new family (M-RoPE GLM trunk) and new tower.
- paddleocr_vl: both missing. https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/paddleocr_vl/vision.py (text: .../paddleocr_vl/language.py, ERNIE-4.5 style). Extend ../crates/runtime/src/families/ and ../crates/runtime/src/vision/siglip.rs. Effort L: new trunk plus native-resolution NaViT tower.
- florence2: both missing. https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/florence2/vision.py (text: .../florence2/language.py, BART encoder-decoder). The decode flow is single-stack; encoder-decoder cross-attention has no home. Effort L: new flow shape plus DaViT.
- indic_ocr: wiring missing. Block OCR is a Qwen3.5 VL fine-tune (trunk and tower likely covered), but the layout detector and reconstruction stages are pipeline code, not a model. https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/indic_ocr/blocks.py and .../pipeline.py. Extend ../crates/repack/src/arch_registry.rs and ../crates/repack/src/qwen36_config.rs. Effort M for the block OCR checkpoint alone (register, gate on real weights); L if the layout stage is wanted.
- minicpm_ocr: registration and gate missing. https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/minicpm_ocr/minicpm_ocr.py (differences: its `vit_merger.act` override to `tanh` is a no-op in MLX, where `precise` is the same function; fp32 norm weights, `4x`/`16x` downsample modes, adaptive upsize in processing_minicpm_ocr.py). Extend ../crates/repack/src/qwen36_config.rs (the `minicpmv4_6`/`minicpmv4_7` gate), ../crates/runtime/src/vision/minicpm.rs, ../crates/vision-io/src/minicpm.rs. Effort M: tower exists, needs a 4x mode check, norm precision and a real-model gate.
- minicpm_ocr_jev: out-of-scope, no gap.
- falcon_ocr: both missing. https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/falcon_ocr/falcon_ocr.py (language.py, layout.py). Early-fusion single trunk, no separate tower. Extend ../crates/runtime/src/families/ and ../crates/model-io/src/arch_config/. Effort L: new family with a bidirectional-image hybrid mask.
- falcon_perception: out-of-scope, no gap.
- oxr: vision tower missing and `oxr` not registered. https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/oxr/vision.py (SigLIP2 NaViT, Qwen3-0.6B trunk via qwen3/language.py). Extend ../crates/repack/src/arch_registry.rs, ../crates/runtime/src/vision/siglip.rs (only the fixed 27x27 DeepSeek-VL2 form today), ../crates/vision-io/src/smart_resize.rs. Effort M: trunk exists, tower is a SigLIP variant with variable grid positions.

Caveat: rows reflect searches of docs, repack, model-io, catalog, runtime and vision-io for each name; text-trunk fit for dots_ocr and oxr was judged from imports, not config comparison (unverified).

## 8. MiniCPM, MoonDream, Molmo, Jina, FastVLM

Summary: 2 supported (minicpmv4_7, minicpmv4_6), 1 family-only (minicpmo), 7 missing (moondream2, moondream3, molmo, molmo2, molmo_point, jina_vlm, fastvlm), 0 out-of-scope.

| mlx-vlm dir | HF model_type | mlx-vlm vision file | TurboSpark text family | TurboSpark vision tower | Status | Files to read |
|---|---|---|---|---|---|---|
| minicpmv4_7 | minicpmv4_7 (text_config qwen3_5_moe_text) | [vision.py (via minicpmv4_6)](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/minicpmv4_6/vision.py) | QwenGdnMoe (text_config fallthrough) | VisionTowerKind::MiniCpmV | supported (real-gated 2026-11-01, thumbnail view only, no `<slice>` crops) | [docs/MINICPMV_VISION_PHASE0.md](../docs/MINICPMV_VISION_PHASE0.md), [docs/MODEL_FAMILY.md](../docs/MODEL_FAMILY.md) line 392, [crates/bench/tests/minicpm_v47_quality_gate.rs](../crates/bench/tests/minicpm_v47_quality_gate.rs) |
| minicpmv4_6 | minicpmv4_6 (text_config qwen3_5_text) | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/minicpmv4_6/vision.py) | QwenGdnDense (text_config fallthrough) | `VisionTowerKind::MiniCpmV46`, sequential positions | supported (2026-11-27, one smoke on the 4-bit build; MLX-quantized combined install only, thumbnail view, no tower parity) | [docs/MINICPMV_VISION_PHASE0.md](../docs/MINICPMV_VISION_PHASE0.md) |
| minicpmo | minicpmv (root), text qwen3 | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/minicpmo/vision.py) | Qwen3Dense ("qwen3") probably | none for this variant (SigLIP-NaViT style, not the 4.6/4.7 merger tower; unverified) | family-only (audio and TTS halves also absent) | [crates/repack/src/arch_registry.rs](../crates/repack/src/arch_registry.rs) |
| moondream2 | moondream2 | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/moondream2/vision.py) | none (Phi-style custom text) | none | missing | [crates/repack/src/arch_registry.rs](../crates/repack/src/arch_registry.rs) |
| moondream3 | moondream3 | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/moondream3/vision.py) | none (custom MoE text) | none | missing | [crates/repack/src/arch_registry.rs](../crates/repack/src/arch_registry.rs) |
| molmo | molmo | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/molmo/vision.py) | none (own Molmo block, not llama/qwen2) | none | missing | [crates/repack/src/arch_registry.rs](../crates/repack/src/arch_registry.rs) |
| molmo2 | molmo2 | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/molmo2/vision.py) | none (Molmo2 decoder) | none | missing | [crates/repack/src/arch_registry.rs](../crates/repack/src/arch_registry.rs) |
| molmo_point | molmo_point (text molmo2_text) | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/molmo_point/vision.py) | none (Molmo2 decoder plus ExtendedLmHead) | none | missing | [crates/repack/src/arch_registry.rs](../crates/repack/src/arch_registry.rs) |
| jina_vlm | jina_vlm | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/jina_vlm/vision.py) | none (custom TextModel with ExtendedEmbedding) | none | missing | [crates/repack/src/arch_registry.rs](../crates/repack/src/arch_registry.rs) |
| fastvlm | llava_qwen2 | [vision.py](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/fastvlm/vision.py) | Qwen2Dense likely (language.py reuses qwen2 Qwen2Model), "llava_qwen2" key not in registry | none (FastViTHD conv/RepMixer hybrid, not SigLIP/CLIP) | missing (text family reachable only after a registry alias; unverified) | [crates/repack/src/arch_registry.rs](../crates/repack/src/arch_registry.rs), [crates/model-io/src/arch_config/sub_configs.rs](../crates/model-io/src/arch_config/sub_configs.rs) |

### Gaps

- minicpmv4_6: WIRED (2026-11-27), see [MINICPMV_VISION_PHASE0.md](MINICPMV_VISION_PHASE0.md). The 4-bit build passed text, image and no-image smokes. Owed: tower parity and a server run. `4x` mode and `<slice>` crops are not built (shared with 4.7).
- minicpmo: vision tower is a different config (siglip_vision_model, Qwen3 text via qwen3_vl mrope default); no tower or wrapper handling here, plus audio/TTS/vocoder out of scope. Port https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/minicpmo/vision.py. Extend ../crates/runtime/src/vision/siglip.rs or minicpm.rs, ../crates/model-io/src/arch_config/sub_configs.rs, ../crates/repack/src/arch_registry.rs (minicpmv root key). Effort M: SigLIP-style tower exists, Qwen3Dense text exists, needs resampler/wrapper reader; text uses mrope.
- moondream2: no text family, no tower. Port https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/moondream2/vision.py and language.py, image_crops.py. Extend ../crates/runtime/src/families/, ../crates/vision-io/src/, ../crates/runtime/src/vision/. Effort L: new text family (Phi-style, parallel attn) plus crop preprocessing and tower.
- moondream3: same, with MoE text. Port https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/moondream3/vision.py. Same TurboSpark files. Effort L: new MoE family, crops, tower.
- molmo: Port https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/molmo/vision.py (CLIP-like ViT with attention pooling connector). Extend ../crates/runtime/src/vision/clip.rs, ../crates/runtime/src/families/, ../crates/vision-io/src/clip.rs. Effort L: new text family (Molmo block), multi-crop preprocessing.
- molmo2: Port https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/molmo2/vision.py. Same files as molmo. Effort L: new decoder family plus tower plus pooling connector.
- molmo_point: Molmo2 base plus pointing head. Port https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/molmo_point/vision.py and point_utils.py. Same files; depends on molmo2. Effort L: molmo2 work plus ExtendedLmHead and point decoding API.
- jina_vlm: Port https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/jina_vlm/vision.py. Extend ../crates/runtime/src/vision/siglip.rs, ../crates/runtime/src/families/, ../crates/vision-io/src/. Effort L: custom text model with extended embedding plus tiling preprocessor.
- fastvlm: Port https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/fastvlm/vision.py (FastViTHD RepMixer/conv blocks). Extend ../crates/runtime/src/vision/, ../crates/vision-io/src/, ../crates/repack/src/arch_registry.rs (add llava_qwen2 to Qwen2Dense via text_config), ../crates/model-io/src/arch_config/sub_configs.rs. Effort L: conv-hybrid tower is wholly new; text side S.


Unverified: the llava_qwen2 registry fallthrough was read from source only, not tested.

## 9. Chinese and other lab VLMs (GLM, Kimi, Hunyuan, Step, MiMo, Youtu, Zaya, ERNIE, MiniMax, PLaMo, LLM-jp)

Summary: 16 dirs checked. 0 supported, 1 supported-unverified (mage_vl), 0 tower-only, 3 family-only (kimi_vl, kimi_k25, llmjpvl), 12 missing. None of these model_types is in `SUPPORTED_HF`, and only mage_vl has a `VisionTowerKind` variant (`MageVit`).

| mlx-vlm dir | HF model_type | mlx-vlm vision file | TurboSpark text family | TurboSpark vision tower | Status | Files to read |
|---|---|---|---|---|---|---|
| glm4v | glm4v (text glm4v_text) | glm4v/vision.py | none (`glm4` unregistered) | none | missing | [docs/MODEL_FAMILY.md](../docs/MODEL_FAMILY.md) |
| glm4v_moe | glm4v_moe | glm4v_moe/vision.py | none (`glm4_moe`/`glm4moe` witnessed, unregistered; [docs/GLM4MOE_PHASE0.md](../docs/GLM4MOE_PHASE0.md)) | none | missing | [crates/repack/src/arch_registry.rs](../crates/repack/src/arch_registry.rs) |
| glm5_next | glm5_next | glm5_next/vision.py | none (KDA + DSA hybrid) | none | missing | [docs/MODEL_FAMILY.md](../docs/MODEL_FAMILY.md) |
| kimi_vl | kimi_vl (text deepseek_v3, vision moonvit) | kimi_vl/vision.py | Deepseek2 (GGUF `deepseek2` only; HF `deepseek_v3` not in SUPPORTED_HF) | none | family-only (unverified for HF intake) | [crates/repack/src/arch_registry.rs](../crates/repack/src/arch_registry.rs), [docs/MODEL_FAMILY.md](../docs/MODEL_FAMILY.md) |
| kimi_k25 | kimi_k25 (text deepseek_v3, moonvit) | kimi_k25/vision.py | Deepseek2 (same caveat) | none | family-only (unverified) | same |
| kimi_k3 | kimi_k3 (text kimi_linear, moonvit3d) | kimi_k3/vision.py | none; refused by measurement (896 experts, unstreamable) | none | missing | [docs/MODEL_FAMILY.md](../docs/MODEL_FAMILY.md) |
| hunyuan_vl | hunyuan_vl | hunyuan_vl/vision.py | none (`hunyuan` unregistered) | none | missing | [docs/MODEL_FAMILY.md](../docs/MODEL_FAMILY.md) |
| step3p7 | step3p7 (text step3p5, perception_encoder) | step3p7/vision.py | none (`step3p5` noted only) | none | missing | [docs/MODEL_FAMILY.md](../docs/MODEL_FAMILY.md) |
| mimo_v2 | mimo_v2 (vision mimovl) | mimo_v2/vision.py | none (`mimo_v2_flash` noted only) | none | missing | [docs/MODEL_FAMILY.md](../docs/MODEL_FAMILY.md) |
| youtu_vl | youtu_vl (siglip2_vision_model) | youtu_vl/vision.py | none (`youtu_llm` unregistered) | none (no SigLIP2 NaFlex kind) | missing | [docs/MODEL_FAMILY.md](../docs/MODEL_FAMILY.md) |
| zaya1_vl | zaya1_vl | zaya1_vl/vision.py | none | none (Qwen25 tower exists, Zaya block differs, unverified reuse) | missing | [crates/runtime/src/vision/qwen25.rs](../crates/runtime/src/vision/qwen25.rs) |
| ernie4_5_moe_vl | ernie4_5_moe_vl (DFNRope vision) | ernie4_5_moe_vl/vision.py | none (`ernie4_5_moe` witnessed, unregistered) | none | missing | [docs/MODEL_FAMILY.md](../docs/MODEL_FAMILY.md) |
| minimax_m3_vl | minimax_m3_vl (text minimax_m3, clip_vision_model) | minimax_m3_vl/vision.py | MiniMaxM2 exists, `minimax_m3` not registered (relation unverified) | none (Clip kind is a different tower) | missing | [crates/model-io/src/arch_config/sub_configs.rs](../crates/model-io/src/arch_config/sub_configs.rs) |
| mage_vl | mage_vl (text qwen3) | mage_vl/vision.py | Qwen3Dense (`qwen3` from `text_config`) | `MageVit` (runtime/vision/qwen2.rs), still images only | supported-unverified (2026-11-27, synthetic only; video refused) | [docs/MAGE_VL_VISION_PHASE0.md](MAGE_VL_VISION_PHASE0.md) |
| plamo2vl | plamo2vl (siglip_vision_model) | plamo2vl/vision.py | none (`plamo2` Mamba hybrid unregistered) | SigLip kind is DeepSeek-VL2 specific, not wired | missing | [docs/MODEL_FAMILY.md](../docs/MODEL_FAMILY.md) |
| llmjpvl | llmjpvl (text llama, siglip_vision_model) | llmjpvl/vision.py | Llama (`llama` in SUPPORTED_HF) | none for this wrapper (SigLip kind has a DeepSeek-VL2 unfold projector, llmjpvl uses `mlp1`) | family-only | [crates/runtime/src/vision/siglip.rs](../crates/runtime/src/vision/siglip.rs) |

### Gaps

- glm4v: text and tower missing. Port https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/glm4v/vision.py ; extend ../crates/repack/src/arch_registry.rs, ../crates/model-io/src/arch_config/sub_configs.rs. Effort L: new text family plus new tower kind.
- glm4v_moe: both missing. https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/glm4v_moe/vision.py ; see ../docs/GLM4MOE_PHASE0.md. Effort L: GLM4 MoE is unregistered, 100B+ class.
- glm5_next: both missing. https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/glm5_next/vision.py ; Effort L: KDA plus DSA hybrid, huge.
- kimi_vl: tower missing (moonvit). https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/kimi_vl/vision.py ; extend ../crates/runtime/src/vision/ (new file), ../crates/vision-io/src/, ../crates/repack/src/arch_registry.rs (HF `deepseek_v3` entry). Effort M: small 16B MoE text rides Deepseek2, tower is a Qwen2-VL-style ViT with 2D rope.
- kimi_k25: same moonvit gap. https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/kimi_k25/vision.py ; Effort L: K2.5 is ~1T params, too big for 36 GB, little to verify locally.
- kimi_k3: both missing. https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/kimi_k3/vision.py ; ../docs/MODEL_FAMILY.md records it as unstreamable. Effort L, not feasible.
- hunyuan_vl: both missing. https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/hunyuan_vl/vision.py ; Effort L.
- step3p7: both missing. https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/step3p7/vision.py ; Effort L: step3p5 text and a perception encoder.
- mimo_v2: both missing. https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/mimo_v2/vision.py ; Effort L (also has an audio branch).
- youtu_vl: both missing. https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/youtu_vl/vision.py ; extend ../crates/runtime/src/vision/siglip.rs. Effort L: new text family plus SigLIP2 variable-resolution.
- zaya1_vl: both missing. https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/zaya1_vl/vision.py ; subclasses the Qwen2.5 vision model, so ../crates/runtime/src/vision/qwen25.rs is a starting point. Effort L: text family unknown here.
- ernie4_5_moe_vl: both missing. https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/ernie4_5_moe_vl/vision.py ; Effort L: ERNIE MoE text plus modality-routed experts.
- minimax_m3_vl: both missing (M3 text not registered; M2 exists but I did not verify M3 shares its flow). https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/minimax_m3_vl/vision.py ; extend ../crates/runtime/src/vision/clip.rs. Effort L.
- mage_vl: still images wired for MLX conversions (docs/MAGE_VL_VISION_PHASE0.md). Owed: a real-weights gate. Not ported: video and codec streams, the OptiQ, BF16 and `llava_onevision2` layouts.
- plamo2vl: both missing. https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/plamo2vl/vision.py ; Effort L: PLaMo 2 is a Mamba hybrid text model.
- llmjpvl: tower missing for this wrapper. https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/llmjpvl/vision.py ; extend ../crates/runtime/src/vision/siglip.rs and ../crates/model-io/src/arch_config/sub_configs.rs. Effort M: Llama text runs and a SigLIP encoder exists, but the existing kind is DeepSeek-VL2 specific (unfold projector) so a new projector (`mlp1`) path is needed.

Caveat: text-family status was checked against `arch_registry.rs` and [docs/MODEL_FAMILY.md](../docs/MODEL_FAMILY.md) only. Nothing was executed.

## 10. LFM, Inkling, Muse, D1

Summary: 4 supported (lfm3_vl, lfm3_omni, muse_glimmer, inkling2), 5 missing (lfm2_vl, d1_lfm3_vl, d1_omni, inkling, muse_spark), 0 out-of-scope. d1_* are non-generative decision models (readout heads, not chat), so they are a different kind of port even though their towers overlap; d1-Omni-2B shares the `lfm3_omni` model_type and is refused by its `D1Model` class or `d1_*` keys.

| mlx-vlm dir | HF model_type | mlx-vlm vision file | TurboSpark text family | TurboSpark vision tower | Status | Files to read |
|---|---|---|---|---|---|---|
| lfm2_vl | `lfm2_vl` (text `lfm2`) | `lfm2_vl/vision.py` (SigLIP2 NaFlex) | none: `lfm2` deliberately unregistered, "a different layer graph" (arch_registry.rs L366-370; MODEL_COVERAGE.md L86 lists it as a planned medium item) | none for this model. `crates/embeddings/src/siglip2` is an embedding model, not wired as a VLM tower; `VisionTowerKind::SigLip` is the DeepSeek-VL2 tower | missing | ../crates/repack/src/arch_registry.rs, ../docs/MODEL_COVERAGE.md |
| lfm3_vl | `lfm3_vl` (trunk `lfm3`) | `lfm3_vl/vision.py` | `ModelFamily::Lfm3` (arch_registry.rs L370) | `VisionTowerKind::Lfm3Vl`, runtime/vision/lfm3_vl.rs | supported (real-gated 2026-11-01 on `LiquidAI/LFM3-VL-2B-MLX-8bit`; [docs/LFM3_VL_VISION_PHASE0.md](../docs/LFM3_VL_VISION_PHASE0.md); test runtime/tests/lfm3_vl_vision_real.rs). Limits: combined install only, 4-bit build not intaken, no sidecar, no FFI/Swift run | ../docs/LFM3_VL_VISION_PHASE0.md, ../crates/runtime/src/vision/lfm3_vl.rs, ../crates/repack/src/gemma4_checkpoint/vision_lfm3.rs |
| lfm3_omni | `lfm3_omni` (trunk `lfm3`) | none own: reuses `lfm3_vl` tower and processor (`lfm3_omni/lfm3_omni.py` subclasses `Lfm3VlModel.encode_images` and `Lfm3VlProcessor`); audio is LFM3-Audio | `ModelFamily::Lfm3` (`lfm3_omni` and `lfm3_vl` rows in arch_registry.rs; the D1 decision-model twin is refused by its `D1Model` class or `d1_*` keys) | `VisionTowerKind::Lfm3Vl`, runtime/vision/lfm3_vl.rs | supported (real-gated 2026-11-27 on `LiquidAI/LFM3-Omni-2B-MLX-8bit@5901d16c`: probe, streamed install with the 14-block tower, greedy text smoke, page transcription with a no-image control; [docs/LFM3_OMNI_VISION_PHASE0.md](../docs/LFM3_OMNI_VISION_PHASE0.md)). Limits: image-only by construction (241 audio tensors dropped by name with a report), no tower-stage parity probe, no video, no quality gate or memory oracle, no server/FFI run | ../docs/LFM3_OMNI_VISION_PHASE0.md, ../crates/repack/src/gemma4_checkpoint/vision_lfm3.rs, ../crates/repack/src/arch_registry.rs |
| d1_lfm3_vl | `lfm3_vl` with `D1Model` arch | `lfm3_vl/vision.py` (shared) | `Lfm3` trunk, but d1 needs set attention and answer-slot readout, no decode | `Lfm3Vl` reusable | missing (decision head and prompt/parse stack absent) | ../docs/LFM3_VL_VISION_PHASE0.md |
| d1_omni | `d1_omni` | `d1_omni/vision.py` (SigLIP2) | none: bidirectional LFM2.5 encoder plus decision head | none (SigLIP2 not wired), FastConformer audio absent | missing | ../crates/repack/src/arch_registry.rs |
| inkling | `inkling` (v1) | `inkling/vision.py` | none: no `inkling` row; only `Inkling2Loop` exists | none | missing | ../crates/model-io/src/arch_config/family.rs |
| inkling2 | `inkling2` / `inkling2_loop` | `inkling2/vision.py` | `ModelFamily::Inkling2Loop` | `VisionTowerKind::Inkling2` | supported (sidecar only; real-gated 2026-11-27 on `inkling2-mini-mlx4`: tower against a numpy transcription, CLI and server image turns; [INKLING2_VISION_PHASE0.md](INKLING2_VISION_PHASE0.md)). Limits: mlx-vlm not executed, no `ts_generate` or app run, no multi-image run | INKLING2_VISION_PHASE0.md, ../crates/repack/tests/inkling2_vision_reader.rs |
| muse_spark | `muse_spark` | `muse_spark/vision.py` (RoPE2D, RMSNorm/LayerNorm blocks, adapter; differs from glimmer) | none: only `muse_glimmer*` rows registered | none for spark (`MuseGlimmer` tower is a different layout) | missing | ../crates/repack/src/arch_registry.rs, ../docs/MUSEGLIMMER_VISION_PHASE0.md |
| muse_glimmer | `muse_glimmer` / `muse_glimmer_text` | `muse_glimmer/vision.py` | `ModelFamily::MuseGlimmer` (arch_registry.rs L201, L276-277) | `VisionTowerKind::MuseGlimmer`, runtime/vision/muse.rs, vision-io/muse.rs | supported (real-gated 2026-11-01 on `mlx-community/Muse-Glimmer-30B-4bit`, tower cosine 0.99999879; [docs/MUSEGLIMMER_VISION_PHASE0.md](../docs/MUSEGLIMMER_VISION_PHASE0.md)). Limits: mlx-vlm not executed, no sidecar run, no logit parity | ../docs/MUSEGLIMMER_VISION_PHASE0.md, ../crates/runtime/src/vision/muse.rs |

### Gaps

- lfm2_vl: both halves missing. Text: LFM2 (conv plus attention, no SWA/sandwich norms) is a new layer graph; vision: SigLIP2 NaFlex with pixel-unshuffle projector. Port from https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/lfm2_vl/vision.py and https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/lfm2_vl/language.py. Extend ../crates/repack/src/arch_registry.rs, ../crates/model-io/src/arch_config/family.rs, ../crates/runtime/src/families/lfm3 (new lfm2 flow), ../crates/runtime/src/vision/lfm3_vl.rs (variant tower). Effort L: new family plus a new tower, though lfm3 shortconv kernels and the Lfm3Vl projector are reusable.
- lfm3_omni: DONE for the image subset (2026-11-27). The registry row, the omni config nesting and the tensor prefixes (`vision_tower.` / `multi_modal_projector.`, same as lfm3_vl; audio under `audio_tower.` / `audio_projector.` / `depth_decoder.` / `codebook_embedding.`) are wired and real-gated; see [docs/LFM3_OMNI_VISION_PHASE0.md](../docs/LFM3_OMNI_VISION_PHASE0.md). Still missing: the audio path (LFM3-Audio encoder, talker, codec) - Effort L if audio in/out is wanted, and the video span.
- d1_lfm3_vl: decision model, no token decoding: set attention plus readout at answer slots, prompt/calibration/pointer/box parsers. Port from https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/d1_lfm3_vl/d1.py. Extend ../crates/runtime/src/families/lfm3 and ../crates/runtime/src/vision/lfm3_vl.rs. Effort L: new inference mode and API surface; tower is reusable. Worth confirming it is wanted at all.
- d1_omni: bidirectional LFM2.5 trunk, SigLIP2 tower, FastConformer audio. Port from https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/d1_omni/vision.py and .../d1_omni.py. Extend ../crates/runtime/src/vision/siglip.rs, ../crates/audio. Effort L: three new components.
- inkling: v1 text family absent (only Inkling2Loop) and no tower. Port from https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/inkling/vision.py and .../language.py. Extend ../crates/model-io/src/arch_config/family.rs, ../crates/repack/src/arch_registry.rs. Effort L: new family, new tower, audio optional. Confirm that v1 checkpoints are public before starting.
- inkling2: DONE (2026-11-27), see [INKLING2_VISION_PHASE0.md](INKLING2_VISION_PHASE0.md). Still owed: the macOS app (it does not attach vision sidecars), a `ts_generate` run, the HF, LAMQ and GGUF installs, multi-image prompts, and a memory and quality row.
- muse_spark: both halves missing. Text: Muse Spark Open 816B-A42B MoE (no registry row). Vision: its own tower (RoPE2D, adapter, patch embedder); also audio and video processors. Port from https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/muse_spark/vision.py, .../language.py, .../muse_spark.py. Extend ../crates/model-io/src/arch_config/family.rs, ../crates/repack/src/arch_registry.rs, ../crates/runtime/src/vision/muse.rs (compare), ../crates/vision-io/src/muse.rs. Effort L, and likely not runnable on a 36 GB Mac at 816B scale, so a real-weight gate is doubtful.

Unverified: whether the `lfm3_vl` HF model_type string is resolved by `config_json_resolution` - RESOLVED 2026-11-27: the string now has its own arch_registry.rs row (read off `LiquidAI/LFM3-VL-2B-MLX-8bit@6990b981`), so resolution no longer depends on the `text_config.model_type` fallback. The Gemma4 checkpoint vision readers were not read beyond name and docs.

## 11. Vision-only, segmentation, depth, detection, and image generation (classification pass)

Summary: 31 dirs classified. 29 out-of-scope (no text output), 1 tower-only (sensenova_loop), 1 missing (cohere_compass). Only those two are image-to-text VLMs.

Note: HF model_type values below are from the mlx-vlm config.py defaults where present; "-" means not read.

| mlx-vlm dir | HF model_type | mlx-vlm vision file | TurboSpark text family | TurboSpark vision tower | Status | Files to read |
|---|---|---|---|---|---|---|
| sam3 | - | sam3/vision.py | none | none (no st hit) | out-of-scope: promptable segmentation, outputs masks | - |
| sam3_1 | - | sam3_1/vision.py | none | none | out-of-scope: segmentation/tracking, outputs masks | - |
| sam3d_body | - | sam3d_body/vision.py | none | none | out-of-scope: 3D body mesh estimation | - |
| sam3d_objects | - | sam3d_objects/vision.py | none | none | out-of-scope: image to 3D object generation | - |
| depth_pro_2 | - | depth_pro_2/encoder.py | none | none | out-of-scope: monocular depth/focal estimation | - |
| moge3 | - | moge3/vision.py | none | none | out-of-scope: geometry (point map) estimation | - |
| video_depth_anything | - | video_depth_anything/dpt.py | none | none | out-of-scope: video depth estimation | - |
| rfdetr | - | rfdetr/vision.py | none | none | out-of-scope: object detection/instance seg head | - |
| rt_detr_v2 | - | rt_detr_v2/vision.py | none | none | out-of-scope: object detector | - |
| yolo11 | - | yolo11/yolo11.py | none | none | out-of-scope: object detector | - |
| dinov2 | dinov2 | dinov2/dinov2.py | none | none | out-of-scope: vision-only feature encoder | - |
| siglip | siglip | siglip/siglip.py | none | SigLip kind exists as a VLM tower only, not the standalone dual encoder | out-of-scope: contrastive image-text encoder, no text generation | ../crates/runtime/src/vision/siglip.rs |
| sapiens2 | - | sapiens2/backbone.py | none | none | out-of-scope: human pose/seg/depth heads | - |
| colqwen2_5 | - | reuses qwen2_5_vl vision | Qwen2 dense (lm_head set to None) | Qwen25Vl tower exists | out-of-scope: multi-vector retrieval embeddings, no text output | ../crates/runtime/src/vision/qwen25.rs |
| cohere_compass | cohere_compass | cohere_compass/vision.py | none (no Cohere/Command-A family in arch_registry.rs) | none for this tower (Qwen3-VL-style packed encoder with deepstack; reuse of Qwen3Vl unverified) | missing | ../crates/repack/src/arch_registry.rs |
| pp_doclayout_v3 | - | pp_doclayout_v3/backbone.py | none | none | out-of-scope: document layout detector | - |
| gliner2_5 | - | n/a (DeBERTa text) | none | none | out-of-scope: text-only span NER, no image input | - |
| embedding_gemma2 | - | reuses Gemma 4 vision/audio towers | Gemma4 family exists (EmbeddingGemma 2 is a bidirectional encoder, not wired) | Gemma4 tower exists | out-of-scope: multimodal embeddings, no text output (text EmbeddingGemma lives in [crates/embeddings](../crates/embeddings)) | ../crates/embeddings/ |
| sensenova_loop | - (inherits qwen3_5; not read) | reuses qwen3_5/vision.py | QwenGdnDense runs plain qwen3_5; Loop schedule and mHC routers (loop_settings, mhc_attn/mhc_ffn) have no st hit | qwen3_5 tower exists (docs/VISION.md) | tower-only | ../docs/VISION.md, ../crates/runtime/src/families/ |
| idol | - | idol/birefnet.py | none | none | out-of-scope: image to 3D Gaussian generation | - |
| qwen_image | - | n/a | n/a (generator; TurboSpark has [crates/image](../crates/image) support for Qwen-Image 2.1 Turbo) | n/a | out-of-scope: text-to-image generator | ../crates/image/, ../docs/QWEN_IMAGE_21_TURBO.md |
| ideogram4 | - | n/a | none | none | out-of-scope: text-to-image generator | - |
| ming_image | - | n/a | none | none | out-of-scope: text-to-image generator | - |
| moss_image | - | n/a | none | none | out-of-scope: text-to-image generator | - |
| llada_image | - | n/a | none | none | out-of-scope: diffusion-LM image generator | - |
| minimax_h3_image | - | n/a | none | none | out-of-scope: image generator | - |
| minimax_h3 | - | n/a | none | none | out-of-scope: video+audio generator | - |
| flux2 | - | n/a | none | none (st hit only in catalog/image.rs mflux handling, not verified as FLUX.2) | out-of-scope: text-to-image generator | ../crates/catalog/src/image.rs |
| flux3 | - | n/a | none | none | out-of-scope: text-to-image generator | - |
| z_image | - | n/a | n/a (generator; TurboSpark has z_image config/scheduler in [crates/catalog](../crates/catalog) and [crates/image](../crates/image)) | n/a | out-of-scope: text-to-image generator | ../crates/image/, ../docs/ZIMAGE_TURBO.md |
| ernie_image | - | n/a | none | none | out-of-scope: text-to-image generator | - |

Code presence (st over crates, docs, swift/TurboSpark): no hits for sam3, rfdetr, rt_detr, yolo11, dinov2, sapiens, colqwen, doclayout, gliner, sensenova, ernie_image, ming_image, moss_image, llada, minimax_h3. Image-generation code exists only for Qwen-Image and Z-Image (crates/image, [crates/catalog/src/image.rs](../crates/catalog/src/image.rs)); not checked beyond that.

### Gaps

- sensenova_loop: text half missing (the Loop interval schedule from `loop_settings` plus mHC routers; Qwen GDN dense runs only the plain qwen3_5 layout). Vision tower exists via qwen3_5. Port from https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/sensenova_loop/language.py (config: https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/sensenova_loop/config.py). Extend ../crates/repack/src/arch_registry.rs, ../crates/model-io/src/arch_config/, ../crates/runtime/src/families/. Effort L: new layer-reuse schedule and multi-stream residual routing in the Qwen GDN forward, plus fp32 mHC weights, and a real-model gate.
- cohere_compass: text family missing (Command-A+-style decoder, no Cohere family) and the vision tower is unverified against the existing Qwen3Vl tower. Port from https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/cohere_compass/language.py and https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/cohere_compass/vision.py. Extend ../crates/repack/src/arch_registry.rs, ../crates/runtime/src/families/, ../crates/runtime/src/vision/. Effort L: new text family plus deepstack tower variant (patch 16, 27 layers, out 3584), with a 131072-vocab Cohere tokenizer.
