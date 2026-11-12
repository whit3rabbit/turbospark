# Vision Towers

## Overview
Metal vision encoders that turn preprocessed image patches into FP16 embedding rows for the trunk. The qwen3_5 tower lives in mod.rs and is built from block.rs, stages.rs, scratch.rs, shape.rs and weights.rs. Sibling towers (qwen25, clip, siglip, onevision, gemma4, pixtral, lfm3_vl, minicpm, muse, deepseek_ocr, unlimited_ocr2) follow the same pattern: resident weights plus transformer blocks pread-streamed from packed_vision/ into pinned slots. budget.rs clamps pixel ceilings to the LoadGuard budget, inject.rs maps prompt placeholders to image rows and rope positions, and overflow.rs is an env-gated FP16 overflow diagnostic.

## Interface
- `VisionTower` [mod.rs:VisionTower]: qwen3_5 tower with `open`, `open_with_sidecar`, `run`, `run_with_stages`, `scratch_bytes`
- `VisionEmbedding`, `DeepstackRows`, `VisionStages`, `VisionTiming` [mod.rs:VisionEmbedding]: per-image FP16 rows, deepstack rows, parity-gate captures
- `VISION_SLOTS` [mod.rs:VISION_SLOTS]: streamer pipelining depth (2)
- `VisionShape::resolve` [shape.rs:VisionShape]: validates VisionConfig once; derives head_dim, merger_input, attention_scale
- `encode_block` [block.rs:encode_block]: one qwen3_5 block (post-norm attention plus row-tiled MLP) into a caller's pass
- `encode_patch_embed`, `pos_embed_rows`, `encode_merger`, `encode_deepstack_merger` [stages.rs:encode_merger]: stages around the block loop
- `VisionScratch`, `scratch_bytes`, `VISION_MLP_TILE_ROWS` [scratch.rs:scratch_bytes]: per-page scratch and the single byte-cost formula
- `VisionResident`, `BlockRoles`, `fp16_view`, `resolve_block_roles`, `BLOCK_ROLES` [weights.rs:fp16_view]: tower weight lookup and validation
- `resolve_max_pixels`, `PixelBudget`, `VisionBudgetTooSmall`, `RealForwardRunner::resolve_vision_pixel_budget` [budget.rs:resolve_max_pixels]: pixel ceiling clamp used by cli/ffi/server
- `PromptVision`, `RopePosition` [inject.rs:PromptVision]: image rows, placeholder spans, (t,h,w) rope triples; `new`, `flat`, `row_for`, `row_index_for`, `rope_position`, deepstack buffer attach
- `VisionOverflowCapture` [overflow.rs:VisionOverflowCapture]: `from_env`, `note_image_start`, `check_block`
- `Qwen25VisionTower`, `window_layout`, `permute_units` [qwen25.rs:window_layout]: Qwen2.5-VL tower with window attention
- `ClipVisionTower`, `clip_shape` [clip.rs:ClipVisionTower]: LLaVA-1.5 CLIP tower (23 blocks, quick_gelu)
- `SiglipVisionTower`, `unfold_2x2` [siglip.rs:unfold_2x2]: DeepSeek-VL2 SigLIP tower with tiles
- `OneVisionTower`, `onevision_shape` [onevision.rs:OneVisionTower]: LLaVA-OneVision SigLIP anyres tower
- `Gemma4VisionTower`, `Gemma4Dims` [gemma4.rs:Gemma4VisionTower]: Gemma 4 tower with host pooling
- `PixtralVisionTower`, `split_rows_by_span` [pixtral.rs:PixtralVisionTower]: Pixtral tower with 2-D rope
- `Lfm3VlVisionTower`, `Lfm3VlDims` [lfm3_vl.rs:Lfm3VlVisionTower]: LFM3-VL conv/attention hybrid tower
- `MiniCpmVisionTower`, `tower_order`, `position_buckets` [minicpm.rs:MiniCpmVisionTower]: MiniCPM-V 4.7 tower with vit_merger
- `MuseVisionTower`, `position_rows`, `pixel_shuffle` [muse.rs:MuseVisionTower]: Muse-Glimmer windowed tower
- `DeepseekOcrVisionTower`, `window_gather_index`, `window_scatter_index` [deepseek_ocr.rs:DeepseekOcrVisionTower]: DeepSeek-OCR SAM plus CLIP-L
- `Uocr2Tower`, `read_bounded`, `parse_tower_index` [unlimited_ocr2.rs:Uocr2Tower]: Unlimited-OCR-2 tower (pub module), loads uocr2_tower.bin/json

## Invariants
- Only weights.rs reads FP16 vision tensors, and `fp16_view` checks dtype tag and exact byte count so same-width BF16 is refused [weights.rs:fp16_view]
- Resident shapes come from VisionShape, never from the index entry [weights.rs:VisionResident]
- packed_vision must declare layer 0 with exactly shape.depth blocks; blocks bind by entry.expert, duplicates and out-of-range refused, in every tower [mod.rs:VisionTower::build]
- Block roles must be fp16, exact size, 4-byte aligned and inside the stride [weights.rs:resolve_block_roles]
- head_dim must satisfy head_dim%4==0 and <= gpu::MAX_ATTENTION_HEAD_DIM, else Unsupported (silent truncation otherwise) [shape.rs:VisionShape]
- Block n uses slot n % VISION_SLOTS with a waited command buffer per block so the next pread into a slot is safe; slot_buffers are declared before the streamer so they drop first [mod.rs:VisionTower::run_inner]
- Rope applies to q and k only, never v; qkv is split into three matmuls at byte offsets [block.rs:encode_block]
- Blocks use tanh GELU, mergers and projectors use erf GELU [stages.rs:encode_merger]
- Qwen2.5-VL attends per window segment except blocks with block%8==7; full attention everywhere is a fluent-wrong model [qwen25.rs:encode_block]
- Scratch aliases attn/proj/m1 onto normed/q/k, valid only under serial single-command-buffer order; seq must be a whole number of merge windows [scratch.rs:VisionScratch]
- Pixel budget: LoadGuard::Off never clamps, bisection is exact, and it refuses rather than clamping below min_pixels [budget.rs:resolve_max_pixels]
- PromptVision::new refuses mismatched triples, embedding/span counts, hidden sizes, unsorted or overlapping spans [inject.rs:PromptVision::new]
- row_for and deepstack adds share one row-index function [inject.rs:row_index_for]
- Narrowing to FP16 requires finite values with |v|<=65504, else Unsupported [gemma4.rs:to_f16]
- SigLIP final LayerNorm eps is 1e-5 while block eps is 1e-6; tiles must have exactly pos_rows patches [siglip.rs:FINAL_LAYER_NORM_EPS]
- Uocr2 index capped at 1 MiB and bin at 2 GiB, offsets must be even, all required tensors present with exact counts [unlimited_ocr2.rs:parse_tower_index]
- LFM3-VL refuses pages over 65,536 patches and video grids (t != 1) [lfm3_vl.rs:forward]

## Side Effects
- Reads env TURBOSPARK_VISION_RESIDENCY (mapped vs pread residency) [mod.rs:vision_mapped_residency_requested]
- Reads env TURBOSPARK_VISION_OVERFLOW and writes a JSON report to that path on Drop [overflow.rs:VisionOverflowCapture::from_env]
- Reads packed_vision/layout.json and pread-streams block blobs through PreadExpertStreamer with pinned slots [mod.rs:VisionTower::build]
- Sidecar open memory-maps model_weights.bin as ResidentGpuWeights, held for the tower's lifetime [mod.rs:VisionTower::open_with_sidecar]
- Reads uocr2_tower.json and uocr2_tower.bin, uploads the bin to a Metal buffer and caches per-grid tables in a HashMap [unlimited_ocr2.rs:Uocr2Tower]
- Allocates per-image Metal scratch inside gpu::autorelease_pool and commits waited passes [mod.rs:VisionTower::run]
- attach_deepstack_buffers allocates GPU buffers, freed on Drop [inject.rs:attach_deepstack_buffers]
- Synchronous GPU readback and eprintln warnings when overflow capture is enabled [overflow.rs:VisionOverflowCapture::check_block]

## Verification
- Run: `cargo test -p turbospark-runtime vision`
