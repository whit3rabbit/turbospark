# vision-io (src)

## Overview
Portable vision preprocessing for `turbospark-vision-io`: image decode with hard limits, PIL-compatible resize, normalize, patchify, position tables (learned pos-embed, vision RoPE, M-RoPE) and per-family pipelines (qwen3_5 default, Gemma 4, CLIP/LLaVA, MiniCPM, DeepSeek-OCR, DeepSeek-VL2, Pixtral, Muse, LLaVA-OneVision, Unlimited OCR2). Plain arithmetic on `Vec<f32>`, no Metal and no macOS dependency.

## Interface
- Decode: `decode_image_bytes`, `decode_image_file`, `image_dimensions`, `Rgb8Image`, `MAX_IMAGE_DIM` [decode.rs:decode_image_bytes]
- Core pipeline: `preprocess`, `PreprocessedImage` [preprocess.rs:preprocess]; `PreprocessParams`, `ResizeRule`, `DEFAULT_RESCALE_FACTOR`, `MAX_ASPECT_RATIO` [params.rs:PreprocessParams]; `resized_dims` [smart_resize.rs:resized_dims]; `rescale_and_normalize` [normalize.rs]; `patch_rows`, `GridThw` [patchify.rs:patch_rows]
- Resize: `resize_bicubic_pil`, `resize_lanczos_pil`, `resample_pil`, `resample_pil_with`, `PilFilter` [resize.rs:resize_bicubic_pil]; `round_half_to_even` [rounding.rs]
- Position tables: `pos_embed_weights`, `PosEmbedTable` [pos_embed.rs]; `vision_rope_freq_rows` [rope.rs]; `rope::VISION_ROPE_THETA` and `rope::vision_rope_freq_rows_default` (module path only), base is 1e4 by design, not the trunk's 1e7 [rope.rs:VISION_ROPE_THETA]; `mrope_position_triples`, `splice_and_walk`, `splice_image_placeholders`, `wrap_bare_image_placeholders`, `ImageSpan`, `VisionSpecialIds`, `MropePositions`, `SplicedPrompt` [mrope.rs]
- Families: `gemma4_preprocess`, `gemma4_resize_dims`, `gemma4_soft_tokens_for`, `gemma4_expand_image_placeholders`, `Gemma4Patches`, `Gemma4ImageIds` [gemma4.rs]; `clip_preprocess`, `clip_resize_dims`, `clip_prepend_bos`, `clip_expand_image_placeholders`, `flat_expand_image_placeholders`, `ClipLayout`, `ClipPreprocessParams`, `ClipPatches` [clip.rs]; `minicpm_preprocess`, `minicpm_resize_dims`, `minicpm_splice_and_positions`, `canvas_positions`, `MiniCpmParams`, `MiniCpmIds` [minicpm.rs]; `deepseek_ocr_preprocess`, `deepseek_ocr_row_layout`, `DeepseekOcrParams` [deepseek_ocr.rs]; `deepseek_vl2_preprocess`, `deepseek_vl2_row_layout`, `deepseek_vl2_image_tokens`, `select_best_resolution`, `pil_pad`, `DeepseekVl2Params`, `DeepseekVl2Tiles` [deepseek_vl2.rs]; `pixtral_preprocess`, `pixtral_resize_target`, `pixtral_rope_freq_rows`, `pixtral_expand_image_placeholders` [pixtral.rs]; `muse_preprocess`, `muse_resized_dims`, `muse_patch_rows` [muse.rs]; `ov_preprocess`, `ov_plan`, `ov_pack_features`, `ov_image_tokens`, `ov_select_best_resolution`, `ov_patch_output_size`, `ov_best_grid` (root alias of `onevision::best_grid`), `Anyres`, `OvPlan`, `ov_max_tiles`, `parse_pinpoints` [onevision.rs]; `uocr2_pages`, `uocr2_gundam_page`, `uocr2_base_page`, `uocr2_crop_grid`, `uocr2_expand_prompt`, `uocr2_grid`, `Uocr2Page`, `Uocr2ImageIds`, `UOCR2_BASE_SIZE`, `UOCR2_CROP_SIZE`, `UOCR2_MAX_CROPS` [unlimited_ocr2.rs] (module is private, only these re-exports are reachable)
- Re-exported per-family constants (fixed geometry and special ids; see each family module): `CLIP_PATCH`, `CLIP_PATCH_DIM` [clip.rs]; `OCR_BASE_SIZE`, `OCR_DOWNSAMPLE`, `OCR_GRID`, `OCR_IMAGE_TOKENS`, `OCR_PATCH` [deepseek_ocr.rs]; `DSVL2_PATCH`, `DSVL2_PATCH_DIM` [deepseek_vl2.rs]; `GEMMA4_DEFAULT_SOFT_TOKENS`, `GEMMA4_PATCH`, `GEMMA4_PATCH_DIM`, `GEMMA4_POOL`, `GEMMA4_SOFT_TOKEN_BUDGETS` [gemma4.rs]; `MINICPM_IMAGE_ID_START`, `MINICPM_IMAGE_ID_END`, `MINICPM_SLICE_START_ID`, `MINICPM_SLICE_END_ID`, `MINICPM_NEWLINE_ID`, `MINICPM_MARKER_TOKENS`, `MINICPM_MERGE`, `MINICPM_PATCH`, `MINICPM_SCALE_RESOLUTION` [minicpm.rs]; `OV_MAX_AREA_TILES`, `OV_PATCH`, `OV_PATCH_DIM`, `OV_SIDE`, `OV_TILE`, `OV_TILE_PATCHES` [onevision.rs]; `PIXTRAL_PATCH`, `PIXTRAL_PATCH_DIM` [pixtral.rs]. Row types `DeepseekOcrRow`, `DeepseekOcrView`, `DeepseekVl2Row` are re-exported alongside their preprocessors.
- Errors: `VisionIoError` (`Decode`, `ImageTooLarge` (declared but never constructed: oversize images fail the decoder limits and surface as `Decode`), `AspectRatioTooExtreme`, `InvalidDimensions`, `BadConfig`, `PlaceholderMismatch`) [error.rs:VisionIoError]; `uocr2_expand_prompt` returns `Result<_, String>` instead [unlimited_ocr2.rs:uocr2_expand_prompt]

## Invariants
- `#![forbid(unsafe_code)]`, pure-Rust codecs only (jpeg, png) so the crate cross-checks off macOS [lib.rs:forbid] [Cargo.toml:image]
- Decoded images are capped at 4096 per side and the decoder allocation is bounded [decode.rs:MAX_IMAGE_DIM]
- Aspect ratio is checked before any rounding [smart_resize.rs:resized_dims] (limit is [params.rs:MAX_ASPECT_RATIO])
- Normalization runs on the resized uint8 samples, after resize [preprocess.rs:preprocess]
- `preprocess` routes `ResizeRule::MuseTokens` to `muse_preprocess`, not the Qwen chain [preprocess.rs:preprocess]
- Config edges are capped at `MAX_IMAGE_DIM`, `min_pixels` above `MAX_CONFIG_PIXELS` is refused while `max_pixels` above it is silently clamped, and the pixel budget has no default [params.rs:MAX_CONFIG_PIXELS] [params.rs:read_pixel_budget]
- Qwen path only: patch sequences are in merge-window order and each patch row is `(T, P_h, P_w, C)`, unlike the mlx-vlm `(C, T, P_h, P_w)` reference [patchify.rs:patch_rows]. Muse emits raster order with `(T, C, P_h, P_w)` rows [muse.rs:muse_patch_rows]; Gemma 4 emits row-major `[16][16][RGB]` rows with no temporal axis [gemma4.rs:Gemma4Patches]
- Placeholder token counts must match the supplied grids [error.rs:VisionIoError::PlaceholderMismatch]
- Token ids use signed 32-bit `i32` [minicpm.rs:MINICPM_SLICE_START_ID]

## Side Effects
- `decode_image_file` reads a file from disk [decode.rs:decode_image_file]
- `onevision::parse_pinpoints` reads from an already parsed JSON value only [onevision.rs:parse_pinpoints]
- None otherwise (no network, no GPU)

## Verification
- Run: `cargo test -p turbospark-vision-io`
