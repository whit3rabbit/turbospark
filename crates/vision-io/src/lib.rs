//! Portable vision preprocessing for the qwen3_5 vision tower: image decode,
//! PIL-bicubic smart resize, normalize, patchify, and the three position
//! tables the tower and the trunk need. The Gemma 4 tower's pipeline is
//! separate and lives in [`gemma4`].
//!
//! Nothing here touches Metal, macOS, or a model install. It is plain
//! arithmetic on `Vec<f32>`, so it builds and tests for a non-macOS target --
//! which is the point, and which the cross-target `cargo check` in AGENTS.md
//! asserts.
//!
//! # What the pipeline produces
//!
//! [`preprocess`] takes decoded RGB8 pixels and returns [`PreprocessedImage`]:
//! the patch-row matrix the tower's patch-embedding GEMM consumes, plus the
//! `(t, h, w)` grid the position tables and the trunk's token expansion are
//! computed from. [`pos_embed_weights`], [`vision_rope_freq_rows`] and
//! [`mrope_position_triples`] are the three tables, and all of them are index
//! and weight arithmetic only -- no gather happens here, because the values
//! being gathered live in GPU buffers this crate cannot see.
//!
//! # Two orders that must not be confused
//!
//! On the Qwen path, every per-patch sequence -- patch rows, pos-embed
//! entries, rope frequency rows -- is emitted in **merge-window order**: the
//! patch grid
//! is cut into `merge_size x merge_size` windows and one whole window is
//! emitted before the next, `(t, wy, wx, ly, lx)`. That is the order the
//! spatial-merge step reads. Muse emits raster order with `(T, C, P_h, P_w)`
//! rows (`muse.rs`) and Gemma 4 emits row-major `[16][16][RGB]` rows with no
//! temporal axis (`gemma4.rs`); neither uses merge-window order.
//!
//! Inside one Qwen patch row the feature order is **`(T, P_h, P_w, C)`**, and this
//! is where this crate DIFFERS from the reference. mlx-vlm's `pixel_values`
//! ship `(C, T, P_h, P_w)`. The tower's `patch_embed.proj.weight` is stored
//! `[1152, 2, 16, 16, 3]`, i.e. `[out, T, P, P, C]`, so emitting `(T,P,P,C)`
//! lets the repack copy that weight verbatim and the GEMM read it row-major
//! with no permutation anywhere (`docs/VISION_PHASE0.md` item 4). The parity
//! test bridges the two orders explicitly rather than loosening a tolerance:
//! a mismatch here is silent wrong numerics, never a crash.

#![forbid(unsafe_code)]

pub mod clip;
pub mod decode;
pub mod deepseek_ocr;
pub mod deepseek_vl2;
pub mod error;
pub mod gemma4;
pub mod minicpm;
pub mod mrope;
pub mod muse;
pub mod normalize;
pub mod onevision;
pub mod params;
pub mod patchify;
pub mod pixtral;
pub mod pos_embed;
pub mod preprocess;
pub mod resize;
pub mod rope;
pub mod rounding;
pub mod smart_resize;
mod unlimited_ocr2;

pub use clip::{
    clip_expand_image_placeholders, clip_prepend_bos, clip_preprocess, clip_resize_dims,
    flat_expand_image_placeholders, ClipLayout, ClipPatches, ClipPreprocessParams, CLIP_PATCH,
    CLIP_PATCH_DIM,
};
pub use decode::{
    decode_image_bytes, decode_image_file, image_dimensions, Rgb8Image, MAX_IMAGE_DIM,
};
pub use deepseek_ocr::{
    deepseek_ocr_preprocess, deepseek_ocr_row_layout, DeepseekOcrParams, DeepseekOcrRow,
    DeepseekOcrView, OCR_BASE_SIZE, OCR_DOWNSAMPLE, OCR_GRID, OCR_IMAGE_TOKENS, OCR_PATCH,
};
pub use deepseek_vl2::{
    deepseek_vl2_image_tokens, deepseek_vl2_preprocess, deepseek_vl2_row_layout, pil_pad,
    select_best_resolution, DeepseekVl2Params, DeepseekVl2Row, DeepseekVl2Tiles, DSVL2_PATCH,
    DSVL2_PATCH_DIM,
};
pub use error::VisionIoError;
pub use gemma4::{
    gemma4_expand_image_placeholders, gemma4_preprocess, gemma4_resize_dims,
    gemma4_soft_tokens_for, Gemma4ImageIds, Gemma4Patches, GEMMA4_DEFAULT_SOFT_TOKENS,
    GEMMA4_PATCH, GEMMA4_PATCH_DIM, GEMMA4_POOL, GEMMA4_SOFT_TOKEN_BUDGETS,
};
pub use minicpm::{
    canvas_positions, minicpm_preprocess, minicpm_resize_dims, minicpm_splice_and_positions,
    MiniCpmIds, MiniCpmParams, MINICPM_IMAGE_ID_END, MINICPM_IMAGE_ID_START, MINICPM_MARKER_TOKENS,
    MINICPM_MERGE, MINICPM_NEWLINE_ID, MINICPM_PATCH, MINICPM_SCALE_RESOLUTION,
    MINICPM_SLICE_END_ID, MINICPM_SLICE_START_ID,
};
pub use mrope::{
    mrope_position_triples, splice_and_walk, splice_image_placeholders,
    wrap_bare_image_placeholders, ImageSpan, MropePositions, SplicedPrompt, VisionSpecialIds,
};
pub use onevision::{
    best_grid as ov_best_grid, ov_image_tokens, ov_max_tiles, ov_pack_features, ov_plan,
    ov_preprocess, Anyres, OvPlan, OV_MAX_AREA_TILES, OV_PATCH, OV_PATCH_DIM, OV_SIDE, OV_TILE,
    OV_TILE_PATCHES,
};
pub use params::{PreprocessParams, ResizeRule, DEFAULT_RESCALE_FACTOR, MAX_ASPECT_RATIO};
pub use patchify::{patch_rows, GridThw};
pub use pixtral::{
    pixtral_expand_image_placeholders, pixtral_preprocess, pixtral_resize_target,
    pixtral_rope_freq_rows, PIXTRAL_PATCH, PIXTRAL_PATCH_DIM,
};
pub use pos_embed::{pos_embed_weights, PosEmbedTable};
pub use preprocess::{preprocess, PreprocessedImage};
pub use rope::vision_rope_freq_rows;
pub use rounding::round_half_to_even;
pub use smart_resize::resized_dims;
pub use unlimited_ocr2::{
    uocr2_base_page, uocr2_crop_grid, uocr2_expand_prompt, uocr2_grid, uocr2_gundam_page,
    uocr2_pages, Uocr2ImageIds, Uocr2Page, UOCR2_BASE_SIZE, UOCR2_CROP_SIZE, UOCR2_MAX_CROPS,
};
