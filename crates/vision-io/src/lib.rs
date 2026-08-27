//! Portable vision preprocessing for the qwen3_5 vision tower: image decode,
//! PIL-bicubic smart resize, normalize, patchify, and the three position
//! tables the tower and the trunk need.
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
//! Every per-patch sequence in this crate -- patch rows, pos-embed entries,
//! rope frequency rows -- is emitted in **merge-window order**: the patch grid
//! is cut into `merge_size x merge_size` windows and one whole window is
//! emitted before the next, `(t, wy, wx, ly, lx)`. That is the order the
//! spatial-merge step reads.
//!
//! Inside one patch row the feature order is **`(T, P_h, P_w, C)`**, and this
//! is where this crate DIFFERS from the reference. mlx-vlm's `pixel_values`
//! ship `(C, T, P_h, P_w)`. The tower's `patch_embed.proj.weight` is stored
//! `[1152, 2, 16, 16, 3]`, i.e. `[out, T, P, P, C]`, so emitting `(T,P,P,C)`
//! lets the repack copy that weight verbatim and the GEMM read it row-major
//! with no permutation anywhere (`docs/VISION_PHASE0.md` item 4). The parity
//! test bridges the two orders explicitly rather than loosening a tolerance:
//! a mismatch here is silent wrong numerics, never a crash.

#![forbid(unsafe_code)]

pub mod decode;
pub mod error;
pub mod mrope;
pub mod normalize;
pub mod params;
pub mod patchify;
pub mod pos_embed;
pub mod preprocess;
pub mod resize;
pub mod rope;
pub mod rounding;
pub mod smart_resize;

pub use decode::{decode_image_bytes, decode_image_file, Rgb8Image, MAX_IMAGE_DIM};
pub use error::VisionIoError;
pub use mrope::{mrope_position_triples, ImageSpan, MropePositions};
pub use params::{PreprocessParams, MAX_ASPECT_RATIO};
pub use patchify::{patch_rows, GridThw};
pub use pos_embed::{pos_embed_weights, PosEmbedTable};
pub use preprocess::{preprocess, PreprocessedImage};
pub use rope::vision_rope_freq_rows;
pub use rounding::round_half_to_even;
pub use smart_resize::resized_dims;
