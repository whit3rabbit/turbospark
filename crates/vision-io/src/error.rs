//! Failure modes for vision preprocessing.
//!
//! Hand-rolled in the shape `model_io::ModelError` uses: struct-style variants
//! carrying the diagnostic values by name, and a manual `Display` that puts
//! those values in the message. The rule the variants are written to is that a
//! reader who sees only the message can tell which input caused it -- a bare
//! "invalid image" sends someone back to the file.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum VisionIoError {
    /// The image bytes could not be decoded. `detail` is the decoder's own
    /// message, which names the format it tried.
    Decode { detail: String },
    /// A decoded image exceeds the per-side cap.
    ImageTooLarge {
        width: u32,
        height: u32,
        max_side: u32,
    },
    /// Long side over short side exceeds the processor's limit. Checked BEFORE
    /// any rounding, which is where the reference checks it.
    AspectRatioTooExtreme { ratio: f64, limit: f64 },
    /// A dimension is zero, or a resized dimension is not a whole number of
    /// patch/merge factors.
    InvalidDimensions { detail: String },
    /// `preprocessor_config.json` is missing a field, or holds a value this
    /// crate cannot act on.
    BadConfig { field: String, why: String },
    /// The number of image placeholder tokens in an id sequence disagrees with
    /// the number of grids supplied beside it.
    PlaceholderMismatch { placeholders: usize, grids: usize },
}

impl fmt::Display for VisionIoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VisionIoError::Decode { detail } => {
                write!(f, "could not decode image bytes: {detail}")
            }
            VisionIoError::ImageTooLarge {
                width,
                height,
                max_side,
            } => write!(
                f,
                "image is {width}x{height}; each side must be at most {max_side} pixels"
            ),
            VisionIoError::AspectRatioTooExtreme { ratio, limit } => write!(
                f,
                "aspect ratio {ratio} exceeds the limit of {limit}; long side over short side"
            ),
            VisionIoError::InvalidDimensions { detail } => {
                write!(f, "invalid image dimensions: {detail}")
            }
            VisionIoError::BadConfig { field, why } => {
                write!(f, "preprocessor config field `{field}` is unusable: {why}")
            }
            VisionIoError::PlaceholderMismatch {
                placeholders,
                grids,
            } => write!(
                f,
                "prompt carries {placeholders} image placeholder token(s) but {grids} image grid(s) were supplied"
            ),
        }
    }
}

impl std::error::Error for VisionIoError {}
