//! Image bytes to RGB8 pixels.
//!
//! Kept separate from [`crate::preprocess`] on purpose, and the reason is a
//! testing constraint rather than tidiness: **JPEG decode is not bit-compatible
//! across decoders**. Pillow and the `image` crate disagree on individual
//! samples of the same JPEG (different IDCT and chroma upsampling), so a parity
//! fixture sourced from a JPEG would compare this crate's arithmetic against
//! the reference's arithmetic PLUS a decoder difference, and a failure could
//! not be attributed. Every oracle fixture therefore starts from synthetic
//! pixels generated identically on both sides, and decode is exercised by its
//! own tests against its own invariants.

use std::io::Cursor;

use crate::error::VisionIoError;

/// Cap on either side of a decoded image.
///
/// Not a format limit. It bounds the allocation a hostile or corrupt file can
/// ask for before any of the pixel work starts. This still admits the
/// checkpoint's 4,064-pixel extreme while bounding an RGB8 decode to 48 MiB.
pub const MAX_IMAGE_DIM: u32 = 4_096;

fn reader(bytes: &[u8]) -> Result<image::ImageReader<Cursor<&[u8]>>, VisionIoError> {
    let mut reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| VisionIoError::Decode {
            detail: e.to_string(),
        })?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_DIM);
    limits.max_image_height = Some(MAX_IMAGE_DIM);
    limits.max_alloc = Some(u64::from(MAX_IMAGE_DIM) * u64::from(MAX_IMAGE_DIM) * 4);
    reader.limits(limits);
    Ok(reader)
}

/// Read an image's dimensions without allocating its pixel buffer.
///
/// The same strict decoder limits used by [`decode_image_bytes`] are installed
/// first, so callers can budget a whole request before decoding any image.
pub fn image_dimensions(bytes: &[u8]) -> Result<(u32, u32), VisionIoError> {
    reader(bytes)?
        .into_dimensions()
        .map_err(|e| VisionIoError::Decode {
            detail: e.to_string(),
        })
}

/// Decoded pixels, 8 bits per sample, three interleaved channels, row-major.
///
/// Interleaved rather than planar because that is what every decoder hands
/// back, and the only consumer ([`crate::preprocess`]) reads it with an
/// explicit stride either way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rgb8Image {
    pub width: usize,
    pub height: usize,
    /// `height * width * 3` samples, `(y, x, c)` order.
    pub data: Vec<u8>,
}

impl Rgb8Image {
    /// Wrap an existing buffer, checking it against the stated dimensions.
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Result<Self, VisionIoError> {
        let expected = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| VisionIoError::InvalidDimensions {
                detail: format!("{width}x{height} overflows a pixel count"),
            })?;
        if data.len() != expected {
            return Err(VisionIoError::InvalidDimensions {
                detail: format!(
                    "{width}x{height} RGB8 needs {expected} samples, got {}",
                    data.len()
                ),
            });
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// One sample, by pixel coordinate and channel.
    pub fn sample(&self, y: usize, x: usize, c: usize) -> u8 {
        self.data[(y * self.width + x) * 3 + c]
    }
}

/// Decode an encoded image, converting to RGB8.
///
/// Grayscale is broadcast to three channels and an alpha channel is dropped,
/// both by `to_rgb8`, matching the reference processor's `convert("RGB")`.
pub fn decode_image_bytes(bytes: &[u8]) -> Result<Rgb8Image, VisionIoError> {
    // Probe first so a dimension bomb is rejected from its header before the
    // decoder can allocate its output buffer.
    let (width, height) = image_dimensions(bytes)?;
    let decoded = reader(bytes)?.decode().map_err(|e| VisionIoError::Decode {
        detail: e.to_string(),
    })?;
    let rgb = decoded.to_rgb8();
    Rgb8Image::new(width as usize, height as usize, rgb.into_raw())
}

/// Read and decode an image file.
pub fn decode_image_file(path: &std::path::Path) -> Result<Rgb8Image, VisionIoError> {
    let bytes = std::fs::read(path).map_err(|e| VisionIoError::Decode {
        detail: format!("{}: {e}", path.display()),
    })?;
    decode_image_bytes(&bytes)
}
