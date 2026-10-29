//! `lfm3`'s `rope_freqs.weight`, checked by VALUE.
//!
//! The tensor name is a refusal for every other family
//! (`GgufNameError::UnsupportedRopeScaling`): in Llama 3.1 it is a LEARNED
//! per-pair frequency scaling that this port's scalar-theta rope kernels
//! cannot express, and dropping it yields a model wrong only at long
//! context. In the LFM3 conversion the same name carries something this port
//! CAN express: llama.cpp's encoding of HF's zero-tail partial rotation,
//! `1.0` over the rotated pairs and `1e30` over the rest (a divisor that
//! drives those pairs' angle to zero, so they pass through unrotated).
//! `partial_rotary_factor` already says that, so the tensor is dropped --
//! but only after its bytes prove it is that pattern and nothing else. A
//! re-conversion that put real scaling factors under the same name refuses
//! here instead of installing a wrong model.

use model_io::{ArchConfig, ModelFamily};

use super::plan::Plan;
use super::types::{read_tensor, GgufRepackError};
use crate::gguf_header::GgufHeader;
use crate::ranged_download::RangeSource;

/// Anything at or above this is "no rotation" in llama.cpp's encoding. The
/// published file writes 1e30; the threshold leaves room for a converter
/// that writes a different huge sentinel without admitting a real factor.
const UNROTATED_FACTOR_FLOOR: f32 = 1e29;

const GGML_TYPE_F32: u32 = 0;

/// Verifies the plan's `rope_freqs.weight` for an `lfm3` install, and
/// requires it: the partial factor this port applies is the one the vector
/// encodes, and a conversion without it would mean the converter rotated
/// the whole head, which the flow would then get wrong.
pub fn verify_plan(
    header: &GgufHeader,
    source: &dyn RangeSource,
    arch: &ArchConfig,
    plan: &Plan<'_>,
) -> Result<(), GgufRepackError> {
    if arch.family != ModelFamily::Lfm3 {
        return Ok(());
    }
    let name = plan
        .rope_freqs
        .ok_or_else(|| GgufRepackError::MissingTensor {
            name: "rope_freqs.weight".to_string(),
        })?;
    let mismatch = |detail: String| GgufRepackError::ShapeMismatch {
        tensor: name.to_string(),
        detail,
    };
    let info = header
        .tensors
        .get(name)
        .ok_or_else(|| GgufRepackError::MissingTensor {
            name: name.to_string(),
        })?;
    if info.ggml_type != GGML_TYPE_F32 {
        return Err(mismatch(format!(
            "ggml type {} where the zero-tail encoding is F32",
            info.ggml_type
        )));
    }
    let bytes = read_tensor(header, source, name)?;
    zero_tail_pattern(&bytes, arch).map_err(mismatch)
}

/// The pure check: `pairs = full_head_dim / 2` little-endian f32 values, the
/// first `partial_rotary_factor * pairs` exactly 1.0 and the rest at or
/// above [`UNROTATED_FACTOR_FLOOR`].
pub fn zero_tail_pattern(bytes: &[u8], arch: &ArchConfig) -> Result<(), String> {
    let pairs = usize::try_from(arch.full_head_dim / 2).map_err(|_| "negative head dim")?;
    let rotated_f = arch.partial_rotary_factor * pairs as f64;
    if rotated_f.fract() != 0.0 || rotated_f <= 0.0 || rotated_f > pairs as f64 {
        return Err(format!(
            "partial_rotary_factor {} does not select a whole number of {pairs} pairs",
            arch.partial_rotary_factor
        ));
    }
    let rotated = rotated_f as usize;
    if bytes.len() != pairs * 4 {
        return Err(format!(
            "{} bytes where {pairs} f32 factors (head_dim {}) are expected",
            bytes.len(),
            arch.full_head_dim
        ));
    }
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let ok = if i < rotated {
            v == 1.0
        } else {
            v.is_finite() && v >= UNROTATED_FACTOR_FLOOR
        };
        if !ok {
            return Err(format!(
                "factor {i} is {v}; the zero-tail encoding is 1.0 over the first {rotated} \
                 pairs and >= {UNROTATED_FACTOR_FLOOR:e} over the rest. Any other vector is \
                 learned rope scaling, which this port's kernels cannot express"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// The exact vector read off `LFM3-1.7B-Q8_0.gguf`: 32 ones, 32 x 1e30.
    fn published() -> Vec<f32> {
        let mut v = vec![1.0f32; 32];
        v.extend(std::iter::repeat_n(1e30f32, 32));
        v
    }

    #[test]
    fn the_published_vector_is_accepted() {
        let arch = model_io::lfm3_1_7b();
        assert_eq!(zero_tail_pattern(&encode(&published()), &arch), Ok(()));
    }

    /// A single real scaling factor in the rotated half is learned scaling,
    /// not the encoding: refused, naming the pair.
    #[test]
    fn a_scaled_rotated_pair_is_refused() {
        let arch = model_io::lfm3_1_7b();
        let mut v = published();
        v[5] = 1.5;
        let err = zero_tail_pattern(&encode(&v), &arch).unwrap_err();
        assert!(err.contains("factor 5 is 1.5"), "{err}");
    }

    /// A tail entry that is merely large is still a rotation (Llama 3.1's
    /// low-frequency factors are 8.0); only the sentinel counts.
    #[test]
    fn a_finite_tail_factor_is_refused() {
        let arch = model_io::lfm3_1_7b();
        let mut v = published();
        v[40] = 8.0;
        assert!(zero_tail_pattern(&encode(&v), &arch).is_err());
        v[40] = f32::INFINITY;
        assert!(zero_tail_pattern(&encode(&v), &arch).is_err());
    }

    /// The boundary moves with the factor: a vector whose ones stop at a
    /// different pair than `partial_rotary_factor` says is refused, which is
    /// what keeps the dropped tensor and the carried factor in agreement.
    #[test]
    fn a_boundary_that_disagrees_with_the_factor_is_refused() {
        let arch = model_io::lfm3_1_7b();
        let mut v = vec![1.0f32; 16];
        v.extend(std::iter::repeat_n(1e30f32, 48));
        assert!(zero_tail_pattern(&encode(&v), &arch).is_err());
    }

    #[test]
    fn a_wrong_length_is_refused() {
        let arch = model_io::lfm3_1_7b();
        let v = &published()[..48];
        assert!(zero_tail_pattern(&encode(v), &arch).is_err());
    }
}
