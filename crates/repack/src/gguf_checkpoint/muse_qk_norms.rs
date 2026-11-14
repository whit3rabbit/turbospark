//! Muse Glimmer's synthesized q/k norm tensors, checked by VALUE.
//!
//! The reference model's per-head q and k norms are scale-free, and Q takes
//! `qk_scale_factor` after its norm. llama.cpp's converter materializes that
//! as two tensors per layer: `attn_q_norm` filled with the factor and
//! `attn_k_norm` filled with ones (`conversion/muse_glimmer.py`). This port's
//! flow keeps the factor as a constant and has no tensor for either, so they
//! are dropped, but only once every element proves to be exactly what the
//! flow assumes. A different factor would make Q silently mis-scaled, which
//! decodes fluently, so it refuses instead.

use model_io::{ArchConfig, ModelFamily};

use super::plan::Plan;
use super::types::{read_tensor, GgufRepackError};
use crate::gguf_header::GgufHeader;
use crate::ranged_download::RangeSource;

/// `text_config.qk_scale_factor`, as the runtime's
/// `families/museglimmer/state.rs::QK_SCALE_FACTOR` holds it. Keep the two in
/// lockstep (neither crate can see the other).
pub const QK_SCALE_FACTOR: f32 = 3.87;

const GGML_TYPE_F32: u32 = 0;

/// Verifies every planned q/k norm tensor of a Muse install.
pub fn verify_plan(
    header: &GgufHeader,
    source: &dyn RangeSource,
    arch: &ArchConfig,
    plan: &Plan<'_>,
) -> Result<(), GgufRepackError> {
    if arch.family != ModelFamily::MuseGlimmer {
        return Ok(());
    }
    // Every layer ships both; a file missing them was not made by the
    // converter this check was read against.
    let want = 2 * arch.num_layers as usize;
    if plan.muse_qk_norms.len() != want {
        return Err(GgufRepackError::MissingTensor {
            name: format!(
                "attn_q_norm / attn_k_norm: {} found, {want} expected (one pair per layer)",
                plan.muse_qk_norms.len()
            ),
        });
    }
    for &(name, is_q) in &plan.muse_qk_norms {
        let info = header
            .tensors
            .get(name)
            .ok_or_else(|| GgufRepackError::MissingTensor {
                name: name.to_string(),
            })?;
        let mismatch = |detail: String| GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail,
        };
        if info.ggml_type != GGML_TYPE_F32 {
            return Err(mismatch(format!(
                "ggml type {} where the synthesized norm is F32",
                info.ggml_type
            )));
        }
        let bytes = read_tensor(header, source, name)?;
        constant_vector(&bytes, arch.full_head_dim as usize, is_q).map_err(mismatch)?;
    }
    Ok(())
}

/// The pure check: `head_dim` little-endian f32 values, all equal to
/// [`QK_SCALE_FACTOR`] for Q or 1.0 for K.
pub fn constant_vector(bytes: &[u8], head_dim: usize, is_q: bool) -> Result<(), String> {
    if bytes.len() != head_dim * 4 {
        return Err(format!(
            "{} bytes where {head_dim} f32 values (head_dim) are expected",
            bytes.len()
        ));
    }
    let want = if is_q { QK_SCALE_FACTOR } else { 1.0 };
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if v != want {
            return Err(format!(
                "element {i} is {v}; the {} norm must be the constant {want} the flow \
                 assumes",
                if is_q { "q" } else { "k" }
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

    #[test]
    fn the_published_constants_are_accepted() {
        assert_eq!(
            constant_vector(&encode(&[QK_SCALE_FACTOR; 128]), 128, true),
            Ok(())
        );
        assert_eq!(constant_vector(&encode(&[1.0; 128]), 128, false), Ok(()));
    }

    #[test]
    fn a_different_scale_or_a_wrong_length_is_refused() {
        assert!(constant_vector(&encode(&[4.0; 128]), 128, true).is_err());
        let mut v = vec![QK_SCALE_FACTOR; 128];
        v[77] = 3.5;
        let e = constant_vector(&encode(&v), 128, true).unwrap_err();
        assert!(e.contains("element 77"), "{e}");
        // A k norm that is not the identity.
        assert!(constant_vector(&encode(&[QK_SCALE_FACTOR; 128]), 128, false).is_err());
        assert!(constant_vector(&encode(&[1.0; 64]), 128, false).is_err());
    }
}
