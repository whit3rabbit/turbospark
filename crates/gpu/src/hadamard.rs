//! Host side of the row-wise signed block-Fast-Walsh-Hadamard transform
//! (`shaders/hadamard.metal`): the activation-side half of the prism
//! Hadamard-folded weight contract (the Bonsai-2 line, `docs/BONSAI2.md`).
//!
//! The kernel is ONE shape for every folded entry because the transform is a
//! property of the ACTIVATION width, not of the weight: one sign vector per
//! distinct width (shared by every folded entry of that width, verified equal
//! across the real checkpoint's 402 packed modules at build time), one block
//! size for the whole checkpoint. `forward` selects signs-then-butterfly on a
//! folded entry's input; the inverse, butterfly-then-signs, un-rotates the
//! embedding's dequantized rows. Both directions are the caller's naming of
//! the same orthogonal map prism's bundled `fwht` applies.
//!
//! Threadgroup budget: the shader's scratch array sizes the largest compiled
//! block (4096 floats = 16 KiB). `HADAMARD_MAX_BLOCK` here is the same
//! constant restated; raising one without the other overruns threadgroup
//! memory or wastes it, so the pair is kept side by side in review.

use metal::FunctionConstantValues;

use crate::context::{GpuError, MetalContext, PassEncoder};

pub const SOURCE: &str = include_str!("shaders/hadamard.metal");

/// The largest butterfly width the shader's threadgroup array compiles for.
pub const HADAMARD_MAX_BLOCK: u32 = 4096;

/// The threadgroup width every dispatch runs at. The butterfly's strided
/// loops make idle threads harmless below the block width, so one width
/// serves every compiled block; it must stay a multiple of 32 (the shader
/// has no SIMD-group reductions, but the ABI wants warps whole).
pub const HADAMARD_THREADS: u64 = 1024;

/// Validates the shape pair a caller is about to dispatch. Split out so the
/// runtime can reject a bad manifest section at OPEN, with the install's
/// name in the message, rather than at first decode.
pub fn hadamard_shape_error(rows: u32, width: u32, block: u32) -> Option<String> {
    if rows == 0 {
        return Some("hadamard dispatch has no rows".to_string());
    }
    if block == 0 || !block.is_power_of_two() || block > HADAMARD_MAX_BLOCK {
        return Some(format!(
            "hadamard block {block} is not a power of two in 2..={HADAMARD_MAX_BLOCK}"
        ));
    }
    if width == 0 || width % block != 0 {
        return Some(format!(
            "hadamard width {width} is not a positive multiple of block {block}"
        ));
    }
    None
}

/// `dst[row, :] = fwht(src[row, :])` for every row, one sign vector per full
/// row width, butterflies over `block`-wide segments. `forward` picks the
/// signs-first order; the embedding's inverse is the same dispatch with it
/// cleared. src and dst may not be the same offset of the same buffer: the
/// kernel reads and writes through one shared threadgroup staging array per
/// (row, segment), but the load and store loops are unordered across
/// threadgroups.
#[allow(clippy::too_many_arguments)]
pub fn encode_hadamard_fwht(
    context: &mut MetalContext,
    pass: &PassEncoder,
    src: (&metal::Buffer, u64),
    dst: (&metal::Buffer, u64),
    signs: (&metal::Buffer, u64),
    rows: u32,
    width: u32,
    block: u32,
    forward: bool,
) -> Result<(), GpuError> {
    if let Some(reason) = hadamard_shape_error(rows, width, block) {
        return Err(GpuError::PipelineCreate(reason));
    }
    let pipeline = context.pipeline(
        SOURCE,
        "hadamard_fwht_rows",
        &FunctionConstantValues::new(),
        b"",
    )?;
    let segments = width / block;
    pass.encode_threads_3d(
        &pipeline,
        &[(src.0, 0, src.1), (dst.0, 1, dst.1), (signs.0, 2, signs.1)],
        &[
            (crate::bytes::u32_bytes(&width), 3),
            (crate::bytes::u32_bytes(&block), 4),
            (crate::bytes::u32_bytes(&(forward as u32)), 5),
        ],
        (rows as u64 * segments as u64 * HADAMARD_THREADS, 1, 1),
        (HADAMARD_THREADS, 1, 1),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_guard_names_the_violation() {
        assert!(hadamard_shape_error(0, 5120, 1024).is_some());
        assert!(hadamard_shape_error(1, 5121, 1024).is_some());
        assert!(hadamard_shape_error(1, 5120, 1000).is_some());
        assert!(hadamard_shape_error(1, 5120, 8192).is_some());
        assert!(hadamard_shape_error(1, 5120, 0).is_some());
        assert!(hadamard_shape_error(1, 5120, 1024).is_none());
    }

    #[test]
    fn max_block_constant_matches_the_shader_array() {
        let source = SOURCE;
        let declared = "constant constexpr uint kHadamardMaxBlock = 4096;";
        assert!(
            source.contains(declared),
            "shader threadgroup array moved; raise HADAMARD_MAX_BLOCK with it"
        );
        assert_eq!(HADAMARD_MAX_BLOCK, 4096);
    }
}
