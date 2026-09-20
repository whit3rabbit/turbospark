//! Host side of the row-wise signed block-Fast-Walsh-Hadamard transform
//! (`shaders/hadamard.metal`): the activation-side half of the prism
//! Hadamard-folded weight contract (the Bonsai-2 line, `docs/BONSAI2.md`).
//!
//! Two kernel shapes behind one dispatch. Blocks 256 and up run the hybrid
//! `hadamard_fwht_block` instantiations: 256 threads per threadgroup, each
//! thread holding `block / 256` elements in registers, shuffles below the
//! simdgroup width, threadgroup memory in the middle stages, and registers
//! above -- bit-identical to the generic kernel, which stays for the narrow
//! blocks the hybrid has no instantiation for. Both are ONE shape per block
//! width because the transform is a property of the ACTIVATION width, not of
//! the weight: one sign vector per distinct width (shared by every folded
//! entry of that width, verified equal across the real checkpoint's 402
//! packed modules at build time), one block size for the whole checkpoint.
//! `forward` selects signs-then-butterfly on a folded entry's input; the
//! inverse, butterfly-then-signs, un-rotates the embedding's dequantized
//! rows. Both directions are the caller's naming of the same orthogonal map
//! prism's bundled `fwht` applies.
//!
//! Threadgroup budget: each kernel's staging array sizes the largest width
//! it compiles for (4096 floats = 16 KiB). `HADAMARD_MAX_BLOCK` here is the
//! same constant restated; raising one without the other overruns
//! threadgroup memory or wastes it, so the pair is kept side by side in
//! review.

use metal::FunctionConstantValues;

use crate::context::{GpuError, MetalContext, PassEncoder};

pub const SOURCE: &str = include_str!("shaders/hadamard.metal");

/// The largest butterfly width the shader's threadgroup array compiles for.
pub const HADAMARD_MAX_BLOCK: u32 = 4096;

/// The threadgroup width the generic strided kernel dispatches at. It now
/// serves only the narrow blocks the hybrid kernel has no instantiation for
/// (below 256); the butterfly's strided loops make idle threads harmless
/// below the block width, so one width serves every compiled block; it must
/// stay a multiple of 32 (the shader has no SIMD-group reductions, but the
/// ABI wants warps whole).
pub const HADAMARD_THREADS: u64 = 1024;

/// The threadgroup width the hybrid block kernel dispatches at: one thread
/// holds `block / 256` elements in registers, so this is also the largest
/// register block the kernel compiles for.
pub const HADAMARD_HYBRID_THREADS: u64 = 256;

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
    // Block widths 256 and up run the hybrid block kernel (shuffles below the
    // simdgroup width, threadgroup memory in the middle, registers above);
    // each instantiation is bit-identical to the generic kernel, only the
    // data movement differs. Narrower blocks keep the generic strided kernel.
    let (function, threads) = match block {
        256 => ("hadamard_fwht_tg_256", HADAMARD_HYBRID_THREADS),
        512 => ("hadamard_fwht_tg_512", HADAMARD_HYBRID_THREADS),
        1024 => ("hadamard_fwht_tg_1024", HADAMARD_HYBRID_THREADS),
        2048 => ("hadamard_fwht_tg_2048", HADAMARD_HYBRID_THREADS),
        4096 => ("hadamard_fwht_tg_4096", HADAMARD_HYBRID_THREADS),
        _ => ("hadamard_fwht_rows", HADAMARD_THREADS),
    };
    let pipeline = context.pipeline(SOURCE, function, &FunctionConstantValues::new(), b"")?;
    let segments = width / block;
    pass.encode_threads_3d(
        &pipeline,
        &[(src.0, 0, src.1), (dst.0, 1, dst.1), (signs.0, 2, signs.1)],
        &[
            (crate::bytes::u32_bytes(&width), 3),
            (crate::bytes::u32_bytes(&block), 4),
            (crate::bytes::u32_bytes(&(forward as u32)), 5),
        ],
        (rows as u64 * segments as u64 * threads, 1, 1),
        (threads, 1, 1),
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

    /// The hybrid instantiations the dispatcher selects by block width must
    /// exist in the shader under the exact names the match arms name: a
    /// renamed or dropped instantiation would only fail at first dispatch.
    #[test]
    fn the_hybrid_instantiations_exist_under_the_dispatched_names() {
        let source = SOURCE;
        for width in [256u32, 512, 1024, 2048, 4096] {
            let host_name = format!("hadamard_fwht_tg_{width}");
            assert!(
                source.contains(&host_name),
                "shader lost the {host_name} instantiation"
            );
            let declared = format!("[[host_name(\"{host_name}\")]]");
            assert!(
                source.contains(&declared),
                "the {host_name} name is not a host_name declaration"
            );
        }
        assert!(source.contains("constant constexpr uint kHadamardHybridThreads = 256;"));
    }
}
