//! Host side of `shaders/gemv_bf16.metal`: GEMV over a resident BF16 matrix.
//!
//! The reader the BF16 resident tag (1) gained for MATRICES with the Bonsai-2
//! line -- the tag was honoured for norms and the embedding before, and every
//! real projection until now was quantized, so `encode_gemv_any` refused tag 1
//! by falling off its dtype match. The first live callers are the dense GDN
//! flow's `in_proj_a`/`in_proj_b`, which the checkpoint ships unquantized; a
//! caller is expected to keep refusing this arm's absence by name rather than
//! looping sequential GEMVs wherever a batched form is missing (the
//! `encode_gemm_any` rule).

use metal::FunctionConstantValues;

use crate::bytes::{f32_bytes, u32_bytes};
use crate::context::{GpuError, MetalContext, PassEncoder};

pub static SOURCE: &str = include_str!("shaders/gemv_bf16.metal");

/// Threads per threadgroup, matched to the shader's `kBf16GemmThreads` and
/// its `partial[]` sizing; widening one without the other overruns.
const THREADS_PER_GROUP: u64 = 256;

/// A whole BF16 weight matrix addressed IN PLACE inside one shared
/// `MTLBuffer` (normally `ResidentGpuWeights::buffer`): `rows * cols * 2`
/// weight bytes at `weights_offset` and nothing else -- BF16 carries no
/// companion planes.
pub struct Bf16ResidentMatrix<'a> {
    pub buffer: &'a metal::Buffer,
    pub weights_offset: u64,
    pub rows: usize,
    pub cols: usize,
}

/// `y[m] = sum_n bf16(W[m,n]) * x[n]` for `x` a row of FP16 at `x.1`, one
/// threadgroup per output row.
pub fn encode_bf16_gemv_resident(
    context: &mut MetalContext,
    pass: &PassEncoder,
    w: &Bf16ResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
) -> Result<(), GpuError> {
    assert!(w.rows > 0);
    assert!(w.cols > 0);
    let m_u32 = w.rows as u32;
    let n_u32 = w.cols as u32;
    let pipeline = context.pipeline(
        SOURCE,
        "bf16_gemv_rows",
        &FunctionConstantValues::new(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[
            (w.buffer, 0, w.weights_offset),
            (x.0, 1, x.1),
            (y.0, 2, y.1),
        ],
        &[(u32_bytes(&m_u32), 3), (u32_bytes(&n_u32), 4)],
        w.rows as u64,
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// Looks up row `token_id` of a resident BF16 embedding table (`[V, d]`,
/// starting at `table.1`) into FP16 `out`, scaled by `out_scale`.
pub fn encode_embed_lookup_bf16(
    context: &mut MetalContext,
    pass: &PassEncoder,
    table: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    token_id: u32,
    d: u32,
    out_scale: f32,
) -> Result<(), GpuError> {
    assert!(d > 0);
    let pipeline = context.pipeline(
        SOURCE,
        "embed_lookup_bf16",
        &FunctionConstantValues::new(),
        b"",
    )?;
    pass.encode_threads_3d(
        &pipeline,
        &[(table.0, 0, table.1), (out.0, 1, out.1)],
        &[
            (u32_bytes(&token_id), 2),
            (u32_bytes(&d), 3),
            (f32_bytes(&out_scale), 4),
        ],
        (d as u64, 1, 1),
        (64, 1, 1),
    );
    Ok(())
}
