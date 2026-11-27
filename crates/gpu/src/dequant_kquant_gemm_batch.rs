//! Host-side dispatch for `shaders/dequant_kquant_batch.metal`: the GGUF
//! Q4_K, Q6_K and Q8_0 GEMVs with B right-hand sides, so the speculative
//! verify pass runs B drafted tokens through one K-quant matrix in ONE
//! dispatch.
//!
//! The kernel header carries the bit-parity argument. This wrapper owns the
//! two host-side ways to break it: the pipeline key MUST carry the baked `B`
//! (crate Gotcha 1: a shared key serves the first-compiled width to every
//! later one, and a wrong B reads the wrong slice of `x` while producing
//! finite output), and the batch cap is an `assert!`, not a clamp, because
//! `acc[B]` is a per-thread register array.

use half::f16;
use metal::{FunctionConstantValues, MTLDataType};

use crate::bytes::{half_slice_to_le_bytes, read_half_buffer, u32_bytes};
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::dequant_int4_batch::MAX_BATCH_ROWS;
use crate::dequant_q4_k_gemv::q4_k_row_bytes;
use crate::dequant_q5_k_gemv::q5_k_row_bytes;
use crate::dequant_q6_k_gemv::q6_k_row_bytes;
use crate::dequant_q8_0_gemv::q8_0_row_bytes;

/// ONE `&'static str`: the pipeline cache keys on its address (Gotcha 4).
const SOURCE: &str = concat!(
    include_str!("shaders/dequant_q4_k.metal"),
    // After Q4_K: it borrows `q4_k_scale_min` (see `dequant_q5_k_gemv`).
    include_str!("shaders/dequant_q5_k.metal"),
    include_str!("shaders/dequant_q6_k.metal"),
    include_str!("shaders/dequant_q8_0.metal"),
    include_str!("shaders/dequant_kquant_batch.metal")
);
const THREADS_PER_GROUP: u64 = 256; // 8 rows/threadgroup * 32 lanes/SIMD group.
const ROWS_PER_THREADGROUP: u64 = 8;

/// The GGUF block types with a batched kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KQuantBatchType {
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
}

impl KQuantBatchType {
    fn kernel(self) -> &'static str {
        match self {
            Self::Q4K => "dequant_q4_k_gemm_simd",
            Self::Q5K => "dequant_q5_k_gemm_simd",
            Self::Q6K => "dequant_q6_k_gemm_simd",
            Self::Q8_0 => "dequant_q8_0_gemm_simd",
        }
    }

    /// Bytes one row of `cols` elements occupies; panics on a ragged row,
    /// like the GEMV helpers it forwards to.
    pub fn row_bytes(self, cols: usize) -> usize {
        match self {
            Self::Q4K => q4_k_row_bytes(cols),
            Self::Q5K => q5_k_row_bytes(cols),
            Self::Q6K => q6_k_row_bytes(cols),
            Self::Q8_0 => q8_0_row_bytes(cols),
        }
    }
}

/// A K-quant weight matrix addressed IN PLACE inside one shared
/// `MTLBuffer`: `rows` consecutive byte runs of `kind.row_bytes(cols)` each,
/// starting at `weights_offset`. The same addressing the per-type
/// `Q*ResidentMatrix` GEMV structs use.
pub struct KQuantResidentMatrix<'a> {
    pub buffer: &'a metal::Buffer,
    pub weights_offset: u64,
    pub rows: usize,
    pub cols: usize,
    pub kind: KQuantBatchType,
}

/// Function constants 102 (B) and 103 (use-FC), with B in the key.
fn specialized_constants(b: u32) -> (FunctionConstantValues, [u8; 4]) {
    let values = FunctionConstantValues::new();
    let on = true;
    values.set_constant_value_at_index((&b as *const u32).cast(), MTLDataType::UInt, 102);
    values.set_constant_value_at_index((&on as *const bool).cast(), MTLDataType::Bool, 103);
    (values, b.to_le_bytes())
}

/// `y[b, m] = sum_n W[m, n] * x[b, n]` for `b` in `0..batch`, over K-quant
/// rows. `x` holds `batch * w.cols` halfs and `y` `batch * w.rows`, both
/// token-major.
///
/// Bit-exact against `batch` calls of the matching GEMV
/// (`encode_dequant_q4_k_gemv_resident` and siblings) at every width in
/// `1..=MAX_BATCH_ROWS`; `crates/gpu/tests/dequant_kquant_gemm_parity.rs` is
/// the assertion.
pub fn encode_dequant_kquant_gemm_resident(
    context: &mut MetalContext,
    pass: &PassEncoder,
    w: &KQuantResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    batch: usize,
) -> Result<(), GpuError> {
    assert!(w.rows > 0);
    let _ = w.kind.row_bytes(w.cols);
    assert!(
        (1..=MAX_BATCH_ROWS).contains(&batch),
        "batch {batch} outside 1..={MAX_BATCH_ROWS}"
    );
    let (m, n, b) = (w.rows as u32, w.cols as u32, batch as u32);
    // The wide Q4_K pair is chosen per matrix exactly as the GEMV chooses it
    // (`q4_k_use_wide`), so verify rows match decode bit for bit.
    let kernel = if w.kind == KQuantBatchType::Q4K
        && crate::dequant_q4_k_gemv::q4_k_use_wide(w.weights_offset)
    {
        if x.1 % 8 != 0 {
            return Err(GpuError::InvalidInput(format!(
                "dequant_q4_k_gemm_wide needs an 8-byte aligned x offset (got {})",
                x.1
            )));
        }
        "dequant_q4_k_gemm_wide"
    } else if w.kind == KQuantBatchType::Q8_0
        && crate::dequant_q8_0_gemv::q8_0_use_wide(w.weights_offset)
    {
        if x.1 % 8 != 0 {
            return Err(GpuError::InvalidInput(format!(
                "dequant_q8_0_gemm_wide needs an 8-byte aligned x offset (got {})",
                x.1
            )));
        }
        "dequant_q8_0_gemm_wide"
    } else {
        w.kind.kernel()
    };
    let (constants, key) = specialized_constants(b);
    let pipeline = context.pipeline(SOURCE, kernel, &constants, &key)?;
    let threadgroups = w.rows.div_ceil(ROWS_PER_THREADGROUP as usize) as u64;
    pass.encode_threadgroups(
        &pipeline,
        &[
            (w.buffer, 0, w.weights_offset),
            (x.0, 1, x.1),
            (y.0, 2, y.1),
        ],
        &[(u32_bytes(&m), 3), (u32_bytes(&n), 4), (u32_bytes(&b), 5)],
        threadgroups,
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// The batched Q4_K verify GEMM with an explicit kernel choice, for the
/// KERN-04 Stage B probe, which times both in one process (the engine path
/// above reads `TURBOSPARK_GGUF_KERNEL` once). Bench only.
pub fn encode_dequant_q4_k_gemm_resident_choose(
    context: &mut MetalContext,
    pass: &PassEncoder,
    w: &KQuantResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    batch: usize,
    wide: bool,
) -> Result<(), GpuError> {
    assert_eq!(w.kind, KQuantBatchType::Q4K);
    assert!((1..=MAX_BATCH_ROWS).contains(&batch));
    if wide && (w.weights_offset % 4 != 0 || x.1 % 8 != 0) {
        return Err(GpuError::InvalidInput(
            "dequant_q4_k_gemm_wide needs 4-byte aligned weights and 8-byte aligned x".into(),
        ));
    }
    let kernel = if wide {
        "dequant_q4_k_gemm_wide"
    } else {
        "dequant_q4_k_gemm_simd"
    };
    let (m, n, b) = (w.rows as u32, w.cols as u32, batch as u32);
    let (constants, key) = specialized_constants(b);
    let pipeline = context.pipeline(SOURCE, kernel, &constants, &key)?;
    pass.encode_threadgroups(
        &pipeline,
        &[
            (w.buffer, 0, w.weights_offset),
            (x.0, 1, x.1),
            (y.0, 2, y.1),
        ],
        &[(u32_bytes(&m), 3), (u32_bytes(&n), 4), (u32_bytes(&b), 5)],
        w.rows.div_ceil(ROWS_PER_THREADGROUP as usize) as u64,
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// `bf16_gemv_rows` and its batched sibling, ONE `&'static str` (Gotcha 4).
const BF16_SOURCE: &str = concat!(
    include_str!("shaders/gemv_bf16.metal"),
    include_str!("shaders/gemv_bf16_batch.metal")
);

/// `y[b, m] = sum_n W[m, n] * x[b, n]` over a resident BF16 matrix, for `b`
/// in `0..batch`, token-major like the K-quant form above. One threadgroup
/// per output row, the GEMV's shape. Bit-exact against `batch` calls of
/// `encode_bf16_gemv_resident`; `gemv_bf16_gemm_parity.rs` is the assertion.
pub fn encode_bf16_gemm_resident(
    context: &mut MetalContext,
    pass: &PassEncoder,
    w: &crate::gemv_bf16::Bf16ResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    batch: usize,
) -> Result<(), GpuError> {
    assert!(w.rows > 0 && w.cols > 0);
    assert!(
        (1..=MAX_BATCH_ROWS).contains(&batch),
        "batch {batch} outside 1..={MAX_BATCH_ROWS}"
    );
    let (m, n, b) = (w.rows as u32, w.cols as u32, batch as u32);
    let (constants, key) = specialized_constants(b);
    let pipeline = context.pipeline(BF16_SOURCE, "bf16_gemm_rows", &constants, &key)?;
    pass.encode_threadgroups(
        &pipeline,
        &[
            (w.buffer, 0, w.weights_offset),
            (x.0, 1, x.1),
            (y.0, 2, y.1),
        ],
        &[(u32_bytes(&m), 3), (u32_bytes(&n), 4), (u32_bytes(&b), 5)],
        w.rows as u64,
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// One-shot [`encode_bf16_gemm_resident`] for the parity tests.
pub fn bf16_gemm_resident(
    context: &mut MetalContext,
    w: &crate::gemv_bf16::Bf16ResidentMatrix<'_>,
    x: &[f16],
    batch: usize,
) -> Result<Vec<f16>, GpuError> {
    assert_eq!(x.len(), w.cols * batch);
    let x_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(x));
    let y_buffer = context.new_output_buffer((w.rows * batch * std::mem::size_of::<u16>()) as u64);
    let pass = context.begin_pass();
    encode_bf16_gemm_resident(context, &pass, w, (&x_buffer, 0), (&y_buffer, 0), batch)?;
    pass.commit_and_wait();
    Ok(read_half_buffer(&y_buffer, w.rows * batch))
}

/// One-shot [`encode_dequant_kquant_gemm_resident`] for the parity tests.
pub fn dequant_kquant_gemm_resident(
    context: &mut MetalContext,
    w: &KQuantResidentMatrix<'_>,
    x: &[f16],
    batch: usize,
) -> Result<Vec<f16>, GpuError> {
    assert_eq!(x.len(), w.cols * batch);
    let x_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(x));
    let y_buffer = context.new_output_buffer((w.rows * batch * std::mem::size_of::<u16>()) as u64);
    let pass = context.begin_pass();
    encode_dequant_kquant_gemm_resident(context, &pass, w, (&x_buffer, 0), (&y_buffer, 0), batch)?;
    pass.commit_and_wait();
    Ok(read_half_buffer(&y_buffer, w.rows * batch))
}
