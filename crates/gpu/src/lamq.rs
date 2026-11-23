//! Host-side dispatch for `shaders/lamq.metal`: LookAtMyQuant (LAMQ) weight
//! codes (`docs/LAMQ.md`). The CPU contract is `turbospark_compute::lamq`,
//! held by `tests/lamq_parity.rs`.
//!
//! Every entry point checks its shape against the kernel's fixed limits and
//! refuses rather than dispatching out of contract: these kernels index
//! threadgroup arrays and packed rows directly and would read out of bounds
//! silently. Every function constant is in the pipeline cache key.

use metal::{FunctionConstantValues, MTLDataType};
use turbospark_compute::lamq::{LamqCode, Rht};

use crate::bytes::u32_bytes;
use crate::{GpuError, MetalContext, PassEncoder};

const SOURCE: &str = include_str!("shaders/lamq.metal");

/// Widest vector `lamq_rht` holds (`kLamqMaxN`).
pub const LAMQ_MAX_N: usize = 3072;
/// Largest dense factor the rotation stages in threadgroup memory.
pub const LAMQ_MAX_NB: usize = 72;
/// Largest table the GEMV stages (`kLamqMaxTable`).
pub const LAMQ_MAX_TABLE: usize = 4096;
/// Rows per GEMV threadgroup, one simdgroup each. Every threadgroup stages the
/// whole lookup table (up to 4096 entries) first; at 8 rows a 1-bit `rtr1`
/// row set read about 3x more table than codes. Must equal the shader's
/// `kLamqRowsPerTG`, and the threadgroup is `ROWS_PER_TG * 32` threads.
const ROWS_PER_TG: u64 = 32;

fn invalid(msg: String) -> GpuError {
    GpuError::InvalidInput(msg)
}

/// One width's Hadamard factors, uploaded: the Sylvester size (applied as an
/// FWHT) and the dense factor `B` as int8 (+1/-1).
pub struct LamqRhtFactor {
    pub n: u32,
    pub na: u32,
    pub nb: u32,
    pub b: metal::Buffer,
}

impl LamqRhtFactor {
    /// Builds and uploads the factor for width `n`, or refuses a width with
    /// no Sylvester x Paley-I split or wider than the kernel holds.
    pub fn new(context: &MetalContext, n: usize) -> Result<Self, GpuError> {
        if n == 0 || n > LAMQ_MAX_N {
            return Err(invalid(format!(
                "LAMQ rotation width {n} outside 1..={LAMQ_MAX_N}"
            )));
        }
        let rht = Rht::new(n).ok_or_else(|| invalid(format!("LAMQ width {n} has no Hadamard")))?;
        let (na, _, nb, b) = rht.factors();
        if nb > LAMQ_MAX_NB {
            return Err(invalid(format!(
                "LAMQ width {n}: dense factor {nb} is wider than {LAMQ_MAX_NB}"
            )));
        }
        // The kernel stages B as int8: refuse a factor that is not exactly
        // +1/-1 rather than truncate it.
        let bytes = b
            .iter()
            .map(|&v| match v {
                1.0 => Ok(1i8 as u8),
                -1.0 => Ok((-1i8) as u8),
                other => Err(invalid(format!(
                    "LAMQ width {n}: dense factor entry {other} is not +1/-1"
                ))),
            })
            .collect::<Result<Vec<u8>, _>>()?;
        Ok(Self {
            n: n as u32,
            na: na as u32,
            nb: nb as u32,
            b: context.new_buffer_with_data(&bytes),
        })
    }
}

/// Rotation direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LamqRhtDir {
    /// `H (s * x) / sqrt(n)`, signs first.
    In,
    /// `s * H^T y / sqrt(n)`, signs last.
    Out,
}

/// The most vectors one batched rotation takes (an Inkling attention group
/// has at most eight projections).
pub const LAMQ_RHT_BATCH_MAX: usize = 8;

/// Input-rotates several FP16 vectors of width `factor.n` in one dispatch,
/// one threadgroup each: job `v` is `(input index into xs, byte offset of its
/// signs in signs_buf)`, written as F32 to `y` at `v * n` elements. Each
/// vector's arithmetic is [`encode_lamq_rht`]'s, so the outputs are
/// bit-identical to rotating them one at a time; the gain is that they run
/// on several cores at once instead of one after another.
pub fn encode_lamq_rht_in_batch(
    context: &mut MetalContext,
    pass: &PassEncoder,
    factor: &LamqRhtFactor,
    xs: &[(&metal::Buffer, u64)],
    signs_buf: &metal::Buffer,
    jobs: &[(u32, u64)],
    y: (&metal::Buffer, u64),
) -> Result<(), GpuError> {
    if jobs.is_empty() || jobs.len() > LAMQ_RHT_BATCH_MAX {
        return Err(invalid(format!(
            "LAMQ batched rotation of {} vectors (1..={LAMQ_RHT_BATCH_MAX})",
            jobs.len()
        )));
    }
    if xs.is_empty() || xs.len() > 3 {
        return Err(invalid(format!(
            "LAMQ batched rotation with {} inputs (1..=3)",
            xs.len()
        )));
    }
    if let Some(&(sel, _)) = jobs.iter().find(|(sel, _)| *sel as usize >= xs.len()) {
        return Err(invalid(format!(
            "LAMQ batched rotation input {sel} of {}",
            xs.len()
        )));
    }
    let split = split_groups(factor);
    let p = context.pipeline(
        SOURCE,
        if split.is_some() {
            "lamq_rht_in_batch_split"
        } else {
            "lamq_rht_in_batch"
        },
        &FunctionConstantValues::new(),
        &[],
    )?;
    let mut params = Vec::with_capacity(16);
    for v in [factor.n, factor.na, factor.nb, 0] {
        params.extend_from_slice(&v.to_le_bytes());
    }
    // LamqRhtJob: x_sel u32, pad u32, s_off u64.
    let mut table = Vec::with_capacity(jobs.len() * 16);
    for &(sel, off) in jobs {
        table.extend_from_slice(&sel.to_le_bytes());
        table.extend_from_slice(&0u32.to_le_bytes());
        table.extend_from_slice(&off.to_le_bytes());
    }
    // Unused input slots repeat the first input; no job selects them.
    let x = |i: usize| xs.get(i).copied().unwrap_or(xs[0]);
    pass.encode_threadgroups(
        &p,
        &[
            (x(0).0, 0, x(0).1),
            (x(1).0, 1, x(1).1),
            (x(2).0, 2, x(2).1),
            (signs_buf, 3, 0),
            (&factor.b, 4, 0),
            (y.0, 5, y.1),
        ],
        &[(&params, 6), (&table, 7)],
        jobs.len() as u64 * split.unwrap_or(1),
        if split.is_some() {
            LAMQ_SPLIT_THREADS
        } else {
            1024
        },
    );
    Ok(())
}

/// Threads per threadgroup of the split rotation kernels. A multiple of 32,
/// so every Sylvester factor of 32 or less tiles it in lane-aligned groups.
const LAMQ_SPLIT_THREADS: u64 = 256;

/// Threadgroups per vector when the split rotation applies (Sylvester
/// factor of 32 or less), mirroring `lamq_split_groups` in lamq.metal.
/// None selects the one-threadgroup kernels (3072 = 256 x 12).
fn split_groups(factor: &LamqRhtFactor) -> Option<u64> {
    let na = u64::from(factor.na);
    if na > 32 {
        return None;
    }
    let cpt = LAMQ_SPLIT_THREADS / na;
    Some(u64::from(factor.nb).div_ceil(cpt))
}

/// Rotates `count` vectors of width `factor.n` (vector `v` at `v * n`
/// elements). `x_half` / `y_half` pick FP16 or F32 for each side.
#[allow(clippy::too_many_arguments)]
pub fn encode_lamq_rht(
    context: &mut MetalContext,
    pass: &PassEncoder,
    factor: &LamqRhtFactor,
    dir: LamqRhtDir,
    x: (&metal::Buffer, u64),
    x_half: bool,
    signs: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    y_half: bool,
    count: u32,
) -> Result<(), GpuError> {
    if count == 0 {
        return Err(invalid("LAMQ rotation of zero vectors".into()));
    }
    let values = FunctionConstantValues::new();
    values.set_constant_value_at_index((&x_half as *const bool).cast(), MTLDataType::Bool, 120);
    values.set_constant_value_at_index((&y_half as *const bool).cast(), MTLDataType::Bool, 121);
    let key = [u8::from(x_half), u8::from(y_half)];
    let split = split_groups(factor);
    let name = if split.is_some() {
        "lamq_rht_split"
    } else {
        "lamq_rht"
    };
    let p = context.pipeline(SOURCE, name, &values, &key)?;
    let dir_v: u32 = match dir {
        LamqRhtDir::In => 0,
        LamqRhtDir::Out => 1,
    };
    let mut params = Vec::with_capacity(16);
    for v in [factor.n, factor.na, factor.nb, dir_v] {
        params.extend_from_slice(&v.to_le_bytes());
    }
    pass.encode_threadgroups(
        &p,
        &[
            (x.0, 0, x.1),
            (signs.0, 1, signs.1),
            (&factor.b, 2, 0),
            (y.0, 3, y.1),
        ],
        &[(&params, 4)],
        u64::from(count) * split.unwrap_or(1),
        // Unsplit, one threadgroup holds the whole vector, so a decode
        // rotation runs on one core: give it the most threads it takes.
        if split.is_some() {
            LAMQ_SPLIT_THREADS
        } else {
            1024
        },
    );
    Ok(())
}

/// The function constants of one code: (k, g, hash shift or 0, levels bits
/// or 0).
fn code_constants(code: LamqCode) -> (u32, u32, u32, u32) {
    let lm = match code {
        LamqCode::Lm6 => 6,
        LamqCode::Lm8 => 8,
        _ => 0,
    };
    (
        code.k(),
        code.g() as u32,
        code.hash_shift().unwrap_or(0),
        lm,
    )
}

fn gemv_pipeline(
    context: &mut MetalContext,
    kernel: &'static str,
    code: LamqCode,
    out_half: bool,
) -> Result<metal::ComputePipelineState, GpuError> {
    let (k, g, shift, lm) = code_constants(code);
    let values = FunctionConstantValues::new();
    values.set_constant_value_at_index((&k as *const u32).cast(), MTLDataType::UInt, 122);
    values.set_constant_value_at_index((&g as *const u32).cast(), MTLDataType::UInt, 123);
    values.set_constant_value_at_index((&shift as *const u32).cast(), MTLDataType::UInt, 124);
    values.set_constant_value_at_index((&out_half as *const bool).cast(), MTLDataType::Bool, 125);
    values.set_constant_value_at_index((&lm as *const u32).cast(), MTLDataType::UInt, 126);
    let mut key = Vec::with_capacity(17);
    for v in [k, g, shift, lm] {
        key.extend_from_slice(&v.to_le_bytes());
    }
    key.push(u8::from(out_half));
    context.pipeline(SOURCE, kernel, &values, &key)
}

/// One coded matrix, as bound for a dispatch.
#[derive(Clone, Copy, Debug)]
pub struct LamqMatrixView<'a> {
    pub code: LamqCode,
    pub rows: u32,
    pub cols: u32,
    pub codes: (&'a metal::Buffer, u64),
    /// f16 per row.
    pub scales: (&'a metal::Buffer, u64),
    /// The shared FP16 LUT (trellis) or the matrix's own F32 levels.
    pub table: (&'a metal::Buffer, u64),
}

impl LamqMatrixView<'_> {
    fn params(&self) -> Result<Vec<u8>, GpuError> {
        let rb = self.code.row_bytes(self.cols as usize).ok_or_else(|| {
            invalid(format!(
                "LAMQ {} width {} does not fit",
                self.code.name(),
                self.cols
            ))
        })?;
        let table_len = self.code.table_rows() * self.code.g();
        if self.rows == 0 || table_len > LAMQ_MAX_TABLE {
            return Err(invalid(format!(
                "LAMQ {} rows {} or table {table_len}",
                self.code.name(),
                self.rows
            )));
        }
        // Trellis rows decode 16 columns per step.
        if self.code.is_trellis() && self.cols % 16 != 0 {
            return Err(invalid(format!(
                "LAMQ {} width {} is not a multiple of 16",
                self.code.name(),
                self.cols
            )));
        }
        // Trellis rows are read as u32 words.
        if self.code.is_trellis() && (self.codes.1 % 4 != 0 || rb % 4 != 0) {
            return Err(invalid(format!(
                "LAMQ {} codes are not 4-byte aligned",
                self.code.name()
            )));
        }
        // f16 scales and LUT entries are 2-byte reads, F32 levels 4-byte.
        let table_align = if self.code.is_trellis() { 2 } else { 4 };
        if self.scales.1 % 2 != 0 || self.table.1 % table_align != 0 {
            return Err(invalid("LAMQ scales or table misaligned".into()));
        }
        let mut b = Vec::with_capacity(16);
        for v in [self.rows, self.cols, rb as u32, table_len as u32] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        Ok(b)
    }
}

/// `y[r] = scale[r] * W~[r] . x` for an already rotated F32 `x[cols]`;
/// `y` is FP16 when `y_half`, else F32.
pub fn encode_lamq_gemv(
    context: &mut MetalContext,
    pass: &PassEncoder,
    m: &LamqMatrixView<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    y_half: bool,
) -> Result<(), GpuError> {
    let params = m.params()?;
    if x.1 % 4 != 0 {
        return Err(invalid("LAMQ gemv input is not 4-byte aligned".into()));
    }
    let p = gemv_pipeline(context, "lamq_gemv", m.code, y_half)?;
    pass.encode_threadgroups(
        &p,
        &[
            (m.codes.0, 0, m.codes.1),
            (m.scales.0, 1, m.scales.1),
            (m.table.0, 2, m.table.1),
            (x.0, 3, x.1),
            (y.0, 4, y.1),
        ],
        &[(&params, 5)],
        u64::from(m.rows).div_ceil(ROWS_PER_TG),
        ROWS_PER_TG * 32,
    );
    Ok(())
}

/// [`encode_lamq_gemv`] over `n_rows` token rows: rotated F32 input row `t`
/// at `x + t * x_stride` floats, output row `t` at `y + t * y_stride`
/// elements. The codes are decoded once per block of 8 rows. A re-summation
/// of [`encode_lamq_gemv`] (within about 1e-6 of a row's RMS, not its bits);
/// a row's bits do not depend on the other rows in the dispatch. Trellis
/// codes only pay: on level codes it is slower than one-row dispatches.
#[allow(clippy::too_many_arguments)]
pub fn encode_lamq_gemv_rows(
    context: &mut MetalContext,
    pass: &PassEncoder,
    m: &LamqMatrixView<'_>,
    x: (&metal::Buffer, u64),
    x_stride: u32,
    y: (&metal::Buffer, u64),
    y_stride: u32,
    n_rows: u32,
    y_half: bool,
) -> Result<(), GpuError> {
    let params = m.params()?;
    if x.1 % 4 != 0 || n_rows == 0 || x_stride < m.cols || y_stride < m.rows {
        return Err(invalid(format!(
            "LAMQ gemv rows: {n_rows} rows, strides {x_stride}/{y_stride} for {}x{}",
            m.rows, m.cols
        )));
    }
    let p = gemv_pipeline(context, "lamq_gemv_rows", m.code, y_half)?;
    pass.encode_threadgroups(
        &p,
        &[
            (m.codes.0, 0, m.codes.1),
            (m.scales.0, 1, m.scales.1),
            (m.table.0, 2, m.table.1),
            (x.0, 3, x.1),
            (y.0, 4, y.1),
        ],
        &[
            (&params, 5),
            (u32_bytes(&n_rows), 6),
            (u32_bytes(&x_stride), 7),
            (u32_bytes(&y_stride), 8),
        ],
        u64::from(m.rows).div_ceil(ROWS_PER_TG),
        ROWS_PER_TG * 32,
    );
    Ok(())
}

/// Row `row` of `m`, decoded and scaled, into F32 `out[cols]`.
pub fn encode_lamq_decode_row(
    context: &mut MetalContext,
    pass: &PassEncoder,
    m: &LamqMatrixView<'_>,
    row: u32,
    out: (&metal::Buffer, u64),
) -> Result<(), GpuError> {
    let params = m.params()?;
    if row >= m.rows {
        return Err(invalid(format!("LAMQ row {row} of {}", m.rows)));
    }
    let p = gemv_pipeline(context, "lamq_decode_row", m.code, false)?;
    let threads = u64::from(m.cols) / m.code.g() as u64;
    pass.encode_threadgroups(
        &p,
        &[
            (m.codes.0, 0, m.codes.1),
            (m.scales.0, 1, m.scales.1),
            (m.table.0, 2, m.table.1),
            (out.0, 3, out.1),
        ],
        &[(&params, 4), (u32_bytes(&row), 5)],
        threads.div_ceil(256),
        256,
    );
    Ok(())
}

/// `dst[order[i]] = half(src[i])` for `i < count`: F32 bucket logits to
/// FP16 logits in vocab order. The caller guarantees `order` is a permutation into `dst` (the
/// intake checks it).
pub fn encode_lamq_scatter(
    context: &mut MetalContext,
    pass: &PassEncoder,
    src: (&metal::Buffer, u64),
    order: (&metal::Buffer, u64),
    dst: (&metal::Buffer, u64),
    count: u32,
) -> Result<(), GpuError> {
    if count == 0 {
        return Err(invalid("LAMQ scatter of zero values".into()));
    }
    let p = context.pipeline(SOURCE, "lamq_scatter", &FunctionConstantValues::new(), b"")?;
    pass.encode_threadgroups(
        &p,
        &[(src.0, 0, src.1), (order.0, 1, order.1), (dst.0, 2, dst.1)],
        &[(u32_bytes(&count), 3)],
        u64::from(count).div_ceil(256),
        256,
    );
    Ok(())
}
