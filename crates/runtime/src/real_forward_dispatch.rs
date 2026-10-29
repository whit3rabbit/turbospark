//! Generic dynamic Metal kernel dispatchers for GEMV, Embedding lookups,
//! and MoE phased execution across affine and GGUF block types.

use model_io::ResidentIndex;

use crate::real_forward_layout::{
    DTYPE_GGUF_IQ1_M, DTYPE_GGUF_IQ1_S, DTYPE_GGUF_IQ2_S, DTYPE_GGUF_IQ2_XS, DTYPE_GGUF_IQ2_XXS,
    DTYPE_GGUF_IQ3_S, DTYPE_GGUF_IQ3_XXS, DTYPE_GGUF_IQ4_NL, DTYPE_GGUF_IQ4_XS, DTYPE_GGUF_Q2_K,
    DTYPE_GGUF_Q3_K, DTYPE_GGUF_Q4_K, DTYPE_GGUF_Q5_K, DTYPE_GGUF_Q6_K, DTYPE_GGUF_Q8_0,
    DTYPE_INT1_AFFINE, DTYPE_INT2_AFFINE, DTYPE_RAW_BF16,
};
use crate::real_forward_types::RealForwardError;
use crate::real_forward_utils::{affine_group_size, entry, resident_matrix};

pub(crate) use crate::real_forward_dispatch_moe::{encode_moe_phase1_any, encode_moe_phase2_any};

/// Shape/size bounds for a block-quantized embedding table: the table must be
/// `[vocab, hidden]`, `hidden` must tile whole superblocks, the byte run must
/// be exactly `rows * row_bytes`, and the selected row must land inside it. A
/// wrong stride or a hand-edited manifest fails here rather than reading a
/// neighbouring token's row.
fn validate_block_quant_embedding(
    e: &model_io::ResidentIndexEntry,
    name: &str,
    token: u32,
    hidden: u32,
    block_elems: usize,
    block_bytes: usize,
    type_name: &str,
) -> Result<(), RealForwardError> {
    let (rows, cols, depth, planes) = e.shape;
    if cols != hidden || depth != 0 || planes != 0 {
        return Err(RealForwardError::Unsupported(format!(
            "embedding table {name}: {type_name} shape {:?} does not match [vocab, {hidden}]",
            e.shape
        )));
    }
    if token >= rows {
        return Err(RealForwardError::Unsupported(format!(
            "embedding table {name}: token {token} is outside {rows} rows"
        )));
    }
    let hidden = u64::from(hidden);
    let block_elems = block_elems as u64;
    if hidden == 0 || hidden % block_elems != 0 {
        return Err(RealForwardError::Unsupported(format!(
            "embedding table {name}: {type_name} width {hidden} is not a whole number of {block_elems}-element blocks"
        )));
    }
    let row_bytes = (hidden / block_elems)
        .checked_mul(block_bytes as u64)
        .ok_or_else(|| {
            RealForwardError::Unsupported(format!(
                "embedding table {name}: {type_name} row size overflows"
            ))
        })?;
    let expected = u64::from(rows).checked_mul(row_bytes).ok_or_else(|| {
        RealForwardError::Unsupported(format!(
            "embedding table {name}: {type_name} table size overflows"
        ))
    })?;
    let selected_end = u64::from(token)
        .checked_add(1)
        .and_then(|row| row.checked_mul(row_bytes))
        .ok_or_else(|| {
            RealForwardError::Unsupported(format!(
                "embedding table {name}: {type_name} selected row overflows"
            ))
        })?;
    if e.size_bytes != expected || selected_end > e.size_bytes {
        return Err(RealForwardError::Unsupported(format!(
            "embedding table {name}: {type_name} packed size {} does not contain {rows}x{hidden} bytes ({expected}) or selected row {token}",
            e.size_bytes
        )));
    }
    Ok(())
}

fn validate_q4_k_embedding(
    e: &model_io::ResidentIndexEntry,
    name: &str,
    token: u32,
    hidden: u32,
) -> Result<(), RealForwardError> {
    validate_block_quant_embedding(
        e,
        name,
        token,
        hidden,
        gpu::Q4_K_BLOCK_ELEMS,
        gpu::Q4_K_BLOCK_BYTES,
        "Q4_K",
    )
}

fn validate_iq4_xs_embedding(
    e: &model_io::ResidentIndexEntry,
    name: &str,
    token: u32,
    hidden: u32,
) -> Result<(), RealForwardError> {
    validate_block_quant_embedding(
        e,
        name,
        token,
        hidden,
        gpu::IQ4_XS_BLOCK_ELEMS,
        gpu::IQ4_XS_BLOCK_BYTES,
        "IQ4_XS",
    )
}

/// Dispatches the embedding lookup matching the table's dtype tag: 4 =
/// INT4-affine, 15 = 1-bit affine, 1 = raw BF16, plus the GGUF block types a
/// real file puts an embedding table in.
///
/// Shared by both real flows rather than written at each one, because it is a
/// property of the tensor and not of the family: Qwen's Q4_K_M keeps
/// `token_embd.weight` at Q4_K, Gemma's published GGUF is Q8_0 throughout,
/// and Swift Qwen3.8 keeps it at IQ4_XS.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_embed_any(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    name: &str,
    out: (&gpu::MetalBuffer, u64),
    token: u32,
    hidden: u32,
    expected_rows: usize,
    embed_scale: f32,
) -> Result<(), RealForwardError> {
    if token as usize >= expected_rows {
        return Err(RealForwardError::Unsupported(format!(
            "token id {token} outside embedding table with {expected_rows} rows"
        )));
    }
    let e = entry(index, name)?;
    let shape = e.shape;
    if shape.0 == 0 || shape.1 != hidden || shape.2 != 0 || shape.3 != 0 {
        return Err(RealForwardError::Unsupported(format!(
            "embedding table {name}: expected [rows, {hidden}], got {shape:?}"
        )));
    }
    if token >= shape.0 {
        return Err(RealForwardError::Unsupported(format!(
            "embedding table {name}: token {token} is outside {} rows",
            shape.0
        )));
    }
    let row_block = match e.dtype {
        DTYPE_GGUF_Q8_0 => Some(gpu::Q8_0_BLOCK_ELEMS),
        DTYPE_GGUF_Q4_K => Some(gpu::Q4_K_BLOCK_ELEMS),
        DTYPE_GGUF_Q6_K => Some(gpu::Q6_K_BLOCK_ELEMS),
        DTYPE_GGUF_Q3_K => Some(gpu::Q3_K_BLOCK_ELEMS),
        DTYPE_GGUF_IQ4_XS => Some(gpu::IQ4_XS_BLOCK_ELEMS),
        DTYPE_GGUF_IQ1_M => Some(gpu::IQ1_M_BLOCK_ELEMS),
        _ => None,
    };
    if let Some(block) = row_block {
        if hidden as usize % block != 0 {
            return Err(RealForwardError::Unsupported(format!(
                "embedding table {name}: row width {hidden} is not divisible by its \
                 {block}-element block"
            )));
        }
    }
    let base = index.header.index_size;
    let table = (weights.buffer(), weights.gpu_offset(e.file_offset - base));
    // A GGUF embedding table is one contiguous byte run per row with its
    // scales inline, so it takes a different kernel rather than the same one
    // with the companion offsets zeroed (ROADMAP Phase G Stage 2).
    match e.dtype {
        DTYPE_GGUF_Q8_0 => {
            gpu::encode_embed_lookup_q8_0(context, pass, table, out, token, hidden, embed_scale)
                .map_err(RealForwardError::Gpu)
        }
        DTYPE_GGUF_Q4_K => {
            validate_q4_k_embedding(e, name, token, hidden)?;
            gpu::encode_embed_lookup_q4_k(context, pass, table, out, token, hidden, embed_scale)
                .map_err(RealForwardError::Gpu)
        }
        DTYPE_GGUF_IQ4_XS => {
            validate_iq4_xs_embedding(e, name, token, hidden)?;
            gpu::encode_embed_lookup_iq4_xs(context, pass, table, out, token, hidden, embed_scale)
                .map_err(RealForwardError::Gpu)
        }
        // Same bounds check the Q4_K arm makes, for the same reason: the
        // stride arithmetic differs from every sibling (110-byte blocks),
        // and a hand-edited manifest fails here rather than reading a
        // neighbouring token's row. No real file keeps `token_embd` at Q3_K
        // today; this is the Q6_K embed lookup's story again.
        DTYPE_GGUF_Q3_K => {
            validate_block_quant_embedding(
                e,
                name,
                token,
                hidden,
                gpu::Q3_K_BLOCK_ELEMS,
                gpu::Q3_K_BLOCK_BYTES,
                "Q3_K",
            )?;
            gpu::encode_embed_lookup_q3_k(context, pass, table, out, token, hidden, embed_scale)
                .map_err(RealForwardError::Gpu)
        }
        // ROADMAP Phase S's candidate puts `token_embd` in Q6_K and ties the
        // head to it, which is what made this kernel worth writing.
        DTYPE_GGUF_Q6_K => {
            gpu::encode_embed_lookup_q6_k(context, pass, table, out, token, hidden, embed_scale)
                .map_err(RealForwardError::Gpu)
        }
        DTYPE_GGUF_IQ1_M => {
            gpu::encode_embed_lookup_iq1_m(context, pass, table, out, token, hidden, embed_scale)
                .map_err(RealForwardError::Gpu)
        }
        // An UNQUANTIZED BF16 table. First carrier: the MLX mxfp4 MiniCPM-V
        // 4.7 checkpoint, which leaves `embed_tokens` BF16. The size is the
        // whole shape check, since there are no companion planes.
        DTYPE_RAW_BF16 => {
            let expected = shape.0 as u64 * u64::from(hidden) * 2;
            if e.size_bytes != expected {
                return Err(RealForwardError::Unsupported(format!(
                    "embedding table {name}: BF16 size {} does not match {} x {hidden} \
                     ({expected})",
                    e.size_bytes, shape.0
                )));
            }
            gpu::encode_embed_lookup_bf16(context, pass, table, out, token, hidden, embed_scale)
                .map_err(RealForwardError::Gpu)
        }
        // The 1-bit table (ROADMAP's 1-bit entry). It exists because the real
        // checkpoint quantizes `embed_tokens` at one bit like everything else,
        // which was read off its safetensors header rather than assumed.
        //
        DTYPE_INT1_AFFINE | DTYPE_INT2_AFFINE => {
            let bits = if e.dtype == DTYPE_INT1_AFFINE { 1 } else { 2 };
            let d = hidden as usize;
            let group_size = affine_group_size(e, name, expected_rows, d, bits)?;
            let scales = (weights.buffer(), weights.gpu_offset(e.scale_offset - base));
            let biases = (weights.buffer(), weights.gpu_offset(e.bias_offset - base));
            let encode = if bits == 1 {
                gpu::encode_embed_lookup_int1
            } else {
                gpu::encode_embed_lookup_int2
            };
            encode(
                context,
                pass,
                table,
                scales,
                biases,
                out,
                token,
                hidden,
                group_size as u32,
                embed_scale,
            )
            .map_err(RealForwardError::Gpu)
        }
        5 => {
            // The INT8-affine table: one byte per element, so the resident
            // byte run IS `rows * hidden` exactly, and the row count falls
            // straight out of the size. First carrier: the dense humanizer
            // Gemma 4 12B, whose embedding is INT8 at group 64 like every
            // other quantized tensor in the checkpoint.
            let d = hidden as usize;
            if d == 0 || e.size_bytes as usize % d != 0 {
                return Err(RealForwardError::Unsupported(format!(
                    "embedding table {name}: {} INT8 bytes is not a whole number of \
                     {d}-element rows",
                    e.size_bytes
                )));
            }
            let rows = e.size_bytes as usize / d;
            let group_size = affine_group_size(e, name, rows, d, 8)?;
            if group_size != 64 {
                return Err(RealForwardError::Unsupported(format!(
                    "embedding table {name}: INT8 group size {group_size} does not match kernel \
                     group size 64"
                )));
            }
            let scales = (weights.buffer(), weights.gpu_offset(e.scale_offset - base));
            let biases = (weights.buffer(), weights.gpu_offset(e.bias_offset - base));
            gpu::encode_embed_lookup_int8(
                context,
                pass,
                table,
                scales,
                biases,
                out,
                token,
                hidden,
                embed_scale,
            )
            .map_err(RealForwardError::Gpu)
        }
        4 => {
            // Resolved inside this arm on purpose: a GGUF entry carries no
            // companions, so its `scale_offset` is 0 and subtracting the
            // index size underflows.
            let scales = (weights.buffer(), weights.gpu_offset(e.scale_offset - base));
            let biases = (weights.buffer(), weights.gpu_offset(e.bias_offset - base));
            let d = hidden as usize;
            if d == 0 || (e.size_bytes as usize * 2) % d != 0 {
                return Err(RealForwardError::Unsupported(format!(
                    "embedding table {name}: {} INT4 bytes is not a whole number of \
                     {d}-element rows",
                    e.size_bytes
                )));
            }
            let rows = e.size_bytes as usize * 2 / d;
            let group_size = affine_group_size(e, name, rows, d, 4)?;
            if group_size != 64 {
                return Err(RealForwardError::Unsupported(format!(
                    "embedding table {name}: INT4 group size {group_size} does not match kernel \
                     group size 64"
                )));
            }
            if token as usize >= rows {
                return Err(RealForwardError::Unsupported(format!(
                    "embedding table {name}: token id {token} outside {rows} rows"
                )));
            }
            gpu::encode_embed_lookup_int4(
                context,
                pass,
                table,
                scales,
                biases,
                out,
                token,
                hidden,
                embed_scale,
            )
            .map_err(RealForwardError::Gpu)
        }
        other => Err(RealForwardError::Unsupported(format!(
            "embedding table {name}: dtype {other} has no dispatched lookup kernel"
        ))),
    }
}

/// `TURBOSPARK_INT4_MMA8=1`, default OFF, read once. Routes every INT4
/// projection of at least [`int4_mma8_min_rows`] output rows that reaches
/// [`encode_gemv_any`] (one row) or [`encode_gemm_any`] (up to 16 rows, in
/// tiles of 8) through the small-M matrix-unit GEMM
/// (`docs/INT4_SMALL_M_MMA.md`). That kernel is not
/// bit-identical to the GEMV, but a row's bits do not depend on the rows
/// beside it, so decode and speculative verify both running it stay
/// bit-exact against each other: the "shared arithmetic" route. It changes
/// decode numerics, so frozen digests do not hold under it. The fused INT4
/// paths that bypass these dispatchers (the FFN fusions and the GDN in-proj
/// fusion) step aside whenever one of their matrices is served
/// ([`int4_mma8_takes`]), so both sides meet the same kernel.
///
/// `TURBOSPARK_INT4_MMA8=verify` is the DRIFT-BOUND arm instead: only the
/// batched dispatcher takes the tile, one-row decode and the fused decode
/// kernels keep GEMV arithmetic. Verify rows then differ from decode in the
/// last bits, so speculation is no longer bit-exact against plain decode;
/// `int4_mma8_drift_probe.rs` measures by how much.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Int4Mma8Mode {
    Off,
    /// Decode and verify both run the tile (bit-exact against each other).
    Shared,
    /// Only batched calls run the tile (drift-bound).
    Verify,
}

pub(crate) fn int4_mma8_mode() -> Int4Mma8Mode {
    static MODE: std::sync::OnceLock<Int4Mma8Mode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("TURBOSPARK_INT4_MMA8").as_deref() {
        Ok("1") => Int4Mma8Mode::Shared,
        Ok("verify") => Int4Mma8Mode::Verify,
        _ => Int4Mma8Mode::Off,
    })
}

pub(crate) fn int4_mma8() -> bool {
    int4_mma8_mode() == Int4Mma8Mode::Shared
}

/// The fewest output rows the shared tile serves under
/// [`int4_mma8`]. Below it the tile loses to the GEMV by 1.8-2.6x (k/v at
/// 1024 rows, the GDN a/b at 48, `mma8_k_split_sweep`), and it does not need
/// to serve them: the GEMV and the shipped batched GEMM are already
/// bit-identical, so a smaller matrix keeps that pair and decode and verify
/// still agree. A function of the SHAPE only, never of the row count.
/// `TURBOSPARK_INT4_MMA8_MIN_ROWS` overrides it, read once, for tests: the
/// synthetic installs are 128 wide, so `1` serves every matrix and a value
/// between their widths gives a mix of served and unserved ones.
fn int4_mma8_min_rows() -> usize {
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("TURBOSPARK_INT4_MMA8_MIN_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096)
    })
}

/// Whether an INT4 `rows x cols` projection runs the shared tile: the flag
/// is on and the shape is one the tile serves. Every INT4 path that can
/// bypass the dispatchers (the fused FFN and GDN in-proj kernels, which are
/// GEMV arithmetic) asks this for each of its matrices and steps aside only
/// when one of them is served.
pub(crate) fn int4_mma8_takes(rows: usize, _cols: usize) -> bool {
    int4_mma8() && rows >= int4_mma8_min_rows()
}

/// [`int4_mma8_takes`] for the BATCHED dispatcher: true in both modes, so
/// the verify-only arm reaches the tile here and only here.
fn int4_mma8_takes_batched(rows: usize) -> bool {
    int4_mma8_mode() != Int4Mma8Mode::Off && rows >= int4_mma8_min_rows()
}

/// [`gpu::encode_dequant_int4_mma8_resident`] over any `batch`, in tiles of
/// [`gpu::INT4_MMA8_MAX_ROWS`] rows (token-major x and y).
fn encode_int4_mma8_rows(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    w: &gpu::Int4ResidentMatrix<'_>,
    x: (&gpu::MetalBuffer, u64),
    y: (&gpu::MetalBuffer, u64),
    batch: usize,
) -> Result<(), RealForwardError> {
    foundation::engaged!("int4_mma8");
    let mut done = 0usize;
    while done < batch {
        let take = (batch - done).min(gpu::INT4_MMA8_MAX_ROWS);
        gpu::encode_dequant_int4_mma8_resident(
            context,
            pass,
            w,
            (x.0, x.1 + (done * w.cols * 2) as u64),
            (y.0, y.1 + (done * w.rows * 2) as u64),
            take,
        )
        .map_err(RealForwardError::Gpu)?;
        done += take;
    }
    Ok(())
}

/// Resolves a packed projection by its dtype tag: 4 = INT4-affine,
/// 5 = INT8-affine, 15 = 1-bit affine, 16 = 2-bit affine (the resident
/// writer's tags), and encodes the matching offset-bound GEMV.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemv_any(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    name: &str,
    rows: usize,
    cols: usize,
    x: (&gpu::MetalBuffer, u64),
    y: (&gpu::MetalBuffer, u64),
) -> Result<(), RealForwardError> {
    let e = entry(index, name)?;
    let base = index.header.index_size;
    match e.dtype {
        5 => {
            let group_size = affine_group_size(e, name, rows, cols, 8)?;
            if group_size != 64 {
                return Err(RealForwardError::Unsupported(format!(
                    "tensor {name}: INT8 group size {group_size} does not match kernel group size 64"
                )));
            }
            let w = gpu::Int8ResidentMatrix {
                buffer: weights.buffer(),
                weights_offset: weights.gpu_offset(e.file_offset - base),
                scales_offset: weights.gpu_offset(e.scale_offset - base),
                biases_offset: weights.gpu_offset(e.bias_offset - base),
                rows,
                cols,
            };
            gpu::encode_dequant_int8_gemv_resident(context, pass, &w, x, y)
                .map_err(RealForwardError::Gpu)
        }
        4 => {
            let w = resident_matrix(weights, index, name, rows, cols)?;
            if int4_mma8_takes(rows, cols) {
                return encode_int4_mma8_rows(context, pass, &w, x, y, 1);
            }
            gpu::encode_dequant_int4_gemv_resident(context, pass, &w, x, y)
                .map_err(RealForwardError::Gpu)
        }
        // 1-BIT AFFINE (ROADMAP's 1-bit entry, step 4). The same three planar
        // regions as the two arms above and NOT a narrower version of them:
        // the companions are FP16 rather than BF16 (same width, so no length
        // check can tell), and the group size is the checkpoint's rather than
        // the siblings' compile-time 64 -- so it is DERIVED from the entry
        // here rather than named, and the derivation doubles as the shape
        // check the other arms spell out.
        //
        // The SYMMETRIC kernel is deliberately not reachable from here. It is
        // a different summation order (`crates/gpu`'s module header), so
        // choosing it per tensor at dispatch time would make the bytes a
        // function of which tensors happened to quantize symmetrically. If it
        // is ever wired, that decision belongs at repack time and needs its
        // own dtype tag.
        DTYPE_INT1_AFFINE => {
            let group_size = affine_group_size(e, name, rows, cols, 1)?;
            let w = gpu::Int1ResidentMatrix {
                buffer: weights.buffer(),
                weights_offset: weights.gpu_offset(e.file_offset - base),
                scales_offset: weights.gpu_offset(e.scale_offset - base),
                biases_offset: weights.gpu_offset(e.bias_offset - base),
                rows,
                cols,
                group_size,
            };
            gpu::encode_dequant_int1_gemv_resident(context, pass, &w, x, y)
                .map_err(RealForwardError::Gpu)
        }
        // 2-BIT AFFINE (ROADMAP's ternary entry). The 1-bit arm above with one
        // constant moved, and the constant is the ONLY thing that separates
        // them at this level: the three planar regions, the FP16 companions
        // and the derived group size are identical, so the dtype TAG is what
        // says how wide a row is. There is no symmetric fast path to choose
        // between here for the 1-bit arm's reason, restated in
        // `crates/compute`'s `quant_2bit` header.
        DTYPE_INT2_AFFINE => {
            let group_size = affine_group_size(e, name, rows, cols, 2)?;
            let w = gpu::Int2ResidentMatrix {
                buffer: weights.buffer(),
                weights_offset: weights.gpu_offset(e.file_offset - base),
                scales_offset: weights.gpu_offset(e.scale_offset - base),
                biases_offset: weights.gpu_offset(e.bias_offset - base),
                rows,
                cols,
                group_size,
            };
            gpu::encode_dequant_int2_gemv_resident(context, pass, &w, x, y)
                .map_err(RealForwardError::Gpu)
        }
        // GGUF Q8_0 (ROADMAP Phase G Stage 2). One byte run, no companions:
        // the entry's scale/bias offsets are zero and are not read.
        DTYPE_GGUF_Q8_0 => {
            let expected = gpu::q8_0_row_bytes(cols) * rows;
            if e.size_bytes as usize != expected {
                return Err(RealForwardError::Unsupported(format!(
                    "tensor {name}: Q8_0 packed size {} does not match {rows}x{cols} ({expected})",
                    e.size_bytes
                )));
            }
            let w = gpu::Q8_0ResidentMatrix {
                buffer: weights.buffer(),
                weights_offset: weights.gpu_offset(e.file_offset - base),
                rows,
                cols,
            };
            gpu::encode_dequant_q8_0_gemv_resident(context, pass, &w, x, y)
                .map_err(RealForwardError::Gpu)
        }
        DTYPE_GGUF_Q2_K => {
            let expected = gpu::q2_k_row_bytes(cols) * rows;
            if e.size_bytes as usize != expected {
                return Err(RealForwardError::Unsupported(format!(
                    "tensor {name}: Q2_K packed size {} does not match {rows}x{cols} ({expected})",
                    e.size_bytes
                )));
            }
            let w = gpu::Q2KResidentMatrix {
                buffer: weights.buffer(),
                weights_offset: weights.gpu_offset(e.file_offset - base),
                rows,
                cols,
            };
            gpu::encode_dequant_q2_k_gemv_resident(context, pass, &w, x, y)
                .map_err(RealForwardError::Gpu)
        }
        // The other two GGUF block types a real file uses for a matrix.
        // Same shape as the Q8_0 arm; only the row-bytes function and the
        // kernel differ, and the size check is what catches a tensor whose
        // dtype tag and byte count disagree.
        DTYPE_GGUF_Q4_K => {
            let expected = gpu::q4_k_row_bytes(cols) * rows;
            if e.size_bytes as usize != expected {
                return Err(RealForwardError::Unsupported(format!(
                    "tensor {name}: Q4_K packed size {} does not match {rows}x{cols} ({expected})",
                    e.size_bytes
                )));
            }
            let w = gpu::Q4KResidentMatrix {
                buffer: weights.buffer(),
                weights_offset: weights.gpu_offset(e.file_offset - base),
                rows,
                cols,
            };
            gpu::encode_dequant_q4_k_gemv_resident(context, pass, &w, x, y)
                .map_err(RealForwardError::Gpu)
        }
        DTYPE_GGUF_Q5_K => {
            let expected = gpu::q5_k_row_bytes(cols) * rows;
            if e.size_bytes as usize != expected {
                return Err(RealForwardError::Unsupported(format!(
                    "tensor {name}: Q5_K packed size {} does not match {rows}x{cols} ({expected})",
                    e.size_bytes
                )));
            }
            let w = gpu::Q5KResidentMatrix {
                buffer: weights.buffer(),
                weights_offset: weights.gpu_offset(e.file_offset - base),
                rows,
                cols,
            };
            gpu::encode_dequant_q5_k_gemv_resident(context, pass, &w, x, y)
                .map_err(RealForwardError::Gpu)
        }
        DTYPE_GGUF_Q6_K => {
            let expected = gpu::q6_k_row_bytes(cols) * rows;
            if e.size_bytes as usize != expected {
                return Err(RealForwardError::Unsupported(format!(
                    "tensor {name}: Q6_K packed size {} does not match {rows}x{cols} ({expected})",
                    e.size_bytes
                )));
            }
            let w = gpu::Q6KResidentMatrix {
                buffer: weights.buffer(),
                weights_offset: weights.gpu_offset(e.file_offset - base),
                rows,
                cols,
            };
            gpu::encode_dequant_q6_k_gemv_resident(context, pass, &w, x, y)
                .map_err(RealForwardError::Gpu)
        }
        // The Dense Qwen2 roadmap item's block type: the pinned Qwen2.5
        // Q3_K_M keeps every attention and FFN projection here, so this arm
        // is the one a real Q3_K install runs on every token. Same shape as
        // the Q8_0 arm; only the row-bytes function (110/256, against the
        // 110/144 confusion with Q4_K) and the kernel differ, and the size
        // check is what catches a tensor whose dtype tag and byte count
        // disagree.
        DTYPE_GGUF_Q3_K => {
            let expected = gpu::q3_k_row_bytes(cols) * rows;
            if e.size_bytes as usize != expected {
                return Err(RealForwardError::Unsupported(format!(
                    "tensor {name}: Q3_K packed size {} does not match {rows}x{cols} ({expected})",
                    e.size_bytes
                )));
            }
            let w = gpu::Q3KResidentMatrix {
                buffer: weights.buffer(),
                weights_offset: weights.gpu_offset(e.file_offset - base),
                rows,
                cols,
            };
            gpu::encode_dequant_q3_k_gemv_resident(context, pass, &w, x, y)
                .map_err(RealForwardError::Gpu)
        }
        dtype @ (DTYPE_GGUF_IQ3_XXS | DTYPE_GGUF_IQ4_NL | DTYPE_GGUF_IQ4_XS
        | DTYPE_GGUF_IQ2_XXS | DTYPE_GGUF_IQ2_XS | DTYPE_GGUF_IQ1_S | DTYPE_GGUF_IQ3_S
        | DTYPE_GGUF_IQ2_S | DTYPE_GGUF_IQ1_M) => {
            let kind = match dtype {
                DTYPE_GGUF_IQ3_XXS => gpu::IqBlockType::Iq3Xxs,
                DTYPE_GGUF_IQ4_NL => gpu::IqBlockType::Iq4Nl,
                DTYPE_GGUF_IQ4_XS => gpu::IqBlockType::Iq4Xs,
                DTYPE_GGUF_IQ2_XXS => gpu::IqBlockType::Iq2Xxs,
                DTYPE_GGUF_IQ2_XS => gpu::IqBlockType::Iq2Xs,
                DTYPE_GGUF_IQ1_S => gpu::IqBlockType::Iq1S,
                DTYPE_GGUF_IQ3_S => gpu::IqBlockType::Iq3S,
                DTYPE_GGUF_IQ2_S => gpu::IqBlockType::Iq2S,
                DTYPE_GGUF_IQ1_M => gpu::IqBlockType::Iq1M,
                _ => unreachable!(),
            };
            let expected = kind.row_bytes(cols) * rows;
            if e.size_bytes as usize != expected {
                return Err(RealForwardError::Unsupported(format!(
                    "tensor {name}: {kind:?} packed size {} does not match {rows}x{cols} ({expected})",
                    e.size_bytes
                )));
            }
            let w = gpu::IqResidentMatrix {
                buffer: weights.buffer(),
                weights_offset: weights.gpu_offset(e.file_offset - base),
                rows,
                cols,
                kind,
            };
            gpu::encode_dequant_iq_gemv_resident(context, pass, &w, x, y)
                .map_err(RealForwardError::Gpu)
        }
        // UNQUANTIZED BF16 matrix (tag 1 narrowed at repack, AGENTS.md
        // Gotcha 45). The reader the Bonsai-2 line needed for its F32-shipped
        // `linear_attn.in_proj_a`/`in_proj_b`: every earlier install
        // quantized them, so this arm never had a live caller until now. No
        // companion planes; the size check is the whole shape assertion.
        DTYPE_RAW_BF16 => {
            let expected = rows * cols * 2;
            if e.size_bytes as usize != expected {
                return Err(RealForwardError::Unsupported(format!(
                    "tensor {name}: BF16 size {} does not match {rows}x{cols} ({expected})",
                    e.size_bytes
                )));
            }
            let w = gpu::Bf16ResidentMatrix {
                buffer: weights.buffer(),
                weights_offset: weights.gpu_offset(e.file_offset - base),
                rows,
                cols,
            };
            gpu::encode_bf16_gemv_resident(context, pass, &w, x, y).map_err(RealForwardError::Gpu)
        }
        // Named rather than defaulted. This arm used to be `_ => int4`,
        // which meant any future dtype tag was read as INT4-affine: no
        // error, just wrong numbers, since the three planar regions an
        // INT4 tensor expects do not exist in a block-quantized one. The
        // GGUF tags (ROADMAP Phase G) are the first tags able to reach it.
        other => Err(RealForwardError::Unsupported(format!(
            "tensor {name}: dtype {other} has no dispatched GEMV kernel"
        ))),
    }
}

/// [`encode_gemv_any`] at `batch` rows: `y[b, m] = sum_n W[m, n] * x[b, n]`.
///
/// `x` holds `batch * cols` halfs and `y` `batch * rows`, both TOKEN-MAJOR,
/// so one token's slice of either stays contiguous and can be handed to a
/// per-token kernel unchanged. That is what lets the batched verify pass
/// batch its GEMVs while leaving norms, RoPE, attention and the recurrent
/// step looping per token, which is exactly the split
/// `docs/MTP_SPECULATIVE.md`'s composite assumes.
///
/// **EVERY DTYPE WITHOUT A BATCHED KERNEL IS REFUSED BY NAME, NEVER
/// LOOPED.** A fallback of `batch` sequential GEMVs is numerically
/// identical, so it would pass every parity and losslessness test there is
/// -- while making a "batched" verify measure the SEQUENTIAL engine and
/// report its cost as the batched one. That is AGENTS.md Gotcha 35's
/// failure (a measurement tool must not inherit a silent default) one
/// layer down, and it would corrupt the only number step 4 exists to
/// produce. The affine sub-byte pair joined INT4 (ROADMAP P3.3): both
/// kernels are bit-exact against their GEMVs at every batch width, which is
/// the gate that admits them. Every GGUF block type still has none.
///
/// The batch bound is an `Err` here and an `assert!` inside
/// `encode_dequant_int4_gemm_resident`. Exceeding `MAX_BATCH_ROWS` is a
/// call-site bug and the kernel is right to abort on it, but a runtime path
/// that can be reached with a caller-chosen block size should say so
/// without killing the process.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemm_any(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    name: &str,
    rows: usize,
    cols: usize,
    x: (&gpu::MetalBuffer, u64),
    y: (&gpu::MetalBuffer, u64),
    batch: usize,
) -> Result<(), RealForwardError> {
    let base = index.header.index_size;
    if !(1..=gpu::MAX_BATCH_ROWS).contains(&batch) {
        return Err(RealForwardError::Unsupported(format!(
            "tensor {name}: batch {batch} outside 1..={} (the batched INT4 kernel's \
             accumulators are a per-thread register array, so the cap is the register \
             file rather than a tuning choice)",
            gpu::MAX_BATCH_ROWS
        )));
    }
    let e = entry(index, name)?;
    match e.dtype {
        // A LITERAL, matching `encode_gemv_any`'s arm above, because INT4
        // and INT8 have no named constants the way the GGUF and sub-4-bit
        // tags do. Spelling it `DTYPE_INT4_AFFINE` here does not fail to
        // compile -- it becomes a catch-all BINDING that matches every
        // dtype and routes a Q4_K tensor into the INT4 kernel. The compiler
        // warns; nothing else would.
        4 => {
            let w = resident_matrix(weights, index, name, rows, cols)?;
            if int4_mma8_takes_batched(rows) {
                return encode_int4_mma8_rows(context, pass, &w, x, y, batch);
            }
            gpu::encode_dequant_int4_gemm_resident(context, pass, &w, x, y, batch)
                .map_err(RealForwardError::Gpu)
        }
        // The INT8-affine batched kernel, the dense humanizer's width. The
        // same refusal logic `encode_gemv_any`'s 5-arm carries, applied to
        // the batched pair: group 64 only, because that is what the kernel
        // is compiled against.
        5 => {
            let group_size = affine_group_size(e, name, rows, cols, 8)?;
            if group_size != 64 {
                return Err(RealForwardError::Unsupported(format!(
                    "tensor {name}: INT8 group size {group_size} does not match kernel \
                     group size 64"
                )));
            }
            let w = gpu::Int8ResidentMatrix {
                buffer: weights.buffer(),
                weights_offset: weights.gpu_offset(e.file_offset - base),
                scales_offset: weights.gpu_offset(e.scale_offset - base),
                biases_offset: weights.gpu_offset(e.bias_offset - base),
                rows,
                cols,
            };
            gpu::encode_dequant_int8_gemm_resident(context, pass, &w, x, y, batch)
                .map_err(RealForwardError::Gpu)
        }
        // The sub-byte affine pair (ROADMAP P3.3), mirroring
        // `encode_gemv_any`'s arms beside them: the companions are FP16 and
        // the group size is the checkpoint's, so it is DERIVED from the
        // entry rather than named, and both batched kernels are held
        // bit-exact against their GEMVs by the parity tests rather than
        // asserted here.
        DTYPE_INT1_AFFINE => {
            let group_size = affine_group_size(e, name, rows, cols, 1)?;
            let w = gpu::Int1ResidentMatrix {
                buffer: weights.buffer(),
                weights_offset: weights.gpu_offset(e.file_offset - base),
                scales_offset: weights.gpu_offset(e.scale_offset - base),
                biases_offset: weights.gpu_offset(e.bias_offset - base),
                rows,
                cols,
                group_size,
            };
            gpu::encode_dequant_int1_gemm_resident(context, pass, &w, x, y, batch)
                .map_err(RealForwardError::Gpu)
        }
        DTYPE_INT2_AFFINE => {
            let group_size = affine_group_size(e, name, rows, cols, 2)?;
            let w = gpu::Int2ResidentMatrix {
                buffer: weights.buffer(),
                weights_offset: weights.gpu_offset(e.file_offset - base),
                scales_offset: weights.gpu_offset(e.scale_offset - base),
                biases_offset: weights.gpu_offset(e.bias_offset - base),
                rows,
                cols,
                group_size,
            };
            gpu::encode_dequant_int2_gemm_resident(context, pass, &w, x, y, batch)
                .map_err(RealForwardError::Gpu)
        }
        other => Err(RealForwardError::Unsupported(format!(
            "tensor {name}: dtype {other} has no BATCHED kernel; INT4-affine (4), \
             INT8-affine (5), 1-bit affine (15) and 2-bit affine (16) do. Refused \
             rather than looped: a sequential fallback here is numerically \
             identical and would silently measure the unbatched engine"
        ))),
    }
}

/// The fewest rows the Steel INT4 GEMM is used for. Its 32-row tile wastes
/// half its work at 16 rows, where it measured only 1.2-1.4x the scalar
/// kernel on Qwen3.8's large projections and 0.61x on the 1024-row K/V ones;
/// from 32 rows it wins on every projection
/// (`crates/gpu/tests/qwen38_int4_prefill_gemm_probe.rs`).
const STEEL_MIN_ROWS: usize = 32;

/// Whether the small-N prefill route is on. It replaces the scalar GEMM's
/// 16-row slices for INT4 matrices too narrow for a whole Steel tile
/// (`gpu::small_n_supports`), only on a scratch that already accepted Steel's
/// rounding. Read once; `TURBOSPARK_PREFILL_SMALL_N=0` restores those slices.
fn small_n_prefill() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TURBOSPARK_PREFILL_SMALL_N").as_deref() != Ok("0"))
}

/// [`encode_gemm_any`] for any positive `batch`, row-major `x` (`cols` halfs
/// per row) and `y` (`rows` halfs per row). With `steel` set, an INT4 tensor
/// the vendored kernel can serve runs as ONE Steel dispatch from
/// [`STEEL_MIN_ROWS`] rows up; an INT4 tensor with too few output rows for a
/// whole Steel tile (Qwen3.8's 48-row GDN `a`/`b`) takes the small-N route
/// (`gpu::encode_dequant_int4_gemm_small_n`); everything else runs the scalar
/// kernel in slices of `gpu::MAX_BATCH_ROWS` rows. Only a caller that has
/// accepted Steel's rounding passes `steel` (the chunked-prefill scratch,
/// `BatchedScratch::steel_gemm`); without it the result is the scalar
/// kernel's, bit for bit, whatever `batch` is.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemm_rows(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    name: &str,
    rows: usize,
    cols: usize,
    x: (&gpu::MetalBuffer, u64),
    y: (&gpu::MetalBuffer, u64),
    batch: usize,
    steel: bool,
) -> Result<(), RealForwardError> {
    if steel && entry(index, name)?.dtype == 4 {
        let w = resident_matrix(weights, index, name, rows, cols)?;
        let tile = gpu::SteelTile::M32K32N32;
        if batch >= STEEL_MIN_ROWS && gpu::steel_supports(&w, tile) {
            foundation::engaged!("prefill_steel");
            return gpu::encode_dequant_int4_gemm_steel(context, pass, &w, x, y, batch, tile)
                .map_err(RealForwardError::Gpu);
        }
        if small_n_prefill() && gpu::small_n_supports(&w) {
            foundation::engaged!("prefill_small_n");
            return gpu::encode_dequant_int4_gemm_small_n(context, pass, &w, x, y, batch)
                .map_err(RealForwardError::Gpu);
        }
    }
    let mut done = 0usize;
    while done < batch {
        let take = (batch - done).min(gpu::MAX_BATCH_ROWS);
        encode_gemm_any(
            context,
            pass,
            weights,
            index,
            name,
            rows,
            cols,
            (x.0, x.1 + (done * cols * 2) as u64),
            (y.0, y.1 + (done * rows * 2) as u64),
            take,
        )?;
        done += take;
    }
    Ok(())
}

/// The `router_topk_select_k8` kernel's semantics on the host: top-`k` by
/// score with ties preferring the lower expert index, softmax over the
/// selected scores only, each weight multiplied by that expert's
/// `per_expert_scale`. (Distinct from the synthetic path's
/// softmax-over-all-then-renormalize `topk_softmax`.)
pub(crate) fn router_topk_gemma4(
    logits: &[f32],
    k: usize,
    per_expert_scale: &[f32],
) -> (Vec<usize>, Vec<f32>) {
    if logits.is_empty() || k == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    let selected: Vec<usize> = order.into_iter().take(k).collect();
    let max = logits[selected[0]];
    let exps: Vec<f32> = selected.iter().map(|&i| (logits[i] - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let weights = selected
        .iter()
        .zip(exps.iter())
        .map(|(&i, e)| {
            let scale = per_expert_scale.get(i).copied().unwrap_or(1.0);
            e / sum * scale
        })
        .collect();
    (selected, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q4_k_embedding(rows: u32, cols: u32) -> model_io::ResidentIndexEntry {
        model_io::ResidentIndexEntry {
            name: "token_embd.weight".to_string(),
            dtype: DTYPE_GGUF_Q4_K,
            file_offset: 4096,
            size_bytes: u64::from(rows)
                * (u64::from(cols) / gpu::Q4_K_BLOCK_ELEMS as u64)
                * gpu::Q4_K_BLOCK_BYTES as u64,
            shape: (rows, cols, 0, 0),
            scale_offset: 0,
            scale_size: 0,
            bias_offset: 0,
            bias_size: 0,
        }
    }

    #[test]
    fn q4_k_embedding_validation_bounds_shape_size_and_selected_row() {
        let valid = q4_k_embedding(256, 256);
        validate_q4_k_embedding(&valid, &valid.name, 255, 256).unwrap();

        let mut wrong_shape = valid.clone();
        wrong_shape.shape.1 = 512;
        assert!(validate_q4_k_embedding(&wrong_shape, &wrong_shape.name, 0, 256).is_err());

        let mut short = valid.clone();
        short.size_bytes -= gpu::Q4_K_BLOCK_BYTES as u64;
        assert!(validate_q4_k_embedding(&short, &short.name, 255, 256).is_err());

        assert!(validate_q4_k_embedding(&valid, &valid.name, 256, 256).is_err());
    }

    fn iq4_xs_embedding(rows: u32, cols: u32) -> model_io::ResidentIndexEntry {
        model_io::ResidentIndexEntry {
            name: "token_embd.weight".to_string(),
            dtype: DTYPE_GGUF_IQ4_XS,
            file_offset: 4096,
            size_bytes: u64::from(rows)
                * (u64::from(cols) / gpu::IQ4_XS_BLOCK_ELEMS as u64)
                * gpu::IQ4_XS_BLOCK_BYTES as u64,
            shape: (rows, cols, 0, 0),
            scale_offset: 0,
            scale_size: 0,
            bias_offset: 0,
            bias_size: 0,
        }
    }

    #[test]
    fn iq4_xs_embedding_validation_bounds_shape_size_and_selected_row() {
        let valid = iq4_xs_embedding(7, 512);
        validate_iq4_xs_embedding(&valid, &valid.name, 6, 512).unwrap();

        let mut wrong_shape = valid.clone();
        wrong_shape.shape.1 = 256;
        assert!(validate_iq4_xs_embedding(&wrong_shape, &wrong_shape.name, 0, 512).is_err());

        let mut short = valid.clone();
        short.size_bytes -= gpu::IQ4_XS_BLOCK_BYTES as u64;
        assert!(validate_iq4_xs_embedding(&short, &short.name, 6, 512).is_err());

        assert!(validate_iq4_xs_embedding(&valid, &valid.name, 7, 512).is_err());
    }

    #[test]
    fn router_topk_gemma4_handles_empty_or_zero_k() {
        let (sel, w) = router_topk_gemma4(&[], 4, &[]);
        assert!(sel.is_empty() && w.is_empty());

        let (sel, w) = router_topk_gemma4(&[1.0, 2.0, 3.0], 0, &[1.0, 1.0, 1.0]);
        assert!(sel.is_empty() && w.is_empty());
    }
}
