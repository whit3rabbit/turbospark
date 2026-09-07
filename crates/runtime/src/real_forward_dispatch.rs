//! Generic dynamic Metal kernel dispatchers for GEMV, Embedding lookups,
//! and MoE phased execution across affine and GGUF block types.

use model_io::ResidentIndex;

use crate::real_forward_layout::{
    DTYPE_GGUF_Q4_K, DTYPE_GGUF_Q5_K, DTYPE_GGUF_Q6_K, DTYPE_GGUF_Q8_0, DTYPE_INT1_AFFINE,
    DTYPE_INT2_AFFINE,
};
use crate::real_forward_types::RealForwardError;
use crate::real_forward_utils::{affine_group_size, entry, resident_matrix};

pub(crate) use crate::real_forward_dispatch_moe::{encode_moe_phase1_any, encode_moe_phase2_any};

/// Dispatches the embedding lookup matching the table's dtype tag: 4 =
/// INT4-affine, 15 = 1-bit affine, plus the GGUF block types a real file puts
/// an embedding table in.
///
/// Shared by both real flows rather than written at each one, because it is a
/// property of the tensor and not of the family: Qwen's Q4_K_M keeps
/// `token_embd.weight` at Q4_K while Gemma's published GGUF is Q8_0
/// throughout, and either family could meet either table.
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
    embed_scale: f32,
) -> Result<(), RealForwardError> {
    let e = entry(index, name)?;
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
            gpu::encode_embed_lookup_q4_k(context, pass, table, out, token, hidden, embed_scale)
                .map_err(RealForwardError::Gpu)
        }
        // ROADMAP Phase S's candidate puts `token_embd` in Q6_K and ties the
        // head to it, which is what made this kernel worth writing.
        DTYPE_GGUF_Q6_K => {
            gpu::encode_embed_lookup_q6_k(context, pass, table, out, token, hidden, embed_scale)
                .map_err(RealForwardError::Gpu)
        }
        // The 1-bit table (ROADMAP's 1-bit entry). It exists because the real
        // checkpoint quantizes `embed_tokens` at one bit like everything else,
        // which was read off its safetensors header rather than assumed.
        //
        // The ROW COUNT is derived rather than passed: this function's callers
        // know the hidden size and the token id, never the vocabulary, and the
        // group size cannot be read off the companion planes without it. One
        // bit per element makes it exact.
        DTYPE_INT1_AFFINE | DTYPE_INT2_AFFINE => {
            let bits = if e.dtype == DTYPE_INT1_AFFINE { 1 } else { 2 };
            let elements_per_byte = 8 / bits;
            let d = hidden as usize;
            if d == 0 || (e.size_bytes as usize * elements_per_byte) % d != 0 {
                return Err(RealForwardError::Unsupported(format!(
                    "embedding table {name}: {} {bits}-bit bytes is not a whole number of \
                     {d}-element rows",
                    e.size_bytes
                )));
            }
            let rows = e.size_bytes as usize * elements_per_byte / d;
            let group_size = affine_group_size(e, name, rows, d, bits)?;
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
        4 => {
            // Resolved inside this arm on purpose: a GGUF entry carries no
            // companions, so its `scale_offset` is 0 and subtracting the
            // index size underflows.
            let scales = (weights.buffer(), weights.gpu_offset(e.scale_offset - base));
            let biases = (weights.buffer(), weights.gpu_offset(e.bias_offset - base));
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
            if e.size_bytes as usize != rows * cols {
                return Err(RealForwardError::Unsupported(format!(
                    "tensor {name}: INT8 packed size {} does not match {rows}x{cols}",
                    e.size_bytes
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
/// **EVERY DTYPE BUT INT4-AFFINE IS REFUSED BY NAME, NEVER LOOPED.** A
/// fallback of `batch` sequential GEMVs is numerically identical, so it
/// would pass every parity and losslessness test there is -- while making a
/// "batched" verify measure the SEQUENTIAL engine and report its cost as the
/// batched one. That is AGENTS.md Gotcha 35's failure (a measurement tool
/// must not inherit a silent default) one layer down, and it would corrupt
/// the only number step 4 exists to produce. The refusal is not a temporary
/// gap either: the 1-bit and 2-bit checkpoints of this same architecture and
/// every GGUF block type have no batched kernel at all.
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
            gpu::encode_dequant_int4_gemm_resident(context, pass, &w, x, y, batch)
                .map_err(RealForwardError::Gpu)
        }
        other => Err(RealForwardError::Unsupported(format!(
            "tensor {name}: dtype {other} has no BATCHED kernel; only INT4-affine (4) \
             does. Refused rather than looped: a sequential fallback here is \
             numerically identical and would silently measure the unbatched engine"
        ))),
    }
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

    #[test]
    fn router_topk_gemma4_handles_empty_or_zero_k() {
        let (sel, w) = router_topk_gemma4(&[], 4, &[]);
        assert!(sel.is_empty() && w.is_empty());

        let (sel, w) = router_topk_gemma4(&[1.0, 2.0, 3.0], 0, &[1.0, 1.0, 1.0]);
        assert!(sel.is_empty() && w.is_empty());
    }
}
