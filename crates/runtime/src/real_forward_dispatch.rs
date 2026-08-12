//! Generic dynamic Metal kernel dispatchers for GEMV, Embedding lookups,
//! and MoE phased execution across affine and GGUF block types.

use model_io::ResidentIndex;

use crate::real_forward_layout::{
    RoutedBlobLayout, DTYPE_GGUF_Q4_K, DTYPE_GGUF_Q5_K, DTYPE_GGUF_Q6_K, DTYPE_GGUF_Q8_0,
};
use crate::real_forward_types::RealForwardError;
use crate::real_forward_utils::{entry, resident_matrix};

/// The routed-expert decode pair, dispatched for whichever blob layout this
/// install carries: the vendored INT4-affine `moe.metal` kernels or one of
/// the port-local GGUF pairs in `moe_gguf.metal` (ROADMAP Phase G Stage 2).
///
/// A pair of forwarders rather than a branch at each call site, so a layout
/// cannot disagree between two of them and read one blob two ways.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_moe_phase1_any(
    layout: RoutedBlobLayout,
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    routed: &gpu::RoutedBlobsBuffer,
    offsets: &gpu::MoeExpertOffsets,
    x: (&gpu::MetalBuffer, u64),
    acts: (&gpu::MetalBuffer, u64),
    d_dim: u32,
    f_dim: u32,
    top_k: u32,
    use_silu: bool,
) -> Result<(), gpu::GpuError> {
    match layout {
        RoutedBlobLayout::GgufQ8_0 => gpu::encode_moe_phase1_q8_0(
            context, pass, routed, offsets, x, acts, d_dim, f_dim, top_k, use_silu,
        ),
        RoutedBlobLayout::GgufQ4K => gpu::encode_moe_phase1_q4_k(
            context, pass, routed, offsets, x, acts, d_dim, f_dim, top_k, use_silu,
        ),
        RoutedBlobLayout::GgufIq3Xxs => gpu::encode_moe_phase1_iq3_xxs(
            context, pass, routed, offsets, x, acts, d_dim, f_dim, top_k, use_silu,
        ),
        RoutedBlobLayout::GgufIq4Xs => gpu::encode_moe_phase1_iq4_xs(
            context, pass, routed, offsets, x, acts, d_dim, f_dim, top_k, use_silu,
        ),
        // MXFP4 IS `gpt-oss` AND NOTHING ELSE, so the activation constants
        // come from that family here rather than being threaded through
        // every caller of this forwarder. `has_bias` is DERIVED rather than
        // assumed, though: a bias offset of 0 is what
        // `moe_offsets_from_layout` writes when the blob has no bias plane,
        // and 0 cannot be a real one because `gate_w` occupies it. If a
        // second MXFP4 checkpoint ever appears with a different activation,
        // this is the line that has to become a parameter.
        RoutedBlobLayout::GgufMxfp4 => gpu::encode_moe_phase1_mxfp4(
            context,
            pass,
            routed,
            offsets,
            x,
            acts,
            d_dim,
            f_dim,
            top_k,
            use_silu,
            gpu::Mxfp4Activation {
                has_bias: offsets.gate_b != 0,
                ..gpu::Mxfp4Activation::GPT_OSS
            },
        ),
        RoutedBlobLayout::Affine => gpu::encode_moe_phase1(
            context, pass, routed, offsets, x, acts, d_dim, f_dim, top_k, use_silu,
        ),
        // No real file puts IQ4_NL in gate/up, so there is no such kernel.
        // Reaching here means an install this port installed but cannot run;
        // `open()`'s dtype gate lets it through because the TYPE is
        // executable, just not in this position. Same shape as Q6_K.
        other => Err(gpu::GpuError::FunctionNotFound(format!(
            "routed phase 1 (gate/up) for {other:?}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_moe_phase2_any(
    layout: RoutedBlobLayout,
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    routed: &gpu::RoutedBlobsBuffer,
    offsets: &gpu::MoeExpertOffsets,
    acts: (&gpu::MetalBuffer, u64),
    routing_w: (&gpu::MetalBuffer, u64),
    residual: (&gpu::MetalBuffer, u64),
    y: (&gpu::MetalBuffer, u64),
    d_dim: u32,
    f_dim: u32,
    use_silu: bool,
) -> Result<(), gpu::GpuError> {
    match layout {
        RoutedBlobLayout::GgufQ8_0 => gpu::encode_moe_phase2_q8_0(
            context, pass, routed, offsets, acts, routing_w, residual, y, d_dim, f_dim, use_silu,
        ),
        RoutedBlobLayout::GgufQ4K => gpu::encode_moe_phase2_q4_k(
            context, pass, routed, offsets, acts, routing_w, residual, y, d_dim, f_dim, use_silu,
        ),
        RoutedBlobLayout::GgufQ6K => gpu::encode_moe_phase2_q6_k(
            context, pass, routed, offsets, acts, routing_w, residual, y, d_dim, f_dim, use_silu,
        ),
        RoutedBlobLayout::GgufIq4Nl => gpu::encode_moe_phase2_iq4_nl(
            context, pass, routed, offsets, acts, routing_w, residual, y, d_dim, f_dim, use_silu,
        ),
        RoutedBlobLayout::GgufMxfp4 => gpu::encode_moe_phase2_mxfp4(
            context,
            pass,
            routed,
            offsets,
            acts,
            routing_w,
            residual,
            y,
            d_dim,
            f_dim,
            use_silu,
            offsets.down_b != 0,
        ),
        RoutedBlobLayout::Affine => gpu::encode_moe_phase2(
            context, pass, routed, offsets, acts, routing_w, residual, y, d_dim, f_dim, use_silu,
        ),
        // No real file puts IQ3_XXS or IQ4_XS in `down`; see the phase-1
        // sibling's note.
        other => Err(gpu::GpuError::FunctionNotFound(format!(
            "routed phase 2 (down) for {other:?}"
        ))),
    }
}

/// Dispatches the embedding lookup matching the table's dtype tag: 4 =
/// INT4-affine, plus the two GGUF block types a real file puts an embedding
/// table in.
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
/// 5 = INT8-affine (the resident writer's tags), and encodes the matching
/// offset-bound GEMV.
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
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    let selected: Vec<usize> = order.into_iter().take(k).collect();
    let max = logits[selected[0]];
    let exps: Vec<f32> = selected.iter().map(|&i| (logits[i] - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let weights = selected
        .iter()
        .zip(exps.iter())
        .map(|(&i, e)| e / sum * per_expert_scale[i])
        .collect();
    (selected, weights)
}
