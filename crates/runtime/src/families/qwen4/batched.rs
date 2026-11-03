//! The M-row resident GEMMs of a `qwen4_exp` chunked-prefill layer
//! (`TURBOSPARK_BATCHED_GEMV`, `docs/BATCHED_PREFILL.md` step 6).
//!
//! A layer's projections depend only on that layer's input rows, so the
//! driver runs every token's `attn_hc` first, then each projection ONCE
//! over all M rows (`encode_gemm_any`, bit-exact against its GEMV at every
//! batch width), then the per-token middle (conv, delta rule, norms, RoPE,
//! KV write, QSA selection, attention) through the same `attn.rs` blocks
//! decode runs, handed this token's rows through [`PreProjected`], then the
//! output projection once over M rows. The recurrent state still advances
//! one token at a time in increasing order, which is what keeps GDN and
//! the QSA indexer equal to sequential decode.
//!
//! The hyper-connection GEMVs, PLE and the INT8 router stay per token.

use model_io::{ArchConfig, ResidentIndex};

use crate::families::qwen4::attn::{encode_full_attention_block, encode_linear_block};
use crate::families::qwen4::hc::encode_hyper_connection;
use crate::families::qwen4::layer_tensor;
use crate::families::qwen4::moe;
use crate::families::qwen4::state::RealQwen4State;
use crate::real_forward_dispatch::encode_gemm_any;
use crate::real_forward_types::{DecodeScratch, RealForwardError, MAX_PREFILL_BATCH};

/// `[MAX_PREFILL_BATCH, width]` FP16 rows, token-major, one buffer per
/// projection a layer runs as a GEMM. Allocated on the first batched layer.
pub(crate) struct BatchedGemvScratch {
    /// `attn_hc`'s output per token, every GEMM's input.
    pub(crate) normed: gpu::MetalBuffer,
    pub(crate) gdn_qkv_raw: gpu::MetalBuffer,
    pub(crate) gdn_z: gpu::MetalBuffer,
    pub(crate) gdn_a: gpu::MetalBuffer,
    pub(crate) gdn_b: gpu::MetalBuffer,
    /// The gated norm's output per token, `out_proj`'s input.
    pub(crate) gdn_out: gpu::MetalBuffer,
    pub(crate) idx_qk: gpu::MetalBuffer,
    pub(crate) q_packed: gpu::MetalBuffer,
    pub(crate) k: gpu::MetalBuffer,
    pub(crate) v: gpu::MetalBuffer,
    /// The gated attention output per token, `o_proj`'s input.
    pub(crate) attn_out: gpu::MetalBuffer,
    /// Either output projection's result per token.
    pub(crate) o: gpu::MetalBuffer,
    pub(crate) widths: Widths,
}

/// Row widths in halfs, so a row offset is `t * width * 2`.
#[derive(Clone, Copy)]
pub(crate) struct Widths {
    pub(crate) hidden: usize,
    pub(crate) qkv: usize,
    pub(crate) value: usize,
    pub(crate) v_heads: usize,
    pub(crate) idx_qk: usize,
    pub(crate) q_dim: usize,
    pub(crate) kv_dim: usize,
}

impl BatchedGemvScratch {
    pub(crate) fn new(
        context: &gpu::MetalContext,
        arch: &ArchConfig,
        qwen4: &RealQwen4State,
    ) -> Self {
        let shape = qwen4.shape;
        let widths = Widths {
            hidden: arch.hidden_size as usize,
            qkv: shape.qkv_dim() as usize,
            value: shape.value_dim() as usize,
            v_heads: shape.num_v_heads as usize,
            idx_qk: ((qwen4.idx_heads + 1) * qwen4.idx_head_dim) as usize,
            q_dim: (arch.num_heads * arch.full_head_dim) as usize,
            kv_dim: (arch.num_full_kv_heads * arch.full_head_dim) as usize,
        };
        let rows = |w: usize| context.new_output_buffer((MAX_PREFILL_BATCH * w.max(1) * 2) as u64);
        Self {
            normed: rows(widths.hidden),
            gdn_qkv_raw: rows(widths.qkv),
            gdn_z: rows(widths.value),
            gdn_a: rows(widths.v_heads),
            gdn_b: rows(widths.v_heads),
            gdn_out: rows(widths.value),
            idx_qk: rows(widths.idx_qk),
            q_packed: rows(2 * widths.q_dim),
            k: rows(widths.kv_dim),
            v: rows(widths.kv_dim),
            attn_out: rows(widths.q_dim),
            o: rows(widths.hidden),
            widths,
        }
    }

    /// Byte offset of row `t` in a buffer `width` halfs wide.
    pub(crate) fn row(t: usize, width: usize) -> u64 {
        (t * width * 2) as u64
    }
}

/// One token's view of the rows the driver projected: the attention blocks
/// read their projections from row `t` instead of running GEMVs, and write
/// their pre-output-projection activation to row `t` instead of projecting.
pub(crate) struct PreProjected<'a> {
    pub(crate) rows: &'a BatchedGemvScratch,
    pub(crate) t: usize,
}

/// A GDN layer's four input projections over `m` rows of `rows.normed`.
pub(crate) fn encode_linear_in_proj_batched(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    rows: &BatchedGemvScratch,
    layer: usize,
    m: usize,
) -> Result<(), RealForwardError> {
    let w = rows.widths;
    for (suffix, width, out) in [
        ("in_proj_qkv.weight", w.qkv, &rows.gdn_qkv_raw),
        ("in_proj_z.weight", w.value, &rows.gdn_z),
        ("in_proj_a.weight", w.v_heads, &rows.gdn_a),
        ("in_proj_b.weight", w.v_heads, &rows.gdn_b),
    ] {
        encode_gemm_any(
            context,
            pass,
            weights,
            index,
            &layer_tensor(layer, &format!("linear_attn.{suffix}")),
            width,
            w.hidden,
            (&rows.normed, 0),
            (out, 0),
            m,
        )?;
    }
    Ok(())
}

/// A QSA layer's indexer, query, key and value projections over `m` rows.
pub(crate) fn encode_attention_in_proj_batched(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    rows: &BatchedGemvScratch,
    layer: usize,
    m: usize,
) -> Result<(), RealForwardError> {
    let w = rows.widths;
    for (suffix, width, out) in [
        ("indexer.index_qk_proj.weight", w.idx_qk, &rows.idx_qk),
        ("q_proj.weight", 2 * w.q_dim, &rows.q_packed),
        ("k_proj.weight", w.kv_dim, &rows.k),
        ("v_proj.weight", w.kv_dim, &rows.v),
    ] {
        encode_gemm_any(
            context,
            pass,
            weights,
            index,
            &layer_tensor(layer, &format!("self_attn.{suffix}")),
            width,
            w.hidden,
            (&rows.normed, 0),
            (out, 0),
            m,
        )?;
    }
    Ok(())
}

/// The layer's output projection over `m` rows into `rows.o`: `out_proj`
/// from the gated norm rows on a GDN layer, `o_proj` from the gated
/// attention rows on a QSA layer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_out_proj_batched(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    rows: &BatchedGemvScratch,
    layer: usize,
    linear: bool,
    m: usize,
) -> Result<(), RealForwardError> {
    let w = rows.widths;
    let (name, cols, input) = if linear {
        ("linear_attn.out_proj.weight", w.value, &rows.gdn_out)
    } else {
        ("self_attn.o_proj.weight", w.q_dim, &rows.attn_out)
    };
    encode_gemm_any(
        context,
        pass,
        weights,
        index,
        &layer_tensor(layer, name),
        w.hidden,
        cols,
        (input, 0),
        (&rows.o, 0),
        m,
    )
}

/// One layer's attention-and-router half for all `m` tokens with the
/// projections batched, onto the caller's `cb1` pass. Same dataflow as the
/// driver's per-token loop, reordered only across independent work:
/// every `attn_hc`, the input GEMMs, the per-token middle in increasing
/// position, the output GEMM, then per token the inject, `mlp_hc` and the
/// router. `attn_hc` writes each token's attention inject gate to its
/// `hc_inject` row, and that row is read by the inject before `mlp_hc`
/// overwrites it, as in the per-token loop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_layer_cb1_batched(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    arch: &ArchConfig,
    qwen4: &mut RealQwen4State,
    scratch: &DecodeScratch,
    kv: &gpu::KvCacheManager,
    layer: usize,
    start_position: usize,
    m: usize,
) -> Result<(), RealForwardError> {
    if qwen4.batched_gemv.is_none() {
        qwen4.batched_gemv = Some(BatchedGemvScratch::new(context, arch, qwen4));
    }
    // Held outside `qwen4` for the layer so the blocks can borrow it while
    // taking `qwen4` mutably; put back on every path.
    let rows = qwen4.batched_gemv.take().expect("allocated above");
    let result = encode_layer_cb1_rows(
        context,
        pass,
        weights,
        index,
        arch,
        qwen4,
        scratch,
        kv,
        layer,
        start_position,
        m,
        &rows,
    );
    qwen4.batched_gemv = Some(rows);
    result
}

#[allow(clippy::too_many_arguments)]
fn encode_layer_cb1_rows(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    arch: &ArchConfig,
    qwen4: &mut RealQwen4State,
    scratch: &DecodeScratch,
    kv: &gpu::KvCacheManager,
    layer: usize,
    start_position: usize,
    m: usize,
    rows: &BatchedGemvScratch,
) -> Result<(), RealForwardError> {
    let w = rows.widths;
    let hidden = w.hidden;
    let hc_count = qwen4.hc_count;
    let wide_dim = hidden * hc_count;
    let num_experts = arch.num_experts as usize;
    let linear = arch.layer_is_linear(layer);
    let wide_buf = qwen4.wide_x.clone();

    for t in 0..m {
        encode_hyper_connection(
            context,
            pass,
            weights,
            index,
            qwen4,
            (&wide_buf, (t * wide_dim * 2) as u64),
            &layer_tensor(layer, "attn_hyper_connection"),
            hidden,
            (&rows.normed, BatchedGemvScratch::row(t, hidden)),
            true,
            (t * hc_count * 2) as u64,
        )?;
    }

    if linear {
        encode_linear_in_proj_batched(context, pass, weights, index, rows, layer, m)?;
    } else {
        encode_attention_in_proj_batched(context, pass, weights, index, rows, layer, m)?;
    }
    for t in 0..m {
        let pre = PreProjected { rows, t };
        let normed = (&rows.normed, BatchedGemvScratch::row(t, hidden));
        if linear {
            encode_linear_block(
                context,
                pass,
                weights,
                index,
                arch,
                qwen4,
                normed,
                scratch,
                layer,
                start_position + t,
                false,
                Some(&pre),
            )?;
        } else {
            encode_full_attention_block(
                context,
                pass,
                weights,
                index,
                arch,
                qwen4,
                normed,
                scratch,
                kv,
                layer,
                start_position + t,
                // Text only: the chunked driver refuses an image prompt
                // before this seam.
                crate::vision::RopePosition::Sequential,
                Some(&pre),
            )?;
        }
    }
    encode_out_proj_batched(context, pass, weights, index, rows, layer, linear, m)?;

    for t in 0..m {
        let x_off = (t * wide_dim * 2) as u64;
        let hc_row = (t * hc_count * 2) as u64;
        gpu::encode_hc_inject_add(
            context,
            pass,
            (&wide_buf, x_off),
            (&rows.o, BatchedGemvScratch::row(t, hidden)),
            (&qwen4.hc_inject, hc_row),
            hc_count as u32,
            hidden as u32,
        )
        .map_err(RealForwardError::Gpu)?;
        let moe_x_off = (t * hidden * 2) as u64;
        encode_hyper_connection(
            context,
            pass,
            weights,
            index,
            qwen4,
            (&wide_buf, x_off),
            &layer_tensor(layer, "mlp_hyper_connection"),
            hidden,
            (&qwen4.moe_x, moe_x_off),
            true,
            hc_row,
        )?;
        moe::encode_moe_router(
            context,
            pass,
            weights,
            index,
            qwen4,
            (&qwen4.moe_x, moe_x_off),
            layer,
            hidden,
            num_experts,
            (t * num_experts * 4) as u64,
        )?;
    }
    Ok(())
}
