//! The batched arms of the dense llama-flow prefill: every projection runs
//! once over the micro-batch's M rows instead of once per token, so the
//! weights are read once per M tokens. Per token they are the whole cost
//! (`docs/SPEEDUP_PLAYBOOK.md`, "Qwen2 prefill on the llama flow").
//!
//! There are two, and they differ in what they promise.
//!
//! - **Exact** (16 rows, [`MAX_PREFILL_BATCH`]): bit-identical to the
//!   per-token loop, and it has to stay that way. The M-row kernels are the
//!   ones speculative verify already holds equal to their GEMVs, the
//!   multi-row norms are per-row reductions, and attention is the one-row
//!   decode kernel, row by row.
//! - **Matrix** (64 rows): Steel INT4 GEMMs and the multi-row GQA attention
//!   partial. Both round and reduce in another order, so it is held to a
//!   distribution bound (`qwen2_prefill_arm_probe`), never to equality.
//!   `TURBOSPARK_PREFILL_STEEL=0` falls back to the exact arm.
//!
//! Biases, q/k norm, RoPE, steering and capture are the per-token arm's own
//! code in both.

use model_io::{ArchConfig, ModelFamily, ResidentIndex};

use super::{attn, layer_tensor, RealLlamaState};
use crate::kv_write::{encode_attention_any, kv_write_target, KvHalf};
use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::{encode_gemm_rows, int4_mma8_mode, Int4Mma8Mode};
use crate::real_forward_types::{
    DecodeScratch, RealForwardError, MAX_PREFILL_BATCH, QWEN_STEEL_PREFILL_BATCH,
};
use crate::real_forward_utils::{entry, norm_view};
use crate::resid_capture::encode_resid_capture;
use crate::steering::encode_steering;
use crate::vision::RopePosition;

/// The projections one dense layer multiplies, all of which must have a
/// batched kernel.
const PROJECTIONS: [&str; 7] = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

/// Families that take this arm with the knob unset. Another family needs its
/// own real-install bit-equality, memory and timing first.
pub(super) fn batched_by_default(family: ModelFamily) -> bool {
    matches!(family, ModelFamily::Qwen2Dense)
}

/// Families that take the matrix arm with the knobs unset. It moves logits,
/// so each one needs its own distribution, quality, memory and timing gates.
fn matrix_by_default(family: ModelFamily) -> bool {
    matches!(family, ModelFamily::Qwen2Dense)
}

/// Transient M-row buffers; the residual rows already live in
/// `DecodeScratch::x`.
pub(super) struct DenseBatchScratch {
    /// Whether the GEMMs may run Steel: the matrix arm.
    matrix: bool,
    /// The multi-row attention partials, matrix arm only.
    rows_attention: Option<gpu::GqaRowsScratch>,
    post_normed: gpu::MetalBuffer,
    normed: gpu::MetalBuffer,
    q: gpu::MetalBuffer,
    attn_out: gpu::MetalBuffer,
    o: gpu::MetalBuffer,
    ffn_gate: gpu::MetalBuffer,
    ffn_up: gpu::MetalBuffer,
    ffn_act: gpu::MetalBuffer,
    ffn_out: gpu::MetalBuffer,
}

impl RealForwardRunner {
    /// Why the batched arm cannot serve this install, or `None` if it can.
    pub(super) fn llama_dense_batched_refusal(&self) -> Option<String> {
        let arch = &self.arch;
        if arch.family == ModelFamily::Llama {
            return Some("dense Llama keeps its own batch-attention route".into());
        }
        if int4_mma8_mode() == Int4Mma8Mode::Verify {
            return Some("TURBOSPARK_INT4_MMA8=verify makes M-row kernels drift".into());
        }
        for layer in 0..arch.num_layers as usize {
            if self.kv.layer_quant(layer).is_some() || self.kv.ring_capacity(layer) != 0 {
                return Some(format!(
                    "layer {layer} does not hold a linear FP16 KV cache"
                ));
            }
            for suffix in PROJECTIONS {
                let name = layer_tensor(layer, suffix);
                let Ok(e) = entry(&self.index, &name) else {
                    return Some(format!("tensor {name} is not resident"));
                };
                // The dtypes `encode_gemm_any` serves: BF16, Q8_0, Q4_K,
                // Q6_K, INT4, INT8, Q5_K and 1/2-bit affine.
                if !matches!(e.dtype, 1 | 4 | 5 | 6 | 7 | 8 | 13 | 15 | 16) {
                    return Some(format!(
                        "tensor {name} (dtype {}) has no M-row kernel",
                        e.dtype
                    ));
                }
            }
        }
        None
    }

    /// Whether this install runs the matrix arm: a family that passed its
    /// gates, with every projection INT4 (the width Steel serves; other
    /// types would take the exact kernels and only the attention would move,
    /// which nothing has gated).
    fn llama_dense_takes_matrix(&self) -> bool {
        self.prefill_steel
            && matrix_by_default(self.arch.family)
            && (0..self.arch.num_layers as usize).all(|layer| {
                PROJECTIONS.iter().all(|suffix| {
                    entry(&self.index, &layer_tensor(layer, suffix)).is_ok_and(|e| e.dtype == 4)
                })
            })
    }

    /// Allocates the scratch for whichever batched arm applies and returns
    /// its micro-batch width.
    pub(super) fn ensure_llama_prefill_batch(&mut self) -> Result<usize, RealForwardError> {
        let matrix = self.llama_dense_takes_matrix();
        let width = if matrix {
            QWEN_STEEL_PREFILL_BATCH
        } else {
            MAX_PREFILL_BATCH
        };
        let rows_attention = matrix && self.prefill_rows_attention;
        let arch = &self.arch;
        let hidden = arch.hidden_size as usize;
        let inter = arch.intermediate_size as usize;
        let q_dim = arch.num_heads as usize * arch.full_head_dim as usize;
        let llama = self
            .real_llama
            .as_mut()
            .ok_or_else(|| RealForwardError::Unsupported("not a llama-flow install".into()))?;
        let (heads, kv_heads, head_dim) = (
            arch.num_heads as u32,
            arch.num_full_kv_heads as u32,
            arch.full_head_dim as u32,
        );
        let rows_attention =
            rows_attention && gpu::attention_decode_gqa_supported(head_dim, heads, kv_heads);
        let current = llama
            .prefill_batch
            .as_ref()
            .map(|b| (b.matrix, b.rows_attention.is_some()));
        if current != Some((matrix, rows_attention)) {
            let context = &self.context;
            let rows = |row: usize| context.new_output_buffer((row * width * 2) as u64);
            llama.prefill_batch = Some(DenseBatchScratch {
                matrix,
                rows_attention: rows_attention.then(|| {
                    gpu::GqaRowsScratch::new(
                        context,
                        width as u32,
                        heads,
                        head_dim,
                        gpu::GQA_ROWS_SCRATCH_CHUNKS,
                    )
                }),
                post_normed: rows(hidden),
                normed: rows(hidden),
                q: rows(q_dim),
                attn_out: rows(q_dim),
                o: rows(hidden),
                ffn_gate: rows(inter),
                ffn_up: rows(inter),
                ffn_act: rows(inter),
                ffn_out: rows(hidden),
            });
        }
        Ok(width)
    }
}

/// One layer over one micro-batch.
pub(super) struct LayerBatch<'a> {
    pub(super) weights: &'a gpu::ResidentGpuWeights,
    pub(super) index: &'a ResidentIndex,
    pub(super) arch: &'a ArchConfig,
    pub(super) llama: &'a RealLlamaState,
    pub(super) scratch: &'a DecodeScratch,
    pub(super) batch: &'a DenseBatchScratch,
    pub(super) kv: &'a gpu::KvCacheManager,
    pub(super) steering: Option<&'a crate::steering::SteeringState>,
    pub(super) resid_capture: Option<&'a crate::resid_capture::ResidCapture>,
    pub(super) layer: usize,
    pub(super) start_position: usize,
    /// One per row; its length is M.
    pub(super) rope_positions: &'a [RopePosition],
}

pub(super) fn encode_layer_batched(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    l: &LayerBatch<'_>,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let LayerBatch {
        weights,
        index,
        arch,
        llama,
        scratch,
        batch,
        kv,
        layer,
        start_position,
        ..
    } = *l;
    let m = l.rope_positions.len();
    let hidden = arch.hidden_size as usize;
    let inter = arch.intermediate_size as usize;
    let num_heads = arch.num_heads as u32;
    let num_kv = arch.num_full_kv_heads as u32;
    let head_dim = arch.full_head_dim as u32;
    let q_dim = (num_heads * head_dim) as usize;
    let kv_dim = (num_kv * head_dim) as usize;
    let row = |width: usize, t: usize| (t * width * 2) as u64;
    let name = |suffix: &str| layer_tensor(layer, suffix);
    let gemm = |context: &mut gpu::MetalContext,
                suffix: &str,
                rows: usize,
                cols: usize,
                x: (&gpu::MetalBuffer, u64),
                y: (&gpu::MetalBuffer, u64)| {
        let tensor = name(suffix);
        encode_gemm_rows(
            context,
            pass,
            weights,
            index,
            &tensor,
            rows,
            cols,
            x,
            y,
            m,
            batch.matrix,
        )
    };
    let input_norm = norm_view(weights, index, &name("input_layernorm.weight"), hidden)?;
    let post_attn_norm = norm_view(
        weights,
        index,
        &name("post_attention_layernorm.weight"),
        hidden,
    )?;

    // The per-head norm kernel with one "head" per row and the row as its
    // width is the per-row norm (`rms_norm_rows_as_heads`).
    gpu::encode_rms_norm_bf16w_perhead(
        context,
        pass,
        (&scratch.x, 0),
        input_norm,
        (&batch.normed, 0),
        m as u32,
        hidden as u32,
        llama.rms_eps,
    )
    .map_err(gpu_err)?;

    // K and V go straight into M consecutive slots of the linear cache.
    let (k_buf, k_off) = kv_write_target(kv, scratch, KvHalf::K, layer, start_position);
    let (v_buf, v_off) = kv_write_target(kv, scratch, KvHalf::V, layer, start_position);
    let normed = (&batch.normed, 0);
    gemm(
        context,
        "self_attn.q_proj.weight",
        q_dim,
        hidden,
        normed,
        (&batch.q, 0),
    )?;
    gemm(
        context,
        "self_attn.k_proj.weight",
        kv_dim,
        hidden,
        normed,
        (k_buf, k_off),
    )?;
    gemm(
        context,
        "self_attn.v_proj.weight",
        kv_dim,
        hidden,
        normed,
        (v_buf, v_off),
    )?;

    // Every K/V row is finished before the first row attends. A row's span
    // ends at its own position, so the later rows are invisible to it.
    for (t, &rope_position) in l.rope_positions.iter().enumerate() {
        let position = start_position + t;
        let (k_buf, k_off) = kv_write_target(kv, scratch, KvHalf::K, layer, position);
        let (v_buf, v_off) = kv_write_target(kv, scratch, KvHalf::V, layer, position);
        attn::encode_attention_row_post(
            context,
            pass,
            weights,
            index,
            arch,
            llama,
            layer,
            position,
            rope_position,
            (&batch.q, row(q_dim, t)),
            (k_buf, k_off),
            (v_buf, v_off),
        )?;
    }
    // The multi-row partial serves a whole micro-batch in one dispatch. One
    // row keeps the decode kernel (the multi-row one is no faster there), as
    // does `TURBOSPARK_ATTN_DECODE_KERNEL=barrier`.
    let rows_attention = batch
        .rows_attention
        .as_ref()
        .filter(|_| m > 1 && scratch.attn.decode_kernel() == gpu::DecodeAttentionKernel::GqaShared);
    if let Some(rows) = rows_attention {
        foundation::engaged!("prefill_rows_attn");
        let (k_all, _) = kv.k_slot(layer, 0);
        let (v_all, _) = kv.v_slot(layer, 0);
        gpu::encode_attention_decode_gqa_rows(
            context,
            pass,
            (&batch.q, 0),
            k_all,
            v_all,
            rows,
            (&batch.attn_out, 0),
            head_dim,
            num_heads,
            num_kv,
            start_position as u32,
            m as u32,
            0,
            0,
            arch.attention_scale as f32,
        )
        .map_err(gpu_err)?;
    } else {
        for t in 0..m {
            let position = start_position + t;
            encode_attention_any(
                context,
                pass,
                (&batch.q, row(q_dim, t)),
                kv,
                scratch,
                (&batch.attn_out, row(q_dim, t)),
                layer,
                position,
                head_dim,
                num_heads,
                num_kv,
                (position + 1) as u32,
                0,
                0,
                arch.attention_scale as f32,
                None,
            )?;
        }
    }
    gemm(
        context,
        "self_attn.o_proj.weight",
        hidden,
        q_dim,
        (&batch.attn_out, 0),
        (&batch.o, 0),
    )?;
    gpu::encode_residual_add(
        context,
        pass,
        (&scratch.x, 0),
        (&batch.o, 0),
        (m * hidden) as u32,
    )
    .map_err(gpu_err)?;
    gpu::encode_rms_norm_bf16w_perhead(
        context,
        pass,
        (&scratch.x, 0),
        post_attn_norm,
        (&batch.post_normed, 0),
        m as u32,
        hidden as u32,
        llama.rms_eps,
    )
    .map_err(gpu_err)?;

    let ffn_in = (&batch.post_normed, 0);
    gemm(
        context,
        "mlp.gate_proj.weight",
        inter,
        hidden,
        ffn_in,
        (&batch.ffn_gate, 0),
    )?;
    gemm(
        context,
        "mlp.up_proj.weight",
        inter,
        hidden,
        ffn_in,
        (&batch.ffn_up, 0),
    )?;
    let act = if arch.hidden_activation.contains("silu") {
        gpu::encode_silu_mul
    } else {
        gpu::encode_gelu_mul
    };
    act(
        context,
        pass,
        (&batch.ffn_gate, 0),
        (&batch.ffn_up, 0),
        (&batch.ffn_act, 0),
        (m * inter) as u32,
    )
    .map_err(gpu_err)?;
    for t in 0..m {
        crate::lab_hooks::encode_dispatch_probe(
            context,
            pass,
            (&batch.ffn_act, row(inter, t)),
            inter,
        )?;
    }
    gemm(
        context,
        "mlp.down_proj.weight",
        hidden,
        inter,
        (&batch.ffn_act, 0),
        (&batch.ffn_out, 0),
    )?;
    gpu::encode_residual_add(
        context,
        pass,
        (&scratch.x, 0),
        (&batch.ffn_out, 0),
        (m * hidden) as u32,
    )
    .map_err(gpu_err)?;

    for t in 0..m {
        let x_off = row(hidden, t);
        encode_steering(context, pass, scratch, l.steering, layer, hidden, 1, x_off)?;
        encode_resid_capture(
            context,
            pass,
            scratch,
            l.resid_capture,
            layer,
            hidden,
            x_off,
        )?;
    }
    Ok(())
}
