//! Layer encoders for Qwen batched verify pass (`produce_batched`).

use model_io::{ArchConfig, ResidentIndex};

use super::batched_layers::{encode_full_attention_block_batched, encode_linear_block_batched};
use super::moe_batch::encode_qwen_layer_moe_batched;
use super::{layer_tensor, BatchedScratch, RealQwenState, RMS_EPS};
use crate::real_forward_init::MappedResidency;
use crate::real_forward_layout::RoutedLayerLayout;
use crate::real_forward_types::{DecodeScratch, PhaseCounters, RealForwardError};

/// One token's router GEMV, into row `m` of the batch's own logits buffer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_router_gemv_batched(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &model_io::ResidentIndex,
    router_name: &str,
    qwen: &RealQwenState,
    batched: &BatchedScratch,
    m: usize,
    hidden: usize,
    num_experts: usize,
) -> Result<(), RealForwardError> {
    let routed = batched.routed.as_ref().ok_or_else(|| {
        RealForwardError::Unsupported("routed scratch missing on a MoE install".to_string())
    })?;
    let router = crate::real_forward_utils::entry(index, router_name)?;
    if router.dtype != 5 || router.size_bytes as usize != num_experts * hidden {
        return Err(RealForwardError::Unsupported(format!(
            "{router_name}: expected INT8 (dtype 5) {num_experts}x{hidden}, got dtype {} \
             with {} packed bytes",
            router.dtype, router.size_bytes
        )));
    }
    let base = index.header.index_size;
    gpu::encode_router_gemv_gemma4(
        context,
        pass,
        (
            weights.buffer(),
            weights.gpu_offset(router.file_offset - base),
        ),
        (
            weights.buffer(),
            weights.gpu_offset(router.scale_offset - base),
        ),
        (
            weights.buffer(),
            weights.gpu_offset(router.bias_offset - base),
        ),
        (&batched.moe_x, (m * hidden * 2) as u64),
        (&qwen.router_ones, 0),
        (
            &routed.batch_router_logits_f32,
            (m * num_experts * 4) as u64,
        ),
        num_experts as u32,
        hidden as u32,
    )
    .map_err(RealForwardError::Gpu)
}

/// Executes the MoE branch of a verify layer: M router GEMVs, commit/wait,
/// retire previous routed buffer, and encode batched MoE.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_moe_layer_batched_step(
    context: &mut gpu::MetalContext,
    pass: gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    scratch: &DecodeScratch,
    qwen: &RealQwenState,
    batched: &BatchedScratch,
    streamers: &mut [Option<streaming::PreadExpertStreamer>],
    slot_buffers: &[Vec<gpu::MetalBuffer>],
    mapped: &MappedResidency,
    moe_offsets: &[gpu::MoeExpertOffsets],
    routed_layouts: &[RoutedLayerLayout],
    router_hist: &mut Option<crate::router_hist::RouterHistogram>,
    phases: &mut PhaseCounters,
    expert_cache_slots: usize,
    layer: usize,
    hidden: usize,
    inter: usize,
    moe_inter: u32,
    num_experts: usize,
    top_k: usize,
    use_silu: bool,
    batch: usize,
    routed_in_flight: &mut Option<gpu::CommittedPass>,
) -> Result<gpu::PassEncoder, RealForwardError> {
    let router_name = layer_tensor(layer, "mlp.gate.weight");
    for m in 0..batch {
        encode_router_gemv_batched(
            context,
            &pass,
            weights,
            index,
            &router_name,
            qwen,
            batched,
            m,
            hidden,
            num_experts,
        )?;
    }
    let t_wait = std::time::Instant::now();
    phases.cb1_gpu_nanos += (pass.commit().wait_with_gpu_time() * 1e9) as u64;
    phases.gpu_wait_nanos += t_wait.elapsed().as_nanos() as u64;

    if let Some(prev) = routed_in_flight.take() {
        phases.routed_cb_gpu_nanos += (prev.wait_with_gpu_time() * 1e9) as u64;
    }
    *routed_in_flight = Some(encode_qwen_layer_moe_batched(
        context,
        weights,
        index,
        scratch,
        qwen,
        batched,
        streamers,
        slot_buffers,
        mapped,
        moe_offsets,
        routed_layouts,
        router_hist,
        phases,
        expert_cache_slots,
        layer,
        hidden,
        inter,
        moe_inter,
        num_experts,
        top_k,
        use_silu,
        batch,
    )?);
    Ok(context.begin_pass_labeled("batched verify"))
}

/// Encodes attention and pre/post norms for M rows in `produce_batched`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_layer_attn_and_norms_batched(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    scratch: &DecodeScratch,
    batched: &BatchedScratch,
    qwen: &RealQwenState,
    kv: &mut gpu::KvCacheManager,
    arch: &ArchConfig,
    input_norm: (&gpu::MetalBuffer, u64),
    post_attn: (&gpu::MetalBuffer, u64),
    layer: usize,
    hidden: usize,
    start_position: usize,
    batch: usize,
    // This layer's tape slot when the pass records one, `None` otherwise.
    // Forwarded to the linear block, whose projections are what the tape
    // copies; attention layers have no tape.
    tape_slot: Option<usize>,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    for m in 0..batch {
        gpu::encode_rms_norm_bf16w(
            context,
            pass,
            (&scratch.x, (m * hidden) as u64 * 2),
            input_norm,
            (&batched.normed, (m * hidden) as u64 * 2),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
    }

    if arch.layer_is_linear(layer) {
        encode_linear_block_batched(
            context, pass, weights, index, arch, qwen, batched, layer, batch, tape_slot,
        )?;
    } else {
        encode_full_attention_block_batched(
            context,
            pass,
            weights,
            index,
            arch,
            qwen,
            scratch,
            batched,
            kv,
            layer,
            start_position,
            batch,
        )?;
    }

    for m in 0..batch {
        let row = (m * hidden) as u64 * 2;
        gpu::encode_residual_add(
            context,
            pass,
            (&scratch.x, row),
            (&batched.o, row),
            hidden as u32,
        )
        .map_err(gpu_err)?;
        gpu::encode_rms_norm_bf16w(
            context,
            pass,
            (&scratch.x, row),
            post_attn,
            (&batched.moe_x, row),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
    }
    Ok(())
}

/// Encodes final rms norm and lm_head GEMM for Qwen verify pass (`produce_batched`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_batched_head(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    scratch: &DecodeScratch,
    batched: &BatchedScratch,
    arch: &ArchConfig,
    embed_name: &str,
    hidden: usize,
    vocab: usize,
    batch: usize,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let final_norm = crate::real_forward_utils::norm_view(
        weights,
        index,
        "language_model.model.norm.weight",
        hidden,
    )?;
    for m in 0..batch {
        gpu::encode_rms_norm_bf16w(
            context,
            pass,
            (&scratch.x, (m * hidden) as u64 * 2),
            final_norm,
            (&batched.normed, (m * hidden) as u64 * 2),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
    }
    let head_name = if arch.tie_word_embeddings {
        embed_name.to_string()
    } else {
        "language_model.lm_head.weight".to_string()
    };
    crate::real_forward_dispatch::encode_gemm_any(
        context,
        pass,
        weights,
        index,
        &head_name,
        vocab,
        hidden,
        (&batched.normed, 0),
        (&batched.logits, 0),
        batch,
    )
}
