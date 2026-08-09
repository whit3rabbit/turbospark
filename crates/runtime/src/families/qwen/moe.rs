//! MoE layer encoding block for Qwen 3.6 decode flow.

use std::time::Instant;

use half::f16;
use model_io::ResidentIndex;

use crate::families::qwen::{layer_tensor, RealQwenState};
use crate::real_forward_dispatch::{
    encode_gemv_any, encode_moe_phase1_any, encode_moe_phase2_any, router_topk_gemma4,
};
use crate::real_forward_layout::RoutedLayerLayout;
use crate::real_forward_types::{DecodeScratch, PhaseCounters, RealForwardError};
use crate::real_forward_utils::f16_slice_to_le_bytes;

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_layer_moe(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    scratch: &DecodeScratch,
    qwen: &RealQwenState,
    streamers: &mut [Option<streaming::PreadExpertStreamer>],
    slot_buffers: &[Vec<gpu::MetalBuffer>],
    routed_blobs: Option<&gpu::RoutedBlobsBuffer>,
    moe_offsets: &[gpu::MoeExpertOffsets],
    routed_layouts: &[RoutedLayerLayout],
    router_hist: &mut Option<crate::router_hist::RouterHistogram>,
    phases: &mut PhaseCounters,
    layer: usize,
    hidden: usize,
    inter: usize,
    moe_inter: u32,
    num_experts: usize,
    top_k: usize,
    use_silu: bool,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let t_router = Instant::now();
    let router_logits = gpu::read_f32_buffer(&qwen.router_logits_f32, num_experts);
    let (selected, route_weights) =
        router_topk_gemma4(&router_logits, top_k, &qwen.per_expert_ones);
    if let Some(hist) = router_hist.as_mut() {
        hist.record(layer, &selected);
    }
    let streamer = streamers[layer].as_mut().ok_or_else(|| {
        RealForwardError::Unsupported(format!(
            "real Qwen 3.6 layer {layer} has no packed-expert streamer"
        ))
    })?;
    let plan = streamer.plan_experts_cached(&selected, &std::collections::HashSet::new());
    let (requests, hits) = (plan.experts.len() as u64, plan.hits as u64);
    phases.expert_requests += requests;
    phases.expert_hits += hits;
    phases.router_nanos += t_router.elapsed().as_nanos() as u64;

    let streamer = streamers[layer]
        .as_mut()
        .expect("streamer presence checked above");
    let t_io = Instant::now();
    let slots = streamer
        .execute_expert_cache_plan(&plan)
        .map_err(|e| RealForwardError::Unsupported(format!("expert stream: {e}")))?;
    phases.expert_io_nanos += t_io.elapsed().as_nanos() as u64;

    let t_bind = Instant::now();
    let order: Vec<usize> = (0..selected.len()).collect();
    let ordered: Vec<(usize, f32)> = order
        .iter()
        .map(|&i| (slots[i], route_weights[i]))
        .collect();
    let mut routing16 = vec![f16::from_f32(0.0); gpu::MAX_STREAMED_EXPERTS];
    for (slot, &(_, weight)) in ordered.iter().enumerate() {
        routing16[slot] = f16::from_f32(weight);
    }
    gpu::write_buffer_bytes(&scratch.routing_w, 0, &f16_slice_to_le_bytes(&routing16));

    let layer_slots = &slot_buffers[layer];
    let blob_refs: Vec<(&gpu::MetalBuffer, u64)> = ordered
        .iter()
        .map(|&(slot, _)| (&layer_slots[slot], 0u64))
        .collect();
    let routed = routed_blobs.ok_or_else(|| {
        RealForwardError::Unsupported("install has no routed-blob buffer".to_string())
    })?;
    let offsets = &moe_offsets[layer];
    routed
        .bind(context, use_silu, &blob_refs)
        .map_err(gpu_err)?;
    phases.bind_nanos += t_bind.elapsed().as_nanos() as u64;

    for &(buffer, _) in &blob_refs {
        pass.use_read_buffer(buffer);
    }

    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &layer_tensor(layer, "mlp.shared_expert.gate_proj.weight"),
        inter,
        hidden,
        (&qwen.moe_x, 0),
        (&scratch.ffn_gate, 0),
    )?;
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &layer_tensor(layer, "mlp.shared_expert.up_proj.weight"),
        inter,
        hidden,
        (&qwen.moe_x, 0),
        (&scratch.ffn_up, 0),
    )?;
    let act = if use_silu {
        gpu::encode_silu_mul
    } else {
        gpu::encode_gelu_mul
    };
    act(
        context,
        pass,
        (&scratch.ffn_gate, 0),
        (&scratch.ffn_up, 0),
        (&scratch.ffn_act, 0),
        inter as u32,
    )
    .map_err(gpu_err)?;
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &layer_tensor(layer, "mlp.shared_expert.down_proj.weight"),
        hidden,
        inter,
        (&scratch.ffn_act, 0),
        (&qwen.h1, 0),
    )?;
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &layer_tensor(layer, "mlp.shared_expert_gate.weight"),
        1,
        hidden,
        (&qwen.moe_x, 0),
        (&qwen.shared_gate_logit, 0),
    )?;
    gpu::encode_sigmoid_scalar_mul(
        context,
        pass,
        (&qwen.h1, 0),
        (&qwen.shared_gate_logit, 0),
        hidden as u32,
    )
    .map_err(gpu_err)?;

    encode_moe_phase1_any(
        routed_layouts[layer].phase1,
        context,
        pass,
        routed,
        offsets,
        (&qwen.moe_x, 0),
        (&scratch.moe_acts, 0),
        hidden as u32,
        moe_inter,
        selected.len() as u32,
        use_silu,
    )
    .map_err(gpu_err)?;
    encode_moe_phase2_any(
        routed_layouts[layer].phase2,
        context,
        pass,
        routed,
        offsets,
        (&scratch.moe_acts, 0),
        (&scratch.routing_w, 0),
        (&qwen.h1, 0),
        (&qwen.h2, 0),
        hidden as u32,
        moe_inter,
        use_silu,
    )
    .map_err(gpu_err)?;
    gpu::encode_residual_add(context, pass, (&scratch.x, 0), (&qwen.h2, 0), hidden as u32)
        .map_err(gpu_err)?;

    Ok(())
}
