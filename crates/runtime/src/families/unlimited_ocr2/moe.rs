//! The MoE half of a layer: `deepseek2`'s function (softmax over ALL experts,
//! top-k, no renormalization, phase 2 seeded with the shared expert), with the
//! gate read as the FP16 rows of a BF16 GEMV. Upstream keeps this router dense
//! on purpose, so it is not transcoded to INT8 the way `flint`'s is.
//!
//! Slot order is the router's ranking (AGENTS.md Gotcha 27), as in every family.

use half::f16;

use crate::families::deepseek2::softmax_topk_no_renorm;
use crate::families::unlimited_ocr2::{layer_tensor, RealUnlimitedOcr2State};
use crate::real_forward_dispatch::{encode_gemv_any, encode_moe_phase1_any, encode_moe_phase2_any};
use crate::real_forward_init::MappedResidency;
use crate::real_forward_layout::RoutedLayerLayout;
use crate::real_forward_types::{DecodeScratch, PhaseCounters, RealForwardError};
use crate::real_forward_utils::f16_slice_to_le_bytes;

/// One MoE layer's routed pass. The caller has committed the pass that wrote
/// `scratch.router_logits`, so reading it here is safe.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_layer_moe(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &model_io::ResidentIndex,
    scratch: &DecodeScratch,
    state: &RealUnlimitedOcr2State,
    streamers: &mut [Option<streaming::PreadExpertStreamer>],
    slot_buffers: &[Vec<gpu::MetalBuffer>],
    mapped: &MappedResidency,
    routed_blobs: Option<&gpu::RoutedBlobsBuffer>,
    moe_offsets: &[gpu::MoeExpertOffsets],
    routed_layouts: &[RoutedLayerLayout],
    phases: &mut PhaseCounters,
    layer: usize,
    hidden: usize,
    moe_inter: u32,
    num_experts: usize,
    top_k: usize,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let blob_layer = layer - state.lead;
    let inter = state.shared_inter;
    let name = |suffix: &str| layer_tensor(layer, &format!("mlp.shared_expert.{suffix}"));

    // First: phase 2's residual seed must hold it when the routed sum lands.
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &name("gate_proj.weight"),
        inter,
        hidden,
        (&state.moe_x, 0),
        (&state.ffn_a, 0),
    )?;
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &name("up_proj.weight"),
        inter,
        hidden,
        (&state.moe_x, 0),
        (&state.ffn_b, 0),
    )?;
    gpu::encode_silu_mul(
        context,
        pass,
        (&state.ffn_a, 0),
        (&state.ffn_b, 0),
        (&state.ffn_a, 0),
        inter as u32,
    )
    .map_err(gpu_err)?;
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &name("down_proj.weight"),
        hidden,
        inter,
        (&state.ffn_a, 0),
        (&state.shared_out, 0),
    )?;

    let logits: Vec<f32> = gpu::read_half_buffer(&scratch.router_logits, num_experts)
        .iter()
        .map(|v| v.to_f32())
        .collect();
    let (selected, route_weights) = softmax_topk_no_renorm(&logits, top_k);

    let mapped_active = mapped.buffers.get(layer).is_some_and(Option::is_some);
    let slots: Vec<usize> = if mapped_active {
        phases.expert_requests += selected.len() as u64;
        phases.expert_hits += selected.len() as u64;
        selected.clone()
    } else {
        let streamer = streamers[layer].as_mut().ok_or_else(|| {
            RealForwardError::Unsupported(format!(
                "unlimited_ocr2 layer {layer} has no packed-expert streamer"
            ))
        })?;
        let plan = streamer.plan_experts_cached(&selected, &std::collections::HashSet::new());
        phases.expert_requests += plan.experts.len() as u64;
        phases.expert_hits += plan.hits as u64;
        let t_io = std::time::Instant::now();
        let slots = streamer
            .execute_expert_cache_plan(&plan)
            .map_err(|e| RealForwardError::Unsupported(format!("expert stream: {e}")))?;
        phases.expert_io_nanos += t_io.elapsed().as_nanos() as u64;
        slots
    };

    let mut routing16 = vec![f16::from_f32(0.0); gpu::MAX_STREAMED_EXPERTS];
    for (dispatch_slot, weight) in route_weights.iter().enumerate() {
        routing16[dispatch_slot] = f16::from_f32(*weight);
    }
    gpu::write_buffer_bytes(&scratch.routing_w, 0, &f16_slice_to_le_bytes(&routing16));

    let blob_refs: Vec<(&gpu::MetalBuffer, u64)> = if mapped_active {
        let buffer = mapped.buffers[layer].as_ref().expect("checked above");
        let mapping = mapped.layers[layer].as_ref().expect("checked above");
        slots
            .iter()
            .map(|&expert| {
                mapping
                    .expert_offset(expert)
                    .map(|offset| (buffer, offset))
                    .map_err(|e| {
                        RealForwardError::Unsupported(format!(
                            "mapped expert layer {layer}, expert {expert}: {e}"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        let layer_slots = &slot_buffers[layer];
        slots.iter().map(|&s| (&layer_slots[s], 0u64)).collect()
    };
    let routed = routed_blobs
        .ok_or_else(|| RealForwardError::Unsupported("install has no routed-blob buffer".into()))?;
    let offsets = &moe_offsets[blob_layer];
    let (blob_source, blob_function) = routed_layouts[blob_layer].phase1.source_function();
    routed
        .bind_for(context, blob_source, blob_function, true, &blob_refs)
        .map_err(gpu_err)?;
    for &(buffer, _) in &blob_refs {
        pass.use_read_buffer(buffer);
    }
    encode_moe_phase1_any(
        routed_layouts[blob_layer],
        context,
        pass,
        routed,
        offsets,
        (&state.moe_x, 0),
        (&scratch.moe_acts, 0),
        hidden as u32,
        moe_inter,
        slots.len() as u32,
        true,
    )
    .map_err(gpu_err)?;
    encode_moe_phase2_any(
        routed_layouts[blob_layer].phase2,
        context,
        pass,
        routed,
        offsets,
        (&scratch.moe_acts, 0),
        (&scratch.routing_w, 0),
        (&state.shared_out, 0),
        (&state.h2, 0),
        hidden as u32,
        moe_inter,
        top_k as u32,
        true,
    )
    .map_err(gpu_err)?;
    gpu::encode_residual_add(
        context,
        pass,
        (&scratch.x, 0),
        (&state.h2, 0),
        hidden as u32,
    )
    .map_err(gpu_err)?;
    Ok(())
}
