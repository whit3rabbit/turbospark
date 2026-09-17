//! The `deepseek2` MoE pass: the shared expert (a plain SwiGLU at the fused
//! width), the INT8 router, softmax-over-ALL top-k with NO renormalization
//! (`norm_topk_prob: false` -- the llama flow's softmax-over-selected is a
//! DIFFERENT function and would reweight every expert), and the routed
//! sum added on top of the shared expert through phase 2's residual seed.
//!
//! Slot order is the router's ranking (AGENTS.md Gotcha 27), as in every
//! family.

use half::f16;

use crate::families::deepseek2::RealDeepseek2State;
use crate::moe_prefill_pipeline::RoutedSlot;
use crate::real_forward_dispatch::{encode_moe_phase1_any, encode_moe_phase2_any};
use crate::real_forward_init::MappedResidency;
use crate::real_forward_layout::RoutedLayerLayout;
use crate::real_forward_types::{DecodeScratch, PhaseCounters, RealForwardError};
use crate::real_forward_utils::f16_slice_to_le_bytes;

/// Softmax over ALL experts, take the top-k, weights are those
/// full-softmax values with no renormalization and no per-expert scale
/// (`routed_scaling_factor` is 1.0; when a checkpoint ever ships a
/// different one, it multiplies here -- not in the router kernel).
pub(crate) fn softmax_topk_no_renorm(logits: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, (l - max).exp()))
        .collect();
    let denom: f32 = probs.iter().map(|(_, p)| p).sum();
    for (_, p) in probs.iter_mut() {
        *p /= denom;
    }
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    probs.truncate(k);
    probs.into_iter().unzip()
}

/// The dense SwiGLU that IS the shared expert: `down(silu(gate(x)) * up(x))`
/// at the fused width, run from `moe_x`'s row into `state.shared_out`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_shared_expert(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &model_io::ResidentIndex,
    state: &RealDeepseek2State,
    layer: usize,
    token: usize,
    hidden: usize,
    inter: usize,
) -> Result<(), RealForwardError> {
    let x_off = (token * hidden * 2) as u64;
    let name = |suffix: &str| {
        crate::families::deepseek2::layer_tensor(layer, &format!("mlp.shared_expert.{suffix}"))
    };
    encode_gemv_rows(
        context,
        pass,
        weights,
        index,
        &name("gate_proj.weight"),
        inter,
        hidden,
        (&state.moe_x, x_off),
        (&state.ffn_a, 0),
    )?;
    encode_gemv_rows(
        context,
        pass,
        weights,
        index,
        &name("up_proj.weight"),
        inter,
        hidden,
        (&state.moe_x, x_off),
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
    .map_err(RealForwardError::Gpu)?;
    encode_gemv_rows(
        context,
        pass,
        weights,
        index,
        &name("down_proj.weight"),
        hidden,
        inter,
        (&state.ffn_a, 0),
        (&state.shared_out, 0),
    )
}

/// `encode_gemv_any` under another name for the shared expert's calls; a
/// re-export would carry the same list of arguments anyway.
#[allow(clippy::too_many_arguments)]
fn encode_gemv_rows(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &model_io::ResidentIndex,
    name: &str,
    rows: usize,
    cols: usize,
    x: (&gpu::MetalBuffer, u64),
    y: (&gpu::MetalBuffer, u64),
) -> Result<(), RealForwardError> {
    crate::real_forward_dispatch::encode_gemv_any(
        context, pass, weights, index, name, rows, cols, x, y,
    )
}

/// One MoE layer's FFN half: shared expert, host top-k, routed phase 1/2
/// with the shared output as phase 2's residual seed, then the stream add.
/// Blob positions are `layer - state.lead` throughout.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_deepseek2_layer_moe(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &model_io::ResidentIndex,
    scratch: &DecodeScratch,
    state: &RealDeepseek2State,
    streamers: &mut [Option<streaming::PreadExpertStreamer>],
    slot_buffers: &[Vec<gpu::MetalBuffer>],
    mapped: &MappedResidency,
    routed_blobs: Option<&gpu::RoutedBlobsBuffer>,
    routed_blobs_banks: &[gpu::RoutedBlobsBuffer],
    moe_offsets: &[gpu::MoeExpertOffsets],
    routed_layouts: &[RoutedLayerLayout],
    router_hist: &mut Option<crate::router_hist::RouterHistogram>,
    phases: &mut PhaseCounters,
    layer: usize,
    hidden: usize,
    moe_inter: u32,
    num_experts: usize,
    top_k: usize,
    slot: &RoutedSlot,
) -> Result<Vec<usize>, RealForwardError> {
    let x_off = (slot.token * hidden * 2) as u64;
    let rw_off = slot.bank * gpu::MAX_STREAMED_EXPERTS * 2;
    let blob_layer = layer - state.lead;

    // The shared expert runs FIRST, on the routed pass, because phase 2's
    // residual seed must already hold its output when the routed sum lands.
    encode_shared_expert(
        context,
        pass,
        weights,
        index,
        state,
        layer,
        slot.token,
        hidden,
        state.shared_inter,
    )?;

    let router_logits = gpu::read_f32_buffer_at(
        &state.router_logits_f32,
        slot.token * num_experts,
        num_experts,
    );
    let (selected, route_weights) = softmax_topk_no_renorm(&router_logits, top_k);
    if let Some(hist) = router_hist.as_mut() {
        hist.record(blob_layer, &selected);
    }
    let mapped_active = mapped.buffers.get(layer).is_some_and(Option::is_some);
    let slots: Vec<usize> = if mapped_active {
        phases.expert_requests += selected.len() as u64;
        phases.expert_hits += selected.len() as u64;
        selected.clone()
    } else {
        let streamer = streamers[layer].as_mut().ok_or_else(|| {
            RealForwardError::Unsupported(format!(
                "deepseek2 layer {layer} has no packed-expert streamer"
            ))
        })?;
        let plan = streamer.plan_experts_cached(&selected, &slot.protect);
        phases.expert_requests += plan.experts.len() as u64;
        phases.expert_hits += plan.hits as u64;
        let t_io = std::time::Instant::now();
        let slots = streamer
            .execute_expert_cache_plan(&plan)
            .map_err(|e| RealForwardError::Unsupported(format!("expert stream: {e}")))?;
        phases.expert_io_nanos += t_io.elapsed().as_nanos() as u64;
        slots
    };

    let ordered: Vec<(usize, f32)> = (0..selected.len())
        .map(|i| (slots[i], route_weights[i]))
        .collect();
    let mut routing16 = vec![f16::from_f32(0.0); gpu::MAX_STREAMED_EXPERTS];
    for (dispatch_slot, &(_, weight)) in ordered.iter().enumerate() {
        routing16[dispatch_slot] = f16::from_f32(weight);
    }
    gpu::write_buffer_bytes(
        &scratch.routing_w,
        rw_off,
        &f16_slice_to_le_bytes(&routing16),
    );

    let blob_refs: Vec<(&gpu::MetalBuffer, u64)> = if mapped_active {
        let buffer = mapped.buffers[layer]
            .as_ref()
            .expect("mapped residency checked above");
        let mapping = mapped.layers[layer]
            .as_ref()
            .expect("mapped residency checked above");
        ordered
            .iter()
            .map(|&(expert, _)| {
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
        ordered
            .iter()
            .map(|&(cache_slot, _)| (&layer_slots[cache_slot], 0u64))
            .collect()
    };
    let routed = match slot.bank {
        0 => routed_blobs,
        n => routed_blobs_banks.get(n - 1),
    }
    .ok_or_else(|| {
        RealForwardError::Unsupported("install has no routed-blob buffer".to_string())
    })?;
    let offsets = &moe_offsets[blob_layer];
    let (blob_source, blob_function) = routed_layouts[blob_layer].phase1.source_function();
    let use_silu = true;
    routed
        .bind_for(context, blob_source, blob_function, use_silu, &blob_refs)
        .map_err(RealForwardError::Gpu)?;
    for &(buffer, _) in &blob_refs {
        pass.use_read_buffer(buffer);
    }

    encode_moe_phase1_any(
        routed_layouts[blob_layer].phase1,
        context,
        pass,
        routed,
        offsets,
        (&state.moe_x, x_off),
        (&scratch.moe_acts, 0),
        hidden as u32,
        moe_inter,
        ordered.len() as u32,
        use_silu,
    )
    .map_err(RealForwardError::Gpu)?;
    // Phase 2's residual input is the SHARED EXPERT's output, so `y =
    // shared + routed sum` lands in h2 in one kernel and the add below is
    // the only place the stream grows (the llama flow seeds zero for the
    // same reason).
    encode_moe_phase2_any(
        routed_layouts[blob_layer].phase2,
        context,
        pass,
        routed,
        offsets,
        (&scratch.moe_acts, 0),
        (&scratch.routing_w, rw_off as u64),
        (&state.shared_out, 0),
        (&state.h2, 0),
        hidden as u32,
        moe_inter,
        top_k as u32,
        use_silu,
    )
    .map_err(RealForwardError::Gpu)?;
    gpu::encode_residual_add(
        context,
        pass,
        (&scratch.x, x_off),
        (&state.h2, 0),
        hidden as u32,
    )
    .map_err(RealForwardError::Gpu)?;

    Ok(ordered.iter().map(|&(cache_slot, _)| cache_slot).collect())
}
