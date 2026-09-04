//! The MoE sublayer every `qwen4_exp` layer runs (`mod.rs`'s "## MoE").
//! Adapted from `families/qwen/moe.rs`, which this closely resembles: the
//! same softmax-before-topk-with-renormalization router
//! (`router_topk_gemma4` is algebraically identical, see `mod.rs`'s own
//! note) and the same gated-shared-expert-seeds-phase-2 choice. **ONE
//! STRUCTURAL DIFFERENCE**: this function writes its result into
//! `qwen4.h2` and returns, WITHOUT a residual add -- `families/qwen/`'s
//! sibling adds directly to `scratch.x`, but this family's residual
//! mechanism is the hyper-connection injection
//! (`gpu::encode_hc_inject_add`), which the caller runs afterward using
//! `mixed_hc`'s `raw`/`inject_w` outputs. Folding the add in here would
//! bury a family-specific mechanism inside a function whose whole point is
//! to be the ordinary part.

use std::time::Instant;

use half::f16;
use model_io::ResidentIndex;

use crate::families::qwen4::layer_tensor;
use crate::families::qwen4::state::RealQwen4State;
use crate::real_forward_dispatch::{
    encode_gemv_any, encode_moe_phase1_any, encode_moe_phase2_any, router_topk_gemma4,
};
use crate::real_forward_init::MappedResidency;
use crate::real_forward_layout::RoutedLayerLayout;
use crate::real_forward_types::{DecodeScratch, PhaseCounters, RealForwardError};
use crate::real_forward_utils::f16_slice_to_le_bytes;

/// Encodes ONLY the router GEMV, on the caller's `cb1` -- the pass that
/// also computed `mixed` (`mlp_hc`'s output). The rest of the MoE step
/// (`encode_moe_layer`) needs a HOST readback of this GEMV's result, so it
/// cannot run in the same uncommitted pass: the caller must `commit`,
/// `wait`, begin a new pass, and only then call `encode_moe_layer`.
/// Splitting this out is what makes that ordering possible; folding the
/// GEMV into `encode_moe_layer` itself would read back a stale (or
/// undefined) buffer, since nothing would have waited for THIS token's
/// GEMV to finish before `encode_moe_layer`'s readback runs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_moe_router(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    qwen4: &RealQwen4State,
    mixed: (&gpu::MetalBuffer, u64),
    layer: usize,
    hidden: usize,
    num_experts: usize,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let router_name = layer_tensor(layer, "mlp.gate.weight");
    let base = index.header.index_size;
    let router = index
        .entries
        .get(&router_name)
        .ok_or_else(|| RealForwardError::MissingTensor(router_name.clone()))?;
    if router.dtype != 5 || router.size_bytes as usize != num_experts * hidden {
        return Err(RealForwardError::Unsupported(format!(
            "{router_name}: expected INT8 (dtype 5) {num_experts}x{hidden}, got dtype {} with \
             {} packed bytes",
            router.dtype, router.size_bytes
        )));
    }
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
        mixed,
        (&qwen4.router_ones, 0),
        (&qwen4.router_logits_f32, 0),
        num_experts as u32,
        hidden as u32,
    )
    .map_err(gpu_err)
}

/// The rest of the MoE step: host readback of the router GEMV
/// [`encode_moe_router`] already committed and waited on, expert plan and
/// bind, the gated shared expert, and the routed phase 1/2 pair. Runs on a
/// NEW pass (`families/qwen/moe.rs`'s "routed cb").
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_moe_layer(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    scratch: &DecodeScratch,
    qwen4: &RealQwen4State,
    mixed: (&gpu::MetalBuffer, u64),
    streamers: &mut [Option<streaming::PreadExpertStreamer>],
    slot_buffers: &[Vec<gpu::MetalBuffer>],
    mapped: &MappedResidency,
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
    let router_logits = gpu::read_f32_buffer(&qwen4.router_logits_f32, num_experts);
    let (selected, route_weights) =
        router_topk_gemma4(&router_logits, top_k, &qwen4.per_expert_ones);
    if let Some(hist) = router_hist.as_mut() {
        hist.record(layer, &selected);
    }
    let mapped_active = mapped.buffers.get(layer).is_some_and(Option::is_some);
    let slots: Vec<usize> = if mapped_active {
        phases.expert_requests += selected.len() as u64;
        phases.expert_hits += selected.len() as u64;
        phases.router_nanos += t_router.elapsed().as_nanos() as u64;
        selected.clone()
    } else {
        let streamer = streamers[layer].as_mut().ok_or_else(|| {
            RealForwardError::Unsupported(format!(
                "real qwen4_exp layer {layer} has no packed-expert streamer"
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
        slots
    };

    let t_bind = Instant::now();
    let ordered: Vec<(usize, f32)> = (0..selected.len())
        .map(|i| (slots[i], route_weights[i]))
        .collect();
    let mut routing16 = vec![f16::from_f32(0.0); gpu::MAX_STREAMED_EXPERTS];
    for (slot, &(_, weight)) in ordered.iter().enumerate() {
        routing16[slot] = f16::from_f32(weight);
    }
    gpu::write_buffer_bytes(&scratch.routing_w, 0, &f16_slice_to_le_bytes(&routing16));

    let blob_refs: Vec<(&gpu::MetalBuffer, u64)> = if mapped_active {
        let buffer = mapped.buffers[layer]
            .as_ref()
            .expect("mapped residency checked above");
        let mapping = mapped.layers[layer]
            .as_ref()
            .expect("mapped residency checked above");
        ordered
            .iter()
            .map(|&(expert, _)| (buffer, mapping.expert_offset(expert)))
            .collect()
    } else {
        let layer_slots = &slot_buffers[layer];
        ordered
            .iter()
            .map(|&(slot, _)| (&layer_slots[slot], 0u64))
            .collect()
    };
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

    // Gated shared expert: SEEDS phase 2's accumulator (`mod.rs`'s own
    // note; `docs/QWEN4_PHASE0.md` item 7).
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &layer_tensor(layer, "mlp.shared_expert.gate_proj.weight"),
        inter,
        hidden,
        mixed,
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
        mixed,
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
        (&qwen4.h1, 0),
    )?;
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &layer_tensor(layer, "mlp.shared_expert_gate.weight"),
        1,
        hidden,
        mixed,
        (&qwen4.shared_gate_logit, 0),
    )?;
    gpu::encode_sigmoid_scalar_mul(
        context,
        pass,
        (&qwen4.h1, 0),
        (&qwen4.shared_gate_logit, 0),
        hidden as u32,
    )
    .map_err(gpu_err)?;

    encode_moe_phase1_any(
        routed_layouts[layer].phase1,
        context,
        pass,
        routed,
        offsets,
        mixed,
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
        (&qwen4.h1, 0),
        (&qwen4.h2, 0),
        hidden as u32,
        moe_inter,
        top_k as u32,
        use_silu,
    )
    .map_err(gpu_err)?;

    // NO residual add here -- see the module doc. `qwen4.h2` holds `moe(mixed)`.
    Ok(())
}
