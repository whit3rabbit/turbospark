//! The `llama` architecture's routed-expert pass (ROADMAP Phase M2).
//!
//! Qwen's sibling with the shared expert removed, which is most of it:
//! Mixtral routes to `top_k` experts and adds nothing else, so phase 2's
//! residual input is a ZERO buffer rather than a gated shared-expert output,
//! and the routed sum is added to the stream afterwards.
//!
//! The slot order is the ROUTER'S RANKING, as it is in both other families,
//! and that is a correctness constraint rather than a style choice: phase 2
//! reduces `blob[slot] * routing_w[slot]` in slot-index order and FP addition
//! is not associative, so whatever the slot order depends on, the generated
//! bytes depend on (AGENTS.md Gotcha 27).
//!
//! Takes a [`RoutedSlot`] since the chunked-prefill driver
//! (`prefill.rs`) pipelines this call across a micro-batch's tokens, the
//! same way `families/gemma4/moe.rs` does. The sequential decode path
//! passes `RoutedSlot::sequential()`, so its bytes cannot move.

use std::time::Instant;

use half::f16;
use model_io::ResidentIndex;

use crate::families::llama::RealLlamaState;
use crate::moe_prefill_pipeline::RoutedSlot;
use crate::real_forward_dispatch::{
    encode_moe_phase1_any, encode_moe_phase2_any, router_topk_gemma4,
};
use crate::real_forward_layout::RoutedLayerLayout;
use crate::real_forward_types::{DecodeScratch, PhaseCounters, RealForwardError};
use crate::real_forward_utils::f16_slice_to_le_bytes;

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_llama_layer_moe(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    index: &ResidentIndex,
    scratch: &DecodeScratch,
    llama: &RealLlamaState,
    streamers: &mut [Option<streaming::PreadExpertStreamer>],
    slot_buffers: &[Vec<gpu::MetalBuffer>],
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
    use_silu: bool,
    slot: &RoutedSlot,
) -> Result<Vec<usize>, RealForwardError> {
    let _ = index;
    let gpu_err = RealForwardError::Gpu;
    let x_off = (slot.token * hidden * 2) as u64;
    let rw_off = slot.bank * gpu::MAX_STREAMED_EXPERTS * 2;
    let t_router = Instant::now();
    let router_logits = gpu::read_f32_buffer_at(
        &llama.router_logits_f32,
        slot.token * num_experts,
        num_experts,
    );
    let (selected, route_weights) =
        router_topk_gemma4(&router_logits, top_k, &llama.per_expert_ones);
    if let Some(hist) = router_hist.as_mut() {
        hist.record(layer, &selected);
    }
    let streamer = streamers[layer].as_mut().ok_or_else(|| {
        RealForwardError::Unsupported(format!("llama layer {layer} has no packed-expert streamer"))
    })?;
    // `protect` is empty on the decode path, so this is the same call it has
    // always made. Inside a chunk it names the slots the previous token's
    // in-flight command buffer is reading; the cache ASSERTS rather than
    // degrades when it cannot honour that plus the misses
    // (`ExpertCache::plan_if_possible`), so the caller has already ensured
    // the arithmetic works or retired the buffer first.
    let plan = streamer.plan_experts_cached(&selected, &slot.protect);
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

    let layer_slots = &slot_buffers[layer];
    let blob_refs: Vec<(&gpu::MetalBuffer, u64)> = ordered
        .iter()
        .map(|&(cache_slot, _)| (&layer_slots[cache_slot], 0u64))
        .collect();
    let routed = match slot.bank {
        0 => routed_blobs,
        n => routed_blobs_banks.get(n - 1),
    }
    .ok_or_else(|| {
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

    encode_moe_phase1_any(
        routed_layouts[layer].phase1,
        context,
        pass,
        routed,
        offsets,
        (&llama.moe_x, x_off),
        (&scratch.moe_acts, 0),
        hidden as u32,
        moe_inter,
        ordered.len() as u32,
        use_silu,
    )
    .map_err(gpu_err)?;
    // `zero_hidden` is the residual input, because there is no shared expert
    // to carry one: phase 2 computes `y = residual + sum(w * expert)`, so a
    // zeroed input makes `y` the routed sum alone and the add below is the
    // only place the stream grows. Passing `x` here instead would add the
    // residual twice.
    encode_moe_phase2_any(
        routed_layouts[layer].phase2,
        context,
        pass,
        routed,
        offsets,
        (&scratch.moe_acts, 0),
        (&scratch.routing_w, rw_off as u64),
        (&scratch.zero_hidden, 0),
        (&llama.h2, 0),
        hidden as u32,
        moe_inter,
        use_silu,
    )
    .map_err(gpu_err)?;
    gpu::encode_residual_add(
        context,
        pass,
        (&scratch.x, x_off),
        (&llama.h2, 0),
        hidden as u32,
    )
    .map_err(gpu_err)?;

    // The cache slots this pass BOUND, so the caller can hand them to the
    // next token as `RoutedSlot::protect` while this command buffer is in
    // flight. Returned rather than recomputed, matching
    // `encode_gemma4_layer_routed_moe`'s exact reasoning.
    Ok(ordered.iter().map(|&(cache_slot, _)| cache_slot).collect())
}
