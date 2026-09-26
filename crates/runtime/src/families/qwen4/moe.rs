//! The MoE sublayer every `qwen4_exp` layer runs (`mod.rs`'s "## MoE").
//! Adapted from `families/qwen/moe.rs`, which this closely resembles: the
//! same softmax-before-topk-with-renormalization router
//! (`router_topk_gemma4` is algebraically identical, see `mod.rs`'s own
//! note). Qwen4's reference adds the gated shared expert after the routed
//! sum is rounded to FP16. **ONE
//! STRUCTURAL DIFFERENCE**: this function writes its result into
//! `qwen4.h2` and returns, WITHOUT a wide residual add -- `families/qwen/`'s
//! sibling adds directly to `scratch.x`, but this family's residual
//! mechanism is the hyper-connection injection
//! (`gpu::encode_hc_inject_add`), which the caller runs afterward using
//! `mixed_hc`'s `raw`/`inject_w` outputs. Folding the add in here would
//! The shared branch joins the routed result here; the wide-stream
//! hyper-connection residual remains with the caller.

use std::time::Instant;

use half::f16;
use model_io::ResidentIndex;

use crate::families::qwen4::layer_tensor;
use crate::families::qwen4::state::RealQwen4State;
// Re-exported (not just imported): `prefill.rs` reaches this type as
// `moe::RoutedSlot`, matching `families/gemma4/moe.rs`'s own precedent for
// why it is re-exported rather than imported separately at each call site.
pub(crate) use crate::moe_prefill_pipeline::RoutedSlot;
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
///
/// `logits_row_offset` is the byte offset into `qwen4.router_logits_f32`
/// this token's logits land at: `0` on the sequential decode path (single
/// row), `t * num_experts * 4` inside a chunked-prefill micro-batch, where
/// the whole layer's `cb1` writes every token's row before ONE host
/// readback covers them all (`prefill.rs`).
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
    logits_row_offset: u64,
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
        (&qwen4.router_logits_f32, logits_row_offset),
        num_experts as u32,
        hidden as u32,
    )
    .map_err(gpu_err)
}

/// The rest of the MoE step: host readback of the router GEMV
/// [`encode_moe_router`] already committed and waited on, expert plan and
/// bind, the gated shared expert, and the routed phase 1/2 pair. Runs on a
/// NEW pass (`families/qwen/moe.rs`'s "routed cb").
///
/// `slot` names which token of a chunked-prefill micro-batch this call is
/// for (`RoutedSlot::sequential()` on the decode path: token 0, bank 0,
/// nothing protected) -- it selects the row `router_logits_f32` is read
/// from, the bank `scratch.routing_w` is written to, and the slots the
/// expert-cache plan may not evict because a previous token's command
/// buffer is still in flight reading them (AGENTS.md Gotcha 64).
/// Returns the cache slots THIS call bound, for the caller to
/// hand to the next token's `RoutedSlot::protect`.
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
    routed_blobs_banks: &[gpu::RoutedBlobsBuffer],
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
    slot: &RoutedSlot,
) -> Result<Vec<usize>, RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let rw_off = slot.bank * gpu::MAX_STREAMED_EXPERTS * 2;
    let t_router = Instant::now();
    let router_logits = gpu::read_f32_buffer_at(
        &qwen4.router_logits_f32,
        slot.token * num_experts,
        num_experts,
    );
    let (selected, route_weights) =
        router_topk_gemma4(&router_logits, top_k, &qwen4.per_expert_ones);
    if let Some(hist) = router_hist.as_mut() {
        hist.record_router_logits(layer, &router_logits);
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
        // `slot.protect` is empty on the sequential decode path, so this is
        // the same call it has always made. Inside a chunk it names the
        // slots a previous token's in-flight command buffer is reading; the
        // cache ASSERTS rather than degrades when it cannot honour that plus
        // the misses (`ExpertCache::plan_if_possible`), so the caller has
        // already ensured the arithmetic works or retired the buffer first
        // (`moe_prefill_pipeline::routed_pipeline_banks`'s `banks == 1`
        // fallback).
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
        slots
    };

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
            .map(|&(slot, _)| (&layer_slots[slot], 0u64))
            .collect()
    };
    // Selected by bank, matching `families/gemma4/moe.rs`'s identical match:
    // a pipelined bank rebinds its OWN argument-buffer object rather than
    // the one a still-in-flight command buffer's dispatches may still be
    // reading (the same host-write-while-GPU-reads hazard `routing_w`'s
    // banking exists for, one buffer over).
    let routed = match slot.bank {
        0 => routed_blobs,
        n => routed_blobs_banks.get(n - 1),
    }
    .ok_or_else(|| {
        RealForwardError::Unsupported("install has no routed-blob buffer".to_string())
    })?;
    let offsets = &moe_offsets[layer];
    let (blob_source, blob_function) = routed_layouts[layer].phase1.source_function();
    routed
        .bind_for(context, blob_source, blob_function, use_silu, &blob_refs)
        .map_err(gpu_err)?;
    phases.bind_nanos += t_bind.elapsed().as_nanos() as u64;
    for &(buffer, _) in &blob_refs {
        pass.use_read_buffer(buffer);
    }

    // Compute the shared branch now, then join it after the routed sum has
    // been rounded to FP16, matching SlotStream's Qwen4 reference.
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
        (&scratch.routing_w, rw_off as u64),
        (&qwen4.moe_zero, 0),
        (&qwen4.h2, 0),
        hidden as u32,
        moe_inter,
        top_k as u32,
        use_silu,
    )
    .map_err(gpu_err)?;
    gpu::encode_residual_add(context, pass, (&qwen4.h2, 0), (&qwen4.h1, 0), hidden as u32)
        .map_err(gpu_err)?;

    // No wide-stream residual add here. h2 holds the complete MoE output;
    // the caller applies the hyper-connection injection.
    // The cache slots this pass BOUND, so the caller can hand them to the
    // next token as `RoutedSlot::protect` while this command buffer is in
    // flight (`families/gemma4/moe.rs`'s identical return, one family over).
    Ok(ordered.iter().map(|&(slot, _)| slot).collect())
}
