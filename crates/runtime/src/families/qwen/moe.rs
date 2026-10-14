//! MoE layer encoding block for Qwen 3.6 decode flow.

use std::time::Instant;

use half::f16;
use model_io::ResidentIndex;

use crate::families::qwen::{layer_tensor, RealQwenState};
pub(crate) use crate::moe_prefill_pipeline::RoutedSlot;
use crate::real_forward_dispatch::{
    encode_gemv_any, encode_moe_phase1_any, encode_moe_phase2_any, router_topk_gemma4,
};
use crate::real_forward_init::MappedResidency;
use crate::real_forward_layout::RoutedLayerLayout;
use crate::real_forward_types::{DecodeScratch, PhaseCounters, RealForwardError};
use crate::real_forward_utils::{entry, f16_slice_to_le_bytes};

/// One token's routed MoE half, plus its shared expert and the post-FFN
/// residual add.
///
/// `slot` is the sequential path's `RoutedSlot::sequential()` on decode
/// (token 0, bank 0, nothing protected), which is what keeps decode's bytes
/// unchanged. The chunked prefill driver passes the token's row inside the
/// micro-batch, which selects the `moe_x`, `router_logits_f32` and
/// `scratch.x` rows, and the bank for the two resources the HOST writes per
/// token (`routing_w` and the routed argument buffer), matching
/// `families/qwen4/moe.rs`.
///
/// Returns the shared-expert overlap pass (when that seam is on), which the
/// caller must keep until its final wait, and the cache slots this pass
/// bound, which the chunked driver hands to the next token as
/// `RoutedSlot::protect` while this command buffer is in flight.
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
) -> Result<(Option<gpu::CommittedPass>, Vec<usize>), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let rw_off = (slot.bank * gpu::MAX_STREAMED_EXPERTS * 2) as u64;
    let x_row = (slot.token * hidden * 2) as u64;
    let t_router = Instant::now();
    let router_logits = gpu::read_f32_buffer_at(
        &qwen.router_logits_f32,
        slot.token * num_experts,
        num_experts,
    );
    let (selected, route_weights) =
        router_topk_gemma4(&router_logits, top_k, &qwen.per_expert_ones);
    if let Some(hist) = router_hist.as_mut() {
        hist.record(layer, &selected);
    }
    // MAPPED RESIDENCY SKIPS BOTH THE PLAN AND THE `pread`, exactly as
    // `families/gemma4/moe.rs` does: every expert is already addressable in
    // its layer's mapping, so there is no cache to consult and nothing to
    // copy. Counted as requests that all hit, so `TURBOSPARK_PHASES`'s hit
    // rate stays comparable across residency modes rather than reading
    // undefined for a run with zero requests.
    let mapped_active = mapped.buffers.get(layer).is_some_and(Option::is_some);
    let routing_nanos = t_router.elapsed().as_nanos() as u64;
    // Submit independent shared work before blocking reads. The same queue
    // orders it before the routed reduction, and the caller retains the
    // draining handle so errors cannot release no-copy backing in flight.
    let shared_submission = if qwen.shared_read_overlap && !mapped_active {
        let shared_pass = context.begin_pass_labeled("qwen shared expert (read overlap)");
        encode_shared_expert(
            context,
            &shared_pass,
            weights,
            index,
            scratch,
            qwen,
            x_row,
            layer,
            hidden,
            inter,
            use_silu,
        )?;
        phases.qwen_shared_submissions += 1;
        Some(shared_pass.commit().waiting_on_drop())
    } else {
        None
    };
    if mapped_active {
        if let Some(mode) = qwen.mapped_demand_preparation {
            let mapping = mapped.layers[layer]
                .as_ref()
                .expect("mapped residency checked above");
            let stats = mapping
                .prepare_selected_experts(&selected, mode, 16 * 1024 * 1024)
                .map_err(|e| {
                    RealForwardError::Unsupported(format!("mapped demand preparation: {e}"))
                })?;
            phases.mapped_prepare_calls += 1;
            phases.mapped_prepared_bytes += stats.bytes_prepared;
            phases.mapped_page_touches += stats.page_touches;
            phases.mapped_advice_calls += stats.advisory_ranges;
            phases.mapped_advice_failures += stats.advisory_errors;
            phases.mapped_prepare_nanos += (stats.elapsed_ms * 1e6) as u64;
        }
    }
    let slots: Vec<usize> = if mapped_active {
        phases.expert_requests += selected.len() as u64;
        phases.expert_hits += selected.len() as u64;
        phases.router_nanos += routing_nanos;
        selected.clone()
    } else {
        let streamer = streamers[layer].as_mut().ok_or_else(|| {
            RealForwardError::Unsupported(format!(
                "real Qwen 3.6 layer {layer} has no packed-expert streamer"
            ))
        })?;
        let t_plan = Instant::now();
        // `slot.protect` is empty on the sequential decode path, so this is
        // the same plan decode always made there.
        let plan = streamer.plan_experts_cached(&selected, &slot.protect);
        let (requests, hits) = (plan.experts.len() as u64, plan.hits as u64);
        phases.expert_requests += requests;
        phases.expert_hits += hits;
        phases.router_nanos += routing_nanos + t_plan.elapsed().as_nanos() as u64;

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
    let order: Vec<usize> = (0..selected.len()).collect();
    let ordered: Vec<(usize, f32)> = order
        .iter()
        .map(|&i| (slots[i], route_weights[i]))
        .collect();
    let mut routing16 = vec![f16::from_f32(0.0); gpu::MAX_STREAMED_EXPERTS];
    for (slot, &(_, weight)) in ordered.iter().enumerate() {
        routing16[slot] = f16::from_f32(weight);
    }
    gpu::write_buffer_bytes(
        &scratch.routing_w,
        rw_off as usize,
        &f16_slice_to_le_bytes(&routing16),
    );

    // The two arms differ only in WHERE a blob lives, never in the order the
    // slots are dispatched: that stays the router's own ranking in both,
    // which is what keeps output byte-identical across residency modes
    // (AGENTS.md Gotcha 27), matching `families/gemma4/moe.rs`.
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
    // Selected by bank, matching `families/qwen4/moe.rs`: a pipelined bank
    // rebinds its OWN argument-buffer object rather than the one a still
    // in-flight command buffer's dispatches may be reading.
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

    if shared_submission.is_none() {
        encode_shared_expert(
            context, pass, weights, index, scratch, qwen, x_row, layer, hidden, inter, use_silu,
        )?;
    }

    encode_moe_phase1_any(
        routed_layouts[layer],
        context,
        pass,
        routed,
        offsets,
        (&qwen.moe_x, x_row),
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
        (&scratch.routing_w, rw_off),
        (&qwen.h1, 0),
        (&qwen.h2, 0),
        hidden as u32,
        moe_inter,
        top_k as u32,
        use_silu,
    )
    .map_err(gpu_err)?;
    gpu::encode_residual_add(
        context,
        pass,
        (&scratch.x, x_row),
        (&qwen.h2, 0),
        hidden as u32,
    )
    .map_err(gpu_err)?;

    Ok((
        shared_submission,
        ordered.iter().map(|&(slot, _)| slot).collect(),
    ))
}

/// The INT8 router GEMV for one token: `moe_x`'s row `x_row` in, f32
/// logits out at `router_logits_f32`'s row `logits_row`. Shared by the
/// sequential flow (row 0, row 0) and the MoE chunked driver (row t, row t),
/// so the two call sites cannot drift apart.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_router(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    qwen: &RealQwenState,
    layer: usize,
    hidden: usize,
    num_experts: usize,
    x_row: u64,
    logits_row: u64,
) -> Result<(), RealForwardError> {
    let base = index.header.index_size;
    let router_name = layer_tensor(layer, "mlp.gate.weight");
    let router = entry(index, &router_name)?;
    if router.dtype != 5 || router.size_bytes as usize != num_experts * hidden {
        return Err(RealForwardError::Unsupported(format!(
            "{router_name}: expected INT8 (dtype 5) {num_experts}x{hidden}, got dtype {} \
             with {} packed bytes",
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
        (&qwen.moe_x, x_row),
        (&qwen.router_ones, 0),
        (&qwen.router_logits_f32, logits_row),
        num_experts as u32,
        hidden as u32,
    )
    .map_err(RealForwardError::Gpu)
}

/// Keep identical shared arithmetic in the inline and overlapping arms.
#[allow(clippy::too_many_arguments)]
fn encode_shared_expert(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    scratch: &DecodeScratch,
    qwen: &RealQwenState,
    x_row: u64,
    layer: usize,
    hidden: usize,
    inter: usize,
    use_silu: bool,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &layer_tensor(layer, "mlp.shared_expert.gate_proj.weight"),
        inter,
        hidden,
        (&qwen.moe_x, x_row),
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
        (&qwen.moe_x, x_row),
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
        (&qwen.moe_x, x_row),
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

    Ok(())
}
