//! Flint's FFN half for a MoE layer: the ungated shared expert, the INT8
//! router, a SIGMOID top-k renormalised over the selected experts, and the
//! routed sum added on top of the shared output through phase 2's residual
//! seed.
//!
//! Structurally `families/deepseek2/moe.rs` (shared SwiGLU seeding phase 2,
//! blob position `layer - lead`). The one function that is Flint's own is
//! [`sigmoid_topk_renorm`], because neither of the existing host top-ks is
//! this router: the llama flow softmaxes over the SELECTED logits, and
//! deepseek2 softmaxes over ALL of them without renormalising.
//!
//! Slot order is the router's ranking (AGENTS.md Gotcha 27): phase 2 reduces
//! in slot-index order and FP addition is not associative.

use half::f16;

use crate::families::flint::{layer_tensor, RealFlintState};
use crate::moe_prefill_pipeline::RoutedSlot;
use crate::real_forward_dispatch::{encode_gemv_any, encode_moe_phase1_any, encode_moe_phase2_any};
use crate::real_forward_init::MappedResidency;
use crate::real_forward_layout::RoutedLayerLayout;
use crate::real_forward_types::{DecodeScratch, PhaseCounters, RealForwardError};
use crate::real_forward_utils::f16_slice_to_le_bytes;

/// `scoring_func: sigmoid`, `norm_topk_prob: true`: score every expert
/// `s_i = sigmoid(logit_i)`, take the top `k`, and weight each by
/// `s_i / sum(s over the selected)`, times `scale` (`routed_scaling_factor`,
/// 1.0 on the published checkpoint).
///
/// **Selection ranks by LOGIT, not by score.** Sigmoid is monotonic, so the
/// order is identical in exact arithmetic, but in f32 a logit above ~17
/// saturates to exactly 1.0 and two different logits would tie, falling back
/// to index order. Ranking by logit keeps the router's own order, which is
/// the slot order phase 2 reduces in. Ties between equal logits keep the
/// lower index first (a stable sort), so the result is deterministic.
pub(crate) fn sigmoid_topk_renorm(logits: &[f32], k: usize, scale: f32) -> (Vec<usize>, Vec<f32>) {
    let mut ranked: Vec<usize> = (0..logits.len()).collect();
    ranked.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked.truncate(k);
    let scores: Vec<f32> = ranked
        .iter()
        .map(|&i| 1.0 / (1.0 + (-logits[i]).exp()))
        .collect();
    let sum: f32 = scores.iter().sum();
    let weights = scores.iter().map(|s| s / sum * scale).collect();
    (ranked, weights)
}

/// The dense SwiGLU `down(silu(gate(x)) * up(x))` at `inter`, from `moe_x`
/// into `out`. Used by both the dense lead (width 6144, output `scratch.o`)
/// and the shared expert (width 1024, output `state.shared_out`): the two
/// differ only in tensor names and width, and never share a layer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_swiglu(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &model_io::ResidentIndex,
    state: &RealFlintState,
    prefix: &str,
    layer: usize,
    hidden: usize,
    inter: usize,
    out: (&gpu::MetalBuffer, u64),
) -> Result<(), RealForwardError> {
    let name = |suffix: &str| layer_tensor(layer, &format!("{prefix}.{suffix}"));
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
    .map_err(RealForwardError::Gpu)?;
    encode_gemv_any(
        context,
        pass,
        weights,
        index,
        &name("down_proj.weight"),
        hidden,
        inter,
        (&state.ffn_a, 0),
        out,
    )
}

/// One MoE layer's FFN half: shared expert, host top-k, routed phase 1/2 with
/// the shared output as phase 2's residual seed, then the stream add. Blob
/// positions are `layer - state.lead` throughout.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_flint_layer_moe(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &model_io::ResidentIndex,
    scratch: &DecodeScratch,
    state: &RealFlintState,
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
    routed_scale: f32,
    slot: &RoutedSlot,
) -> Result<Vec<usize>, RealForwardError> {
    let x_off = (slot.token * hidden * 2) as u64;
    let rw_off = slot.bank * gpu::MAX_STREAMED_EXPERTS * 2;
    let blob_layer = layer - state.lead;

    // The shared expert runs FIRST on the routed pass: phase 2's residual
    // seed must already hold its output when the routed sum lands. It is
    // UNGATED (no `shared_expert_gate` tensor exists) and summed raw.
    encode_swiglu(
        context,
        pass,
        weights,
        index,
        state,
        "mlp.shared_expert",
        layer,
        hidden,
        state.shared_inter,
        (&state.shared_out, 0),
    )?;

    let router_logits = gpu::read_f32_buffer_at(
        &state.router_logits_f32,
        slot.token * num_experts,
        num_experts,
    );
    let (selected, route_weights) = sigmoid_topk_renorm(&router_logits, top_k, routed_scale);
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
                "flint layer {layer} has no packed-expert streamer"
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
        routed_layouts[blob_layer],
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
    // Phase 2's residual input is the SHARED EXPERT's output, so
    // `y = shared + routed sum` lands in h2 in one kernel and the add below
    // is the only place the stream grows.
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

#[cfg(test)]
mod tests {
    use super::sigmoid_topk_renorm;

    #[test]
    fn weights_are_sigmoid_scores_renormalised_over_the_selected() {
        let logits = [0.0f32, 2.0, -1.0, 1.0, 3.0, -4.0];
        let (sel, w) = sigmoid_topk_renorm(&logits, 3, 1.0);
        assert_eq!(sel, vec![4, 1, 3]);
        let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
        let denom = sig(3.0) + sig(2.0) + sig(1.0);
        for (got, want) in w.iter().zip([sig(3.0), sig(2.0), sig(1.0)]) {
            assert!((got - want / denom).abs() < 1e-6, "{got} vs {}", want / denom);
        }
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    /// NOT softmax-over-selected: that function weights by exp(logit), this
    /// one by sigmoid(logit), and the two disagree on every non-degenerate
    /// input. A flow wired to the wrong one is fluent and wrong.
    #[test]
    fn it_is_not_the_llama_flows_softmax_over_the_selected() {
        let logits = [3.0f32, 2.0, 1.0];
        let (_, w) = sigmoid_topk_renorm(&logits, 3, 1.0);
        let denom: f32 = logits.iter().map(|l| l.exp()).sum();
        let softmax0 = 3.0f32.exp() / denom;
        assert!((w[0] - softmax0).abs() > 0.01, "{} vs {softmax0}", w[0]);
    }

    /// Past ~17 an f32 sigmoid is exactly 1.0, so ranking by score would tie
    /// two different logits and fall back to index order.
    #[test]
    fn selection_survives_sigmoid_saturation() {
        let logits = [30.0f32, 40.0, 25.0, 0.0];
        let (sel, _) = sigmoid_topk_renorm(&logits, 3, 1.0);
        assert_eq!(sel, vec![1, 0, 2]);
    }

    #[test]
    fn scale_multiplies_after_renormalisation() {
        let (_, w) = sigmoid_topk_renorm(&[1.0, 2.0], 2, 2.5);
        assert!((w.iter().sum::<f32>() - 2.5).abs() < 1e-5);
    }

    #[test]
    fn equal_logits_keep_index_order() {
        let (sel, _) = sigmoid_topk_renorm(&[1.0, 1.0, 1.0, 0.0], 2, 1.0);
        assert_eq!(sel, vec![0, 1]);
    }
}
