//! The batched routed half of one `qwen3_5` MoE layer: the M-row sibling
//! of `families/qwen/moe.rs` (ROADMAP Phase 3).
//!
//! It is `families/gemma4/moe_batch.rs`'s problem solved a second time,
//! and the two differ in exactly the ways the FAMILIES differ:
//!
//! - **A SHARED EXPERT, gated.** Gemma's routed sum starts from nothing
//!   and its shared branch joins in the sandwich tail after a norm; this
//!   family's phase 2 SEEDS its accumulator with the gated shared-expert
//!   output. That is why `encode_moe_prefill_phase2_fused` takes a
//!   residual at all. FP addition is not associative, so seeding is not
//!   the same function as adding the shared expert to a finished routed
//!   sum, and a driver that appended a residual add would be a different
//!   model that still reads fluently.
//! - **NO SANDWICH NORMS.** The tail here is one raw residual add.
//!   Normalizing between the routed output and the stream is what took
//!   Qwen 3.6's reference perplexity from 6.2536 to 255,409 once already
//!   (crate Gotcha 11), and it is the single easiest thing to import by
//!   accident from the file next door.
//! - **`per_expert_ones` rather than a learned per-expert scale**, which
//!   is what `router_topk_gemma4` takes here.
//!
//! Everything else is deliberately the same, including the two properties
//! the batched routed pair exists to preserve. Routes stay in PAIR order
//! (token-major, rank within) because the fused phase 2 looks them up by
//! pair. And sub-batches are bounded by the expert UNION, because
//! `ExpertCache::plan_if_possible` ASSERTS `experts.len() <= slot_count`
//! rather than degrading (AGENTS.md Gotcha 54) -- at top-8 a block of M
//! tokens can want up to `8M` experts, so M=2 wants 16 and M=4 wants 32,
//! and past that neither the slot cache nor the 32-pointer argument
//! buffer can hold the round. That ceiling agrees with the measured
//! shape on both speculative pages: small blocks are what pay on this
//! engine.

use std::time::Instant;

use model_io::ResidentIndex;

use crate::families::qwen::batched_scratch::BatchedScratch;
use crate::families::qwen::{layer_tensor, RealQwenState};
use crate::moe_prefill_pipeline::{encode_routes, next_routed_sub_batch, plan_and_stream_union};
use crate::real_forward_dispatch::{encode_gemv_any, router_topk_gemma4};
use crate::real_forward_init::MappedResidency;
use crate::real_forward_layout::{RoutedBlobLayout, RoutedLayerLayout};
use crate::real_forward_types::{DecodeScratch, PhaseCounters, RealForwardError};
use crate::real_forward_utils::f16_slice_to_le_bytes;

/// One layer's routed half for all `batch` rows.
///
/// The caller has already committed and WAITED on the pass carrying this
/// layer's attention and its M router GEMVs, so the router logits below
/// are readable. Returns the last committed routed buffer for the caller
/// to retire, exactly as the per-token path pipelines its final token.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen_layer_moe_batched(
    context: &mut gpu::MetalContext,
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
) -> Result<gpu::CommittedPass, RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let routed = batched.routed.as_ref().ok_or_else(|| {
        RealForwardError::Unsupported(
            "batched routed half on an install with no routed scratch".to_string(),
        )
    })?;
    let layout = routed_layouts[layer];
    if layout.phase1 != RoutedBlobLayout::Affine || layout.phase2 != RoutedBlobLayout::Affine {
        // Refused rather than looped: a caller that asked for the batched
        // verify and quietly got the per-token kernels would measure the
        // UNBATCHED engine and report it as batched (AGENTS.md Gotcha 35,
        // and `batched.rs`'s own header says the same of `encode_gemm_any`).
        return Err(RealForwardError::Unsupported(format!(
            "batched routed verify is wired for INT4-affine blobs only; layer {layer} \
             phase1 {:?} phase2 {:?}",
            layout.phase1, layout.phase2
        )));
    }
    // MAPPED RESIDENCY AND THE BATCHED VERIFY PASS CANNOT BOTH RUN, the same
    // structural conflict `families/gemma4/moe_batch.rs` refuses: this driver
    // binds `slot_buffers[layer]` ONCE per layer, an array indexed by CACHE
    // SLOT, and mapped residency has no slot cache at all. No published MoE
    // conversion of this architecture carries an ingestible speculative head
    // (`crates/runtime/CLAUDE.md` Gotcha 0), so a real install can never
    // reach this combination -- but a synthetic MoE+MTP fixture can, and an
    // unrefused combination would bind an empty `slot_buffers[layer]` and
    // trip a plain `assert!` several frames downstream in `gpu`'s argument
    // encoder, naming neither seam.
    if mapped.buffers.get(layer).is_some_and(Option::is_some) {
        return Err(RealForwardError::Unsupported(format!(
            "the batched verify pass and mapped expert residency \
             (TURBOSPARK_EXPERT_RESIDENCY=mapped) cannot be combined: this pass binds one \
             buffer per CACHE SLOT and mapped residency has no slot cache (layer {layer}). \
             Pick one"
        )));
    }
    if slot_buffers[layer].len() > gpu::MAX_PREFILL_EXPERT_BINDINGS {
        return Err(RealForwardError::Unsupported(format!(
            "batched routed verify needs slot indices below {}; this install has {} slots",
            gpu::MAX_PREFILL_EXPERT_BINDINGS,
            slot_buffers[layer].len()
        )));
    }
    if router_hist
        .as_ref()
        .is_some_and(crate::router_hist::RouterHistogram::pilot_enabled)
    {
        return Err(RealForwardError::Unsupported(format!(
            "batched routed verify and the router pilot probe (TURBOSPARK_PILOT_PROBE) cannot be combined (layer {layer})"
        )));
    }

    // Router readback for the WHOLE batch at once, then host top-k per
    // token. This is the per-layer blocking wait a batched pass exists to
    // pay once instead of M times.
    let t_router = Instant::now();
    let router_logits =
        gpu::read_f32_buffer_at(&routed.batch_router_logits_f32, 0, batch * num_experts);
    let mut selected_all = Vec::with_capacity(batch);
    let mut weights_all = Vec::with_capacity(batch);
    for t in 0..batch {
        let (selected, route_weights) = router_topk_gemma4(
            &router_logits[t * num_experts..(t + 1) * num_experts],
            top_k,
            &qwen.per_expert_ones,
        );
        if let Some(hist) = router_hist.as_mut() {
            hist.record(layer, &selected);
        }
        selected_all.push(selected);
        weights_all.push(route_weights);
    }
    phases.router_nanos += t_router.elapsed().as_nanos() as u64;

    // The wide argument buffer is bound ONCE per layer with every cache
    // slot: sub-batches share the layer's slot cache, and the only bind
    // that could race an in-flight reader is the NEXT layer's, which
    // happens after this layer's last sub-batch is retired.
    let blob_refs: Vec<(gpu::MetalBuffer, u64)> = slot_buffers[layer]
        .iter()
        .map(|b| (b.clone(), 0u64))
        .collect();
    let t_bind = Instant::now();
    let bind_refs: Vec<(&gpu::MetalBuffer, u64)> = blob_refs.iter().map(|(b, o)| (b, *o)).collect();
    routed
        .wide_blobs
        .bind(context, use_silu, &bind_refs)
        .map_err(gpu_err)?;
    phases.bind_nanos += t_bind.elapsed().as_nanos() as u64;

    let cap = expert_cache_slots.min(gpu::MAX_PREFILL_EXPERT_BINDINGS);
    let moe_inter_us = moe_inter as usize;
    let offsets = &moe_offsets[layer];
    let mut committed: Option<gpu::CommittedPass> = None;
    let mut sub_start = 0usize;
    while sub_start < batch {
        // Nothing this sub-batch's `pread` evicts may still be named by an
        // in-flight dispatch, and a `pread` is host work the queue knows
        // nothing about -- committing alone does not order it. So the
        // previous sub-batch is waited out before this one is planned.
        if let Some(prev) = committed.take() {
            let t_wait = Instant::now();
            prev.wait();
            phases.pipeline_wait_nanos += t_wait.elapsed().as_nanos() as u64;
        }
        let sub = next_routed_sub_batch(&selected_all, sub_start, cap, layer, top_k)?;
        let streamer = streamers[layer].as_mut().ok_or_else(|| {
            RealForwardError::Unsupported(format!("real Qwen layer {layer} has no expert streamer"))
        })?;
        let slot_of = plan_and_stream_union(streamer, phases, &sub.union)?;
        let (routes, routing16) = encode_routes(&selected_all, &weights_all, &sub, &slot_of, top_k);
        let sub_len = sub.sub_len;
        let x_off = (sub_start * hidden * 2) as u64;
        let acts_off = (sub_start * top_k * moe_inter_us * 2) as u64;
        let rw_off = (sub_start * top_k * 2) as u64;
        let routes_off = (sub_start * top_k * gpu::MoePrefillRoute::STRIDE_BYTES) as u64;
        gpu::write_buffer_bytes(
            &routed.batch_routing_w,
            rw_off as usize,
            &f16_slice_to_le_bytes(&routing16),
        );
        gpu::write_buffer_bytes(
            &routed.batch_routes,
            routes_off as usize,
            &gpu::MoePrefillRoute::bytes(&routes),
        );

        let pass = context.begin_pass_labeled("routed cb (batched verify)");
        for (blob, _) in &blob_refs {
            pass.use_read_buffer(blob);
        }

        // The GATED shared-expert branch per token, into `batch_h1`. It
        // shares this sub-batch's command buffer with the routed pair
        // below, which reads its output as phase 2's seed: dispatches on
        // one serial encoder run in encode order, so no fence is needed.
        for i in 0..sub_len {
            let t = sub_start + i;
            let x_row = (t * hidden * 2) as u64;
            let inter_row = (t * inter * 2) as u64;
            encode_gemv_any(
                context,
                &pass,
                weights,
                index,
                &layer_tensor(layer, "mlp.shared_expert.gate_proj.weight"),
                inter,
                hidden,
                (&batched.moe_x, x_row),
                (&batched.ffn_gate, inter_row),
            )?;
            encode_gemv_any(
                context,
                &pass,
                weights,
                index,
                &layer_tensor(layer, "mlp.shared_expert.up_proj.weight"),
                inter,
                hidden,
                (&batched.moe_x, x_row),
                (&batched.ffn_up, inter_row),
            )?;
            let act = if use_silu {
                gpu::encode_silu_mul
            } else {
                gpu::encode_gelu_mul
            };
            act(
                context,
                &pass,
                (&batched.ffn_gate, inter_row),
                (&batched.ffn_up, inter_row),
                (&batched.ffn_act, inter_row),
                inter as u32,
            )
            .map_err(gpu_err)?;
            encode_gemv_any(
                context,
                &pass,
                weights,
                index,
                &layer_tensor(layer, "mlp.shared_expert.down_proj.weight"),
                hidden,
                inter,
                (&batched.ffn_act, inter_row),
                (&routed.batch_h1, x_row),
            )?;
            encode_gemv_any(
                context,
                &pass,
                weights,
                index,
                &layer_tensor(layer, "mlp.shared_expert_gate.weight"),
                1,
                hidden,
                (&batched.moe_x, x_row),
                (&routed.batch_gate_logit, (t * 2) as u64),
            )?;
            gpu::encode_sigmoid_scalar_mul(
                context,
                &pass,
                (&routed.batch_h1, x_row),
                (&routed.batch_gate_logit, (t * 2) as u64),
                hidden as u32,
            )
            .map_err(gpu_err)?;
        }

        gpu::encode_moe_prefill_phase1(
            context,
            &pass,
            &routed.wide_blobs,
            offsets,
            (&batched.moe_x, x_off),
            (&routed.batch_acts, acts_off),
            (&routed.batch_routes, routes_off),
            hidden as u32,
            moe_inter,
            top_k as u32,
            (sub_len * top_k) as u32,
            use_silu,
        )
        .map_err(gpu_err)?;
        gpu::encode_moe_prefill_phase2_fused(
            context,
            &pass,
            &routed.wide_blobs,
            offsets,
            (&routed.batch_acts, acts_off),
            (&routed.batch_routing_w, rw_off),
            (&routed.batch_routes, routes_off),
            // The SEED, not a later addend: `moe_phase2_down_reduce_k8`
            // starts its accumulator at the gated shared expert and then
            // adds the eight ranks, so this is where `batch_h1` has to go
            // for the batched pass to be the same function.
            (&routed.batch_h1, x_off),
            (&routed.batch_y, x_off),
            hidden as u32,
            moe_inter,
            top_k as u32,
            sub_len as u32,
            use_silu,
        )
        .map_err(gpu_err)?;

        // The tail: ONE RAW RESIDUAL ADD per token. No norm belongs here
        // (crate Gotcha 11) -- see this module's header.
        for i in 0..sub_len {
            let row = ((sub_start + i) * hidden * 2) as u64;
            gpu::encode_residual_add(
                context,
                &pass,
                (&scratch.x, row),
                (&routed.batch_y, row),
                hidden as u32,
            )
            .map_err(gpu_err)?;
        }

        sub_start += sub_len;
        committed = Some(pass.commit());
    }

    Ok(committed.expect("batch >= 1 guarantees at least one sub-batch"))
}
