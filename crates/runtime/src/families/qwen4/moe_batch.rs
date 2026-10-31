//! The batched routed half of one `qwen4_exp` layer for a chunked-prefill
//! micro-batch (`TURBOSPARK_ROUTED_BATCH`, `docs/BATCHED_PREFILL.md`):
//! the M-row sibling of `moe::encode_moe_layer`, and
//! `families/qwen/moe_batch.rs`'s `encode_qwen_layer_routed_prefill` with
//! three differences, each forced by this family:
//!
//! - **Top-10, past the eight-wide reduce.** The pair is
//!   `encode_moe_prefill_phase1_wide` / `encode_moe_prefill_phase2_fused_wide`,
//!   whose phase 2 reduces exactly `top_k` ranks after the seed, the decode
//!   kernel's loop. The eight-wide pair asserts on top-10 rather than
//!   truncating.
//! - **The shared expert joins AFTER the routed sum is rounded.** Decode
//!   seeds phase 2 with `moe_zero` and adds the gated shared expert to the
//!   FP16 result (`moe.rs`'s header). So phase 2 here seeds from zero rows,
//!   and each token's shared branch is added to its `y` row afterwards, the
//!   same two FP16 operands in the same addition. Seeding with the shared
//!   expert instead, as `qwen3_5` does, would be a different model.
//! - **No residual add; a hyper-connection inject instead.** Each token's
//!   `y` row goes into its `wide_x` row through `encode_hc_inject_add` with
//!   its own `hc_inject` row, which `cb1` wrote.
//!
//! Each sub-batch binds only its own expert union (at most
//! [`gpu::MAX_PREFILL_EXPERT_BINDINGS`]), so the CLI's `auto` slot count is
//! not refused. At top-10 that cap keeps at least three tokens per
//! sub-batch, more when routes overlap.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use model_io::ResidentIndex;

use crate::families::qwen4::layer_tensor;
use crate::families::qwen4::state::RealQwen4State;
use crate::moe_prefill_pipeline::{
    encode_routes, next_routed_sub_batch, plan_and_stream_union, try_plan_and_stream_union,
};
use crate::real_forward_dispatch::{encode_gemv_any, router_topk_gemma4};
use crate::real_forward_init::MappedResidency;
use crate::real_forward_layout::{RoutedBlobLayout, RoutedLayerLayout};
use crate::real_forward_types::{
    DecodeScratch, PhaseCounters, RealForwardError, MAX_PREFILL_BATCH,
};
use crate::real_forward_utils::f16_slice_to_le_bytes;

/// The batched pair's own buffers, allocated on the first batched layer
/// rather than at open: an install that never sets the seam pays nothing.
pub(crate) struct RoutedBatchScratch {
    /// `[M * top_k, moe_inter]` FP16, phase 1's output over the route list.
    acts: gpu::MetalBuffer,
    /// `[M, hidden]` FP16, phase 2's output, then routed plus shared.
    y: gpu::MetalBuffer,
    /// `[M, hidden]` FP16 zeros, phase 2's seed (`moe_zero`, M rows).
    zeros: gpu::MetalBuffer,
    /// `[M * top_k]` FP16 routing weights in pair order.
    routing_w: gpu::MetalBuffer,
    /// The encoded route list, `MoePrefillRoute::STRIDE_BYTES` per route.
    routes: gpu::MetalBuffer,
    wide_blobs: gpu::RoutedBlobsWideBuffer,
    /// The read-ahead's second argument buffer
    /// (`TURBOSPARK_ROUTED_BATCH_OVERLAP`).
    wide_blobs_alt: gpu::RoutedBlobsWideBuffer,
}

impl RoutedBatchScratch {
    pub(crate) fn new(
        context: &mut gpu::MetalContext,
        hidden: usize,
        moe_inter: usize,
        top_k: usize,
        use_silu: bool,
    ) -> Result<Self, gpu::GpuError> {
        let wide_blobs = gpu::RoutedBlobsWideBuffer::new(context, use_silu)?;
        let wide_blobs_alt = gpu::RoutedBlobsWideBuffer::new(context, use_silu)?;
        let m = MAX_PREFILL_BATCH;
        let zeros = context.new_output_buffer((m * hidden * 2) as u64);
        gpu::write_buffer_bytes(&zeros, 0, &vec![0u8; m * hidden * 2]);
        Ok(Self {
            acts: context.new_output_buffer((m * top_k * moe_inter * 2) as u64),
            y: context.new_output_buffer((m * hidden * 2) as u64),
            zeros,
            routing_w: context.new_output_buffer((m * top_k * 2) as u64),
            routes: context
                .new_output_buffer((m * top_k * gpu::MoePrefillRoute::STRIDE_BYTES) as u64),
            wide_blobs,
            wide_blobs_alt,
        })
    }
}

/// One layer's routed half for all `batch` rows of a micro-batch. The
/// caller has committed and waited on the layer's `cb1`, so every row of
/// `qwen4.router_logits_f32`, `qwen4.moe_x` and `qwen4.hc_inject` is
/// readable, and has retired the previous layer's last routed buffer.
/// Returns the last committed routed buffer for the caller to retire after
/// the next layer's router wait, as the per-token loop does with its last
/// token.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qwen4_layer_routed_prefill(
    context: &mut gpu::MetalContext,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    scratch: &DecodeScratch,
    qwen4: &RealQwen4State,
    batch_scratch: &RoutedBatchScratch,
    streamers: &mut [Option<streaming::PreadExpertStreamer>],
    slot_buffers: &[Vec<gpu::MetalBuffer>],
    mapped: &MappedResidency,
    moe_offsets: &[gpu::MoeExpertOffsets],
    routed_layouts: &[RoutedLayerLayout],
    router_hist: &mut Option<crate::router_hist::RouterHistogram>,
    phases: &mut PhaseCounters,
    expert_cache_slots: usize,
    overlap: bool,
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
    let layout = routed_layouts[layer];
    if layout.phase1 != RoutedBlobLayout::Affine || layout.phase2 != RoutedBlobLayout::Affine {
        return Err(RealForwardError::Unsupported(format!(
            "TURBOSPARK_ROUTED_BATCH is wired for INT4-affine routed experts only on the \
             qwen4_exp chunked driver; layer {layer} phase1 {:?} phase2 {:?}",
            layout.phase1, layout.phase2
        )));
    }
    if top_k > gpu::MAX_STREAMED_EXPERTS {
        return Err(RealForwardError::Unsupported(format!(
            "TURBOSPARK_ROUTED_BATCH: top-{top_k} is past the batched pair's {} ranks",
            gpu::MAX_STREAMED_EXPERTS
        )));
    }
    // Mapped residency has no slot cache to bind from. Refused here, not
    // at open, because `set_routed_batch_prefill` can flip the seam later.
    if mapped.buffers.get(layer).is_some_and(Option::is_some) {
        return Err(RealForwardError::Unsupported(format!(
            "TURBOSPARK_ROUTED_BATCH and mapped expert residency cannot be combined: the \
             batched routed pair binds cache slots and mapped residency has none (layer {layer})"
        )));
    }
    if router_hist
        .as_ref()
        .is_some_and(crate::router_hist::RouterHistogram::pilot_enabled)
    {
        return Err(RealForwardError::Unsupported(format!(
            "TURBOSPARK_ROUTED_BATCH and the router pilot probe (TURBOSPARK_PILOT_PROBE) \
             cannot be combined (layer {layer})"
        )));
    }

    // One readback for the whole micro-batch, then the per-token loop's
    // top-k, weights and histogram records, in the same token order.
    let t_router = Instant::now();
    let router_logits = gpu::read_f32_buffer_at(&qwen4.router_logits_f32, 0, batch * num_experts);
    let mut selected_all = Vec::with_capacity(batch);
    let mut weights_all = Vec::with_capacity(batch);
    for t in 0..batch {
        let row = &router_logits[t * num_experts..(t + 1) * num_experts];
        let (selected, route_weights) = router_topk_gemma4(row, top_k, &qwen4.per_expert_ones);
        if let Some(hist) = router_hist.as_mut() {
            hist.record_router_logits(layer, row);
            hist.record(layer, &selected);
        }
        selected_all.push(selected);
        weights_all.push(route_weights);
    }
    phases.router_nanos += t_router.elapsed().as_nanos() as u64;

    let cap = expert_cache_slots.min(gpu::MAX_PREFILL_EXPERT_BINDINGS);
    let moe_inter_us = moe_inter as usize;
    let hc_count = qwen4.hc_count;
    let wide_dim = hidden * hc_count;
    let offsets = &moe_offsets[layer];
    let layer_slots = &slot_buffers[layer];
    let mut committed: Option<gpu::CommittedPass> = None;
    let mut in_flight: HashSet<usize> = HashSet::new();
    let mut alt = false;
    let retire = |prev: gpu::CommittedPass, phases: &mut PhaseCounters| {
        let t_wait = Instant::now();
        phases.routed_cb_gpu_nanos += (prev.wait_with_gpu_time() * 1e9) as u64;
        phases.pipeline_wait_nanos += t_wait.elapsed().as_nanos() as u64;
    };
    let mut sub_start = 0usize;
    while sub_start < batch {
        let sub = next_routed_sub_batch(&selected_all, sub_start, cap, layer, top_k)?;
        let streamer = streamers[layer].as_mut().ok_or_else(|| {
            RealForwardError::Unsupported(format!(
                "real qwen4_exp layer {layer} has no packed-expert streamer"
            ))
        })?;
        // A `pread` is host work the queue knows nothing about: nothing this
        // sub-batch evicts may still be read by the previous one.
        let ahead = match (&committed, overlap) {
            (Some(_), true) => try_plan_and_stream_union(streamer, phases, &sub.union, &in_flight)?,
            _ => None,
        };
        let slot_of = match ahead {
            Some(slot_of) => {
                foundation::engaged!("routed_batch_overlap");
                phases.routed_read_aheads += 1;
                slot_of
            }
            None => {
                if let Some(prev) = committed.take() {
                    retire(prev, phases);
                }
                plan_and_stream_union(streamer, phases, &sub.union)?
            }
        };
        let blobs = if alt {
            &batch_scratch.wide_blobs_alt
        } else {
            &batch_scratch.wide_blobs
        };

        // Bind the union in union order; routes name binding indices. The
        // reduce runs over ranks, never bindings, so the remap moves no bit.
        let t_bind = Instant::now();
        let bound: Vec<&gpu::MetalBuffer> =
            sub.union.iter().map(|e| &layer_slots[slot_of[e]]).collect();
        let binding_of: HashMap<usize, usize> =
            sub.union.iter().enumerate().map(|(i, &e)| (e, i)).collect();
        let bind_refs: Vec<(&gpu::MetalBuffer, u64)> = bound.iter().map(|&b| (b, 0u64)).collect();
        blobs.bind(context, use_silu, &bind_refs).map_err(gpu_err)?;
        phases.bind_nanos += t_bind.elapsed().as_nanos() as u64;

        let (routes, routing16) =
            encode_routes(&selected_all, &weights_all, &sub, &binding_of, top_k);
        let sub_len = sub.sub_len;
        let x_off = (sub_start * hidden * 2) as u64;
        let acts_off = (sub_start * top_k * moe_inter_us * 2) as u64;
        let rw_off = (sub_start * top_k * 2) as u64;
        let routes_off = (sub_start * top_k * gpu::MoePrefillRoute::STRIDE_BYTES) as u64;
        gpu::write_buffer_bytes(
            &batch_scratch.routing_w,
            rw_off as usize,
            &f16_slice_to_le_bytes(&routing16),
        );
        gpu::write_buffer_bytes(
            &batch_scratch.routes,
            routes_off as usize,
            &gpu::MoePrefillRoute::bytes(&routes),
        );

        let pass = context.begin_pass_labeled("routed cb (batched prefill)");
        for &blob in &bound {
            pass.use_read_buffer(blob);
        }
        gpu::encode_moe_prefill_phase1_wide(
            context,
            &pass,
            blobs,
            offsets,
            (&qwen4.moe_x, x_off),
            (&batch_scratch.acts, acts_off),
            (&batch_scratch.routes, routes_off),
            hidden as u32,
            moe_inter,
            top_k as u32,
            (sub_len * top_k) as u32,
            use_silu,
        )
        .map_err(gpu_err)?;
        gpu::encode_moe_prefill_phase2_fused_wide(
            context,
            &pass,
            blobs,
            offsets,
            (&batch_scratch.acts, acts_off),
            (&batch_scratch.routing_w, rw_off),
            (&batch_scratch.routes, routes_off),
            // Zero seed: decode's `moe_zero`, so the routed sum is rounded
            // to FP16 before the shared expert joins.
            (&batch_scratch.zeros, x_off),
            (&batch_scratch.y, x_off),
            hidden as u32,
            moe_inter,
            top_k as u32,
            sub_len as u32,
            use_silu,
        )
        .map_err(gpu_err)?;

        // Per token: the gated shared expert through the single-row scratch
        // `encode_moe_layer` uses (dispatches on one encoder run in order),
        // joined to its routed row, then the hyper-connection inject.
        for i in 0..sub_len {
            let t = sub_start + i;
            let row = (t * hidden * 2) as u64;
            let mixed = (&qwen4.moe_x, row);
            encode_gemv_any(
                context,
                &pass,
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
                &pass,
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
                &pass,
                (&scratch.ffn_gate, 0),
                (&scratch.ffn_up, 0),
                (&scratch.ffn_act, 0),
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
                (&scratch.ffn_act, 0),
                (&qwen4.h1, 0),
            )?;
            encode_gemv_any(
                context,
                &pass,
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
                &pass,
                (&qwen4.h1, 0),
                (&qwen4.shared_gate_logit, 0),
                hidden as u32,
            )
            .map_err(gpu_err)?;
            gpu::encode_residual_add(
                context,
                &pass,
                (&batch_scratch.y, row),
                (&qwen4.h1, 0),
                hidden as u32,
            )
            .map_err(gpu_err)?;
            gpu::encode_hc_inject_add(
                context,
                &pass,
                (&qwen4.wide_x, (t * wide_dim * 2) as u64),
                (&batch_scratch.y, row),
                (&qwen4.hc_inject, (t * hc_count * 2) as u64),
                hc_count as u32,
                hidden as u32,
            )
            .map_err(gpu_err)?;
        }

        sub_start += sub_len;
        // At most one routed buffer is in flight when this returns.
        if let Some(prev) = committed.take() {
            retire(prev, phases);
        }
        in_flight = slot_of.values().copied().collect();
        alt = overlap && !alt;
        committed = Some(pass.commit());
    }

    Ok(committed.expect("batch >= 1 guarantees at least one sub-batch"))
}
