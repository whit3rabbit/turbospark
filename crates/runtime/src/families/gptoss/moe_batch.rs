//! The batched routed half of a `gpt-oss` prefill micro-batch layer
//! (`docs/BATCHED_PREFILL.md` step 5): ONE expert-cache plan and `pread`
//! round per union-bounded sub-batch, the MXFP4 route-list kernels from
//! `gpu::moe_prefill_batch_gguf`, and the per-token residual add unchanged.
//!
//! This is `families/gemma4/moe_batch.rs`'s shape with this family's two
//! subtractions and one addition, and every one of them is already in the
//! sequential flow rather than new here:
//!
//!  - **no shared expert**, so there is no branch to encode up front and
//!    phase 2's accumulator seed is `batch_zero` rather than a shared
//!    output (`families/gptoss/moe.rs`'s own note: passing `x` there would
//!    add the residual twice);
//!  - **no sandwich norms**, so the tail is ONE raw residual add per token
//!    rather than Gemma's norm/add/norm/add/scalar-mul;
//!  - **the ROUTER BIAS is added to each token's read-back logits before
//!    its top-k**, which is `moe.rs`'s first six lines and the reason that
//!    file is written out rather than shared with `families/llama/`.
//!    Adding it after the top-k selects the wrong experts and still reads
//!    fluently.
//!
//! Byte-identity with the per-token path is structural rather than hoped
//! for: the batched MXFP4 kernels are bit-exact against M sequential
//! decode-pair calls (`crates/gpu/tests/moe_prefill_batch_gguf_parity.rs`),
//! and the tail is the same dispatch at row offsets.
//!
//! **This family reaches M=16 where Gemma 4 and `qwen3moe` cap at M=8**,
//! which is why step 5's MXFP4 arm was built before its Q4_K/Q6_K one: 32
//! experts at top-4 keep the measured union under the slot count at every
//! M, so `MAX_PREFILL_BATCH` is the binding constraint rather than
//! `union(M) <= slot_count`. That the greedy shrink below essentially never
//! fires is MEASURED and not inferred from the mean union: on the real 20B
//! install the batched arm makes 75,762 plan requests against the 73,478
//! full-width sub-batches predict, a 3.1% gap. The shrink stays because the
//! bound is a property of the ROUTING rather than of the architecture, and a
//! layer whose routing is unusually scattered must still not trip
//! `ExpertCache::plan`, which aborts rather than degrades.

use std::collections::HashSet;
use std::time::Instant;

use half::f16;

use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::router_topk_gemma4;
use crate::real_forward_layout::RoutedBlobLayout;
use crate::real_forward_types::RealForwardError;
use crate::real_forward_utils::f16_slice_to_le_bytes;
use crate::resid_capture::encode_resid_capture;
use crate::steering::encode_steering;

impl RealForwardRunner {
    /// One layer's routed half for all `m` tokens of the micro-batch: the
    /// whole micro-batch's router readback and top-k, then one routed
    /// command buffer per union-bounded sub-batch carrying its route-list
    /// dispatch pair and its tokens' residual adds. Returns the last
    /// committed buffer for the driver to retire after the next layer's
    /// router wait, exactly as the per-token path does.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_gpt_oss_layer_routed_moe_batched(
        &mut self,
        layer: usize,
        hidden: usize,
        moe_inter: u32,
        num_experts: usize,
        top_k: usize,
        m: usize,
    ) -> Result<gpu::CommittedPass, RealForwardError> {
        let gpu_err = RealForwardError::Gpu;
        let layout = self.routed_layouts[layer];
        if layout.phase1 != RoutedBlobLayout::GgufMxfp4
            || layout.phase2 != RoutedBlobLayout::GgufMxfp4
        {
            // Refused rather than looped, for the reason the affine sibling
            // gives: a caller that asked for the batched half and quietly
            // got the per-token kernels would measure the unbatched engine
            // and report it as batched.
            return Err(RealForwardError::Unsupported(format!(
                "batched routed prefill for this family is wired for MXFP4 blobs only; \
                 layer {layer} phase1 {:?} phase2 {:?}",
                layout.phase1, layout.phase2
            )));
        }
        if self.slot_buffers[layer].len() > gpu::MAX_PREFILL_EXPERT_BINDINGS {
            return Err(RealForwardError::Unsupported(format!(
                "batched routed prefill needs slot indices below {}; this install has {} slots",
                gpu::MAX_PREFILL_EXPERT_BINDINGS,
                self.slot_buffers[layer].len()
            )));
        }

        // Router readback for the WHOLE micro-batch at once, then the bias
        // and the host top-k per token. The readback is the per-layer
        // blocking wait the chunk driver exists to amortize, so pay it once.
        let t_router = Instant::now();
        let router_logits = {
            let state = self
                .real_gpt_oss
                .as_ref()
                .expect("real gpt-oss state present");
            gpu::read_f32_buffer_at(&state.router_logits_f32, 0, m * num_experts)
        };
        let mut selected_all = Vec::with_capacity(m);
        let mut weights_all = Vec::with_capacity(m);
        for t in 0..m {
            let state = self
                .real_gpt_oss
                .as_ref()
                .expect("real gpt-oss state present");
            let mut row = router_logits[t * num_experts..(t + 1) * num_experts].to_vec();
            // THE BIAS, BEFORE THE TOP-K -- `moe.rs`'s placement, per token.
            for (logit, bias) in row.iter_mut().zip(&state.router_bias[layer]) {
                *logit += bias;
            }
            let (selected, route_weights) = router_topk_gemma4(&row, top_k, &state.per_expert_ones);
            if let Some(hist) = self.router_hist.as_mut() {
                hist.record(layer, &selected);
            }
            selected_all.push(selected);
            weights_all.push(route_weights);
        }
        self.phases.router_nanos += t_router.elapsed().as_nanos() as u64;

        // The wide argument buffer is bound ONCE per layer with every cache
        // slot: sub-batches share the layer's slot cache, and the only bind
        // that could race an in-flight reader is the NEXT layer's, which
        // happens after this layer's last sub-batch is retired.
        let blob_refs: Vec<(gpu::MetalBuffer, u64)> = self.slot_buffers[layer]
            .iter()
            .map(|b| (b.clone(), 0u64))
            .collect();
        let wide = self
            .real_gpt_oss
            .as_ref()
            .expect("real gpt-oss state present")
            .batched()
            .wide_blobs
            .clone();
        let t_bind = Instant::now();
        let bind_refs: Vec<(&gpu::MetalBuffer, u64)> =
            blob_refs.iter().map(|(b, o)| (b, *o)).collect();
        gpu::bind_routed_blobs_wide_mxfp4(&wide, &mut self.context, true, &bind_refs)
            .map_err(gpu_err)?;
        self.phases.bind_nanos += t_bind.elapsed().as_nanos() as u64;

        // Sub-batches: greedily as many tokens as the union of their routes
        // fits the slot cache. EACH COMMITS ITS OWN COMMAND BUFFER and the
        // next one's plan WAITS for it first -- a later sub-batch's `pread`
        // evicts experts from slots an earlier sub-batch's dispatches still
        // name, and those `pread`s are host-side work the queue knows
        // nothing about, so committing alone does not order them. The LAST
        // sub-batch stays in flight for the driver to retire.
        let cap = self
            .expert_cache_slots
            .min(gpu::MAX_PREFILL_EXPERT_BINDINGS);
        let moe_inter_us = moe_inter as usize;
        let mut committed: Option<gpu::CommittedPass> = None;
        let mut sub_start = 0usize;
        while sub_start < m {
            self.retire_routed(&mut committed);
            let mut seen: HashSet<usize> = selected_all[sub_start].iter().copied().collect();
            if seen.len() > cap {
                return Err(RealForwardError::Unsupported(format!(
                    "layer {layer}: one token routes {top_k} experts against {cap} slots"
                )));
            }
            let mut sub_len = 1usize;
            while sub_start + sub_len < m {
                let new = selected_all[sub_start + sub_len]
                    .iter()
                    .filter(|e| !seen.contains(*e))
                    .count();
                if seen.len() + new > cap {
                    break;
                }
                seen.extend(selected_all[sub_start + sub_len].iter().copied());
                sub_len += 1;
            }

            // One plan and one pread round for the sub-batch's union, in
            // first-seen order. The plan order cannot move bytes: the fused
            // phase 2 reduces over RANKS per token, never over slots, so
            // cache-slot assignment is numerically invisible here -- a
            // stronger property than the decode kernel has.
            let mut union: Vec<usize> = Vec::with_capacity(seen.len());
            let mut in_union = HashSet::new();
            for selected in &selected_all[sub_start..sub_start + sub_len] {
                for &e in selected {
                    if in_union.insert(e) {
                        union.push(e);
                    }
                }
            }
            let t_io = Instant::now();
            let streamer = self.streamers[layer].as_mut().ok_or_else(|| {
                RealForwardError::Unsupported(format!(
                    "gpt-oss layer {layer} has no packed-expert streamer"
                ))
            })?;
            let plan = streamer.plan_experts_cached(&union, &HashSet::new());
            self.phases.expert_requests += plan.experts.len() as u64;
            self.phases.expert_hits += plan.hits as u64;
            let streamer = self.streamers[layer]
                .as_mut()
                .expect("streamer presence checked above");
            let slots = streamer
                .execute_expert_cache_plan(&plan)
                .map_err(|e| RealForwardError::Unsupported(format!("expert stream: {e}")))?;
            self.phases.expert_io_nanos += t_io.elapsed().as_nanos() as u64;
            let slot_of: std::collections::HashMap<usize, usize> =
                union.iter().copied().zip(slots).collect();

            // Routes stay in PAIR order (token-major, rank within): the
            // fused phase 2 looks its routes up BY PAIR, so a blob-locality
            // sort would break it.
            let mut routes = Vec::with_capacity(sub_len * top_k);
            let mut routing16 = Vec::with_capacity(sub_len * top_k);
            for i in 0..sub_len {
                for r in 0..top_k {
                    let expert = selected_all[sub_start + i][r];
                    routes.push(gpu::MoePrefillRoute {
                        token: i as u32,
                        rank: r as u32,
                        slot: *slot_of
                            .get(&expert)
                            .expect("plan assigned every union expert a slot")
                            as u32,
                    });
                    routing16.push(f16::from_f32(weights_all[sub_start + i][r]));
                }
            }
            let state = self
                .real_gpt_oss
                .as_ref()
                .expect("real gpt-oss state present");
            let x_off = (sub_start * hidden * 2) as u64;
            let acts_off = (sub_start * top_k * moe_inter_us * 2) as u64;
            let rw_off = (sub_start * top_k * 2) as u64;
            let routes_off = (sub_start * top_k * 16) as u64;
            gpu::write_buffer_bytes(
                &state.batched().batch_routing_w,
                rw_off as usize,
                &f16_slice_to_le_bytes(&routing16),
            );
            gpu::write_buffer_bytes(
                &state.batched().batch_routes,
                routes_off as usize,
                &gpu::MoePrefillRoute::bytes(&routes),
            );

            let offsets = &self.moe_offsets[layer];
            let pass = self
                .context
                .begin_pass_labeled("gpt-oss routed cb (batched)");
            for (blob, _) in &blob_refs {
                pass.use_read_buffer(blob);
            }
            // `has_bias` derived exactly as `encode_moe_phase{1,2}_any`
            // derives it -- from the OFFSETS rather than from the family, so
            // a blob without bias planes takes the same path here as there.
            gpu::encode_moe_prefill_phase1_mxfp4(
                &mut self.context,
                &pass,
                &wide,
                offsets,
                (&state.moe_x, x_off),
                (&state.batched().batch_acts, acts_off),
                (&state.batched().batch_routes, routes_off),
                hidden as u32,
                moe_inter,
                top_k as u32,
                (sub_len * top_k) as u32,
                true,
                gpu::Mxfp4Activation {
                    has_bias: offsets.gate_b != 0,
                    ..gpu::Mxfp4Activation::GPT_OSS
                },
            )
            .map_err(gpu_err)?;
            gpu::encode_moe_prefill_phase2_fused_mxfp4(
                &mut self.context,
                &pass,
                &wide,
                offsets,
                (&state.batched().batch_acts, acts_off),
                (&state.batched().batch_routing_w, rw_off),
                (&state.batched().batch_routes, routes_off),
                // Seed from ZEROS: no shared expert, so the routed sum
                // starts from nothing and reaches the stream through the
                // residual add below -- which is what the decode path's
                // `zero_hidden` argument says too.
                (&state.batched().batch_zero, x_off),
                (&state.batched().batch_y, x_off),
                hidden as u32,
                moe_inter,
                top_k as u32,
                sub_len as u32,
                true,
                offsets.down_b != 0,
            )
            .map_err(gpu_err)?;

            // The tail, per token, on the same command buffer after the
            // fused phase 2: ONE raw residual add, this family having no
            // sandwich norms.
            for i in 0..sub_len {
                let row_off = x_off + (i * hidden * 2) as u64;
                let x_row = ((sub_start + i) * hidden * 2) as u64;
                gpu::encode_residual_add(
                    &mut self.context,
                    &pass,
                    (&self.scratch.x, x_row),
                    (&state.batched().batch_y, row_off),
                    hidden as u32,
                )
                .map_err(gpu_err)?;
                // This row's own offset (Gotcha 21): `scratch.x` holds every
                // token of the whole micro-batch, each at its own slot, so
                // steering and capture must operate on `x_row` and not row 0.
                // Dispatched for every row so every prompt token is steered;
                // capture's destination is one fixed region per layer and
                // sub-batches commit in increasing `sub_start` order, so the
                // globally LAST row processed is what `record_pass` reads.
                encode_steering(
                    &mut self.context,
                    &pass,
                    &self.scratch,
                    self.steering.as_ref(),
                    layer,
                    hidden,
                    1,
                    x_row,
                )?;
                encode_resid_capture(
                    &mut self.context,
                    &pass,
                    &self.scratch,
                    self.resid_capture.as_ref(),
                    layer,
                    hidden,
                    x_row,
                )?;
            }

            sub_start += sub_len;
            committed = Some(pass.commit());
        }

        Ok(committed.expect("m >= 1 guarantees at least one sub-batch"))
    }
}
