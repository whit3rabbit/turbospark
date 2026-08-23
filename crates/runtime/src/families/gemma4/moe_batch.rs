//! The batched routed half of a prefill micro-batch layer
//! (`docs/BATCHED_PREFILL.md` steps 2 and 3): ONE expert-cache plan and
//! `pread` round per union-bounded sub-batch, the route-list kernels from
//! `gpu::moe_prefill_batch`, and the per-token sandwich tail unchanged.
//!
//! Byte-identity with the per-token path is structural, not hoped for:
//! the batched kernels are bit-exact against M sequential decode-pair
//! calls (`crates/gpu/tests/moe_prefill_batch_parity.rs`), and the tail
//! kernels are the same dispatches at row offsets. A sub-batch keeps
//! every expert of its UNION resident at once, which is what caps it
//! near M=8 at 32 slots -- the constraint the design doc measured, and
//! the reason the greedy shrink below exists (`union(M) <= slot_count`,
//! or `ExpertCache::plan` aborts rather than degrades).

use std::collections::HashSet;
use std::time::Instant;

use half::f16;

use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::router_topk_gemma4;
use crate::real_forward_layout::RoutedBlobLayout;
use crate::real_forward_types::RealForwardError;
use crate::real_forward_utils::{f16_slice_to_le_bytes, layer_tensor, norm_view};

use super::moe;

const RMS_EPS: f32 = 1e-6;

impl RealForwardRunner {
    /// One layer's routed half for all `m` tokens of the micro-batch:
    /// shared-expert branches first (each its own committed command
    /// buffer, overlapping the plan and `pread` below), then one routed
    /// command buffer carrying every sub-batch's route-list dispatches
    /// and every token's sandwich tail. Returns the committed buffer for
    /// the driver to retire after the next layer's router wait, exactly
    /// as the per-token path does.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_gemma4_layer_routed_moe_batched(
        &mut self,
        layer: usize,
        hidden: usize,
        inter: usize,
        moe_inter: u32,
        num_experts: usize,
        top_k: usize,
        use_silu: bool,
        m: usize,
    ) -> Result<gpu::CommittedPass, RealForwardError> {
        let gpu_err = RealForwardError::Gpu;
        let layout = self.routed_layouts[layer];
        if layout.phase1 != RoutedBlobLayout::Affine || layout.phase2 != RoutedBlobLayout::Affine {
            // Refused rather than looped: a caller that asked for the
            // batched half and quietly got the per-token kernels would
            // measure the unbatched engine and report it as batched (the
            // `encode_gemm_any` doctrine). The GGUF pairs are step 5.
            return Err(RealForwardError::Unsupported(format!(
                "batched routed prefill is wired for INT4-affine blobs only; layer {layer} phase1 {:?} phase2 {:?}",
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

        let per_expert_scale = self
            .real
            .as_ref()
            .expect("real state present")
            .per_expert_scale[layer]
            .clone();
        let batch_h1 = self
            .real
            .as_ref()
            .expect("real state present")
            .batched()
            .batch_h1
            .clone();

        // Shared-expert branches for every token up front, so the GPU
        // drains them while the host plans and preads the routed union.
        // (`MFERENCE_SHARED_CB=0` reverts the DECODE path to inline
        // encoding; the batched path always overlaps, because the union
        // `pread` below is one large host stall to hide.)
        //
        // Under `MFERENCE_BATCHED_GEMV` all M tokens share ONE command
        // buffer and three M-row GEMMs; otherwise each rides its own
        // committed buffer with three GEMVs, which is what shipped.
        if self.batched_gemv_prefill {
            self.encode_shared_expert_branch_batched(layer, hidden, inter, use_silu, m)?;
        } else {
            for t in 0..m {
                let slot = moe::RoutedSlot {
                    token: t,
                    bank: 0,
                    protect: HashSet::new(),
                };
                self.encode_shared_expert_branch(
                    layer,
                    hidden,
                    inter,
                    use_silu,
                    &slot,
                    (&batch_h1, (t * hidden * 2) as u64),
                )?;
            }
        }

        // Router readback for the WHOLE micro-batch at once, then host
        // top-k per token -- the readback is the per-layer blocking wait
        // the chunk driver exists to amortize, so pay it once.
        let t_router = Instant::now();
        let router_logits = {
            let real = self.real.as_ref().expect("real state present");
            gpu::read_f32_buffer_at(&real.router_logits_f32, 0, m * num_experts)
        };
        let mut selected_all = Vec::with_capacity(m);
        let mut weights_all = Vec::with_capacity(m);
        for t in 0..m {
            let (selected, route_weights) = router_topk_gemma4(
                &router_logits[t * num_experts..(t + 1) * num_experts],
                top_k,
                &per_expert_scale,
            );
            if let Some(hist) = self.router_hist.as_mut() {
                hist.record(layer, &selected);
            }
            selected_all.push(selected);
            weights_all.push(route_weights);
        }
        self.phases.router_nanos += t_router.elapsed().as_nanos() as u64;

        // The wide argument buffer is bound ONCE per layer with every
        // cache slot: sub-batches share the layer's slot cache, the bind
        // is a host write that must precede the commit of any dispatch
        // reading it, and the only bind that could race an in-flight
        // reader is the NEXT layer's -- which happens after this layer's
        // last sub-batch is retired.
        let blob_refs: Vec<(gpu::MetalBuffer, u64)> = self.slot_buffers[layer]
            .iter()
            .map(|b| (b.clone(), 0u64))
            .collect();
        let wide = self
            .real
            .as_ref()
            .expect("real state present")
            .batched()
            .wide_blobs
            .clone();
        let t_bind = Instant::now();
        let bind_refs: Vec<(&gpu::MetalBuffer, u64)> =
            blob_refs.iter().map(|(b, o)| (b, *o)).collect();
        wide.bind(&mut self.context, use_silu, &bind_refs)
            .map_err(gpu_err)?;
        self.phases.bind_nanos += t_bind.elapsed().as_nanos() as u64;

        // Sub-batches: greedily as many tokens as the union of their
        // routes fits the slot cache. At 32 slots that lands near M=8 on
        // the measured union table, at 16 near M=2; a layer whose routing
        // is unusually scattered shrinks further, which is why the bound
        // is computed per layer rather than frozen.
        //
        // EACH SUB-BATCH COMMITS ITS OWN COMMAND BUFFER, and the next
        // sub-batch's plan WAITS for it first: a later sub-batch's
        // `pread` evicts experts from slots an earlier sub-batch's
        // dispatches still name, and those `pread`s are host-side work
        // the queue knows nothing about -- committing alone does not
        // order them. (One command buffer per layer was tried and the
        // 8-slot runtime test caught exactly this: fluent wrong logits,
        // because union-of-micro-batch > slots is what forces a second
        // sub-batch.) The LAST sub-batch stays in flight for the driver
        // to retire after the next layer's router wait, exactly as the
        // per-token path pipelines its final token.
        let cap = self
            .expert_cache_slots
            .min(gpu::MAX_PREFILL_EXPERT_BINDINGS);
        let moe_inter_us = moe_inter as usize;
        let mut committed: Option<gpu::CommittedPass> = None;
        let mut sub_start = 0usize;
        while sub_start < m {
            // Nothing this sub-batch's `pread` evicts may still be named
            // by an in-flight dispatch: wait out the previous sub-batch's
            // command buffer before planning this one.
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
            // first-seen order. The plan order cannot move bytes: the
            // fused phase 2 reduces over RANKS per token, never over
            // slots, so cache-slot assignment is numerically invisible
            // here -- a stronger property than the decode kernel has.
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
                    "real Gemma 4 layer {layer} has no packed-expert streamer"
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
            // fused phase 2 looks its routes up BY PAIR, so a
            // blob-locality sort would break it -- and each blob is
            // ~3.2 MB against caches far smaller, so the sort would buy
            // nothing anyway.
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
            let real = self.real.as_ref().expect("real state present");
            let x_off = (sub_start * hidden * 2) as u64;
            let acts_off = (sub_start * top_k * moe_inter_us * 2) as u64;
            let rw_off = (sub_start * top_k * 2) as u64;
            let routes_off = (sub_start * top_k * 16) as u64;
            gpu::write_buffer_bytes(
                &real.batched().batch_routing_w,
                rw_off as usize,
                &f16_slice_to_le_bytes(&routing16),
            );
            gpu::write_buffer_bytes(
                &real.batched().batch_routes,
                routes_off as usize,
                &gpu::MoePrefillRoute::bytes(&routes),
            );

            let offsets = &self.moe_offsets[layer];
            let pass = self.context.begin_pass_labeled("routed cb (batched)");
            for (blob, _) in &blob_refs {
                pass.use_read_buffer(blob);
            }
            gpu::encode_moe_prefill_phase1(
                &mut self.context,
                &pass,
                &wide,
                offsets,
                (&real.routed_x, x_off),
                (&real.batched().batch_acts, acts_off),
                (&real.batched().batch_routes, routes_off),
                hidden as u32,
                moe_inter,
                top_k as u32,
                (sub_len * top_k) as u32,
                use_silu,
            )
            .map_err(gpu_err)?;
            gpu::encode_moe_prefill_phase2_fused(
                &mut self.context,
                &pass,
                &wide,
                offsets,
                (&real.batched().batch_acts, acts_off),
                (&real.batched().batch_routing_w, rw_off),
                (&real.batched().batch_routes, routes_off),
                // Seed from ZEROS: this family adds its shared expert in
                // the sandwich tail below, after a norm, so the routed sum
                // starts from nothing -- which is what the decode path's
                // `zero_hidden` argument says too. `0.0f + p` is the
                // operation the kernel's old hardcoded seed performed, so
                // this row's bytes do not move.
                (&real.batched().batch_zero, x_off),
                (&real.batched().batch_y, x_off),
                hidden as u32,
                moe_inter,
                top_k as u32,
                sub_len as u32,
                use_silu,
            )
            .map_err(gpu_err)?;

            // The sandwich tail, per token, on the same command buffer
            // after the fused phase 2: the same kernels the per-token
            // path dispatches, at this sub-batch's row offsets.
            let layer_scalar = real.layer_scalar[layer];
            let post_ffn2 = norm_view(
                &self.weights,
                &self.index,
                &layer_tensor(layer, "post_feedforward_layernorm_2.weight"),
                hidden,
            )?;
            let post_ffn = norm_view(
                &self.weights,
                &self.index,
                &layer_tensor(layer, "post_feedforward_layernorm.weight"),
                hidden,
            )?;
            for i in 0..sub_len {
                let row_off = x_off + (i * hidden * 2) as u64;
                let x_row = ((sub_start + i) * hidden * 2) as u64;
                gpu::encode_rms_norm_bf16w(
                    &mut self.context,
                    &pass,
                    (&real.batched().batch_y, row_off),
                    post_ffn2,
                    (&real.batched().batch_y, row_off),
                    hidden as u32,
                    RMS_EPS,
                )
                .map_err(gpu_err)?;
                gpu::encode_residual_add(
                    &mut self.context,
                    &pass,
                    (&batch_h1, row_off),
                    (&real.batched().batch_y, row_off),
                    hidden as u32,
                )
                .map_err(gpu_err)?;
                gpu::encode_rms_norm_bf16w(
                    &mut self.context,
                    &pass,
                    (&batch_h1, row_off),
                    post_ffn,
                    (&self.scratch.ffn_normed, 0),
                    hidden as u32,
                    RMS_EPS,
                )
                .map_err(gpu_err)?;
                gpu::encode_residual_add(
                    &mut self.context,
                    &pass,
                    (&self.scratch.x, x_row),
                    (&self.scratch.ffn_normed, 0),
                    hidden as u32,
                )
                .map_err(gpu_err)?;
                gpu::encode_scalar_mul(
                    &mut self.context,
                    &pass,
                    (&self.scratch.x, x_row),
                    layer_scalar,
                    hidden as u32,
                )
                .map_err(gpu_err)?;
            }

            sub_start += sub_len;
            committed = Some(pass.commit());
        }

        Ok(committed.expect("m >= 1 guarantees at least one sub-batch"))
    }
}
