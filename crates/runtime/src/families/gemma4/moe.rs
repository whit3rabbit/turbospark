//! Routed MoE pass encoding for Gemma 4 decode flow.

use std::time::Instant;

use half::f16;

// Re-exported (not just imported) because `moe_batch.rs`, `mod.rs` and
// `shared_expert.rs` all reach this type as `moe::RoutedSlot`, and moving
// the definition to a shared module shouldn't force three call sites to
// change their qualified paths.
pub(crate) use crate::moe_prefill_pipeline::RoutedSlot;
use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::{
    encode_moe_phase1_any, encode_moe_phase2_any, router_topk_gemma4,
};
use crate::real_forward_types::RealForwardError;
use crate::real_forward_utils::{f16_slice_to_le_bytes, layer_tensor, norm_view};

const RMS_EPS: f32 = 1e-6;

impl RealForwardRunner {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_gemma4_layer_routed_moe(
        &mut self,
        pass: &gpu::PassEncoder,
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
        let x_off = (slot.token * hidden * 2) as u64;
        let rw_off = slot.bank * gpu::MAX_STREAMED_EXPERTS * 2;
        let t_router = Instant::now();
        let real = self.real.as_ref().expect("real state present");
        let router_logits = gpu::read_f32_buffer_at(
            &real.router_logits_f32,
            slot.token * num_experts,
            num_experts,
        );
        let (selected, route_weights) =
            router_topk_gemma4(&router_logits, top_k, &real.per_expert_scale[layer]);
        if let Some(hist) = self.router_hist.as_mut() {
            hist.record(layer, &selected);
        }
        let layer_scalar = real.layer_scalar[layer];

        // MAPPED RESIDENCY SKIPS BOTH THE PLAN AND THE `pread`. Every expert
        // is already addressable in its layer's mapping, so there is no cache
        // to consult and nothing to copy -- which is the whole of what this
        // mode buys (the `expert io` bucket was 3.86 ms/token at 32 slots on
        // this install, and the slot cache it replaces was 1.5-3.0 GiB).
        //
        // `slots` carries a different KIND of handle in each arm (a cache
        // slot index against an expert id), which is why the two blob-ref
        // constructions below are separate rather than sharing a lookup: the
        // two index different things and a shared one would read plausibly
        // wrong bytes rather than failing.
        let mapped_active = self.mapped.buffers.get(layer).is_some_and(Option::is_some);
        let order: Vec<usize> = (0..selected.len()).collect();
        let slots: Vec<usize> = if mapped_active {
            // Counted as requests that all hit, because on this path every
            // expert IS resident by construction. Reporting zero requests
            // would make `MFERENCE_PHASES`'s hit rate undefined rather than
            // perfect, and the two modes have to stay comparable.
            self.phases.expert_requests += selected.len() as u64;
            self.phases.expert_hits += selected.len() as u64;
            self.phases.router_nanos += t_router.elapsed().as_nanos() as u64;
            selected.clone()
        } else {
            let streamer = self.streamers[layer].as_mut().ok_or_else(|| {
                RealForwardError::Unsupported(format!(
                    "real Gemma 4 layer {layer} has no packed-expert streamer"
                ))
            })?;
            // `protect` is empty on the decode path, so this is the same call it
            // has always made. Inside a chunk it names the slots the previous
            // token's in-flight command buffer is reading; the cache ASSERTS
            // rather than degrades when it cannot honour that plus the misses
            // (`ExpertCache::plan_if_possible`), so the caller has already
            // ensured the arithmetic works or retired the buffer first.
            let plan = streamer.plan_experts_cached(&selected, &slot.protect);
            let (requests, hits) = (plan.experts.len() as u64, plan.hits as u64);
            self.phases.expert_requests += requests;
            self.phases.expert_hits += hits;
            self.phases.router_nanos += t_router.elapsed().as_nanos() as u64;

            let streamer = self.streamers[layer]
                .as_mut()
                .expect("streamer presence checked above");
            let t_io = Instant::now();
            let slots = streamer
                .execute_expert_cache_plan(&plan)
                .map_err(|e| RealForwardError::Unsupported(format!("expert stream: {e}")))?;
            self.phases.expert_io_nanos += t_io.elapsed().as_nanos() as u64;
            slots
        };

        if !self.shared_cb_overlap {
            let h1 = self.real.as_ref().expect("real state present").h1.clone();
            self.encode_shared_expert_branch(layer, hidden, inter, use_silu, slot, (&h1, 0))?;
        }
        let real = self.real.as_ref().expect("real state present");

        let t_bind = Instant::now();
        let ordered: Vec<(usize, f32)> = order
            .iter()
            .map(|&i| (slots[i], route_weights[i]))
            .collect();
        let mut routing16 = vec![f16::from_f32(0.0); gpu::MAX_STREAMED_EXPERTS];
        for (dispatch_slot, &(_, weight)) in ordered.iter().enumerate() {
            routing16[dispatch_slot] = f16::from_f32(weight);
        }
        gpu::write_buffer_bytes(
            &self.scratch.routing_w,
            rw_off,
            &f16_slice_to_le_bytes(&routing16),
        );

        // The two arms differ only in WHERE a blob lives, never in the order
        // the slots are dispatched: that stays the router's own ranking in
        // both, which is what keeps output byte-identical across residency
        // modes for the same reason it is byte-identical across slot counts
        // (AGENTS.md Gotcha 27).
        let blob_refs: Vec<(&gpu::MetalBuffer, u64)> = if mapped_active {
            let buffer = self.mapped.buffers[layer]
                .as_ref()
                .expect("mapped residency checked above");
            let mapping = self.mapped.layers[layer]
                .as_ref()
                .expect("mapped residency checked above");
            ordered
                .iter()
                .map(|&(expert, _)| (buffer, mapping.expert_offset(expert)))
                .collect()
        } else {
            let layer_slots = &self.slot_buffers[layer];
            ordered
                .iter()
                .map(|&(cache_slot, _)| (&layer_slots[cache_slot], 0u64))
                .collect()
        };
        // Selected by FIELD rather than through a `&self` accessor: this
        // binding is live across `routed.bind(&mut self.context, ..)` and
        // the two phase encodes, and a method call would borrow all of
        // `self` for that whole span (crate Gotcha 5's E0502 shape).
        let routed = match slot.bank {
            0 => self.routed_blobs.as_ref(),
            n => self.routed_blobs_banks.get(n - 1),
        }
        .ok_or_else(|| {
            RealForwardError::Unsupported("install has no routed-blob buffer".to_string())
        })?;
        let offsets = &self.moe_offsets[layer];
        let layer_layout = self.routed_layouts[layer];
        routed
            .bind(&mut self.context, use_silu, &blob_refs)
            .map_err(gpu_err)?;
        self.phases.bind_nanos += t_bind.elapsed().as_nanos() as u64;

        for &(buffer, _) in &blob_refs {
            pass.use_read_buffer(buffer);
        }

        let main_phase1_k = order.len();
        if main_phase1_k > 0 {
            encode_moe_phase1_any(
                layer_layout.phase1,
                &mut self.context,
                pass,
                routed,
                offsets,
                (&real.routed_x, x_off),
                (&self.scratch.moe_acts, 0),
                hidden as u32,
                moe_inter,
                main_phase1_k as u32,
                use_silu,
            )
            .map_err(gpu_err)?;
        }
        encode_moe_phase2_any(
            layer_layout.phase2,
            &mut self.context,
            pass,
            routed,
            offsets,
            (&self.scratch.moe_acts, 0),
            (&self.scratch.routing_w, rw_off as u64),
            (&self.scratch.zero_hidden, 0),
            (&real.h2, 0),
            hidden as u32,
            moe_inter,
            use_silu,
        )
        .map_err(gpu_err)?;
        let post_ffn2 = norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "post_feedforward_layernorm_2.weight"),
            hidden,
        )?;
        gpu::encode_rms_norm_bf16w(
            &mut self.context,
            pass,
            (&real.h2, 0),
            post_ffn2,
            (&real.h2, 0),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;

        gpu::encode_residual_add(
            &mut self.context,
            pass,
            (&real.h1, 0),
            (&real.h2, 0),
            hidden as u32,
        )
        .map_err(gpu_err)?;
        let post_ffn = norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "post_feedforward_layernorm.weight"),
            hidden,
        )?;
        gpu::encode_rms_norm_bf16w(
            &mut self.context,
            pass,
            (&real.h1, 0),
            post_ffn,
            (&self.scratch.ffn_normed, 0),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
        gpu::encode_residual_add(
            &mut self.context,
            pass,
            (&self.scratch.x, x_off),
            (&self.scratch.ffn_normed, 0),
            hidden as u32,
        )
        .map_err(gpu_err)?;
        gpu::encode_scalar_mul(
            &mut self.context,
            pass,
            (&self.scratch.x, x_off),
            layer_scalar,
            hidden as u32,
        )
        .map_err(gpu_err)?;

        // The cache slots this pass BOUND, so the caller can hand them to
        // the next token as `RoutedSlot::protect` while this command buffer
        // is in flight. Returned rather than recomputed because the plan's
        // assignment is the only thing that knows them.
        Ok(ordered.iter().map(|&(cache_slot, _)| cache_slot).collect())
    }
}
