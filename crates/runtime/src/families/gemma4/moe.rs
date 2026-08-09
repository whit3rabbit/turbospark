//! Routed MoE pass encoding for Gemma 4 decode flow.

use std::time::Instant;

use half::f16;

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
    ) -> Result<(), RealForwardError> {
        let gpu_err = RealForwardError::Gpu;
        let t_router = Instant::now();
        let real = self.real.as_ref().expect("real state present");
        let router_logits = gpu::read_f32_buffer(&real.router_logits_f32, num_experts);
        let (selected, route_weights) =
            router_topk_gemma4(&router_logits, top_k, &real.per_expert_scale[layer]);
        if let Some(hist) = self.router_hist.as_mut() {
            hist.record(layer, &selected);
        }
        let layer_scalar = real.layer_scalar[layer];

        let streamer = self.streamers[layer].as_mut().ok_or_else(|| {
            RealForwardError::Unsupported(format!(
                "real Gemma 4 layer {layer} has no packed-expert streamer"
            ))
        })?;
        let plan = streamer.plan_experts_cached(&selected, &std::collections::HashSet::new());
        let (requests, hits) = (plan.experts.len() as u64, plan.hits as u64);
        self.phases.expert_requests += requests;
        self.phases.expert_hits += hits;
        self.phases.router_nanos += t_router.elapsed().as_nanos() as u64;

        let order: Vec<usize> = (0..selected.len()).collect();

        let streamer = self.streamers[layer]
            .as_mut()
            .expect("streamer presence checked above");
        let t_io = Instant::now();
        let slots = streamer
            .execute_expert_cache_plan(&plan)
            .map_err(|e| RealForwardError::Unsupported(format!("expert stream: {e}")))?;
        self.phases.expert_io_nanos += t_io.elapsed().as_nanos() as u64;

        if !self.shared_cb_overlap {
            self.encode_shared_expert_branch(layer, hidden, inter, use_silu)?;
        }
        let real = self.real.as_ref().expect("real state present");

        let t_bind = Instant::now();
        let ordered: Vec<(usize, f32)> = order
            .iter()
            .map(|&i| (slots[i], route_weights[i]))
            .collect();
        let mut routing16 = vec![f16::from_f32(0.0); gpu::MAX_STREAMED_EXPERTS];
        for (slot, &(_, weight)) in ordered.iter().enumerate() {
            routing16[slot] = f16::from_f32(weight);
        }
        gpu::write_buffer_bytes(
            &self.scratch.routing_w,
            0,
            &f16_slice_to_le_bytes(&routing16),
        );

        let layer_slots = &self.slot_buffers[layer];
        let blob_refs: Vec<(&gpu::MetalBuffer, u64)> = ordered
            .iter()
            .map(|&(slot, _)| (&layer_slots[slot], 0u64))
            .collect();
        let routed = self.routed_blobs.as_ref().ok_or_else(|| {
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
                (&real.routed_x, 0),
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
            (&self.scratch.routing_w, 0),
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
            (&self.scratch.x, 0),
            (&self.scratch.ffn_normed, 0),
            hidden as u32,
        )
        .map_err(gpu_err)?;
        gpu::encode_scalar_mul(
            &mut self.context,
            pass,
            (&self.scratch.x, 0),
            layer_scalar,
            hidden as u32,
        )
        .map_err(gpu_err)?;

        Ok(())
    }
}
