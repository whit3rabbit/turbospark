//! Per-layer encoding helper for the synthetic short-name fallback forward pass.

use half::f16;

use crate::real_forward::RealForwardRunner;
use crate::real_forward_types::RealForwardError;
use crate::real_forward_utils::{
    f16_slice_to_le_bytes, f16_to_f32, f32_to_f16, layer_name, resident_matrix, topk_softmax,
};

const RMS_EPS: f32 = 1e-6;

impl RealForwardRunner {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_synthetic_layer(
        &mut self,
        pass: &mut gpu::PassEncoder,
        layer: usize,
        position: usize,
        seq_len: u32,
        hidden: usize,
        inter: usize,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        qk_dim: usize,
        kv_dim: usize,
        rotated_pairs: u32,
        theta: f32,
        attn_scale: f32,
        use_silu: bool,
        sandwich: bool,
    ) -> Result<(), RealForwardError> {
        let gpu_err = RealForwardError::Gpu;
        let q_proj = resident_matrix(
            &self.weights,
            &self.index,
            &layer_name("q_proj", layer),
            qk_dim,
            hidden,
        )?;
        let k_proj = resident_matrix(
            &self.weights,
            &self.index,
            &layer_name("k_proj", layer),
            kv_dim,
            hidden,
        )?;
        let o_proj = resident_matrix(
            &self.weights,
            &self.index,
            &layer_name("o_proj", layer),
            hidden,
            qk_dim,
        )?;
        let (k_buf, k_off) = self.kv.k_slot(layer, position);

        gpu::encode_rms_norm_no_scale(
            &mut self.context,
            pass,
            (&self.scratch.x, 0),
            (&self.scratch.normed, 0),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
        gpu::encode_dequant_int4_gemv_resident(
            &mut self.context,
            pass,
            &q_proj,
            (&self.scratch.normed, 0),
            (&self.scratch.q, 0),
        )
        .map_err(gpu_err)?;
        gpu::encode_dequant_int4_gemv_resident(
            &mut self.context,
            pass,
            &k_proj,
            (&self.scratch.normed, 0),
            (k_buf, k_off as u64),
        )
        .map_err(gpu_err)?;
        gpu::encode_rope_proportional_neox(
            &mut self.context,
            pass,
            (&self.scratch.q, 0),
            position as u32,
            num_heads,
            head_dim,
            rotated_pairs,
            theta,
        )
        .map_err(gpu_err)?;
        gpu::encode_rope_proportional_neox(
            &mut self.context,
            pass,
            (k_buf, k_off as u64),
            position as u32,
            num_kv_heads,
            head_dim,
            rotated_pairs,
            theta,
        )
        .map_err(gpu_err)?;

        let (kv_start, active_ring) = if self.arch.full_attention_layer_mask[layer] == 0 {
            let ring = self.kv.ring_capacity(layer) as u32;
            (
                seq_len.saturating_sub(self.arch.sliding_window as u32),
                if ring > 0 && seq_len > ring { ring } else { 0 },
            )
        } else {
            (0, 0)
        };
        gpu::encode_attention_decode(
            &mut self.context,
            pass,
            (&self.scratch.q, 0),
            k_buf,
            k_buf,
            &self.scratch.attn,
            (&self.scratch.attn_out, 0),
            head_dim,
            num_heads,
            num_kv_heads,
            seq_len,
            kv_start,
            active_ring,
            attn_scale,
            None,
        )
        .map_err(gpu_err)?;
        gpu::encode_dequant_int4_gemv_resident(
            &mut self.context,
            pass,
            &o_proj,
            (&self.scratch.attn_out, 0),
            (&self.scratch.o, 0),
        )
        .map_err(gpu_err)?;
        let attn_delta = if sandwich {
            gpu::encode_rms_norm_no_scale(
                &mut self.context,
                pass,
                (&self.scratch.o, 0),
                (&self.scratch.o_normed, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
            &self.scratch.o_normed
        } else {
            &self.scratch.o
        };
        gpu::encode_residual_add(
            &mut self.context,
            pass,
            (&self.scratch.x, 0),
            (attn_delta, 0),
            hidden as u32,
        )
        .map_err(gpu_err)?;

        gpu::encode_rms_norm_no_scale(
            &mut self.context,
            pass,
            (&self.scratch.x, 0),
            (&self.scratch.normed, 0),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;

        if self.arch.num_experts > 0 && self.streamers[layer].is_some() {
            let num_experts = self.arch.num_experts as usize;
            let top_k = self.arch.top_k_experts as usize;
            let moe_inter = self.arch.moe_intermediate_size as u32;
            let router = resident_matrix(
                &self.weights,
                &self.index,
                &layer_name("router", layer),
                num_experts,
                hidden,
            )?;
            gpu::encode_dequant_int4_gemv_resident(
                &mut self.context,
                pass,
                &router,
                (&self.scratch.normed, 0),
                (&self.scratch.router_logits, 0),
            )
            .map_err(gpu_err)?;
            let old_pass = std::mem::replace(pass, self.context.begin_pass());
            old_pass.commit_and_wait();

            let router_logits = f16_to_f32(&gpu::read_buffer_f16(
                &self.scratch.router_logits,
                0,
                num_experts,
            ));
            let (selected, route_weights) = topk_softmax(&router_logits, top_k);
            let streamer = self.streamers[layer].as_mut().expect("checked above");
            let plan = streamer.plan_experts_cached(&selected, &std::collections::HashSet::new());
            let slots = streamer
                .execute_expert_cache_plan(&plan)
                .map_err(|e| RealForwardError::Unsupported(format!("expert stream: {e}")))?;

            let mut routing16 = vec![f16::from_f32(0.0); gpu::MAX_STREAMED_EXPERTS];
            for (i, &w) in route_weights.iter().enumerate() {
                routing16[i] = f16::from_f32(w);
            }
            gpu::write_buffer_bytes(
                &self.scratch.routing_w,
                0,
                &f16_slice_to_le_bytes(&routing16),
            );

            let layer_slots = &self.slot_buffers[layer];
            let blob_refs: Vec<(&gpu::MetalBuffer, u64)> =
                slots.iter().map(|&s| (&layer_slots[s], 0u64)).collect();
            let routed = self.routed_blobs.as_ref().expect("layout implies blobs");
            let offsets = &self.moe_offsets[layer];
            routed
                .bind(&mut self.context, use_silu, &blob_refs)
                .map_err(gpu_err)?;

            for &(buffer, _) in &blob_refs {
                pass.use_read_buffer(buffer);
            }
            gpu::encode_moe_phase1(
                &mut self.context,
                pass,
                routed,
                offsets,
                (&self.scratch.normed, 0),
                (&self.scratch.moe_acts, 0),
                hidden as u32,
                moe_inter,
                top_k as u32,
                use_silu,
            )
            .map_err(gpu_err)?;
            gpu::encode_moe_phase2(
                &mut self.context,
                pass,
                routed,
                offsets,
                (&self.scratch.moe_acts, 0),
                (&self.scratch.routing_w, 0),
                (&self.scratch.zero_hidden, 0),
                (&self.scratch.ffn_out, 0),
                hidden as u32,
                moe_inter,
                top_k as u32,
                use_silu,
            )
            .map_err(gpu_err)?;
        } else if self.arch.num_experts > 0 {
            let old_pass = std::mem::replace(pass, self.context.begin_pass());
            old_pass.commit_and_wait();
            let pre_ffn32 = f16_to_f32(&gpu::read_buffer_f16(&self.scratch.normed, 0, hidden));
            let combined = self.moe_ffn_host(layer, &pre_ffn32)?;
            gpu::write_buffer_bytes(
                &self.scratch.ffn_out,
                0,
                &f16_slice_to_le_bytes(&f32_to_f16(&combined)),
            );
        } else {
            let gate_proj = resident_matrix(
                &self.weights,
                &self.index,
                &layer_name("gate_proj", layer),
                inter,
                hidden,
            )?;
            let up_proj = resident_matrix(
                &self.weights,
                &self.index,
                &layer_name("up_proj", layer),
                inter,
                hidden,
            )?;
            let down_proj = resident_matrix(
                &self.weights,
                &self.index,
                &layer_name("down_proj", layer),
                hidden,
                inter,
            )?;
            gpu::encode_dequant_int4_gemv_resident(
                &mut self.context,
                pass,
                &gate_proj,
                (&self.scratch.normed, 0),
                (&self.scratch.ffn_gate, 0),
            )
            .map_err(gpu_err)?;
            gpu::encode_dequant_int4_gemv_resident(
                &mut self.context,
                pass,
                &up_proj,
                (&self.scratch.normed, 0),
                (&self.scratch.ffn_up, 0),
            )
            .map_err(gpu_err)?;
            let act = if use_silu {
                gpu::encode_silu_mul
            } else {
                gpu::encode_gelu_mul
            };
            act(
                &mut self.context,
                pass,
                (&self.scratch.ffn_gate, 0),
                (&self.scratch.ffn_up, 0),
                (&self.scratch.ffn_act, 0),
                inter as u32,
            )
            .map_err(gpu_err)?;
            gpu::encode_dequant_int4_gemv_resident(
                &mut self.context,
                pass,
                &down_proj,
                (&self.scratch.ffn_act, 0),
                (&self.scratch.ffn_out, 0),
            )
            .map_err(gpu_err)?;
        }

        let ffn_delta = if sandwich {
            gpu::encode_rms_norm_no_scale(
                &mut self.context,
                pass,
                (&self.scratch.ffn_out, 0),
                (&self.scratch.ffn_normed, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
            &self.scratch.ffn_normed
        } else {
            &self.scratch.ffn_out
        };
        gpu::encode_residual_add(
            &mut self.context,
            pass,
            (&self.scratch.x, 0),
            (ffn_delta, 0),
            hidden as u32,
        )
        .map_err(gpu_err)?;

        Ok(())
    }
}
