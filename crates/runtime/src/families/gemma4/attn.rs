//! Attention and router pass encoding for Gemma 4 decode flow.

use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::encode_gemv_any;
use crate::real_forward_types::RealForwardError;
use crate::real_forward_utils::{entry, layer_tensor, norm_view};

const RMS_EPS: f32 = 1e-6;

impl RealForwardRunner {
    /// `slot` is the token's index inside a prefill micro-batch, and 0 for
    /// the sequential path. It selects the row of every buffer that has to
    /// outlive this call: the residual stream `x`, the two pre-FFN norms
    /// the routed half reads back, and the router logits. Everything else
    /// here is consumed before the next dispatch and stays single-row.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_gemma4_layer_attn_and_router(
        &mut self,
        pass: &gpu::PassEncoder,
        layer: usize,
        position: usize,
        seq_len: u32,
        base: u64,
        slot: usize,
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let hidden = arch.hidden_size as usize;
        let num_heads = arch.num_heads as u32;
        let num_experts = arch.num_experts as usize;
        let attn_scale = arch.attention_scale as f32;
        let gpu_err = RealForwardError::Gpu;
        let x_off = (slot * hidden * 2) as u64;
        let logits_off = (slot * num_experts * 4) as u64;

        let is_full = arch.full_attention_layer_mask[layer] == 1;
        let head_dim_l = if is_full {
            arch.full_head_dim
        } else {
            arch.head_dim
        } as u32;
        let num_kv_l = if is_full {
            arch.num_full_kv_heads
        } else {
            arch.num_kv_heads
        } as u32;
        let q_dim = (num_heads * head_dim_l) as usize;
        let kv_dim = (num_kv_l * head_dim_l) as usize;

        let input_norm = norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "input_layernorm.weight"),
            hidden,
        )?;
        gpu::encode_rms_norm_bf16w(
            &mut self.context,
            pass,
            (&self.scratch.x, x_off),
            input_norm,
            (&self.scratch.normed, 0),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;

        let q_name = layer_tensor(layer, "self_attn.q_proj.weight");
        let k_name = layer_tensor(layer, "self_attn.k_proj.weight");
        let v_name = if is_full && arch.attention_k_eq_v {
            k_name.clone()
        } else {
            layer_tensor(layer, "self_attn.v_proj.weight")
        };
        let o_name = layer_tensor(layer, "self_attn.o_proj.weight");
        let (k_buf, k_off) = self.kv.k_slot(layer, position);
        let (v_buf, v_off) = self.kv.v_slot(layer, position);
        encode_gemv_any(
            &mut self.context,
            pass,
            &self.weights,
            &self.index,
            &q_name,
            q_dim,
            hidden,
            (&self.scratch.normed, 0),
            (&self.scratch.q, 0),
        )?;
        encode_gemv_any(
            &mut self.context,
            pass,
            &self.weights,
            &self.index,
            &k_name,
            kv_dim,
            hidden,
            (&self.scratch.normed, 0),
            (k_buf, k_off as u64),
        )?;
        encode_gemv_any(
            &mut self.context,
            pass,
            &self.weights,
            &self.index,
            &v_name,
            kv_dim,
            hidden,
            (&self.scratch.normed, 0),
            (v_buf, v_off as u64),
        )?;

        let q_norm = norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "self_attn.q_norm.weight"),
            head_dim_l as usize,
        )?;
        let k_norm = norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "self_attn.k_norm.weight"),
            head_dim_l as usize,
        )?;
        gpu::encode_rms_norm_bf16w_perhead(
            &mut self.context,
            pass,
            (&self.scratch.q, 0),
            q_norm,
            (&self.scratch.q, 0),
            num_heads,
            head_dim_l,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
        gpu::encode_rms_norm_bf16w_perhead(
            &mut self.context,
            pass,
            (k_buf, k_off as u64),
            k_norm,
            (k_buf, k_off as u64),
            num_kv_l,
            head_dim_l,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
        gpu::encode_rms_norm_no_scale_perhead(
            &mut self.context,
            pass,
            (v_buf, v_off as u64),
            (v_buf, v_off as u64),
            num_kv_l,
            head_dim_l,
            RMS_EPS,
        )
        .map_err(gpu_err)?;

        let (rotated_pairs, theta) = if is_full {
            (
                (arch.full_head_dim as f64 * arch.partial_rotary_factor / 2.0) as u32,
                arch.full_rope_theta as f32,
            )
        } else {
            (head_dim_l / 2, arch.rope_theta as f32)
        };
        gpu::encode_rope_proportional_neox(
            &mut self.context,
            pass,
            (&self.scratch.q, 0),
            position as u32,
            num_heads,
            head_dim_l,
            rotated_pairs,
            theta,
        )
        .map_err(gpu_err)?;
        gpu::encode_rope_proportional_neox(
            &mut self.context,
            pass,
            (k_buf, k_off as u64),
            position as u32,
            num_kv_l,
            head_dim_l,
            rotated_pairs,
            theta,
        )
        .map_err(gpu_err)?;

        let (kv_start, active_ring) = if is_full {
            (0, 0)
        } else {
            let ring = self.kv.ring_capacity(layer) as u32;
            (
                seq_len.saturating_sub(arch.sliding_window as u32),
                if ring > 0 && seq_len > ring { ring } else { 0 },
            )
        };
        gpu::encode_attention_decode(
            &mut self.context,
            pass,
            (&self.scratch.q, 0),
            k_buf,
            v_buf,
            &self.scratch.attn,
            (&self.scratch.attn_out, 0),
            head_dim_l,
            num_heads,
            num_kv_l,
            seq_len,
            kv_start,
            active_ring,
            attn_scale,
            // Gemma has no attention sinks; only `gpt-oss` does.
            None,
        )
        .map_err(gpu_err)?;
        encode_gemv_any(
            &mut self.context,
            pass,
            &self.weights,
            &self.index,
            &o_name,
            hidden,
            q_dim,
            (&self.scratch.attn_out, 0),
            (&self.scratch.o, 0),
        )?;

        let post_attn = norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "post_attention_layernorm.weight"),
            hidden,
        )?;
        gpu::encode_rms_norm_bf16w(
            &mut self.context,
            pass,
            (&self.scratch.o, 0),
            post_attn,
            (&self.scratch.o_normed, 0),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
        gpu::encode_residual_add(
            &mut self.context,
            pass,
            (&self.scratch.x, x_off),
            (&self.scratch.o_normed, 0),
            hidden as u32,
        )
        .map_err(gpu_err)?;

        let real = self.real.as_ref().expect("real state present");
        gpu::encode_rms_norm_no_scale(
            &mut self.context,
            pass,
            (&self.scratch.x, x_off),
            (&real.router_x, 0),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
        let pre_ffn = norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "pre_feedforward_layernorm.weight"),
            hidden,
        )?;
        gpu::encode_rms_norm_bf16w(
            &mut self.context,
            pass,
            (&self.scratch.x, x_off),
            pre_ffn,
            (&real.dense_x, x_off),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
        let pre_ffn2 = norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "pre_feedforward_layernorm_2.weight"),
            hidden,
        )?;
        gpu::encode_rms_norm_bf16w(
            &mut self.context,
            pass,
            (&self.scratch.x, x_off),
            pre_ffn2,
            (&real.routed_x, x_off),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;

        let router_name = layer_tensor(layer, "router.proj.weight");
        let router_entry = entry(&self.index, &router_name)?;
        if router_entry.dtype != 5 {
            return Err(RealForwardError::Unsupported(format!(
                "{router_name}: expected INT8 (dtype 5), got dtype {}",
                router_entry.dtype
            )));
        }
        if router_entry.size_bytes as usize != num_experts * hidden {
            return Err(RealForwardError::Unsupported(format!(
                "{router_name}: packed size {} does not match {num_experts}x{hidden}",
                router_entry.size_bytes
            )));
        }
        gpu::encode_router_gemv_gemma4(
            &mut self.context,
            pass,
            (
                self.weights.buffer(),
                self.weights.gpu_offset(router_entry.file_offset - base),
            ),
            (
                self.weights.buffer(),
                self.weights.gpu_offset(router_entry.scale_offset - base),
            ),
            (
                self.weights.buffer(),
                self.weights.gpu_offset(router_entry.bias_offset - base),
            ),
            (&real.router_x, 0),
            (&real.effective_scale[layer], 0),
            (&real.router_logits_f32, logits_off),
            num_experts as u32,
            hidden as u32,
        )
        .map_err(gpu_err)?;

        Ok(())
    }
}
