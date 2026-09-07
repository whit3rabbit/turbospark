//! Attention and router pass encoding for Gemma 4 decode flow.

use crate::kv_write::{encode_attention_any, encode_kv_commit, kv_write_target, KvHalf};
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
        // FP16 layer: straight into the cache slot. TurboQuant-quantized
        // layer (full attention only -- SWA layers are never quantized, so
        // `is_full == false` always takes the FP16 arm regardless of
        // `--kv-bits`): into the staging row, normed/RoPE'd there exactly
        // as below, then `encode_kv_commit` quantizes it into the cache.
        let (k_buf, k_off) = kv_write_target(&self.kv, &self.scratch, KvHalf::K, layer, position);
        let (v_buf, v_off) = kv_write_target(&self.kv, &self.scratch, KvHalf::V, layer, position);
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
            (k_buf, k_off),
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
            (v_buf, v_off),
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
            (k_buf, k_off),
            k_norm,
            (k_buf, k_off),
            num_kv_l,
            head_dim_l,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
        gpu::encode_rms_norm_no_scale_perhead(
            &mut self.context,
            pass,
            (v_buf, v_off),
            (v_buf, v_off),
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
            (k_buf, k_off),
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
        // No-op on an FP16 (or SWA) layer; quantizes the normed, RoPE'd
        // staging row into the cache on a TurboQuant-quantized one.
        encode_kv_commit(
            &mut self.context,
            pass,
            &self.kv,
            &self.scratch,
            layer,
            position,
            head_dim_l,
            num_kv_l,
        )?;
        encode_attention_any(
            &mut self.context,
            pass,
            (&self.scratch.q, 0),
            &self.kv,
            &self.scratch,
            (&self.scratch.attn_out, 0),
            layer,
            position,
            head_dim_l,
            num_heads,
            num_kv_l,
            seq_len,
            kv_start,
            active_ring,
            attn_scale,
            // Gemma has no attention sinks; only `gpt-oss` does.
            None,
        )?;
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
        let (router_w, router_scale, router_bias) =
            self.router_offsets_gemma4(&router_name, num_experts, hidden)?;
        gpu::encode_router_gemv_gemma4(
            &mut self.context,
            pass,
            (
                self.weights.buffer(),
                self.weights.gpu_offset(router_w - base),
            ),
            (
                self.weights.buffer(),
                self.weights.gpu_offset(router_scale - base),
            ),
            (
                self.weights.buffer(),
                self.weights.gpu_offset(router_bias - base),
            ),
            (&real.router_x, 0),
            (&real.effective_scale[layer], 0),
            (&real.router_logits_f32, logits_off),
            num_experts as u32,
            hidden as u32,
        )
        .map_err(gpu_err)?;

        self.encode_gemma4_pilot_probe(pass, layer, hidden, num_experts, base)?;

        Ok(())
    }

    /// One-layer-ahead router probe (`TURBOSPARK_PILOT_PROBE`), the measurement
    /// behind `docs/EXPERT_ROUTING.md`'s prefetch-ceiling section.
    ///
    /// Runs layer L+1's router on layer L's post-attention residual, which is
    /// colibri's PILOT predictor. It costs ONE extra GEMV rather than a norm
    /// plus a GEMV because Gemma's router pre-norm is
    /// `encode_rms_norm_no_scale` -- weightless, so it carries no per-layer
    /// tensor and `real.router_x` computed here IS already the input layer
    /// L+1's router would see if it ran now. A family whose router norm has a
    /// weight cannot reuse the buffer this way and owes its own norm encode.
    ///
    /// Encoded into the SAME pass as the production router deliberately: the
    /// MoE encoder reads both results after this command buffer commits, so
    /// the probe needs no synchronisation of its own and cannot read a value
    /// the GPU has not written.
    fn encode_gemma4_pilot_probe(
        &mut self,
        pass: &gpu::PassEncoder,
        layer: usize,
        hidden: usize,
        num_experts: usize,
        base: u64,
    ) -> Result<(), RealForwardError> {
        if !self
            .router_hist
            .as_ref()
            .is_some_and(crate::router_hist::RouterHistogram::pilot_enabled)
        {
            return Ok(());
        }
        // `streamers` carries one entry per layer, so this single lookup
        // answers both guards: out of range (the last layer, which predicts
        // nothing) and present-but-dense (no router to borrow).
        let next = layer
            + self
                .router_hist
                .as_ref()
                .map_or(1, crate::router_hist::RouterHistogram::pilot_offset);
        if self.streamers.get(next).is_none_or(Option::is_none) {
            return Ok(());
        }
        let gpu_err = RealForwardError::Gpu;
        let next_name = layer_tensor(next, "router.proj.weight");
        let (w, scale, bias) = self.router_offsets_gemma4(&next_name, num_experts, hidden)?;
        let real = self.real.as_ref().expect("real state present");
        gpu::encode_router_gemv_gemma4(
            &mut self.context,
            pass,
            (self.weights.buffer(), self.weights.gpu_offset(w - base)),
            (self.weights.buffer(), self.weights.gpu_offset(scale - base)),
            (self.weights.buffer(), self.weights.gpu_offset(bias - base)),
            (&real.router_x, 0),
            (&real.effective_scale[next], 0),
            (&real.pilot_logits_f32, 0),
            num_experts as u32,
            hidden as u32,
        )
        .map_err(gpu_err)?;
        Ok(())
    }

    /// The router weight's three resident offsets, after the two shape
    /// checks both layer encoders make of it. Shared so the batched form
    /// cannot drift into accepting a tensor the per-token form refuses.
    ///
    /// It returns plain offsets rather than the `&ResidentIndexEntry` they
    /// come from because a `&self` borrow lives as long as its result, and
    /// every caller needs `&mut self.context` on the next line.
    pub(crate) fn router_offsets_gemma4(
        &self,
        router_name: &str,
        num_experts: usize,
        hidden: usize,
    ) -> Result<(u64, u64, u64), RealForwardError> {
        let router_entry = entry(&self.index, router_name)?;
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
        Ok((
            router_entry.file_offset,
            router_entry.scale_offset,
            router_entry.bias_offset,
        ))
    }
}
