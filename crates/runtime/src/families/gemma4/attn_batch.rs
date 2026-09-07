//! Batched attention and router pass encoding for Gemma 4 prefill flow.

use super::attn::encode_gemma4_layer_tail;
use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::encode_gemm_any;
use crate::real_forward_types::RealForwardError;
use crate::real_forward_utils::{layer_tensor, norm_view, ring_spans};

const RMS_EPS: f32 = 1e-6;

impl RealForwardRunner {
    /// The same layer half for ALL `m` tokens of a prefill micro-batch,
    /// with the four resident projections as M-row GEMMs
    /// (`TURBOSPARK_BATCHED_GEMV`, `docs/BATCHED_PREFILL.md`'s 29.7% row).
    ///
    /// **WHAT BATCHES AND WHAT DOES NOT, and the split is the composite's
    /// rather than a first cut.** `q_proj` / `k_proj` / `v_proj` / `o_proj`
    /// batch, because a GEMM amortizes one weight matrix across M rows.
    /// Everything else loops per token: the input norm, the three per-head
    /// norms, both RoPE calls, attention itself, the post-attention norm,
    /// the residual add, the two pre-FFN norms and the router GEMV. Norms
    /// and elementwise work are the 21.2% of prefill with no weights to
    /// amortize; attention has its own batched form and it is step 4, not
    /// this. `families/qwen/batched_layers.rs` draws the line in the same
    /// place for the same reason.
    ///
    /// Single-row scratch is reused across the M tokens wherever it is a
    /// GPU-only intermediate consumed before the next dispatch writes it
    /// (`o_normed`, `router_x`, `scratch.attn`): dispatches inside one
    /// serial compute encoder run in order, which is `crates/gpu` Gotcha 8.
    /// The buffers that need a row per token are the ones that outlive the
    /// layer or are read by the HOST, and all five of those -- `x`,
    /// `dense_x`, `routed_x`, `router_logits_f32`, `batch_h1` -- already
    /// had one.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_gemma4_layer_attn_and_router_batched(
        &mut self,
        pass: &gpu::PassEncoder,
        layer: usize,
        start_position: usize,
        base: u64,
        m: usize,
    ) -> Result<(), RealForwardError> {
        // `TURBOSPARK_BATCHED_GEMV` writes K/V straight into the cache slot
        // via an M-row GEMM, bypassing `kv_write::kv_write_target`'s
        // staging fork entirely -- wiring `--kv-bits` into this path is a
        // named follow-up (`docs/TRUBOQUANT.md`), not yet done, so a
        // quantized layer is refused by name here rather than silently
        // writing raw FP16 bytes into a buffer TurboQuant has sized for
        // packed words.
        if self.kv.layer_quant(layer).is_some() {
            return Err(RealForwardError::Unsupported(
                "--kv-bits is not yet supported with TURBOSPARK_BATCHED_GEMV".to_string(),
            ));
        }
        let arch = self.arch.clone();
        let hidden = arch.hidden_size as usize;
        let num_heads = arch.num_heads as u32;
        let num_experts = arch.num_experts as usize;
        let attn_scale = arch.attention_scale as f32;
        let gpu_err = RealForwardError::Gpu;

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

        // What makes "M adjacent slots in one dispatch" a checked fact
        // rather than a coincidence: the batched kernel's output is
        // token-major with a row stride of `kv_dim` halfs, and that has to
        // be the cache's own per-token stride.
        if self.kv.k_stride(layer) != kv_dim * 2 {
            return Err(RealForwardError::Unsupported(format!(
                "layer {layer}: KV stride {} is not {kv_dim} halfs, so a batched projection \
                 cannot write M adjacent slots in one dispatch",
                self.kv.k_stride(layer)
            )));
        }

        let batched = self.real.as_ref().expect("real state present").batched();
        let (b_normed, b_q, b_attn_out, b_o) = (
            batched.batch_normed.clone(),
            batched.batch_q.clone(),
            batched.batch_attn_out.clone(),
            batched.batch_o.clone(),
        );

        // `batch_q` and `batch_attn_out` are sized at the model's LARGEST
        // `q_dim`, and a full layer's is twice a sliding-window layer's on
        // the real install (16 heads at 512 against 256). Sizing them from
        // `head_dim` alone would under-allocate every full layer by half
        // and the GEMM would run off the end.
        //
        // **NO FIXTURE IN THIS REPO CAN REACH THIS**, which is why it is a
        // guard and not a test: `tiny_gemma4_arch` sets `head_dim` and
        // `full_head_dim` to the same constant, so the two sizings agree
        // there and a mutation of the `max` survives every case in
        // `real_forward_gemma4_chunked.rs`. Checked, not assumed.
        let want = (m * q_dim * 2) as u64;
        for (buffer, what) in [(&b_q, "batch_q"), (&b_attn_out, "batch_attn_out")] {
            if buffer.length() < want {
                return Err(RealForwardError::Unsupported(format!(
                    "layer {layer}: {what} holds {} bytes, short of the {want} a batch of {m} at \
                     q_dim {q_dim} writes; it is sized at the model's widest head",
                    buffer.length()
                )));
            }
        }

        let input_norm = norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "input_layernorm.weight"),
            hidden,
        )?;
        for t in 0..m {
            let row = (t * hidden * 2) as u64;
            gpu::encode_rms_norm_bf16w(
                &mut self.context,
                pass,
                (&self.scratch.x, row),
                input_norm,
                (&b_normed, row),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
        }

        let q_name = layer_tensor(layer, "self_attn.q_proj.weight");
        let k_name = layer_tensor(layer, "self_attn.k_proj.weight");
        let v_name = if is_full && arch.attention_k_eq_v {
            k_name.clone()
        } else {
            layer_tensor(layer, "self_attn.v_proj.weight")
        };
        let o_name = layer_tensor(layer, "self_attn.o_proj.weight");

        encode_gemm_any(
            &mut self.context,
            pass,
            &self.weights,
            &self.index,
            &q_name,
            q_dim,
            hidden,
            (&b_normed, 0),
            (&b_q, 0),
            m,
        )?;
        // Both projections into the K/V slots directly. A sliding-window
        // buffer is a RING, so a micro-batch that straddles its wrap needs
        // two dispatches; `spans[1].1` is zero on every write that does
        // not, and then this is exactly the single call it would have been.
        let spans = ring_spans(self.kv.capacity(layer), start_position, m);
        for (name, is_k) in [(&k_name, true), (&v_name, false)] {
            for (offset, rows) in spans {
                if rows == 0 {
                    continue;
                }
                let position = start_position + offset;
                let (buf, off) = if is_k {
                    self.kv.k_slot(layer, position)
                } else {
                    self.kv.v_slot(layer, position)
                };
                encode_gemm_any(
                    &mut self.context,
                    pass,
                    &self.weights,
                    &self.index,
                    name,
                    kv_dim,
                    hidden,
                    (&b_normed, (offset * hidden * 2) as u64),
                    (buf, off as u64),
                    rows,
                )?;
            }
        }

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
        let (rotated_pairs, theta) = if is_full {
            (
                (arch.full_head_dim as f64 * arch.partial_rotary_factor / 2.0) as u32,
                arch.full_rope_theta as f32,
            )
        } else {
            (head_dim_l / 2, arch.rope_theta as f32)
        };

        for t in 0..m {
            let position = start_position + t;
            let q_row = (t * q_dim * 2) as u64;
            let (k_buf, k_off) = self.kv.k_slot(layer, position);
            let k_row = (k_buf, k_off as u64);
            let (v_buf, v_off) = self.kv.v_slot(layer, position);
            let v_row = (v_buf, v_off as u64);

            gpu::encode_rms_norm_bf16w_perhead(
                &mut self.context,
                pass,
                (&b_q, q_row),
                q_norm,
                (&b_q, q_row),
                num_heads,
                head_dim_l,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
            gpu::encode_rms_norm_bf16w_perhead(
                &mut self.context,
                pass,
                k_row,
                k_norm,
                k_row,
                num_kv_l,
                head_dim_l,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
            gpu::encode_rms_norm_no_scale_perhead(
                &mut self.context,
                pass,
                v_row,
                v_row,
                num_kv_l,
                head_dim_l,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
            gpu::encode_rope_proportional_neox(
                &mut self.context,
                pass,
                (&b_q, q_row),
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
                k_row,
                position as u32,
                num_kv_l,
                head_dim_l,
                rotated_pairs,
                theta,
            )
            .map_err(gpu_err)?;

            let seq_len = (position + 1) as u32;
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
                (&b_q, q_row),
                k_buf,
                v_buf,
                &self.scratch.attn,
                (&b_attn_out, q_row),
                head_dim_l,
                num_heads,
                num_kv_l,
                seq_len,
                kv_start,
                active_ring,
                attn_scale,
                None,
            )
            .map_err(gpu_err)?;
        }

        encode_gemm_any(
            &mut self.context,
            pass,
            &self.weights,
            &self.index,
            &o_name,
            hidden,
            q_dim,
            (&b_attn_out, 0),
            (&b_o, 0),
            m,
        )?;

        let post_attn = norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "post_attention_layernorm.weight"),
            hidden,
        )?;
        let pre_ffn = norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "pre_feedforward_layernorm.weight"),
            hidden,
        )?;
        let pre_ffn2 = norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "pre_feedforward_layernorm_2.weight"),
            hidden,
        )?;
        let router_name = layer_tensor(layer, "router.proj.weight");
        let (router_w, router_scale, router_bias) =
            self.router_offsets_gemma4(&router_name, num_experts, hidden)?;

        let real = self.real.as_ref().expect("real state present");
        for t in 0..m {
            let x_off = (t * hidden * 2) as u64;
            let logits_off = (t * num_experts * 4) as u64;
            encode_gemma4_layer_tail(
                &mut self.context,
                pass,
                &self.weights,
                &self.scratch,
                real,
                (&b_o, x_off),
                x_off,
                logits_off,
                layer,
                hidden,
                num_experts,
                base,
                post_attn,
                pre_ffn,
                pre_ffn2,
                (router_w, router_scale, router_bias),
            )?;
        }

        Ok(())
    }
}
