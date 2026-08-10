//! Synthetic short-name checkpoint execution flow and host-bridged MoE FFN pass
//! for [`RealForwardRunner`].

mod layer;

use foundation::LogitValue;

use crate::real_forward::RealForwardRunner;
use crate::real_forward_types::RealForwardError;
use crate::real_forward_utils::{
    f16_to_f32, f32_to_f16, layer_name, owned_rows, resident_matrix, topk_softmax,
};

const RMS_EPS: f32 = 1e-6;

impl RealForwardRunner {
    /// The routed-expert FFN host bridge: a real GPU router GEMV, host
    /// top-k, then each selected expert's gate/up/down + gated activation
    /// via `compute::run_ffn`, weighted and summed. Expert weights come
    /// from the per-layer `PreadExpertStreamer` (LFU slot cache, parallel
    /// pread on misses) when the install packs them, or from the resident
    /// region by name otherwise. Phase B moves the expert math onto the
    /// GPU reading the slots directly.
    pub(crate) fn moe_ffn_host(
        &mut self,
        layer: usize,
        x: &[f32],
    ) -> Result<Vec<f32>, RealForwardError> {
        let hidden = self.arch.hidden_size as usize;
        let moe_inter = self.arch.moe_intermediate_size as usize;
        let num_experts = self.arch.num_experts as usize;
        let top_k = self.arch.top_k_experts as usize;

        let x16 = f32_to_f16(x);
        let router = resident_matrix(
            &self.weights,
            &self.index,
            &layer_name("router", layer),
            num_experts,
            hidden,
        )?;
        let logits16 = gpu::dequant_int4_gemv_resident(&mut self.context, &router, &x16)
            .map_err(RealForwardError::Gpu)?;
        let (selected, weights) = topk_softmax(&f16_to_f32(&logits16), top_k);

        let mut combined = vec![0f32; hidden];
        let data = self.weights.data();
        for (&e, &w) in selected.iter().zip(weights.iter()) {
            let gate_rows = owned_rows(
                &self.index,
                data,
                &format!("layer{layer}.expert{e}.gate_proj"),
                moe_inter,
                hidden,
            )?;
            let up_rows = owned_rows(
                &self.index,
                data,
                &format!("layer{layer}.expert{e}.up_proj"),
                moe_inter,
                hidden,
            )?;
            let down_rows = owned_rows(
                &self.index,
                data,
                &format!("layer{layer}.expert{e}.down_proj"),
                hidden,
                moe_inter,
            )?;
            let out = compute::run_ffn(&gate_rows, &up_rows, &down_rows, x, hidden, moe_inter);
            for (c, o) in combined.iter_mut().zip(out.iter()) {
                *c += w * o;
            }
        }
        Ok(combined)
    }

    pub(crate) fn produce_inner(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        if self.real_qwen.is_some() {
            return self.produce_real_qwen36(token, position, logits);
        }
        if self.real_llama.is_some() {
            return self.produce_real_llama(token, position, logits);
        }
        if self.real.is_some() {
            return self.produce_real_gemma4(token, position, logits);
        }
        let hidden = self.arch.hidden_size as usize;
        let inter = self.arch.intermediate_size as usize;
        let num_heads = self.arch.num_heads as u32;
        let num_kv_heads = self.arch.num_full_kv_heads as u32;
        let head_dim = self.arch.full_head_dim as u32;
        let qk_dim = (num_heads * head_dim) as usize;
        let kv_dim = (num_kv_heads * head_dim) as usize;
        let vocab = self.arch.vocab_size as usize;
        let rotated_pairs =
            ((head_dim as f64 * self.arch.partial_rotary_factor) / 2.0).round() as u32;
        let theta = self.arch.full_rope_theta as f32;
        let attn_scale = self.arch.attention_scale as f32;
        let softcap = self.arch.final_logit_softcap as f32;
        let embed_scale = if self.arch.embedding_scaled_by_sqrt_hidden {
            (hidden as f32).sqrt()
        } else {
            1.0
        };
        let use_silu = self.arch.hidden_activation.contains("silu");
        let sandwich = self.arch.ffn_sandwich_norms;

        if position != self.kv.position() {
            return Err(RealForwardError::Unsupported(format!(
                "non-sequential position {position}; KV cache is at {}",
                self.kv.position()
            )));
        }
        if !self.arch.attention_k_eq_v {
            return Err(RealForwardError::Unsupported(
                "only attention_k_eq_v architectures are supported".to_string(),
            ));
        }
        let seq_len = (position + 1) as u32;

        let gpu_err = RealForwardError::Gpu;
        let embed = self
            .index
            .entries
            .get("embed_lm_head")
            .ok_or_else(|| RealForwardError::MissingTensor("embed_lm_head".to_string()))?;
        if (token as usize) >= self.arch.vocab_size as usize {
            return Err(RealForwardError::Unsupported(format!(
                "token id {token} outside vocab {}",
                self.arch.vocab_size
            )));
        }
        let base = self.index.header.index_size;
        let embed_table = self.weights.gpu_offset(embed.file_offset - base);
        let embed_scales = self.weights.gpu_offset(embed.scale_offset - base);
        let embed_biases = self.weights.gpu_offset(embed.bias_offset - base);

        let mut pass = self.context.begin_pass();
        gpu::encode_embed_lookup_int4(
            &mut self.context,
            &pass,
            (self.weights.buffer(), embed_table),
            (self.weights.buffer(), embed_scales),
            (self.weights.buffer(), embed_biases),
            (&self.scratch.x, 0),
            token as u32,
            hidden as u32,
            embed_scale,
        )
        .map_err(gpu_err)?;

        for layer in 0..self.arch.num_layers as usize {
            self.encode_synthetic_layer(
                &mut pass,
                layer,
                position,
                seq_len,
                hidden,
                inter,
                num_heads,
                num_kv_heads,
                head_dim,
                qk_dim,
                kv_dim,
                rotated_pairs,
                theta,
                attn_scale,
                use_silu,
                sandwich,
            )?;
        }

        if !self.skip_head {
            let lm_head =
                resident_matrix(&self.weights, &self.index, "embed_lm_head", vocab, hidden)?;
            gpu::encode_rms_norm_no_scale(
                &mut self.context,
                &pass,
                (&self.scratch.x, 0),
                (&self.scratch.normed, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
            gpu::encode_dequant_int4_gemv_resident(
                &mut self.context,
                &pass,
                &lm_head,
                (&self.scratch.normed, 0),
                (&self.scratch.logits, 0),
            )
            .map_err(gpu_err)?;
            if softcap > 0.0 {
                gpu::encode_logit_softcap(
                    &mut self.context,
                    &pass,
                    (&self.scratch.logits, 0),
                    softcap,
                    vocab as u32,
                )
                .map_err(gpu_err)?;
            }
        }
        pass.commit_and_wait();
        self.kv.advance();

        if self.skip_head {
            return Ok(());
        }
        let head = gpu::read_buffer_f16(&self.scratch.logits, 0, vocab);
        if head.len() != logits.len() {
            return Err(RealForwardError::Unsupported(format!(
                "vocab mismatch: model has {}, caller expected {}",
                head.len(),
                logits.len()
            )));
        }
        logits.copy_from_slice(&head);
        Ok(())
    }
}
