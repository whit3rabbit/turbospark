//! The `llama`-architecture decode flow for [`RealForwardRunner`] (ROADMAP
//! Phase M2). MoE half only: Mixtral 8x7B / 8x22B run, dense Llama and
//! Mistral are refused at open by `state.rs` until the dense FFN has a GPU
//! path.
//!
//! One decoder layer, which is the pseudocode `docs/NEW_MODEL.md` Phase 0
//! asks for before any of this is written:
//!
//! ```text
//! h      = rms_norm(x, input_layernorm)
//! q,k,v  = h @ {q,k,v}_proj                 // 32 q heads over 8 kv heads
//! q,k    = rope(q, k, pos, base 1e6)        // full-head NeoX
//! a      = attention(q, k, v) @ o_proj      // no gate, no window, no norms
//! x      = x + a                            // RAW, no sandwich norm
//! m      = rms_norm(x, post_attention_layernorm)
//! r      = softmax_topk(m @ router)[:2]     // 8 experts, top-2
//! x      = x + sum(r_w * expert(m))         // no shared expert
//! ```
//!
//! Then a final norm and an untied head, with no softcap.
//!
//! Every difference from the two existing flows is an ABSENCE, which is why
//! this file is the shortest of the three: Gemma's sandwich norms, per-head
//! q/k/v norms, router scales, shared expert and logit softcap are all gone,
//! and so are Qwen's linear layers, output gate and gated shared expert.

mod attn;
mod moe;
mod state;

pub(crate) use state::RealLlamaState;

use std::time::Instant;

use foundation::LogitValue;

use crate::real_forward::{RealForwardError, RealForwardRunner};
use crate::real_forward_dispatch::{encode_embed_any, encode_gemv_any};
use crate::real_forward_utils::{entry, norm_view};

pub(crate) const RMS_EPS: f32 = 1e-5;

pub(crate) fn layer_tensor(layer: usize, suffix: &str) -> String {
    format!("language_model.model.layers.{layer}.{suffix}")
}

impl RealForwardRunner {
    pub(crate) fn produce_real_llama(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let result = self.produce_real_llama_inner(token, position, logits);
        self.phases.calls += 1;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn produce_real_llama_inner(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let hidden = arch.hidden_size as usize;
        let moe_inter = arch.moe_intermediate_size as u32;
        let vocab = arch.vocab_size as usize;
        let num_experts = arch.num_experts as usize;
        let top_k = arch.top_k_experts as usize;
        let use_silu = arch.hidden_activation.contains("silu");
        let gpu_err = RealForwardError::Gpu;

        if position != self.kv.position() {
            return Err(RealForwardError::Unsupported(format!(
                "non-sequential position {position}; KV cache is at {}",
                self.kv.position()
            )));
        }
        if (token as usize) >= vocab {
            return Err(RealForwardError::Unsupported(format!(
                "token id {token} outside vocab {vocab}"
            )));
        }

        let embed_name = "language_model.model.embed_tokens.weight";
        let base = self.index.header.index_size;

        let mut pass = self.context.begin_pass_labeled("cb1 (attn+router)");
        // No `sqrt(hidden)` embedding scale: that is Gemma's, and this
        // architecture's manifest says so (`embeddingScaledBySqrtHidden`).
        encode_embed_any(
            &mut self.context,
            &pass,
            &self.weights,
            &self.index,
            embed_name,
            (&self.scratch.x, 0),
            token as u32,
            hidden as u32,
            1.0,
        )?;

        let (
            context,
            weights,
            index,
            arch,
            scratch,
            kv,
            llama,
            streamers,
            slot_buffers,
            routed_blobs,
            moe_offsets,
            routed_layouts,
            router_hist,
            phases,
        ) = (
            &mut self.context,
            &self.weights,
            &self.index,
            &self.arch,
            &self.scratch,
            &mut self.kv,
            self.real_llama.as_ref().expect("real llama state present"),
            &mut self.streamers,
            &self.slot_buffers,
            &self.routed_blobs,
            &self.moe_offsets,
            &self.routed_layouts,
            &mut self.router_hist,
            &mut self.phases,
        );

        for layer in 0..arch.num_layers as usize {
            let input_norm = norm_view(
                weights,
                index,
                &layer_tensor(layer, "input_layernorm.weight"),
                hidden,
            )?;
            gpu::encode_rms_norm_bf16w(
                context,
                &pass,
                (&scratch.x, 0),
                input_norm,
                (&scratch.normed, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;

            attn::encode_attention_block(
                context, &pass, weights, index, arch, llama, scratch, kv, layer, position,
            )?;

            // RAW residual add. Gemma normalizes the attention output before
            // adding it back (its sandwich norms) and Qwen normalizes it with
            // `post_attention_layernorm`; this architecture does neither, and
            // getting that wrong is invisible in a greedy smoke.
            gpu::encode_residual_add(
                context,
                &pass,
                (&scratch.x, 0),
                (&scratch.o, 0),
                hidden as u32,
            )
            .map_err(gpu_err)?;

            let post_attn = norm_view(
                weights,
                index,
                &layer_tensor(layer, "post_attention_layernorm.weight"),
                hidden,
            )?;
            gpu::encode_rms_norm_bf16w(
                context,
                &pass,
                (&scratch.x, 0),
                post_attn,
                (&llama.moe_x, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;

            let router_name = layer_tensor(layer, "mlp.gate.weight");
            let router = entry(index, &router_name)?;
            if router.dtype != 5 || router.size_bytes as usize != num_experts * hidden {
                return Err(RealForwardError::Unsupported(format!(
                    "{router_name}: expected INT8 (dtype 5) {num_experts}x{hidden}, got dtype {} \
                     with {} packed bytes",
                    router.dtype, router.size_bytes
                )));
            }
            gpu::encode_router_gemv_gemma4(
                context,
                &pass,
                (
                    weights.buffer(),
                    weights.gpu_offset(router.file_offset - base),
                ),
                (
                    weights.buffer(),
                    weights.gpu_offset(router.scale_offset - base),
                ),
                (
                    weights.buffer(),
                    weights.gpu_offset(router.bias_offset - base),
                ),
                (&llama.moe_x, 0),
                (&llama.router_ones, 0),
                (&llama.router_logits_f32, 0),
                num_experts as u32,
                hidden as u32,
            )
            .map_err(gpu_err)?;

            let t_wait = Instant::now();
            phases.cb1_gpu_nanos += (pass.commit().wait_with_gpu_time() * 1e9) as u64;
            phases.gpu_wait_nanos += t_wait.elapsed().as_nanos() as u64;

            pass = context.begin_pass_labeled("routed cb");
            moe::encode_llama_layer_moe(
                context,
                &pass,
                index,
                scratch,
                llama,
                streamers,
                slot_buffers,
                routed_blobs.as_ref(),
                moe_offsets,
                routed_layouts,
                router_hist,
                phases,
                layer,
                hidden,
                moe_inter,
                num_experts,
                top_k,
                use_silu,
            )?;
        }

        if !self.skip_head {
            pass.relabel("final cb (head)");
            let final_norm = norm_view(weights, index, "language_model.model.norm.weight", hidden)?;
            gpu::encode_rms_norm_bf16w(
                context,
                &pass,
                (&scratch.x, 0),
                final_norm,
                (&scratch.normed, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
            let head_name = if arch.tie_word_embeddings {
                embed_name.to_string()
            } else {
                "language_model.lm_head.weight".to_string()
            };
            encode_gemv_any(
                context,
                &pass,
                weights,
                index,
                &head_name,
                vocab,
                hidden,
                (&scratch.normed, 0),
                (&scratch.logits, 0),
            )?;
        } else {
            pass.relabel("final cb (prefill, no head)");
        }
        let t_wait = Instant::now();
        phases.final_cb_gpu_nanos += (pass.commit_and_wait_with_gpu_time() * 1e9) as u64;
        phases.final_wait_nanos += t_wait.elapsed().as_nanos() as u64;
        self.kv.advance();

        if self.skip_head {
            return Ok(());
        }
        // Raw logits, never probabilities: `selection::select` softmaxes
        // whatever it is handed (AGENTS.md Gotcha 16). There is no softcap to
        // apply here either.
        let head = gpu::read_buffer_f16(&scratch.logits, 0, vocab);
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
