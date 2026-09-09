//! The `spark2_5` decode flow (`XHToken/Spark-X2.5-4B`), the NINTH flow.
//!
//! A dense 36-layer GQA stack, 16 query heads over 4 KV heads at head_dim
//! 256, with the muse-shaped three-sliding/one-full window at 512. The
//! conventions come from `modeling_spark.py` cross-checked against the
//! upstream llama.cpp merge (PR 27868); the facts live in
//! `docs/SPARK_PHASE0.md`.
//!
//! One decoder layer, transcribed:
//!
//! ```text
//! h        = embed[id]                               // NO scale, NO norm
//! per layer i:
//!   x      = rms(h, input_layernorm, 1e-6)           // plain w, pre-norm
//!   qkv    = x @ q_k_v_proj                          // [4096 | 1024 | 1024]
//!   g      = sigmoid(x @ g_proj)                     // [16], one PER HEAD
//!   q,k,v  = split(qkv); k, v land in the cache slots
//!   full = mask[i] == 1:
//!       rope(q, k; theta = 5e6, rotary_dim = 64)     // leading quarter only
//!   else:
//!       rope(q, k; theta = 1e4, rotary_dim = 256)    // whole head
//!   a      = attention(q, k, v, scale = 256^-0.5, window 512 or causal)
//!   a      = a * g[broadcast over head_dim]          // BEFORE o_proj
//!   h      = h + a @ o_proj                          // raw residual add
//!   x      = rms(h, post_attention_layernorm, 1e-6)
//!   h      = h + down(gelu_erf(x @ gate_proj) * (x @ up_proj))
//! h        = rms(h, model.norm, 1e-6)
//! logits   = h @ embed_tokens                        // TIED head, no softcap
//! ```
//!
//! What this family does NOT have, which is most of what makes the file
//! short: no experts, no router, no shared expert, no streamed expert blob,
//! no learned or no-scale q/k norms, no sandwich norms, no softcap, no
//! embedding scaling, no per-projection biases, no linear or compressed
//! layers. What it DOES have that no neighbour shares is all inside the
//! layer and listed in `attn.rs`'s header: the fused QKV, the per-class
//! rope, and the headwise scalar gate, plus the erf GELU in `mlp.rs`.

mod attn;
mod mlp;
mod prefill;
mod state;

pub(crate) use state::RealSparkState;
use state::RMS_EPS;

use std::time::Instant;

use foundation::LogitValue;

use crate::real_forward::{RealForwardError, RealForwardRunner};
use crate::real_forward_dispatch::{encode_embed_any, encode_gemv_any};
use crate::real_forward_utils::norm_view;
use crate::resid_capture::encode_resid_capture;
use crate::steering::encode_steering;

pub(crate) fn layer_tensor(layer: usize, suffix: &str) -> String {
    format!("language_model.model.layers.{layer}.{suffix}")
}

impl RealForwardRunner {
    pub(crate) fn produce_real_spark(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let result = self.produce_real_spark_inner(token, position, logits);
        self.phases.calls += 1;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn produce_real_spark_inner(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let hidden = arch.hidden_size as usize;
        let inter = arch.intermediate_size as usize;
        let vocab = arch.vocab_size as usize;
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

        let pass = self.context.begin_pass_labeled("cb1 (attn+ffn)");
        // NO `sqrt(hidden)` scale and NO embedding norm: the reference uses
        // the raw row (`llama`'s convention; Gemma scales, muse norms).
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
            spark,
            resid_capture,
            steering,
            phases,
            ffn_hist,
        ) = (
            &mut self.context,
            &self.weights,
            &self.index,
            &self.arch,
            &self.scratch,
            &mut self.kv,
            self.real_spark.as_ref().expect("real spark state present"),
            self.resid_capture.as_ref(),
            self.steering.as_ref(),
            &mut self.phases,
            self.ffn_hist.as_ref(),
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
                context, &pass, weights, index, arch, spark, scratch, kv, layer, position,
            )?;

            // RAW residual add: this family has no sandwich norms, so the
            // attention output joins the stream unnormalized (`llama`'s
            // shape, not muse's).
            gpu::encode_residual_add(
                context,
                &pass,
                (&scratch.x, 0),
                (&scratch.o, 0),
                hidden as u32,
            )
            .map_err(gpu_err)?;

            mlp::encode_mlp_block(
                context, &pass, weights, index, scratch, ffn_hist, layer, hidden, inter, 0,
            )?;

            // The FFN residual add (inside `encode_mlp_block`) is the layer's
            // whole contribution to the one residual stream: steering first,
            // then the capture, matching every other flow's order.
            encode_steering(context, &pass, scratch, steering, layer, hidden, 1, 0)?;
            encode_resid_capture(context, &pass, scratch, resid_capture, layer, hidden, 0)?;
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
            // TIED: the head re-reads the embedding row (`RealSparkState`
            // refuses an untied install). No softcap, no output multiplier.
            encode_gemv_any(
                context,
                &pass,
                weights,
                index,
                embed_name,
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
        // The command buffer has been waited on, so every layer's
        // resid-capture region is final (`crates/runtime` Gotcha 20).
        let skip_head = self.skip_head;
        if let Some(capture) = self.resid_capture.as_mut() {
            capture.record_pass(position, skip_head);
        }
        if let Some(hist) = self.ffn_hist.as_mut() {
            if self.skip_head {
                hist.note_prefill_pass();
            } else {
                hist.record_pass();
            }
        }

        if self.skip_head {
            return Ok(());
        }
        // Raw LOGITS, never probabilities (`selection::select` softmaxes
        // whatever it is handed -- AGENTS.md Gotcha 16), read straight into
        // the caller's slice.
        if vocab != logits.len() {
            return Err(RealForwardError::Unsupported(format!(
                "vocab mismatch: model has {}, caller expected {}",
                vocab,
                logits.len()
            )));
        }
        gpu::read_buffer_f16_into(&scratch.logits, 0, logits);
        Ok(())
    }
}
