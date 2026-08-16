//! The `muse_glimmer` decode flow (`mlx-community/Muse-Glimmer-30B-4bit`),
//! the SIXTH flow and the SEVENTH family.
//!
//! A dense 52-layer GQA stack, 32 query heads over 2 KV heads at head_dim
//! 128, with a three-sliding/one-full window at 2048 and a logit softcap.
//! Structurally closest to Gemma 4 -- sandwich norms, an alternating window,
//! a softcap -- and a separate flow anyway, on the `gpt-oss` precedent: TEN
//! differences, every one INSIDE the layer, and every one of them fluent
//! rather than fatal if a neighbour's flow is used.
//!
//! One decoder layer, transcribed from
//! `mlx_vlm.models.muse_glimmer.language.DecoderLayer`:
//!
//! ```text
//! h        = rms_norm_no_scale(embed[id], 1e-5)      // NO sqrt(hidden) scale
//! per layer i:
//!   residual = h
//!   x      = centered_rms_norm(h, input_layernorm, 1e-5)     // scale is (1 + w)
//!   q      = x @ q_proj      -> [32, 128]
//!   k, v   = x @ {k,v}_proj  -> [2, 128]
//!   g      = x @ self_attn.gate_proj                          // [32*128]
//!   q      = rms_norm_no_scale_perhead(q, 1e-5); q *= 3.87
//!   k      = rms_norm_no_scale_perhead(k, 1e-5)               // v NOT normed
//!   if layer_rope_theta[i] != 0:                              // SLIDING only
//!       q, k = rope_neox_full_head(q, k, theta = 500000)
//!   a      = attention(q, k, v, scale = 128^-0.5, window 2048 or causal)
//!   a     *= sigmoid(g)
//!   a      = a @ o_proj
//!   residual = residual + centered_rms_norm(a, post_attention_layernorm, 1e-8)
//!   m_in   =            centered_rms_norm(residual, pre_feedforward_layernorm, 1e-5)
//!   m      = down_proj(silu(gate_proj(m_in)) * up_proj(m_in))
//!   h      = residual + centered_rms_norm(m, post_feedforward_layernorm, 1e-8)
//! h        = rms_norm(h, model.norm.weight, 1e-5)   // PLAIN w, NOT (1 + w)
//! logits   = (h @ lm_head) * 0.19611613513818404
//! logits   = 20 * tanh(logits / 20)
//! ```
//!
//! **THE TWO NORM CONVENTIONS IN THAT LISTING ARE THE THING TO NOT GET
//! WRONG.** The four per-layer norms are `CenteredRMSNorm` (`x * (1 + w)`)
//! and the FINAL norm is a plain `nn.RMSNorm` (`x * w`). They are different
//! kernels here (`rmsnorm_bf16w_centered` against `rmsnorm_bf16w`), selected
//! per TENSOR and never per family. Using one everywhere decodes fluently and
//! is a different model.
//!
//! **AND THE TWO EPSILONS.** 1e-5 on the input, pre-FFN, q/k and embedding
//! norms; 1e-8 on the two POST norms. No other flow here carries a pair.
//!
//! What this family does NOT have, which is most of what makes the file
//! short: no experts, no router, no shared expert, no streamed expert blob,
//! no learned q/k norms, no packed query/gate projection, no linear or
//! compressed layers, and no per-projection biases.

mod attn;
mod state;

pub(crate) use state::RealMuseState;
use state::{OUTPUT_MULTIPLIER, POST_NORM_EPS, RMS_EPS};

use std::time::Instant;

use foundation::LogitValue;

use crate::real_forward::{RealForwardError, RealForwardRunner};
use crate::real_forward_dispatch::{encode_embed_any, encode_gemv_any};
use crate::real_forward_utils::norm_view;

pub(crate) fn layer_tensor(layer: usize, suffix: &str) -> String {
    format!("language_model.model.layers.{layer}.{suffix}")
}

impl RealForwardRunner {
    pub(crate) fn produce_real_muse(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let result = self.produce_real_muse_inner(token, position, logits);
        self.phases.calls += 1;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn produce_real_muse_inner(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let hidden = arch.hidden_size as usize;
        let inter = arch.intermediate_size as usize;
        let vocab = arch.vocab_size as usize;
        let softcap = arch.final_logit_softcap as f32;
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
        // NO `sqrt(hidden)` scale: this family NORMS the embedding row
        // instead (`TextModel.embed_norm`, a no-scale RMS). Gemma scales,
        // `llama` does neither, and the manifest says which
        // (`embeddingScaledBySqrtHidden`, false here).
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

        let (context, weights, index, arch, scratch, kv, muse, phases, ffn_hist) = (
            &mut self.context,
            &self.weights,
            &self.index,
            &self.arch,
            &self.scratch,
            &mut self.kv,
            self.real_muse.as_ref().expect("real muse state present"),
            &mut self.phases,
            self.ffn_hist.as_ref(),
        );

        // `embed_norm`. A no-scale RMS over the embedding row, before any
        // layer runs. Nothing else in this port does this.
        gpu::encode_rms_norm_no_scale(
            context,
            &pass,
            (&scratch.x, 0),
            (&scratch.x, 0),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;

        for layer in 0..arch.num_layers as usize {
            // CENTERED, at the STANDARD epsilon.
            let input_norm = norm_view(
                weights,
                index,
                &layer_tensor(layer, "input_layernorm.weight"),
                hidden,
            )?;
            gpu::encode_rms_norm_bf16w_centered(
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
                context, &pass, weights, index, arch, muse, scratch, kv, layer, position,
            )?;

            // SANDWICH TAIL, HALF ONE. The attention output is normalized
            // BEFORE being added back, at the POST epsilon. `llama` adds it
            // raw and Qwen normalizes with a different tensor; getting this
            // wrong is invisible in a greedy smoke.
            let post_attn = norm_view(
                weights,
                index,
                &layer_tensor(layer, "post_attention_layernorm.weight"),
                hidden,
            )?;
            gpu::encode_rms_norm_bf16w_centered(
                context,
                &pass,
                (&scratch.o, 0),
                post_attn,
                (&scratch.o_normed, 0),
                hidden as u32,
                POST_NORM_EPS,
            )
            .map_err(gpu_err)?;
            gpu::encode_residual_add(
                context,
                &pass,
                (&scratch.x, 0),
                (&scratch.o_normed, 0),
                hidden as u32,
            )
            .map_err(gpu_err)?;

            // The FFN's input norm, at the STANDARD epsilon again.
            let pre_ffn = norm_view(
                weights,
                index,
                &layer_tensor(layer, "pre_feedforward_layernorm.weight"),
                hidden,
            )?;
            gpu::encode_rms_norm_bf16w_centered(
                context,
                &pass,
                (&scratch.x, 0),
                pre_ffn,
                (&scratch.ffn_normed, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;

            // A plain dense gated FFN. No router, no experts, and therefore
            // no mid-layer commit: nothing here is data-dependent on a host
            // readback the way an MoE top-k is, so the whole token stays in
            // one command buffer.
            for (suffix, out) in [
                ("mlp.gate_proj.weight", &scratch.ffn_gate),
                ("mlp.up_proj.weight", &scratch.ffn_up),
            ] {
                encode_gemv_any(
                    context,
                    &pass,
                    weights,
                    index,
                    &layer_tensor(layer, suffix),
                    inter,
                    hidden,
                    (&scratch.ffn_normed, 0),
                    (out, 0),
                )?;
            }
            // Under the activation census, `silu_mul` writes into a
            // per-layer region of the capture buffer instead of the shared
            // `ffn_act` scratch (which the next layer would overwrite), and
            // `down_proj` reads from the same region. Same kernel, same
            // inputs, different destination address: the math and the
            // generated text are byte-identical either way. See
            // `ffn_hist.rs`.
            let act = match ffn_hist {
                Some(hist) => (&hist.capture, (layer * inter * 2) as u64),
                None => (&scratch.ffn_act, 0),
            };
            gpu::encode_silu_mul(
                context,
                &pass,
                (&scratch.ffn_gate, 0),
                (&scratch.ffn_up, 0),
                act,
                inter as u32,
            )
            .map_err(gpu_err)?;
            encode_gemv_any(
                context,
                &pass,
                weights,
                index,
                &layer_tensor(layer, "mlp.down_proj.weight"),
                hidden,
                inter,
                act,
                (&scratch.ffn_out, 0),
            )?;

            // SANDWICH TAIL, HALF TWO. Same shape as the attention half,
            // same POST epsilon, and it reuses `ffn_normed` because that
            // buffer's contents are dead by now.
            let post_ffn = norm_view(
                weights,
                index,
                &layer_tensor(layer, "post_feedforward_layernorm.weight"),
                hidden,
            )?;
            gpu::encode_rms_norm_bf16w_centered(
                context,
                &pass,
                (&scratch.ffn_out, 0),
                post_ffn,
                (&scratch.ffn_normed, 0),
                hidden as u32,
                POST_NORM_EPS,
            )
            .map_err(gpu_err)?;
            gpu::encode_residual_add(
                context,
                &pass,
                (&scratch.x, 0),
                (&scratch.ffn_normed, 0),
                hidden as u32,
            )
            .map_err(gpu_err)?;
        }

        if !self.skip_head {
            pass.relabel("final cb (head)");
            // **PLAIN, NOT CENTERED.** `TextModel.norm` is an `nn.RMSNorm`
            // where every norm above is a `CenteredRMSNorm`. One model, two
            // conventions; see the module header.
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
            // Untied: `RealMuseState::build` refuses a tied install.
            encode_gemv_any(
                context,
                &pass,
                weights,
                index,
                "language_model.lm_head.weight",
                vocab,
                hidden,
                (&scratch.normed, 0),
                (&scratch.logits, 0),
            )?;
            // THE ORDER OF THESE TWO IS THE FUNCTION. `softcap(z * m)` is
            // not `softcap(z) * m`, and the reference multiplies first.
            gpu::encode_scalar_mul(
                context,
                &pass,
                (&scratch.logits, 0),
                OUTPUT_MULTIPLIER,
                vocab as u32,
            )
            .map_err(gpu_err)?;
            gpu::encode_logit_softcap(context, &pass, (&scratch.logits, 0), softcap, vocab as u32)
                .map_err(gpu_err)?;
        } else {
            pass.relabel("final cb (prefill, no head)");
        }
        let t_wait = Instant::now();
        phases.final_cb_gpu_nanos += (pass.commit_and_wait_with_gpu_time() * 1e9) as u64;
        phases.final_wait_nanos += t_wait.elapsed().as_nanos() as u64;
        self.kv.advance();

        // The command buffer has been waited on, so every layer's capture
        // region is final. Prefill passes are counted but not read back:
        // prefill routes differently and a 2 MB readback per prompt token
        // buys data the analysis would exclude (`ffn_hist.rs`).
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
        // Raw (softcapped) LOGITS, never probabilities: `selection::select`
        // softmaxes whatever it is handed (AGENTS.md Gotcha 16). The softcap
        // is what HF's `*ForCausalLM.forward` returns, and the multiplier is
        // part of the head rather than of the sampler.
        //
        // Read the head STRAIGHT into the caller's slice. The owned-`Vec`
        // form of this cost a 512 KiB allocation and a second 512 KiB copy
        // per decoded token, outside every profiling bucket in this repo
        // (AGENTS.md Gotcha 23). The length check moves ahead of the read
        // because it was only ever comparing `vocab` to `logits.len()`.
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
