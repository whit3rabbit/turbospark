//! The `lfm3` decode flow (`LiquidAI/LFM3-1.7B`).
//!
//! A dense 24-layer hybrid: 13 gated short-convolution layers, 6 full and 5
//! sliding-window (256) GQA layers, 16 query heads over 4 KV heads at
//! head_dim 128. Conventions from `modeling_lfm3.py`, cross-checked against
//! llama.cpp's `src/models/lfm3.cpp`; facts in `docs/LFM3_PHASE0.md`.
//!
//! One decoder layer, transcribed:
//!
//! ```text
//! h        = embed[id]                               // NO scale, NO norm
//! per layer i:
//!   x      = rms(h, operator_norm, 1e-5)
//!   conv (mask 6):
//!     B,C,u  = split3(x @ in_proj)                    // [2048 | 2048 | 2048]
//!     y      = causal_conv7(tail | B * u)             // no activation
//!     o      = (C * y) @ out_proj                     // tail <- last 6 rows of B*u
//!   attention (mask 1 full, 0 sliding):
//!     q,k,v  = x @ q_proj, x @ k_proj, x @ v_proj     // k, v into the cache
//!     q,k    = rms_per_head(q, q_norm), rms_per_head(k, k_norm)
//!     full:    rope_proportional_neox(theta 1e6, 32 of 64 pairs)
//!     sliding: rope_proportional_neox(theta 1e4, all 64 pairs)
//!     a      = attention(q, k, v, scale 128^-0.5, causal or window 256)
//!     a      = a * sigmoid(x @ gate_proj)[per head]   // BEFORE o_proj
//!     o      = a @ o_proj
//!   h      = h + rms(o, post_operator_norm)           // sandwich
//!   h      = h + rms(w2(silu(w1 f) * w3 f), post_ffn_norm), f = rms(h, ffn_norm)
//! h        = rms(h, embedding_norm)                   // the FINAL norm
//! logits   = h @ embed_tokens                         // TIED, no softcap
//! ```
//!
//! What it deliberately does NOT have yet: a chunked prefill driver. The
//! sequential `produce` path serves prefill (the `deepseek2` precedent), so
//! `supports_chunked_prefill` is false; `encode_short_conv_prefill` exists
//! and is parity-tested for the driver that comes later.
//!
//! The steering and residual-capture hooks are not wired, and
//! `family_dispatches_steering` answers false until they are.

mod attn;
mod conv;
mod mlp;
mod state;

pub(crate) use state::RealLfm3State;
use state::RMS_EPS;

use std::time::Instant;

use foundation::LogitValue;

use crate::real_forward::{RealForwardError, RealForwardRunner};
use crate::real_forward_dispatch::{encode_embed_any, encode_gemv_any};
use crate::real_forward_utils::norm_view;

pub(crate) fn layer_tensor(layer: usize, suffix: &str) -> String {
    format!("language_model.model.layers.{layer}.{suffix}")
}

impl RealForwardRunner {
    pub(crate) fn produce_real_lfm3(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let result = self.produce_real_lfm3_inner(token, position, logits);
        self.phases.calls += 1;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn produce_real_lfm3_inner(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let hidden = self.arch.hidden_size as usize;
        let inter = self.arch.intermediate_size as usize;
        let vocab = self.arch.vocab_size as usize;
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
        let pass = self.context.begin_pass_labeled("cb1 (lfm3 layers)");
        encode_embed_any(
            &mut self.context,
            &pass,
            &self.weights,
            &self.index,
            embed_name,
            (&self.scratch.x, 0),
            token as u32,
            hidden as u32,
            vocab,
            1.0,
        )?;

        let (context, weights, index, arch, scratch, kv, lfm3, phases, ffn_hist) = (
            &mut self.context,
            &self.weights,
            &self.index,
            &self.arch,
            &self.scratch,
            &self.kv,
            self.real_lfm3.as_ref().expect("real lfm3 state present"),
            &mut self.phases,
            self.ffn_hist.as_ref(),
        );

        for layer in 0..arch.num_layers as usize {
            let operator_norm = norm_view(
                weights,
                index,
                &layer_tensor(layer, "operator_norm.weight"),
                hidden,
            )?;
            gpu::encode_rms_norm_bf16w(
                context,
                &pass,
                (&scratch.x, 0),
                operator_norm,
                (&scratch.normed, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;

            if arch.layer_is_short_conv(layer) {
                conv::encode_conv_block(
                    context, &pass, weights, index, arch, lfm3, scratch, layer,
                )?;
            } else {
                attn::encode_attention_block(
                    context, &pass, weights, index, arch, lfm3, scratch, kv, layer, position,
                )?;
            }

            // The mixer-side sandwich norm, into `normed` (the mixer has
            // consumed it), then the residual add.
            let post_operator = norm_view(
                weights,
                index,
                &layer_tensor(layer, "post_operator_norm.weight"),
                hidden,
            )?;
            gpu::encode_rms_norm_bf16w(
                context,
                &pass,
                (&scratch.o, 0),
                post_operator,
                (&scratch.normed, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
            gpu::encode_residual_add(
                context,
                &pass,
                (&scratch.x, 0),
                (&scratch.normed, 0),
                hidden as u32,
            )
            .map_err(gpu_err)?;

            mlp::encode_mlp_block(
                context, &pass, weights, index, scratch, ffn_hist, layer, hidden, inter,
            )?;
        }

        if !self.skip_head {
            pass.relabel("final cb (head)");
            let final_norm = norm_view(
                weights,
                index,
                "language_model.model.embedding_norm.weight",
                hidden,
            )?;
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
        // Raw logits; the sampler normalizes (AGENTS.md Gotcha 16).
        if vocab != logits.len() {
            return Err(RealForwardError::Unsupported(format!(
                "vocab mismatch: model has {}, caller expected {}",
                vocab,
                logits.len()
            )));
        }
        gpu::read_buffer_f16_into(&self.scratch.logits, 0, logits);
        Ok(())
    }

    /// Largest absolute value in one short-conv layer's tail, or `None` when
    /// the layer is not a conv layer or the install is not `lfm3`. Exists so
    /// a synthetic test can prove the conv ran and that `reset()` rewound
    /// it: output-only assertions on untrained weights cannot.
    pub fn short_conv_tail_abs_max(&self, layer: usize) -> Option<f32> {
        let lfm3 = self.real_lfm3.as_ref()?;
        if !lfm3.conv.is_conv(layer) {
            return None;
        }
        let buf = lfm3.conv.tail_buffer(layer);
        let mut halves = vec![half::f16::ZERO; buf.length() as usize / 2];
        gpu::read_buffer_f16_into(buf, 0, &mut halves);
        Some(halves.iter().fold(0.0f32, |m, v| m.max(v.to_f32().abs())))
    }
}
