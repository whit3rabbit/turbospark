//! The real-checkpoint Qwen decode flow for [`RealForwardRunner`], serving
//! BOTH the `qwen36` family and (ROADMAP's 1-bit entry) the dense `qwen3_5`
//! one, which differ in their FFN half and in nothing else.

mod attn;
mod batched;
mod dense;
mod moe;
mod mtp;
mod state;

pub(crate) use attn::{encode_full_attention_block, encode_linear_block, QkNormConvention};
pub(crate) use mtp::{draft_depth_from_env, MtpState};
pub(crate) use state::RealQwenState;

use std::time::Instant;

use foundation::LogitValue;

use crate::real_forward::{RealForwardError, RealForwardRunner};
use crate::real_forward_dispatch::{encode_embed_any, encode_gemv_any};
use crate::real_forward_utils::{entry, norm_view};

pub(crate) const RMS_EPS: f32 = 1e-6;

/// The trunk's tensor-name prefix.
pub(crate) const TRUNK_PREFIX: &str = "language_model.model";

/// The multi-token-prediction head's, for `prefixed_layer_tensor`.
///
/// **No trailing dot**, unlike `repack`'s `classify::MTP_PREFIX`. The two
/// answer different questions and are deliberately not shared: that one
/// MATCHES a name (`name.starts_with("mtp.")`) and this one BUILDS one
/// (`format!("{prefix}.layers.{layer}.{suffix}")`), so a single constant
/// would be wrong at one of the two sites.
pub(crate) const MTP_PREFIX: &str = "mtp";

pub(crate) fn layer_tensor(layer: usize, suffix: &str) -> String {
    prefixed_layer_tensor(TRUNK_PREFIX, layer, suffix)
}

/// [`layer_tensor`] under an explicit prefix.
///
/// The ONLY reason this exists is the multi-token-prediction head
/// (`docs/MTP_SPECULATIVE.md`), whose single block is shape-identical to a
/// trunk full-attention layer field for field and so can run the SAME
/// encoders under `mtp.layers.0.*` -- rather than a second copy of them,
/// which is what Gotcha 11 is about the cost of.
///
/// It is a STRING change and not a flow change: every caller passing
/// [`TRUNK_PREFIX`] resolves the byte-identical name it resolved before, and
/// `qwen38_quality_gate` was re-run to say so rather than to hope so.
pub(crate) fn prefixed_layer_tensor(prefix: &str, layer: usize, suffix: &str) -> String {
    format!("{prefix}.layers.{layer}.{suffix}")
}

impl RealForwardRunner {
    pub fn gdn_state_abs_max(&mut self, layer: usize) -> Option<f32> {
        let qwen = self.real_qwen.as_ref()?;
        if !qwen.gdn.is_linear(layer) {
            return None;
        }
        let buf = qwen.gdn.state_buffer(layer);
        let len = (buf.length() as usize) / 4;
        let contents = gpu::read_f32_buffer(buf, len);
        let mut max = 0.0f32;
        for &v in &contents {
            max = max.max(v.abs());
        }
        Some(max)
    }

    pub(crate) fn produce_real_qwen(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let result = self.produce_real_qwen_inner(token, position, logits);
        self.phases.calls += 1;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn produce_real_qwen_inner(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let hidden = arch.hidden_size as usize;
        let inter = arch.intermediate_size as usize;
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

        let qwen = self.real_qwen.as_ref().expect("real Qwen state present");
        let qwen_rotary_dim = qwen.rotary_dim;
        let qwen_shape = qwen.shape;
        let _ = (qwen_rotary_dim, qwen_shape);

        let (
            context,
            weights,
            index,
            arch,
            scratch,
            kv,
            qwen,
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
            self.real_qwen.as_ref().expect("checked above"),
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

            if arch.layer_is_linear(layer) {
                encode_linear_block(context, &pass, weights, index, arch, qwen, scratch, layer)?;
            } else {
                encode_full_attention_block(
                    context,
                    &pass,
                    weights,
                    index,
                    arch,
                    qwen,
                    scratch,
                    kv,
                    TRUNK_PREFIX,
                    layer,
                    position,
                    // The TRUNK's q/k norms are plain. Its MTP head's, which
                    // carry the same names through the same call, are not.
                    QkNormConvention::Plain,
                )?;
            }

            // The RAW attention output joins the residual stream. Qwen has
            // no sandwich norms (`ffn_sandwich_norms: false` in
            // `qwen_gdn_moe_35b_a3b()`), so normalizing `scratch.o` before this
            // add is a Gemma habit, not a Qwen one -- and it applies
            // `post_attention_layernorm`, a tensor that belongs to the
            // stream below, to the attention output as well. Doing both
            // took the reference-answer perplexity from 6.25 to 255,409.
            gpu::encode_residual_add(
                context,
                &pass,
                (&scratch.x, 0),
                (&scratch.o, 0),
                hidden as u32,
            )
            .map_err(gpu_err)?;

            // ONE post-attention norm feeds the router, the shared expert,
            // and the routed experts. Gemma splits this three ways; Qwen
            // does not, and adding the split would change every number.
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
                (&qwen.moe_x, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;

            // THE DENSE HALF DIVERGES HERE AND NOWHERE ELSE. Everything above
            // -- embedding, both norms, both attention blocks, the raw
            // residual -- is the same code for a `qwen3_5` as for a Qwen 3.6,
            // and so is the head below. A dense layer also needs no mid-layer
            // commit, because nothing in it is data-dependent on a host
            // readback the way the router's top-k is, so the pass stays open
            // across the whole token.
            if qwen.dense {
                dense::encode_qwen_layer_dense(
                    context,
                    &pass,
                    weights,
                    index,
                    scratch,
                    qwen,
                    &scratch.x,
                    TRUNK_PREFIX,
                    layer,
                    hidden,
                    inter,
                    use_silu,
                )?;
                continue;
            }

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
                (&qwen.moe_x, 0),
                (&qwen.router_ones, 0),
                (&qwen.router_logits_f32, 0),
                num_experts as u32,
                hidden as u32,
            )
            .map_err(gpu_err)?;

            let t_wait = Instant::now();
            phases.cb1_gpu_nanos += (pass.commit().wait_with_gpu_time() * 1e9) as u64;
            phases.gpu_wait_nanos += t_wait.elapsed().as_nanos() as u64;

            pass = context.begin_pass_labeled("routed cb");
            moe::encode_qwen_layer_moe(
                context,
                &pass,
                weights,
                index,
                scratch,
                qwen,
                streamers,
                slot_buffers,
                routed_blobs.as_ref(),
                moe_offsets,
                routed_layouts,
                router_hist,
                phases,
                layer,
                hidden,
                inter,
                moe_inter,
                num_experts,
                top_k,
                use_silu,
            )?;
        }

        // Final norm + head. No softcap: Qwen has none, and the head must
        // hand `selection::select` raw logits either way.
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
