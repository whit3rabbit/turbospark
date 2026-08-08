//! The real-checkpoint Qwen 3.6 decode flow for [`RealForwardRunner`]: a
//! hybrid of gated-DeltaNet linear-attention layers (mask 2) and gated
//! full-attention layers (mask 1), every layer MoE with a sigmoid-gated
//! shared expert. Selected by `open()` from `ArchConfig.family`, NOT from
//! the resident index's naming -- Gemma 4 and Qwen 3.6 both carry
//! `language_model.model.embed_tokens.weight`, so the naming probe that
//! picks the Gemma flow cannot tell them apart.
//!
//! ```text
//! h = embed_lookup_int4(token)            # embed_scale 1.0, no softcap anywhere
//! for L:
//!   a = rmsnorm_bf16w(h, input_layernorm)
//!   mask 2 (linear):                      # see real_forward_qwen_attn.rs
//!     qkv,z,ap,bp = gdn_in_proj(a)        # fused 4-way INT4 GEMV
//!     conv = gdn_conv_mix_decode(tail, qkv)
//!     gdn_qk_norm(conv)                   # in place, delta scales folded in
//!     y = gdn_delta_step_decode(state, conv, ap, bp, A_log, dt_bias)
//!     o = out_proj(gdn_gated_norm(y, z, norm.weight))
//!   mask 1 (gated full attention):
//!     q,gate = split_q_gate(q_proj(a));  k,v -> KV slots
//!     per-head rmsnorm_bf16w on q and k   # NO v norm
//!     rope_neox_subdim(q, k)              # rotary_dim = full_head_dim * prf
//!     attn = attention_decode(...);  attn *= sigmoid(gate);  o = o_proj(attn)
//!   h += o
//!   m = rmsnorm_bf16w(h, post_attention_layernorm)   # ONE norm feeds all three
//!   idx, w = router_topk(int8_gemv(mlp.gate, m))     # unit scales
//!   h1 = silu_glu(mlp.shared_expert, m) * sigmoid(int8_gemv(shared_expert_gate, m))
//!   h += moe_phase2(..., residual = h1)              # phase 2 fuses the add
//! logits = int4_gemv(rmsnorm_bf16w(h, model.norm), lm_head)
//! ```
//!
//! No sandwich norms, no `layer_scalar`, no embedding scale, no logit
//! softcap: the head returns RAW logits, which is what `LogitProducer`
//! documents and what `selection::select` softmaxes (crate Gotcha 1).
//!
//! **Command-buffer shape is deliberately plain**: one buffer per layer up
//! to the router (`cb1`), a host readback for top-k plus the expert
//! `pread`, then one buffer for the MoE tail. The three overlap seams the
//! Gemma path carries (`MFERENCE_SHARED_CB`,
//! `MFERENCE_ROUTED_PIPELINE`) are throughput-only and are not wired here;
//! see DEVIATIONS.md.
//!
//! Memory, at the real 35B shape (hidden 2048, 40 layers, 30 of them
//! linear): the GDN state is `Hv * Dv * Dk * 4 = 32 * 128 * 128 * 4` = 2
//! MiB per linear layer, 60 MiB total, plus a `(K-1) * qkv_dim * 2` = 48
//! KiB conv tail each. Both are FIXED -- they do not grow with context,
//! which is the entire point of a linear-attention layer. KV rows exist
//! only on the 10 full-attention layers; the other 30 share one
//! page-sized placeholder (`KvCacheManager`). Everything else is the
//! per-token scratch allocated once in [`RealQwenState::build`]; the
//! decode hot path allocates no Metal buffer, which the no-allocation
//! test in `tests/real_forward_qwen.rs` is the guard for.

use std::time::Instant;

use foundation::LogitValue;
use half::f16;

use crate::real_forward::{f16_slice_to_le_bytes, RealForwardError, RealForwardRunner};
use crate::real_forward_gemma4::{
    encode_embed_any, encode_gemv_any, encode_moe_phase1_any, encode_moe_phase2_any, entry,
    norm_view, router_topk_gemma4,
};
use crate::real_forward_qwen_attn::{encode_full_attention_block, encode_linear_block};
pub(crate) use crate::real_forward_qwen_state::RealQwenState;

pub(crate) const RMS_EPS: f32 = 1e-6;

pub(crate) fn layer_tensor(layer: usize, suffix: &str) -> String {
    format!("language_model.model.layers.{layer}.{suffix}")
}

impl RealForwardRunner {
    /// Test hook: `max |S|` over one linear layer's recurrent state, or
    /// `None` if the layer is not linear (or the install is not Qwen).
    ///
    /// Untrained synthetic weights make output-only assertions weak
    /// (AGENTS.md Gotcha 12): a GDN block that silently produced zeros
    /// would still decode finite, deterministic tokens. A non-zero state
    /// after a decode step is direct evidence the recurrence ran.
    #[doc(hidden)]
    pub fn gdn_state_abs_max(&self, layer: usize) -> Option<f32> {
        let qwen = self.real_qwen.as_ref()?;
        if !qwen.gdn.is_linear(layer) {
            return None;
        }
        let buffer = qwen.gdn.state_buffer(layer);
        let count = buffer.length() as usize / 4;
        Some(
            gpu::read_f32_buffer(buffer, count)
                .into_iter()
                .fold(0.0f32, |acc, v| acc.max(v.abs())),
        )
    }

    pub(crate) fn produce_real_qwen36(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let result = self.produce_real_qwen36_inner(token, position, logits);
        self.phases.calls += 1;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn produce_real_qwen36_inner(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        // Disjoint field borrows for the whole pass, so the block encoders
        // can take `&mut context` alongside `&real_qwen` without the
        // re-binding the Gemma flow needs (crate Gotcha 3).
        let Self {
            context,
            weights,
            index,
            arch,
            kv,
            scratch,
            slot_buffers,
            streamers,
            moe_offsets,
            routed_blobs,
            real_qwen,
            phases,
            skip_head,
            routed_layout,
            ..
        } = self;
        let routed_layout = *routed_layout;
        let qwen = real_qwen.as_ref().expect("qwen state present");
        let gpu_err = RealForwardError::Gpu;
        let hidden = arch.hidden_size as usize;
        let inter = arch.intermediate_size as usize;
        let moe_inter = arch.moe_intermediate_size as u32;
        let num_experts = arch.num_experts as usize;
        let top_k = arch.top_k_experts as usize;
        let vocab = arch.vocab_size as usize;
        // Qwen's hidden activation is silu; the MoE kernels take it as a
        // specialization flag rather than a separate pipeline.
        let use_silu = arch.hidden_activation.contains("silu");

        if position != kv.position() {
            return Err(RealForwardError::Unsupported(format!(
                "non-sequential position {position}; KV cache is at {}",
                kv.position()
            )));
        }
        if (token as usize) >= vocab {
            return Err(RealForwardError::Unsupported(format!(
                "token id {token} outside vocab {vocab}"
            )));
        }

        let embed_name = "language_model.model.embed_tokens.weight";
        // Resident entry offsets are file-relative and the mapping starts
        // after the index, so this is subtracted at each use below.
        let base = index.header.index_size;
        let mut pass = context.begin_pass_labeled("cb1 (attn+router)");
        // Qwen has no embedding scale (Gemma's sqrt(H)), hence the 1.0. The
        // table's dtype picks the kernel: a Q4_K_M GGUF keeps this tensor at
        // Q4_K where an MLX install has it INT4-affine.
        encode_embed_any(
            context,
            &pass,
            weights,
            index,
            embed_name,
            (&scratch.x, 0),
            token as u32,
            hidden as u32,
            1.0,
        )?;

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
                    context, &pass, weights, index, arch, qwen, scratch, kv, layer, position,
                )?;
            }
            gpu::encode_residual_add(
                context,
                &pass,
                (&scratch.x, 0),
                (&scratch.o, 0),
                hidden as u32,
            )
            .map_err(gpu_err)?;

            // One post-attention norm feeds the router, the shared expert,
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

            let t_router = Instant::now();
            let router_logits = gpu::read_f32_buffer(&qwen.router_logits_f32, num_experts);
            let (selected, route_weights) =
                router_topk_gemma4(&router_logits, top_k, &qwen.per_expert_ones);
            let streamer = streamers[layer].as_mut().ok_or_else(|| {
                RealForwardError::Unsupported(format!(
                    "Qwen 3.6 layer {layer} has no packed-expert streamer"
                ))
            })?;
            let plan = streamer.plan_experts_cached(&selected, &std::collections::HashSet::new());
            phases.expert_requests += plan.experts.len() as u64;
            phases.expert_hits += plan.hits as u64;
            phases.router_nanos += t_router.elapsed().as_nanos() as u64;

            let t_io = Instant::now();
            let slots = streamer
                .execute_expert_cache_plan(&plan)
                .map_err(|e| RealForwardError::Unsupported(format!("expert stream: {e}")))?;
            phases.expert_io_nanos += t_io.elapsed().as_nanos() as u64;

            let t_bind = Instant::now();
            let mut routing16 = vec![f16::from_f32(0.0); gpu::MAX_STREAMED_EXPERTS];
            for (slot, &w) in route_weights.iter().enumerate() {
                routing16[slot] = f16::from_f32(w);
            }
            gpu::write_buffer_bytes(&scratch.routing_w, 0, &f16_slice_to_le_bytes(&routing16));
            let layer_slots = &slot_buffers[layer];
            let blob_refs: Vec<(&gpu::MetalBuffer, u64)> =
                slots.iter().map(|&s| (&layer_slots[s], 0u64)).collect();
            let routed = routed_blobs.as_ref().ok_or_else(|| {
                RealForwardError::Unsupported("install has no routed-blob buffer".to_string())
            })?;
            let offsets = moe_offsets.as_ref().expect("layout implies offsets");
            routed
                .bind(context, use_silu, &blob_refs)
                .map_err(gpu_err)?;
            phases.bind_nanos += t_bind.elapsed().as_nanos() as u64;

            pass = context.begin_pass_labeled("routed cb");
            for &(buffer, _) in &blob_refs {
                pass.use_read_buffer(buffer);
            }

            // Shared expert: INT8 SwiGLU on moe_x, then the scalar sigmoid
            // gate. It has to land in h1 before phase 2 reads h1 as its
            // residual; one serial encoder guarantees that ordering.
            for (suffix, out) in [
                ("gate_proj", &scratch.ffn_gate),
                ("up_proj", &scratch.ffn_up),
            ] {
                encode_gemv_any(
                    context,
                    &pass,
                    weights,
                    index,
                    &layer_tensor(layer, &format!("mlp.shared_expert.{suffix}.weight")),
                    inter,
                    hidden,
                    (&qwen.moe_x, 0),
                    (out, 0),
                )?;
            }
            let act = if use_silu {
                gpu::encode_silu_mul
            } else {
                gpu::encode_gelu_mul
            };
            act(
                context,
                &pass,
                (&scratch.ffn_gate, 0),
                (&scratch.ffn_up, 0),
                (&scratch.ffn_act, 0),
                inter as u32,
            )
            .map_err(gpu_err)?;
            encode_gemv_any(
                context,
                &pass,
                weights,
                index,
                &layer_tensor(layer, "mlp.shared_expert.down_proj.weight"),
                hidden,
                inter,
                (&scratch.ffn_act, 0),
                (&qwen.h1, 0),
            )?;
            encode_gemv_any(
                context,
                &pass,
                weights,
                index,
                &layer_tensor(layer, "mlp.shared_expert_gate.weight"),
                1,
                hidden,
                (&qwen.moe_x, 0),
                (&qwen.shared_gate_logit, 0),
            )?;
            gpu::encode_sigmoid_scalar_mul(
                context,
                &pass,
                (&qwen.h1, 0),
                (&qwen.shared_gate_logit, 0),
                hidden as u32,
            )
            .map_err(gpu_err)?;

            encode_moe_phase1_any(
                routed_layout,
                context,
                &pass,
                routed,
                offsets,
                (&qwen.moe_x, 0),
                (&scratch.moe_acts, 0),
                hidden as u32,
                moe_inter,
                selected.len() as u32,
                use_silu,
            )
            .map_err(gpu_err)?;
            // Phase 2 fuses the residual add: h2 = h1 + sum_slot w * down.
            encode_moe_phase2_any(
                routed_layout,
                context,
                &pass,
                routed,
                offsets,
                (&scratch.moe_acts, 0),
                (&scratch.routing_w, 0),
                (&qwen.h1, 0),
                (&qwen.h2, 0),
                hidden as u32,
                moe_inter,
                use_silu,
            )
            .map_err(gpu_err)?;
            gpu::encode_residual_add(
                context,
                &pass,
                (&scratch.x, 0),
                (&qwen.h2, 0),
                hidden as u32,
            )
            .map_err(gpu_err)?;
        }

        // Final norm + head. No softcap: Qwen has none, and the head must
        // hand `selection::select` raw logits either way.
        if !*skip_head {
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
        kv.advance();

        if *skip_head {
            return Ok(());
        }
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
