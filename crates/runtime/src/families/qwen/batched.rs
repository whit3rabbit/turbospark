//! The M-row trunk forward: `docs/MTP_SPECULATIVE.md` step 4's verify pass.
//!
//! This is the FIRST batched math path on a decode flow in this repo, and
//! the distinction matters because one already exists that is not one:
//! `prefill_chunk_real_gemma4` batches COMMAND BUFFERS and still calls
//! `encode_attention_decode` once per token, which is why its win is the
//! per-layer blocking wait rather than the kernels. Here the GEMVs really do
//! become GEMMs.
//!
//! **WHAT IS BATCHED AND WHAT IS NOT COMES FROM THE MEASURED COMPUTE SPLIT**
//! (`docs/MTP_SPECULATIVE.md`: GEMV 93.1%, norms and elementwise 4.1%, GDN
//! recurrence 2.2%, attention 0.6%, un-amortizable floor 6.4%). Every GEMV is
//! batched; norms, RoPE, attention and the recurrent step loop per token.
//! That is not a first-cut simplification to be tightened later -- it is what
//! the composite those numbers feed already assumes, and widening attention
//! at decode context would buy 0.6% of a pass.
//!
//! **THE ROUTED HALF RUNS SINCE ROADMAP PHASE 3** and this file's first
//! refusal used to be "dense only". `moe_batch.rs` drives the batched
//! routed pair; what remains true is the CEILING that refusal cited. A
//! block of M tokens reads the UNION of their routes, up to `8M` experts
//! at top-8, and `ExpertCache::plan_if_possible` ASSERTS
//! `experts.len() <= slot_count` rather than degrading (AGENTS.md
//! Gotcha 54). So M=2 wants 16 and M=4 wants 32, and past that the round
//! is refused by name rather than asserting. That ceiling is not a
//! limitation to work around: small blocks are what pay on this engine,
//! measured independently on both speculative pages.
//!
//! A MoE layer also costs a MID-LAYER COMMIT the dense path does not: the
//! router's top-k is a HOST decision downstream of a readback, so the
//! attention half must complete before the routed half can be planned.
//! That is the same shape the per-token flow has, and it is why `pass` is
//! reassigned here rather than being one buffer for the whole forward.
//!
//! Three refusals remain, all BY NAME rather than by falling back:
//!
//! 1. **INT4 only**, enforced one level down in `encode_gemm_any` and again
//!    in the routed half's `RoutedBlobLayout` check. The 1-bit and 2-bit
//!    checkpoints of this same architecture have no batched kernel.
//! 2. **A union that outgrows the slot cache**, per the paragraph above.
//! 3. **No KV wrap.** M consecutive positions occupy M ADJACENT slots only
//!    while `position % capacity` does not roll over inside the block.
//!
//! The first would otherwise be silent: a sequential fallback is
//! numerically identical, so it would pass every losslessness test while
//! making the measurement describe the wrong engine.
//!
//! **THE STEERING EDIT RUNS HERE**, at each layer's output and ahead of the
//! drafter's aux capture, through the same `encode_steering` the per-token
//! path calls with `rows` of 1. That is what let `real_forward_open.rs` stop
//! refusing steering and speculation together: a verify commits, so an
//! unsteered verify beside a steered sequential fallback is a run of tokens
//! from two models. Measured end to end on the real install, a speculative
//! steered generation is byte-identical to a sequential steered one
//! (`docs/OBLITERATION.md`).
//!
//! **NOTHING REACHES THE ROUTED HALF END TO END YET**, and that is a
//! property of the checkpoints rather than of this file:
//! `speculation_policy`'s `speculation_blocker` still refuses a MoE
//! install, because drafting needs a HEAD and no published MoE
//! conversion of this architecture carries an ingestible one (Ornith's
//! lives in its BF16 repo's last shard and is itself MoE, where
//! `MtpState::REQUIRED` names dense FFN tensors). The path is reached by
//! tests calling `produce_batched` directly.

use foundation::LogitValue;

use super::batched_layers::{
    encode_dense_ffn_batched, encode_full_attention_block_batched, encode_linear_block_batched,
};
pub(crate) use super::BatchedScratch;
use crate::families::qwen::{layer_tensor, RMS_EPS};
use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::{encode_embed_any, encode_gemm_any};
use crate::real_forward_types::RealForwardError;
use crate::real_forward_utils::norm_view;
use crate::steering::encode_steering;

/// One token's router GEMV, into row `m` of the batch's own logits
/// buffer.
///
/// It LOOPS rather than batching for the same reason the norms do: the
/// router is `num_experts x hidden` against this family's 2048 hidden and
/// 256 experts, which is a rounding error beside the expert GEMVs it
/// selects, and `router_gemv_gemma4_r4` has no M-row form.
#[allow(clippy::too_many_arguments)]
fn encode_router_gemv_batched(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &model_io::ResidentIndex,
    router_name: &str,
    qwen: &super::RealQwenState,
    batched: &BatchedScratch,
    m: usize,
    hidden: usize,
    num_experts: usize,
) -> Result<(), RealForwardError> {
    let routed = batched.routed.as_ref().ok_or_else(|| {
        RealForwardError::Unsupported("routed scratch missing on a MoE install".to_string())
    })?;
    let router = crate::real_forward_utils::entry(index, router_name)?;
    if router.dtype != 5 || router.size_bytes as usize != num_experts * hidden {
        return Err(RealForwardError::Unsupported(format!(
            "{router_name}: expected INT8 (dtype 5) {num_experts}x{hidden}, got dtype {} \
             with {} packed bytes",
            router.dtype, router.size_bytes
        )));
    }
    let base = index.header.index_size;
    gpu::encode_router_gemv_gemma4(
        context,
        pass,
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
        (&batched.moe_x, (m * hidden * 2) as u64),
        (&qwen.router_ones, 0),
        (
            &routed.batch_router_logits_f32,
            (m * num_experts * 4) as u64,
        ),
        num_experts as u32,
        hidden as u32,
    )
    .map_err(RealForwardError::Gpu)
}

impl RealForwardRunner {
    /// Runs `tokens` through the trunk in ONE pass, writing `tokens.len() *
    /// vocab` logits.
    ///
    /// Token `m` occupies `start_position + m` and attends over
    /// `[0, start_position + m]`, so the causal structure is identical to
    /// running the same tokens sequentially -- and that is not arranged here,
    /// it falls out of `encode_attention_decode` taking its span from the
    /// position ARGUMENT rather than from the cache cursor. All M KV rows are
    /// written before any query reads, and a query simply does not look at
    /// the rows above its own span.
    ///
    /// The KV cursor advances by `tokens.len()`, exactly as the same tokens
    /// run one at a time would leave it, so `rollback` needs no batched
    /// variant.
    pub fn produce_batched(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let hidden = arch.hidden_size as usize;
        let inter = arch.intermediate_size as usize;
        let vocab = arch.vocab_size as usize;
        let use_silu = arch.hidden_activation.contains("silu");
        let batch = tokens.len();
        let gpu_err = RealForwardError::Gpu;

        if batch == 0 {
            return Err(RealForwardError::Unsupported(
                "batched forward needs at least one token".to_string(),
            ));
        }
        // A DFlash2 install carries no MTP head but needs the SAME verify
        // pass, so its state owns a BatchedScratch of its own and either
        // drafter's scratch serves.
        let batched =
            match (&self.real_mtp, &self.real_dflash) {
                (Some(m), _) => &m.batched,
                (None, Some(d)) => &d.batched,
                (None, None) => return Err(RealForwardError::Unsupported(
                    "no batched scratch; set MFERENCE_MTP_DRAFT or MFERENCE_DFLASH_DRAFT before \
                     opening the model"
                        .to_string(),
                )),
            };
        if batch > batched.batch {
            return Err(RealForwardError::Unsupported(format!(
                "batched forward of {batch} rows against scratch sized for {}",
                batched.batch
            )));
        }
        if start_position != self.kv.position() {
            return Err(RealForwardError::Unsupported(format!(
                "non-sequential batched start {start_position}; KV cache is at {}",
                self.kv.position()
            )));
        }
        if logits.len() != batch * vocab {
            return Err(RealForwardError::Unsupported(format!(
                "batched logits buffer is {}, expected {batch} x {vocab}",
                logits.len()
            )));
        }
        for &t in tokens {
            if (t as usize) >= vocab {
                return Err(RealForwardError::Unsupported(format!(
                    "token id {t} outside vocab {vocab}"
                )));
            }
        }
        // M consecutive positions are M ADJACENT slots only while
        // `position % capacity` does not roll over inside the block. This
        // family has no sliding window so its full layers are linear, but
        // "linear" still wraps at `max_context`; a batched projection writing
        // across that boundary would scatter into row 0 with no symptom
        // beyond wrong attention.
        for layer in 0..arch.num_layers as usize {
            if arch.layer_is_linear(layer) {
                continue;
            }
            let capacity = self.kv.capacity(layer);
            if start_position % capacity + batch > capacity {
                return Err(RealForwardError::Unsupported(format!(
                    "batched forward of {batch} rows at position {start_position} wraps \
                     layer {layer}'s KV capacity {capacity}; the batched projection writes \
                     M adjacent slots and cannot straddle the boundary"
                )));
            }
        }

        let embed_name = "language_model.model.embed_tokens.weight";
        let mut pass = self.context.begin_pass_labeled("batched verify");
        // A MoE layer commits mid-layer, so the routed half's host-side
        // resources have to be reachable from inside the loop. They are
        // taken as disjoint fields here for the same reason
        // `families/qwen/moe.rs` is a free function taking them one by
        // one: a `&mut self` call between a `let real = ...` binding and
        // its last use is E0502 (crate Gotcha 5).
        let (
            context,
            weights,
            index,
            scratch,
            qwen,
            kv,
            dflash,
            steering,
            streamers,
            slot_buffers,
            moe_offsets,
            routed_layouts,
            router_hist,
            phases,
        ) = (
            &mut self.context,
            &self.weights,
            &self.index,
            &self.scratch,
            self.real_qwen
                .as_ref()
                .ok_or_else(|| RealForwardError::Unsupported("not a Qwen install".to_string()))?,
            &mut self.kv,
            self.real_dflash.as_ref(),
            self.steering.as_ref(),
            &mut self.streamers,
            &self.slot_buffers,
            &self.moe_offsets,
            &self.routed_layouts,
            &mut self.router_hist,
            &mut self.phases,
        );
        let expert_cache_slots = self.expert_cache_slots;
        let top_k = arch.top_k_experts as usize;
        let num_experts = arch.num_experts as usize;
        let moe_inter = arch.moe_intermediate_size;
        // The last MoE layer's routed buffer is still in flight when the
        // loop ends; the head below reads `scratch.x`, which it writes.
        let mut routed_in_flight: Option<gpu::CommittedPass> = None;

        // An embedding lookup has no trip count to amortize, so it loops for
        // the same reason the norms below do.
        for (m, &token) in tokens.iter().enumerate() {
            encode_embed_any(
                context,
                &pass,
                weights,
                index,
                embed_name,
                (&scratch.x, (m * hidden) as u64 * 2),
                token as u32,
                hidden as u32,
                1.0,
            )?;
        }

        for layer in 0..arch.num_layers as usize {
            let input_norm = norm_view(
                weights,
                index,
                &layer_tensor(layer, "input_layernorm.weight"),
                hidden,
            )?;
            for m in 0..batch {
                gpu::encode_rms_norm_bf16w(
                    context,
                    &pass,
                    (&scratch.x, (m * hidden) as u64 * 2),
                    input_norm,
                    (&batched.normed, (m * hidden) as u64 * 2),
                    hidden as u32,
                    RMS_EPS,
                )
                .map_err(gpu_err)?;
            }

            if arch.layer_is_linear(layer) {
                encode_linear_block_batched(
                    context, &pass, weights, index, &arch, qwen, batched, layer, batch,
                )?;
            } else {
                encode_full_attention_block_batched(
                    context,
                    &pass,
                    weights,
                    index,
                    &arch,
                    qwen,
                    scratch,
                    batched,
                    kv,
                    layer,
                    start_position,
                    batch,
                )?;
            }

            let post_attn = norm_view(
                weights,
                index,
                &layer_tensor(layer, "post_attention_layernorm.weight"),
                hidden,
            )?;
            for m in 0..batch {
                let row = (m * hidden) as u64 * 2;
                // RAW residual add: this family has no sandwich norms, and
                // normalizing here took the Qwen 3.6 reference perplexity
                // from 6.25 to 255,409 once already (crate Gotcha 11).
                gpu::encode_residual_add(
                    context,
                    &pass,
                    (&scratch.x, row),
                    (&batched.o, row),
                    hidden as u32,
                )
                .map_err(gpu_err)?;
                gpu::encode_rms_norm_bf16w(
                    context,
                    &pass,
                    (&scratch.x, row),
                    post_attn,
                    (&batched.moe_x, row),
                    hidden as u32,
                    RMS_EPS,
                )
                .map_err(gpu_err)?;
            }

            if num_experts == 0 {
                // A dense layer stays in the SAME command buffer: nothing
                // in it is data-dependent on a host readback the way the
                // router's top-k is, so the whole row of layers can ride
                // one buffer exactly as it did before the routed half
                // existed. This branch's bytes do not move.
                encode_dense_ffn_batched(
                    context, &pass, weights, index, scratch, batched, layer, hidden, inter,
                    use_silu, batch,
                )?;
            } else {
                // M router GEMVs into the batch's own logits rows, then
                // COMMIT AND WAIT: the top-k below is a host decision and
                // cannot be encoded.
                let router_name = layer_tensor(layer, "mlp.gate.weight");
                for m in 0..batch {
                    encode_router_gemv_batched(
                        context,
                        &pass,
                        weights,
                        index,
                        &router_name,
                        qwen,
                        batched,
                        m,
                        hidden,
                        num_experts,
                    )?;
                }
                let t_wait = std::time::Instant::now();
                phases.cb1_gpu_nanos += (pass.commit().wait_with_gpu_time() * 1e9) as u64;
                phases.gpu_wait_nanos += t_wait.elapsed().as_nanos() as u64;

                // Retire the PREVIOUS layer's routed buffer. It has
                // provably completed already -- it was committed before
                // this layer's attention buffer on the same queue, and
                // that one was just waited out -- so this costs nothing
                // and is taken for the GPU-time ATTRIBUTION, exactly as
                // the Gemma chunk driver's `retire_routed` is.
                if let Some(prev) = routed_in_flight.take() {
                    phases.routed_cb_gpu_nanos += (prev.wait_with_gpu_time() * 1e9) as u64;
                }
                routed_in_flight = Some(super::moe_batch::encode_qwen_layer_moe_batched(
                    context,
                    weights,
                    index,
                    scratch,
                    qwen,
                    batched,
                    streamers,
                    slot_buffers,
                    moe_offsets,
                    routed_layouts,
                    router_hist,
                    phases,
                    expert_cache_slots,
                    layer,
                    hidden,
                    inter,
                    moe_inter as u32,
                    num_experts,
                    top_k,
                    use_silu,
                    batch,
                )?);
                pass = context.begin_pass_labeled("batched verify");
            }

            // THE STEERING EDIT at M rows, at the same boundary and in the
            // same order as the per-token path: this layer's OUTPUT, ahead of
            // the drafter's capture. ONE dispatch for the whole block rather
            // than M, because the kernel is row-parallel already (one
            // threadgroup per row, `coeff[row]`), so a verify costs the same
            // per-layer dispatch a single token does.
            //
            // A block whose rows are NOT all steered is the failure this
            // cannot be allowed to have: the verify commits every accepted
            // row, so a partially-steered block emits a run of tokens drawn
            // from two different models. `encode_steering` refuses a block
            // wider than the coefficient buffer by name rather than clamping.
            encode_steering(context, &pass, scratch, steering, layer, hidden, batch, 0)?;

            // THE DFLASH2 AUX CAPTURE at M rows: the residual rows this
            // layer just produced, into the fc input's layout. Same point
            // as the per-token hook (`families/qwen/mod.rs`), same zero
            // dispatches when no drafter is open.
            if let Some(d) = dflash {
                if let Some(aux) = d.aux_slot(layer) {
                    gpu::encode_dflash_copy_rows(
                        context,
                        &pass,
                        (&scratch.x, 0),
                        (&d.capture, (aux * hidden) as u64 * 2),
                        batch as u32,
                        hidden as u32,
                        (d.shape.aux_count * hidden) as u32,
                    )
                    .map_err(gpu_err)?;
                }
            }
        }

        // The last MoE layer's routed buffer, retired for its GPU time.
        // The head below is encoded into a buffer committed after it, so
        // the queue already orders the residual stream it wrote.
        if let Some(last) = routed_in_flight.take() {
            phases.routed_cb_gpu_nanos += (last.wait_with_gpu_time() * 1e9) as u64;
        }

        let final_norm = norm_view(weights, index, "language_model.model.norm.weight", hidden)?;
        for m in 0..batch {
            gpu::encode_rms_norm_bf16w(
                context,
                &pass,
                (&scratch.x, (m * hidden) as u64 * 2),
                final_norm,
                (&batched.normed, (m * hidden) as u64 * 2),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
        }
        let head_name = if arch.tie_word_embeddings {
            embed_name.to_string()
        } else {
            "language_model.lm_head.weight".to_string()
        };
        encode_gemm_any(
            context,
            &pass,
            weights,
            index,
            &head_name,
            vocab,
            hidden,
            (&batched.normed, 0),
            (&batched.logits, 0),
            batch,
        )?;

        pass.commit_and_wait();
        for _ in 0..batch {
            kv.advance();
        }
        // THE LAST ROW'S RESIDUAL HAS TO LAND AT ROW 0, because that is where
        // a SEQUENTIAL run of the same tokens leaves it and because
        // `mtp_draft_step` reads `h_t` from `scratch.x` at offset 0.
        //
        // Without this the head drafts off the FIRST token of the block
        // instead of the last, and the failure is invisible in every place
        // one would look: the trunk is untouched, so the committed stream
        // stays byte-identical to a non-speculative run and the losslessness
        // gate passes. What moves is the DRAFTER's quality -- measured on the
        // real install, accept length fell from 1.84 to 1.10 per round and
        // rollbacks went from 0 to 62 of 123 rounds, which reads as a verdict
        // about MTP rather than as a bug in the pass.
        //
        // A host copy rather than a blit: `commit_and_wait` has already run,
        // it is `hidden` halfs (10 KiB here), and it needs no kernel.
        if batch > 1 {
            let row = hidden * 2;
            let last = gpu::read_buffer_bytes(&scratch.x, (batch - 1) * row, row);
            gpu::write_buffer_bytes(&scratch.x, 0, &last);
        }
        gpu::read_buffer_f16_into(&batched.logits, 0, logits);
        // The dflash capture's bookkeeping, last so it cannot alias the
        // scratch borrow above: `batch` rows whose positions start at
        // `start_position`, ready for the next round's context write over
        // the accepted prefix.
        if let Some(d) = self.real_dflash.as_mut() {
            d.note_capture(start_position, batch);
        }
        Ok(())
    }
}
