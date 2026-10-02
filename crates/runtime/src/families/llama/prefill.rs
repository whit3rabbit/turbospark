//! Chunked prefill for the DENSE half of the `llama` architecture (Mistral,
//! Llama 2/3.x): the second [`crate::producer::ChunkedPrefillRunner`]
//! implementation after Gemma 4's, and structurally simpler than it.
//!
//! Gemma 4's driver (`families/gemma4/prefill.rs`) commits one command
//! buffer PER LAYER because its routed half needs the router's top-k read
//! back on the host before the experts can be bound. A dense layer has no
//! such host round trip (`dense.rs`'s own doc: "the whole token stays in
//! one command buffer and the layer loop never syncs"), so a whole
//! micro-batch runs every layer of every token in ONE command buffer,
//! committed once at the end.
//!
//! Only `scratch.x` needs a row per token (the residual stream, read and
//! written across the whole micro-batch); every other scratch buffer
//! (`normed`, `q`, `attn_out`, `o`, `ffn_gate`, `ffn_up`, `ffn_act`,
//! `llama.h2`, `llama.moe_x`) is a transient GPU-only intermediate reused at
//! offset 0 per token, safe because dispatches within one command buffer
//! execute in commit order (`crates/gpu/CLAUDE.md` Gotcha 8) -- the same
//! reasoning Gemma 4's driver already established.

use std::time::Instant;

use foundation::LogitValue;

use super::{attn, dense, layer_tensor};
use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::{encode_embed_any, encode_gemv_any};
use crate::real_forward_types::{RealForwardError, MAX_PREFILL_BATCH};
use crate::real_forward_utils::norm_view;
use crate::resid_capture::encode_resid_capture;
use crate::steering::encode_steering;

#[cfg(all(test, target_os = "macos"))]
thread_local! {
    static BATCH_ATTENTION_DISPATCH_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(all(test, target_os = "macos"))]
pub(super) fn reset_batch_attention_dispatch_count() {
    BATCH_ATTENTION_DISPATCH_COUNT.with(|value| value.set(0));
}

#[cfg(all(test, target_os = "macos"))]
pub(super) fn batch_attention_dispatch_count() -> usize {
    BATCH_ATTENTION_DISPATCH_COUNT.with(std::cell::Cell::get)
}

pub(super) fn batch_attention_route_eligible(
    arch: &model_io::ArchConfig,
    kv: &gpu::KvCacheManager,
    live_rows: usize,
) -> bool {
    let Ok(num_layers) = usize::try_from(arch.num_layers) else {
        return false;
    };
    if live_rows <= 1
        || arch.family != model_io::ModelFamily::Llama
        || arch.num_experts != 0
        || num_layers == 0
        || arch.full_attention_layer_mask.len() != num_layers
        || arch.full_attention_layer_mask.iter().any(|&mask| mask != 1)
    {
        return false;
    }
    (0..num_layers).all(|layer| kv.layer_quant(layer).is_none())
}

/// Encodes one live layer's query and K/V rows, then waits until every cache
/// slot write is visible before returning the batch-attention inputs.
///
/// The preparation helper visits rows in cache-position order. Keeping the
/// pass local here makes that ordering and its completion a prerequisite for
/// exposing the query buffer to batched attention.
#[allow(clippy::too_many_arguments)]
pub(super) fn stage_attention_inputs_batch<'a>(
    context: &mut gpu::MetalContext,
    weights: &gpu::ResidentGpuWeights,
    index: &model_io::ResidentIndex,
    arch: &model_io::ArchConfig,
    llama: &'a super::RealLlamaState,
    scratch: &crate::real_forward_types::DecodeScratch,
    kv: &gpu::KvCacheManager,
    layer: usize,
    first_query_position: usize,
    rope_positions: &[crate::vision::RopePosition],
) -> Result<attn::PreparedAttentionRows<'a>, RealForwardError> {
    let pass = context.begin_pass_labeled("llama batch attention staging");
    let prepared = attn::encode_attention_inputs_batch(
        context,
        &pass,
        weights,
        index,
        arch,
        llama,
        scratch,
        kv,
        layer,
        first_query_position,
        rope_positions,
    )?;
    pass.commit_and_wait();
    Ok(prepared)
}

impl RealForwardRunner {
    /// Runs a whole prefill chunk through the dense `llama` flow, writing
    /// the logits for the position after its last token. Call only once
    /// [`RealLlamaState::dense`] is known true; an MoE install is refused at
    /// [`crate::producer::ChunkedPrefillRunner::prefill_chunk`], by name,
    /// before this is reached.
    pub(crate) fn prefill_chunk_real_llama_dense(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        if tokens.is_empty() {
            return Err(RealForwardError::Unsupported(
                "prefill_chunk called with an empty chunk".to_string(),
            ));
        }
        // REFUSED BY NAME rather than ignored. A dense driver ignores
        // `TURBOSPARK_ROUTED_BATCH` legitimately (no routed half to refer
        // to), but this seam names the driver's RESIDENT GEMVs, which every
        // family has; the M-row GEMM is wired in Gemma 4's driver alone
        // (INT4-affine, step 6), so running the per-token loop anyway would
        // measure the unbatched engine under the batched arm's label
        // (crate Gotcha 22's rule).
        if self.batched_gemv_prefill {
            return Err(RealForwardError::Unsupported(
                "TURBOSPARK_BATCHED_GEMV is not wired for this family: the M-row \
                 resident GEMM exists in the gemma4 chunked driver alone \
                 (INT4-affine), and this driver keeps every resident GEMV per \
                 token"
                    .to_string(),
            ));
        }
        let mut offset = 0usize;
        while offset < tokens.len() {
            let take = (tokens.len() - offset).min(MAX_PREFILL_BATCH);
            let last = offset + take == tokens.len();
            self.prefill_micro_batch_llama_dense(
                &tokens[offset..offset + take],
                start_position + offset,
                last,
                logits,
            )?;
            offset += take;
        }
        Ok(())
    }

    fn prefill_micro_batch_llama_dense(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        want_head: bool,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let result =
            self.prefill_micro_batch_llama_dense_inner(tokens, start_position, want_head, logits);
        // One forward pass per TOKEN, matching every other phase divisor in
        // this port (`crates/runtime/CLAUDE.md` Gotcha 6).
        self.phases.calls += tokens.len() as u64;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn prefill_micro_batch_llama_dense_inner(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        want_head: bool,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let hidden = arch.hidden_size as usize;
        let dense_inter = arch.intermediate_size as usize;
        let vocab = arch.vocab_size as usize;
        let use_silu = arch.hidden_activation.contains("silu");
        let gpu_err = RealForwardError::Gpu;
        let m = tokens.len();

        if start_position != self.kv.position() {
            return Err(RealForwardError::Unsupported(format!(
                "non-sequential chunk start {start_position}; KV cache is at {}",
                self.kv.position()
            )));
        }
        if let Some(&bad) = tokens.iter().find(|&&t| (t as usize) >= vocab) {
            return Err(RealForwardError::Unsupported(format!(
                "token id {bad} outside vocab {vocab}"
            )));
        }
        let use_batch_attention = batch_attention_route_eligible(&arch, &self.kv, m);

        let embed_name = "language_model.model.embed_tokens.weight";

        // Vision: resolve everything the pass needs off `prompt_vision`
        // BEFORE the field split below hands `&mut self.context` out. Rows
        // are copied (the blit is a byte copy anyway), rope triples are
        // `Copy`, and the deepstack buffers clone a retain handle -- none of
        // them keep a borrow of `self` alive across the split. The same
        // precompute pattern `families/qwen/prefill.rs` runs for its driver.
        let (embed_blits, rope_positions, deepstack_rows, deepstack_buffers) =
            match self.prompt_vision.as_ref() {
                Some(pv) => {
                    let blits: Vec<Option<Vec<u8>>> = (0..m)
                        .map(|t| pv.row_for(start_position + t).map(<[u8]>::to_vec))
                        .collect();
                    let ropes: Vec<crate::vision::RopePosition> = (0..m)
                        .map(|t| {
                            let (a, b, c) = pv.rope_position(start_position + t);
                            crate::vision::RopePosition::Triple(a, b, c)
                        })
                        .collect();
                    // Per token, the merged-row index every deepstack
                    // merger's rows share (the mergers all read the same
                    // image positions), `None` for a text position.
                    let ds_rows: Vec<Option<usize>> = (0..m)
                        .map(|t| pv.row_index_for(start_position + t))
                        .collect();
                    (blits, ropes, ds_rows, pv.deepstack_buffers().to_vec())
                }
                // No image map: every position is an ordinary lookup at its
                // own position, the engine this flow shipped as.
                None => {
                    let no_blits: Vec<Option<Vec<u8>>> = vec![None; m];
                    let no_ropes = vec![crate::vision::RopePosition::Sequential; m];
                    (no_blits, no_ropes, Vec::new(), Vec::new())
                }
            };

        let mut pass = Some(self.context.begin_pass_labeled("llama dense chunk cb"));
        // No `sqrt(hidden)` embedding scale, matching the sequential flow:
        // this architecture's manifest declares `embeddingScaledBySqrtHidden`
        // false and `RealLlamaState::build` refuses an install that says
        // otherwise.
        for (t, &token) in tokens.iter().enumerate() {
            match &embed_blits[t] {
                // An image-pad position blits the tower's FP16 row INSTEAD
                // of encoding a lookup -- the placeholder id carries no
                // meaning, so blending the table's row under it would mix a
                // text embedding into every patch. The write is a host write
                // into shared storage landing BEFORE the single commit at
                // the end of this pass, which is what makes it visible to
                // every dispatch below (`crate` Gotcha 27's argument, one
                // flow over).
                Some(row) => {
                    gpu::write_buffer_bytes(&self.scratch.x, t * hidden * 2, row);
                }
                None => {
                    encode_embed_any(
                        &mut self.context,
                        pass.as_ref().expect("chunk pass remains active"),
                        &self.weights,
                        &self.index,
                        embed_name,
                        (&self.scratch.x, (t * hidden * 2) as u64),
                        token as u32,
                        hidden as u32,
                        vocab,
                        1.0,
                    )?;
                }
            }
        }

        let (context, weights, index, scratch, kv, llama, resid_capture, steering) = (
            &mut self.context,
            &self.weights,
            &self.index,
            &self.scratch,
            &mut self.kv,
            self.real_llama.as_ref().expect("real llama state present"),
            self.resid_capture.as_ref(),
            self.steering.as_ref(),
        );

        for layer in 0..arch.num_layers as usize {
            let input_norm = norm_view(
                weights,
                index,
                &layer_tensor(layer, "input_layernorm.weight"),
                hidden,
            )?;
            let post_attn_norm = norm_view(
                weights,
                index,
                &layer_tensor(layer, "post_attention_layernorm.weight"),
                hidden,
            )?;

            if use_batch_attention {
                pass.take()
                    .expect("chunk pass remains active before batch staging")
                    .commit_and_wait();
                let prepared = stage_attention_inputs_batch(
                    context,
                    weights,
                    index,
                    &arch,
                    llama,
                    scratch,
                    kv,
                    layer,
                    start_position,
                    &rope_positions,
                )?;
                let batch = llama.batch_attention_buffers.as_ref().ok_or_else(|| {
                    RealForwardError::Unsupported(
                        "dense Llama batch attention buffers are unavailable".into(),
                    )
                })?;
                let (k_buffer, _) = kv.k_slot(layer, 0);
                let (v_buffer, _) = kv.v_slot(layer, 0);
                let first_query_position =
                    u32::try_from(prepared.first_query_position).map_err(|_| {
                        RealForwardError::Unsupported(format!(
                            "query position {} exceeds the GPU position range",
                            prepared.first_query_position
                        ))
                    })?;
                let head_dim = arch.full_head_dim as u32;
                let num_q_heads = arch.num_heads as u32;
                let num_kv_heads = arch.num_full_kv_heads as u32;
                let batch_pass = context.begin_pass_labeled("llama dense batch attention cb");
                #[cfg(all(test, target_os = "macos"))]
                BATCH_ATTENTION_DISPATCH_COUNT.with(|count| count.set(count.get() + 1));
                gpu::encode_attention_decode_batch(
                    context,
                    &batch_pass,
                    (prepared.query, 0),
                    k_buffer,
                    v_buffer,
                    &batch.scratch,
                    (&batch.output, 0),
                    gpu::BatchAttentionInputContract {
                        kv_layout: gpu::BatchAttentionKvLayout::Linear,
                        kv_format: gpu::BatchAttentionKvFormat::Fp16,
                        kv_start: 0,
                        ring_capacity: kv.ring_capacity(layer),
                        sink_count: 0,
                    },
                    first_query_position,
                    prepared.row_count,
                    head_dim,
                    num_q_heads,
                    num_kv_heads,
                    arch.attention_scale as f32,
                )
                .map_err(gpu_err)?;

                let q_dim = (num_q_heads * head_dim) as usize;
                for t in 0..prepared.row_count {
                    let x_off = (t * hidden * 2) as u64;
                    let attention_off = (t * q_dim * 2) as u64;
                    encode_gemv_any(
                        context,
                        &batch_pass,
                        weights,
                        index,
                        &layer_tensor(layer, "self_attn.o_proj.weight"),
                        hidden,
                        q_dim,
                        (&batch.output, attention_off),
                        (&scratch.o, 0),
                    )?;

                    gpu::encode_residual_add(
                        context,
                        &batch_pass,
                        (&scratch.x, x_off),
                        (&scratch.o, 0),
                        hidden as u32,
                    )
                    .map_err(gpu_err)?;
                    gpu::encode_rms_norm_bf16w(
                        context,
                        &batch_pass,
                        (&scratch.x, x_off),
                        post_attn_norm,
                        (&llama.moe_x, 0),
                        hidden as u32,
                        llama.rms_eps,
                    )
                    .map_err(gpu_err)?;
                    dense::encode_llama_layer_dense(
                        context,
                        &batch_pass,
                        weights,
                        index,
                        scratch,
                        llama,
                        layer,
                        hidden,
                        dense_inter,
                        use_silu,
                        x_off,
                    )?;

                    // Steering runs before capture, matching the existing row path.
                    encode_steering(
                        context,
                        &batch_pass,
                        scratch,
                        steering,
                        layer,
                        hidden,
                        1,
                        x_off,
                    )?;
                    encode_resid_capture(
                        context,
                        &batch_pass,
                        scratch,
                        resid_capture,
                        layer,
                        hidden,
                        x_off,
                    )?;
                }
                pass = Some(batch_pass);
            } else {
                let row_pass = pass.as_ref().expect("chunk pass remains active");
                for (t, rope_position) in rope_positions.iter().enumerate() {
                    let rope_position = *rope_position;
                    let position = start_position + t;
                    let x_off = (t * hidden * 2) as u64;

                    gpu::encode_rms_norm_bf16w(
                        context,
                        row_pass,
                        (&scratch.x, x_off),
                        input_norm,
                        (&scratch.normed, 0),
                        hidden as u32,
                        llama.rms_eps,
                    )
                    .map_err(gpu_err)?;
                    attn::encode_attention_block(
                        context,
                        row_pass,
                        weights,
                        index,
                        &arch,
                        llama,
                        scratch,
                        kv,
                        layer,
                        position,
                        rope_position,
                    )?;

                    // The architecture adds attention and FFN outputs raw to this row.
                    gpu::encode_residual_add(
                        context,
                        row_pass,
                        (&scratch.x, x_off),
                        (&scratch.o, 0),
                        hidden as u32,
                    )
                    .map_err(gpu_err)?;
                    gpu::encode_rms_norm_bf16w(
                        context,
                        row_pass,
                        (&scratch.x, x_off),
                        post_attn_norm,
                        (&llama.moe_x, 0),
                        hidden as u32,
                        llama.rms_eps,
                    )
                    .map_err(gpu_err)?;
                    dense::encode_llama_layer_dense(
                        context,
                        row_pass,
                        weights,
                        index,
                        scratch,
                        llama,
                        layer,
                        hidden,
                        dense_inter,
                        use_silu,
                        x_off,
                    )?;
                    encode_steering(
                        context, row_pass, scratch, steering, layer, hidden, 1, x_off,
                    )?;
                    encode_resid_capture(
                        context,
                        row_pass,
                        scratch,
                        resid_capture,
                        layer,
                        hidden,
                        x_off,
                    )?;
                }
            }

            // DEEPSTACK: merger `layer`'s rows raw-add at this chunk's image
            // positions once the whole layer has run -- the reference adds
            // to the layer's OUTPUT (`h = layer(h)` then
            // `_deepstack_process`), and injection depth `layer` is merger
            // `layer`'s slot, never block `indexes[layer]`'s. One add
            // dispatch per image position, from the GPU-resident rows
            // uploaded at `set_prompt_vision`; a text position adds nothing.
            if layer < deepstack_buffers.len() {
                for (t, row) in deepstack_rows.iter().enumerate() {
                    if let Some(row) = row {
                        gpu::encode_residual_add(
                            context,
                            pass.as_ref().expect("chunk pass remains active"),
                            (&scratch.x, (t * hidden * 2) as u64),
                            (&deepstack_buffers[layer], (row * hidden * 2) as u64),
                            hidden as u32,
                        )
                        .map_err(gpu_err)?;
                    }
                }
            }
        }

        let pass = pass.expect("chunk pass remains active at the final layer");
        if want_head {
            pass.relabel("llama dense final cb (head)");
            let last_off = ((m - 1) * hidden * 2) as u64;
            let final_norm = norm_view(weights, index, "language_model.model.norm.weight", hidden)?;
            gpu::encode_rms_norm_bf16w(
                context,
                &pass,
                (&scratch.x, last_off),
                final_norm,
                (&scratch.normed, 0),
                hidden as u32,
                llama.rms_eps,
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
            // No softcap: `RealLlamaState::build` refuses an install that
            // declares one, matching the sequential flow.
        } else {
            pass.relabel("llama dense chunk cb (no head)");
        }
        let t_wait = Instant::now();
        self.phases.final_cb_gpu_nanos += (pass.commit_and_wait_with_gpu_time() * 1e9) as u64;
        self.phases.final_wait_nanos += t_wait.elapsed().as_nanos() as u64;
        self.kv.advance_by(m);

        let skip_head = !want_head;
        let last_position = start_position + m - 1;
        if let Some(capture) = self.resid_capture.as_mut() {
            capture.record_pass(last_position, skip_head);
        }

        if !want_head {
            return Ok(());
        }
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
}
