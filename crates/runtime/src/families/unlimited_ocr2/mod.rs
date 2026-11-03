//! The `unlimited_ocr2` decode flow: a DeepSeek-V2-shaped MoE decoder whose
//! attention is DASH block-sparse over page2d RoPE. `state.rs`'s header has
//! the layer pseudocode, `compute::dash` the routing arithmetic, and
//! `docs/UNLIMITED_OCR2_PHASE0.md` the facts.
//!
//! Each layer commits twice: once so the host can read the normed row and
//! route, once so it can read the expert gate. Prefill is the same per-token
//! path as decode, which is equal to upstream's chunks because a query only
//! routes to blocks that end `local_blocks` before its own, so they are
//! complete under either driver. Tensor names are the canonical install's
//! (`crates/repack/src/unlimited_ocr2_config.rs`).

mod moe;
mod state;

pub(crate) use state::RealUnlimitedOcr2State;

use std::time::Instant;

use compute::dash::{block_capacity, dash_select, router_index_qk, summarize_block};
use foundation::LogitValue;
use half::f16;

use crate::real_forward::{RealForwardError, RealForwardRunner};
use crate::real_forward_dispatch::{encode_embed_any, encode_gemv_any};
use crate::real_forward_utils::{f16_slice_to_le_bytes, norm_view};

pub(crate) fn layer_tensor(layer: usize, suffix: &str) -> String {
    format!("language_model.model.layers.{layer}.{suffix}")
}

/// Route one layer's query: fold its index key into the block state, select
/// the keys, and write the list the attention kernel reads.
fn route_layer(
    state: &mut RealUnlimitedOcr2State,
    layer: usize,
    position: usize,
    normed: &[f32],
    in_prompt: bool,
) {
    let d = state.params.index_dim;
    let bs = state.params.block_size;
    let mut q = vec![0.0f32; d];
    let mut k = vec![0.0f32; d];
    router_index_qk(
        normed,
        &state.routers[layer],
        &state.angles,
        state.rms_eps,
        &mut q,
        &mut k,
    );
    let router = &mut state.layers[layer];
    router.pending.extend_from_slice(&k);
    if router.pending.len() == bs * d {
        let n = router.summaries.len();
        router.summaries.resize(n + d, 0.0);
        summarize_block(
            &router.pending,
            &state.routers[layer].head_cls,
            &mut router.summaries[n..],
        );
        router.pending.clear();
    }
    let nblk = (position + 1).div_ceil(bs);
    let capacity = block_capacity(nblk);
    if in_prompt {
        state
            .pages
            .prefill_candidates(position, nblk, &mut state.candidates);
    } else {
        state
            .decode
            .candidates(&state.pages, capacity, &mut state.candidates);
    }
    dash_select(
        &q,
        &state.layers[layer].summaries,
        &state.candidates,
        capacity,
        position,
        state.dash.idx_scale[layer],
        state.dash.floor[layer],
        &state.params,
        &mut state.selection,
    );
    let positions: Vec<u8> = state
        .selection
        .positions
        .iter()
        .flat_map(|p| p.to_le_bytes())
        .collect();
    let bias: Vec<u8> = state
        .selection
        .bias
        .iter()
        .flat_map(|b| b.to_le_bytes())
        .collect();
    gpu::write_buffer_bytes(&state.positions, 0, &positions);
    gpu::write_buffer_bytes(&state.bias, 0, &bias);
    gpu::write_buffer_bytes(
        &state.count,
        0,
        &(state.selection.positions.len() as u32).to_le_bytes(),
    );
}

impl RealForwardRunner {
    /// Install the prompt the next generation will prefill: its full token ids
    /// and one embedding row per image token. DASH needs the whole prompt up
    /// front, because a position's page depends on where the last image is.
    pub fn set_unlimited_ocr2_prompt(
        &mut self,
        ids: &[i32],
        image_rows: Vec<f16>,
    ) -> Result<(), RealForwardError> {
        let hidden = self.arch.hidden_size as usize;
        self.real_unlimited_ocr2
            .as_mut()
            .ok_or_else(|| {
                RealForwardError::Unsupported("this install is not unlimited_ocr2".to_string())
            })?
            .set_prompt(ids, image_rows, hidden)
    }

    /// The last layer's routed keys and bias, and the current rope angles. An
    /// untrained fixture's logits cannot show which keys a query read, because
    /// rows also travel between tokens through the local window.
    #[doc(hidden)]
    pub fn unlimited_ocr2_last_route(&self) -> Option<(Vec<u32>, Vec<f32>, Vec<f32>)> {
        self.real_unlimited_ocr2.as_ref().map(|s| {
            (
                s.selection.positions.clone(),
                s.selection.bias.clone(),
                s.angles.clone(),
            )
        })
    }

    /// The ids an image prompt needs, when this install is this family.
    pub fn unlimited_ocr2_image_ids(&self) -> Option<turbospark_vision_io::Uocr2ImageIds> {
        self.real_unlimited_ocr2
            .as_ref()
            .map(|s| turbospark_vision_io::Uocr2ImageIds {
                bos: 0,
                page: s.params.page_token_id,
                image: s.params.image_token_id,
            })
    }

    /// Encode `pages` and install `expanded`, the prompt from
    /// `uocr2_expand_prompt`, as the next generation's. Encode and generate
    /// belong under one lock, as for every tower here.
    pub fn attach_unlimited_ocr2_pages(
        &mut self,
        pages: &[turbospark_vision_io::Uocr2Page],
        expanded: &[i32],
    ) -> Result<(), RealForwardError> {
        let mut rows = Vec::new();
        for page in pages {
            rows.extend(self.encode_unlimited_ocr2_page(page)?);
        }
        self.set_unlimited_ocr2_prompt(expanded, rows)
    }

    /// One page's fused image rows, in prompt order. Reads the tower file on the
    /// first image.
    pub fn encode_unlimited_ocr2_page(
        &mut self,
        page: &turbospark_vision_io::Uocr2Page,
    ) -> Result<Vec<f16>, RealForwardError> {
        let mut tower = self.take_unlimited_ocr2_tower()?;
        let rows = tower.encode_page(&mut self.context, page);
        self.real_unlimited_ocr2
            .as_mut()
            .expect("checked by take")
            .tower = Some(tower);
        rows
    }

    /// One view with its intermediate rows, for the cross-engine parity gate.
    #[doc(hidden)]
    pub fn unlimited_ocr2_view_stages(
        &mut self,
        pixels: &[f32],
        side: usize,
    ) -> Result<crate::vision::unlimited_ocr2::Uocr2Stages, RealForwardError> {
        let mut stages = crate::vision::unlimited_ocr2::Uocr2Stages::default();
        let mut tower = self.take_unlimited_ocr2_tower()?;
        let ran = tower.run_view(&mut self.context, pixels, side, Some(&mut stages));
        self.real_unlimited_ocr2
            .as_mut()
            .expect("checked by take")
            .tower = Some(tower);
        ran?;
        Ok(stages)
    }

    /// The tower, opened if this is the first image. Taken rather than borrowed,
    /// because running it needs `self.context` mutably; callers put it back.
    fn take_unlimited_ocr2_tower(
        &mut self,
    ) -> Result<crate::vision::unlimited_ocr2::Uocr2Tower, RealForwardError> {
        let state = self.real_unlimited_ocr2.as_mut().ok_or_else(|| {
            RealForwardError::Unsupported("this install is not unlimited_ocr2".to_string())
        })?;
        match state.tower.take() {
            Some(tower) => Ok(tower),
            None => crate::vision::unlimited_ocr2::Uocr2Tower::open(&self.context, &state.dir),
        }
    }

    /// Forget the installed prompt, so the next generation is a text prompt.
    pub fn clear_unlimited_ocr2_prompt(&mut self) {
        if let Some(state) = self.real_unlimited_ocr2.as_mut() {
            state.clear_prompt();
        }
    }

    pub(crate) fn produce_real_unlimited_ocr2(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let mut state = self
            .real_unlimited_ocr2
            .take()
            .expect("unlimited_ocr2 state present");
        let result = self.produce_unlimited_ocr2_inner(&mut state, token, position, logits);
        self.real_unlimited_ocr2 = Some(state);
        self.phases.calls += 1;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn produce_unlimited_ocr2_inner(
        &mut self,
        state: &mut RealUnlimitedOcr2State,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let hidden = arch.hidden_size as usize;
        let vocab = arch.vocab_size as usize;
        let head_dim = arch.head_dim as u32;
        let heads = arch.num_heads as u32;
        let inter = state.dense_inter;
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
        let in_prompt = position < state.pages.length;
        // The bookkeeping was derived from these ids. A caller feeding others
        // would attend the wrong pages with no other symptom.
        if in_prompt && state.prompt_ids[position] != token {
            return Err(RealForwardError::Unsupported(format!(
                "position {position} carries token {token}, but the installed prompt has {}; \
                 call set_unlimited_ocr2_prompt for this prompt",
                state.prompt_ids[position]
            )));
        }

        state.angles_for(token, position);
        let angle_bytes: Vec<u8> = state.angles.iter().flat_map(|a| a.to_le_bytes()).collect();
        gpu::write_buffer_bytes(&state.angles_buffer, 0, &angle_bytes);

        let mut pass = self.context.begin_pass_labeled("uocr2 embed+norm");
        let image_row = if in_prompt {
            state.image_row_at.get(position).copied().flatten()
        } else {
            None
        };
        match image_row {
            Some(row) => {
                let at = row as usize * hidden;
                gpu::write_buffer_bytes(
                    &self.scratch.x,
                    0,
                    &f16_slice_to_le_bytes(&state.image_rows[at..at + hidden]),
                );
            }
            None => encode_embed_any(
                &mut self.context,
                &pass,
                &self.weights,
                &self.index,
                "language_model.model.embed_tokens.weight",
                (&self.scratch.x, 0),
                token as u32,
                hidden as u32,
                vocab,
                1.0,
            )?,
        }

        for layer in 0..arch.num_layers as usize {
            let input_norm = norm_view(
                &self.weights,
                &self.index,
                &layer_tensor(layer, "input_layernorm.weight"),
                hidden,
            )?;
            gpu::encode_rms_norm_bf16w(
                &mut self.context,
                &pass,
                (&self.scratch.x, 0),
                input_norm,
                (&self.scratch.normed, 0),
                hidden as u32,
                state.rms_eps,
            )
            .map_err(gpu_err)?;
            let t_wait = Instant::now();
            let committed = pass.commit();
            self.phases.cb1_gpu_nanos += (committed.wait_with_gpu_time() * 1e9) as u64;
            self.phases.gpu_wait_nanos += t_wait.elapsed().as_nanos() as u64;

            let normed: Vec<f32> = gpu::read_half_buffer(&self.scratch.normed, hidden)
                .iter()
                .map(|v| v.to_f32())
                .collect();
            route_layer(state, layer, position, &normed, in_prompt);

            pass = self.context.begin_pass_labeled("uocr2 attn+router");
            let (k_buf, k_off) = self.kv.k_slot(layer, position);
            let (v_buf, v_off) = self.kv.v_slot(layer, position);
            let (k_row, v_row) = ((k_buf, k_off as u64), (v_buf, v_off as u64));
            for (suffix, out) in [
                ("self_attn.q_proj.weight", (&self.scratch.q, 0u64)),
                ("self_attn.k_proj.weight", k_row),
                ("self_attn.v_proj.weight", v_row),
            ] {
                encode_gemv_any(
                    &mut self.context,
                    &pass,
                    &self.weights,
                    &self.index,
                    &layer_tensor(layer, suffix),
                    (heads * head_dim) as usize,
                    hidden,
                    (&self.scratch.normed, 0),
                    out,
                )?;
            }
            // The kernel's angle is `position * table[pair]`, so position 1
            // makes the table the angles, page2d or yarn alike. V stays
            // unrotated.
            for row in [(&self.scratch.q, 0u64), k_row] {
                gpu::encode_rope_neox_freqs(
                    &mut self.context,
                    &pass,
                    row,
                    1,
                    heads,
                    head_dim,
                    head_dim / 2,
                    (&state.angles_buffer, 0),
                    1.0,
                )
                .map_err(gpu_err)?;
            }
            let (k_base, _) = self.kv.k_slot(layer, 0);
            let (v_base, _) = self.kv.v_slot(layer, 0);
            gpu::encode_attention_decode_indexed_biased(
                &mut self.context,
                &pass,
                (&self.scratch.q, 0),
                k_base,
                v_base,
                (&state.positions, 0),
                (&state.bias, 0),
                (&state.count, 0),
                state.params.max_keys() as u32,
                &self.scratch.attn,
                (&self.scratch.attn_out, 0),
                head_dim,
                heads,
                heads,
                arch.attention_scale as f32,
            )
            .map_err(gpu_err)?;
            encode_gemv_any(
                &mut self.context,
                &pass,
                &self.weights,
                &self.index,
                &layer_tensor(layer, "self_attn.o_proj.weight"),
                hidden,
                (heads * head_dim) as usize,
                (&self.scratch.attn_out, 0),
                (&self.scratch.o, 0),
            )?;
            gpu::encode_residual_add(
                &mut self.context,
                &pass,
                (&self.scratch.x, 0),
                (&self.scratch.o, 0),
                hidden as u32,
            )
            .map_err(gpu_err)?;
            let post_attn = norm_view(
                &self.weights,
                &self.index,
                &layer_tensor(layer, "post_attention_layernorm.weight"),
                hidden,
            )?;
            gpu::encode_rms_norm_bf16w(
                &mut self.context,
                &pass,
                (&self.scratch.x, 0),
                post_attn,
                (&state.moe_x, 0),
                hidden as u32,
                state.rms_eps,
            )
            .map_err(gpu_err)?;

            if layer < state.lead {
                encode_gemv_any(
                    &mut self.context,
                    &pass,
                    &self.weights,
                    &self.index,
                    &layer_tensor(layer, "mlp.gate_proj.weight"),
                    inter,
                    hidden,
                    (&state.moe_x, 0),
                    (&state.ffn_a, 0),
                )?;
                encode_gemv_any(
                    &mut self.context,
                    &pass,
                    &self.weights,
                    &self.index,
                    &layer_tensor(layer, "mlp.up_proj.weight"),
                    inter,
                    hidden,
                    (&state.moe_x, 0),
                    (&state.ffn_b, 0),
                )?;
                gpu::encode_silu_mul(
                    &mut self.context,
                    &pass,
                    (&state.ffn_a, 0),
                    (&state.ffn_b, 0),
                    (&state.ffn_a, 0),
                    inter as u32,
                )
                .map_err(gpu_err)?;
                encode_gemv_any(
                    &mut self.context,
                    &pass,
                    &self.weights,
                    &self.index,
                    &layer_tensor(layer, "mlp.down_proj.weight"),
                    hidden,
                    inter,
                    (&state.ffn_a, 0),
                    (&self.scratch.o, 0),
                )?;
                gpu::encode_residual_add(
                    &mut self.context,
                    &pass,
                    (&self.scratch.x, 0),
                    (&self.scratch.o, 0),
                    hidden as u32,
                )
                .map_err(gpu_err)?;
                continue;
            }

            encode_gemv_any(
                &mut self.context,
                &pass,
                &self.weights,
                &self.index,
                &layer_tensor(layer, "mlp.gate.weight"),
                arch.num_experts as usize,
                hidden,
                (&state.moe_x, 0),
                (&self.scratch.router_logits, 0),
            )?;
            let t_wait = Instant::now();
            let committed = pass.commit();
            self.phases.cb1_gpu_nanos += (committed.wait_with_gpu_time() * 1e9) as u64;
            self.phases.gpu_wait_nanos += t_wait.elapsed().as_nanos() as u64;

            pass = self.context.begin_pass_labeled("uocr2 routed");
            moe::encode_layer_moe(
                &mut self.context,
                &pass,
                &self.weights,
                &self.index,
                &self.scratch,
                state,
                &mut self.streamers,
                &self.slot_buffers,
                &self.mapped,
                self.routed_blobs.as_ref(),
                &self.moe_offsets,
                &self.routed_layouts,
                &mut self.phases,
                layer,
                hidden,
                arch.moe_intermediate_size as u32,
                arch.num_experts as usize,
                arch.top_k_experts as usize,
            )?;
        }

        if !self.skip_head {
            pass.relabel("uocr2 head");
            let final_norm = norm_view(
                &self.weights,
                &self.index,
                "language_model.model.norm.weight",
                hidden,
            )?;
            gpu::encode_rms_norm_bf16w(
                &mut self.context,
                &pass,
                (&self.scratch.x, 0),
                final_norm,
                (&self.scratch.normed, 0),
                hidden as u32,
                state.rms_eps,
            )
            .map_err(gpu_err)?;
            encode_gemv_any(
                &mut self.context,
                &pass,
                &self.weights,
                &self.index,
                "language_model.lm_head.weight",
                vocab,
                hidden,
                (&self.scratch.normed, 0),
                (&self.scratch.logits, 0),
            )?;
        }
        let t_wait = Instant::now();
        self.phases.final_cb_gpu_nanos += (pass.commit_and_wait_with_gpu_time() * 1e9) as u64;
        self.phases.final_wait_nanos += t_wait.elapsed().as_nanos() as u64;
        self.kv.advance();

        if self.skip_head {
            return Ok(());
        }
        // Raw logits, never probabilities (AGENTS.md Gotcha 16); no softcap.
        if vocab != logits.len() {
            return Err(RealForwardError::Unsupported(format!(
                "vocab mismatch: model has {vocab}, caller expected {}",
                logits.len()
            )));
        }
        gpu::read_buffer_f16_into(&self.scratch.logits, 0, logits);
        Ok(())
    }
}
