//! The real-checkpoint Gemma 4 decode flow for [`RealForwardRunner`]:
//! learned (BF16) norms everywhere, per-head q/k norms and the no-scale
//! per-head v norm, the INT8 router GEMV with its pre-folded effective
//! scale, softmax-over-top-k routing weights times `per_expert_scale`
//! (the `router_topk_select_k8` kernel's semantics, computed on the host
//! where the logits are already read back for the expert `pread`), the
//! INT8 shared-expert branch summed alongside the routed experts through
//! Gemma's FFN sandwich norms, and the per-layer `layer_scalar` multiply.
//! Mirrors the layer flow documented at the top of the Swift
//! `RealForwardRunner.swift`:
//!
//! ```text
//! h = embed_lookup_int4(token) * sqrt(H)
//! for L:
//!   a = rmsnorm_bf16w(h, input_layernorm)
//!   Q = q_proj(a)  K = k_proj(a)  V = (SWA) v_proj(a) | (full) k_proj(a)
//!   per-head q/k_norm (bf16w), per-head v_norm (no_scale)
//!   NeoX RoPE on Q + K (full rotation for SWA, proportional for full)
//!   attn = attention(Q, K-cache, V-cache); h += rmsnorm_bf16w(o_proj(attn), post_attn)
//!   router_x = rmsnorm_no_scale(h); dense_x = rmsnorm_bf16w(h, pre_ffn)
//!   routed_x = rmsnorm_bf16w(h, pre_ffn2)
//!   idx, w = router_topk_gemma4(router_x, effective_scale[L], per_expert_scale[L])
//!   h1 = rmsnorm_bf16w(SharedExpertInt8(dense_x), post_ffn_1)
//!   h2 = rmsnorm_bf16w(moe(routed_x, idx, w), post_ffn_2)
//!   h += rmsnorm_bf16w(h1 + h2, post_ffn); h *= layer_scalar[L]
//! logits = softcap(int4_gemv(rmsnorm_bf16w(h, model.norm), embed^T))
//! ```
//!
//! The head stops at the softcapped logits (what HF's
//! `Gemma*ForCausalLM.forward` returns and what `LogitProducer` documents),
//! not at probabilities: the softmax belongs to `selection::select`, which
//! runs its own. The Swift original fuses cap+softmax because it samples on
//! the GPU from probs; doing that here and then handing probs to `select`
//! softmaxes twice and flattens the distribution to near-uniform over the
//! surviving top-k.
//!
//! Selected by `open()` when the resident index carries the source
//! checkpoint's verbatim `language_model.` tensor names (what
//! `mrefrust_repack::write_gemma4_install` writes). The synthetic
//! short-name installs keep the plain flow in `real_forward.rs`.

use std::time::Instant;

use foundation::LogitValue;
use half::f16;
use model_io::{ArchConfig, ResidentIndex};

use crate::real_forward::{
    f16_slice_to_le_bytes, resident_matrix, RealForwardError, RealForwardRunner,
};

const RMS_EPS: f32 = 1e-6;

/// Per-layer real-checkpoint state built once at open.
pub(crate) struct RealGemmaState {
    /// BF16 `[hidden]` per layer: the checkpoint's `router.scale` with
    /// `1/sqrt(hidden)` pre-folded (the Swift effective-scale buffers).
    effective_scale: Vec<gpu::MetalBuffer>,
    /// Decoded `router.per_expert_scale` per layer.
    per_expert_scale: Vec<Vec<f32>>,
    /// Decoded `layer_scalar` per layer.
    layer_scalar: Vec<f32>,
    /// FP32 router logits (the INT8 router kernel writes float).
    router_logits_f32: gpu::MetalBuffer,
    /// `[hidden]` FP16 branch scratch.
    dense_x: gpu::MetalBuffer,
    routed_x: gpu::MetalBuffer,
    router_x: gpu::MetalBuffer,
    h1: gpu::MetalBuffer,
    h2: gpu::MetalBuffer,
}

pub(crate) fn layer_tensor(layer: usize, suffix: &str) -> String {
    format!("language_model.model.layers.{layer}.{suffix}")
}

pub(crate) fn entry<'a>(
    index: &'a ResidentIndex,
    name: &str,
) -> Result<&'a model_io::ResidentIndexEntry, RealForwardError> {
    index
        .entries
        .get(name)
        .ok_or_else(|| RealForwardError::MissingTensor(name.to_string()))
}

/// Decodes a resident BF16 tensor to `f32` host values.
pub(crate) fn read_bf16_host(
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    name: &str,
) -> Result<Vec<f32>, RealForwardError> {
    let e = entry(index, name)?;
    let base = index.header.index_size;
    let local = (e.file_offset - base) as usize;
    let bytes = &weights.data()[local..local + e.size_bytes as usize];
    Ok(bytes
        .chunks_exact(2)
        .map(|c| compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect())
}

/// Resolves a raw (norm) tensor to a `(buffer, gpu offset)` view, checking
/// its byte size against the expected element count.
pub(crate) fn norm_view<'a>(
    weights: &'a gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    name: &str,
    expect_elems: usize,
) -> Result<(&'a gpu::MetalBuffer, u64), RealForwardError> {
    let e = entry(index, name)?;
    if e.size_bytes as usize != expect_elems * 2 {
        return Err(RealForwardError::Unsupported(format!(
            "tensor {name}: {} bytes does not match expected BF16 [{expect_elems}]",
            e.size_bytes
        )));
    }
    let base = index.header.index_size;
    Ok((weights.buffer(), weights.gpu_offset(e.file_offset - base)))
}

/// Resolves a packed projection by its dtype tag: 4 = INT4-affine,
/// 5 = INT8-affine (the resident writer's tags), and encodes the matching
/// offset-bound GEMV.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemv_any(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    name: &str,
    rows: usize,
    cols: usize,
    x: (&gpu::MetalBuffer, u64),
    y: (&gpu::MetalBuffer, u64),
) -> Result<(), RealForwardError> {
    let e = entry(index, name)?;
    let base = index.header.index_size;
    match e.dtype {
        5 => {
            if e.size_bytes as usize != rows * cols {
                return Err(RealForwardError::Unsupported(format!(
                    "tensor {name}: INT8 packed size {} does not match {rows}x{cols}",
                    e.size_bytes
                )));
            }
            let w = gpu::Int8ResidentMatrix {
                buffer: weights.buffer(),
                weights_offset: weights.gpu_offset(e.file_offset - base),
                scales_offset: weights.gpu_offset(e.scale_offset - base),
                biases_offset: weights.gpu_offset(e.bias_offset - base),
                rows,
                cols,
            };
            gpu::encode_dequant_int8_gemv_resident(context, pass, &w, x, y)
                .map_err(RealForwardError::Gpu)
        }
        4 => {
            let w = resident_matrix(weights, index, name, rows, cols)?;
            gpu::encode_dequant_int4_gemv_resident(context, pass, &w, x, y)
                .map_err(RealForwardError::Gpu)
        }
        // Named rather than defaulted. This arm used to be `_ => int4`,
        // which meant any future dtype tag was read as INT4-affine: no
        // error, just wrong numbers, since the three planar regions an
        // INT4 tensor expects do not exist in a block-quantized one. The
        // GGUF tags (ROADMAP Phase G) are the first tags able to reach it.
        other => Err(RealForwardError::Unsupported(format!(
            "tensor {name}: dtype {other} has no dispatched GEMV kernel"
        ))),
    }
}

/// The `router_topk_select_k8` kernel's semantics on the host: top-`k` by
/// score with ties preferring the lower expert index, softmax over the
/// selected scores only, each weight multiplied by that expert's
/// `per_expert_scale`. (Distinct from the synthetic path's
/// softmax-over-all-then-renormalize `topk_softmax`.)
pub(crate) fn router_topk_gemma4(
    logits: &[f32],
    k: usize,
    per_expert_scale: &[f32],
) -> (Vec<usize>, Vec<f32>) {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    let selected: Vec<usize> = order.into_iter().take(k).collect();
    let max = logits[selected[0]];
    let exps: Vec<f32> = selected.iter().map(|&i| (logits[i] - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let weights = selected
        .iter()
        .zip(exps.iter())
        .map(|(&i, e)| e / sum * per_expert_scale[i])
        .collect();
    (selected, weights)
}

impl RealGemmaState {
    pub(crate) fn build(
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        if !arch.ffn_sandwich_norms || !arch.router_scaled {
            return Err(RealForwardError::Unsupported(
                "real-checkpoint flow currently implements Gemma 4 only \
                 (ffn_sandwich_norms + router_scaled)"
                    .to_string(),
            ));
        }
        if arch.num_experts <= 0 || arch.top_k_experts <= 0 {
            return Err(RealForwardError::Unsupported(
                "real Gemma 4 installs are MoE; num_experts/top_k must be positive".to_string(),
            ));
        }
        if arch.top_k_experts as usize > gpu::MAX_STREAMED_EXPERTS {
            return Err(RealForwardError::Unsupported(format!(
                "top_k {} exceeds the {}-slot MoE kernels",
                arch.top_k_experts,
                gpu::MAX_STREAMED_EXPERTS
            )));
        }

        let hidden = arch.hidden_size as usize;
        let num_layers = arch.num_layers as usize;
        let num_experts = arch.num_experts as usize;
        let inv_sqrt_d = 1.0f32 / (hidden as f32).sqrt();

        let mut effective_scale = Vec::with_capacity(num_layers);
        let mut per_expert_scale = Vec::with_capacity(num_layers);
        let mut layer_scalar = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            // Pre-fold 1/sqrt(D) into router.scale once per open, as the
            // Swift runner does per generation.
            let scale = read_bf16_host(weights, index, &layer_tensor(layer, "router.scale"))?;
            if scale.len() != hidden {
                return Err(RealForwardError::Unsupported(format!(
                    "router.scale layer {layer}: {} elements, expected {hidden}",
                    scale.len()
                )));
            }
            let folded: Vec<u8> = scale
                .iter()
                .flat_map(|&v| compute::f32_to_bf16(v * inv_sqrt_d).to_le_bytes())
                .collect();
            let buffer = context.new_output_buffer(folded.len() as u64);
            gpu::write_buffer_bytes(&buffer, 0, &folded);
            effective_scale.push(buffer);

            let pes = read_bf16_host(
                weights,
                index,
                &layer_tensor(layer, "router.per_expert_scale"),
            )?;
            if pes.len() != num_experts {
                return Err(RealForwardError::Unsupported(format!(
                    "per_expert_scale layer {layer}: {} elements, expected {num_experts}",
                    pes.len()
                )));
            }
            per_expert_scale.push(pes);

            let scalar = read_bf16_host(weights, index, &layer_tensor(layer, "layer_scalar"))?;
            layer_scalar.push(*scalar.first().ok_or_else(|| {
                RealForwardError::Unsupported(format!("layer_scalar layer {layer} is empty"))
            })?);
        }

        let halfs = |n: usize| context.new_output_buffer((n.max(1) * 2) as u64);
        Ok(Self {
            router_logits_f32: context.new_output_buffer((num_experts * 4) as u64),
            dense_x: halfs(hidden),
            routed_x: halfs(hidden),
            router_x: halfs(hidden),
            h1: halfs(hidden),
            h2: halfs(hidden),
            effective_scale,
            per_expert_scale,
            layer_scalar,
        })
    }
}

impl RealForwardRunner {
    /// The shared (dense) expert branch: INT8 gate/up on `dense_x`, gated
    /// activation, down projection, then `post_feedforward_layernorm_1`,
    /// encoded into its own command buffer and committed without waiting.
    ///
    /// It reads only `dense_x`, which the router's command buffer already
    /// produced, so the caller can commit this before waiting on the
    /// router and let it run on the GPU through the router readback and
    /// the blocking expert `pread`. Buffers on one queue execute in commit
    /// order, so nothing needs an explicit fence. That overlap is what the
    /// Swift original buys with an `MTLSharedEvent` signalled mid-buffer;
    /// metal-rs 0.33 binds no wait-until-signaled-value, and spinning on
    /// `signaledValue` would steal the SoC power budget the GPU needs, so
    /// a second command buffer buys it instead.
    fn encode_shared_expert_branch(
        &mut self,
        layer: usize,
        hidden: usize,
        inter: usize,
        use_silu: bool,
    ) -> Result<(), RealForwardError> {
        let gpu_err = RealForwardError::Gpu;
        let shared_pass = self.context.begin_pass_labeled("shared-expert cb");
        // Shared INT8 branch on dense_x -> h1, then post_ffn_1.
        let real = self.real.as_ref().expect("real state present");
        for (name, rows, cols, x_buf, y_buf) in [
            (
                layer_tensor(layer, "mlp.gate_proj.weight"),
                inter,
                hidden,
                &real.dense_x,
                &self.scratch.ffn_gate,
            ),
            (
                layer_tensor(layer, "mlp.up_proj.weight"),
                inter,
                hidden,
                &real.dense_x,
                &self.scratch.ffn_up,
            ),
        ] {
            encode_gemv_any(
                &mut self.context,
                &shared_pass,
                &self.weights,
                &self.index,
                &name,
                rows,
                cols,
                (x_buf, 0),
                (y_buf, 0),
            )?;
        }
        let act = if use_silu {
            gpu::encode_silu_mul
        } else {
            gpu::encode_gelu_mul
        };
        act(
            &mut self.context,
            &shared_pass,
            (&self.scratch.ffn_gate, 0),
            (&self.scratch.ffn_up, 0),
            (&self.scratch.ffn_act, 0),
            inter as u32,
        )
        .map_err(gpu_err)?;
        let real = self.real.as_ref().expect("real state present");
        encode_gemv_any(
            &mut self.context,
            &shared_pass,
            &self.weights,
            &self.index,
            &layer_tensor(layer, "mlp.down_proj.weight"),
            hidden,
            inter,
            (&self.scratch.ffn_act, 0),
            (&real.h1, 0),
        )?;
        let post_ffn1 = norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "post_feedforward_layernorm_1.weight"),
            hidden,
        )?;
        gpu::encode_rms_norm_bf16w(
            &mut self.context,
            &shared_pass,
            (&real.h1, 0),
            post_ffn1,
            (&real.h1, 0),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
        shared_pass.commit();
        Ok(())
    }

    /// Times the whole forward pass into `phases.total_nanos`; the inner
    /// function accumulates the per-phase buckets it is carved into.
    pub(crate) fn produce_real_gemma4(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = Instant::now();
        let result = self.produce_real_gemma4_inner(token, position, logits);
        self.phases.calls += 1;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn produce_real_gemma4_inner(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let hidden = arch.hidden_size as usize;
        let inter = arch.intermediate_size as usize;
        let moe_inter = arch.moe_intermediate_size as u32;
        let num_heads = arch.num_heads as u32;
        let vocab = arch.vocab_size as usize;
        let num_experts = arch.num_experts as usize;
        let top_k = arch.top_k_experts as usize;
        let attn_scale = arch.attention_scale as f32;
        let use_silu = arch.hidden_activation.contains("silu");
        let embed_scale = if arch.embedding_scaled_by_sqrt_hidden {
            (hidden as f32).sqrt()
        } else {
            1.0
        };
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
        let seq_len = (position + 1) as u32;

        let embed_name = "language_model.model.embed_tokens.weight";
        let embed = entry(&self.index, embed_name)?;
        let base = self.index.header.index_size;
        let embed_table = self.weights.gpu_offset(embed.file_offset - base);
        let embed_scales = self.weights.gpu_offset(embed.scale_offset - base);
        let embed_biases = self.weights.gpu_offset(embed.bias_offset - base);

        let mut pass = self.context.begin_pass_labeled("cb1 (attn+router)");
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

        // Swift's one-layer-pipelined routed command buffer: a layer's
        // routed-expert work commits at the END of that layer (so the GPU
        // starts it during the host's next-layer attention encode) and is
        // retired after the next layer's router wait, where it has
        // provably completed (committed earlier on the same queue). Local
        // on purpose: `CommittedPass` is not Send, and a local guarantees
        // the pipeline drains within this call. Depth is exactly one.
        let mut pending_routed: Option<gpu::CommittedPass> = None;

        for layer in 0..arch.num_layers as usize {
            let is_full = arch.full_attention_layer_mask[layer] == 1;
            let head_dim_l = if is_full {
                arch.full_head_dim
            } else {
                arch.head_dim
            } as u32;
            let num_kv_l = if is_full {
                arch.num_full_kv_heads
            } else {
                arch.num_kv_heads
            } as u32;
            let q_dim = (num_heads * head_dim_l) as usize;
            let kv_dim = (num_kv_l * head_dim_l) as usize;

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
                RMS_EPS,
            )
            .map_err(gpu_err)?;

            // Projections. Under the K=V quirk, full layers reuse the K
            // projection for V (a separately-written, separately-normed V
            // row); SWA layers have a real v_proj.
            let q_name = layer_tensor(layer, "self_attn.q_proj.weight");
            let k_name = layer_tensor(layer, "self_attn.k_proj.weight");
            let v_name = if is_full && arch.attention_k_eq_v {
                k_name.clone()
            } else {
                layer_tensor(layer, "self_attn.v_proj.weight")
            };
            let o_name = layer_tensor(layer, "self_attn.o_proj.weight");
            let (k_buf, k_off) = self.kv.k_slot(layer, position);
            let (v_buf, v_off) = self.kv.v_slot(layer, position);
            encode_gemv_any(
                &mut self.context,
                &pass,
                &self.weights,
                &self.index,
                &q_name,
                q_dim,
                hidden,
                (&self.scratch.normed, 0),
                (&self.scratch.q, 0),
            )?;
            encode_gemv_any(
                &mut self.context,
                &pass,
                &self.weights,
                &self.index,
                &k_name,
                kv_dim,
                hidden,
                (&self.scratch.normed, 0),
                (k_buf, k_off as u64),
            )?;
            encode_gemv_any(
                &mut self.context,
                &pass,
                &self.weights,
                &self.index,
                &v_name,
                kv_dim,
                hidden,
                (&self.scratch.normed, 0),
                (v_buf, v_off as u64),
            )?;

            // Per-head norms, in place: q/k learned (bf16w), v no-scale.
            let q_norm = norm_view(
                &self.weights,
                &self.index,
                &layer_tensor(layer, "self_attn.q_norm.weight"),
                head_dim_l as usize,
            )?;
            let k_norm = norm_view(
                &self.weights,
                &self.index,
                &layer_tensor(layer, "self_attn.k_norm.weight"),
                head_dim_l as usize,
            )?;
            gpu::encode_rms_norm_bf16w_perhead(
                &mut self.context,
                &pass,
                (&self.scratch.q, 0),
                q_norm,
                (&self.scratch.q, 0),
                num_heads,
                head_dim_l,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
            gpu::encode_rms_norm_bf16w_perhead(
                &mut self.context,
                &pass,
                (k_buf, k_off as u64),
                k_norm,
                (k_buf, k_off as u64),
                num_kv_l,
                head_dim_l,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
            gpu::encode_rms_norm_no_scale_perhead(
                &mut self.context,
                &pass,
                (v_buf, v_off as u64),
                (v_buf, v_off as u64),
                num_kv_l,
                head_dim_l,
                RMS_EPS,
            )
            .map_err(gpu_err)?;

            // NeoX RoPE: SWA layers rotate the full head at the SWA theta;
            // full layers rotate the proportional subset at the full theta
            // (the Swift runner truncates the pair count).
            let (rotated_pairs, theta) = if is_full {
                (
                    (arch.full_head_dim as f64 * arch.partial_rotary_factor / 2.0) as u32,
                    arch.full_rope_theta as f32,
                )
            } else {
                (head_dim_l / 2, arch.rope_theta as f32)
            };
            gpu::encode_rope_proportional_neox(
                &mut self.context,
                &pass,
                (&self.scratch.q, 0),
                position as u32,
                num_heads,
                head_dim_l,
                rotated_pairs,
                theta,
            )
            .map_err(gpu_err)?;
            gpu::encode_rope_proportional_neox(
                &mut self.context,
                &pass,
                (k_buf, k_off as u64),
                position as u32,
                num_kv_l,
                head_dim_l,
                rotated_pairs,
                theta,
            )
            .map_err(gpu_err)?;

            // SWA layers switch to the ring pipeline once seq_len outgrows
            // the ring (Swift activation rule; identity mapping below
            // capacity keeps the linear pipeline byte-identical until the
            // first wrap).
            let (kv_start, active_ring) = if is_full {
                (0, 0)
            } else {
                let ring = self.kv.ring_capacity(layer) as u32;
                (
                    seq_len.saturating_sub(arch.sliding_window as u32),
                    if ring > 0 && seq_len > ring { ring } else { 0 },
                )
            };
            gpu::encode_attention_decode(
                &mut self.context,
                &pass,
                (&self.scratch.q, 0),
                k_buf,
                v_buf,
                &self.scratch.attn,
                (&self.scratch.attn_out, 0),
                head_dim_l,
                num_heads,
                num_kv_l,
                seq_len,
                kv_start,
                active_ring,
                attn_scale,
            )
            .map_err(gpu_err)?;
            encode_gemv_any(
                &mut self.context,
                &pass,
                &self.weights,
                &self.index,
                &o_name,
                hidden,
                q_dim,
                (&self.scratch.attn_out, 0),
                (&self.scratch.o, 0),
            )?;

            // h += rmsnorm_bf16w(attn, post_attention_layernorm)
            let post_attn = norm_view(
                &self.weights,
                &self.index,
                &layer_tensor(layer, "post_attention_layernorm.weight"),
                hidden,
            )?;
            gpu::encode_rms_norm_bf16w(
                &mut self.context,
                &pass,
                (&self.scratch.o, 0),
                post_attn,
                (&self.scratch.o_normed, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
            gpu::encode_residual_add(
                &mut self.context,
                &pass,
                (&self.scratch.x, 0),
                (&self.scratch.o_normed, 0),
                hidden as u32,
            )
            .map_err(gpu_err)?;

            // Branch inputs: router (no-scale), shared (pre_ffn), routed
            // (pre_ffn2) — the fused_post_attn_setup split, unfused.
            let real = self.real.as_ref().expect("real state present");
            gpu::encode_rms_norm_no_scale(
                &mut self.context,
                &pass,
                (&self.scratch.x, 0),
                (&real.router_x, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
            let pre_ffn = norm_view(
                &self.weights,
                &self.index,
                &layer_tensor(layer, "pre_feedforward_layernorm.weight"),
                hidden,
            )?;
            gpu::encode_rms_norm_bf16w(
                &mut self.context,
                &pass,
                (&self.scratch.x, 0),
                pre_ffn,
                (&real.dense_x, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
            let pre_ffn2 = norm_view(
                &self.weights,
                &self.index,
                &layer_tensor(layer, "pre_feedforward_layernorm_2.weight"),
                hidden,
            )?;
            gpu::encode_rms_norm_bf16w(
                &mut self.context,
                &pass,
                (&self.scratch.x, 0),
                pre_ffn2,
                (&real.routed_x, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;

            // INT8 router GEMV with the pre-folded effective scale.
            let router_name = layer_tensor(layer, "router.proj.weight");
            let router_entry = entry(&self.index, &router_name)?;
            if router_entry.dtype != 5 {
                return Err(RealForwardError::Unsupported(format!(
                    "{router_name}: expected INT8 (dtype 5), got dtype {}",
                    router_entry.dtype
                )));
            }
            if router_entry.size_bytes as usize != num_experts * hidden {
                return Err(RealForwardError::Unsupported(format!(
                    "{router_name}: packed size {} does not match {num_experts}x{hidden}",
                    router_entry.size_bytes
                )));
            }
            gpu::encode_router_gemv_gemma4(
                &mut self.context,
                &pass,
                (
                    self.weights.buffer(),
                    self.weights.gpu_offset(router_entry.file_offset - base),
                ),
                (
                    self.weights.buffer(),
                    self.weights.gpu_offset(router_entry.scale_offset - base),
                ),
                (
                    self.weights.buffer(),
                    self.weights.gpu_offset(router_entry.bias_offset - base),
                ),
                (&real.router_x, 0),
                (&real.effective_scale[layer], 0),
                (&real.router_logits_f32, 0),
                num_experts as u32,
                hidden as u32,
            )
            .map_err(gpu_err)?;
            let cb1 = pass.commit();
            if self.shared_cb_overlap {
                self.encode_shared_expert_branch(layer, hidden, inter, use_silu)?;
            }

            let t_wait = Instant::now();
            self.phases.cb1_gpu_nanos += (cb1.wait_with_gpu_time() * 1e9) as u64;
            self.phases.gpu_wait_nanos += t_wait.elapsed().as_nanos() as u64;

            // Retire the previous layer's pipelined routed buffer. It was
            // committed before `cb1`, so it has already completed and this
            // wait is ~free; being explicit makes every host buffer write
            // below safe without completion-order reasoning.
            if let Some(pending) = pending_routed.take() {
                let t_retire = Instant::now();
                self.phases.routed_cb_gpu_nanos += (pending.wait_with_gpu_time() * 1e9) as u64;
                self.phases.pipeline_wait_nanos += t_retire.elapsed().as_nanos() as u64;
            }

            // Host: kernel-semantics top-k + expert streaming.
            let t_router = Instant::now();
            let real = self.real.as_ref().expect("real state present");
            let router_logits = gpu::read_f32_buffer(&real.router_logits_f32, num_experts);
            let (selected, route_weights) =
                router_topk_gemma4(&router_logits, top_k, &real.per_expert_scale[layer]);
            let layer_scalar = real.layer_scalar[layer];

            let streamer = self.streamers[layer].as_mut().ok_or_else(|| {
                RealForwardError::Unsupported(format!(
                    "real Gemma 4 layer {layer} has no packed-expert streamer"
                ))
            })?;
            let plan = streamer.plan_experts_cached(&selected, &std::collections::HashSet::new());
            let (requests, hits) = (plan.experts.len() as u64, plan.hits as u64);
            self.phases.expert_requests += requests;
            self.phases.expert_hits += hits;
            // The router bucket is the readback, the top-k and the slot
            // plan; the hit dispatch and the pread have their own.
            self.phases.router_nanos += t_router.elapsed().as_nanos() as u64;

            // Slot order for this layer's MoE dispatches: cache MISSES
            // first, then cache hits. Both phase-1 kernels read
            // `blob[slot]` for `slot < top_k` and write `acts[slot * F]`,
            // so a misses-first order lets the miss dispatch keep the full
            // argument buffer and offset 0, while the hit dispatch takes
            // its own argument buffer and an `acts` offset. Nothing else in
            // the layer depends on the router's own ranking, as long as
            // `routing_w` is permuted to match.
            let mut is_miss = vec![false; selected.len()];
            for &index in &plan.misses {
                is_miss[index] = true;
            }
            let mut order: Vec<usize> = plan.misses.clone();
            let miss_count = order.len();
            order.extend((0..selected.len()).filter(|&i| !is_miss[i]));
            let hit_count = order.len() - miss_count;

            // The hit experts are already in slot memory, so their phase-1
            // GEMV can run on the GPU through the blocking pread below.
            // It needs its OWN argument buffer: the host rebinds the main
            // one for the full slot list after the pread, which would race
            // a dispatch still reading it. Slot memory itself is safe --
            // `plan` reserves hit slots before picking eviction victims, so
            // the parallel miss reads never write a slot this reads.
            let t_hit = Instant::now();
            let dispatched_hits = self.hit_cb_overlap && hit_count > 0;
            if dispatched_hits {
                let layer_slots = &self.slot_buffers[layer];
                let hit_refs: Vec<(&gpu::MetalBuffer, u64)> = order[miss_count..]
                    .iter()
                    .map(|&i| (&layer_slots[plan.assigned_slots[i]], 0u64))
                    .collect();
                let routed_hits = self
                    .routed_blobs_hits
                    .as_ref()
                    .expect("layout implies blobs");
                routed_hits
                    .bind(&mut self.context, use_silu, &hit_refs)
                    .map_err(gpu_err)?;
                let hit_pass = self.context.begin_pass_labeled("hit-expert cb");
                for &(buffer, _) in &hit_refs {
                    hit_pass.use_read_buffer(buffer);
                }
                let real = self.real.as_ref().expect("real state present");
                let offsets = self.moe_offsets.as_ref().expect("layout implies offsets");
                gpu::encode_moe_phase1(
                    &mut self.context,
                    &hit_pass,
                    routed_hits,
                    offsets,
                    (&real.routed_x, 0),
                    (
                        &self.scratch.moe_acts,
                        miss_count as u64 * moe_inter as u64 * 2,
                    ),
                    hidden as u32,
                    moe_inter,
                    hit_count as u32,
                    use_silu,
                )
                .map_err(gpu_err)?;
                // Committed, never waited on: buffers execute in commit
                // order, so these `acts` rows land before the phase 2
                // encoded below reads them.
                hit_pass.commit();
            }
            self.phases.hit_cb_nanos += t_hit.elapsed().as_nanos() as u64;

            let streamer = self.streamers[layer]
                .as_mut()
                .expect("streamer presence checked above");
            let t_io = Instant::now();
            let slots = streamer
                .execute_expert_cache_plan(&plan)
                .map_err(|e| RealForwardError::Unsupported(format!("expert stream: {e}")))?;
            self.phases.expert_io_nanos += t_io.elapsed().as_nanos() as u64;

            if !self.shared_cb_overlap {
                self.encode_shared_expert_branch(layer, hidden, inter, use_silu)?;
            }
            // Re-borrowed after the encode above, which needs `&mut self`.
            let real = self.real.as_ref().expect("real state present");

            let t_bind = Instant::now();
            // One list in the misses-first slot order, holding each slot's
            // blob AND its routing weight together: the kernel pairs
            // `blob[slot]` with `routing_w[slot]`, and pairing them here
            // too is what keeps the permutation from drifting apart.
            let ordered: Vec<(usize, f32)> = order
                .iter()
                .map(|&i| (slots[i], route_weights[i]))
                .collect();
            let mut routing16 = vec![f16::from_f32(0.0); gpu::MAX_STREAMED_EXPERTS];
            for (slot, &(_, weight)) in ordered.iter().enumerate() {
                routing16[slot] = f16::from_f32(weight);
            }
            gpu::write_buffer_bytes(
                &self.scratch.routing_w,
                0,
                &f16_slice_to_le_bytes(&routing16),
            );

            let layer_slots = &self.slot_buffers[layer];
            let blob_refs: Vec<(&gpu::MetalBuffer, u64)> = ordered
                .iter()
                .map(|&(slot, _)| (&layer_slots[slot], 0u64))
                .collect();
            let routed = self.routed_blobs.as_ref().ok_or_else(|| {
                RealForwardError::Unsupported("install has no routed-blob buffer".to_string())
            })?;
            let offsets = self.moe_offsets.as_ref().expect("layout implies offsets");
            routed
                .bind(&mut self.context, use_silu, &blob_refs)
                .map_err(gpu_err)?;
            self.phases.bind_nanos += t_bind.elapsed().as_nanos() as u64;

            pass = self.context.begin_pass_labeled("routed cb");
            for &(buffer, _) in &blob_refs {
                pass.use_read_buffer(buffer);
            }

            // Routed branch on routed_x -> h2 (zero residual: the sandwich
            // combine needs the raw routed output), then post_ffn_2. Phase
            // 1 here covers only the slots the hit command buffer did not
            // already take, which are the low ones by construction.
            let main_phase1_k = if dispatched_hits {
                miss_count
            } else {
                order.len()
            };
            if main_phase1_k > 0 {
                gpu::encode_moe_phase1(
                    &mut self.context,
                    &pass,
                    routed,
                    offsets,
                    (&real.routed_x, 0),
                    (&self.scratch.moe_acts, 0),
                    hidden as u32,
                    moe_inter,
                    main_phase1_k as u32,
                    use_silu,
                )
                .map_err(gpu_err)?;
            }
            gpu::encode_moe_phase2(
                &mut self.context,
                &pass,
                routed,
                offsets,
                (&self.scratch.moe_acts, 0),
                (&self.scratch.routing_w, 0),
                (&self.scratch.zero_hidden, 0),
                (&real.h2, 0),
                hidden as u32,
                moe_inter,
                use_silu,
            )
            .map_err(gpu_err)?;
            let post_ffn2 = norm_view(
                &self.weights,
                &self.index,
                &layer_tensor(layer, "post_feedforward_layernorm_2.weight"),
                hidden,
            )?;
            gpu::encode_rms_norm_bf16w(
                &mut self.context,
                &pass,
                (&real.h2, 0),
                post_ffn2,
                (&real.h2, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;

            // Tail: h += rmsnorm_bf16w(h1 + h2, post_ffn); h *= layer_scalar.
            gpu::encode_residual_add(
                &mut self.context,
                &pass,
                (&real.h1, 0),
                (&real.h2, 0),
                hidden as u32,
            )
            .map_err(gpu_err)?;
            let post_ffn = norm_view(
                &self.weights,
                &self.index,
                &layer_tensor(layer, "post_feedforward_layernorm.weight"),
                hidden,
            )?;
            gpu::encode_rms_norm_bf16w(
                &mut self.context,
                &pass,
                (&real.h1, 0),
                post_ffn,
                (&self.scratch.ffn_normed, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
            gpu::encode_residual_add(
                &mut self.context,
                &pass,
                (&self.scratch.x, 0),
                (&self.scratch.ffn_normed, 0),
                hidden as u32,
            )
            .map_err(gpu_err)?;
            gpu::encode_scalar_mul(
                &mut self.context,
                &pass,
                (&self.scratch.x, 0),
                layer_scalar,
                hidden as u32,
            )
            .map_err(gpu_err)?;

            // The one seam branch: commit this layer's routed work as its
            // own command buffer now (pipelined arm) instead of letting it
            // roll uncommitted into the next layer's first buffer.
            if self.routed_pipeline {
                debug_assert!(pending_routed.is_none(), "routed pipeline depth is 1");
                pending_routed = Some(pass.commit());
                pass = self.context.begin_pass_labeled("cb1 (attn+router)");
            }
        }

        // Drain the last layer's routed buffer; nothing is left to hide it
        // behind, so this is the one retire that costs real wait time.
        if let Some(pending) = pending_routed.take() {
            let t_retire = Instant::now();
            self.phases.routed_cb_gpu_nanos += (pending.wait_with_gpu_time() * 1e9) as u64;
            self.phases.pipeline_wait_nanos += t_retire.elapsed().as_nanos() as u64;
        }

        // Final norm + tied LM head + softcap. This rides whatever buffer the
        // last layer left open, which the phase counters bill as the final
        // CB; say so for the dispatch profile too. On a prefill token whose
        // logits the caller discards the whole head is skipped, but the open
        // buffer still has to be committed and waited on (the next token
        // overwrites this one's scratch) and the KV still has to advance.
        if !self.skip_head {
            pass.relabel("final cb (head)");
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
                RMS_EPS,
            )
            .map_err(gpu_err)?;
            let head_name = if arch.tie_word_embeddings {
                embed_name.to_string()
            } else {
                "language_model.lm_head.weight".to_string()
            };
            encode_gemv_any(
                &mut self.context,
                &pass,
                &self.weights,
                &self.index,
                &head_name,
                vocab,
                hidden,
                (&self.scratch.normed, 0),
                (&self.scratch.logits, 0),
            )?;
            if arch.final_logit_softcap > 0.0 {
                gpu::encode_logit_softcap(
                    &mut self.context,
                    &pass,
                    (&self.scratch.logits, 0),
                    arch.final_logit_softcap as f32,
                    vocab as u32,
                )
                .map_err(gpu_err)?;
            }
        } else {
            pass.relabel("final cb (prefill, no head)");
        }
        let t_wait = Instant::now();
        self.phases.final_cb_gpu_nanos += (pass.commit_and_wait_with_gpu_time() * 1e9) as u64;
        self.phases.final_wait_nanos += t_wait.elapsed().as_nanos() as u64;
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
