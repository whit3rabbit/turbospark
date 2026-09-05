//! `RealQwen4State`: everything [`crate::real_forward::RealForwardRunner`]
//! allocates once when it opens a `qwen4_exp` install -- the wide (10240)
//! residual buffer and its hyper-connection scratch, the GDN recurrent
//! state, the PLE layer's own buffers and its mmap'd n-gram table, and
//! every MoE scratch buffer `families/qwen/state.rs` already established
//! the shape of.
//!
//! **THE WIDE RESIDUAL IS FAMILY-LOCAL, NOT `DecodeScratch::x`.** That
//! shared field is sized `hidden_size * MAX_PREFILL_BATCH` for every OTHER
//! family; widening it to `hidden_size * hc_count` here would move every
//! other family's footprint and disturb their frozen memory-oracle rows
//! for a buffer they never touch (`docs/QWEN4_PHASE0.md`'s Phase 3 plan,
//! `BatchedScratch`'s own precedent one struct over). `wide_x` is this
//! struct's own field instead; `DecodeScratch`'s hidden-width buffers
//! (`normed`, `q`, `attn_out`, `o`, the FFN quartet, `logits`) ARE reused,
//! because every one of them already operates at `H = 2560` -- the width
//! of a hyper-connection's `mixed` output, never the wide stream itself.

use std::path::Path;

use model_io::{ArchConfig, NgramContext, NgramTableLayout, ResidentBuffer, ResidentIndex};

use crate::families::qwen4::layer_tensor;
use crate::real_forward::RealForwardError;
use crate::real_forward_utils::entry;

/// `(taps - 1) * dilation` for PLE's depthwise conv: `kernel_size=4`,
/// `dilation=ngram_size=3` (`docs/QWEN4_PHASE0.md` item 4).
pub(crate) const PLE_CONV_HISTORY: usize = 9;

/// Per-open `qwen4_exp` decode state.
pub(crate) struct RealQwen4State {
    pub(crate) shape: gpu::GdnShape,
    pub(crate) gdn: gpu::GdnStateManager,
    pub(crate) rotary_dim: u32,
    pub(crate) hc_count: usize,
    pub(crate) hc_lowrank: usize,
    /// Zero-based index of the one PLE layer (`arch.ple.layer_indices()`'s
    /// single entry, converted from the checkpoint's one-based spelling).
    pub(crate) ple_layer: usize,

    /// The residual stream, `hidden_size * hc_count` wide. See the module
    /// doc for why this is not `DecodeScratch::x`.
    pub(crate) wide_x: gpu::MetalBuffer,
    /// `hc_norm`'s output, wide -- shared by both per-layer hyper-connection
    /// calls and the final mixer, which never overlap within one dispatch
    /// sequence (commit-order execution, `crates/gpu` Gotcha 8).
    pub(crate) hc_normed: gpu::MetalBuffer,
    /// `input_mix_weight_down`'s output, `hc_lowrank` wide.
    pub(crate) hc_low: gpu::MetalBuffer,
    /// `input_mix_weight_up`'s output, wide -- this is `w` in `hc_mix`'s
    /// contract.
    pub(crate) hc_up: gpu::MetalBuffer,
    /// `block_inject_weight`'s output, `hc_count` wide.
    pub(crate) hc_inject: gpu::MetalBuffer,

    /// `[2 * num_heads * full_head_dim]`: QSA's packed query/gate rows.
    pub(crate) q_packed: gpu::MetalBuffer,
    pub(crate) attn_gate: gpu::MetalBuffer,
    pub(crate) gdn_qkv_raw: gpu::MetalBuffer,
    pub(crate) gdn_conv_out: gpu::MetalBuffer,
    pub(crate) gdn_z: gpu::MetalBuffer,
    pub(crate) gdn_a: gpu::MetalBuffer,
    pub(crate) gdn_b: gpu::MetalBuffer,
    pub(crate) gdn_y: gpu::MetalBuffer,
    pub(crate) gdn_out: gpu::MetalBuffer,

    /// BF16 `[hidden]` of ones -- the router kernel's unused per-element
    /// scale, matching `families/qwen/state.rs`'s `router_ones` exactly
    /// (this checkpoint has no `router.scale` either).
    pub(crate) router_ones: gpu::MetalBuffer,
    pub(crate) per_expert_ones: Vec<f32>,
    pub(crate) router_logits_f32: gpu::MetalBuffer,
    /// The gated shared-expert output, which SEEDS phase 2's accumulator
    /// (`docs/QWEN4_PHASE0.md` item 7, matching `families/qwen/moe.rs`'s
    /// existing choice: FP addition is not associative, so seeding differs
    /// from appending to a finished routed sum).
    pub(crate) h1: gpu::MetalBuffer,
    /// Shared + routed, `moe(mixed)`'s final output -- the `out` the layer
    /// pseudocode injects back into the wide stream.
    pub(crate) h2: gpu::MetalBuffer,
    pub(crate) shared_gate_logit: gpu::MetalBuffer,

    /// PLE's concatenated n-gram lookup, `[hidden]` -- host dequant, GPU
    /// upload.
    pub(crate) ngram_emb: gpu::MetalBuffer,
    /// `norm_key(key_proj(emb))`, wide.
    pub(crate) ple_key: gpu::MetalBuffer,
    /// `value_proj(emb)`, `[hidden]` -- shared across all `hc_count`
    /// streams by `ple_gate`'s own broadcast.
    pub(crate) ple_value: gpu::MetalBuffer,
    /// `norm_query(hidden)`, wide -- reads the WIDE residual directly,
    /// never `mixed`.
    pub(crate) ple_query: gpu::MetalBuffer,
    /// `ple_gate`'s output (`gv.flatten()`), wide.
    pub(crate) ple_gv: gpu::MetalBuffer,
    /// `norm_conv(gv.flatten())`, wide -- the dilated conv's input.
    pub(crate) ple_conv_normed: gpu::MetalBuffer,
    /// `silu(dilated_conv(...))`, wide -- added to `ple_gv` (NOT to
    /// `ple_conv_normed`) to produce PLE's final output, per
    /// `docs/QWEN4_PHASE0.md` item 4: the conv reads the NORMED gated
    /// value and its output joins the UN-normed one.
    pub(crate) ple_conv_out: gpu::MetalBuffer,
    /// The dilated conv's recurrent state: [`PLE_CONV_HISTORY`] rows of
    /// `hidden * hc_count` RAW (pre-conv) values, FP16. Standalone rather
    /// than part of [`RealQwen4State::gdn`]'s conv tails: `GdnStateManager`
    /// sizes its tails from `LinearAttentionConfig`, at the GDN chain's
    /// `qkv_dim` width, and this is a differently-shaped buffer belonging
    /// to exactly one layer.
    pub(crate) ple_conv_tail: gpu::MetalBuffer,
    /// The n-gram table's own addressing (multipliers, per-head vocab
    /// sizes and offsets), read from `ngram_table/header.json`.
    pub(crate) ngram_layout: NgramTableLayout,
    /// The WHOLE table, mmap'd once at open. Demand-paged, so mapping the
    /// full ~32 GB costs no resident memory until a row is actually
    /// touched -- `model_io::ngram_table`'s module doc has the measured
    /// argument for `mmap` over the `pread` streamer at this record size.
    pub(crate) ngram_table: ResidentBuffer,
    /// The decode-time EOS-boundary-aware n-gram context (recurrent state
    /// slot 2 in `docs/QWEN4_PHASE0.md` item 4's table). Reset alongside
    /// the GDN state.
    pub(crate) ngram_context: NgramContext,
    /// `ngram_size - 1`: `NgramContext` does not expose its own length, and
    /// re-deriving it from another stored constant (`PLE_CONV_HISTORY` is
    /// `(taps - 1) * dilation`, an unrelated product that only coincides in
    /// one factor) is exactly the kind of clever-looking arithmetic that
    /// reads correct and is not; a plain stored field has no such trap.
    pub(crate) ngram_context_len: usize,
    /// `arch.ple.eos_token_id`, kept for [`RealQwen4State::reset`] and for
    /// feeding [`model_io::NgramContext::step`] each decode step.
    pub(crate) eos_token_id: i64,
}

impl RealQwen4State {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build(
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &ResidentIndex,
        arch: &ArchConfig,
        install_dir: &Path,
        max_context: usize,
        expert_cache_slots: usize,
    ) -> Result<Self, RealForwardError> {
        let unsupported = |detail: String| Err(RealForwardError::Unsupported(detail));

        // **NO INDEXER CODE EXISTS IN THIS PORT** (`mod.rs`'s "##
        // QSA-as-dense-attention"): below `indexer_budget`,
        // `docs/QWEN4_PHASE0.md` section 5 proves from source that QSA's
        // block selection is a no-op and every full-attention layer is
        // exactly dense causal attention. Refusing above that budget is
        // what licenses reading no `self_attn.indexer.*` tensor anywhere
        // in this flow -- not an approximation of QSA, the exact function
        // it computes under the budget. Three tokens of margin exist in
        // the reference's own proof (`visible <= 2051`); this refuses at
        // the round number the checkpoint itself declares instead of
        // trying to claim those three.
        let budget = arch.compressed_attention.index_budget;
        if budget <= 0 {
            return unsupported(
                "qwen4_exp declares no positive indexer_budget; this flow needs one to refuse \
                 context above it"
                    .to_string(),
            );
        }
        if max_context > budget as usize {
            return unsupported(format!(
                "qwen4_exp needs the QSA indexer above {budget} tokens of context, which this \
                 port does not implement; --max-context must not exceed {budget} \
                 (docs/QWEN4_PHASE0.md section 5)"
            ));
        }

        if arch.num_experts <= 0 || arch.top_k_experts <= 0 {
            return unsupported(format!(
                "qwen4_exp is MoE-only in this port: num_experts {}, top_k_experts {} must both \
                 be positive",
                arch.num_experts, arch.top_k_experts
            ));
        }
        if arch.top_k_experts as usize > gpu::MAX_STREAMED_EXPERTS {
            return unsupported(format!(
                "top_k {} exceeds the {}-slot MoE kernels",
                arch.top_k_experts,
                gpu::MAX_STREAMED_EXPERTS
            ));
        }
        // ONE TOKEN'S TOP-K MUST FIT THE CACHE OUTRIGHT, independent of
        // AGENTS.md Gotcha 64's `2 * top_k` pipelining margin: that margin
        // protects a PREVIOUS token's still-in-flight slots from an
        // overlapping plan, which only exists on a CHUNKED prefill driver.
        // This flow has none (`mod.rs`'s own doc: chunked prefill is
        // refused by name), so `moe::encode_moe_layer` always plans with an
        // empty `protect` set and the only hard requirement is that
        // `top_k_experts` distinct experts fit `expert_cache_slots` slots at
        // all -- below that, `ExpertCache::plan` cannot select this layer's
        // routing regardless of pipelining and aborts the process rather
        // than degrading. A future chunked-prefill driver for this family
        // would need to raise this to `2 * top_k`, matching every other
        // family's driver.
        if expert_cache_slots < arch.top_k_experts as usize {
            return unsupported(format!(
                "qwen4_exp routes {} experts per token, which does not fit a \
                 {expert_cache_slots}-slot cache; raise --expert-cache-slots to at least {} \
                 (AGENTS.md Gotcha 64)",
                arch.top_k_experts, arch.top_k_experts
            ));
        }
        if !arch.shared_expert_gated || !arch.attn_output_gate || !arch.rope_neox_subdim {
            return unsupported(
                "the qwen4_exp flow needs sharedExpertGated + attnOutputGate + ropeNeoxSubdim"
                    .to_string(),
            );
        }
        if arch.ffn_sandwich_norms || arch.router_scaled || arch.embedding_scaled_by_sqrt_hidden {
            return unsupported(
                "the qwen4_exp flow has no sandwich norms, no router scale, and no \
                 sqrt(hidden) embedding scale"
                    .to_string(),
            );
        }
        if arch.final_logit_softcap != 0.0 {
            return unsupported("qwen4_exp has no final logit softcap".to_string());
        }
        if arch.tie_word_embeddings {
            return unsupported("qwen4_exp's lm_head is its own tensor, never tied".to_string());
        }
        if arch
            .full_attention_layer_mask
            .iter()
            .any(|&m| m != 1 && m != 2)
        {
            return unsupported(
                "this flow's layers are full attention (1, QSA-below-budget) or gated \
                 DeltaNet (2) only"
                    .to_string(),
            );
        }
        if !arch.has_linear_attention_layers() {
            return unsupported(
                "qwen4_exp with no linear layers is not one this flow can run".to_string(),
            );
        }
        if arch.hyper_connections.mult <= 1 {
            return unsupported(format!(
                "hyper_connections.mult {} must be at least 2; this flow's whole point is \
                 the wide multi-stream residual",
                arch.hyper_connections.mult
            ));
        }
        if arch.hyper_connections.lowrank <= 0 {
            return unsupported(format!(
                "hyper_connections.lowrank {} must be positive",
                arch.hyper_connections.lowrank
            ));
        }
        if !arch.ple.is_active() {
            return unsupported(
                "qwen4_exp with no active PLE table is not one this flow can run".to_string(),
            );
        }
        let ple_layers = arch.ple.layer_indices();
        let [ple_layer_signed] = ple_layers[..] else {
            return unsupported(format!(
                "arch.ple declares {} PLE layers; this flow places exactly one",
                ple_layers.len()
            ));
        };
        if ple_layer_signed < 0 || ple_layer_signed >= arch.num_layers {
            return unsupported(format!(
                "PLE layer index {ple_layer_signed} is outside 0..{}",
                arch.num_layers
            ));
        }
        let ple_layer = ple_layer_signed as usize;
        if arch.layer_is_linear(ple_layer) {
            // Confirmed by the checkpoint (`docs/QWEN4_PHASE0.md` section
            // 0 finding 1: `ple_layer_ids: [2]` resolves to layer index 1,
            // which this baseline's `full_attention_layer_mask` marks
            // linear/GDN). Stated as an invariant the flow relies on
            // rather than assumed silently: PLE's own dataflow runs
            // BEFORE the layer's GDN-or-attention branch either way, so a
            // PLE layer that happened to be a QSA one would still be
            // structurally fine -- this check exists so a future config
            // that moves PLE onto a QSA layer is a loud refusal instead of
            // an unexercised code path.
        }
        if arch.ple.eos_token_id == 0 {
            return unsupported(
                "arch.ple.eos_token_id is unset; PLE's n-gram context resets at EOS \
                 boundaries and needs one (docs/QWEN4_PHASE0.md item 4)"
                    .to_string(),
            );
        }
        if !arch.linear_attention.output_gate_sigmoid {
            return unsupported(
                "qwen4_exp's GDN gated norm is the sigmoid variant \
                 (docs/QWEN4_PHASE0.md item 0 finding 4); outputGateSigmoid must be true"
                    .to_string(),
            );
        }

        let shape = gpu::GdnShape {
            num_k_heads: arch.linear_attention.num_k_heads as u32,
            num_v_heads: arch.linear_attention.num_v_heads as u32,
            key_head_dim: arch.linear_attention.key_head_dim as u32,
            value_head_dim: arch.linear_attention.value_head_dim as u32,
            conv_kernel_size: arch.linear_attention.conv_kernel_size as u32,
        };
        shape.validate().map_err(RealForwardError::Gpu)?;

        let head_dim = arch.full_head_dim;
        let rotary_dim = (head_dim as f64 * arch.partial_rotary_factor).round() as i64;
        if rotary_dim <= 0 || rotary_dim % 2 != 0 || rotary_dim > head_dim {
            return unsupported(format!(
                "rotary_dim {rotary_dim} (full_head_dim {head_dim} x partial_rotary_factor \
                 {}) must be positive, even, and at most full_head_dim",
                arch.partial_rotary_factor
            ));
        }

        // Fail at open, not at token 1: the top-level tensors, plus one
        // representative layer of each kind (attn_hyper_connection /
        // mlp_hyper_connection are on EVERY layer, so those are probed for
        // all of them).
        let hidden = arch.hidden_size as usize;
        for name in [
            "language_model.model.embed_tokens.weight".to_string(),
            "language_model.lm_head.weight".to_string(),
            "language_model.model.hyper_connection_mixer.hc_norm.weight".to_string(),
            "language_model.model.hyper_connection_mixer.input_mix_weight_down.weight".to_string(),
            "language_model.model.hyper_connection_mixer.input_mix_weight_up.weight".to_string(),
        ] {
            entry(index, &name)?;
        }
        for layer in 0..arch.num_layers as usize {
            for hc in ["attn_hyper_connection", "mlp_hyper_connection"] {
                for suffix in [
                    "hc_norm.weight",
                    "input_mix_weight_down.weight",
                    "input_mix_weight_up.weight",
                    "block_inject_weight.weight",
                ] {
                    entry(index, &layer_tensor(layer, &format!("{hc}.{suffix}")))?;
                }
            }
            entry(index, &layer_tensor(layer, "mlp.gate.weight"))?;
            entry(index, &layer_tensor(layer, "mlp.shared_expert_gate.weight"))?;
            entry(
                index,
                &layer_tensor(layer, "mlp.shared_expert.gate_proj.weight"),
            )?;
            if arch.layer_is_linear(layer) {
                for suffix in [
                    "linear_attn.in_proj_qkv.weight",
                    "linear_attn.conv1d.weight",
                    "linear_attn.A_log",
                    "linear_attn.dt_bias",
                ] {
                    entry(index, &layer_tensor(layer, suffix))?;
                }
            } else {
                for suffix in ["self_attn.q_proj.weight", "self_attn.q_norm.weight"] {
                    entry(index, &layer_tensor(layer, suffix))?;
                }
            }
            if layer == ple_layer {
                for suffix in [
                    "ple.key_proj.weight",
                    "ple.value_proj.weight",
                    "ple.conv1d.weight",
                    "ple.norm_key.weight",
                    "ple.norm_query.weight",
                    "ple.norm_conv.weight",
                ] {
                    entry(index, &layer_tensor(layer, suffix))?;
                }
            }
        }
        let _ = weights;

        let ngram_layout = model_io::load_ngram_table_layout(install_dir)
            .map_err(RealForwardError::Model)?
            .ok_or_else(|| {
                RealForwardError::Unsupported(
                    "qwen4_exp needs an ngram_table/ store; this install has none".to_string(),
                )
            })?;
        let expected_head_dim = arch.ple.head_dim() as u64;
        if ngram_layout.head_dim != expected_head_dim {
            return unsupported(format!(
                "ngram_table/header.json declares head_dim {}, but arch.ple resolves \
                 {expected_head_dim} ({}head/{}heads)",
                ngram_layout.head_dim,
                arch.ple.ple_embed_dim,
                arch.ple.ngram_heads()
            ));
        }
        let ngram_table = ResidentBuffer::map(
            &install_dir
                .join(model_io::NGRAM_TABLE_DIR)
                .join(model_io::NGRAM_TABLE_BLOB),
            0,
            ngram_layout.blob_bytes().ok_or_else(|| {
                RealForwardError::Unsupported("ngram_table row/record count overflows".to_string())
            })?,
        )
        .map_err(RealForwardError::Model)?;

        let ones: Vec<u8> = (0..hidden)
            .flat_map(|_| 0x3F80u16.to_le_bytes()) // BF16 1.0
            .collect();
        let router_ones = context.new_output_buffer(ones.len() as u64);
        gpu::write_buffer_bytes(&router_ones, 0, &ones);

        let halfs = |n: usize| context.new_output_buffer((n.max(1) * 2) as u64);
        let hc_count = arch.hyper_connections.mult as usize;
        let hc_lowrank = arch.hyper_connections.lowrank as usize;
        let wide_dim = hidden * hc_count;
        let num_experts = arch.num_experts as usize;
        let q_dim = (arch.num_heads * head_dim) as usize;
        let qkv_dim = shape.qkv_dim() as usize;
        let value_dim = shape.value_dim() as usize;
        let v_heads = shape.num_v_heads as usize;

        let ngram_context_len = (arch.ple.ngram_size - 1).max(0) as usize;
        let ple_conv_tail = halfs(PLE_CONV_HISTORY * wide_dim);
        gpu::write_buffer_bytes(
            &ple_conv_tail,
            0,
            &vec![0u8; PLE_CONV_HISTORY * wide_dim * 2],
        );

        Ok(Self {
            gdn: gpu::GdnStateManager::new(context.device(), arch),
            shape,
            rotary_dim: rotary_dim as u32,
            hc_count,
            hc_lowrank,
            ple_layer,

            wide_x: halfs(wide_dim),
            hc_normed: halfs(wide_dim),
            hc_low: halfs(hc_lowrank),
            hc_up: halfs(wide_dim),
            hc_inject: halfs(hc_count),

            q_packed: halfs(2 * q_dim),
            attn_gate: halfs(q_dim),
            gdn_qkv_raw: halfs(qkv_dim),
            gdn_conv_out: halfs(qkv_dim),
            gdn_z: halfs(value_dim),
            gdn_a: halfs(v_heads),
            gdn_b: halfs(v_heads),
            gdn_y: halfs(value_dim),
            gdn_out: halfs(value_dim),

            router_ones,
            per_expert_ones: vec![1.0; num_experts],
            router_logits_f32: context.new_output_buffer((num_experts.max(1) * 4) as u64),
            h1: halfs(hidden),
            h2: halfs(hidden),
            shared_gate_logit: halfs(1),

            ngram_emb: halfs(hidden),
            ple_key: halfs(wide_dim),
            ple_value: halfs(hidden),
            ple_query: halfs(wide_dim),
            ple_gv: halfs(wide_dim),
            ple_conv_normed: halfs(wide_dim),
            ple_conv_out: halfs(wide_dim),
            ple_conv_tail,
            ngram_context: NgramContext::new(ngram_context_len, arch.ple.eos_token_id),
            ngram_context_len,
            eos_token_id: arch.ple.eos_token_id,
            ngram_layout,
            ngram_table,
        })
    }

    /// Rewinds the recurrent state to empty context: the GDN chain's delta
    /// rule and conv tail, the PLE n-gram context, AND the PLE dilated
    /// conv's own tail.
    ///
    /// **THIS WAS THE qwen4_exp QUALITY GATE'S DETERMINISM BUG
    /// (`docs/QWEN4_EXP.md`'s "The quality gate is BLOCKED" section).** The
    /// module doc used to claim the conv tail's staleness across `reset()`
    /// was inert because "today nothing [resets mid-process]" -- that claim
    /// was false the moment `crates/bench`'s quality gate called `reset()`
    /// between two back-to-back warm generations on one open runner, which
    /// is exactly the mid-process reset the old comment said did not exist
    /// yet. Leaving `ple_conv_tail` stale meant generation 2 started PLE's
    /// dilated conv from generation 1's leftover history instead of from
    /// zero, diverging the wide residual from the very first PLE-layer
    /// token onward and cascading into a completely different greedy
    /// digest -- while a fresh process always starts from the zeros
    /// `RealQwen4State::build` writes, which is why cross-process
    /// generation reproduced exactly throughout that investigation.
    pub(crate) fn reset(&mut self) {
        self.gdn.reset();
        self.ngram_context = NgramContext::new(self.ngram_context_len, self.eos_token_id);
        let tail_len = self.ple_conv_tail.length() as usize;
        gpu::write_buffer_bytes(&self.ple_conv_tail, 0, &vec![0u8; tail_len]);
    }
}
