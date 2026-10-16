//! `RealQwenState`: everything [`crate::real_forward::RealForwardRunner`]
//! allocates once when it opens a Qwen 3.6 or `qwen3_5` install -- the GDN
//! recurrent buffers, the unit router scales that stand in for Qwen's absent
//! `router.scale`/`per_expert_scale`, and every piece of per-token
//! scratch the flow in `mod.rs` writes.
//!
//! **TWO FAMILIES, one state, and they differ in the FFN alone** (ROADMAP's
//! 1-bit entry). `qwen_gdn_dense_27b()`'s every BEHAVIOURAL field equals
//! `qwen_gdn_moe_35b_a3b()`'s and every SHAPE field differs, which is what licenses
//! sharing the flow rather than forking it; `shared_expert_gated` is the one
//! that legitimately parts company, and it does so BECAUSE the model is
//! dense. The split is read off `num_experts`, never off tensor naming, for
//! the reason `families/llama/state.rs` reads its own off the same field.

use model_io::{ArchConfig, ResidentIndex};

use crate::families::qwen::{layer_tensor, BatchedScratch, HadamardPlan};
use crate::real_forward::RealForwardError;
use crate::real_forward_types::MAX_PREFILL_BATCH;
use crate::real_forward_utils::entry;

/// BF16 bit pattern for 1.0.
const BF16_ONE: u16 = 0x3F80;

/// What the pilot probe writes: the next layer's post-attention norm of THIS
/// layer's residual, and the router logits it produces.
pub(crate) struct PilotBuffers {
    /// BF16 `[hidden]`.
    pub(crate) x: gpu::MetalBuffer,
    /// F32 `[num_experts]`.
    pub(crate) logits: gpu::MetalBuffer,
}

/// Per-open Qwen decode state: the recurrent GDN buffers plus every piece
/// of scratch the per-token flow writes.
pub(crate) struct RealQwenState {
    pub(crate) shape: gpu::GdnShape,
    pub(crate) gdn: gpu::GdnStateManager,
    /// `rotary_dim`: `full_head_dim * partial_rotary_factor`, the width of
    /// the rotated prefix of each head.
    pub(crate) rotary_dim: u32,
    /// DENSE, i.e. `num_experts == 0`: the `qwen3_5` half of this flow, whose
    /// layers carry one `mlp.{gate,up,down}_proj` where Qwen 3.6 carries a
    /// router, a gated shared expert and a routed table.
    ///
    /// It changes the FFN half of a layer and nothing above it: the
    /// embedding, both norms, both attention blocks, the raw residual and the
    /// head are the same code either way.
    pub(crate) dense: bool,
    /// Experimental scheduling is captured once per open, keeping env
    /// lookups out of the per-layer hot path and the shipping path unchanged.
    pub(crate) shared_read_overlap: bool,
    /// `false` when `TURBOSPARK_QWEN_MOE_CHUNKED=0` was set at open: the
    /// kill switch for the MoE chunked prefill driver
    /// (`moe_prefill.rs`), which keeps the sequential path one env var away
    /// (`docs/SPEEDUP_PLAYBOOK.md`). Read once here, never per token.
    pub(crate) moe_chunked_prefill: bool,
    pub(crate) mapped_demand_preparation: Option<streaming::MappedPagePreparationMode>,
    /// BF16 `[hidden]` of ones. The INT8 router kernel multiplies `x[n]`
    /// by an effective scale per element; Qwen has no `router.scale` and
    /// `router_scaled` is false, so the scale is identically 1.
    pub(crate) router_ones: gpu::MetalBuffer,
    /// `[num_experts]` of ones, for the same reason: Qwen has no
    /// `per_expert_scale`, so `router_topk_gemma4`'s weighting is a no-op
    /// and the selection reduces to softmax-over-the-selected.
    pub(crate) per_expert_ones: Vec<f32>,
    pub(crate) router_logits_f32: gpu::MetalBuffer,
    /// Scratch for the one-layer-ahead router probe (`TURBOSPARK_PILOT_PROBE`).
    /// `None` unless the env var was set at open, so a normal run allocates
    /// and dispatches nothing for it.
    pub(crate) pilot: Option<PilotBuffers>,
    /// `TURBOSPARK_QWEN_PREFETCH=<k>`: read the next layer's guessed experts
    /// early on a worker. An experiment (see `prefetch.rs`).
    pub(crate) prefetch: Option<super::prefetch::Prefetcher>,

    /// `[2 * num_heads * full_head_dim]`: q_proj's packed query/gate rows.
    pub(crate) q_packed: gpu::MetalBuffer,
    /// `[num_heads * full_head_dim]`: the gate half, after the split.
    pub(crate) attn_gate: gpu::MetalBuffer,
    pub(crate) gdn_qkv_raw: gpu::MetalBuffer,
    pub(crate) gdn_conv_out: gpu::MetalBuffer,
    pub(crate) gdn_z: gpu::MetalBuffer,
    pub(crate) gdn_a: gpu::MetalBuffer,
    pub(crate) gdn_b: gpu::MetalBuffer,
    pub(crate) gdn_y: gpu::MetalBuffer,
    pub(crate) gdn_out: gpu::MetalBuffer,
    /// `[hidden]`: the one post-attention norm that feeds the router, the
    /// shared expert, and the routed experts alike.
    pub(crate) moe_x: gpu::MetalBuffer,
    /// `[hidden]`: the gated shared-expert output, which doubles as the
    /// residual phase 2 adds its routed sum onto.
    pub(crate) h1: gpu::MetalBuffer,
    /// `[hidden]`: shared + routed, added back to the stream.
    pub(crate) h2: gpu::MetalBuffer,
    /// One half: the shared expert's scalar gate logit.
    pub(crate) shared_gate_logit: gpu::MetalBuffer,
    /// The M-row scratch the chunked-prefill driver's batched-GEMV arm
    /// writes (`TURBOSPARK_BATCHED_GEMV`, `docs/BATCHED_PREFILL.md` step 6).
    ///
    /// **Allocated on FIRST USE, and only when the seam is ON**, which is a
    /// stronger condition than the Gemma 4 sibling this otherwise copies:
    /// `RealGemmaState::batched` serves `TURBOSPARK_ROUTED_BATCH` as well, so
    /// its driver allocates at the entry point unconditionally. Here the one
    /// consumer is the seam, and `qwen38_memory_oracle`'s frozen row has to
    /// keep describing the engine that shipped before this arm existed --
    /// this struct is the same [`BatchedScratch`] the drafters allocate, so
    /// it is ~10 MiB on the real install rather than a rounding error.
    ///
    /// It is never allocated BESIDE a drafter's copy: the chunked driver
    /// refuses an open drafter by name, so at most one of the two exists.
    pub(crate) batched_prefill: Option<BatchedScratch>,
    /// The Hadamard-folded weight contract's plan (the Bonsai-2 line), built
    /// only when the manifest carries a `hadamard` section. `None` on every
    /// install written before the contract existed, which makes every
    /// consumer site byte-identical to its pre-contract code.
    pub(crate) hadamard: Option<HadamardPlan>,
}

impl RealQwenState {
    pub(crate) fn build(
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &ResidentIndex,
        arch: &ArchConfig,
        hadamard: Option<(&model_io::ManifestHadamard, &[u8])>,
        headless: bool,
    ) -> Result<Self, RealForwardError> {
        let unsupported = |detail: String| Err(RealForwardError::Unsupported(detail));
        // DENSE AND MoE ARE BOTH THIS FLOW. The two are mutually exclusive
        // rather than a spectrum, so a file claiming experts with no `top_k`
        // (or the reverse) is malformed and is refused rather than guessed at.
        if arch.num_experts < 0 || arch.top_k_experts < 0 {
            return unsupported(format!(
                "negative expert counts: num_experts {}, top_k_experts {}",
                arch.num_experts, arch.top_k_experts
            ));
        }
        let dense = arch.num_experts == 0;
        let shared_read_overlap =
            !dense && std::env::var("TURBOSPARK_QWEN_SHARED_READ_OVERLAP").as_deref() == Ok("1");
        let moe_chunked_prefill =
            !dense && std::env::var("TURBOSPARK_QWEN_MOE_CHUNKED").as_deref() != Ok("0");
        let mapped_demand_preparation =
            match std::env::var("TURBOSPARK_MAPPED_DEMAND_PREP").as_deref() {
                Err(_) | Ok("off") => None,
                Ok("advice") => Some(streaming::MappedPagePreparationMode::Advice),
                Ok("touch") => Some(streaming::MappedPagePreparationMode::Touch),
                Ok(value) => {
                    return unsupported(format!(
                "invalid TURBOSPARK_MAPPED_DEMAND_PREP {value:?}; expected off, advice, or touch"
            ))
                }
            };
        if dense != (arch.top_k_experts == 0) {
            return unsupported(format!(
                "num_experts {} and top_k_experts {} disagree about whether this install is \
                 dense; both must be zero or both positive",
                arch.num_experts, arch.top_k_experts
            ));
        }
        // The deepstack injection lives in the LLAMA flow (the qwen3_vl
        // trunk). A qwen3_5-family tower declaring deepstack mergers would
        // run here and silently skip every one of them -- competent output
        // from a trunk that never saw the picture's deep features -- so an
        // arch this flow cannot inject is refused at open, the same
        // shape the qwen35 chunked driver's batched-GEMV refusal takes.
        if !arch.vision.deepstack_visual_indexes.is_empty() {
            return unsupported(format!(
                "this install's vision tower declares {} deepstack merger(s), which no qwen \
                 flow injects; deepstack belongs to the qwen3_vl trunk (the shared llama flow)",
                arch.vision.deepstack_visual_indexes.len()
            ));
        }
        if arch.ffn_sandwich_norms
            || arch.router_scaled
            || arch.embedding_scaled_by_sqrt_hidden
            || !arch.attn_output_gate
            || !arch.rope_neox_subdim
        {
            return unsupported(
                "the Qwen flow needs attnOutputGate + ropeNeoxSubdim and none of \
                 ffnSandwichNorms / routerScaled / embeddingScaledBySqrtHidden"
                    .to_string(),
            );
        }
        // `sharedExpertGated` is the ONE behavioural field the two halves
        // disagree on, and the disagreement follows from the FFN shape rather
        // than being an independent axis: there is no shared expert to gate on
        // a dense model. Checked in both directions, because a MoE install
        // that lost the flag would silently skip the sigmoid gate and a dense
        // one that claimed it would be describing a tensor it does not have.
        if arch.shared_expert_gated == dense {
            return unsupported(format!(
                "sharedExpertGated is {} on a {} install; a MoE Qwen gates its shared expert and \
                 a dense one has none",
                arch.shared_expert_gated,
                if dense { "dense" } else { "MoE" }
            ));
        }
        if arch.final_logit_softcap != 0.0 {
            return unsupported("no Qwen family here has a final logit softcap".to_string());
        }
        if dense && arch.intermediate_size <= 0 {
            return unsupported(format!(
                "a dense Qwen install needs a positive ffnIntermediate, got {}",
                arch.intermediate_size
            ));
        }
        if arch.top_k_experts as usize > gpu::MAX_STREAMED_EXPERTS {
            return unsupported(format!(
                "top_k {} exceeds the {}-slot MoE kernels",
                arch.top_k_experts,
                gpu::MAX_STREAMED_EXPERTS
            ));
        }
        if arch
            .full_attention_layer_mask
            .iter()
            .any(|&m| m != 1 && m != 2)
        {
            return unsupported(
                "this flow's layers are full attention (1) or gated DeltaNet (2) only".to_string(),
            );
        }
        if !arch.has_linear_attention_layers() {
            return unsupported(
                "a hybrid Qwen install with no linear layers is not one this flow can run"
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

        // Fail at open, not at token 1: probe one layer of each kind for
        // the tensors this flow will bind.
        let hidden = arch.hidden_size as usize;
        let num_experts = arch.num_experts as usize;
        for layer in 0..arch.num_layers as usize {
            let mut probes = vec![
                layer_tensor(layer, "input_layernorm.weight"),
                layer_tensor(layer, "post_attention_layernorm.weight"),
            ];
            // The FFN half is the only thing the two shapes disagree on. A
            // dense layer's three names are the ones the MoE half's SHARED
            // expert spells `mlp.shared_expert.*`, one level shallower.
            probes.extend(
                if dense {
                    [
                        "mlp.gate_proj.weight",
                        "mlp.up_proj.weight",
                        "mlp.down_proj.weight",
                    ]
                    .as_slice()
                } else {
                    [
                        "mlp.gate.weight",
                        "mlp.shared_expert_gate.weight",
                        "mlp.shared_expert.gate_proj.weight",
                    ]
                    .as_slice()
                }
                .iter()
                .map(|s| layer_tensor(layer, s)),
            );
            probes.extend(if arch.layer_is_linear(layer) {
                [
                    "linear_attn.in_proj_qkv.weight",
                    "linear_attn.conv1d.weight",
                    // No `.weight` suffix on either of these.
                    "linear_attn.A_log",
                    "linear_attn.dt_bias",
                ]
                .iter()
                .map(|s| layer_tensor(layer, s))
                .collect::<Vec<_>>()
            } else {
                ["self_attn.q_proj.weight", "self_attn.q_norm.weight"]
                    .iter()
                    .map(|s| layer_tensor(layer, s))
                    .collect::<Vec<_>>()
            });
            for name in probes {
                entry(index, &name)?;
            }
        }
        // A headless trunk (`model_io::trunk_role`) has no head by design: the
        // model that owns it applies its own to the final hidden state. Every
        // other install must carry one. The final norm is required by both.
        if !headless {
            let head_name = if arch.tie_word_embeddings {
                "language_model.model.embed_tokens.weight"
            } else {
                "language_model.lm_head.weight"
            };
            entry(index, head_name)?;
        }
        entry(index, "language_model.model.norm.weight")?;
        let _ = weights;

        let ones: Vec<u8> = (0..hidden).flat_map(|_| BF16_ONE.to_le_bytes()).collect();
        let router_ones = context.new_output_buffer(ones.len() as u64);
        gpu::write_buffer_bytes(&router_ones, 0, &ones);

        let q_dim = (arch.num_heads * head_dim) as usize;
        let qkv_dim = shape.qkv_dim() as usize;
        let value_dim = shape.value_dim() as usize;
        let v_heads = shape.num_v_heads as usize;
        // The folded-weight plan, built BEFORE the scratch closure below so
        // the two never hold `context` at once. `value_dim` rather than a
        // second `q_dim` is deliberate: the two coincide on the real
        // checkpoint (24 heads of 256; 48 v-heads of 128) but describe
        // different tensors, and the scratch pairs with the BLOCK that
        // consumes it, not with a width.
        let hadamard = match hadamard {
            Some((section, signs_bytes)) => Some(HadamardPlan::build(
                context,
                section,
                signs_bytes,
                index,
                hidden,
                q_dim,
                arch.intermediate_size as usize,
                value_dim,
            )?),
            None => None,
        };
        // The prefetch experiment reads the same one-layer-ahead guess the
        // probe makes, so it needs the probe's buffers even with no
        // histogram.
        let prefetch = if num_experts > 0 {
            super::prefetch::width_from_env(arch.top_k_experts.max(0) as usize)
                .map(|k| super::prefetch::Prefetcher::spawn(k, arch.top_k_experts as usize))
        } else {
            None
        };
        let pilot = (num_experts > 0
            && (prefetch.is_some() || std::env::var_os("TURBOSPARK_PILOT_PROBE").is_some()))
        .then(|| PilotBuffers {
            x: context.new_output_buffer((hidden.max(1) * 2) as u64),
            logits: context.new_output_buffer((num_experts * 4) as u64),
        });
        let halfs = |n: usize| context.new_output_buffer((n.max(1) * 2) as u64);
        Ok(Self {
            gdn: gpu::GdnStateManager::new(context.device(), arch),
            shape,
            rotary_dim: rotary_dim as u32,
            dense,
            shared_read_overlap,
            moe_chunked_prefill,
            mapped_demand_preparation,
            router_ones,
            per_expert_ones: vec![1.0; num_experts],
            // `new_output_buffer(0)` is not a thing worth finding out about at
            // the first dispatch, and a dense flow never binds this.
            // `MAX_PREFILL_BATCH` rows: the MoE chunked driver reads every
            // token's router logits back in one host round trip after the
            // layer's single `cb1` commit, so a single row would hand every
            // token the last token's routing. Decode uses row 0.
            router_logits_f32: context
                .new_output_buffer((num_experts.max(1) * MAX_PREFILL_BATCH * 4) as u64),
            pilot,
            prefetch,
            q_packed: halfs(2 * q_dim),
            attn_gate: halfs(q_dim),
            gdn_qkv_raw: halfs(qkv_dim),
            gdn_conv_out: halfs(qkv_dim),
            gdn_z: halfs(value_dim),
            gdn_a: halfs(v_heads),
            gdn_b: halfs(v_heads),
            gdn_y: halfs(value_dim),
            gdn_out: halfs(value_dim),
            // `MAX_PREFILL_BATCH` rows, for the same reason as
            // `router_logits_f32`: the MoE chunked driver writes every token's
            // post-attention norm in `cb1` and reads it one commit later in
            // that token's routed pass. Decode and the dense driver use row 0.
            moe_x: halfs(hidden * MAX_PREFILL_BATCH),
            h1: halfs(hidden),
            h2: halfs(hidden),
            shared_gate_logit: halfs(1),
            batched_prefill: None,
            hadamard,
        })
    }

    /// Allocates the chunked driver's M-row scratch the first time its
    /// batched-GEMV arm asks for it. Afterwards it only adds or drops the
    /// multi-row attention partials to match `rows_attention`.
    ///
    /// Idempotent rather than "call once", because the entry point that
    /// calls it (`prefill_chunk_real_qwen_dense`) is reached once per CHUNK
    /// and not once per run -- `RealGemmaState::ensure_batched`'s reason,
    /// verbatim.
    ///
    /// Sized at [`crate::real_forward_types::MAX_PREFILL_BATCH`], which is
    /// `gpu::MAX_BATCH_ROWS` and not a second 16 that agrees by luck: the
    /// batched INT4 GEMM caps a dispatch at its per-thread accumulator
    /// array, and the driver's micro-batch is capped at the same constant,
    /// so a full micro-batch is exactly one dispatch with no sub-batching.
    /// `steel` sizes the scratch for the Steel micro-batch
    /// (`QWEN_STEEL_PREFILL_BATCH` rows) and lets its GEMMs take the Steel
    /// kernel; otherwise it is the 16-row scalar arm's. Fixed at first
    /// allocation: the choice is the runner's, read once at open.
    pub(crate) fn ensure_batched_prefill(
        &mut self,
        context: &mut gpu::MetalContext,
        arch: &ArchConfig,
        steel: bool,
        rows_attention: bool,
    ) -> Result<(), RealForwardError> {
        if self.batched_prefill.is_none() {
            self.allocate_batched_prefill(context, arch, steel)?;
        }
        // Reconciled on every call, unlike the Steel choice, so one runner
        // can A/B it (`qwen38_prefill_arm_probe`). 3.1 MiB of partials at
        // Qwen3.8's shape and 64 rows, allocated only where the shape is one
        // the kernel serves.
        let (heads, kv_heads, head_dim) = (
            arch.num_heads as u32,
            arch.num_full_kv_heads as u32,
            arch.full_head_dim as u32,
        );
        let scratch = self.batched_prefill.as_mut().expect("allocated just above");
        let want = rows_attention && gpu::attention_decode_gqa_supported(head_dim, heads, kv_heads);
        if want && scratch.rows_attention.is_none() {
            scratch.rows_attention = Some(gpu::GqaRowsScratch::new(
                context,
                scratch.batch as u32,
                heads,
                head_dim,
                gpu::GQA_ROWS_SCRATCH_CHUNKS,
            ));
        } else if !want {
            scratch.rows_attention = None;
        }
        Ok(())
    }

    fn allocate_batched_prefill(
        &mut self,
        context: &mut gpu::MetalContext,
        arch: &ArchConfig,
        steel: bool,
    ) -> Result<(), RealForwardError> {
        let rows = if steel {
            crate::real_forward_types::QWEN_STEEL_PREFILL_BATCH
        } else {
            crate::real_forward_types::MAX_PREFILL_BATCH
        };
        let mut scratch = BatchedScratch::new(
            context, arch, self.shape, rows,
            // No tape: a prefill chunk is never rolled back, so the
            // buffers would be dead weight on every chunked prompt.
            false,
        )
        .map_err(RealForwardError::Gpu)?;
        scratch.steel_gemm = steel;
        self.batched_prefill = Some(scratch);
        Ok(())
    }

    /// The chunked driver's M-row scratch, reachable only from inside that
    /// driver's batched arm -- downstream of the `ensure_batched_prefill` at
    /// its entry point, so the `expect` is a structural invariant rather
    /// than a hope.
    pub(crate) fn batched_prefill(&self) -> &BatchedScratch {
        self.batched_prefill.as_ref().expect(
            "both qwen chunked drivers call ensure_batched_prefill before their batched arm runs",
        )
    }

    /// Rewinds the recurrent state to empty context. Both the delta rule
    /// and the causal conv define that state as zero.
    pub(crate) fn reset(&mut self) {
        self.gdn.reset();
    }
}
