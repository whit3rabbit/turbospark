//! `RealDeepseek2State`: everything [`crate::real_forward::RealForwardRunner`]
//! allocates once when it opens a `deepseek2` install. The MLA geometry,
//! the yarn rope table, the shared-expert and router scratch, and the
//! validations the family refuses on (docs/DEEPSEEK2_PHASE0.md).
//!
//! One decoder layer:
//!
//! ```text
//! a      = rms_norm(x, input_layernorm)              // eps 1e-6
//! q      = a @ q_proj                                // [heads * (nope+rope)]
//! row    = a @ kv_a_proj                             // [latent + rope]
//! row    = mla_kv_norm(row, kv_a_norm)               // latent half, in place
//! k_pe   = yarn_rope(row[latent..], pos)             // in place, in the cache
//! K[pos] = row                                       // ONE row, no V row
//! q_pe   = yarn_rope(q pe windows, pos)              // per-head tail
//! q'     = [W_uk_h @ q_nope_h ; q_pe_h]              // absorb, per head
//! attn_h = softmax(q' . K * scale) . c               // MQA, V = K row
//! out_h  = W_uv_h @ attn_h                           // v-combine, per head
//! x      = x + o_proj(out)
//! f      = rms_norm(x, post_attention_layernorm)
//! layer 0:  x += dense SwiGLU(f)                     // dense lead
//! else:     h = softmax(f @ router) top-6, NO renorm
//!           x += routed_sum + shared SwiGLU(f)       // fused 2-expert shexp
//! ```
//!
//! Final: `rms_norm(x, model.norm)`, raw-logit head (untied).

use model_io::{ArchConfig, ModelFamily, ResidentIndex};

use crate::real_forward::RealForwardError;
use crate::real_forward_types::MAX_PREFILL_BATCH;

/// BF16 bit pattern for 1.0 (the INT8 router kernel's effective scale).
const BF16_ONE: u16 = 0x3F80;

pub(crate) struct RealDeepseek2State {
    /// RMS epsilon, read off the family: the architecture publishes 1e-6.
    pub(crate) rms_eps: f32,
    /// BF16 `[hidden]` of ones: the INT8 router kernel's effective scale,
    /// identically 1 (this architecture has no `router.scale`).
    pub(crate) router_ones: gpu::MetalBuffer,

    // ---- MLA geometry (halves; docs/DEEPSEEK2_PHASE0.md) ----
    pub(crate) heads: u32,
    /// Per-head non-rotated query/key width, `qk_nope_head_dim`.
    pub(crate) nope: u32,
    /// Per-head rope-carried width; also the rope dimension count.
    pub(crate) rope_dim: u32,
    /// The latent rank; the compressed row's head and the v-read width.
    pub(crate) kv_lora: u32,
    /// Per-head value width the v-combine produces.
    pub(crate) v_dim: u32,
    /// `nope + rope`: the q projection's per-head width.
    pub(crate) head_dim: u32,
    /// `kv_lora + rope`: the cache row, and the fused q' row's width.
    pub(crate) cache_row: u32,
    /// YaRN magnitude multiplier, `1 + 0.1 * mscale_param * ln(factor)`.
    /// Scales the rope rotation; its square is already folded into
    /// `arch.attention_scale` at repack, exactly as the HF reference does.
    pub(crate) rope_mscale: f32,
    /// Position-independent per-pair frequencies from
    /// `compute::yarn_frequencies`, `rope_dim / 2` floats.
    pub(crate) rope_frequencies: gpu::MetalBuffer,

    // ---- per-token scratch (preallocated at open; the hot path
    // allocates no Metal buffers, per NEW_MODEL.md Phase 3) ----
    /// `[heads * head_dim]`: the q projection, pe halves roped in place.
    pub(crate) q: gpu::MetalBuffer,
    /// `[heads * cache_row]`: the fused absorbed query the attention reads.
    pub(crate) q_abs: gpu::MetalBuffer,
    /// `[heads * kv_lora]`: the softmax-weighted latent, pre v-combine.
    pub(crate) attn_c: gpu::MetalBuffer,
    /// `[heads * v_dim]`: the per-head v-combine outputs, concatenated;
    /// o_proj consumes this.
    pub(crate) mla_out: gpu::MetalBuffer,
    /// `[max(dense lead width, shared expert width)]`: FFN gate/up/act
    /// scratch, sized to the WIDEST intermediate this architecture writes.
    /// The dense lead runs 10944 wide and the shared expert 2816 against a
    /// 2048-wide residual stream -- reusing a hidden-sized buffer for
    /// either overflows into whatever the allocator placed next (here: the
    /// yarn frequency table, whose clobber was invisible at position 0,
    /// where any table gives the identity rope, and corrupted every later
    /// position nondeterministically).
    pub(crate) ffn_a: gpu::MetalBuffer,
    pub(crate) ffn_b: gpu::MetalBuffer,
    /// `[hidden]`: the shared expert's SwiGLU output, phase 2's residual
    /// seed (`y = shared + sum(w * expert)`).
    pub(crate) shared_out: gpu::MetalBuffer,

    // ---- MoE scratch, the llama shapes ----
    /// FP32 router logits, one `[num_experts]` row per micro-batch token.
    pub(crate) router_logits_f32: gpu::MetalBuffer,
    /// `[hidden]` per token: the post-attention norm that feeds router,
    /// routed experts and the shared expert.
    pub(crate) moe_x: gpu::MetalBuffer,
    /// `[hidden]`: the routed-plus-shared sum, added back to the stream.
    pub(crate) h2: gpu::MetalBuffer,
    /// Dense lead layer count (`first_k_dense_replace`): these layers run
    /// the plain dense FFN and hold no expert blob, so every blob position
    /// and streamer index is `layer - lead`.
    pub(crate) lead: usize,
    /// The dense lead's FFN width, a DIFFERENT number from the shared
    /// expert's (`intermediate_size`).
    pub(crate) dense_inter: usize,
    /// The shared expert's SwiGLU width (`intermediate_size` on this
    /// architecture: the fused pair's width).
    pub(crate) shared_inter: usize,
}

impl RealDeepseek2State {
    pub(crate) fn build(
        context: &mut gpu::MetalContext,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        let unsupported = |detail: String| Err(RealForwardError::Unsupported(detail));
        if arch.family != ModelFamily::Deepseek2 {
            return unsupported("deepseek2 state built for another family".into());
        }
        // Every behavioural extension this architecture does NOT have,
        // checked against the manifest rather than assumed (each is a field
        // with a Gemma fallback, AGENTS.md Gotcha 24).
        if arch.ffn_sandwich_norms
            || arch.router_scaled
            || arch.embedding_scaled_by_sqrt_hidden
            || arch.attn_output_gate
            || arch.shared_expert_gated
            || arch.attention_k_eq_v
        {
            return unsupported(
                "the deepseek2 flow takes none of ffnSandwichNorms / routerScaled / \
                 embeddingScaledBySqrtHidden / attnOutputGate / sharedExpertGated / \
                 attentionKEqV"
                    .to_string(),
            );
        }
        if arch.final_logit_softcap != 0.0 {
            return unsupported("the deepseek2 architecture has no final logit softcap".into());
        }
        if arch.full_attention_layer_mask.iter().any(|&m| m != 5) {
            return unsupported(
                "every deepseek2 layer is MLA (mask 5); a mixed mask is not wired".to_string(),
            );
        }
        let mla = arch.mla;
        if !mla.is_active() {
            return unsupported("deepseek2 install declares no MLA geometry".into());
        }
        if mla.q_lora_rank != 0 {
            return unsupported(format!(
                "q low-rank {} is the full V2/V3 branch, which this port does not execute",
                mla.q_lora_rank
            ));
        }
        let heads = arch.num_heads as u32;
        let nope = mla.nope_head_dim as u32;
        let rope_dim = mla.rope_head_dim as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        let v_dim = mla.v_head_dim as u32;
        let head_dim = mla.key_head_dim() as u32;
        let cache_row = mla.cache_row_dim() as u32;
        // The projection shapes the manifest claims must agree with the
        // geometry: heads * head_dim rows of q, heads * (nope + v) rows of
        // kv_b, heads * v_dim of o.
        let q_rows = arch.num_heads * mla.key_head_dim();
        let kv_b_rows = arch.num_heads * (mla.nope_head_dim + mla.v_head_dim);
        let o_rows = arch.num_heads * mla.v_head_dim;
        if q_rows != heads as i64 * head_dim as i64
            || kv_b_rows != arch.num_heads * (mla.nope_head_dim + mla.v_head_dim)
            || o_rows != arch.num_heads * mla.v_head_dim
        {
            return unsupported(
                "deepseek2 head geometry disagrees with itself (heads x (nope + rope) / \
                 (nope + v) / v)"
                    .to_string(),
            );
        }
        // The kernels' fixed contracts: the attention accumulator is
        // 512 wide (2 halves per thread over 256 threads), the Q8_0 dots
        // walk whole 32-element blocks, and the GGUF phase-2 reduce
        // hardcodes 8 slots.
        if kv_lora != 512 {
            return unsupported(format!(
                "the MLA attention kernel fixes its accumulator at kv_lora 512; this \
                 install declares {kv_lora}"
            ));
        }
        if nope % 32 != 0 || kv_lora % 32 != 0 || v_dim % 8 != 0 || rope_dim % 2 != 0 {
            return unsupported(
                "deepseek2 MLA dims must keep nope/kv_lora on 32-block and v_dim on \
                 8-row boundaries, and rope even"
                    .to_string(),
            );
        }
        let top_k = arch.top_k_experts as usize;
        if top_k == 0 || arch.num_experts == 0 {
            return unsupported("deepseek2 installs are MoE; experts must be positive".into());
        }
        if top_k > gpu::PHASE2_FIXED_SLOTS {
            return unsupported(format!(
                "top_k {top_k} exceeds the {}-slot GGUF MoE kernels",
                gpu::PHASE2_FIXED_SLOTS
            ));
        }
        // The routed blob files start after the dense lead, and the flow
        // maps checkpoint layer L to blob position L - lead.
        let lead = arch.num_dense_leading_layers.max(0) as usize;
        if lead >= arch.num_layers as usize {
            return unsupported("dense lead leaves no MoE layers".into());
        }
        if arch.dense_lead_intermediate_size <= 0 {
            return unsupported(
                "deepseek2 declares a dense lead count but no dense lead FFN width".into(),
            );
        }

        // YaRN: the frequency table from the checkpoint's own scalars. The
        // rope MAGNITUDE is 1.0, deliberately, and this is the subtlest
        // constant in the family: llama.cpp's DEEPSEEK2 yarn setup
        // (`llama-context.cpp`, the TAG_DEEPSEEK2_YARN_LOG_MUL_FIX block)
        // sets `yarn_attn_factor = get_mscale(factor, mscale) /
        // get_mscale(factor, mscale_all_dim) * (1 + 0.1*log(factor))^-1`,
        // which for mscale == mscale_all_dim (this checkpoint's 0.707/0.707)
        // is exactly 0.7305 -- so ggml's internal rope multiplier
        // `yarn_attn_factor * (1 + 0.1*log(1/freq_scale))` nets to 1.0 and
        // the rotated values carry NO magnitude scaling. The mscale
        // parameter's whole effect lands in the attention score instead,
        // whose scale (1 + 0.1 * mscale_all_dim * ln factor)^2 / sqrt(key)
        // is the manifest's `attentionScale` and already correct. Applying
        // the HF-style 1.2608 rope magnitude HERE as well scores the rope
        // terms 1.5896x too high: position 0 still decodes (rope is the
        // identity there) and the first generated token survives, and every
        // position after degrades -- found against llama.cpp on the same
        // Q8_0 bytes, whose teacher-forced logits this flow now reproduces
        // (`docs/DEEPSEEK2_PHASE0.md`, the cross-engine section).
        let _ = arch.rope_scaling.yarn_mscale(); // documented above; NOT applied
        let rope_mscale: f32 = 1.0;
        let yarn = compute::yarn_frequencies(
            rope_dim as usize,
            arch.full_rope_theta as f32,
            arch.rope_scaling.factor as f32,
            arch.rope_scaling.original_context,
            arch.rope_scaling.beta_fast as f32,
            arch.rope_scaling.beta_slow as f32,
        );
        let freq_bytes: Vec<u8> = yarn
            .frequencies
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let rope_frequencies = context.new_output_buffer(freq_bytes.len().max(4) as u64);
        gpu::write_buffer_bytes(&rope_frequencies, 0, &freq_bytes);

        // Little-endian, exactly as the llama state's own fill: the buffer
        // is read as native u16 BF16 patterns.
        let ones: Vec<u8> = (0..arch.hidden_size as usize)
            .flat_map(|_| BF16_ONE.to_le_bytes())
            .collect();
        let router_ones = context.new_output_buffer(ones.len() as u64);
        gpu::write_buffer_bytes(&router_ones, 0, &ones);

        // Every tensor the flow will name must EXIST at open, not at
        // token 1. The dense lead's FFN triple and the first MoE layer's
        // router + shexp triple are checked so the error names the real gap.
        for (layer, suffix) in [
            (0usize, "mlp.gate_proj.weight"),
            (0usize, "mlp.up_proj.weight"),
            (0usize, "mlp.down_proj.weight"),
            (lead, "mlp.gate.weight"),
            (lead, "mlp.shared_expert.gate_proj.weight"),
            (lead, "mlp.shared_expert.up_proj.weight"),
            (lead, "mlp.shared_expert.down_proj.weight"),
        ] {
            let name = crate::families::deepseek2::layer_tensor(layer, suffix);
            if !index.entries.contains_key(&name) {
                return Err(RealForwardError::MissingTensor(name));
            }
        }

        let hidden = arch.hidden_size as usize;
        let bf16 = |n: usize| (n * 2) as u64;
        Ok(Self {
            rms_eps: 1e-6,
            router_ones,
            heads,
            nope,
            rope_dim,
            kv_lora,
            v_dim,
            head_dim,
            cache_row,
            rope_mscale,
            rope_frequencies,
            q: context.new_output_buffer(bf16(heads as usize * head_dim as usize)),
            q_abs: context.new_output_buffer(bf16(heads as usize * cache_row as usize)),
            attn_c: context.new_output_buffer(bf16(heads as usize * kv_lora as usize)),
            mla_out: context.new_output_buffer(bf16(heads as usize * v_dim as usize)),
            ffn_a: context.new_output_buffer(bf16(
                arch.dense_lead_intermediate_size
                    .max(arch.intermediate_size) as usize,
            )),
            ffn_b: context.new_output_buffer(bf16(
                arch.dense_lead_intermediate_size
                    .max(arch.intermediate_size) as usize,
            )),
            shared_out: context.new_output_buffer(bf16(hidden)),
            router_logits_f32: context
                .new_output_buffer((MAX_PREFILL_BATCH * arch.num_experts as usize * 4) as u64),
            moe_x: context.new_output_buffer(bf16(MAX_PREFILL_BATCH * hidden)),
            h2: context.new_output_buffer(bf16(hidden)),
            lead,
            dense_inter: arch.dense_lead_intermediate_size as usize,
            shared_inter: arch.intermediate_size as usize,
        })
    }
}
