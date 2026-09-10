use crate::arch_config::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

fn qwen_gdn_moe_layer_mask() -> Vec<u8> {
    qwen_hybrid_layer_mask(40)
}

/// The Qwen hybrid layer mask: gated-DeltaNet linear everywhere except
/// every 4th layer, which is full attention.
///
/// Layer kinds: 2 = gated-DeltaNet linear, 1 = full attention on every 4th
/// layer (`(i + 1) % 4 == 0`). Shared by Qwen 3.6 at 40 layers and
/// `qwen3_5` at 64, which both declare `full_attention_interval: 4` and
/// whose `layer_types` lists reproduce exactly this pattern.
fn qwen_hybrid_layer_mask(layers: usize) -> Vec<u8> {
    let mut mask = vec![2u8; layers];
    let mut i = 3;
    while i < layers {
        mask[i] = 1;
        i += 4;
    }
    mask
}

/// Canonical `qwen3_5` baseline: a 64-layer hybrid of 48 gated-DeltaNet
/// linear-attention layers and 16 full-attention layers (every 4th), a DENSE
/// SwiGLU FFN, untied lm_head, no logit softcap and no sliding window.
///
/// **It is named for the ARCHITECTURE rather than for a checkpoint because it
/// serves two published ones**, which is what the rename off `bonsai_27b`
/// records. `prism-ml/Bonsai-27B-mlx-1bit` (ROADMAP's 1-bit entry) came
/// first; `Qwen/Qwen3.8-27B` arrived 2026-08-14 and its `text_config` agrees
/// with Bonsai's on 33 of 35 fields -- everything below is identical, and the
/// only differences are `eos_token_id` (248044 against 248046, a tokenizer
/// concern that reaches no field here) and the quantization block, which is
/// not part of an `ArchConfig` either. The two therefore share this baseline
/// exactly, and the QUANTIZATION is what a reader should expect to differ
/// between installs of them: 1-bit at group 128 for Bonsai, INT4 at group 64
/// for the mlx-community Qwen3.8 artifact.
///
/// **Every behavioural field here is Qwen 3.6's and every shape field
/// differs**, which is what makes this the same relationship dense Mistral
/// has to Mixtral. All of it is read off the checkpoint's own `config.json`
/// (`model_type: qwen3_5`, `text_config.model_type: qwen3_5_text`) rather
/// than inferred from the sibling: `head_dim` 256, `vocab_size` 248320,
/// `rope_theta` 1e7, `partial_rotary_factor` 0.25, `attn_output_gate`, the
/// linear key/value head dims and the conv kernel all coincide with Qwen
/// 3.6 as published, while hidden is 5120 against 2048, layers 64 against
/// 40, and there are no experts at all.
///
/// Two fields are worth reading twice. `intermediate_size` is the DENSE FFN
/// width (17408) where Qwen 3.6's is its shared expert's 512, so the same
/// field name means a different thing in the two baselines -- the dense
/// `llama` half has exactly this collision. And `rope_scaling` is `NONE`
/// even though the checkpoint declares `mrope_section [11, 11, 10]`:
/// [`RopeScalingConfig`] carries YARN's four scalars, and mrope is not
/// YaRN. On TEXT positions mrope's three sections are equal and it reduces
/// to the `rope_neox_subdim` this baseline already sets, which is a claim
/// the cross-engine check has to verify rather than one to assume.
pub fn qwen_gdn_dense_27b() -> ArchConfig {
    ArchConfig {
        hidden_size: 5120,
        // The dense FFN, not a shared expert. See the doc above.
        intermediate_size: 17408,
        // No routed experts, so no routed width.
        moe_intermediate_size: 0,
        num_heads: 24,
        num_kv_heads: 4,
        num_full_kv_heads: 4,
        head_dim: 256,
        full_head_dim: 256,
        vocab_size: 248_320,
        sliding_window: 0,
        final_logit_softcap: 0.0,
        rope_theta: 10_000_000.0,
        full_rope_theta: 10_000_000.0,
        partial_rotary_factor: 0.25,
        num_layers: 64,
        num_experts: 0,
        top_k_experts: 0,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: qwen_hybrid_layer_mask(64),
        hidden_activation: "silu".to_string(),
        family: ModelFamily::QwenGdnDense,
        attn_output_gate: true,
        attention_scale: 0.0625, // 256^-0.5, a binary fraction (Gotcha 24)
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        // No shared expert to gate: the FFN is dense.
        shared_expert_gated: false,
        rope_neox_subdim: true,
        linear_attention: LinearAttentionConfig {
            num_k_heads: 16,
            // 48, against Qwen 3.6's 32: three V heads per K head rather
            // than two.
            num_v_heads: 48,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel_size: 4,
            output_gate_sigmoid: false,
        },
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: RopeScalingConfig::NONE,
        vision: VisionConfig::NONE,
        ple: PleConfig::NONE,
    }
}

/// Canonical Qwen3.6-35B-A3B baseline: a 40-layer hybrid of 30
/// gated-DeltaNet linear-attention layers and 10 full-attention layers
/// (every 4th layer), 256 routed experts (top-8) plus a sigmoid-gated
/// shared expert, SwiGLU activations, untied lm_head, no logit softcap.
pub fn qwen_gdn_moe_35b_a3b() -> ArchConfig {
    ArchConfig {
        hidden_size: 2048,
        intermediate_size: 512,
        moe_intermediate_size: 512,
        num_heads: 16,
        num_kv_heads: 2,
        num_full_kv_heads: 2,
        head_dim: 256,
        full_head_dim: 256,
        vocab_size: 248_320,
        sliding_window: 0,
        final_logit_softcap: 0.0,
        rope_theta: 10_000_000.0,
        full_rope_theta: 10_000_000.0,
        partial_rotary_factor: 0.25,
        num_layers: 40,
        num_experts: 256,
        top_k_experts: 8,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: qwen_gdn_moe_layer_mask(),
        hidden_activation: "silu".to_string(),
        family: ModelFamily::QwenGdnMoe,
        attn_output_gate: true,
        attention_scale: 0.0625, // 256^-0.5
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: true,
        rope_neox_subdim: true,
        linear_attention: LinearAttentionConfig {
            num_k_heads: 16,
            num_v_heads: 32,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel_size: 4,
            output_gate_sigmoid: false,
        },
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: RopeScalingConfig::NONE,
        vision: VisionConfig::NONE,
        ple: PleConfig::NONE,
    }
}

/// Canonical Qwen3-30B-A3B baseline (ROADMAP Phase M, the fine-grained MoE
/// follow-on): 48 full-attention layers, 32 query heads over 4 KV heads at
/// head_dim 128, 128 routed experts at top-8 with NO shared expert, SwiGLU,
/// untied lm_head, no logit softcap, no sliding window.
///
/// **Every number here was read off the published
/// `Qwen/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf` header**, not from a
/// model card (`gguf_checkpoint_network.rs::scopes_qwen3moe_*`).
///
/// `head_dim` is 128 while `hidden_size / num_heads` is 64, so the query
/// projection emits 4096 rows against a 2048-wide stream. That is the usual
/// case rather than the exception (`docs/NEW_MODEL.md` Phase 0 says so), and
/// deriving the head dim from the hidden size would halve every GEMV here.
///
/// `intermediate_size` is the published `feed_forward_length`, which this
/// checkpoint carries and never uses: every layer is MoE and there is no
/// dense or shared FFN tensor in the file.
///
/// THE FIELD THAT MADE THIS FAMILY WORTH BRINGING UP is
/// `moe_intermediate_size = 768`: one expert is `3 x 768 x 2048` at Q4_K,
/// i.e. 2.5 MiB, so 16 slots over 48 layers pin 1.90 GiB. Mixtral's same
/// arithmetic reads 108.9 MiB and 54.5 GiB (AGENTS.md Gotcha 36).
pub fn qwen3_30b_a3b() -> ArchConfig {
    ArchConfig {
        hidden_size: 2048,
        intermediate_size: 6144,
        moe_intermediate_size: 768,
        num_heads: 32,
        num_kv_heads: 4,
        num_full_kv_heads: 4,
        head_dim: 128,
        full_head_dim: 128,
        vocab_size: 151_936,
        sliding_window: 0,
        final_logit_softcap: 0.0,
        rope_theta: 1_000_000.0,
        full_rope_theta: 1_000_000.0,
        // Full rotary: the whole head is rotated, as on the `llama` side.
        partial_rotary_factor: 1.0,
        num_layers: 48,
        num_experts: 128,
        top_k_experts: 8,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: vec![1u8; 48],
        hidden_activation: "silu".to_string(),
        family: ModelFamily::Qwen3Moe,
        attn_output_gate: false,
        // 128^-0.5 = 2^-3.5, the same non-binary-fraction value Mixtral's
        // head dim produces; AGENTS.md Gotcha 24's round-trip warning
        // applies and `crates/model-io/tests/arch_config.rs` pins it.
        attention_scale: 0.088_388_347_648_318_45,
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: false,
        rope_neox_subdim: false,
        linear_attention: LinearAttentionConfig::NONE,
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: RopeScalingConfig::NONE,
        vision: VisionConfig::NONE,
        ple: PleConfig::NONE,
    }
}

/// Canonical dense `qwen3` baseline (`ModelFamily::Qwen3Dense`): plain GQA
/// with per-head QK-norm, a dense SwiGLU FFN, full-head NeoX RoPE, no
/// sliding window, no softcap. Shape fields are `Qwen/Qwen3-4B`'s
/// (`config.json`, read 2026-09-09); behavioral fields are
/// [`qwen3_30b_a3b`]'s, since the two share the SAME attention block
/// (`docs/QWEN3_PHASE0.md`).
///
/// **This baseline serves every dense `qwen3` size (0.6B through 32B,
/// Qwen3-Coder's dense sizes, DeepSeek-R1-0528-Qwen3-8B), not just the 4B
/// checkpoint it is named for.** As with every other GGUF-intake family,
/// shape fields (`hidden_size`, `num_layers`, `num_heads`, `num_kv_heads`,
/// `vocab_size`, `tie_word_embeddings`, ...) are overridden per-checkpoint
/// by `arch_from_gguf` reading the file's own metadata
/// (`crates/repack/src/gguf_config/mod.rs`); only the numbers below that
/// `arch_from_gguf` never touches -- `attn_output_gate`, `attention_scale`,
/// `ffn_sandwich_norms`, `rope_neox_subdim`, etc. -- are the ones a real
/// install actually inherits from this function. `head_dim` is 128 at every
/// published dense `qwen3` size checked (0.6B/4B/8B), independent of
/// `hidden_size / num_heads`, so `attention_scale` (`head_dim^-0.5`) is
/// stable across the whole family exactly as it is for
/// [`qwen3_30b_a3b`]'s MoE sibling; `arch_from_gguf` still reads
/// `attention.key_length` per file rather than trusting that agreement.
///
/// `tie_word_embeddings` varies by published size (Qwen3-4B: true,
/// Qwen3-8B: false) and is read from the GGUF's own `output.weight` tensor
/// presence at repack time, not fixed here.
pub fn qwen3_4b() -> ArchConfig {
    ArchConfig {
        hidden_size: 2560,
        intermediate_size: 9728,
        // Dense: no routed experts, so no routed width.
        moe_intermediate_size: 0,
        num_heads: 32,
        num_kv_heads: 8,
        num_full_kv_heads: 8,
        head_dim: 128,
        full_head_dim: 128,
        vocab_size: 151_936,
        sliding_window: 0,
        final_logit_softcap: 0.0,
        rope_theta: 1_000_000.0,
        full_rope_theta: 1_000_000.0,
        // Full rotary, as `qwen3moe`.
        partial_rotary_factor: 1.0,
        num_layers: 36,
        num_experts: 0,
        top_k_experts: 0,
        // `Qwen/Qwen3-4B`'s own value; per-checkpoint after that (see doc).
        tie_word_embeddings: true,
        attention_k_eq_v: false,
        // Every layer is full attention: dense `qwen3` publishes no
        // `attention.sliding_window` key, same as `qwen3moe`.
        full_attention_layer_mask: vec![1u8; 36],
        hidden_activation: "silu".to_string(),
        family: ModelFamily::Qwen3Dense,
        attn_output_gate: false,
        // 128^-0.5 = 2^-3.5, the same value `qwen3_30b_a3b` uses for the
        // same head_dim; AGENTS.md Gotcha 24's round-trip warning applies.
        attention_scale: 0.088_388_347_648_318_45,
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        // No shared expert to gate: the FFN is dense.
        shared_expert_gated: false,
        rope_neox_subdim: false,
        linear_attention: LinearAttentionConfig::NONE,
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: RopeScalingConfig::NONE,
        vision: VisionConfig::NONE,
        ple: PleConfig::NONE,
    }
}

/// Canonical `qwen4_exp` (Qwen3.8-Flash-Next) baseline: a 48-layer hybrid of
/// 36 gated-DeltaNet linear layers and 12 full-attention ones (every 4th),
/// 512 routed experts at top-10 plus a gated shared expert, a FOUR-STREAM
/// residual, and a hashed n-gram per-layer embedding at layer index 1.
///
/// **Named for the ARCHITECTURE rather than a checkpoint, on
/// [`qwen_gdn_dense_27b`]'s precedent, because it serves two published ones.**
/// `pipenetwork/Qwen3.8-Flash-Next-MLX-4bit` declares 512 experts and
/// `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit` declares 288; the two
/// `text_config` blocks are otherwise EQUAL, field for field, which
/// `crates/repack`'s `every_published_checkpoint_parses_to_one_baseline`
/// asserts offline. `num_experts` here is the unpruned 512 because that is the
/// architecture's own shape, and a REAP install carries its own count in the
/// manifest -- `arch_validation` compares against the config the repack
/// DERIVED, so this value is the GGUF-derivation fallback and the canonical
/// reference, never a constraint on an install.
///
/// Everything below is read off the checkpoint's own `config.json`
/// (`model_type: qwen4_exp`, `text_config.model_type: qwen4_exp_text`).
///
/// **Three fields carry the whole reason this is not [`qwen_gdn_moe_35b_a3b`]
/// at different shapes.**
///
/// `hyper_connections` is ACTIVE at `mult: 4`, so the residual stream is
/// 10,240 wide rather than 2,560 for the entire stack, and every residual add
/// becomes a gated read/inject pair. `lowrank: 320` is the mixing bottleneck;
/// `sinkhorn_iters` and `eps` stay 0 because those belong to DeepSeek's mHC,
/// which is a different mixer sharing the struct.
///
/// `linear_attention.output_gate_sigmoid` is TRUE, from the checkpoint's
/// `output_gate_type: "sigmoid"`. Every earlier family reaching that kernel
/// declares silu, and the difference is one character that yields fluent WRONG
/// output rather than an error (`crates/gpu` Gotcha 12).
///
/// `ple` is active, which no other family here has at all. It is 30.8% of the
/// checkpoint's bytes and streams from its own table.
///
/// **The indexer fields are all dispatched on** (`families/qwen4/attn.rs`,
/// since 2026-09-05): heads and dim size `index_qk_proj` and the per-head
/// norms, `csa_compress_rate` is the pooling block, and `index_top_k` is the
/// number of BLOCKS kept (this architecture's unit; DeepSeek's counts
/// tokens). The reference's indexer returns early at `kv_len <= budget`, so
/// at or below `index_budget` selection keeps every block and attention is
/// exactly plain causal -- `select_blocks` reproduces that by construction.
///
/// `ple.seed` is the one value NOT in the file. `config.json` carries no
/// `seed` key and the reference defaults it to 1234, which is what the hash
/// multipliers derive from when a checkpoint omits its `layer_multipliers`
/// buffer. Both published checkpoints DO ship that buffer, so the seed is a
/// cross-check rather than a source of truth -- but recording it as the
/// format's own default is what AGENTS.md Gotcha 39 requires of a value
/// standing in for an absent key.
///
/// `vision` is `NONE` because this port ingests the TEXT tower only, as it
/// already does for `qwen3_5` and `muse_glimmer`, even though the checkpoint
/// declares a `vision_config` and `language_model_only: false`.
pub fn qwen4_exp_125b_a6b() -> ArchConfig {
    ArchConfig {
        hidden_size: 2560,
        // The SHARED expert's width, matching Qwen 3.6's use of this field.
        // This architecture happens to give the shared and routed experts the
        // same 640, so the two read alike here and mean different things.
        intermediate_size: 640,
        moe_intermediate_size: 640,
        num_heads: 24,
        num_kv_heads: 2,
        num_full_kv_heads: 2,
        head_dim: 256,
        full_head_dim: 256,
        vocab_size: 248_320,
        sliding_window: 0,
        final_logit_softcap: 0.0,
        rope_theta: 10_000_000.0,
        full_rope_theta: 10_000_000.0,
        partial_rotary_factor: 0.25,
        num_layers: 48,
        num_experts: 512,
        top_k_experts: 10,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: qwen_hybrid_layer_mask(48),
        hidden_activation: "silu".to_string(),
        family: ModelFamily::Qwen4Exp,
        attn_output_gate: true,
        attention_scale: 0.0625, // 256^-0.5
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: true,
        rope_neox_subdim: true,
        linear_attention: LinearAttentionConfig {
            num_k_heads: 16,
            num_v_heads: 48,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel_size: 4,
            output_gate_sigmoid: true,
        },
        compressed_attention: CompressedAttentionConfig {
            index_n_heads: 4,
            index_kv_heads: 1,
            index_head_dim: 128,
            // Blocks, not tokens: `indexer_budget / indexer_compress_ratio`.
            index_top_k: 512,
            index_budget: 2048,
            csa_compress_rate: 4,
            // The rest are DeepSeek's MLA terms and this architecture has
            // none of them. Its full layers are ordinary GQA with a selector
            // in front, not a compressed-KV attention.
            q_lora_rank: 0,
            o_lora_rank: 0,
            o_groups: 0,
            rope_head_dim: 0,
            hca_compress_rate: 0,
            compress_rope_theta: 0.0,
            rope_scaling_factor: 0.0,
            rope_scaling_original_max: 0,
            rope_scaling_beta_fast: 0.0,
            rope_scaling_beta_slow: 0.0,
        },
        hyper_connections: HyperConnectionConfig {
            mult: 4,
            lowrank: 320,
            // DeepSeek mHC's terms; this mixer is not Sinkhorn-normalised.
            sinkhorn_iters: 0,
            eps: 0.0,
        },
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: RopeScalingConfig::NONE,
        vision: VisionConfig::NONE,
        ple: PleConfig {
            ngram_size: 3,
            heads_per_ngram: 8,
            ngram_vocab_size_base: 20_000_000,
            make_divisible_by: 128,
            split_ngram_parts: 128,
            ple_embed_dim: 2560,
            conv_kernel_size: 4,
            // ONE-BASED, exactly as `config.json` spells it. Layer INDEX 1.
            layer_ids: vec![2],
            // NOT in the file; the reference's default. See the doc above.
            seed: 1234,
            // `text_config.eos_token_id`, `docs/QWEN4_PHASE0.md` item 4.
            eos_token_id: 248_044,
        },
    }
}
