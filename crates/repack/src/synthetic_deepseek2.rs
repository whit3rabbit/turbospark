//! Builds a tiny `deepseek2` install through the REAL GGUF walk.
//!
//! GGUF is this family's only intake, so like `synthetic_spark` this
//! fixture constructs a small `GgufBuilder` file shaped like the real
//! `mradermacher/DeepSeek-V2-Lite-Chat-GGUF` header (MLA projections,
//! separate gate/up routed experts, a fused shared expert, one dense lead
//! layer) and pushes it through `write_gguf_install_streamed` exactly as a
//! real `pull` would. The derived `ArchConfig` is the walk's own output,
//! which is also why the returned arch is what a test must pass to
//! `RealForwardRunner::open`.
//!
//! Shape notes: `KV_LORA + ROPE_DIM` and every weight tensor's leading
//! (fastest-varying) dim are multiples of 32 so Q8_0 blocks never straddle
//! rows; the dense lead width (96) deliberately differs from the fused
//! shared expert's width (64), because those two being different numbers is
//! the whole point of `dense_lead_intermediate_size`.

use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig, MlaConfig,
    ModelFamily, PleConfig, VisionConfig,
};

use crate::gguf_header::{parse_header as parse_gguf_header, DEFAULT_MAX_HEADER_BYTES};
use crate::ranged_download::MemoryRangeSource;
use crate::synthetic_gguf::GgufBuilder;
use crate::write_gguf_install_streamed;

const HIDDEN: usize = 128;
const NUM_HEADS: usize = 4;
/// The latent rank: the compressed cache row is `KV_LORA + ROPE_DIM = 80`
/// halves wide.
const KV_LORA: usize = 64;
/// The rope-carried tail: 8 pairs, even, and the per-head key width is
/// `NOPE + ROPE_DIM = 48`.
const ROPE_DIM: usize = 16;
const NOPE: usize = 32;
/// The per-head value width the v-combine produces.
const V_DIM: usize = 32;
const MOE_INTER: usize = 32;
const EXPERTS: usize = 8;
const TOP_K: usize = 2;
/// The shared expert count, FUSED into one SwiGLU of `MOE_INTER * SHARED`
/// = 64 wide -- deliberately DIFFERENT from the dense lead's 96 below.
const SHARED: usize = 2;
/// The dense lead layer's FFN width.
const DENSE_INTER: usize = 96;

/// A tiny `deepseek2`-shaped architecture. Every non-shape field takes
/// [`model_io::deepseek_v2_lite_16b`]'s own value, because the GGUF walk
/// derives behavioural fields from `known_architecture(Deepseek2)` and the
/// fixture must agree with what the walk produces -- the assert in
/// [`build_synthetic_deepseek2_install`] is that agreement, checked field
/// by field.
pub fn tiny_deepseek2_arch(vocab_size: i64, num_layers: i64) -> ArchConfig {
    let mut mask = vec![5u8; num_layers as usize];
    let _ = &mut mask;
    ArchConfig {
        hidden_size: HIDDEN as i64,
        // The fused shared expert: MOE_INTER * SHARED.
        intermediate_size: (MOE_INTER * SHARED) as i64,
        moe_intermediate_size: MOE_INTER as i64,
        num_heads: NUM_HEADS as i64,
        // The compressed cache is ONE shared row per layer; the walk
        // hard-sets this for the family regardless of `head_count_kv`.
        num_kv_heads: 1,
        num_full_kv_heads: 1,
        head_dim: (NOPE + ROPE_DIM) as i64,
        full_head_dim: V_DIM as i64,
        vocab_size,
        sliding_window: 0,
        final_logit_softcap: 0.0,
        rope_theta: 10_000.0,
        full_rope_theta: 10_000.0,
        partial_rotary_factor: 0.0,
        num_layers,
        dense_lead_intermediate_size: DENSE_INTER as i64,
        num_dense_leading_layers: 1,
        num_experts: EXPERTS as i64,
        top_k_experts: TOP_K as i64,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: mask,
        hidden_activation: "silu".to_string(),
        family: ModelFamily::Deepseek2,
        attn_output_gate: false,
        // THE BASELINE'S VALUE, NOT A TOY ONE: attention scale is not in
        // any GGUF key, so the walk takes
        // `known_architecture(Deepseek2).attention_scale` and this field
        // must equal it or the builder's assert reddens. It round-trips
        // through the manifest exactly (pinned beside the baseline), which
        // is what lets the synthetic open at all.
        attention_scale: model_io::deepseek_v2_lite_16b().attention_scale,
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: false,
        rope_neox_subdim: true,
        linear_attention: LinearAttentionConfig::NONE,
        mla: MlaConfig {
            kv_lora_rank: KV_LORA as i64,
            q_lora_rank: 0,
            nope_head_dim: NOPE as i64,
            rope_head_dim: ROPE_DIM as i64,
            v_head_dim: V_DIM as i64,
        },
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        // THE BASELINE'S YaRN, not NONE: `arch_from_gguf` starts from
        // `known_architecture(Deepseek2)` and only OVERWRITES rope scaling
        // when the file declares it, so a fixture without yarn keys derives
        // the baseline's. The synthetic flow therefore exercises the same
        // yarn rope the real checkpoint does.
        rope_scaling: model_io::deepseek_v2_lite_16b().rope_scaling,
        vision: VisionConfig::NONE,
        ple: PleConfig::NONE,
    }
}

/// Writes a tiny `deepseek2` `.gturbo` install via the REAL GGUF walk and
/// returns the derived `ArchConfig`. `num_layers` must be at least 2: layer
/// 0 is the dense lead and at least one MoE layer must follow it.
pub fn build_synthetic_deepseek2_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    assert!(num_layers >= 2, "a dense lead plus one MoE layer minimum");
    let vocab = vocab_size as usize;
    let heads = NUM_HEADS as u64;
    let hidden = HIDDEN as u64;
    let q_out = (NUM_HEADS * (NOPE + ROPE_DIM)) as u64;
    let kv_a_out = (KV_LORA + ROPE_DIM) as u64;
    let kv_b_out = (NUM_HEADS * (NOPE + V_DIM)) as u64;
    let shexp = (MOE_INTER * SHARED) as u64;

    let mut b = GgufBuilder::new()
        .metadata_str("general.architecture", "deepseek2")
        .metadata_u32("deepseek2.block_count", num_layers as u32)
        .metadata_u32("deepseek2.embedding_length", hidden as u32)
        // The dense lead's width; the shexp tensors carry their own.
        .metadata_u32("deepseek2.feed_forward_length", DENSE_INTER as u32)
        .metadata_u32("deepseek2.leading_dense_block_count", 1)
        .metadata_u32("deepseek2.attention.head_count", heads as u32)
        // The real file says 16; the walk overrides it to the compressed
        // cache's 1 for this family, so the fixture records the expanded
        // form's answer to prove the override fires.
        .metadata_u32("deepseek2.attention.head_count_kv", 4)
        .metadata_u32("deepseek2.attention.key_length", (NOPE + ROPE_DIM) as u32)
        .metadata_u32("deepseek2.attention.value_length", V_DIM as u32)
        .metadata_u32("deepseek2.attention.kv_lora_rank", KV_LORA as u32)
        .metadata_f32("deepseek2.attention.layer_norm_rms_epsilon", 1e-6)
        .metadata_u32("deepseek2.rope.dimension_count", ROPE_DIM as u32)
        .metadata_f32("deepseek2.rope.freq_base", 10_000.0)
        .metadata_u32("deepseek2.expert_count", EXPERTS as u32)
        .metadata_u32("deepseek2.expert_used_count", TOP_K as u32)
        .metadata_u32("deepseek2.expert_feed_forward_length", MOE_INTER as u32)
        .metadata_u32("deepseek2.expert_shared_count", SHARED as u32)
        .metadata_f32("deepseek2.expert_weights_scale", 1.0)
        // The vocabulary comes off the embedding tensor's own dims.
        .q8_0_tensor("token_embd.weight", &[hidden, vocab as u64], 1);

    for l in 0..num_layers as usize {
        let p = format!("blk.{l}");
        let base = (l as u8).wrapping_mul(16);
        let slot = |i: u8| base.wrapping_add(i);
        b = b
            .f32_tensor(&format!("{p}.attn_norm.weight"), &[hidden], slot(1))
            .q8_0_tensor(&format!("{p}.attn_q.weight"), &[hidden, q_out], slot(2))
            .q8_0_tensor(
                &format!("{p}.attn_kv_a_mqa.weight"),
                &[hidden, kv_a_out],
                slot(3),
            )
            .f32_tensor(
                &format!("{p}.attn_kv_a_norm.weight"),
                &[KV_LORA as u64],
                slot(4),
            )
            .q8_0_tensor(
                &format!("{p}.attn_kv_b.weight"),
                &[KV_LORA as u64, kv_b_out],
                slot(5),
            )
            .q8_0_tensor(
                &format!("{p}.attn_output.weight"),
                &[(NUM_HEADS * V_DIM) as u64, hidden],
                slot(6),
            )
            .f32_tensor(&format!("{p}.ffn_norm.weight"), &[hidden], slot(7));
        if l == 0 {
            // The dense lead: a plain SwiGLU at DENSE_INTER.
            b = b
                .q8_0_tensor(
                    &format!("{p}.ffn_gate.weight"),
                    &[hidden, DENSE_INTER as u64],
                    slot(8),
                )
                .q8_0_tensor(
                    &format!("{p}.ffn_up.weight"),
                    &[hidden, DENSE_INTER as u64],
                    slot(9),
                )
                .q8_0_tensor(
                    &format!("{p}.ffn_down.weight"),
                    &[DENSE_INTER as u64, hidden],
                    slot(10),
                );
        } else {
            b = b
                .f32_tensor(
                    &format!("{p}.ffn_gate_inp.weight"),
                    &[hidden, EXPERTS as u64],
                    slot(8),
                )
                .q8_0_tensor(
                    &format!("{p}.ffn_gate_exps.weight"),
                    &[hidden, MOE_INTER as u64, EXPERTS as u64],
                    slot(9),
                )
                .q8_0_tensor(
                    &format!("{p}.ffn_up_exps.weight"),
                    &[hidden, MOE_INTER as u64, EXPERTS as u64],
                    slot(10),
                )
                .q8_0_tensor(
                    &format!("{p}.ffn_down_exps.weight"),
                    &[MOE_INTER as u64, hidden, EXPERTS as u64],
                    slot(11),
                )
                .q8_0_tensor(
                    &format!("{p}.ffn_gate_shexp.weight"),
                    &[hidden, shexp],
                    slot(12),
                )
                .q8_0_tensor(
                    &format!("{p}.ffn_up_shexp.weight"),
                    &[hidden, shexp],
                    slot(13),
                )
                .q8_0_tensor(
                    &format!("{p}.ffn_down_shexp.weight"),
                    &[shexp, hidden],
                    slot(14),
                );
        }
    }
    let b = b
        .f32_tensor("output_norm.weight", &[hidden], 15)
        // Untied head, exactly as the real file ships.
        .q8_0_tensor("output.weight", &[hidden, vocab as u64], 16);

    let (bytes, _) = b.build();
    let header = parse_gguf_header(&bytes, DEFAULT_MAX_HEADER_BYTES)?;
    let arch = write_gguf_install_streamed(
        dir,
        &header,
        &MemoryRangeSource::new(&bytes),
        model_id,
        |_| {},
    )?;
    // The walk derives the config from the file; a mismatch against the
    // intended tiny arch means the fixture's metadata is wrong, not a
    // tolerance to paper over.
    let expected = tiny_deepseek2_arch(vocab_size, num_layers);
    assert_eq!(arch, expected, "the walk derived a different arch");
    Ok(arch)
}
