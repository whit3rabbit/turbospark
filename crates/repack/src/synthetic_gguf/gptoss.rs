//! A tiny but STRUCTURALLY REAL `gpt-oss` GGUF (ROADMAP M5).
//!
//! [`super::gemma4::SyntheticGgufShape::mxfp4`] already builds a fixture with
//! gpt-oss's block types, and that is all it builds: a GEMMA-shaped install
//! that proves the MXFP4 routed pair dispatches inside a forward pass. It
//! carries none of the four things that make `gpt-oss` a fifth flow rather
//! than a sixth family on an existing one, so it cannot answer a single
//! question about the layer.
//!
//! This one carries all four, and the reason to build it BEFORE the 12.1 GB
//! stream is `crates/repack/CLAUDE.md` Gotcha 8: on the dense `llama` half,
//! every hole the real file exposed was in the SHAPE of the model rather than
//! in the block format, and each one cost a five-minute re-stream because no
//! fixture had asked first. The four shape questions here are
//!
//! - **rank-2 routed tensors.** The per-expert biases are `[width, experts]`,
//!   where every routed tensor the walk has ever seen is rank 3. `plan.rs`
//!   refuses a non-rank-3 routed source by name, so this is the first thing
//!   that breaks and it breaks loudly.
//! - **six routed roles per layer**, not three, and the three new ones sort
//!   outside `plan.rs`'s `["gate", "up", "down"]` order.
//! - **a bias beside every projection**, which no family here has had, so the
//!   resident transcode meets F32 vectors it has to narrow like a norm.
//! - **an untied head next to a `token_embd` of the same shape**, which is
//!   Gemma's arrangement inverted.
//!
//! GATE AND UP ARE NOT FUSED, unlike Gemma's `ffn_gate_up_exps`. That is the
//! real file's shape and it is why this fixture cannot be a `QuantMix` on the
//! Gemma builder: the tensor inventory differs, not just the types in it.
//!
//! THE FIXTURE'S GATE/UP AND DOWN BIASES HAVE DIFFERENT LENGTHS ON PURPOSE.
//! On the real 20b they are both `[2880, 32]`, because `hidden` and the expert
//! width are both 2880 there, so the file cannot distinguish "the bias of a
//! projection that outputs `n_ff`" from "one that outputs `n_embd`". Here they
//! are 32 and 64, so a walk that took the wrong one is a size mismatch rather
//! than a silent transposition -- the same reason `SyntheticGgufShape` keeps
//! `head_dim` and `full_head_dim` apart.

use super::builder::{GgufBuilder, GgufFileAndRanges};

/// Shape knobs for [`build_synthetic_gpt_oss_gguf`].
///
/// Deliberately NOT a `QuantMix` variant on [`super::gemma4::SyntheticGgufShape`]:
/// that type's fields describe Gemma's tensor inventory (a fused gate/up, a
/// per-expert router scale, a `layer_output_scale`), and `gpt-oss` has a
/// different one rather than the same one quantized differently.
#[derive(Debug, Clone, Copy)]
pub struct SyntheticGptOssShape {
    pub num_layers: usize,
    pub hidden: u64,
    pub num_heads: u64,
    pub head_dim: u64,
    pub num_kv_heads: u64,
    /// Routed expert width. A whole number of MXFP4 blocks (32) or the
    /// packing is a partial block, which ggml never emits.
    pub moe_intermediate: u64,
    pub num_experts: u64,
    pub top_k: u64,
    pub vocab: u64,
    pub sliding_window: u64,
}

impl Default for SyntheticGptOssShape {
    /// TWO LAYERS, because the window alternates and one layer cannot show
    /// an alternation. Layer 0 slides and layer 1 does not, which is the real
    /// model's polarity (`is_swa[il] = (il % 2) < 1`, so EVEN layers slide)
    /// and the half of it a fixture can actually pin: a mask built with the
    /// phase inverted produces a different `full_attention_layer_mask` here
    /// and an install that is wrong only past 128 tokens on the real file.
    ///
    /// `moe_intermediate` is 32, one MXFP4 block, and `hidden` is 64, two.
    /// Keeping them UNEQUAL is the point of the pair -- the real 20b has
    /// `hidden == moe_intermediate == 2880`, so it cannot tell a gate/up bias
    /// from a down bias, and neither could a fixture that copied its ratio.
    fn default() -> Self {
        Self {
            num_layers: 2,
            hidden: 64,
            num_heads: 8,
            head_dim: 16,
            num_kv_heads: 2,
            moe_intermediate: 32,
            num_experts: 4,
            top_k: 2,
            vocab: 128,
            sliding_window: 8,
        }
    }
}

impl SyntheticGptOssShape {
    fn q_dim(&self) -> u64 {
        self.num_heads * self.head_dim
    }

    fn kv_dim(&self) -> u64 {
        self.num_kv_heads * self.head_dim
    }
}

/// A complete, tiny `gpt-oss` GGUF: every tensor name and metadata key the
/// real `gpt-oss-20b-MXFP4.gguf` carries, at toy dimensions.
///
/// The inventory is `gguf_names.rs::map_gpt_oss_layer`'s domain exactly, which
/// is the property that makes this fixture worth having: a name the table maps
/// and the fixture omits is untested, and a name the fixture emits and the
/// table does not map fails the walk here rather than 20 minutes into a
/// stream.
pub fn build_synthetic_gpt_oss_gguf(shape: SyntheticGptOssShape) -> GgufFileAndRanges {
    let s = shape;
    let mut b = GgufBuilder::new()
        .metadata_str("general.architecture", "gpt-oss")
        .metadata_u32("gpt-oss.block_count", s.num_layers as u32)
        .metadata_u32("gpt-oss.embedding_length", s.hidden as u32)
        .metadata_u32("gpt-oss.attention.head_count", s.num_heads as u32)
        .metadata_u32("gpt-oss.attention.head_count_kv", s.num_kv_heads as u32)
        .metadata_u32("gpt-oss.attention.key_length", s.head_dim as u32)
        .metadata_u32("gpt-oss.attention.value_length", s.head_dim as u32)
        .metadata_u32("gpt-oss.expert_count", s.num_experts as u32)
        .metadata_u32("gpt-oss.expert_used_count", s.top_k as u32)
        .metadata_u32("gpt-oss.feed_forward_length", s.moe_intermediate as u32)
        .metadata_u32(
            "gpt-oss.expert_feed_forward_length",
            s.moe_intermediate as u32,
        )
        // NO `attention.sliding_window_pattern`, exactly as the real file
        // omits it, so `gpt_oss_layer_mask`'s period-2 default is what runs.
        // Publishing one here would test the fixture instead of the loader.
        .metadata_u32("gpt-oss.attention.sliding_window", s.sliding_window as u32)
        .metadata_f32("gpt-oss.attention.layer_norm_rms_epsilon", 1e-5)
        .metadata_f32("gpt-oss.rope.freq_base", 150_000.0)
        // YaRN, and the TYPE key is what arms it: `rope.scaling.factor` alone
        // also spells linear scaling, and reading it as YaRN would apply a
        // ramp that is finite, fluent and wrong.
        .metadata_str("gpt-oss.rope.scaling.type", "yarn")
        .metadata_f32("gpt-oss.rope.scaling.factor", 32.0)
        .metadata_u32("gpt-oss.rope.scaling.original_context_length", 4096)
        .metadata_f32("gpt-oss.rope.scaling.yarn_beta_fast", 32.0)
        .metadata_f32("gpt-oss.rope.scaling.yarn_beta_slow", 1.0)
        .q8_0_tensor("token_embd.weight", &[s.hidden, s.vocab], 1)
        // UNTIED, and the walk reads exactly this tensor's presence to decide
        // (`tie_word_embeddings = !tensors.contains_key("output.weight")`).
        .q8_0_tensor("output.weight", &[s.hidden, s.vocab], 2)
        .f32_upcast_bf16_tensor("output_norm.weight", &[s.hidden], 3);

    for l in 0..s.num_layers {
        let seed = (l as u8).wrapping_mul(41).wrapping_add(5);
        let n = |tail: &str| format!("blk.{l}.{tail}");

        b = b
            .f32_upcast_bf16_tensor(&n("attn_norm.weight"), &[s.hidden], seed)
            .f32_upcast_bf16_tensor(&n("post_attention_norm.weight"), &[s.hidden], seed + 1)
            .q8_0_tensor(&n("attn_q.weight"), &[s.hidden, s.q_dim()], seed + 2)
            .q8_0_tensor(&n("attn_k.weight"), &[s.hidden, s.kv_dim()], seed + 3)
            .q8_0_tensor(&n("attn_v.weight"), &[s.hidden, s.kv_dim()], seed + 4)
            .q8_0_tensor(&n("attn_output.weight"), &[s.q_dim(), s.hidden], seed + 5);

        // THE BIASES, F32 in the file as every one of gpt-oss's is. Upcast
        // BF16 values rather than arbitrary F32, so `transcode_f32` narrows
        // them back bit-exactly and a fixture install carries no lossy
        // warnings it did not mean to (AGENTS.md Gotcha 29's measured
        // property, reproduced here so the count is a signal).
        b = b
            .f32_upcast_bf16_tensor(&n("attn_q.bias"), &[s.q_dim()], seed + 6)
            .f32_upcast_bf16_tensor(&n("attn_k.bias"), &[s.kv_dim()], seed + 7)
            .f32_upcast_bf16_tensor(&n("attn_v.bias"), &[s.kv_dim()], seed + 8)
            .f32_upcast_bf16_tensor(&n("attn_output.bias"), &[s.hidden], seed + 9)
            // One learned logit per QUERY head, not per KV head. Getting that
            // wrong is a length mismatch here and a wrong softmax denominator
            // on the real file.
            .f32_upcast_bf16_tensor(&n("attn_sinks.weight"), &[s.num_heads], seed + 10);

        // The router: F32 weight (the repack INT8s it) plus an F32 bias that
        // is added to the logits BEFORE the top-k.
        b = b
            .f32_tensor(
                &n("ffn_gate_inp.weight"),
                &[s.hidden, s.num_experts],
                seed + 11,
            )
            .f32_upcast_bf16_tensor(&n("ffn_gate_inp.bias"), &[s.num_experts], seed + 12);

        // Routed experts, NOT fused, all three MXFP4.
        b = b
            .mxfp4_tensor(
                &n("ffn_gate_exps.weight"),
                &[s.hidden, s.moe_intermediate, s.num_experts],
                seed + 13,
            )
            .mxfp4_tensor(
                &n("ffn_up_exps.weight"),
                &[s.hidden, s.moe_intermediate, s.num_experts],
                seed + 14,
            )
            .mxfp4_tensor(
                &n("ffn_down_exps.weight"),
                &[s.moe_intermediate, s.hidden, s.num_experts],
                seed + 15,
            );

        // The per-expert biases: RANK 2, and the whole reason this fixture
        // exists. Gate and up are the expert width; down is `hidden`, because
        // it is the projection back to the residual stream.
        b = b
            .f32_tensor(
                &n("ffn_gate_exps.bias"),
                &[s.moe_intermediate, s.num_experts],
                seed + 16,
            )
            .f32_tensor(
                &n("ffn_up_exps.bias"),
                &[s.moe_intermediate, s.num_experts],
                seed + 17,
            )
            .f32_tensor(
                &n("ffn_down_exps.bias"),
                &[s.hidden, s.num_experts],
                seed + 18,
            );
    }

    b.build()
}
