use super::builder::{GgufBuilder, GgufFileAndRanges};
use crate::gguf_header::GgufValue;

/// Shape knobs for [`build_synthetic_gemma4_gguf`]. Tiny by design, but
/// every dimension is chosen so each tensor's element count is a whole
/// number of 32-element Q8_0 blocks, PER EXPERT as well as in total, which
/// is the constraint a naive shrink of a real config violates first.
#[derive(Debug, Clone, Copy)]
pub struct SyntheticGgufShape {
    pub num_layers: usize,
    pub hidden: u64,
    pub num_heads: u64,
    pub head_dim: u64,
    pub full_head_dim: u64,
    pub num_kv_heads: u64,
    pub num_full_kv_heads: u64,
    pub intermediate: u64,
    pub moe_intermediate: u64,
    pub num_experts: u64,
    pub top_k: u64,
    pub vocab: u64,
    pub sliding_window: u64,
    /// Which block types the fixture puts where.
    pub mix: QuantMix,
}

/// Which mixture of block types a fixture carries.
///
/// A three-way enum rather than a pair of flags because the mixtures are
/// mutually exclusive by construction and two bools would have an invalid
/// fourth state.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum QuantMix {
    /// Q8_0 throughout, as Gemma 4's published GGUF is.
    #[default]
    Q8_0,
    /// What a real `Q4_K_M` does: routed experts and the embedding table at
    /// Q4_K, the attention projections at Q6_K, everything else Q8_0. Needs
    /// [`SyntheticGgufShape::k_quant`]'s dimensions, since every K-quant row
    /// has to tile 256 elements where Q8_0 needs 32.
    KQuant,
    /// What ROADMAP Phase S's candidate does: IQ3_XXS routed gate/up over
    /// IQ4_NL routed down, a Q6_K embedding table with the head tied to it,
    /// Q8_0 attention - and a LAST LAYER that differs from all the others
    /// (IQ4_XS gate/up over Q8_0 down), which is the property no fixture
    /// before this one had. Needs [`SyntheticGgufShape::iq_mixed`]'s
    /// dimensions.
    Iq,
    /// What ROADMAP M5's `gpt-oss` does with its block types: MXFP4 routed
    /// experts over a Q8_0 everything-else. The first mixture whose expert
    /// type has NO resident GEMV, which is the property this fixture exists
    /// to exercise -- the manifest gate must let it through on the routed
    /// slot while the resident dtype backstop would refuse the same type on
    /// an attention tensor. Needs [`SyntheticGgufShape::mxfp4`]'s dimensions.
    ///
    /// It carries none of gpt-oss's FLOW differences (biases, sinks, YaRN,
    /// the clamped SwiGLU): this is a Gemma-shaped install with gpt-oss's
    /// block types, exactly as [`QuantMix::Iq`] is a Gemma-shaped install
    /// with the Phase S candidate's. What it proves is that the MXFP4 pair
    /// dispatches and decodes inside a whole forward pass.
    Mxfp4,
}

impl Default for SyntheticGgufShape {
    /// Two layers, one sliding-window and one global, so the per-layer
    /// `head_count_kv` array and the `sliding_window_pattern` inversion are
    /// both exercised by the smallest possible fixture.
    fn default() -> Self {
        Self {
            num_layers: 2,
            hidden: 64,
            num_heads: 4,
            head_dim: 16,
            full_head_dim: 32,
            num_kv_heads: 2,
            num_full_kv_heads: 1,
            intermediate: 32,
            moe_intermediate: 16,
            num_experts: 4,
            top_k: 2,
            vocab: 128,
            sliding_window: 8,
            mix: QuantMix::Q8_0,
        }
    }
}

impl SyntheticGgufShape {
    /// The smallest shape a K-quant fixture can take: every dimension that
    /// becomes a Q4_K or Q6_K ROW is 256, because a superblock is 256
    /// elements and ggml never emits a partial one. That is `hidden` (the
    /// embedding, the experts' gate/up, and every attention projection),
    /// `moe_intermediate` (the experts' down), and `num_heads * head_dim`
    /// (attn_output). Shrinking any of them is what breaks first.
    pub fn k_quant() -> Self {
        Self {
            hidden: 256,
            num_heads: 8,
            head_dim: 32,
            full_head_dim: 32,
            intermediate: 256,
            moe_intermediate: 256,
            vocab: 256,
            mix: QuantMix::KQuant,
            ..Self::default()
        }
    }

    /// The Phase S mixture, at the smallest dimensions its block types allow.
    ///
    /// `hidden` and `moe_intermediate` are both 256 because IQ3_XXS, IQ4_XS
    /// and Q6_K are all 256-element superblock types; IQ4_NL's 32 divides that
    /// anyway. `num_layers` is 3 rather than the default 2 so the odd LAST
    /// layer is genuinely a minority - with two layers, "the last one" and
    /// "half of them" are the same fixture and a plumbing bug that used layer
    /// 0's types everywhere would still look mixed.
    pub fn iq_mixed() -> Self {
        Self {
            num_layers: 3,
            hidden: 256,
            num_heads: 8,
            head_dim: 32,
            full_head_dim: 32,
            intermediate: 256,
            moe_intermediate: 256,
            vocab: 256,
            mix: QuantMix::Iq,
            ..Self::default()
        }
    }

    /// ROADMAP M5's mixture, at the smallest dimensions its block types
    /// allow.
    ///
    /// MXFP4's block is 32, so this needs NONE of the 256s the two mixtures
    /// above do -- the default shape's dimensions almost all qualify. The one
    /// that does not is `moe_intermediate`, which defaults to 16 and is a
    /// routed row length; the same widening `k_quant` needs for a different
    /// reason. Left at 32 rather than raised to 256 on purpose: a fixture
    /// whose every dimension is the largest block in the file cannot catch a
    /// kernel striding by the wrong one.
    pub fn mxfp4() -> Self {
        Self {
            moe_intermediate: 32,
            mix: QuantMix::Mxfp4,
            ..Self::default()
        }
    }

    /// `true` on the one layer whose routed experts differ from the rest.
    /// Only [`QuantMix::Iq`] has one.
    fn odd_expert_layer(&self, layer: usize) -> bool {
        self.mix == QuantMix::Iq && layer + 1 == self.num_layers
    }

    /// `true` where the layer slides, matching GGUF's own polarity.
    fn slides(&self, layer: usize) -> bool {
        // Last layer global, the rest sliding: enough to produce both kinds
        // without reproducing Gemma's every-sixth-layer pattern.
        layer + 1 != self.num_layers
    }
}

/// A complete, tiny Gemma 4 GGUF: every tensor name and metadata key the
/// real `gemma-4-26B-A4B-it-Q8_0.gguf` carries, at toy dimensions.
///
/// The name and metadata inventory is copied from the real file (see
/// `gguf_names.rs`), including the parts that are easy to get wrong: gate
/// and up FUSED into `ffn_gate_up_exps`, the router's per-expert scale
/// hiding under `ffn_down_exps.scale`, no `output.weight` because Gemma
/// ties its embeddings, and NO `attn_v` on global layers.
pub fn build_synthetic_gemma4_gguf(shape: SyntheticGgufShape) -> GgufFileAndRanges {
    let s = shape;
    let mut b = GgufBuilder::new()
        .metadata_str("general.architecture", "gemma4")
        .metadata_u32("gemma4.block_count", s.num_layers as u32)
        .metadata_u32("gemma4.embedding_length", s.hidden as u32)
        .metadata_u32("gemma4.attention.head_count", s.num_heads as u32)
        .metadata_u32("gemma4.expert_count", s.num_experts as u32)
        .metadata_u32("gemma4.expert_used_count", s.top_k as u32)
        .metadata_u32(
            "gemma4.expert_feed_forward_length",
            s.moe_intermediate as u32,
        )
        .metadata_u32("gemma4.feed_forward_length", s.intermediate as u32)
        .metadata_u32("gemma4.attention.key_length", s.full_head_dim as u32)
        .metadata_u32("gemma4.attention.key_length_swa", s.head_dim as u32)
        .metadata_u32("gemma4.attention.sliding_window", s.sliding_window as u32)
        .metadata_f32("gemma4.final_logit_softcapping", 30.0)
        .metadata_f32("gemma4.rope.freq_base", 1_000_000.0)
        .metadata_f32("gemma4.rope.freq_base_swa", 10_000.0)
        .metadata(
            "gemma4.attention.sliding_window_pattern",
            GgufValue::Array(
                (0..s.num_layers)
                    .map(|l| GgufValue::Bool(s.slides(l)))
                    .collect(),
            ),
        )
        .metadata(
            "gemma4.attention.head_count_kv",
            GgufValue::Array(
                (0..s.num_layers)
                    .map(|l| {
                        GgufValue::I32(if s.slides(l) {
                            s.num_kv_heads as i32
                        } else {
                            s.num_full_kv_heads as i32
                        })
                    })
                    .collect(),
            ),
        )
        .f32_upcast_bf16_tensor("output_norm.weight", &[s.hidden], 2);
    b = embed_tensor(b, &s, "token_embd.weight", &[s.hidden, s.vocab], 1);

    for l in 0..s.num_layers {
        let sliding = s.slides(l);
        let hd = if sliding { s.head_dim } else { s.full_head_dim };
        let kv = if sliding {
            s.num_kv_heads
        } else {
            s.num_full_kv_heads
        };
        let q_dim = s.num_heads * hd;
        let seed = (l as u8).wrapping_mul(37).wrapping_add(3);

        b = attn_tensor(
            b,
            &s,
            &format!("blk.{l}.attn_q.weight"),
            &[s.hidden, q_dim],
            seed,
        );
        b = attn_tensor(
            b,
            &s,
            &format!("blk.{l}.attn_k.weight"),
            &[s.hidden, kv * hd],
            seed.wrapping_add(1),
        );
        b = attn_tensor(
            b,
            &s,
            &format!("blk.{l}.attn_output.weight"),
            &[q_dim, s.hidden],
            seed.wrapping_add(2),
        );
        b = b
            .f32_upcast_bf16_tensor(&format!("blk.{l}.attn_q_norm.weight"), &[hd], seed)
            .f32_upcast_bf16_tensor(
                &format!("blk.{l}.attn_k_norm.weight"),
                &[hd],
                seed.wrapping_add(1),
            );

        // Global layers carry no V projection at all - a property of the
        // model, reproduced here so the walk is exercised against it.
        if sliding {
            b = attn_tensor(
                b,
                &s,
                &format!("blk.{l}.attn_v.weight"),
                &[s.hidden, kv * hd],
                seed.wrapping_add(3),
            );
        }

        for (n, norm) in [
            "attn_norm",
            "post_attention_norm",
            "ffn_norm",
            "pre_ffw_norm_2",
            "post_ffw_norm",
            "post_ffw_norm_1",
            "post_ffw_norm_2",
        ]
        .iter()
        .enumerate()
        {
            b = b.f32_upcast_bf16_tensor(
                &format!("blk.{l}.{norm}.weight"),
                &[s.hidden],
                seed.wrapping_add(10 + n as u8),
            );
        }

        b = b
            .q8_0_tensor(
                &format!("blk.{l}.ffn_gate.weight"),
                &[s.hidden, s.intermediate],
                seed.wrapping_add(4),
            )
            .q8_0_tensor(
                &format!("blk.{l}.ffn_up.weight"),
                &[s.hidden, s.intermediate],
                seed.wrapping_add(5),
            )
            .q8_0_tensor(
                &format!("blk.{l}.ffn_down.weight"),
                &[s.intermediate, s.hidden],
                seed.wrapping_add(6),
            )
            // Router: F32 in GGUF, where an MLX install carries INT8. Real
            // values rather than upcast BF16, because the repack quantizes
            // this one instead of narrowing it.
            .f32_tensor(
                &format!("blk.{l}.ffn_gate_inp.weight"),
                &[s.hidden, s.num_experts],
                seed.wrapping_add(20),
            )
            .f32_upcast_bf16_tensor(
                &format!("blk.{l}.ffn_gate_inp.scale"),
                &[s.hidden],
                seed.wrapping_add(21),
            )
            .f32_upcast_bf16_tensor(
                &format!("blk.{l}.ffn_down_exps.scale"),
                &[s.num_experts],
                seed.wrapping_add(22),
            )
            .f32_upcast_bf16_tensor(
                &format!("blk.{l}.layer_output_scale.weight"),
                &[1],
                seed.wrapping_add(23),
            );

        // Gate and up fused along the output dim.
        b = expert_tensor(
            b,
            &s,
            &format!("blk.{l}.ffn_gate_up_exps.weight"),
            &[s.hidden, 2 * s.moe_intermediate, s.num_experts],
            seed.wrapping_add(7),
            l,
            true,
        );
        b = expert_tensor(
            b,
            &s,
            &format!("blk.{l}.ffn_down_exps.weight"),
            &[s.moe_intermediate, s.hidden, s.num_experts],
            seed.wrapping_add(8),
            l,
            false,
        );
    }

    b.build()
}

/// The roles a mixed fixture moves off Q8_0, each mirroring where the real
/// file it models puts that block type. Written as functions rather than
/// methods on the builder because the choice belongs to the fixture's shape,
/// not to GGUF.
fn embed_tensor(
    b: GgufBuilder,
    s: &SyntheticGgufShape,
    name: &str,
    dims: &[u64],
    seed: u8,
) -> GgufBuilder {
    match s.mix {
        QuantMix::Q8_0 => b.q8_0_tensor(name, dims, seed),
        QuantMix::KQuant => b.q4_k_tensor(name, dims, seed),
        // The Phase S candidate keeps `token_embd` at Q6_K and ties the head
        // to it, which is what made `embed_lookup_q6_k` worth writing.
        QuantMix::Iq => b.q6_k_tensor(name, dims, seed),
        // gpt-oss keeps `token_embd` at Q8_0 and unties its head.
        QuantMix::Mxfp4 => b.q8_0_tensor(name, dims, seed),
    }
}

/// A routed expert tensor. Takes the LAYER and which half it is, because
/// [`QuantMix::Iq`] is the first mixture where those matter: its phases carry
/// different types from each other, and its last layer differs from the rest.
fn expert_tensor(
    b: GgufBuilder,
    s: &SyntheticGgufShape,
    name: &str,
    dims: &[u64],
    seed: u8,
    layer: usize,
    gate_up: bool,
) -> GgufBuilder {
    match (s.mix, gate_up, s.odd_expert_layer(layer)) {
        (QuantMix::Q8_0, _, _) => b.q8_0_tensor(name, dims, seed),
        (QuantMix::KQuant, _, _) => b.q4_k_tensor(name, dims, seed),
        (QuantMix::Iq, true, false) => b.iq_tensor(name, 18, dims, seed),
        (QuantMix::Iq, true, true) => b.iq_tensor(name, 23, dims, seed),
        (QuantMix::Iq, false, false) => b.iq_tensor(name, 20, dims, seed),
        (QuantMix::Iq, false, true) => b.q8_0_tensor(name, dims, seed),
        // Both phases, every layer: gpt-oss is uniform in a way the two
        // mixtures above deliberately are not.
        (QuantMix::Mxfp4, _, _) => b.mxfp4_tensor(name, dims, seed),
    }
}

/// Q6_K, where a real `Q4_K_M` would carry it on `output.weight`. A Gemma
/// fixture ties its embeddings and so has no such tensor, and the attention
/// projections are the next place a resident GEMV reads every token. The IQ
/// mixture leaves attention at Q8_0, as its candidate does.
fn attn_tensor(
    b: GgufBuilder,
    s: &SyntheticGgufShape,
    name: &str,
    dims: &[u64],
    seed: u8,
) -> GgufBuilder {
    match s.mix {
        QuantMix::KQuant => b.q6_k_tensor(name, dims, seed),
        QuantMix::Q8_0 | QuantMix::Iq | QuantMix::Mxfp4 => b.q8_0_tensor(name, dims, seed),
    }
}
