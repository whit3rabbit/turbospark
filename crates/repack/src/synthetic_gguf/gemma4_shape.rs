//! Synthetic GGUF shape and quantization mixture specifications for Gemma 4 test fixtures.

/// Shape knobs for [`super::gemma4::build_synthetic_gemma4_gguf`]. Tiny by design, but
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
    /// What a real `Q3_K_M` does: the attention projections and the dense
    /// FFN at Q3_K, the embedding table and the routed experts at Q4_K, the
    /// rest Q8_0 -- the pinned Qwen2.5 Q3_K_M's census over a Gemma-shaped
    /// carrier, since a dense fixture has no routed slot to show the mixture
    /// on. Q3_K has a resident GEMV and an embedding lookup and NO routed
    /// pair, so its experts deliberately stay on a type with one.
    /// Needs [`SyntheticGgufShape::k_quant`]'s dimensions, since a Q3_K row
    /// also tiles 256 elements.
    Q3K,
    /// A llama.cpp legacy-quant file (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`): the
    /// embedding table, every attention projection and the dense FFN on that
    /// one type, the routed experts on Q8_0. The legacy blocks have a resident
    /// GEMV and an embedding lookup and NO routed pair, so their experts
    /// deliberately stay on a type with one. Tiles 32 elements, so the
    /// default dimensions fit. The payload is the ggml type id (2, 3, 6, 7).
    Legacy(u32),
    /// The oracle twin of [`QuantMix::Legacy`]: the same weights, decoded and
    /// stored as Q8_0, so the Q8_0 kernels compute what the legacy kernels
    /// must. Compare the two installs' logits.
    LegacyAsQ8_0(u32),
    /// A file whose ROUTED EXPERTS are a block type from the resident-only
    /// list above (ggml ids 2, 3, 6, 7, 34, 35, 39, 40, 41, 42) and everything
    /// else is Q8_0: what a Mixtral Q4_0 conversion looks like. These types
    /// have routed kernels now, so this installs and runs.
    LegacyExperts(u32),
    /// The oracle twin of [`QuantMix::LegacyExperts`]: the same expert weights
    /// decoded and stored as Q8_0.
    LegacyExpertsAsQ8_0(u32),
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

    /// The Q3_K mixture, at [`k_quant`]'s dimensions: Q3_K is a 256-element
    /// superblock type like Q4_K and Q6_K, so it needs exactly the same
    /// widening.
    pub fn q3_k() -> Self {
        Self {
            mix: QuantMix::Q3K,
            ..Self::k_quant()
        }
    }

    /// A legacy-quant mixture: ggml type 2 (Q4_0), 3 (Q4_1), 6 (Q5_0) or
    /// 7 (Q5_1) at the default dimensions, or 34, 35, 40, 41 (TQ1_0, TQ2_0,
    /// NVFP4, Q1_0) at the 256-wide ones.
    pub fn legacy(ggml_type: u32) -> Self {
        assert!(
            matches!(ggml_type, 2 | 3 | 6 | 7 | 34 | 35 | 39 | 40 | 41 | 42 | 64 | 65 | 66),
            "not a legacy ggml type: {ggml_type}"
        );
        // Q1_0 (41), TQ1_0 (34), TQ2_0 (35) and NVFP4 (40) are resident-only
        // blocks like the legacy four, but their blocks are 64 to 256
        // elements, so they take the 256-wide dimensions.
        // MXFP4 (39) and Q2_0 (42) tile 32 and 64 elements, so they take the
        // 256-wide dimensions too (every row stays a whole number of blocks).
        if matches!(ggml_type, 34 | 35 | 39 | 40 | 41 | 42 | 64 | 65 | 66) {
            return Self {
                mix: QuantMix::Legacy(ggml_type),
                ..Self::k_quant()
            };
        }
        // `moe_intermediate` 32: the Q8_0 routed experts need rows that tile
        // 32 elements, as `mxfp4()` does.
        Self {
            moe_intermediate: 32,
            mix: QuantMix::Legacy(ggml_type),
            ..Self::default()
        }
    }

    /// Legacy-typed routed experts over Q8_0 everything-else; must be refused.
    pub fn legacy_experts(ggml_type: u32) -> Self {
        Self {
            mix: QuantMix::LegacyExperts(ggml_type),
            ..Self::legacy(ggml_type)
        }
    }

    /// The Q8_0 oracle for [`Self::legacy_experts`].
    pub fn legacy_experts_as_q8_0(ggml_type: u32) -> Self {
        Self {
            mix: QuantMix::LegacyExpertsAsQ8_0(ggml_type),
            ..Self::legacy(ggml_type)
        }
    }

    /// The Q8_0 oracle for [`Self::legacy`].
    pub fn legacy_as_q8_0(ggml_type: u32) -> Self {
        Self {
            mix: QuantMix::LegacyAsQ8_0(ggml_type),
            ..Self::legacy(ggml_type)
        }
    }

    /// `true` on the one layer whose routed experts differ from the rest.
    /// Only [`QuantMix::Iq`] has one.
    pub(crate) fn odd_expert_layer(&self, layer: usize) -> bool {
        self.mix == QuantMix::Iq && layer + 1 == self.num_layers
    }

    /// `true` where the layer slides, matching GGUF's own polarity.
    pub(crate) fn slides(&self, layer: usize) -> bool {
        // Last layer global, the rest sliding: enough to produce both kinds
        // without reproducing Gemma's every-sixth-layer pattern.
        layer + 1 != self.num_layers
    }
}
