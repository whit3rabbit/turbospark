/// Model family discriminator. Selects the tensor-name contract, the layer
/// graph shape, and family-specific kernel behavior. Stored in
/// `manifest.json -> arch.family`; absent means Gemma 4 (the format's
/// original architecture).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFamily {
    Gemma4,
    QwenGdnMoe,
    DeepseekV4Flash,
    /// The `llama` GGUF architecture, which covers dense Llama 2/3.x and
    /// Mistral AND the Mixtral MoEs -- one string, distinguished only by
    /// `expert_count` (ROADMAP Phase M2). The baseline is Mixtral's because
    /// every behavioural field is shared and only shape fields differ.
    Llama,
    /// The `qwen3moe` GGUF architecture (Qwen3-30B-A3B and siblings), the
    /// first FINE-GRAINED MoE brought up after Mixtral showed that being MoE
    /// is not enough for this engine's memory result (AGENTS.md Gotcha 36).
    ///
    /// It runs through the SAME decode flow as [`ModelFamily::Llama`]
    /// (`crates/runtime/src/families/llama/`), because the layer graph is
    /// identical: plain GQA, raw residual add, one post-attention norm
    /// feeding router and routed experts, no shared expert, no softcap,
    /// full-head NeoX RoPE. It differs in exactly two places, both carried
    /// by `RealLlamaState`: it norms q and k PER HEAD before RoPE, and its
    /// RMS epsilon is 1e-6 where the `llama` architecture's is 1e-5.
    Qwen3Moe,
    /// The `qwen3` GGUF architecture / HF `model_type: "qwen3"` (Qwen3-0.6B
    /// through 32B dense, Qwen3-Coder's dense sizes,
    /// DeepSeek-R1-0528-Qwen3-8B), the TENTH family and the SECOND to run on
    /// [`ModelFamily::Llama`]'s flow rather than its own.
    ///
    /// It is exactly the combination of two switches that flow already
    /// carries separately, tested together here for the first time: the
    /// per-head q/k norm and 1e-6 RMS epsilon [`ModelFamily::Qwen3Moe`]
    /// already keys on `RealLlamaState`, plus the DENSE FFN arm `Llama`
    /// already derives from `num_experts == 0` for Mistral/Llama 3.x. The
    /// GGUF tensor inventory confirms it structurally: llama.cpp's
    /// `MODEL_ARCH.QWEN3` tensor list is `MODEL_ARCH.QWEN3MOE`'s list minus
    /// the four routed-expert rows, plus plain `FFN_GATE`/`FFN_UP`/
    /// `FFN_DOWN` -- i.e. `qk_norm=true, dense=true`, nothing else moved.
    /// Facts: `docs/QWEN3_PHASE0.md`.
    ///
    /// **NOT to be confused with [`ModelFamily::QwenGdnDense`]**, which is
    /// the unrelated gated-DeltaNet dense family (`model_type: "qwen3_5"`).
    /// The two share only the word "dense": this one is plain GQA, that one
    /// is a hybrid linear-attention architecture at a different `head_dim`.
    ///
    /// `qwen2` / `qwen2.5` (QwQ-32B, the older R1-distill-Qwen line) is a
    /// SEPARATE architecture string with a different config schema (no
    /// per-head QK-norm) and is represented by [`ModelFamily::Qwen2Dense`].
    Qwen3Dense,
    /// Dense Qwen2/Qwen2.5 full-attention models. This shares the Llama flow
    /// with Qwen3 dense, but carries Q/K/V projection biases and uses the
    /// Qwen2 RMS epsilon of 1e-6 without Q/K normalization.
    Qwen2Dense,
    /// The `gpt-oss` GGUF architecture (ROADMAP M5), the third fine-grained
    /// MoE and the FIFTH real family. Chosen by the M5 Phase 0 survey on
    /// AGENTS.md Gotcha 36's axis: 12.6 MiB per expert against Llama 4
    /// Scout's 77.8 and DeepSeek-V3's 34.5, both of which want tens of GiB
    /// of slot cache and cannot stream here.
    ///
    /// Unlike [`ModelFamily::Qwen3Moe`] this is NOT another family on an
    /// existing flow. Its layer differs from every flow here in four ways,
    /// each a first for this port: per-projection BIASES on q/k/v/output and
    /// on the router and every routed expert, ATTENTION SINKS (one learned
    /// logit per q head, added to the softmax denominator only), YARN rope
    /// scaling, and a CLAMPED SwiGLU whose activation is a swish with alpha
    /// times `(up + 1)` rather than silu times `up`. Its alternating
    /// 128-token sliding window is the one part that is free, because
    /// Gemma's SWA ring already exists.
    GptOss,
    /// The `qwen3_5` HF architecture: the gated-DeltaNet hybrid in its DENSE
    /// form. Two published checkpoints, `prism-ml/Bonsai-27B-mlx-1bit`
    /// (ROADMAP's 1-bit entry) and `Qwen/Qwen3.8-27B`, which share one
    /// `ArchConfig` exactly and differ only in quantization.
    ///
    /// **NAMED FOR THE ARCHITECTURE, NOT A VERSION, AND ITS WIRE STRING IS
    /// FROZEN AT THE OLD SPELLING.** [`ModelFamily::as_str`] still returns
    /// `"qwen35"` and [`ModelFamily::parse`] still reads it, because that
    /// string is written into every install's `manifest.json` and read back
    /// at load: renaming it would invalidate every `.gturbo` directory ever
    /// built. So the on-disk identifier is a FORMAT CONSTANT, historical and
    /// deliberately not descriptive, while this Rust name is free to say
    /// what the family is. Do not "fix" the string to match the variant.
    ///
    /// The variant was called `Qwen35` until 2026-08-15 and the rename is
    /// what stopped the drift: upstream keeps `model_type: qwen3_5` stable
    /// across checkpoints named 3.5, 3.6 and 3.8, so a version-shaped name
    /// reads as "the 3.5 one" when it means "the gated-DeltaNet dense one".
    /// Its sibling was worse -- `Qwen36` was named after the Qwen 3.6
    /// checkpoint while matching `model_type: qwen3_5_moe`.
    ///
    /// It runs through the SAME decode flow as [`ModelFamily::QwenGdnMoe`]
    /// (`crates/runtime/src/families/qwen/`), because every BEHAVIOURAL
    /// field is shared -- gated DeltaNet on the linear layers, gated full
    /// attention on every fourth, `attn_output_gate`, `head_dim` 256,
    /// `partial_rotary_factor` 0.25 at theta 1e7, no sandwich norms, no
    /// softcap, silu -- and only SHAPE fields differ (hidden 5120 against
    /// 2048, 64 layers against 40, 24 q heads over 4 kv). Read off the
    /// checkpoint's own `config.json`, not assumed.
    ///
    /// It differs in exactly two ways, and the first is why it needs a
    /// branch rather than just a baseline: it is DENSE, one
    /// `mlp.{gate,up,down}_proj` per layer where Qwen 3.6 has a router, a
    /// shared expert and 256 routed ones. The second is mrope
    /// (`mrope_section [11, 11, 10]`), which on TEXT positions reduces to
    /// the `rope_neox_subdim` already here -- a claim to verify against
    /// the reference, not to assume.
    ///
    /// **A SEPARATE VARIANT DESPITE SHARING A FLOW, and the precedent is
    /// [`ModelFamily::Qwen3Moe`] rather than [`ModelFamily::Llama`].**
    /// `qwen3moe` shares `families/llama/`'s flow ENTIRELY and is still
    /// its own variant, because its architecture string differs. `llama`
    /// covers a dense and an MoE half under one variant only because
    /// Mixtral and Mistral report the SAME string. Strings decide the
    /// variant; flows are shared separately.
    QwenGdnDense,
    /// The `muse_glimmer` HF architecture (`mlx-community/Muse-Glimmer-30B-4bit`,
    /// an MLX INT4 conversion of `meta-models/Muse-Glimmer-30B`), the SEVENTH
    /// family and the SIXTH decode flow.
    ///
    /// A dense 52-layer GQA stack (32 q heads over 2 kv, head_dim 128) with an
    /// alternating three-sliding/one-full window at 2048, sandwich norms and a
    /// logit softcap of 20. Its closest existing graph is Gemma 4's, and it
    /// still needs its own flow, because TEN differences are inside the layer
    /// and every one of them produces fluent WRONG text rather than an error
    /// if a neighbour's flow is used (the `gpt-oss` precedent, which bought a
    /// fifth flow with four such differences):
    ///
    /// 1. Its four per-layer norms are CENTERED -- the effective scale is
    ///    `1 + w` -- where every other family here, Gemma included, applies a
    ///    plain `w`.
    /// 2. Its FINAL norm is a plain `w`. Two conventions in one model, so
    ///    this is not a family-wide switch on the norm kernel.
    /// 3. TWO epsilons: 1e-5 on the input, pre-FFN, q/k and embedding norms,
    ///    and 1e-8 on the two POST norms.
    /// 4. NoPE on the thirteen full-attention layers (`layer_rope_theta` is
    ///    literally 0 there), so the rotation is per layer rather than
    ///    per model.
    /// 5. `qk_scale_factor` 3.87 multiplies Q after its norm, separately from
    ///    the `128^-0.5` attention scale.
    /// 6. The attention output gate is its OWN tensor, `self_attn.gate_proj`,
    ///    not Qwen's packing into `q_proj` -- see `muse_glimmer_30b`'s header
    ///    for why `attn_output_gate` is nonetheless false.
    /// 7. Its q/k norms are NO-SCALE and per-head; Gemma's and `qwen3moe`'s
    ///    are learned, and no `q_norm.weight` exists in the checkpoint.
    /// 8. The embedding row is NORMED (no-scale RMS) where Gemma scales it by
    ///    `sqrt(hidden)` and `llama` does neither.
    /// 9. An `output_multiplier` of `26^-0.5` multiplies the logits before
    ///    the softcap.
    /// 10. The FFN is dense, so there is no router, no shared expert and no
    ///     streamed expert blob at all.
    ///
    /// It is a VISION-language model and this port ingests the TEXT tower
    /// only: `vision_tower.`, `vision_adapter.` and `vision_projection.` are
    /// excluded at repack, as `qwen3_5`'s vision tensors already are.
    MuseGlimmer,
    /// The `qwen4_exp` HF architecture (`Qwen3.8-Flash-Next`), the EIGHTH
    /// family and the first of the Qwen 4 line. Two published checkpoints
    /// share it exactly: `pipenetwork/Qwen3.8-Flash-Next-MLX-4bit` at 512
    /// experts and `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit`, which is
    /// expert-pruned to 288 and differs in `num_experts` and NOTHING ELSE.
    ///
    /// 48 layers on a three-linear/one-full pattern, so 36 gated-DeltaNet
    /// (mask 2) against 12 full-attention (mask 1). 512 routed experts at
    /// top-10 plus a gated shared expert, `head_dim` 256 over 24 q and 2 kv
    /// heads, `partial_rotary_factor` 0.25 at theta 1e7, `attn_output_gate`,
    /// per-head q/k norms, no sandwich norms, no softcap. That much is
    /// [`ModelFamily::QwenGdnMoe`]'s graph at different shapes.
    ///
    /// **IT NEEDS ITS OWN FLOW BECAUSE THE RESIDUAL STREAM IS NOT ONE STREAM.**
    /// The embedding is tiled `hc_count` times, so the stream is
    /// `4 * hidden_size` wide for the whole stack, and every residual ADD is
    /// replaced by a gated-residual read/inject pair
    /// ([`super::sub_configs::HyperConnectionConfig`]). There is no final
    /// `model.norm` at all -- the closing mixer carries it. No line of
    /// `families/qwen/`'s per-token function survives that unchanged, which is
    /// the `gpt-oss` precedent rather than the `qwen3moe` one.
    ///
    /// Three further firsts, each documented where it lives. Its GDN output
    /// gate is SIGMOID where every earlier family's is silu
    /// (`output_gate_sigmoid`, `crates/gpu` Gotcha 12). It carries a hashed
    /// n-gram per-layer embedding at layer index 1
    /// ([`super::sub_configs::PleConfig`]), which is 30.8% of the checkpoint
    /// and streams from its own table. And its full-attention layers carry a
    /// query-sparse INDEXER that selects blocks of compressed keys
    /// (`families/qwen4/attn.rs`) -- see
    /// [`super::sub_configs::CompressedAttentionConfig::sparse_below`] for why
    /// attention at or below `indexer_budget` is exactly dense.
    ///
    /// It is a VISION-language model and this port ingests the TEXT tower
    /// only, as `qwen3_5` and `muse_glimmer` already do.
    Qwen4Exp,
    /// The `spark2_5` architecture (`XHToken/Spark-X2.5-4B`), the NINTH
    /// decode flow and the first family whose GGUF conventions come from an
    /// upstream llama.cpp merge rather than a fork (PR 27868, 2026-09-06).
    ///
    /// A dense 36-layer stack whose hybrid window is byte-identical to
    /// [`ModelFamily::MuseGlimmer`]'s: `[swa, swa, swa, full]` repeating, 27
    /// sliding layers at window 512 against 9 full. It runs its own flow
    /// (`crates/runtime/src/families/spark/`) because four things inside the
    /// layer are each a first for this port, and every one of them produces
    /// fluent WRONG text on a neighbour's flow:
    ///
    /// 1. PER-CLASS RoPE WITH A PER-CLASS PARTIAL FACTOR: full layers rotate
    ///    only the leading 64 of 256 dims at theta 5e6, SWA layers rotate all
    ///    256 at theta 1e4. Gemma 4 shares the per-class theta shape and its
    ///    factor-1.0-on-SWA hard-coding, but applies `rope_proportional_neox`
    ///    semantics; Spark's divisor is the ROTARY dim (64), which is
    ///    `rope_neox_subdim`'s convention.
    /// 2. A HEADWISE SCALAR output gate: `g_proj` maps hidden to num_heads (16
    ///    scalars), sigmoid, broadcast over head_dim before `o_proj`. Qwen
    ///    packs a full-width per-head gate into `q_proj`; muse's separate
    ///    gate is full-width. `attn_output_gate` is false with the muse
    ///    precedent: the gate is its own tensor and the flow applies it
    ///    unconditionally.
    /// 3. EXACT-ERF GELU on a gated MLP (`hidden_act: "gelu"`, and the
    ///    reference implementation refuses anything else). Trunk paths only
    ///    had silu and tanh-GELU.
    /// 4. FUSED `q_k_v_proj` with no gate packed inside, so the QKV split is
    ///    plain 4096/1024/1024.
    ///
    /// No QK-norm, no sandwich norms, no softcap, no embedding scaling,
    /// attention scale exactly `256^-0.5`, tied embeddings. Everything else
    /// -- the SWA ring, the split-KV decode attention, the dequant GEMV
    /// matrix -- is composition over existing kernels. Facts:
    /// `docs/SPARK_PHASE0.md`.
    Spark25,
    /// MiniMax-M2: full GQA with whole-projection Q/K norms and sigmoid MoE.
    MiniMaxM2,
    /// The `qwen3_vl` HF architecture (`Qwen3-VL-4B-Instruct` and siblings),
    /// the FIFTEENTH family and the THIRD to run on
    /// [`ModelFamily::Llama`]'s flow rather than its own.
    ///
    /// A dense 36-layer GQA stack (32 q heads over 8 kv, `head_dim` 128 -- an
    /// independent field, since 32 x 128 does not divide hidden 2560), full
    /// rotary over the whole head, per-head q/k RMSNorm before RoPE at eps
    /// 1e-6, a plain gated FFN, TIED embeddings, no softcap, no sliding
    /// window, no biases. That is exactly [`ModelFamily::Qwen3Dense`]'s
    /// switch combination (`QkNorm::PerHead` plus the dense-FFN arm) at
    /// different shapes, so `RealLlamaState` carries it with one new family
    /// arm and no new flow.
    ///
    /// **`partial_rotary_factor` is 1.0 and `rope_neox_subdim` is false**, so
    /// the flow dispatches full-width `rope_proportional_neox`. The
    /// checkpoint declares `mrope_interleaved: true` with
    /// `mrope_section [24, 20, 20]`, but on TEXT positions the reference's
    /// mRoPE gives every token `t == h == w` and the three sections collapse
    /// to one position, which is bit-identical to plain full-width NeoX rope
    /// (`docs/VISION_PHASE0.md` item 2, and `families/qwen/attn.rs`'s
    /// degenerate-arm note). The mRoPE kernel is only needed where an image
    /// makes the three components diverge, which is the vision bring-up this
    /// family's text-first pass deliberately defers.
    ///
    /// It is a VISION-language model -- the checkpoint ships a SigLIP-class
    /// tower this port already has the kernels for, plus three DEEPSTACK
    /// mergers (the one genuinely new component: intermediate block outputs
    /// merged and raw-added into trunk layers 0-2's residuals) -- and this
    /// port ingests the TEXT tower only in this pass, as `qwen3_5` and
    /// `muse_glimmer` already do: every `vision_tower.*` tensor, the 18
    /// deepstack ones included, is `ExcludedMultimodal` at repack. Facts:
    /// `docs/QWEN3VL_PHASE0.md`.
    Qwen3Vl,
    /// The `deepseek2` GGUF architecture / HF `model_type: "deepseek_v2"`:
    /// multi-head latent attention plus fine-grained MoE. The pinned
    /// witness is `DeepSeek-V2-Lite-Chat` (16B, 64 experts top-6 plus a
    /// fused shared expert, one dense lead layer); the same string reports
    /// DeepSeek V2/V3, Kimi K2.5/K2.6, GLM-4.7-Flash and Mistral-Large-3,
    /// which is why the roadmap calls it the highest-leverage unlock.
    ///
    /// Its attention compresses each token's KV into ONE row per layer --
    /// `[latent 512 ; rope-carried key 64]` -- and runs the ABSORBED form
    /// (q_nope folded through the up-projection, V read from the cached
    /// row itself), which converts the attention into MQA over that row.
    /// Every layer is mask 5, a kind no earlier family carries. Facts:
    /// `docs/DEEPSEEK2_PHASE0.md`.
    Deepseek2,
}

impl ModelFamily {
    /// Exhaustive list of all 15 model families.
    pub const ALL: [ModelFamily; 15] = [
        ModelFamily::Gemma4,
        ModelFamily::QwenGdnMoe,
        ModelFamily::DeepseekV4Flash,
        ModelFamily::Llama,
        ModelFamily::Qwen3Moe,
        ModelFamily::GptOss,
        ModelFamily::QwenGdnDense,
        ModelFamily::MuseGlimmer,
        ModelFamily::Qwen4Exp,
        ModelFamily::Spark25,
        ModelFamily::Qwen3Dense,
        ModelFamily::MiniMaxM2,
        ModelFamily::Qwen2Dense,
        ModelFamily::Deepseek2,
        ModelFamily::Qwen3Vl,
    ];

    /// Returns static string identifier for the model family.
    ///
    /// **THESE STRINGS ARE AN ON-DISK FORMAT AND TWO OF THEM NO LONGER MATCH
    /// THEIR VARIANT'S NAME. That is deliberate.** Every `.gturbo` install
    /// records this value in `manifest.json`, and [`ModelFamily::parse`]
    /// reads it back at load, so a string here is a compatibility promise to
    /// artifacts already on disk -- not a label to keep tidy.
    ///
    /// `QwenGdnMoe` therefore still writes `"qwen36"` and `QwenGdnDense`
    /// still writes `"qwen35"`, the version-shaped names both variants were
    /// called before 2026-08-15. Changing either would make every existing
    /// install of those families unloadable (`parse` returns `None`, and the
    /// open path reports an unknown family) for a cosmetic gain. The Rust
    /// names carry the meaning; these carry the history.
    ///
    /// A NEW family is free to pick a matching string, because nothing has
    /// been written with it yet.
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelFamily::Gemma4 => "gemma4",
            ModelFamily::QwenGdnMoe => "qwen36",
            ModelFamily::DeepseekV4Flash => "deepseekV4Flash",
            ModelFamily::Llama => "llama",
            ModelFamily::Qwen3Moe => "qwen3moe",
            ModelFamily::GptOss => "gptOss",
            ModelFamily::QwenGdnDense => "qwen35",
            // A NEW family, so the string is free to match the variant: no
            // `.gturbo` directory has ever been written with it.
            ModelFamily::MuseGlimmer => "museGlimmer",
            ModelFamily::Qwen4Exp => "qwen4exp",
            // Also new, and it matches the GGUF `general.architecture` string
            // AND the HF `model_type`, which agree ("spark2_5").
            ModelFamily::Spark25 => "spark2_5",
            // Also new, and matches the GGUF `general.architecture` string
            // AND the HF `model_type`, which agree ("qwen3").
            ModelFamily::Qwen3Dense => "qwen3",
            ModelFamily::MiniMaxM2 => "minimax_m2",
            ModelFamily::Qwen2Dense => "qwen2",
            // New, and it matches the GGUF `general.architecture` string.
            // The HF `model_type` is "deepseek_v2"; the GGUF spelling won
            // because the pinned witness is a GGUF and the registry key
            // must not have a second copy.
            ModelFamily::Deepseek2 => "deepseek2",
            // New, and it matches the HF `model_type` ("qwen3_vl"). The
            // pinned witness is an MLX safetensors conversion, so the HF
            // spelling won; there is no GGUF `general.architecture` row for
            // this family (its `gguf_names` arm is `None`).
            ModelFamily::Qwen3Vl => "qwen3_vl",
        }
    }

    /// Parses string identifier into model family.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "gemma4" => Some(ModelFamily::Gemma4),
            "qwen36" => Some(ModelFamily::QwenGdnMoe),
            "deepseekV4Flash" => Some(ModelFamily::DeepseekV4Flash),
            "llama" => Some(ModelFamily::Llama),
            "qwen3moe" => Some(ModelFamily::Qwen3Moe),
            "gptOss" => Some(ModelFamily::GptOss),
            "qwen35" => Some(ModelFamily::QwenGdnDense),
            "museGlimmer" => Some(ModelFamily::MuseGlimmer),
            "qwen4exp" => Some(ModelFamily::Qwen4Exp),
            "spark2_5" => Some(ModelFamily::Spark25),
            "qwen3" => Some(ModelFamily::Qwen3Dense),
            "minimax_m2" => Some(ModelFamily::MiniMaxM2),
            "qwen2" => Some(ModelFamily::Qwen2Dense),
            "deepseek2" => Some(ModelFamily::Deepseek2),
            "qwen3_vl" => Some(ModelFamily::Qwen3Vl),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_families_matches_every_variant_exhaustively() {
        for (idx, &family) in ModelFamily::ALL.iter().enumerate() {
            let expected_idx = match family {
                ModelFamily::Gemma4 => 0,
                ModelFamily::QwenGdnMoe => 1,
                ModelFamily::DeepseekV4Flash => 2,
                ModelFamily::Llama => 3,
                ModelFamily::Qwen3Moe => 4,
                ModelFamily::GptOss => 5,
                ModelFamily::QwenGdnDense => 6,
                ModelFamily::MuseGlimmer => 7,
                ModelFamily::Qwen4Exp => 8,
                ModelFamily::Spark25 => 9,
                ModelFamily::Qwen3Dense => 10,
                ModelFamily::MiniMaxM2 => 11,
                ModelFamily::Qwen2Dense => 12,
                ModelFamily::Deepseek2 => 13,
                ModelFamily::Qwen3Vl => 14,
            };
            assert_eq!(idx, expected_idx);
        }
        assert_eq!(ModelFamily::ALL.len(), 15);
    }
}
