//! `muse_glimmer` `config.json` -> [`ArchConfig`], plus the four published
//! scalars that deliberately do NOT live in an `ArchConfig`.
//!
//! The family-specific piece of the repack path for the seventh family.
//! Everything downstream of it -- `classify_for_family`, `manifest_quant`,
//! `write_muse_glimmer_install_streamed` -- already takes a `ModelFamily`.
//!
//! Measured against `mlx-community/Muse-Glimmer-30B-4bit`
//! @ `3e7677d7a40d348a3daba263a2b1c0aa41910710`:
//!
//! | `ArchConfig` field | `config.json -> text_config` key |
//! |---|---|
//! | `intermediate_size` | `intermediate_size`, the DENSE FFN width |
//! | `hidden_activation` | `hidden_activation` (Gemma's spelling, not Qwen's `hidden_act`) |
//! | `full_attention_layer_mask` | `layer_types`, `sliding_attention` -> **0**, `full_attention` -> **1** |
//! | `rope_theta` | the NONZERO entries of `layer_rope_theta` |
//! | `full_rope_theta` | the entries of `layer_rope_theta` on FULL layers, which are 0 |
//! | `final_logit_softcap` | `final_logit_softcapping` |
//! | `sliding_window` | `sliding_window` |
//!
//! **`layer_rope_theta` AND `layer_types` ARE TWO INDEPENDENT ARRAYS THAT
//! MUST AGREE, and this parser checks that they do.** The file publishes the
//! window pattern once as strings and once as thetas, and the reference reads
//! them separately -- `is_sliding` off `layer_types`, `use_rope` off
//! `bool(layer_rope_theta[i])`. Nothing upstream makes them consistent. If a
//! future revision moves one and not the other, an engine that reads only
//! `layer_types` rotates a NoPE layer (or fails to rotate a rotating one) and
//! generates fluent wrong text, which is AGENTS.md Gotcha 33's failure mode
//! exactly. Cross-checking costs one pass and turns that into a refusal.
//!
//! `attention_scale` has no key at all. It is `head_dim ** -0.5`, taken from
//! the reference (`mlx_vlm.models.muse_glimmer.language.Attention.__init__`
//! sets `self.scale = self.head_dim**-0.5`) and NOT from the formula anyone
//! remembers -- `docs/NEW_MODEL.md` Phase 0 explains why the distinction
//! matters, and this family is precisely a case where the obvious guess is
//! wrong twice over: there is a SECOND scale (`qk_scale_factor`) applied to Q
//! elsewhere, and folding the two would give an arbitrary decimal on
//! `arch_validation`'s `!=` path (AGENTS.md Gotcha 24).

use model_io::{
    muse_glimmer_layer_mask, ArchConfig, CompressedAttentionConfig, HyperConnectionConfig,
    LinearAttentionConfig, MlaConfig, ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

use crate::gemma4_checkpoint::Gemma4Error;

/// Layer-mask code for a sliding-window attention layer.
const MASK_SLIDING: u8 = 0;
/// Layer-mask code for a full-attention layer.
const MASK_FULL: u8 = 1;

/// The four published scalars that are family CONSTANTS in
/// `crates/runtime`'s `families/museglimmer/state.rs` rather than
/// [`ArchConfig`] fields.
///
/// **They are parsed here even though nothing in the repack path consumes
/// them, and that is the whole point.** Putting them in `ArchConfig` would
/// route them through `manifest.json`, where `arch_validation` compares
/// floats with `!=` on `f64` and serde_json's default parser is accurate only
/// to ~1 ULP -- and not one of these four is a binary fraction, so that is
/// AGENTS.md Gotcha 24 with four fresh chances to fire. Keeping them as
/// constants follows the `rms_eps` precedent (`llama`'s 1e-5 against
/// `qwen3moe`'s 1e-6 are constants too).
///
/// The risk that trade takes on is the one AGENTS.md Gotcha 38 names: a
/// constant is a value RECALLED where the file states one, and it stays
/// correct exactly as long as one checkpoint exercises it.
/// `tests/museglimmer_config.rs` closes that by asserting these parsed values
/// against the flow's constants, so a future checkpoint that moves any of
/// them reddens a millisecond offline test instead of decoding subtly wrong.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MuseGlimmerScalars {
    /// `text_config.qk_scale_factor`. Multiplies Q AFTER its per-head
    /// no-scale norm and BEFORE RoPE, in FP32, separately from
    /// `ArchConfig::attention_scale`.
    pub qk_scale_factor: f64,
    /// `text_config.output_multiplier`. Multiplies the logits BEFORE the
    /// softcap. Equals `26^-0.5`.
    pub output_multiplier: f64,
    /// `text_config.rms_norm_eps`. The input, pre-FFN, q/k and embedding
    /// norms.
    pub rms_norm_eps: f64,
    /// `text_config.post_norm_eps`. The two POST norms only, and four orders
    /// of magnitude smaller than its sibling above.
    pub post_norm_eps: f64,
}

/// Parses a `muse_glimmer` `config.json` into an [`ArchConfig`].
///
/// The checkpoint is multimodal (`MuseGlimmerForConditionalGeneration`), so
/// the text tower lives under a `text_config` wrapper and `vision_config` is
/// ignored -- `classify_for_family` drops the `vision_tower.`,
/// `vision_adapter.` and `vision_projection.` tensors separately.
///
/// Cross-check the result against [`model_io::muse_glimmer_30b`]: if the two
/// disagree field for field, one of them is wrong.
pub fn parse_muse_glimmer_config(json: &str) -> Result<ArchConfig, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    crate::arch_registry::refuse_foreign_config(&root, ModelFamily::MuseGlimmer)
        .map_err(Gemma4Error::Config)?;
    // Text-only conversions drop the wrapper; accept both shapes.
    let tc = root.get("text_config").unwrap_or(&root);

    let i = |k: &str| -> Result<i64, Gemma4Error> {
        tc.get(k)
            .and_then(|v| v.as_i64())
            .ok_or_else(|| Gemma4Error::Config(format!("missing {k}")))
    };
    let f = |k: &str| -> Result<f64, Gemma4Error> {
        tc.get(k)
            .and_then(|v| v.as_f64())
            .ok_or_else(|| Gemma4Error::Config(format!("missing {k}")))
    };
    let b = |k: &str| tc.get(k).and_then(|v| v.as_bool()).unwrap_or(false);

    let num_layers = i("num_hidden_layers")?;
    let mask = parse_layer_types(tc, num_layers)?;
    let rope_theta = parse_layer_rope_theta(tc, &mask)?;

    // A DENSE architecture. A config that also declares experts is a
    // contradiction rather than extra information: reading only half of it
    // yields an `ArchConfig` whose FFN width and expert count disagree, which
    // validates structurally and then dispatches a branch this family has no
    // tensors for. Refused, following `parse_qwen_gdn_dense_config`.
    for key in ["num_experts", "num_local_experts", "n_routed_experts"] {
        if let Some(n) = tc.get(key).and_then(|v| v.as_i64()) {
            if n > 0 {
                return Err(Gemma4Error::Config(format!(
                    "a muse_glimmer config declares {key} {n}; this family is dense and has \
                     no routed experts"
                )));
            }
        }
    }

    let head_dim = i("head_dim")?;
    let kv_heads = i("num_key_value_heads")?;

    Ok(ArchConfig {
        hidden_size: i("hidden_size")?,
        intermediate_size: i("intermediate_size")?,
        moe_intermediate_size: 0,
        num_heads: i("num_attention_heads")?,
        // One head dim and one KV head count for both layer kinds: unlike
        // Gemma, this architecture publishes no separate global-attention
        // keys, and the reference builds every layer's projections from the
        // same two numbers.
        num_kv_heads: kv_heads,
        num_full_kv_heads: kv_heads,
        head_dim,
        full_head_dim: head_dim,
        vocab_size: i("vocab_size")?,
        sliding_window: i("sliding_window")?,
        final_logit_softcap: f("final_logit_softcapping")?,
        rope_theta,
        // ZERO, and it is the file's own value rather than an absent
        // sentinel: the full-attention layers are NoPE. See
        // `parse_layer_rope_theta`.
        full_rope_theta: 0.0,
        // The reference builds its rope at the full `head_dim` with no
        // partial factor, so every pair of a rotating layer's head rotates.
        partial_rotary_factor: 1.0,
        num_layers,
        dense_lead_intermediate_size: 0,
        num_dense_leading_layers: 0,
        num_experts: 0,
        top_k_experts: 0,
        tie_word_embeddings: b("tie_word_embeddings"),
        attention_k_eq_v: false,
        full_attention_layer_mask: mask,
        hidden_activation: tc
            .get("hidden_activation")
            .and_then(|v| v.as_str())
            .unwrap_or("silu")
            .to_string(),
        family: ModelFamily::MuseGlimmer,
        // FALSE even though this architecture HAS an attention output gate:
        // the field asks whether `q_proj` emits packed `[query; gate]` rows
        // (Qwen's shape), and this family's gate is its own tensor. See
        // `model_io::muse_glimmer_30b`'s header.
        attn_output_gate: false,
        // See the module header: `qk_scale_factor` is deliberately NOT
        // folded in here.
        attention_scale: (head_dim as f64).powf(-0.5),
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: true,
        shared_expert_gated: false,
        rope_neox_subdim: false,
        linear_attention: LinearAttentionConfig::NONE,
        mla: MlaConfig::NONE,
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: RopeScalingConfig::NONE,
        vision: VisionConfig::NONE,
        ple: PleConfig::NONE,
    })
}

/// Reads the four published scalars the [`ArchConfig`] deliberately omits.
///
/// See [`MuseGlimmerScalars`] for why they are not fields.
pub fn parse_muse_glimmer_scalars(json: &str) -> Result<MuseGlimmerScalars, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    crate::arch_registry::refuse_foreign_config(&root, ModelFamily::MuseGlimmer)
        .map_err(Gemma4Error::Config)?;
    let tc = root.get("text_config").unwrap_or(&root);
    let f = |k: &str| -> Result<f64, Gemma4Error> {
        tc.get(k)
            .and_then(|v| v.as_f64())
            .ok_or_else(|| Gemma4Error::Config(format!("missing {k}")))
    };
    Ok(MuseGlimmerScalars {
        qk_scale_factor: f("qk_scale_factor")?,
        output_multiplier: f("output_multiplier")?,
        rms_norm_eps: f("rms_norm_eps")?,
        post_norm_eps: f("post_norm_eps")?,
    })
}

/// `text_config.layer_types` -> the 0/1 window mask.
///
/// Required and length-checked rather than defaulted, for the reason Qwen's
/// is: a short or absent mask would still validate structurally while running
/// the wrong block on every layer past its end.
fn parse_layer_types(tc: &serde_json::Value, num_layers: i64) -> Result<Vec<u8>, Gemma4Error> {
    let types = tc
        .get("layer_types")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Gemma4Error::Config("missing layer_types".to_string()))?;
    let mask = types
        .iter()
        .map(|t| match t.as_str() {
            Some("sliding_attention") => Ok(MASK_SLIDING),
            Some("full_attention") => Ok(MASK_FULL),
            other => Err(Gemma4Error::Config(format!(
                "unknown layer_types entry {other:?}"
            ))),
        })
        .collect::<Result<Vec<u8>, _>>()?;
    if mask.len() as i64 != num_layers {
        return Err(Gemma4Error::Config(format!(
            "layer_types has {} entries but num_hidden_layers is {num_layers}",
            mask.len()
        )));
    }
    Ok(mask)
}

/// `text_config.layer_rope_theta` -> the ONE nonzero theta, cross-checked
/// against the window mask.
///
/// Three things are refused rather than smoothed over, and each corresponds
/// to a way this port's single-`rope_theta` `ArchConfig` would misrepresent
/// the file:
///
/// 1. **A nonzero theta on a FULL layer, or a zero on a SLIDING one.** The
///    two arrays disagree, so one of the reference's two reads is not what
///    this parser is about to encode. See the module header.
/// 2. **Two different nonzero thetas.** `ArchConfig` carries one per layer
///    KIND, not one per layer, so a genuinely per-layer schedule cannot be
///    represented and must not be silently collapsed to the first.
/// 3. **No nonzero theta at all.** A model with no rotating layer is not
///    this architecture, and `rope_theta: 0.0` would read as "absent".
fn parse_layer_rope_theta(tc: &serde_json::Value, mask: &[u8]) -> Result<f64, Gemma4Error> {
    let thetas = tc
        .get("layer_rope_theta")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Gemma4Error::Config("missing layer_rope_theta".to_string()))?;
    if thetas.len() != mask.len() {
        return Err(Gemma4Error::Config(format!(
            "layer_rope_theta has {} entries but layer_types has {}",
            thetas.len(),
            mask.len()
        )));
    }
    let mut sliding_theta: Option<f64> = None;
    for (layer, (t, &kind)) in thetas.iter().zip(mask).enumerate() {
        let theta = t.as_f64().ok_or_else(|| {
            Gemma4Error::Config(format!("layer_rope_theta[{layer}] is not a number"))
        })?;
        let rotates = theta != 0.0;
        let is_sliding = kind == MASK_SLIDING;
        if rotates != is_sliding {
            return Err(Gemma4Error::Config(format!(
                "layer {layer} is {} by layer_types but layer_rope_theta[{layer}] is {theta}; \
                 this architecture rotates its sliding layers and leaves its full layers NoPE, \
                 and the two arrays must agree",
                if is_sliding { "sliding" } else { "full" }
            )));
        }
        if rotates {
            match sliding_theta {
                None => sliding_theta = Some(theta),
                Some(first) if first != theta => {
                    return Err(Gemma4Error::Config(format!(
                        "layer_rope_theta carries two different nonzero values ({first} and \
                         {theta} at layer {layer}); ArchConfig carries one theta per layer KIND \
                         and cannot represent a per-layer schedule"
                    )))
                }
                Some(_) => {}
            }
        }
    }
    sliding_theta.ok_or_else(|| {
        Gemma4Error::Config(
            "layer_rope_theta is zero on every layer, so no layer rotates at all".to_string(),
        )
    })
}

/// The `[0, 0, 0, 1]` mask this family's real checkpoint declares, for a
/// fixture that wants the pattern without a `config.json`.
///
/// Re-exported from `model_io` so the repack-side fixtures and the baseline
/// cannot disagree about the period.
pub fn muse_glimmer_mask(num_layers: i64) -> Vec<u8> {
    muse_glimmer_layer_mask(num_layers)
}
