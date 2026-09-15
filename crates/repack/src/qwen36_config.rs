//! Qwen 3.6 `config.json` -> [`ArchConfig`].
//!
//! The ONE family-specific piece of the repack path. Everything downstream
//! of it -- `classify_for_family`, `manifest_quant`, `write_qwen_gdn_moe_install`,
//! `write_qwen_gdn_moe_install_streamed` -- already takes a `ModelFamily` and
//! needs nothing else from this module.
//!
//! Shaped like [`crate::parse_gemma4_config`], but almost every key name
//! differs, so it is a separate function rather than a parameterized one.
//! Measured against `mlx-community/Qwen3.6-35B-A3B-4bit`:
//!
//! | `ArchConfig` field | `config.json -> text_config` key |
//! |---|---|
//! | `intermediate_size` | `shared_expert_intermediate_size` (NOT `intermediate_size`, which the text config does not carry) |
//! | `top_k_experts` | `num_experts_per_tok` |
//! | `hidden_activation` | `hidden_act` (Gemma spells it `hidden_activation`) |
//! | `full_attention_layer_mask` | `layer_types`, `linear_attention` -> **2**, `full_attention` -> **1** (Gemma's mapping is 1/0) |
//! | `rope_theta`, `partial_rotary_factor` | `rope_parameters`, a FLAT object here; Gemma nests one sub-object per attention kind |
//! | `num_full_kv_heads`, `full_head_dim` | no separate global-attention keys: the full-attention layers reuse `num_key_value_heads` / `head_dim` |
//! | `linear_attention` | the five `linear_*` keys |
//!
//! `attention_scale` has no key at all. It is `head_dim ** -0.5`, taken from
//! the reference implementation (mlx-lm `Qwen3NextAttention.__init__` sets
//! `self.scale = self.head_dim**-0.5`), not assumed from the formula --
//! `docs/NEW_MODEL.md` Phase 0 explains why that distinction matters. At
//! `head_dim = 256` it is exactly 1/16, a binary fraction, so it survives
//! the serde_json round trip AGENTS.md Gotcha 24 warns about.
//!
//! Fields with no config key (`router_scaled`, `ffn_sandwich_norms`,
//! `rope_neox_subdim`, ...) are family constants, written out the same way
//! `parse_gemma4_config` writes Gemma's.

use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

use crate::gemma4_checkpoint::Gemma4Error;

/// The seed `qwen4_exp`'s n-gram hash multipliers derive from when a
/// checkpoint omits its `layer_multipliers` buffer.
///
/// NOT a `config.json` key on either published checkpoint. It is
/// transformers' own default, which the reference implementation names
/// explicitly (`seed: int = 1234  # transformers default; config.json has no
/// seed key`). Recorded because an absent optional key means the FORMAT's
/// default and not a neighbour's value (AGENTS.md Gotcha 39), and because
/// both published checkpoints DO ship the buffer -- so this is a cross-check
/// on those bytes rather than the value anything reads.
const QWEN4_PLE_DEFAULT_SEED: i64 = 1234;

/// Layer-mask code for a gated-DeltaNet linear-attention layer.
const MASK_LINEAR: u8 = 2;
/// Layer-mask code for a full-attention layer.
const MASK_FULL: u8 = 1;

/// Parses a Qwen 3.6 `config.json` into an [`ArchConfig`].
///
/// The checkpoint is multimodal (`Qwen3_5MoeForConditionalGeneration`), so
/// the text tower lives under a `text_config` wrapper and `vision_config`
/// is ignored -- `classify_for_family` drops the `vision_tower.` tensors
/// separately.
///
/// Cross-check the result against [`model_io::qwen_gdn_moe_35b_a3b`]: if the two
/// disagree field for field, one of them is wrong.
pub fn parse_qwen_gdn_moe_config(json: &str) -> Result<ArchConfig, Gemma4Error> {
    parse_qwen_family_config(json, ModelFamily::QwenGdnMoe)
}

/// Parses a `qwen3_5` `config.json` into an [`ArchConfig`]
/// (`prism-ml/Bonsai-27B-mlx-1bit`, ROADMAP's 1-bit entry).
///
/// **The same parser as [`parse_qwen_gdn_moe_config`] with FOUR fields resolved
/// differently, which is why this is a parameterized body and Qwen 3.6's own
/// parser is not a parameterized Gemma one.** Against Gemma almost every key
/// name differs; against Qwen 3.6 almost none does, because the two share a
/// behavioural profile entirely (see [`model_io::qwen_gdn_dense_27b`]). What differs
/// is exactly the FFN:
///
/// | field | `qwen3_5` | `qwen3_5_moe` |
/// |---|---|---|
/// | `intermediate_size` | `intermediate_size`, the DENSE FFN | `shared_expert_intermediate_size` |
/// | `moe_intermediate_size` | 0 | `moe_intermediate_size` |
/// | `num_experts` / `top_k_experts` | 0 | `num_experts` / `num_experts_per_tok` |
/// | `shared_expert_gated` | false | true |
///
/// Cross-check the result against [`model_io::qwen_gdn_dense_27b`]: if the two
/// disagree field for field, one of them is wrong.
pub fn parse_qwen_gdn_dense_config(json: &str) -> Result<ArchConfig, Gemma4Error> {
    parse_qwen_family_config(json, ModelFamily::QwenGdnDense)
}

/// Parses a `qwen4_exp` `config.json` into an [`ArchConfig`]
/// (Qwen3.8-Flash-Next, the first of the Qwen 4 line).
///
/// **THE SAME PARSER AGAIN, because this config is `qwen3_5_moe`'s plus ten
/// keys rather than a different vocabulary.** Every key
/// [`parse_qwen_gdn_moe_config`] reads is present here at the same name and
/// the same meaning, down to `layer_types` and the flat `rope_parameters`
/// object. What `qwen4_exp` ADDS is three components no earlier family has:
///
/// | `ArchConfig` field | `text_config` keys |
/// |---|---|
/// | `hyper_connections` | `hc_count`, `hc_lowrank` |
/// | `compressed_attention` | the five `indexer_*` keys |
/// | `ple` | `ngram_size`, `heads_per_ngram`, `ngram_vocab_size_base`, `make_ngram_vocab_size_divisible_by`, `split_ngram_parts`, `ple_embed_dim`, `ple_conv_kernel_size`, `ple_layer_ids` |
/// | `linear_attention.output_gate_sigmoid` | `output_gate_type` |
///
/// **ONE FIELD IS A FAMILY CONSTANT HERE AND A CONFIG KEY THERE, and reading
/// it the shared way is silently wrong.** `attn_output_gate` is a real
/// `text_config` key on both `qwen3_5` checkpoints, and `qwen4_exp` declares
/// it NOWHERE -- while its reference `Attention` sizes `q_proj` at
/// `n_heads * head_dim * 2` and splits `[query; gate]` unconditionally. So
/// `b("attn_output_gate")` returns false on a model that has one, which
/// halves the projection and drops the gate with no error (AGENTS.md Gotcha
/// 39: a default is a claim about what silence means, and here silence means
/// yes). It is set from the family instead.
///
/// Cross-check the result against [`model_io::qwen4_exp_125b_a6b`]: the two
/// published checkpoints differ from it in `num_experts` alone.
pub fn parse_qwen4_exp_config(json: &str) -> Result<ArchConfig, Gemma4Error> {
    parse_qwen_family_config(json, ModelFamily::Qwen4Exp)
}

/// The only vision-tower depth supported by the current Qwen 3.5 kernels and
/// packed layout. All published checkpoints use this depth.
pub(crate) const SUPPORTED_VISION_DEPTH: i64 = 27;

/// Parses `qwen4_exp`'s hyper-connection, indexer and PLE blocks.
///
/// Split out because it is the only part of `parse_qwen_family_config` that
/// does not apply to all three families, and because each of the three blocks
/// is REQUIRED once the family is known: every field is a shape some kernel
/// strides by or a threshold a refusal quotes, so a default would be this
/// port inventing a number the checkpoint declined to state.
fn parse_qwen4_extensions(
    tc: &serde_json::Value,
) -> Result<(HyperConnectionConfig, CompressedAttentionConfig, PleConfig), Gemma4Error> {
    let i = |k: &str| -> Result<i64, Gemma4Error> {
        tc.get(k)
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| Gemma4Error::Config(format!("missing {k}")))
    };

    let hc_count = i("hc_count")?;
    if hc_count < 1 {
        return Err(Gemma4Error::Config(format!(
            "hc_count {hc_count} would make the residual stream narrower than one stream"
        )));
    }
    let hyper_connections = HyperConnectionConfig {
        mult: hc_count,
        lowrank: i("hc_lowrank")?,
        // DeepSeek mHC's terms. This mixer is a low-rank silu/sigmoid blend
        // with no Sinkhorn normalisation, so both stay zero.
        sinkhorn_iters: 0,
        eps: 0.0,
    };

    let compress = i("indexer_compress_ratio")?;
    let budget = i("indexer_budget")?;
    if compress < 1 {
        return Err(Gemma4Error::Config(format!(
            "indexer_compress_ratio {compress} does not pool a positive number of tokens"
        )));
    }
    if budget % compress != 0 {
        return Err(Gemma4Error::Config(format!(
            "indexer_budget {budget} is not a whole number of {compress}-token blocks"
        )));
    }
    let compressed_attention = CompressedAttentionConfig {
        index_n_heads: i("indexer_n_heads")?,
        index_kv_heads: i("indexer_kv_heads")?,
        index_head_dim: i("indexer_head_dim")?,
        // BLOCKS, which is this architecture's selection unit. DeepSeek's
        // `index_top_k` counts tokens; the reference derives this one as
        // `token_budget // compress_ratio` and takes a top-k over blocks.
        index_top_k: budget / compress,
        index_budget: budget,
        csa_compress_rate: compress,
        ..CompressedAttentionConfig::NONE
    };

    let ngram_size = i("ngram_size")?;
    let heads_per_ngram = i("heads_per_ngram")?;
    let layer_ids = tc
        .get("ple_layer_ids")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Gemma4Error::Config("missing ple_layer_ids".to_string()))?
        .iter()
        .map(|v| {
            v.as_i64()
                .ok_or_else(|| Gemma4Error::Config("ple_layer_ids entry is not an integer".into()))
        })
        .collect::<Result<Vec<i64>, _>>()?;
    if layer_ids.is_empty() {
        return Err(Gemma4Error::Config(
            "ple_layer_ids is empty, which this port cannot tell from an absent PLE table; \
             a checkpoint without one should omit the block"
                .to_string(),
        ));
    }
    // ONE-BASED in the checkpoint, so 0 is not a layer anyone can name and a
    // zero entry means the file is using a convention this parser does not.
    if let Some(bad) = layer_ids.iter().find(|id| **id < 1) {
        return Err(Gemma4Error::Config(format!(
            "ple_layer_ids contains {bad}; the ids are one-based, so the first layer is 1"
        )));
    }
    let ple = PleConfig {
        ngram_size,
        heads_per_ngram,
        ngram_vocab_size_base: i("ngram_vocab_size_base")?,
        make_divisible_by: i("make_ngram_vocab_size_divisible_by")?,
        split_ngram_parts: i("split_ngram_parts")?,
        ple_embed_dim: i("ple_embed_dim")?,
        conv_kernel_size: i("ple_conv_kernel_size")?,
        layer_ids,
        // NOT a config key. The reference defaults it to 1234 (transformers'
        // own), and it is what the hash multipliers derive from when a
        // checkpoint omits its `layer_multipliers` buffer. Both published
        // checkpoints ship that buffer, so this is a cross-check rather than
        // a source of truth -- but the FORMAT's default is what an absent key
        // means, never a neighbour's value (AGENTS.md Gotcha 39).
        seed: QWEN4_PLE_DEFAULT_SEED,
        // The n-gram context resets at EOS boundaries
        // (`docs/QWEN4_PHASE0.md` item 4); `text_config.eos_token_id` is a
        // SCALAR, distinct from `generation_config.json`'s two-entry stop
        // list, which is why this reads `tc` and not that other file. A
        // missing key is a hard parse error (`i` returns `Err`), matching
        // the reference's own `validate_architecture`, which refuses PLE
        // with no EOS set rather than guessing one.
        eos_token_id: i("eos_token_id")?,
    };
    // The head count divides the embedding width in the reference
    // (`head_dim = embed_dim // ngram_heads`), so a config where it does not
    // divide evenly would silently truncate every row by the remainder.
    let heads = ple.ngram_heads();
    if heads < 1 {
        return Err(Gemma4Error::Config(format!(
            "ngram_size {ngram_size} and heads_per_ngram {heads_per_ngram} give {heads} hash heads"
        )));
    }
    if ple.ple_embed_dim < 1 {
        return Err(Gemma4Error::Config(format!(
            "ple_embed_dim {} is not positive",
            ple.ple_embed_dim
        )));
    }
    if ple.ple_embed_dim % heads != 0 {
        return Err(Gemma4Error::Config(format!(
            "ple_embed_dim {} is not divisible by {heads} hash heads",
            ple.ple_embed_dim
        )));
    }
    Ok((hyper_connections, compressed_attention, ple))
}

/// Parses `vision_config` into a [`VisionConfig`], or `NONE` when the
/// checkpoint declares no tower (ROADMAP M-V3).
///
/// **THIS IS A SEPARATE PARSE FROM [`parse_qwen_gdn_dense_config`] AND THE
/// SEPARATION IS THE DESIGN, NOT A CONVENIENCE.** `ArchConfig.vision`
/// describes what an INSTALL carries; this function describes what a
/// CHECKPOINT declares. Folding it into the family parser would break both
/// ends at once:
///
/// - `arch_validation` compares a manifest field by field against a
///   per-architecture baseline, and every baseline carries
///   `VisionConfig::NONE`. An `ArchConfig` that read 27 blocks off the config
///   would compare 27 against 0 for every EXISTING install on disk, none of
///   which declares a `visionDepth` at all -- AGENTS.md Gotcha 24 exactly,
///   and it would refuse every qwen35 and qwen36 install ever written.
/// - `every_published_checkpoint_parses_to_one_baseline` and its Ornith
///   siblings assert `derived == baseline` on the whole struct, so the two
///   would have to disagree about a field neither is wrong about.
///
/// **AND THE ARTIFACT DOES NOT FOLLOW THE CONFIG, WHICH IS WHAT SETTLES IT.**
/// All three `qwen3_5` checkpoints plus BOTH Ornith releases declare the
/// identical tower -- depth 27, hidden 1152, intermediate 4304,
/// `num_position_embeddings` 2304, `mrope_section` [11, 11, 10] and the same
/// four token ids, verified against the published files rather than inferred.
/// But `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` ships NO `vision_tower.`
/// tensors at all. So a config-derived tower would mark that install as
/// carrying one, `validate_manifest` would then demand `packed_vision/` files
/// the walk never wrote, and a working install would stop opening. It is the
/// same "DECLARED AND UNSHIPPED" split `ornith_config.rs` already records for
/// `mtp.*`, and the same answer the MTP head reached: what an install has is
/// answered by its BYTES, so nothing can disagree with them.
///
/// A PARTIAL block is refused rather than defaulted. Every field is a shape
/// some kernel strides by, and a default would be this port inventing a number
/// the checkpoint declined to state (AGENTS.md Gotcha 39). `mrope_section` is
/// the one exception in SOURCE rather than in strictness: it is not in
/// `vision_config` at all but in `text_config.rope_parameters`, because it
/// describes how the TRUNK consumes an image's positions rather than anything
/// the tower computes.
pub fn parse_vision_config(json: &str) -> Result<VisionConfig, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    let tc = root.get("text_config").unwrap_or(&root);
    let root = &root;
    let Some(vc) = root.get("vision_config") else {
        return Ok(VisionConfig::NONE);
    };
    let i = |k: &str| -> Result<i64, Gemma4Error> {
        vc.get(k)
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| Gemma4Error::Config(format!("missing vision_config.{k}")))
    };

    // From the TRUNK's rope block, not the tower's. Required whenever a tower
    // exists: without it the trunk cannot place an image token's three
    // positions, and mRoPE degenerating to plain RoPE is precisely the silent
    // wrong answer (`docs/VISION_PHASE0.md` item 2).
    let section = tc
        .get("rope_parameters")
        .and_then(|r| r.get("mrope_section"))
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            Gemma4Error::Config(
                "a vision_config with no text_config.rope_parameters.mrope_section; the trunk \
                 cannot place image tokens without it"
                    .to_string(),
            )
        })?;
    if section.len() != 3 {
        return Err(Gemma4Error::Config(format!(
            "mrope_section has {} entries, expected the (t, h, w) triple",
            section.len()
        )));
    }
    let mut mrope_section = [0i64; 3];
    for (slot, v) in mrope_section.iter_mut().zip(section) {
        *slot = v
            .as_i64()
            .ok_or_else(|| Gemma4Error::Config("mrope_section entry is not an integer".into()))?;
    }

    let vision = VisionConfig {
        depth: i("depth")?,
        hidden_size: i("hidden_size")?,
        intermediate_size: i("intermediate_size")?,
        num_heads: i("num_heads")?,
        patch_size: i("patch_size")?,
        temporal_patch_size: i("temporal_patch_size")?,
        in_channels: i("in_channels")?,
        spatial_merge_size: i("spatial_merge_size")?,
        num_position_embeddings: i("num_position_embeddings")?,
        out_hidden_size: i("out_hidden_size")?,
        mrope_section,
        // Token ids live at the ROOT, beside the wrapper rather than inside
        // either config: they belong to the tokenizer's vocabulary, which the
        // text and vision halves share.
        vision_start_token_id: root_i(root, "vision_start_token_id")?,
        vision_end_token_id: root_i(root, "vision_end_token_id")?,
        image_token_id: root_i(root, "image_token_id")?,
        video_token_id: root_i(root, "video_token_id")?,
    };

    if vision.depth != SUPPORTED_VISION_DEPTH {
        return Err(Gemma4Error::Config(format!(
            "vision depth {} is unsupported; expected {SUPPORTED_VISION_DEPTH}",
            vision.depth
        )));
    }

    // Two consistency checks the fields cannot make individually, both of
    // which produce a wrong STRIDE rather than an error if they fail.
    if vision.num_heads == 0 || vision.hidden_size % vision.num_heads != 0 {
        return Err(Gemma4Error::Config(format!(
            "vision hidden_size {} is not divisible by num_heads {}",
            vision.hidden_size, vision.num_heads
        )));
    }
    // The position table is a SQUARE grid the tower interpolates from, so a
    // non-square count means the grid edge this port derives (`sqrt`) is not
    // the one the checkpoint trained.
    let edge = (vision.num_position_embeddings as f64).sqrt() as i64;
    if edge * edge != vision.num_position_embeddings {
        return Err(Gemma4Error::Config(format!(
            "vision num_position_embeddings {} is not a square; the position table is \
             interpolated from a square grid",
            vision.num_position_embeddings
        )));
    }
    if !vision.is_active() {
        return Err(Gemma4Error::Config(
            "vision_config declares depth 0, which this port cannot tell from an absent \
             tower; a checkpoint with no tower should omit the block"
                .to_string(),
        ));
    }
    Ok(vision)
}

fn root_i(root: &serde_json::Value, key: &str) -> Result<i64, Gemma4Error> {
    root.get(key)
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| Gemma4Error::Config(format!("missing {key} beside vision_config")))
}

fn parse_qwen_family_config(json: &str, family: ModelFamily) -> Result<ArchConfig, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    crate::arch_registry::refuse_foreign_config(&root, family).map_err(Gemma4Error::Config)?;
    // Text-only conversions drop the wrapper; accept both shapes.
    let tc = root.get("text_config").unwrap_or(&root);

    let i = |k: &str| -> Result<i64, Gemma4Error> {
        tc.get(k)
            .and_then(|v| v.as_i64())
            .ok_or_else(|| Gemma4Error::Config(format!("missing {k}")))
    };
    let b = |k: &str| tc.get(k).and_then(|v| v.as_bool()).unwrap_or(false);

    // `layer_types` is what makes a Qwen install a hybrid rather than a
    // plain transformer, and an empty or short mask would still validate
    // structurally while running the wrong block on every layer, so it is
    // required and length-checked rather than defaulted.
    let num_layers = i("num_hidden_layers")?;
    let types = tc
        .get("layer_types")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Gemma4Error::Config("missing layer_types".to_string()))?;
    let mask = types
        .iter()
        .map(|t| match t.as_str() {
            Some("linear_attention") => Ok(MASK_LINEAR),
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

    // Flat here, unlike Gemma's per-attention-kind sub-objects. The
    // partial rotary factor is duplicated at the text_config top level;
    // prefer the rope object and fall back to it.
    let rope = tc.get("rope_parameters");
    let rope_f = |key: &str| rope.and_then(|r| r.get(key)).and_then(|v| v.as_f64());
    let rope_theta = rope_f("rope_theta")
        .ok_or_else(|| Gemma4Error::Config("missing rope_parameters.rope_theta".to_string()))?;
    let prf = rope_f("partial_rotary_factor")
        .or_else(|| tc.get("partial_rotary_factor").and_then(|v| v.as_f64()))
        .ok_or_else(|| Gemma4Error::Config("missing partial_rotary_factor".to_string()))?;

    let head_dim = i("head_dim")?;
    // The NeoX sub-dimension RoPE rotates `rotary_dim / 2` pairs, so an odd
    // or non-integral rotary_dim would silently drop a channel.
    let rotary_dim = prf * head_dim as f64;
    if rotary_dim <= 0.0 || rotary_dim.fract() != 0.0 || (rotary_dim as i64) % 2 != 0 {
        return Err(Gemma4Error::Config(format!(
            "partial_rotary_factor {prf} x head_dim {head_dim} = {rotary_dim}, \
             which is not a positive even integer"
        )));
    }

    let kv_heads = i("num_key_value_heads")?;

    // `qwen4_exp`'s three extra components, and the ONE field it resolves
    // from the family rather than from a key. See `parse_qwen4_exp_config`.
    let qwen4 = family == ModelFamily::Qwen4Exp;
    let (hyper_connections, compressed_attention, ple) = if qwen4 {
        parse_qwen4_extensions(tc)?
    } else {
        (
            HyperConnectionConfig::NONE,
            CompressedAttentionConfig::NONE,
            PleConfig::NONE,
        )
    };
    // Both `qwen3_5` checkpoints carry this key; `qwen4_exp` carries none and
    // gates unconditionally in its reference, so silence means YES here and
    // NO there. Reading it the shared way halves `q_proj` and drops the gate.
    let attn_output_gate = if qwen4 { true } else { b("attn_output_gate") };
    // `output_gate_type` selects the activation the gated DeltaNet output norm
    // applies to `z`. Absent means silu, which is what every family before
    // `qwen4_exp` declares and what the kernel did unconditionally.
    let output_gate_sigmoid = tc
        .get("output_gate_type")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|s| s == "sigmoid");

    // The four fields the two Qwen configs resolve differently. Everything
    // else above and below is shared verbatim.
    let dense = family == ModelFamily::QwenGdnDense;
    let (intermediate_size, moe_intermediate_size, num_experts, top_k_experts) = if dense {
        // A dense config that ALSO declares experts is a contradiction, and
        // reading only half of it produces an `ArchConfig` whose FFN width
        // and expert count disagree -- which validates structurally and
        // dispatches the wrong branch. Refused rather than ignored.
        if let Some(n) = tc.get("num_experts").and_then(|v| v.as_i64()) {
            if n > 0 {
                return Err(Gemma4Error::Config(format!(
                    "a qwen3_5 config declares num_experts {n}; the dense family has none,                      and a checkpoint with experts is qwen3_5_moe"
                )));
            }
        }
        (i("intermediate_size")?, 0, 0, 0)
    } else {
        (
            i("shared_expert_intermediate_size")?,
            i("moe_intermediate_size")?,
            i("num_experts")?,
            i("num_experts_per_tok")?,
        )
    };

    Ok(ArchConfig {
        hidden_size: i("hidden_size")?,
        intermediate_size,
        moe_intermediate_size,
        num_heads: i("num_attention_heads")?,
        num_kv_heads: kv_heads,
        num_full_kv_heads: kv_heads,
        head_dim,
        full_head_dim: head_dim,
        vocab_size: i("vocab_size")?,
        sliding_window: 0,
        final_logit_softcap: 0.0,
        rope_theta,
        full_rope_theta: rope_theta,
        partial_rotary_factor: prf,
        num_layers,
        num_experts,
        top_k_experts,
        tie_word_embeddings: b("tie_word_embeddings"),
        attention_k_eq_v: false,
        full_attention_layer_mask: mask,
        hidden_activation: tc
            .get("hidden_act")
            .and_then(|v| v.as_str())
            .unwrap_or("silu")
            .to_string(),
        family,
        attn_output_gate,
        attention_scale: (head_dim as f64).powf(-0.5),
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: !dense,
        rope_neox_subdim: true,
        linear_attention: LinearAttentionConfig {
            num_k_heads: i("linear_num_key_heads")?,
            num_v_heads: i("linear_num_value_heads")?,
            key_head_dim: i("linear_key_head_dim")?,
            value_head_dim: i("linear_value_head_dim")?,
            conv_kernel_size: i("linear_conv_kernel_dim")?,
            output_gate_sigmoid,
        },
        compressed_attention,
        hyper_connections,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: RopeScalingConfig::NONE,
        // NOT parsed from `vision_config`, though all three published
        // checkpoints declare one. See `parse_vision_config`: this field
        // describes what an INSTALL carries, and the config describes what the
        // architecture has. `qwen4_exp` declares one too and is text-only here
        // for the same reason.
        vision: VisionConfig::NONE,
        // NOT held to `vision`'s rule, and the difference is which side the
        // bytes are on. A tower is OPTIONAL in the artifact -- one published
        // checkpoint declares one and ships none -- so the manifest must state
        // what the install HAS. The n-gram table is not optional: a
        // `qwen4_exp` checkpoint without it has no layer 1, so declaring it
        // from the config states a property of the architecture that the walk
        // then has to satisfy rather than a claim it might contradict.
        ple,
    })
}

#[cfg(test)]
mod vision_depth_tests {
    use super::*;

    fn vision_json(depth: i64) -> String {
        serde_json::json!({
            "text_config": {"rope_parameters": {"mrope_section": [11, 11, 10]}},
            "vision_config": {
                "depth": depth,
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_heads": 16,
                "patch_size": 16,
                "temporal_patch_size": 2,
                "in_channels": 3,
                "spatial_merge_size": 2,
                "num_position_embeddings": 2304,
                "out_hidden_size": 5120
            },
            "vision_start_token_id": 248056,
            "vision_end_token_id": 248057,
            "image_token_id": 248058,
            "video_token_id": 248059
        })
        .to_string()
    }

    #[test]
    fn vision_depth_is_bounded_at_the_untrusted_config_boundary() {
        let error = parse_vision_config(&vision_json(i64::MAX))
            .expect_err("an attacker-controlled depth must be rejected");
        assert_eq!(
            error.to_string(),
            format!(
                "config.json invalid: vision depth {} is unsupported; expected {}",
                i64::MAX,
                27
            )
        );
    }
}
