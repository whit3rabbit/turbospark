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
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig, MlaConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig, MAX_VISION_DEEPSTACK_MERGERS,
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

/// The vision-tower depths the current Qwen tower kernels and packed layout
/// support: 27 is every published `qwen3_5` / `qwen3_5_moe` tower, 24 is the
/// `qwen3_vl`-4B tower. The tower walk itself is depth-parametrized (blocks
/// stream as one "layer" of `depth` blobs); this list is the set of depths
/// some real checkpoint has been read against, so a tower of an unseen depth
/// refuses at parse rather than streaming into an unpackaged layout.
pub(crate) const SUPPORTED_VISION_DEPTHS: [i64; 2] = [24, 27];

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
/// `vision_config` at all but in the trunk's rope block -- under
/// `text_config.rope_parameters` for the `qwen3_5` conversions and
/// `text_config.rope_scaling` for the HF-native `qwen3_vl` one -- because it
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
    // Qwen2.5-VL spells the tower differently (`depth`, `in_chans`, a window
    // size and a full-attention block list, no position table), so it parses
    // on its own rather than through the qwen3 reader's required keys.
    if root.get("model_type").and_then(serde_json::Value::as_str) == Some("qwen2_5_vl") {
        return parse_qwen25_vl_vision_config(root, vc);
    }
    if root.get("model_type").and_then(serde_json::Value::as_str) == Some("muse_glimmer") {
        return parse_muse_vision_config(root, vc);
    }
    if root.get("model_type").and_then(serde_json::Value::as_str) == Some("minicpmv4_7") {
        return parse_minicpm_v_vision_config(root, vc);
    }
    let i = |k: &str| -> Result<i64, Gemma4Error> {
        vc.get(k)
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| Gemma4Error::Config(format!("missing vision_config.{k}")))
    };

    // From the TRUNK's rope block, not the tower's. Required whenever a tower
    // exists: without it the trunk cannot place an image token's three
    // positions, and mRoPE degenerating to plain RoPE is precisely the silent
    // wrong answer (`docs/VISION_PHASE0.md` item 2).
    //
    // TWO PUBLISHER SPELLINGS of the same block, read like
    // `canonicalize_vision_header`'s two tower spellings: the `qwen3_5`
    // conversions carry `text_config.rope_parameters.mrope_section` and the
    // HF-native `qwen3_vl` conversion carries
    // `text_config.rope_scaling.mrope_section`. A tower whose trunk declares
    // neither is refused by name.
    let section = tc
        .get("rope_parameters")
        .and_then(|r| r.get("mrope_section"))
        .or_else(|| tc.get("rope_scaling").and_then(|r| r.get("mrope_section")))
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            Gemma4Error::Config(
                "a vision_config with no text_config.rope_parameters.mrope_section or \
                 text_config.rope_scaling.mrope_section; the trunk cannot place image tokens \
                 without it"
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

    // Optional, and ABSENT MEANS EMPTY: the reference config's own default
    // is `default_factory=list` (`qwen3_vl/config.py`), so silence is the
    // format's answer and not a sibling's (AGENTS.md Gotcha 39). Every
    // published `qwen3_5` tower declares the key with `[]`.
    let deepstack_visual_indexes = match vc.get("deepstack_visual_indexes") {
        Some(v) => {
            let list = v.as_array().ok_or_else(|| {
                Gemma4Error::Config("vision_config.deepstack_visual_indexes is not an array".into())
            })?;
            let mut indexes = Vec::with_capacity(list.len());
            for entry in list {
                let idx = entry.as_i64().ok_or_else(|| {
                    Gemma4Error::Config(
                        "vision_config.deepstack_visual_indexes holds a non-integer".into(),
                    )
                })?;
                indexes.push(idx);
            }
            indexes
        }
        None => Vec::new(),
    };

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
        deepstack_visual_indexes,
        // Token ids live at the ROOT, beside the wrapper rather than inside
        // either config: they belong to the tokenizer's vocabulary, which the
        // text and vision halves share.
        vision_start_token_id: root_i(root, "vision_start_token_id")?,
        vision_end_token_id: root_i(root, "vision_end_token_id")?,
        image_token_id: root_i(root, "image_token_id")?,
        video_token_id: root_i(root, "video_token_id")?,
        tower: model_io::VisionTowerKind::QwenSiglip,
    };

    if !SUPPORTED_VISION_DEPTHS.contains(&vision.depth) {
        return Err(Gemma4Error::Config(format!(
            "vision depth {} is unsupported; expected one of {SUPPORTED_VISION_DEPTHS:?}",
            vision.depth
        )));
    }
    if vision.deepstack_visual_indexes.len() > MAX_VISION_DEEPSTACK_MERGERS {
        return Err(Gemma4Error::Config(format!(
            "deepstack_visual_indexes has {} entries; at most \
             {MAX_VISION_DEEPSTACK_MERGERS} are supported",
            vision.deepstack_visual_indexes.len()
        )));
    }
    // Every index names a block of THIS tower, and no block feeds two
    // mergers: a repeated index would run the same features through two
    // different mergers into two different trunk layers, which no published
    // checkpoint does and no reference implementation could mean.
    let mut seen = std::collections::BTreeSet::new();
    for &idx in &vision.deepstack_visual_indexes {
        if idx < 0 || idx >= vision.depth {
            return Err(Gemma4Error::Config(format!(
                "deepstack_visual_indexes holds {idx}, outside this tower's depth of {}",
                vision.depth
            )));
        }
        if !seen.insert(idx) {
            return Err(Gemma4Error::Config(format!(
                "deepstack_visual_indexes holds {idx} twice"
            )));
        }
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

/// Blocks in the one published Muse-Glimmer tower.
pub(crate) const SUPPORTED_MUSE_VISION_DEPTHS: [i64; 1] = [50];

/// `<|image_start|>` and `<|image_end|>` in the pinned Muse-Glimmer
/// tokenizer (`tokenizer.json` `added_tokens` 200080 and 200081). The
/// processor wraps every image's placeholder run in them but the checkpoint's
/// `config.json` names neither, so they are constants of the tower kind,
/// read off the pinned revision's tokenizer.
pub const MUSE_IMAGE_START_TOKEN_ID: i64 = 200080;
pub const MUSE_IMAGE_END_TOKEN_ID: i64 = 200081;

/// Window edge in PATCHES. The reference windows by `pos_emb_height`, which
/// every published checkpoint declares as 32.
pub(crate) const MUSE_VISION_WINDOW_PATCHES: i64 = 32;

/// Muse-Glimmer's full-attention rule: every fourth block, and the last.
pub(crate) fn muse_full_attention_block(block: i64, depth: i64) -> bool {
    (block + 1) % 4 == 0 || block == depth - 1
}

/// `vision_config` of a `muse_glimmer` checkpoint (read off
/// `mlx-community/Muse-Glimmer-30B-4bit` @
/// `3e7677d7a40d348a3daba263a2b1c0aa41910710`,
/// `docs/MUSEGLIMMER_VISION_PHASE0.md`).
///
/// The kind's constants are checked, not stored: the 32-patch window, the
/// 32x32 position table, LayerNorm eps 1e-5, rope theta 10000, exact GELU,
/// patch 14 over 2 frames and 3 channels, and the block-by-block
/// window/full pattern, which the config lists explicitly and the parser
/// compares against the rule so a revision that moves one layer is refused
/// instead of run with the wrong attention.
fn parse_muse_vision_config(
    root: &serde_json::Value,
    vc: &serde_json::Value,
) -> Result<VisionConfig, Gemma4Error> {
    let i = |k: &str| -> Result<i64, Gemma4Error> {
        vc.get(k)
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| Gemma4Error::Config(format!("missing vision_config.{k}")))
    };
    let text_hidden = root
        .get("text_config")
        .and_then(|t| t.get("hidden_size"))
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| Gemma4Error::Config("missing text_config.hidden_size".into()))?;
    let depth = i("num_hidden_layers")?;
    let (pos_h, pos_w) = (i("pos_emb_height")?, i("pos_emb_width")?);
    let vision = VisionConfig {
        depth,
        hidden_size: i("hidden_size")?,
        intermediate_size: i("intermediate_size")?,
        num_heads: i("num_attention_heads")?,
        patch_size: i("patch_size")?,
        temporal_patch_size: i("patch_temporal")?,
        in_channels: 3,
        spatial_merge_size: i("merge_size")?,
        num_position_embeddings: pos_h * pos_w,
        // The projection writes straight into the trunk's residual stream.
        out_hidden_size: text_hidden,
        mrope_section: [0, 0, 0],
        deepstack_visual_indexes: Vec::new(),
        vision_start_token_id: MUSE_IMAGE_START_TOKEN_ID,
        vision_end_token_id: MUSE_IMAGE_END_TOKEN_ID,
        image_token_id: root_i(root, "image_token_id")?,
        video_token_id: root_i(root, "video_token_id")?,
        tower: model_io::VisionTowerKind::MuseGlimmer,
    };
    if !SUPPORTED_MUSE_VISION_DEPTHS.contains(&vision.depth) {
        return Err(Gemma4Error::Config(format!(
            "muse vision depth {} is unsupported; expected one of \
             {SUPPORTED_MUSE_VISION_DEPTHS:?}",
            vision.depth
        )));
    }
    if vision.patch_size != 14
        || vision.temporal_patch_size != 2
        || vision.spatial_merge_size != 2
        || pos_h != MUSE_VISION_WINDOW_PATCHES
        || pos_w != MUSE_VISION_WINDOW_PATCHES
        || i("max_position_embeddings")? != pos_h * pos_w
    {
        return Err(Gemma4Error::Config(format!(
            "muse vision geometry patch {} x temporal {} x merge {}, table {pos_h}x{pos_w} is \
             not the 14 x 2 x 2, 32x32 every published checkpoint declares",
            vision.patch_size, vision.temporal_patch_size, vision.spatial_merge_size
        )));
    }
    if vision.num_heads <= 0
        || vision.hidden_size % vision.num_heads != 0
        || (vision.hidden_size / vision.num_heads) % 4 != 0
    {
        return Err(Gemma4Error::Config(format!(
            "muse vision hidden_size {} over {} heads must give a head_dim divisible by 4",
            vision.hidden_size, vision.num_heads
        )));
    }
    if vc.get("hidden_act").and_then(serde_json::Value::as_str) != Some("gelu") {
        return Err(Gemma4Error::Config(
            "muse vision_config.hidden_act must be gelu (the exact erf form)".into(),
        ));
    }
    let eps = vc
        .get("layer_norm_eps")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(f64::NAN);
    if (eps - 1e-5).abs() > 1e-12 {
        return Err(Gemma4Error::Config(format!(
            "muse vision layer_norm_eps {eps} is not the 1e-5 this port encodes"
        )));
    }
    let theta = vc
        .get("rope_parameters")
        .and_then(|r| r.get("rope_theta"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(f64::NAN);
    if (theta - 10000.0).abs() > 1e-9 {
        return Err(Gemma4Error::Config(format!(
            "muse vision rope_theta {theta} is not the 10000 this port encodes"
        )));
    }
    let types = vc
        .get("layer_types")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Gemma4Error::Config("missing vision_config.layer_types".into()))?;
    let want: Vec<&str> = (0..depth)
        .map(|b| {
            if muse_full_attention_block(b, depth) {
                "full_attention"
            } else {
                "window_attention"
            }
        })
        .collect();
    let got: Vec<&str> = types.iter().filter_map(serde_json::Value::as_str).collect();
    if got != want {
        return Err(Gemma4Error::Config(
            "muse vision layer_types is not the every-fourth-block-plus-last full-attention \
             pattern this port encodes"
                .into(),
        ));
    }
    // The adapter reads merge^2 patch rows, so the root's out_hidden_size
    // (the adapter input) is a cross-check on the shuffle width.
    let adapter_in = root
        .get("out_hidden_size")
        .and_then(serde_json::Value::as_i64);
    if adapter_in.is_some_and(|w| w != vision.hidden_size * vision.spatial_merge_size.pow(2)) {
        return Err(Gemma4Error::Config(format!(
            "muse out_hidden_size {} is not hidden {} x merge^2",
            adapter_in.unwrap_or(0),
            vision.hidden_size
        )));
    }
    Ok(vision)
}

/// Blocks in every published Qwen2.5-VL tower (3B, 7B, 32B and 72B all
/// declare depth 32).
pub(crate) const SUPPORTED_QWEN25_VISION_DEPTHS: [i64; 1] = [32];

/// The Qwen2.5-VL attention window in pixels, which every published
/// checkpoint declares as 112. A property of the tower kind rather than a
/// stored field (`VisionTowerKind::Qwen25Vl`), so a checkpoint declaring
/// another value is refused here instead of being run with the wrong windows.
pub(crate) const QWEN25_VISION_WINDOW_SIZE: i64 = 112;

/// Every published Qwen2.5-VL tower runs full attention on blocks 7, 15, 23
/// and 31 and window attention on the rest: every eighth block, counting
/// from the eighth.
pub(crate) fn qwen25_full_attention_blocks(depth: i64) -> Vec<i64> {
    (0..depth).filter(|b| b % 8 == 7).collect()
}

/// `vision_config` of a `qwen2_5_vl` checkpoint (read off
/// `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`, `docs/QWEN25_VL_VISION_PHASE0.md`).
///
/// Everything the kernels stride by is read from the file and everything the
/// file cannot vary is checked against the kind's constants: patch 14, two
/// frames per patch, a 2x2 merge, three channels, SiLU, the 112-pixel window
/// and the every-eighth-block full-attention pattern. A checkpoint that
/// disagrees is a tower this port has not seen, and the tower would run with
/// the wrong windows and answer fluently.
fn parse_qwen25_vl_vision_config(
    root: &serde_json::Value,
    vc: &serde_json::Value,
) -> Result<VisionConfig, Gemma4Error> {
    let i = |k: &str| -> Result<i64, Gemma4Error> {
        vc.get(k)
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| Gemma4Error::Config(format!("missing vision_config.{k}")))
    };
    let section = root
        .get("rope_scaling")
        .and_then(|r| r.get("mrope_section"))
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            Gemma4Error::Config(
                "a qwen2_5_vl vision_config with no rope_scaling.mrope_section; the trunk \
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
        in_channels: i("in_chans")?,
        spatial_merge_size: i("spatial_merge_size")?,
        // No learned position table: the tower is positioned by rope alone.
        num_position_embeddings: 0,
        out_hidden_size: i("out_hidden_size")?,
        mrope_section,
        deepstack_visual_indexes: Vec::new(),
        vision_start_token_id: root_i(root, "vision_start_token_id")?,
        vision_end_token_id: root_i(root, "vision_end_token_id")?,
        image_token_id: root_i(root, "image_token_id")?,
        video_token_id: root_i(root, "video_token_id")?,
        tower: model_io::VisionTowerKind::Qwen25Vl,
    };
    if !SUPPORTED_QWEN25_VISION_DEPTHS.contains(&vision.depth) {
        return Err(Gemma4Error::Config(format!(
            "qwen2_5_vl vision depth {} is unsupported; expected one of \
             {SUPPORTED_QWEN25_VISION_DEPTHS:?}",
            vision.depth
        )));
    }
    if vision.patch_size != 14
        || vision.temporal_patch_size != 2
        || vision.spatial_merge_size != 2
        || vision.in_channels != 3
    {
        return Err(Gemma4Error::Config(format!(
            "qwen2_5_vl tower geometry patch {} x temporal {} x merge {} x {} channels is not \
             the 14 x 2 x 2 x 3 every published checkpoint declares",
            vision.patch_size,
            vision.temporal_patch_size,
            vision.spatial_merge_size,
            vision.in_channels
        )));
    }
    if vision.num_heads <= 0 || vision.hidden_size % vision.num_heads != 0 {
        return Err(Gemma4Error::Config(format!(
            "qwen2_5_vl vision hidden_size {} is not divisible by num_heads {}",
            vision.hidden_size, vision.num_heads
        )));
    }
    if (vision.hidden_size / vision.num_heads) % 4 != 0 {
        return Err(Gemma4Error::Config(format!(
            "qwen2_5_vl vision head_dim {} must be a multiple of 4 for the 2-D rope",
            vision.hidden_size / vision.num_heads
        )));
    }
    if vc.get("hidden_act").and_then(serde_json::Value::as_str) != Some("silu") {
        return Err(Gemma4Error::Config(
            "qwen2_5_vl vision_config.hidden_act must be silu".into(),
        ));
    }
    if i("window_size")? != QWEN25_VISION_WINDOW_SIZE {
        return Err(Gemma4Error::Config(format!(
            "qwen2_5_vl vision window_size {} is not the {QWEN25_VISION_WINDOW_SIZE} every \
             published checkpoint declares",
            i("window_size")?
        )));
    }
    let full: Vec<i64> = vc
        .get("fullatt_block_indexes")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Gemma4Error::Config("missing vision_config.fullatt_block_indexes".into()))?
        .iter()
        .map(|v| v.as_i64())
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            Gemma4Error::Config("vision_config.fullatt_block_indexes holds a non-integer".into())
        })?;
    if full != qwen25_full_attention_blocks(vision.depth) {
        return Err(Gemma4Error::Config(format!(
            "qwen2_5_vl fullatt_block_indexes {full:?} is not the every-eighth-block pattern {:?} \
             this port encodes",
            qwen25_full_attention_blocks(vision.depth)
        )));
    }
    // The merger writes straight into the trunk's residual stream.
    let trunk_hidden = root.get("hidden_size").and_then(serde_json::Value::as_i64);
    if trunk_hidden.is_some_and(|h| h != vision.out_hidden_size) {
        return Err(Gemma4Error::Config(format!(
            "qwen2_5_vl vision out_hidden_size {} does not match the trunk hidden_size {}",
            vision.out_hidden_size,
            trunk_hidden.unwrap_or(0)
        )));
    }
    Ok(vision)
}

/// The MiniCPM-V 4.7 tower's constants, checked against the checkpoint rather
/// than stored (`VisionTowerKind::MiniCpmV`). Read off
/// `mlx-community/MiniCPM-V-4.7-35B-A3B-mxfp4` and mlx-vlm's
/// `minicpmv4_6`/`minicpmv4_7` source (`docs/MINICPMV_VISION_PHASE0.md`).
pub const MINICPM_V_DEPTH: i64 = 27;
/// The encoder block after which the 2x2 window attention-merger runs.
pub const MINICPM_V_INSERT_LAYER: i64 = 6;
/// Per-axis downsample of the default `16x` mode: the window merger (2) times
/// the final merger (2).
pub const MINICPM_V_MERGE: i64 = 4;
/// `<slice>` / `</slice>` and the `\n` row separator. Slicing is not built, so
/// these only decide which marker tokens the canvas rope absorbs around a
/// thumbnail.
pub const MINICPM_V_SLICE_START_ID: i64 = 248088;
pub const MINICPM_V_SLICE_END_ID: i64 = 248089;
pub const MINICPM_V_NEWLINE_ID: i64 = 198;

/// `config.json` of a `minicpmv4_7` wrapper. Everything that varies is read
/// from the file; everything the 4.7 reference hardcodes (the 27-block SigLIP,
/// the insert layer, the `16x` mode, the 2x2 mergers, the canvas rope, the
/// marker ids) is checked, and a checkpoint that disagrees is refused rather
/// than run with the wrong merger geometry.
fn parse_minicpm_v_vision_config(
    root: &serde_json::Value,
    vc: &serde_json::Value,
) -> Result<VisionConfig, Gemma4Error> {
    let i = |k: &str| -> Result<i64, Gemma4Error> {
        vc.get(k)
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| Gemma4Error::Config(format!("missing vision_config.{k}")))
    };
    let tc = root.get("text_config").unwrap_or(root);
    let section = tc
        .get("rope_parameters")
        .and_then(|r| r.get("mrope_section"))
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            Gemma4Error::Config(
                "a minicpmv4_7 config with no text_config.rope_parameters.mrope_section".into(),
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
    let patch = i("patch_size")?;
    let image_size = i("image_size")?;
    if patch <= 0 || image_size % patch != 0 {
        return Err(Gemma4Error::Config(format!(
            "minicpmv4_7 vision image_size {image_size} is not a multiple of patch_size {patch}"
        )));
    }
    let side = image_size / patch;
    let vision = VisionConfig {
        depth: i("num_hidden_layers")?,
        hidden_size: i("hidden_size")?,
        intermediate_size: i("intermediate_size")?,
        num_heads: i("num_attention_heads")?,
        patch_size: patch,
        temporal_patch_size: 1,
        in_channels: i("num_channels")?,
        spatial_merge_size: MINICPM_V_MERGE,
        num_position_embeddings: side * side,
        out_hidden_size: tc
            .get("hidden_size")
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| Gemma4Error::Config("missing text_config.hidden_size".into()))?,
        mrope_section,
        deepstack_visual_indexes: Vec::new(),
        vision_start_token_id: root_i(root, "image_start_id")?,
        vision_end_token_id: root_i(root, "image_end_id")?,
        image_token_id: root_i(root, "image_token_id")?,
        video_token_id: root_i(root, "video_token_id")?,
        tower: model_io::VisionTowerKind::MiniCpmV,
    };
    let refuse = |what: String| Err(Gemma4Error::Config(format!("minicpmv4_7: {what}")));
    if vision.depth != MINICPM_V_DEPTH
        || vision.patch_size != 14
        || vision.in_channels != 3
        || vision.num_heads <= 0
        || vision.hidden_size % vision.num_heads != 0
    {
        return refuse(format!(
            "tower geometry depth {} patch {} channels {} heads {} hidden {} is not the \
             27-block, patch-14 SigLIP the reference runs",
            vision.depth,
            vision.patch_size,
            vision.in_channels,
            vision.num_heads,
            vision.hidden_size
        ));
    }
    if vc.get("hidden_act").and_then(serde_json::Value::as_str) != Some("gelu_pytorch_tanh") {
        return refuse("vision hidden_act must be gelu_pytorch_tanh".into());
    }
    if vc
        .get("layer_norm_eps")
        .and_then(serde_json::Value::as_f64)
        .map(|e| (e - 1e-6).abs() > 1e-12)
        .unwrap_or(true)
    {
        return refuse("vision layer_norm_eps must be 1e-6".into());
    }
    let root_i64 = |k: &str| root.get(k).and_then(serde_json::Value::as_i64);
    if root_i64("insert_layer_id") != Some(MINICPM_V_INSERT_LAYER) {
        return refuse(format!(
            "insert_layer_id {:?} is not the {MINICPM_V_INSERT_LAYER} the window merger is built for",
            root_i64("insert_layer_id")
        ));
    }
    if root
        .get("downsample_mode")
        .and_then(serde_json::Value::as_str)
        != Some("16x")
    {
        return refuse("only the default 16x downsample mode is built".into());
    }
    if root.get("mrope_mode").and_then(serde_json::Value::as_str) != Some("canvas") {
        return refuse("mrope_mode must be canvas".into());
    }
    for key in ["merge_kernel_size", "window_kernel_size"] {
        for holder in [root, vc] {
            if let Some(v) = holder.get(key) {
                if v != &serde_json::json!([2, 2]) {
                    return refuse(format!(
                        "{key} {v} is not the 2x2 the mergers are built for"
                    ));
                }
            }
        }
    }
    if root_i64("merger_times").is_some_and(|t| t != 1) {
        return refuse("merger_times must be 1".into());
    }
    for (key, want) in [
        ("slice_start_id", MINICPM_V_SLICE_START_ID),
        ("slice_end_id", MINICPM_V_SLICE_END_ID),
        ("newline_id", MINICPM_V_NEWLINE_ID),
    ] {
        if root_i64(key) != Some(want) {
            return refuse(format!(
                "{key} {:?} is not the {want} the canvas rope absorbs",
                root_i64(key)
            ));
        }
    }
    if mrope_section.iter().sum::<i64>() <= 0 {
        return refuse("mrope_section is empty".into());
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
        dense_lead_intermediate_size: 0,
        num_dense_leading_layers: 0,
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
        mla: MlaConfig::NONE,
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
        short_conv_kernel_size: 0,
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

    fn qwen3vl_vision_json(depth: i64, deepstack: serde_json::Value) -> String {
        serde_json::json!({
            "text_config": {"rope_parameters": {"mrope_section": [24, 20, 20]}},
            "vision_config": {
                "depth": depth,
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "num_heads": 16,
                "patch_size": 16,
                "temporal_patch_size": 2,
                "in_channels": 3,
                "spatial_merge_size": 2,
                "num_position_embeddings": 2304,
                "out_hidden_size": 2560,
                "deepstack_visual_indexes": deepstack
            },
            "vision_start_token_id": 151652,
            "vision_end_token_id": 151653,
            "image_token_id": 151655,
            "video_token_id": 151656
        })
        .to_string()
    }

    #[test]
    fn the_qwen3vl_tower_parses_with_its_deepstack_indexes() {
        let vision = parse_vision_config(&qwen3vl_vision_json(24, serde_json::json!([5, 11, 17])))
            .expect("the 4B tower's config parses");
        assert_eq!(vision.depth, 24);
        assert_eq!(vision.hidden_size, 1024);
        assert_eq!(vision.out_hidden_size, 2560);
        assert_eq!(vision.mrope_section, [24, 20, 20]);
        assert_eq!(vision.deepstack_visual_indexes, vec![5, 11, 17]);
        assert_eq!(vision.merger_input_dim(), 4096);
    }

    #[test]
    fn an_absent_deepstack_key_means_empty_and_not_a_siblings_answer() {
        // The KEY ABSENT, not `[]`: the reference config's own default is an
        // empty list, so silence is the format's answer (AGENTS.md Gotcha
        // 39). Same value as every qwen3_5 tower that declares `[]`.
        let vision = parse_vision_config(&vision_json(27)).expect("parses");
        assert!(vision.deepstack_visual_indexes.is_empty());
    }

    #[test]
    fn a_deepstack_index_past_the_depth_is_refused() {
        let err = parse_vision_config(&qwen3vl_vision_json(24, serde_json::json!([5, 24])))
            .expect_err("index 24 is past a depth-24 tower");
        assert!(
            err.to_string().contains("outside this tower's depth"),
            "{err}"
        );
    }

    #[test]
    fn a_repeated_deepstack_index_is_refused() {
        let err = parse_vision_config(&qwen3vl_vision_json(24, serde_json::json!([5, 5])))
            .expect_err("two mergers reading one block is ambiguous");
        assert!(err.to_string().contains("twice"), "{err}");
    }

    #[test]
    fn too_many_deepstack_mergers_are_refused() {
        let err = parse_vision_config(&qwen3vl_vision_json(24, serde_json::json!([2, 5, 11, 17])))
            .expect_err("deepstack retention must have a fixed allocation bound");
        assert!(err.to_string().contains("at most 3 are supported"), "{err}");
    }

    #[test]
    fn vision_depth_is_bounded_at_the_untrusted_config_boundary() {
        let error = parse_vision_config(&vision_json(i64::MAX))
            .expect_err("an attacker-controlled depth must be rejected");
        assert_eq!(
            error.to_string(),
            format!(
                "config.json invalid: vision depth {} is unsupported; expected one of [24, 27]",
                i64::MAX
            )
        );
    }
}
