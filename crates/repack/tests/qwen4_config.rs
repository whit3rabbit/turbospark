// `serde_json::json!` recurses once per key and this fixture carries ~40,
// which is past the default 128. The sibling `qwen35_config.rs` needs no such
// attribute because its `text_config` is smaller -- the depth is a property of
// the checkpoint's key count, not of anything this file does.
#![recursion_limit = "256"]

//! `parse_qwen4_exp_config` against the PRODUCTION field values of BOTH
//! published `qwen4_exp` checkpoints (Qwen3.8-Flash-Next):
//! `pipenetwork/Qwen3.8-Flash-Next-MLX-4bit` at 512 routed experts and
//! `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit`, expert-pruned to 288.
//!
//! Its own target rather than a case in `qwen36_config.rs`, following
//! `qwen35_config.rs`, and the fixture is the real `text_config` verbatim for
//! that file's reason: the whole test reduces to one assertion, that the parse
//! equals `model_io::qwen4_exp_125b_a6b()` field for field. A shape-only
//! fixture cannot catch a key mapped to the wrong field, because every
//! dimension would be some other made-up number either way.
//!
//! **THIS IS THE PHASE 0 GATE FOR THE FAMILY, AND IT COSTS MILLISECONDS
//! AGAINST A 68 GiB STREAM.** AGENTS.md Gotcha 47's workflow: a `config.json`
//! is a few KB over HTTP, so diffing it against every shipped baseline answers
//! "is this new, and is it still what we think" before any of the expensive
//! questions are asked. If a future point release of either checkpoint moves a
//! shape key, THIS is what reddens, rather than a multi-GB stream failing at
//! some tensor offset.
//!
//! **ONE `text_config()` SERVES BOTH CHECKPOINTS AND THEY DIFFER IN
//! `num_experts` ALONE**, read side by side off the published files. That is a
//! stronger statement than the `qwen3_5` trio makes -- there the two that
//! differed (`eos_token_id`, the quantization object) reached no `ArchConfig`
//! field at all, so the configs agreed on everything that mattered. Here the
//! one difference DOES reach a field, and `every_published_checkpoint_parses_
//! to_one_baseline` is what pins that it is the only one.

use model_io::ModelFamily;
use turbospark_repack::{parse_qwen4_exp_config, parse_qwen_gdn_dense_config};

/// 48 layers, gated-DeltaNet everywhere except every 4th
/// (`full_attention_interval = 4`), which reproduces the checkpoint's own
/// `layer_types` list.
fn layer_types() -> Vec<&'static str> {
    (0..48)
        .map(|i| {
            if (i + 1) % 4 == 0 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect()
}

/// The production `text_config`, trimmed to the keys the parser reads plus the
/// mrope and `mtp` fields it must ignore.
///
/// `num_experts` is the one key that differs across the published checkpoints
/// (512 unpruned, 288 after REAP). It is a parameter rather than a constant so
/// the difference is stated, and it is the ONLY parameter -- which is the
/// claim `every_published_checkpoint_parses_to_one_baseline` turns into an
/// assertion.
///
/// Note what is NOT here: `attn_output_gate`. Both `qwen3_5` checkpoints carry
/// that key and this family carries none, while gating unconditionally in its
/// reference. Its absence from this fixture is deliberate and load-bearing --
/// see `the_attention_output_gate_is_a_family_constant_not_a_key`.
fn text_config_with_experts(num_experts: u32) -> serde_json::Value {
    serde_json::json!({
        "bos_token_id": 248_044,
        "eos_token_id": 248_044,
        "full_attention_interval": 4,
        "hc_count": 4,
        "hc_lowrank": 320,
        "head_dim": 256,
        "heads_per_ngram": 8,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
        "indexer_head_dim": 128,
        "indexer_kv_heads": 1,
        "indexer_n_heads": 4,
        "layer_types": layer_types(),
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "make_ngram_vocab_size_divisible_by": 128,
        "max_position_embeddings": 262_144,
        "model_type": "qwen4_exp_text",
        "moe_intermediate_size": 640,
        // Declared and unimplemented, exactly as `qwen3_5`'s is.
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": false,
        "ngram_size": 3,
        "ngram_vocab_size_base": 20_000_000,
        "num_attention_heads": 24,
        "num_experts": num_experts,
        "num_experts_per_tok": 10,
        "num_hidden_layers": 48,
        "num_key_value_heads": 2,
        // SIGMOID here. `qwen3_5` declares `swish` at this same key, which is
        // the discriminating pair `the_output_gate_activation_is_read_off_the_
        // config` rests on.
        "output_gate_type": "sigmoid",
        "partial_rotary_factor": 0.25,
        "ple_conv_kernel_size": 4,
        "ple_embed_dim": 2560,
        // ONE-BASED: this is layer INDEX 1.
        "ple_layer_ids": [2],
        "rms_norm_eps": 1.0e-6,
        "rope_parameters": {
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000_000,
            "rope_type": "default"
        },
        "shared_expert_intermediate_size": 640,
        "split_ngram_parts": 128,
        "tie_word_embeddings": false,
        "vocab_size": 248_320
    })
}

fn config_with_experts(num_experts: u32) -> String {
    serde_json::json!({
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "image_token_id": 248_056,
        "language_model_only": false,
        "model_type": "qwen4_exp",
        "text_config": text_config_with_experts(num_experts),
        "video_token_id": 248_057,
        "vision_config": {"depth": 27},
        "vision_end_token_id": 248_054,
        "vision_start_token_id": 248_053,
        "quantization": {"group_size": 64, "bits": 4}
    })
    .to_string()
}

/// `pipenetwork/Qwen3.8-Flash-Next-MLX-4bit`, the unpruned checkpoint.
fn full_config_json() -> String {
    config_with_experts(512)
}

/// `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit`, expert-pruned to 288.
fn reap_config_json() -> String {
    config_with_experts(288)
}

#[test]
fn the_unpruned_checkpoint_parses_to_the_pinned_baseline() {
    let derived = parse_qwen4_exp_config(&full_config_json()).expect("parses");
    assert_eq!(derived, model_io::qwen4_exp_125b_a6b());
}

/// The claim the fixture's single parameter encodes, stated as an assertion.
///
/// Both published checkpoints reduce to one baseline and the ONLY field that
/// moves is `num_experts`. Checked by patching the baseline's expert count and
/// requiring whole-struct equality, so a second divergence anywhere -- a shape
/// key, a rope term, one of the three new sub-configs -- reddens this rather
/// than passing on a field-by-field spot check that happened not to look.
#[test]
fn every_published_checkpoint_parses_to_one_baseline() {
    let full = parse_qwen4_exp_config(&full_config_json()).expect("parses");
    let reap = parse_qwen4_exp_config(&reap_config_json()).expect("parses");

    assert_eq!(full.num_experts, 512);
    assert_eq!(reap.num_experts, 288);

    let mut expected = model_io::qwen4_exp_125b_a6b();
    assert_eq!(full, expected);
    expected.num_experts = 288;
    assert_eq!(reap, expected);
}

/// **`attn_output_gate` IS A FAMILY CONSTANT HERE AND A CONFIG KEY ON
/// `qwen3_5`, AND READING IT THE SHARED WAY IS SILENTLY WRONG.**
///
/// The fixture carries no `attn_output_gate` because neither published
/// checkpoint does, while the reference sizes `q_proj` at
/// `n_heads * head_dim * 2` and splits `[query; gate]` unconditionally. A
/// parser falling through to `b("attn_output_gate")` reads FALSE, which halves
/// the projection and drops the gate with no error at all.
///
/// The `qwen3_5` half of the pair is what stops the fix from being "just set
/// it true everywhere": that family really does answer from the key.
#[test]
fn the_attention_output_gate_is_a_family_constant_not_a_key() {
    let derived = parse_qwen4_exp_config(&full_config_json()).expect("parses");
    assert!(
        derived.attn_output_gate,
        "qwen4_exp gates unconditionally and declares no key for it"
    );
    assert!(
        !text_config_with_experts(512)
            .as_object()
            .expect("object")
            .contains_key("attn_output_gate"),
        "the fixture must not carry the key, or this case proves nothing"
    );
}

/// The GDN output gate's activation, as a DISCRIMINATING PAIR.
///
/// `crates/gpu`'s Gotcha 12 is that `gdn_gated_norm` hardcodes silu because
/// every family reaching it declared silu. Both halves are asserted here:
/// `qwen4_exp` says `sigmoid` and `qwen3_5` says `swish` at the SAME key, so a
/// parser that hardcoded either answer fails one of the two.
#[test]
fn the_output_gate_activation_is_read_off_the_config() {
    let qwen4 = parse_qwen4_exp_config(&full_config_json()).expect("parses");
    assert!(qwen4.linear_attention.output_gate_sigmoid);

    let qwen35 = parse_qwen_gdn_dense_config(&qwen35_config_json()).expect("parses");
    assert!(
        !qwen35.linear_attention.output_gate_sigmoid,
        "qwen3_5 declares output_gate_type swish, which is silu"
    );
}

/// A minimal real-shaped `qwen3_5` config, for the pair above and the refusal
/// below. Trimmed to what `parse_qwen_gdn_dense_config` reads.
fn qwen35_config_json() -> String {
    serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "attn_output_gate": true,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "layer_types": (0..64)
                .map(|i| if (i + 1) % 4 == 0 { "full_attention" } else { "linear_attention" })
                .collect::<Vec<_>>(),
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "model_type": "qwen3_5_text",
            "num_attention_heads": 24,
            "num_hidden_layers": 64,
            "num_key_value_heads": 4,
            "output_gate_type": "swish",
            "partial_rotary_factor": 0.25,
            "rope_parameters": {"rope_theta": 10_000_000, "partial_rotary_factor": 0.25},
            "tie_word_embeddings": false,
            "vocab_size": 248_320
        }
    })
    .to_string()
}

/// The three sub-configs reach their fields, checked as VALUES rather than as
/// presence.
///
/// Presence is not enough on any of them: every field here is an integer, so a
/// key mapped to the wrong field produces a populated struct holding another
/// key's number. `index_top_k` is the sharpest of the three because it is
/// DERIVED (`indexer_budget / indexer_compress_ratio`) rather than read, and
/// counts BLOCKS where DeepSeek's field of that name counts tokens.
#[test]
fn the_three_new_sub_configs_carry_the_checkpoints_values() {
    let a = parse_qwen4_exp_config(&full_config_json()).expect("parses");

    assert_eq!(a.hyper_connections.mult, 4);
    assert_eq!(a.hyper_connections.lowrank, 320);
    assert!(a.hyper_connections.is_active());
    // DeepSeek mHC's terms, which this mixer does not have.
    assert_eq!(a.hyper_connections.sinkhorn_iters, 0);

    assert_eq!(a.compressed_attention.index_n_heads, 4);
    assert_eq!(a.compressed_attention.index_kv_heads, 1);
    assert_eq!(a.compressed_attention.index_head_dim, 128);
    assert_eq!(a.compressed_attention.index_budget, 2048);
    assert_eq!(a.compressed_attention.csa_compress_rate, 4);
    assert_eq!(a.compressed_attention.index_top_k, 512, "2048 / 4 blocks");
    assert_eq!(a.compressed_attention.sparse_below(), 2048);
    // DeepSeek's MLA terms; this family's full layers are ordinary GQA with a
    // selector in front, so none of them applies.
    assert_eq!(a.compressed_attention.q_lora_rank, 0);

    assert_eq!(a.ple.ngram_size, 3);
    assert_eq!(a.ple.heads_per_ngram, 8);
    assert_eq!(a.ple.ngram_heads(), 16, "(3 - 1) * 8");
    assert_eq!(a.ple.ple_embed_dim, 2560);
    assert_eq!(a.ple.head_dim(), 160, "2560 / 16");
    assert_eq!(a.ple.split_ngram_parts, 128);
    assert_eq!(a.ple.conv_kernel_size, 4);
    assert!(a.ple.is_active());
    // ONE-BASED as the checkpoint spells it, ZERO-BASED for a layer loop. The
    // pair is asserted because reading either as the other puts the PLE block
    // on the wrong layer, which is fluent and wrong.
    assert_eq!(a.ple.layer_ids, vec![2]);
    assert_eq!(a.ple.layer_indices(), vec![1]);
    assert_eq!(a.ple.seed, 1234, "transformers' default; no config key");
}

/// The layer mask, and that it is the hybrid rather than all-full.
///
/// 36 linear to 12 full at every 4th, which is what makes this family reach
/// `families/qwen/`'s gated-DeltaNet kernels at all.
#[test]
fn the_layer_mask_is_three_linear_to_one_full() {
    let a = parse_qwen4_exp_config(&full_config_json()).expect("parses");
    assert_eq!(a.full_attention_layer_mask.len(), 48);
    assert_eq!(
        a.full_attention_layer_mask
            .iter()
            .filter(|m| **m == 2)
            .count(),
        36
    );
    assert_eq!(
        a.full_attention_layer_mask
            .iter()
            .filter(|m| **m == 1)
            .count(),
        12
    );
    // Layer 0 is LINEAR, which is why the manifest quant probe reads
    // `linear_attn.in_proj_qkv` and not `self_attn.q_proj`.
    assert_eq!(a.full_attention_layer_mask[0], 2);
    assert_eq!(a.full_attention_layer_mask[3], 1);
    assert_eq!(a.family, ModelFamily::Qwen4Exp);
}

/// A `qwen3_5` config must not parse as `qwen4_exp` and vice versa.
///
/// The strings are far apart (`qwen4_exp` shares no prefix with `qwen3_5`), so
/// this is not the near-miss `arch_registry` warns about between the two
/// `qwen3_5` rows. What makes it worth pinning is the opposite: the two
/// configs share almost every KEY, so a `qwen3_5` file fed to this parser gets
/// all the way to the extension block before anything is missing, and would
/// otherwise fail with "missing hc_count" rather than naming the family.
#[test]
fn the_two_qwen_configs_refuse_each_other() {
    let err = parse_qwen4_exp_config(&qwen35_config_json()).expect_err("refused");
    let msg = err.to_string();
    // **THE MESSAGE MUST QUOTE THE FILE.** It used to print the resolved
    // family's WIRE STRING under the label `model_type`, which is a frozen
    // on-disk constant no config contains (`qwen35` for a file that says
    // `qwen3_5`), so a reader grepping the config for the token in the error
    // found nothing. Asserting the real `model_type` appears is what keeps
    // that from coming back.
    assert!(
        msg.contains("qwen3_5"),
        "the refusal should quote the config's own model_type, got: {msg}"
    );

    parse_qwen_gdn_dense_config(&full_config_json()).expect_err("refused");
}

/// Each extension block is REQUIRED once the family is known, and a missing
/// key is a refusal rather than a default.
///
/// Every field is a shape some kernel strides by or a threshold a refusal
/// quotes, so defaulting one would be this port inventing a number the
/// checkpoint declined to state (AGENTS.md Gotcha 39). Three keys, one per
/// block, so a block dropped wholesale is caught as well as a key.
#[test]
fn a_missing_extension_key_is_refused_rather_than_defaulted() {
    for key in ["hc_count", "indexer_budget", "ngram_size", "ple_layer_ids"] {
        let mut root: serde_json::Value =
            serde_json::from_str(&full_config_json()).expect("valid json");
        root["text_config"]
            .as_object_mut()
            .expect("object")
            .remove(key)
            .unwrap_or_else(|| {
                panic!("{key} must be in the fixture for this case to mean anything")
            });
        let err = parse_qwen4_exp_config(&root.to_string()).expect_err("refused");
        assert!(
            err.to_string().contains(key),
            "dropping {key} should be refused by name, got: {err}"
        );
    }
}

/// Two consistency checks the individual fields cannot make, both of which
/// produce a wrong STRIDE rather than an error if they fail.
#[test]
fn an_inconsistent_ple_or_indexer_block_is_refused() {
    // `ple_embed_dim` must divide evenly across the hash heads: the reference
    // computes `head_dim = embed_dim // ngram_heads`, so a remainder silently
    // truncates every row.
    let mut root: serde_json::Value =
        serde_json::from_str(&full_config_json()).expect("valid json");
    root["text_config"]["ple_embed_dim"] = serde_json::json!(2561);
    parse_qwen4_exp_config(&root.to_string()).expect_err("indivisible embed dim refused");

    let mut root: serde_json::Value =
        serde_json::from_str(&full_config_json()).expect("valid json");
    root["text_config"]["ple_embed_dim"] = serde_json::json!(0);
    parse_qwen4_exp_config(&root.to_string()).expect_err("zero embed dim refused");

    // The budget must be a whole number of blocks, or the derived block top-k
    // does not describe the budget the checkpoint declares.
    let mut root: serde_json::Value =
        serde_json::from_str(&full_config_json()).expect("valid json");
    root["text_config"]["indexer_budget"] = serde_json::json!(2049);
    parse_qwen4_exp_config(&root.to_string()).expect_err("ragged budget refused");

    // One-based ids, so 0 names no layer and means a different convention.
    let mut root: serde_json::Value =
        serde_json::from_str(&full_config_json()).expect("valid json");
    root["text_config"]["ple_layer_ids"] = serde_json::json!([0]);
    parse_qwen4_exp_config(&root.to_string()).expect_err("zero layer id refused");
}
