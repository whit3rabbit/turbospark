//! Ornith-1.5's two text checkpoints against the PINNED baselines, offline.
//!
//! **THE HEADLINE IS THAT NEITHER IS A NEW FAMILY, AND THIS FILE IS WHERE
//! THAT CLAIM LIVES** -- AGENTS.md Gotcha 47's workflow (diff `config.json`
//! against every shipped baseline before budgeting a bring-up), turned into a
//! millisecond assertion so a future point release reddens here rather than
//! failing a 21 GB stream at some tensor offset.
//!
//! | | `ornith-ai/Ornith-1.5-35B-A3B` | `ornith-ai/Ornith-1.5-9B` |
//! |---|---|---|
//! | `model_type` | `qwen3_5_moe` | `qwen3_5` |
//! | family | `QwenGdnMoe` | `QwenGdnDense` |
//! | baseline | equals `qwen_gdn_moe_35b_a3b()` FIELD FOR FIELD | `qwen_gdn_dense_27b()` behaviourally, new shape |
//!
//! The 35B is the sharper of the two: its `text_config` reproduces the Qwen
//! 3.6 baseline on every field this port reads, so Ornith-1.5 is a retrain of
//! that architecture rather than a new one and NOTHING in `crates/model-io`,
//! `crates/gpu` or `crates/runtime`'s flow moved for it.
//!
//! The 9B is the case `qwen35_config.rs` does not cover: a checkpoint of a
//! shipped family at a SHAPE no shipped baseline carries. It cannot be
//! asserted with one equality, so it is asserted as the PAIR of claims that
//! actually matter -- every behavioural field equals the dense baseline's
//! (which is what licenses reusing `families/qwen/`'s flow) and the shape
//! fields are the published ones. That is the same split
//! `model-io`'s `qwen_gdn_dense_shares_the_moe_flows_behaviour_and_differs_in_shape`
//! makes one crate down.
//!
//! Both bodies are the real `text_config`, trimmed to the keys the parser
//! reads plus the ones it must IGNORE: the mrope pair (settled by reading
//! mlx-lm -- `rope_type: "default"` maps to a plain `nn.RoPE`, so neither
//! field reaches the text path) and `mtp_num_hidden_layers`, which is the
//! subject of its own trap below.

use turbospark_repack::{parse_qwen_gdn_dense_config, parse_qwen_gdn_moe_config};

/// Gated-DeltaNet everywhere except every 4th layer, which is what both
/// checkpoints' own `layer_types` lists spell out at their own depths.
fn layer_types(layers: usize) -> Vec<&'static str> {
    (0..layers)
        .map(|i| {
            if (i + 1) % 4 == 0 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect()
}

/// `ornith-ai/Ornith-1.5-35B-A3B`'s production `text_config`.
///
/// Read off the published file. Every value here is Qwen 3.6's, which is the
/// finding; the keys that are NOT Qwen 3.6's (`max_position_embeddings`,
/// `mtp_num_hidden_layers`, the mrope pair) all reach no `ArchConfig` field.
fn moe_text_config() -> serde_json::Value {
    serde_json::json!({
        "attn_output_gate": true,
        "bos_token_id": 248_044,
        "eos_token_id": 248_044,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "layer_types": layer_types(40),
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        // 256K. Lands in `manifest.json` as `arch.trainedContext` and reaches
        // no field here, by the deliberate design AGENTS.md Gotcha 55 records.
        "max_position_embeddings": 262_144,
        "model_type": "qwen3_5_moe_text",
        "moe_intermediate_size": 512,
        // The head this checkpoint really ships (785 `mtp.*` tensors), unlike
        // the 9B's identical claim. Reaches no `ArchConfig` field either way.
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": false,
        "num_attention_heads": 16,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 40,
        "num_key_value_heads": 2,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1.0e-6,
        "rope_parameters": {
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000_000,
            "rope_type": "default"
        },
        // The FFN width the MoE parser reads. Qwen 3.6 spells it the same way
        // and `qwen3_5` dense uses `intermediate_size` instead -- the one
        // collision `qwen36_config.rs`'s mapping table calls out.
        "shared_expert_intermediate_size": 512,
        "tie_word_embeddings": false,
        "vocab_size": 248_320
    })
}

fn moe_config_json() -> String {
    serde_json::json!({
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": moe_text_config(),
        "vision_config": {"depth": 27, "model_type": "qwen3_5_moe_vision"}
    })
    .to_string()
}

/// `ornith-ai/Ornith-1.5-9B`'s production `text_config`.
fn dense_text_config() -> serde_json::Value {
    serde_json::json!({
        "attn_output_gate": true,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 4096,
        // The DENSE FFN width, which is the field name the MoE half uses for
        // something else entirely.
        "intermediate_size": 12288,
        "layer_types": layer_types(32),
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262_144,
        "model_type": "qwen3_5_text",
        // DECLARED AND UNSHIPPED: this checkpoint publishes 760 tensors and
        // NONE of them is `mtp.*`, so the key is a claim about the
        // architecture rather than about the weights. Whether an install can
        // draft is answered by the resident index (`mtp.fc.weight`) and never
        // by a config key, which is exactly why this one costs nothing.
        "mtp_num_hidden_layers": 1,
        "num_attention_heads": 16,
        "num_hidden_layers": 32,
        "num_key_value_heads": 4,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1.0e-6,
        "rope_parameters": {
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000_000,
            "rope_type": "default"
        },
        "tie_word_embeddings": false,
        "vocab_size": 248_320
    })
}

fn dense_config_json() -> String {
    serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": dense_text_config(),
        "vision_config": {"depth": 27}
    })
    .to_string()
}

/// **THE WHOLE BRING-UP DECISION IN ONE ASSERTION.**
///
/// `Ornith-1.5-35B-A3B` is 35B of retrained weights on Qwen 3.6's
/// architecture, so the correct amount of new `ArchConfig`, kernel and flow
/// work is zero. If this ever reddens, the checkpoint moved a shape key and
/// the bring-up is a real one.
#[test]
fn the_moe_checkpoint_parses_to_the_pinned_qwen36_baseline() {
    let arch = parse_qwen_gdn_moe_config(&moe_config_json()).expect("config parses");
    assert_eq!(
        arch,
        model_io::qwen_gdn_moe_35b_a3b(),
        "Ornith-1.5-35B-A3B does not match the pinned qwen3_5_moe baseline"
    );
}

/// The 9B's SHAPE, read off its own config.
///
/// Asserted field by field rather than against a baseline, because no shipped
/// baseline carries this shape and inventing a `qwen_gdn_dense_9b()` would be
/// a second copy of numbers only one checkpoint uses. Nothing needs one:
/// `peek_manifest_arch` overwrites every shape field from the manifest before
/// `validate_arch` runs, so shapes compare against themselves at load and only
/// the family-EXTENSION fields bind to a baseline (AGENTS.md Gotcha 24).
#[test]
fn the_dense_checkpoint_parses_to_its_published_shape() {
    let arch = parse_qwen_gdn_dense_config(&dense_config_json()).expect("config parses");

    assert_eq!(arch.hidden_size, 4096);
    assert_eq!(arch.intermediate_size, 12288, "the DENSE FFN width");
    assert_eq!(arch.moe_intermediate_size, 0, "no routed experts");
    assert_eq!(arch.num_layers, 32);
    assert_eq!(arch.num_heads, 16);
    assert_eq!(arch.num_kv_heads, 4);
    assert_eq!(arch.num_full_kv_heads, 4);
    assert_eq!(arch.head_dim, 256);
    assert_eq!(arch.full_head_dim, 256);
    assert_eq!(arch.vocab_size, 248_320);
    assert_eq!(arch.num_experts, 0);
    assert_eq!(arch.top_k_experts, 0);
    assert_eq!(arch.linear_attention.num_k_heads, 16);
    assert_eq!(arch.linear_attention.num_v_heads, 32);
    assert_eq!(arch.linear_attention.key_head_dim, 128);
    assert_eq!(arch.linear_attention.value_head_dim, 128);
    assert_eq!(arch.linear_attention.conv_kernel_size, 4);

    // 24 gated-DeltaNet layers to 8 full ones, every 4th full.
    assert_eq!(arch.full_attention_layer_mask.len(), 32);
    assert_eq!(
        arch.full_attention_layer_mask
            .iter()
            .filter(|&&m| m == 1)
            .count(),
        8,
        "eight full-attention layers"
    );
    assert_eq!(
        arch.full_attention_layer_mask[3], 1,
        "the fourth layer is the first full one"
    );
    assert_eq!(arch.full_attention_layer_mask[0], 2, "layer 0 is linear");
}

/// **What licenses running the 9B on the SHIPPED dense flow**: every
/// behavioural field equals `qwen_gdn_dense_27b()`'s, so `families/qwen/`
/// needs no arm for it. Only the shape moves, and shape is what the manifest
/// carries.
///
/// A coherence smoke cannot see a behavioural field at all, which is why this
/// is asserted rather than commented.
#[test]
fn the_dense_checkpoint_shares_the_shipped_flows_behaviour() {
    let arch = parse_qwen_gdn_dense_config(&dense_config_json()).expect("config parses");
    let baseline = model_io::qwen_gdn_dense_27b();

    assert_eq!(arch.family, baseline.family);
    assert_eq!(arch.hidden_activation, baseline.hidden_activation);
    assert_eq!(arch.attn_output_gate, baseline.attn_output_gate);
    assert_eq!(arch.attention_scale, baseline.attention_scale);
    assert_eq!(arch.rope_theta, baseline.rope_theta);
    assert_eq!(arch.full_rope_theta, baseline.full_rope_theta);
    assert_eq!(arch.partial_rotary_factor, baseline.partial_rotary_factor);
    assert_eq!(arch.rope_neox_subdim, baseline.rope_neox_subdim);
    assert_eq!(arch.attention_k_eq_v, baseline.attention_k_eq_v);
    assert_eq!(arch.ffn_sandwich_norms, baseline.ffn_sandwich_norms);
    assert_eq!(arch.router_scaled, baseline.router_scaled);
    assert_eq!(arch.shared_expert_gated, baseline.shared_expert_gated);
    assert_eq!(arch.sliding_window, baseline.sliding_window);
    assert_eq!(arch.final_logit_softcap, baseline.final_logit_softcap);
    assert_eq!(arch.tie_word_embeddings, baseline.tie_word_embeddings);
    assert_eq!(
        arch.embedding_scaled_by_sqrt_hidden,
        baseline.embedding_scaled_by_sqrt_hidden
    );
    assert_eq!(arch.rope_scaling, baseline.rope_scaling);
}

/// The two parsers must not accept each other's files.
///
/// `model_type` is one suffix apart (`qwen3_5` against `qwen3_5_moe`) and the
/// two configs differ in exactly four FFN fields, so a parser that fell
/// through would produce something that still validates structurally and then
/// dispatches the wrong FFN branch. `refuse_foreign_config` is what stops it;
/// this is the Ornith pair exercising the same guard `qwen35_config.rs` pins
/// for the Qwen pair.
///
/// The message names the two FAMILIES rather than the raw `model_type`
/// strings, which is the more useful pair: it says what the file is and what
/// the parser wanted, in the vocabulary the rest of the port uses.
#[test]
fn the_two_ornith_checkpoints_do_not_parse_as_each_other() {
    let err = parse_qwen_gdn_dense_config(&moe_config_json())
        .expect_err("the MoE config must not parse as dense");
    assert_eq!(
        format!("{err}"),
        "config.json invalid: model_type says qwen36, not qwen35",
        "the refusal must name what it found and what it wanted"
    );

    let err = parse_qwen_gdn_moe_config(&dense_config_json())
        .expect_err("the dense config must not parse as MoE");
    assert_eq!(
        format!("{err}"),
        "config.json invalid: model_type says qwen35, not qwen36",
        "the refusal must name what it found and what it wanted"
    );
}
