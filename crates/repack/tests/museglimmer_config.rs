//! `parse_muse_glimmer_config` against the PRODUCTION field values of
//! `mlx-community/Muse-Glimmer-30B-4bit`
//! @ `3e7677d7a40d348a3daba263a2b1c0aa41910710`.
//!
//! The fixture is the real `text_config` verbatim, for the reason
//! `qwen35_config.rs` states: the headline test reduces to ONE assertion,
//! that the parse equals `model_io::muse_glimmer_30b()` field for field, and
//! a shape-only fixture cannot catch a key wired to the wrong field because
//! every dimension in it would be some other made-up number either way.
//!
//! **THIS FILE ALSO CARRIES THE HALF OF THE ARCHITECTURE THAT IS NOT IN
//! `ArchConfig`.** Four published scalars -- `qk_scale_factor`,
//! `output_multiplier` and the two RMS epsilons -- are family CONSTANTS in
//! `crates/runtime`'s `families/museglimmer/state.rs` rather than manifest
//! fields, because none of them is a binary fraction and `arch_validation`
//! compares manifest floats with `!=` on `f64` (AGENTS.md Gotcha 24). That
//! trade buys safety against a round-trip and takes on Gotcha 38's risk in
//! exchange: a value RECALLED where the file states one stays correct
//! exactly as long as one checkpoint exercises it.
//! `the_published_scalars_equal_the_flows_constants` is what closes that, and
//! it is the reason this test target exists rather than a case elsewhere.
//!
//! Offline: no network, no install, milliseconds.

use model_io::{muse_glimmer_30b, ModelFamily};
use turbospark_repack::{
    parse_gemma4_quantization, parse_muse_glimmer_config, parse_muse_glimmer_scalars,
};

/// 52 layers, `[sliding, sliding, sliding, full]` repeated 13 times, which
/// reproduces the checkpoint's own `layer_types` list.
fn layer_types() -> Vec<&'static str> {
    (0..52)
        .map(|i| {
            if i % 4 == 3 {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect()
}

/// The SECOND array the file publishes for the same pattern: 500000 on every
/// sliding layer and 0 on every full one.
///
/// A separate function rather than a derivation from [`layer_types`], because
/// the parser's job is to cross-check two independent arrays and a fixture
/// that derived one from the other could not exercise the disagreement cases
/// below.
fn layer_rope_theta() -> Vec<f64> {
    (0..52)
        .map(|i| if i % 4 == 3 { 0.0 } else { 500_000.0 })
        .collect()
}

/// The production `text_config`, trimmed to the keys the parser reads plus
/// the ones it must ignore.
fn text_config() -> serde_json::Value {
    serde_json::json!({
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": 200000,
        "eos_token_id": 200001,
        "final_logit_softcapping": 20.0,
        "head_dim": 128,
        "hidden_activation": "silu",
        "hidden_size": 6656,
        "initializer_range": 0.02,
        "intermediate_size": 19968,
        "layer_rope_theta": layer_rope_theta(),
        "layer_types": layer_types(),
        "max_position_embeddings": 131072,
        "model_type": "muse_glimmer_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 52,
        "num_key_value_heads": 2,
        "output_multiplier": 0.19611613513818404,
        "pad_token_id": serde_json::Value::Null,
        "post_norm_eps": 1e-08,
        "qk_scale_factor": 3.87,
        "rms_norm_eps": 1e-05,
        "rope_parameters": { "rope_theta": 500000.0, "rope_type": "default" },
        "sliding_window": 2048,
        "tie_word_embeddings": false,
        "use_cache": true,
        "vocab_size": 202048,
    })
}

/// The full production `config.json`, wrapper and all.
///
/// The `vision_config` is present and trimmed rather than dropped: this is a
/// vision-language checkpoint and the parser must ignore that tower, which a
/// fixture without one cannot demonstrate.
fn config() -> String {
    serde_json::json!({
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "dtype": "bfloat16",
        "eos_token_id": [200001, 200008],
        "image_token_id": 200092,
        "model_type": "muse_glimmer",
        "out_hidden_size": 6144,
        "projector_hidden_act": "gelu",
        "projector_hidden_size": 4096,
        "quantization": { "group_size": 64, "bits": 4, "mode": "affine" },
        "quantization_config": { "group_size": 64, "bits": 4, "mode": "affine" },
        "text_config": text_config(),
        "transformers_version": "5.15.0.dev0",
        "video_token_id": 200091,
        "vision_config": {
            "hidden_act": "gelu",
            "hidden_size": 1536,
            "intermediate_size": 8960,
            "model_type": "muse_glimmer_vision",
            "num_attention_heads": 16,
            "num_hidden_layers": 50,
            "patch_size": 14,
        },
    })
    .to_string()
}

/// Replaces one `text_config` key, for the refusal cases.
fn config_with(key: &str, value: serde_json::Value) -> String {
    let mut tc = text_config();
    tc[key] = value;
    let mut root: serde_json::Value = serde_json::from_str(&config()).unwrap();
    root["text_config"] = tc;
    root.to_string()
}

/// THE HEADLINE. One equality, field for field, against the shipped
/// baseline.
#[test]
fn the_production_config_parses_to_the_baseline() {
    let parsed = parse_muse_glimmer_config(&config()).expect("parses");
    assert_eq!(parsed, muse_glimmer_30b());
}

/// The four scalars that are NOT `ArchConfig` fields, against the constants
/// the decode flow holds. See the module header.
///
/// Exact equality rather than a tolerance: both sides are `f64` literals
/// transcribed from the same file, so any difference is a transcription
/// error, which is exactly what this exists to catch.
#[test]
fn the_published_scalars_equal_the_flows_constants() {
    let s = parse_muse_glimmer_scalars(&config()).expect("parses");
    assert_eq!(s.qk_scale_factor, 3.87);
    assert_eq!(s.output_multiplier, 0.19611613513818404);
    assert_eq!(s.rms_norm_eps, 1e-5);
    assert_eq!(s.post_norm_eps, 1e-8);

    // The two epsilons are FOUR ORDERS OF MAGNITUDE apart, which is the
    // whole reason this family needs a pair rather than the single `rms_eps`
    // every other flow carries. Stated as an assertion so a future edit that
    // collapsed them would redden here rather than in a perplexity row.
    assert!(
        s.rms_norm_eps > s.post_norm_eps * 100.0,
        "the post-norm epsilon must be far smaller than the standard one"
    );

    // `output_multiplier` is 26^-0.5, which is worth pinning because it is
    // the kind of value that reads as noise and would survive a typo.
    assert!((s.output_multiplier - 26f64.powf(-0.5)).abs() < 1e-15);
}

/// The `attention_scale` is `head_dim ** -0.5` from the reference, and it is
/// NOT the `qk_scale_factor` and not the two folded together.
#[test]
fn the_attention_scale_is_the_head_dim_alone() {
    let parsed = parse_muse_glimmer_config(&config()).expect("parses");
    assert_eq!(parsed.attention_scale, 128f64.powf(-0.5));

    let s = parse_muse_glimmer_scalars(&config()).expect("parses");
    let folded = s.qk_scale_factor * parsed.attention_scale;
    assert_ne!(
        parsed.attention_scale, folded,
        "folding qk_scale_factor into attention_scale would change the value \
         `encode_attention_decode` specializes into its pipeline"
    );
}

/// The full-attention layers are NoPE, and `full_rope_theta` zero is the
/// FILE's value rather than an absent-field default.
#[test]
fn the_full_attention_layers_are_nope() {
    let parsed = parse_muse_glimmer_config(&config()).expect("parses");
    assert_eq!(parsed.rope_theta, 500_000.0);
    assert_eq!(parsed.full_rope_theta, 0.0);

    // Thirteen full layers of 52, the last one being layer 51.
    let full: Vec<usize> = parsed
        .full_attention_layer_mask
        .iter()
        .enumerate()
        .filter(|(_, &m)| m == 1)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(full.len(), 13);
    assert_eq!(full[0], 3);
    assert_eq!(*full.last().unwrap(), 51);
}

/// The two arrays are cross-checked. A `layer_rope_theta` that rotates a
/// FULL layer is refused rather than believed.
///
/// This is the case the module header is about: nothing upstream keeps the
/// two arrays consistent, and an engine that reads only one of them produces
/// fluent wrong text rather than an error.
#[test]
fn a_rotating_full_attention_layer_is_refused() {
    let mut thetas = layer_rope_theta();
    thetas[3] = 500_000.0; // layer 3 is full, so it must be NoPE
    let err =
        parse_muse_glimmer_config(&config_with("layer_rope_theta", serde_json::json!(thetas)))
            .expect_err("a full layer with a theta must be refused");
    let msg = err.to_string();
    assert!(msg.contains("layer 3"), "{msg}");
    assert!(msg.contains("must agree"), "{msg}");
}

/// The mirror: a sliding layer with no theta.
#[test]
fn a_nope_sliding_layer_is_refused() {
    let mut thetas = layer_rope_theta();
    thetas[0] = 0.0; // layer 0 is sliding, so it must rotate
    let err =
        parse_muse_glimmer_config(&config_with("layer_rope_theta", serde_json::json!(thetas)))
            .expect_err("a sliding layer without a theta must be refused");
    assert!(err.to_string().contains("layer 0"), "{err}");
}

/// `ArchConfig` carries one theta per layer KIND, so a genuinely per-layer
/// schedule must be refused rather than collapsed to its first value.
#[test]
fn two_different_nonzero_thetas_are_refused() {
    let mut thetas = layer_rope_theta();
    thetas[4] = 1_000_000.0;
    let err =
        parse_muse_glimmer_config(&config_with("layer_rope_theta", serde_json::json!(thetas)))
            .expect_err("a per-layer theta schedule must be refused");
    assert!(err.to_string().contains("per-layer schedule"), "{err}");
}

/// A mask shorter than `num_hidden_layers` runs the wrong block on every
/// layer past its end, so it is length-checked rather than padded.
#[test]
fn a_short_layer_types_is_refused() {
    let short: Vec<&str> = layer_types().into_iter().take(51).collect();
    let err = parse_muse_glimmer_config(&config_with("layer_types", serde_json::json!(short)))
        .expect_err("a short layer_types must be refused");
    assert!(err.to_string().contains("51 entries"), "{err}");
}

/// This architecture is dense. A config that also declares experts is a
/// contradiction, not extra information.
#[test]
fn a_config_declaring_experts_is_refused() {
    let err = parse_muse_glimmer_config(&config_with("num_experts", serde_json::json!(128)))
        .expect_err("a dense family with experts must be refused");
    assert!(err.to_string().contains("dense"), "{err}");
}

/// The registry guard: another family's `config.json` must not parse here.
///
/// Worth having because every parser hardcodes `family:` on the way out, so
/// a mis-routed config produces an `ArchConfig` that CLAIMS to be this family
/// while carrying another's shapes.
#[test]
fn a_foreign_config_is_refused() {
    let mut root: serde_json::Value = serde_json::from_str(&config()).unwrap();
    root["model_type"] = serde_json::json!("qwen3_5");
    root["text_config"]["model_type"] = serde_json::json!("qwen3_5_text");
    let err = parse_muse_glimmer_config(&root.to_string())
        .expect_err("a qwen3_5 config must not parse as muse_glimmer");
    assert!(err.to_string().contains("model_type"), "{err}");
}

/// The quantization block is the ordinary INT4 affine at group 64 with BF16
/// companions, which is the shape `model_io::validate_quant` has always
/// accepted -- no new width, unlike the two sub-4-bit checkpoints.
#[test]
fn the_quantization_is_the_ordinary_int4_affine() {
    let quant = parse_gemma4_quantization(&config()).expect("parses");
    assert_eq!(quant.default_bits, 4);
    assert_eq!(quant.group_size, 64);
    assert!(
        turbospark_repack::is_supported_affine_shape(quant.default_bits, quant.group_size),
        "4-bit at group 64 has kernels"
    );
}

/// The family resolves from BOTH `model_type` strings the checkpoint carries,
/// and the lookup is exact equality.
#[test]
fn both_model_type_strings_resolve_to_this_family() {
    for s in ["muse_glimmer", "muse_glimmer_text"] {
        assert_eq!(
            turbospark_repack::hf_family_for_model_type(s),
            Some(ModelFamily::MuseGlimmer),
            "{s}"
        );
    }
    // A prefix match would be wrong here for the reason it is wrong for the
    // two `qwen3_5` rows: nothing guarantees a future `muse_glimmer_moe`
    // would share this baseline.
    assert_eq!(
        turbospark_repack::hf_family_for_model_type("muse_glimmer_moe"),
        None
    );
}
