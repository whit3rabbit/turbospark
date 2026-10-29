//! `lfm3` config parsing and refusal gates.

use turbospark_repack::parse_lfm3_config;

/// The published checkpoint's own config, verbatim except for `auto_map`
/// and the token ids (keys this parser does not read). Values are the REAL
/// production numbers from `LiquidAI/LFM3-1.7B` @ 40ebe0aa, which its MLX
/// conversions repeat, so the headline equality cannot pass on a fixture
/// whose every number is made up.
fn config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["Lfm3ForCausalLM"],
        "conv_L_cache": 7,
        "conv_bias": false,
        "dtype": "bfloat16",
        "head_dim": 128,
        "hidden_size": 2048,
        "intermediate_size": 7680,
        "layer_types": [
            "conv", "conv", "full_attention", "conv", "sliding_attention", "conv",
            "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
            "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv",
            "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv"
        ],
        "max_position_embeddings": 262144,
        "model_type": "lfm3",
        "norm_eps": 1e-05,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "num_key_value_heads": 4,
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.5,
                "rope_theta": 1000000.0,
                "rope_type": "default"
            },
            "sliding_attention": { "rope_theta": 10000, "rope_type": "default" }
        },
        "sliding_window": 256,
        "tie_word_embeddings": true,
        "use_cache": true,
        "use_sandwich_norm": true,
        "vocab_size": 128000
    })
}

fn parse(v: &serde_json::Value) -> Result<model_io::ArchConfig, String> {
    parse_lfm3_config(&v.to_string()).map_err(|e| e.to_string())
}

/// ONE equality against the family baseline: a key wired to the wrong field
/// shows up here immediately, where a synthetic fixture's made-up numbers
/// would agree with either wiring (NEW_MODEL.md Phase 1).
#[test]
fn the_published_config_parses_to_the_baseline() {
    assert_eq!(parse(&config()).unwrap(), model_io::lfm3_1_7b());
}

#[test]
fn a_foreign_model_type_is_refused() {
    let mut v = config();
    v["model_type"] = "qwen3_5".into();
    let e = parse(&v).unwrap_err();
    assert!(e.contains("not lfm3"), "{e}");
}

#[test]
fn turning_off_sandwich_norms_is_refused() {
    let mut v = config();
    v["use_sandwich_norm"] = false.into();
    assert!(parse(&v).unwrap_err().contains("use_sandwich_norm"));
}

#[test]
fn a_conv_bias_is_refused() {
    let mut v = config();
    v["conv_bias"] = true.into();
    assert!(parse(&v).unwrap_err().contains("conv_bias"));
}

#[test]
fn an_unknown_layer_type_is_refused() {
    let mut v = config();
    v["layer_types"][3] = "linear_attention".into();
    assert!(parse(&v).unwrap_err().contains("layer_types entry"));
}

#[test]
fn a_layer_types_length_mismatch_is_refused() {
    let mut v = config();
    v["num_hidden_layers"] = 23.into();
    assert!(parse(&v).unwrap_err().contains("24 entries for 23 layers"));
}

/// The flow rotates the whole head on sliding layers, so a partial factor
/// there is a different model, not a detail.
#[test]
fn a_partial_factor_on_sliding_layers_is_refused() {
    let mut v = config();
    v["rope_parameters"]["sliding_attention"]["partial_rotary_factor"] = 0.5.into();
    assert!(parse(&v)
        .unwrap_err()
        .contains("sliding_attention partial_rotary_factor"));
}

/// 0.3 x 128 = 38.4, not a whole number of dims.
#[test]
fn a_fractional_rotary_dim_is_refused() {
    let mut v = config();
    v["rope_parameters"]["full_attention"]["partial_rotary_factor"] = 0.3.into();
    assert!(parse(&v).unwrap_err().contains("positive even integer"));
}

#[test]
fn a_scaled_rope_type_is_refused() {
    let mut v = config();
    v["rope_parameters"]["full_attention"]["rope_type"] = "yarn".into();
    assert!(parse(&v).unwrap_err().contains("only \"default\""));
}

#[test]
fn a_different_epsilon_is_refused() {
    let mut v = config();
    v["norm_eps"] = 1e-6.into();
    assert!(parse(&v).unwrap_err().contains("norm_eps"));
}
