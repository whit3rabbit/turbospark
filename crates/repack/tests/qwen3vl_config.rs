//! `qwen3_vl` config parsing and refusal gates.

use turbospark_repack::parse_qwen3_vl_config;

/// The pinned checkpoint's own config, trimmed to the keys this parser
/// reads. Values are the REAL production numbers from
/// `mlx-community/Qwen3-VL-4B-Instruct-4bit`, so the headline equality
/// below cannot pass on a fixture whose every number is made up.
fn config_json() -> String {
    serde_json::json!({
        "model_type": "qwen3_vl",
        "image_token_id": 151655,
        "video_token_id": 151656,
        "quantization": { "group_size": 64, "bits": 4, "mode": "affine" },
        "text_config": {
            "model_type": "qwen3_vl_text",
            "vocab_size": 151936,
            "max_position_embeddings": 262144,
            "hidden_size": 2560,
            "intermediate_size": 9728,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "hidden_act": "silu",
            "rms_norm_eps": 1.0e-6,
            "rope_theta": 5.0e6,
            "rope_scaling": {
                "mrope_interleaved": true,
                "mrope_section": [24, 20, 20],
                "rope_type": "default"
            },
            "attention_bias": false,
            "tie_word_embeddings": true
        }
    })
    .to_string()
}

#[test]
fn parses_qwen3vl_into_the_pinned_baseline() {
    let arch = parse_qwen3_vl_config(&config_json()).expect("qwen3_vl config parses");
    assert_eq!(arch, model_io::qwen3_vl_4b());
}

#[test]
fn head_dim_is_independent_of_hidden_over_heads() {
    // 32 x 128 = 4096 != 2560: the q/k/v/o projections do not preserve the
    // hidden width. A parser that derived head_dim instead of reading it
    // would land on 80 and fail the headline equality above; this case
    // names the property so the failure says why.
    let arch = parse_qwen3_vl_config(&config_json()).unwrap();
    assert_eq!(arch.head_dim, 128);
    assert_eq!(arch.hidden_size / arch.num_heads, 80);
    assert_eq!(arch.full_head_dim, arch.head_dim);
}

#[test]
fn accepts_a_flat_config_without_the_wrapper() {
    let mut root: serde_json::Value = serde_json::from_str(&config_json()).unwrap();
    let obj = root.as_object_mut().unwrap();
    let tc = obj.remove("text_config").unwrap();
    let mut flat = tc.as_object().unwrap().clone();
    for (key, value) in obj {
        flat.entry(key.clone()).or_insert(value.clone());
    }
    let arch = parse_qwen3_vl_config(&flat_json(flat)).expect("flat qwen3_vl config parses");
    assert_eq!(arch, model_io::qwen3_vl_4b());
}

fn flat_json(mut flat: serde_json::Map<String, serde_json::Value>) -> String {
    flat.insert("model_type".into(), serde_json::json!("qwen3_vl"));
    serde_json::Value::Object(flat).to_string()
}

#[test]
fn refuses_wrong_behavioral_contracts() {
    for (key, value, expected) in [
        ("rms_norm_eps", serde_json::json!(1.0e-5), "rms_norm_eps"),
        ("hidden_act", serde_json::json!("gelu"), "hidden_act"),
        ("attention_bias", serde_json::json!(true), "attention_bias"),
        ("head_dim", serde_json::json!(0), "head_dim"),
        ("num_experts", serde_json::json!(64), "num_experts"),
    ] {
        let mut config: serde_json::Value = serde_json::from_str(&config_json()).unwrap();
        config["text_config"][key] = value;
        let error =
            parse_qwen3_vl_config(&config.to_string()).expect_err(&format!("{key} refused"));
        assert!(
            error.to_string().contains(expected),
            "{key}: expected the error to name {expected:?}, got: {error}"
        );
    }
}

#[test]
fn refuses_a_rope_scaling_type_this_port_does_not_know() {
    let mut config: serde_json::Value = serde_json::from_str(&config_json()).unwrap();
    config["text_config"]["rope_scaling"]["rope_type"] = serde_json::json!("yarn");
    let error = parse_qwen3_vl_config(&config.to_string()).expect_err("yarn refused");
    assert!(error.to_string().contains("rope_type"));
}

#[test]
fn validates_mrope_section_arithmetic_when_present() {
    // Sections partition the rotated sub-frequencies: they must sum to
    // head_dim / 2.
    for section in [vec![24, 20, 21], vec![24, 20], vec![]] {
        let mut config: serde_json::Value = serde_json::from_str(&config_json()).unwrap();
        config["text_config"]["rope_scaling"]["mrope_section"] =
            serde_json::to_value(&section).unwrap();
        let error = parse_qwen3_vl_config(&config.to_string())
            .expect_err(&format!("section {section:?} refused"));
        assert!(
            error.to_string().contains("mrope_section"),
            "expected the error to name mrope_section, got: {error}"
        );
    }
}

#[test]
fn accepts_an_absent_rope_scaling_block() {
    // The mRoPE declaration is a vision-time concern; its absence changes
    // nothing on the text-only path and must not refuse the config.
    let mut config: serde_json::Value = serde_json::from_str(&config_json()).unwrap();
    config["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("rope_scaling");
    assert_eq!(
        parse_qwen3_vl_config(&config.to_string()).unwrap(),
        model_io::qwen3_vl_4b()
    );
}

#[test]
fn refuses_another_familys_config_by_name() {
    let mut config: serde_json::Value = serde_json::from_str(&config_json()).unwrap();
    config["model_type"] = serde_json::json!("qwen3_5");
    let error = parse_qwen3_vl_config(&config.to_string()).expect_err("foreign config refused");
    assert!(error.to_string().contains("qwen3_5"), "got: {error}");
}
