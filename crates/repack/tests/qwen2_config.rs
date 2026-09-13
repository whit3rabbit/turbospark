//! Qwen2/Qwen2.5 config parsing and refusal gates.

use turbospark_repack::parse_qwen2_config;

fn config_json() -> String {
    serde_json::json!({
        "model_type": "qwen2",
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "num_hidden_layers": 28,
        "vocab_size": 152064,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1.0e-6,
        "rope_theta": 1.0e6,
        "hidden_act": "silu",
        "tie_word_embeddings": false,
        "use_sliding_window": false,
        "sliding_window": null
    })
    .to_string()
}

#[test]
fn parses_qwen25_into_the_pinned_baseline() {
    let arch = parse_qwen2_config(&config_json()).expect("Qwen2 config parses");
    assert_eq!(arch, model_io::qwen2_5_7b());
}

#[test]
fn accepts_the_text_config_wrapper() {
    let mut root: serde_json::Value = serde_json::from_str(&config_json()).unwrap();
    let text = root.as_object_mut().unwrap().remove("model_type");
    let mut text_config = root.as_object_mut().unwrap().clone();
    text_config.insert("model_type".into(), text.unwrap());
    let wrapped = serde_json::json!({
        "model_type": "qwen2",
        "text_config": text_config
    });
    assert_eq!(
        parse_qwen2_config(&wrapped.to_string()).unwrap(),
        model_io::qwen2_5_7b()
    );
}

#[test]
fn refuses_wrong_behavioral_contracts() {
    for (key, value, expected) in [
        ("rms_norm_eps", serde_json::json!(1.0e-5), "rms_norm_eps"),
        ("hidden_act", serde_json::json!("gelu"), "hidden_act"),
        (
            "use_sliding_window",
            serde_json::json!(true),
            "sliding-window",
        ),
    ] {
        let mut config: serde_json::Value = serde_json::from_str(&config_json()).unwrap();
        config[key] = value;
        let error = parse_qwen2_config(&config.to_string())
            .expect_err("invalid Qwen2 behavior must be refused")
            .to_string();
        assert!(error.contains(expected), "{key}: {error}");
    }
}

#[test]
fn refuses_non_qwen2_model_types() {
    let mut config: serde_json::Value = serde_json::from_str(&config_json()).unwrap();
    config["model_type"] = serde_json::json!("qwen3_5_moe");
    let error = parse_qwen2_config(&config.to_string())
        .expect_err("a different registered family must be refused")
        .to_string();
    assert!(error.contains("qwen36"), "{error}");
}
