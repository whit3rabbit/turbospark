//! `parse_qwen3_config` against `Qwen/Qwen3-4B`'s `config.json`.
//!
//! The fixture (`fixtures/qwen3_4b_config.json`) is RECONSTRUCTED from the
//! published values (hidden 2560, 36 layers, 32 heads / 8 KV heads,
//! `head_dim` 128, intermediate 9728, vocab 151936, tied embeddings,
//! `rope_theta` 1e6, no scaling); it was not downloaded in this session.
//! The assertions are inline so a key wired to the wrong field fails without
//! a network. A byte-pinned copy is owed to the first real intake.

use model_io::{qwen3_4b, ModelFamily};
use turbospark_repack::parse_qwen3_config;

const REAL: &str = include_str!("fixtures/qwen3_4b_config.json");

fn edited(f: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>)) -> String {
    let mut v: serde_json::Value = serde_json::from_str(REAL).unwrap();
    f(v.as_object_mut().unwrap());
    v.to_string()
}

#[test]
fn the_qwen3_4b_config_parses_to_the_published_shape() {
    let arch = parse_qwen3_config(REAL).expect("the pinned config parses");
    assert_eq!(arch.family, ModelFamily::Qwen3Dense);
    assert_eq!(arch.hidden_size, 2560);
    assert_eq!(arch.intermediate_size, 9728);
    assert_eq!(arch.num_layers, 36);
    assert_eq!(arch.num_heads, 32);
    assert_eq!(arch.num_kv_heads, 8);
    assert_eq!(arch.vocab_size, 151_936);
    assert!(arch.tie_word_embeddings);
    assert_eq!(arch.rope_theta, 1_000_000.0);
    assert_eq!(arch.partial_rotary_factor, 1.0);
    assert_eq!(arch.full_attention_layer_mask, vec![1u8; 36]);
    assert_eq!(arch.num_experts, 0);
    assert!(!arch.vision.is_active());
    // head_dim is the config's 128, NOT hidden/heads = 80.
    assert_eq!(arch.head_dim, 128);
    assert_eq!(arch.full_head_dim, 128);
    // The whole parse agrees with the GGUF-path baseline for this size.
    assert_eq!(arch, qwen3_4b());
}

#[test]
fn head_dim_is_required_not_derived() {
    let json = edited(|m| {
        m.remove("head_dim");
    });
    let err = parse_qwen3_config(&json).unwrap_err().to_string();
    assert!(err.contains("head_dim"), "{err}");
}

#[test]
fn an_untied_size_reads_its_own_flag() {
    let arch = parse_qwen3_config(&edited(|m| {
        m.insert("tie_word_embeddings".into(), false.into());
    }))
    .unwrap();
    assert!(!arch.tie_word_embeddings);
}

#[test]
fn the_2507_instruct_context_length_parses() {
    let arch = parse_qwen3_config(&edited(|m| {
        m.insert("max_position_embeddings".into(), 262_144.into());
    }))
    .unwrap();
    assert_eq!(arch.family, ModelFamily::Qwen3Dense);
}

#[test]
fn configs_the_flow_cannot_run_are_refused_by_name() {
    let cases: Vec<(&str, String)> = vec![
        (
            "attention_bias",
            edited(|m| {
                m.insert("attention_bias".into(), true.into());
            }),
        ),
        (
            "sliding-window",
            edited(|m| {
                m.insert("use_sliding_window".into(), true.into());
            }),
        ),
        (
            "rope_scaling",
            edited(|m| {
                m.insert(
                    "rope_scaling".into(),
                    serde_json::json!({"rope_type": "yarn", "factor": 4.0}),
                );
            }),
        ),
        (
            "layer_types",
            edited(|m| {
                m.insert(
                    "layer_types".into(),
                    serde_json::json!(["sliding_attention"]),
                );
            }),
        ),
        (
            "num_experts",
            edited(|m| {
                m.insert("num_experts".into(), 128.into());
            }),
        ),
        (
            "rms_norm_eps",
            edited(|m| {
                m.insert("rms_norm_eps".into(), 1e-5.into());
            }),
        ),
        (
            "hidden_act",
            edited(|m| {
                m.insert("hidden_act".into(), "gelu".into());
            }),
        ),
    ];
    for (needle, json) in cases {
        let err = parse_qwen3_config(&json).expect_err(needle).to_string();
        assert!(err.contains(needle), "{needle}: {err}");
    }
}

#[test]
fn a_sibling_family_model_type_is_refused() {
    for sibling in ["qwen3_vl", "qwen3_5", "qwen2", "gemma4"] {
        let json = edited(|m| {
            m.insert("model_type".into(), sibling.into());
        });
        let err = parse_qwen3_config(&json).expect_err(sibling).to_string();
        assert!(err.contains("resolves to"), "{sibling}: {err}");
    }
}
