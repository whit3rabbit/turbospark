//! Fish Audio Drama 3 intake (`crates/repack/src/fish_drama3.rs`).
//!
//! The real `config.json` is copied verbatim from
//! `mlx-community/fish-audio-drama-3-4bit` at rev
//! cc639e27aba7edb73429d3e3806f471dcd79d316 (git blob
//! fa8042126101484c6eb000a70a72e4be89eb839d). The install tests stream the
//! synthetic Drama checkpoint (`synthetic_qwen/drama3.rs`) through the same
//! streamed writer a real install uses. Facts: `docs/FISH_DRAMA3_PHASE0.md`.

use std::path::PathBuf;

use turbospark_repack::{
    canonicalize_drama3_header, config_json_family, drama3_trunk_config, headless_tts_trunk_owner,
    parse_gemma4_quantization, parse_header, parse_qwen_gdn_moe_config,
    synthetic_fish_drama3_checkpoint, write_synthetic_fish_drama3, Drama3NormStorage, Gemma4Error,
    DRAMA3_SIDE_FILE, DRAMA3_TRUNK_DIR,
};

const CONFIG: &str = include_str!("fixtures/fish_drama3_4bit_config.json");

fn tempdir(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("repack_fish_drama3_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn with_text(edit: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>)) -> String {
    let mut root: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    edit(root["text_config"].as_object_mut().unwrap());
    root.to_string()
}

#[test]
fn the_real_trunk_is_qwen36_with_its_own_vocabulary() {
    let derived = drama3_trunk_config(CONFIG).expect("the real config is accepted");
    let parsed = parse_qwen_gdn_moe_config(&derived).expect("derived config parses");
    let mut expected = model_io::qwen_gdn_moe_35b_a3b();
    expected.vocab_size = 252_288;
    assert_eq!(parsed, expected);
}

#[test]
fn quantization_keys_move_onto_the_walk_namespace() {
    let derived = drama3_trunk_config(CONFIG).unwrap();
    let quant = parse_gemma4_quantization(&derived).unwrap();
    // The global default is 8; the routed experts are 4 only through their
    // per-path keys, which name `backbone.model.` in the source config.
    assert_eq!(quant.default_bits, 8);
    let routed = "language_model.model.layers.0.mlp.switch_mlp.gate_proj";
    assert_eq!(quant.bits_for(routed), 4);
    assert_eq!(
        quant.bits_for("language_model.model.layers.3.self_attn.q_proj"),
        8
    );
    // Nothing keeps the source spelling.
    let root: serde_json::Value = serde_json::from_str(&derived).unwrap();
    assert!(root["quantization"]
        .as_object()
        .unwrap()
        .keys()
        .all(|k| !k.starts_with("backbone.")));
}

#[test]
fn generic_intake_refuses_drama_by_name() {
    let root: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    assert_eq!(headless_tts_trunk_owner(&root), Some("fish_drama3"));
    // And it resolves to no chat family: neither `fish_drama3` nor
    // `fish_omni_text` is in the supported table.
    assert_eq!(config_json_family(&root), None);
    let qwen = serde_json::json!({"model_type": "qwen3_5_moe"});
    assert_eq!(headless_tts_trunk_owner(&qwen), None);
}

#[test]
fn trunk_config_refusals_name_the_setting() {
    let cases: Vec<(String, &str)> = vec![
        (
            with_text(|t| {
                t.insert("rms_norm_eps".into(), 1e-5.into());
            }),
            "rms_norm_eps",
        ),
        (
            with_text(|t| {
                t.insert("rotary_interleaved".into(), true.into());
            }),
            "rotary_interleaved",
        ),
        (
            with_text(|t| {
                t.insert("tie_word_embeddings".into(), true.into());
            }),
            "tie",
        ),
        (
            with_text(|t| {
                let types: Vec<&str> = (0..40)
                    .map(|i| {
                        if i % 2 == 1 {
                            "full_attention"
                        } else {
                            "linear_attention"
                        }
                    })
                    .collect();
                t.insert("layer_types".into(), serde_json::json!(types));
            }),
            "three linear-attention layers",
        ),
        (
            with_text(|t| {
                let mut types = vec!["moe"; 40];
                types[7] = "dense";
                t.insert("mlp_layer_types".into(), serde_json::json!(types));
            }),
            "mixture-of-experts",
        ),
    ];
    for (config, needle) in cases {
        let err = drama3_trunk_config(&config).expect_err(needle).to_string();
        assert!(err.contains(needle), "{needle}: {err}");
    }
    let mut root: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    root["model_type"] = "qwen3_5_moe".into();
    assert!(drama3_trunk_config(&root.to_string()).is_err());
    let mut root: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    root["quantization"]["backbone.lm_head"] = serde_json::json!({"group_size": 64, "bits": 8});
    let err = drama3_trunk_config(&root.to_string())
        .unwrap_err()
        .to_string();
    assert!(err.contains("backbone.lm_head"), "{err}");
}

#[test]
fn header_rename_refuses_names_outside_the_three_namespaces() {
    let checkpoint = synthetic_fish_drama3_checkpoint(256, 4, Drama3NormStorage::Shifted);
    let mut header = parse_header(&checkpoint.blob, 1 << 20).unwrap();
    canonicalize_drama3_header(&mut header).expect("the fixture renames cleanly");
    assert!(header
        .tensors
        .keys()
        .all(|k| k.starts_with("language_model.model.") || k.starts_with("audio_tower.")));
    assert!(header
        .tensors
        .contains_key("audio_tower.fish_drama3.semantic_head.weight"));

    for stray in [
        "backbone.lm_head.weight",
        "language_model.model.norm.weight",
        "mtp.fc.weight",
    ] {
        let mut header = parse_header(&checkpoint.blob, 1 << 20).unwrap();
        let info = header.tensors.values().next().unwrap().clone();
        header.tensors.insert(stray.to_string(), info);
        let err = canonicalize_drama3_header(&mut header).unwrap_err();
        assert!(
            matches!(&err, Gemma4Error::UnknownTensor(n) if n == stray)
                || err.to_string().contains("duplicate"),
            "{stray}: {err}"
        );
    }
}

#[test]
fn streamed_intake_writes_a_marked_headless_trunk_and_the_side_file() {
    let dir = tempdir("install");
    let checkpoint = synthetic_fish_drama3_checkpoint(256, 4, Drama3NormStorage::Shifted);
    let arch = write_synthetic_fish_drama3(&dir, &checkpoint, "synthetic/fish-drama3")
        .expect("streamed intake");
    assert_eq!(arch.family, model_io::ModelFamily::QwenGdnMoe);
    assert!(!arch.tie_word_embeddings);

    let trunk = dir.join(DRAMA3_TRUNK_DIR);
    assert!(trunk.join("manifest.json").exists());
    let role = model_io::trunk_role::read(&trunk).expect("marker published with the trunk");
    assert_eq!(role.owner, "fish_drama3");
    // The intake writes nothing else at the root: the root files are the
    // catalog's verbatim downloads.
    let mut root_entries: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    root_entries.sort();
    assert_eq!(root_entries, [DRAMA3_TRUNK_DIR, DRAMA3_SIDE_FILE]);

    let index = model_io::load_resident_index(&trunk.join("model_weights.bin")).unwrap();
    assert!(
        !index.entries.contains_key("language_model.lm_head.weight"),
        "the trunk is headless"
    );
    assert!(index.entries.keys().all(|k| !k.starts_with("audio_tower.")));
    for layer in 0..4 {
        let router =
            &index.entries[&format!("language_model.model.layers.{layer}.mlp.gate.weight")];
        assert_eq!(
            router.dtype, 5,
            "layer {layer} router must be INT8 for the qwen flow"
        );
    }

    // Every side tensor, under its original name, bytes verbatim.
    let side = model_io::SafetensorsFile::open(&dir.join(DRAMA3_SIDE_FILE)).unwrap();
    let source = parse_header(&checkpoint.blob, 1 << 20).unwrap();
    let data_start = 8 + source.header_len as usize;
    let mut want: Vec<&str> = source
        .tensors
        .keys()
        .filter(|k| k.starts_with("tower.") || k.starts_with("semantic_head."))
        .map(String::as_str)
        .collect();
    want.sort_unstable();
    let mut got: Vec<&str> = side.tensor_names().collect();
    got.sort_unstable();
    assert_eq!(got, want);
    for name in want {
        let (a, b) = source.tensors[name].data_offsets;
        assert_eq!(
            side.raw_bytes(name).unwrap(),
            &checkpoint.blob[data_start + a as usize..data_start + b as usize],
            "{name}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn unshifted_norms_are_refused_before_anything_is_written() {
    let dir = tempdir("unshifted");
    let checkpoint = synthetic_fish_drama3_checkpoint(256, 4, Drama3NormStorage::Unshifted);
    let err = write_synthetic_fish_drama3(&dir, &checkpoint, "synthetic/fish-drama3")
        .expect_err("an unshifted export is refused")
        .to_string();
    assert!(err.contains("1 + w"), "{err}");
    assert!(!dir.join(DRAMA3_TRUNK_DIR).exists());
    assert!(!dir.join(DRAMA3_SIDE_FILE).exists());
    let _ = std::fs::remove_dir_all(&dir);
}
