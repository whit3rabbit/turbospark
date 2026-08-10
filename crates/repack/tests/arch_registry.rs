//! The architecture registry (ROADMAP Phase M Stage 1): what resolves, what
//! is merely recognized, and the `config.json` family guard.
//!
//! The row VALUES are checked against real published files by
//! `tests/arch_registry_network.rs`; this file checks the logic around them,
//! and runs in the default suite.

use model_io::ModelFamily;
use turbospark_repack::{
    arch_from_gguf, describe_gguf_architecture, gguf_arch_support, hf_family_for_model_type,
    parse_gemma4_config, parse_qwen36_config, planned_gguf_architectures, ArchSupport,
};

#[test]
fn the_two_running_architectures_resolve_to_their_family() {
    assert_eq!(
        gguf_arch_support("gemma4"),
        Some(ArchSupport::Supported(ModelFamily::Gemma4))
    );
    // Not "qwen36": llama.cpp's converter name is what a real file carries.
    assert_eq!(
        gguf_arch_support("qwen35moe"),
        Some(ArchSupport::Supported(ModelFamily::Qwen36))
    );
    // Supported PARTIALLY (ROADMAP Phase M2): the MoE half of this
    // architecture runs and the dense half is refused at open, which no
    // string-keyed table can express -- only `expert_count` says which half a
    // file is.
    assert_eq!(
        gguf_arch_support("llama"),
        Some(ArchSupport::Supported(ModelFamily::Llama))
    );
}

/// Recognition is not support. A planned row must NOT leak into the family
/// resolution the repack walk keys on, or a checkpoint with no decode flow
/// would be installed under some other family's rules.
#[test]
fn a_planned_architecture_is_recognized_but_has_no_family() {
    for (key, planned) in planned_gguf_architectures() {
        assert!(
            matches!(gguf_arch_support(key), Some(ArchSupport::Planned(_))),
            "{key} should be planned"
        );
        assert!(!planned.needs.is_empty(), "{key} has no needs clause");
        assert!(
            planned.witness.starts_with("https://") && planned.witness.ends_with(".gguf"),
            "{key}'s witness is not a fetchable GGUF url: {}",
            planned.witness
        );
    }
}

#[test]
fn an_unknown_architecture_resolves_to_nothing() {
    assert_eq!(gguf_arch_support("not-a-real-architecture"), None);
}

/// The point of the whole table: the refusal names the string, and for a
/// recognized one it also names the next step.
#[test]
fn the_refusal_names_the_architecture_and_the_checklist() {
    let planned = describe_gguf_architecture("qwen3moe");
    assert!(planned.contains("qwen3moe"), "{planned}");
    assert!(planned.contains("docs/NEW_MODEL.md"), "{planned}");

    let unknown = describe_gguf_architecture("not-a-real-architecture");
    assert!(unknown.contains("not-a-real-architecture"), "{unknown}");
    assert!(unknown.contains("docs/NEW_MODEL.md"), "{unknown}");
}

/// `arch_from_gguf` is the caller that matters, and its error renders
/// through the registry. A synthetic header is enough: only the
/// architecture string is read before the refusal.
#[test]
fn arch_from_gguf_refuses_a_planned_architecture_with_the_registry_message() {
    let bytes = minimal_gguf("qwen3moe");
    let header =
        turbospark_repack::parse_gguf_header(&bytes, bytes.len() as u64).expect("parse header");
    let message = arch_from_gguf(&header)
        .expect_err("qwen3moe has no flow")
        .to_string();
    assert!(message.contains("qwen3moe"), "{message}");
    assert!(message.contains("docs/NEW_MODEL.md"), "{message}");
}

/// Both `model_type` spellings a real checkpoint carries. The `_text` suffix
/// is what the multimodal wrapper's inner config uses, and it is a distinct
/// string rather than a prefix match.
#[test]
fn both_model_type_spellings_resolve() {
    assert_eq!(
        hf_family_for_model_type("gemma4"),
        Some(ModelFamily::Gemma4)
    );
    assert_eq!(
        hf_family_for_model_type("gemma4_text"),
        Some(ModelFamily::Gemma4)
    );
    assert_eq!(
        hf_family_for_model_type("qwen3_5_moe"),
        Some(ModelFamily::Qwen36)
    );
    assert_eq!(
        hf_family_for_model_type("qwen3_5_moe_text"),
        Some(ModelFamily::Qwen36)
    );
    // Qwen2-MoE is a different model, and an earlier draft of
    // docs/MODEL_FAMILY.md claimed this row existed.
    assert_eq!(hf_family_for_model_type("qwen2_moe"), None);
}

/// The silent failure this guard exists for: every parser hardcodes
/// `family:` on the way out, so without it a Qwen config parsed by the Gemma
/// parser yields a Gemma-labelled `ArchConfig`.
#[test]
fn each_config_parser_refuses_the_other_family() {
    let qwen = r#"{"model_type": "qwen3_5_moe", "text_config": {}}"#;
    let gemma = r#"{"model_type": "gemma4", "text_config": {}}"#;

    let e = parse_gemma4_config(qwen)
        .expect_err("gemma parser must refuse a qwen config")
        .to_string();
    assert!(e.contains("qwen36"), "{e}");

    let e = parse_qwen36_config(gemma)
        .expect_err("qwen parser must refuse a gemma config")
        .to_string();
    assert!(e.contains("gemma4"), "{e}");
}

/// An absent or unrecognized `model_type` is NOT evidence of the wrong
/// family, and the trimmed fixtures elsewhere in this directory omit the key
/// entirely. Both must get past the guard and fail (or pass) on their own
/// keys instead.
#[test]
fn a_config_claiming_nothing_recognizable_still_reaches_the_parser() {
    for json in [
        r#"{"text_config": {}}"#,
        r#"{"model_type": "something-else", "text_config": {}}"#,
    ] {
        let e = parse_gemma4_config(json)
            .expect_err("empty text_config")
            .to_string();
        assert!(
            e.contains("missing"),
            "guard should not have fired, got: {e}"
        );
    }
}

/// The smallest GGUF that carries an architecture string: v3 header, one
/// metadata entry, no tensors.
fn minimal_gguf(architecture: &str) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes()); // version
    out.extend_from_slice(&0u64.to_le_bytes()); // tensor count
    out.extend_from_slice(&1u64.to_le_bytes()); // metadata count
    let key = b"general.architecture";
    out.extend_from_slice(&(key.len() as u64).to_le_bytes());
    out.extend_from_slice(key);
    out.extend_from_slice(&8u32.to_le_bytes()); // value type: string
    out.extend_from_slice(&(architecture.len() as u64).to_le_bytes());
    out.extend_from_slice(architecture.as_bytes());
    out
}
