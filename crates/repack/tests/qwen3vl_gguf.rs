//! Qwen3-VL's GGUF text-trunk intake, offline.
//!
//! The authoritative checks are in `qwen3vl_gguf_network.rs` (real bytes).
//! These cover the registry, the derivation on a small header with the real
//! file's keys, the refusal, and the whole walk.

use model_io::ModelFamily;
use turbospark_repack::{
    arch_from_gguf, family_for_architecture, gguf_architecture, map_gguf_name,
    orchestrate_gguf_checkpoint, parse_gguf_header, GgufBuilder, GgufConfigError, GgufMapping,
    GgufValue, MemoryRangeSource, ResidentEntrySpec, GGUF_DEFAULT_MAX_HEADER_BYTES,
};

const LAYERS: u64 = 2;
const HIDDEN: u64 = 128;
const HEADS: u64 = 4;
const KV_HEADS: u64 = 2;
const HEAD_DIM: u64 = 128;
const INTER: u64 = 256;

fn builder(sections: Vec<u32>) -> GgufBuilder {
    GgufBuilder::new()
        .metadata_str("general.architecture", "qwen3vl")
        .metadata_u32("qwen3vl.block_count", LAYERS as u32)
        .metadata_u32("qwen3vl.embedding_length", HIDDEN as u32)
        .metadata_u32("qwen3vl.feed_forward_length", INTER as u32)
        .metadata_u32("qwen3vl.attention.head_count", HEADS as u32)
        .metadata_u32("qwen3vl.attention.head_count_kv", KV_HEADS as u32)
        .metadata_u32("qwen3vl.attention.key_length", HEAD_DIM as u32)
        .metadata_u32("qwen3vl.attention.value_length", HEAD_DIM as u32)
        .metadata_f32("qwen3vl.rope.freq_base", 5_000_000.0)
        .metadata_f32("qwen3vl.attention.layer_norm_rms_epsilon", 1e-6)
        .metadata_u32("qwen3vl.n_deepstack_layers", 3)
        .metadata(
            "qwen3vl.rope.dimension_sections",
            GgufValue::Array(sections.into_iter().map(GgufValue::U32).collect()),
        )
}

fn with_tensors(mut b: GgufBuilder) -> GgufBuilder {
    b = b
        .q8_0_tensor("token_embd.weight", &[HIDDEN, 256], 1)
        .tensor(
            "output_norm.weight",
            0,
            &[HIDDEN],
            vec![0u8; HIDDEN as usize * 4],
        );
    for l in 0..LAYERS {
        let t = |s: &str| format!("blk.{l}.{s}.weight");
        b = b
            .q8_0_tensor(&t("attn_q"), &[HIDDEN, HEADS * HEAD_DIM], 3)
            .q8_0_tensor(&t("attn_k"), &[HIDDEN, KV_HEADS * HEAD_DIM], 4)
            .q8_0_tensor(&t("attn_v"), &[HIDDEN, KV_HEADS * HEAD_DIM], 5)
            .q8_0_tensor(&t("attn_output"), &[HEADS * HEAD_DIM, HIDDEN], 6)
            .q8_0_tensor(&t("ffn_gate"), &[HIDDEN, INTER], 7)
            .q8_0_tensor(&t("ffn_up"), &[HIDDEN, INTER], 8)
            .q8_0_tensor(&t("ffn_down"), &[INTER, HIDDEN], 9)
            .tensor(
                &t("attn_norm"),
                0,
                &[HIDDEN],
                vec![0u8; HIDDEN as usize * 4],
            )
            .tensor(&t("ffn_norm"), 0, &[HIDDEN], vec![0u8; HIDDEN as usize * 4])
            .tensor(
                &t("attn_q_norm"),
                0,
                &[HEAD_DIM],
                vec![0u8; HEAD_DIM as usize * 4],
            )
            .tensor(
                &t("attn_k_norm"),
                0,
                &[HEAD_DIM],
                vec![0u8; HEAD_DIM as usize * 4],
            );
    }
    b
}

fn arch(b: GgufBuilder) -> Result<model_io::ArchConfig, GgufConfigError> {
    let (bytes, _) = b.build();
    let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).expect("parse");
    arch_from_gguf(&h)
}

#[test]
fn the_architecture_string_is_the_text_trunks_and_not_the_moe_rows() {
    assert_eq!(
        family_for_architecture("qwen3vl"),
        Some(ModelFamily::Qwen3Vl)
    );
    assert_eq!(gguf_architecture(ModelFamily::Qwen3Vl), Some("qwen3vl"));
    // One word away, and a different family.
    assert_eq!(
        family_for_architecture("qwen3vlmoe"),
        Some(ModelFamily::Qwen3Moe)
    );
}

#[test]
fn derives_a_full_attention_tied_trunk() {
    let a = arch(with_tensors(builder(vec![24, 20, 20, 0]))).expect("derives");
    assert_eq!(a.family, ModelFamily::Qwen3Vl);
    assert_eq!(a.num_layers, LAYERS as i64);
    assert_eq!(a.full_attention_layer_mask, vec![1, 1]);
    assert!(a.tie_word_embeddings, "no output.weight in the file");
    assert_eq!((a.head_dim, a.full_head_dim), (128, 128));
    assert_eq!(a.rope_theta, 5_000_000.0);
    // The 3-section spelling is accepted too.
    assert!(arch(with_tensors(builder(vec![24, 20, 20]))).is_ok());
}

#[test]
fn other_rope_sections_are_another_model_and_refused() {
    for sections in [vec![16, 24, 24, 0], vec![24, 20]] {
        match arch(with_tensors(builder(sections.clone()))) {
            Err(GgufConfigError::BadValue { key, .. }) => {
                assert_eq!(key, "qwen3vl.rope.dimension_sections", "{sections:?}")
            }
            other => panic!("{sections:?}: expected BadValue, got {other:?}"),
        }
    }
    // Absent entirely is refused too: the key is what marks a VL trunk.
    let (bytes, _) = with_tensors(
        GgufBuilder::new()
            .metadata_str("general.architecture", "qwen3vl")
            .metadata_u32("qwen3vl.block_count", 2)
            .metadata_u32("qwen3vl.embedding_length", 128)
            .metadata_u32("qwen3vl.attention.head_count", 4),
    )
    .build();
    let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    assert!(arch_from_gguf(&h).is_err());
}

#[test]
fn names_are_the_dense_qwen3_table_and_a_tower_name_is_not_mapped() {
    let m = |n: &str| map_gguf_name(n, ModelFamily::Qwen3Vl);
    let p = "language_model.model.layers.1.";
    for (g, c) in [
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("attn_q_norm", "self_attn.q_norm"),
        ("attn_k_norm", "self_attn.k_norm"),
        ("attn_norm", "input_layernorm"),
        ("ffn_norm", "post_attention_layernorm"),
        ("ffn_gate", "mlp.gate_proj"),
        ("ffn_up", "mlp.up_proj"),
        ("ffn_down", "mlp.down_proj"),
    ] {
        assert_eq!(
            m(&format!("blk.1.{g}.weight")),
            Ok(GgufMapping::Resident(format!("{p}{c}.weight")))
        );
    }
    // The tower lives in the mmproj file; none of its names reach this table.
    assert!(m("v.blk.0.attn_q.weight").is_err());
    assert!(m("mm.0.weight").is_err());
}

#[test]
fn the_walk_installs_the_trunk_under_the_resident_names() {
    let (bytes, _) = with_tensors(builder(vec![24, 20, 20, 0])).build();
    let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    let out = orchestrate_gguf_checkpoint(&h, &MemoryRangeSource::new(&bytes)).expect("walks");
    assert_eq!(out.arch.family, ModelFamily::Qwen3Vl);
    let names: Vec<String> = out
        .resident
        .iter()
        .map(|e| match e {
            ResidentEntrySpec::Raw(r) => r.name.clone(),
            ResidentEntrySpec::Int4(r) | ResidentEntrySpec::Int8(r) => r.name.clone(),
            _ => String::new(),
        })
        .collect();
    for want in [
        "language_model.model.layers.0.self_attn.q_norm.weight",
        "language_model.model.layers.1.mlp.down_proj.weight",
        "language_model.model.norm.weight",
        "language_model.model.embed_tokens.weight",
    ] {
        assert!(
            names.iter().any(|n| n == want),
            "{want} missing from {names:?}"
        );
    }
}
