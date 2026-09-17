//! `deepseek2` Phase 1 gates: the real-walk synthetic install, the tensor
//! name table, and the two derived floats. Facts: `docs/DEEPSEEK2_PHASE0.md`.

use model_io::{self, ModelFamily};
use turbospark_repack::{
    build_synthetic_deepseek2_install, describe_gguf_architecture, family_for_architecture,
    gguf_arch_support, gguf_architecture, map_gguf_name, tiny_deepseek2_arch, ArchSupport,
    GgufMapping,
};

const MODEL_ID: &str = "tiny-deepseek2";

fn tempdir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("deepseek2-phase1-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The real-walk fixture: a 2-layer install (dense lead plus one MoE layer)
/// whose derived arch must equal the intended tiny arch FIELD BY FIELD. A
/// mismatch here is the fixture's metadata lying, and this assert is what
/// catches it before any runtime test builds on the fixture.
#[test]
fn the_synthetic_walk_derives_the_intended_tiny_arch() {
    let dir = tempdir("walk");
    let arch = build_synthetic_deepseek2_install(&dir, 1024, 2, MODEL_ID)
        .map_err(|e| {
            eprintln!("install failed: {e}");
            e
        })
        .expect("install");
    assert_eq!(arch, tiny_deepseek2_arch(1024, 2));

    // The compressed-cache shape the runtime keys on: ONE shared row per
    // layer, whatever the file's `head_count_kv` said (the fixture writes 4
    // to prove the override fires), and the MLA block the flow reads.
    assert_eq!(arch.num_kv_heads, 1);
    assert_eq!(arch.num_full_kv_heads, 1);
    assert_eq!(arch.mla.kv_lora_rank, 64);
    assert_eq!(arch.mla.rope_head_dim, 16);
    assert_eq!(arch.mla.nope_head_dim, 32);
    assert_eq!(arch.mla.v_head_dim, 32);
    assert_eq!(arch.mla.key_head_dim(), 48);
    assert_eq!(arch.mla.cache_row_dim(), 80);
    // The two FFN widths are DIFFERENT numbers on this architecture, which
    // is the whole reason the dense lead carries its own field.
    assert_eq!(arch.intermediate_size, 64);
    assert_eq!(arch.dense_lead_intermediate_size, 96);
    assert_eq!(arch.full_attention_layer_mask, vec![5u8, 5]);

    std::fs::remove_dir_all(&dir).ok();
}

/// The witness's real tensor names, one per row of the mapping, asserted in
/// both directions: every name the real file carries maps, and the ones it
/// does not are refused rather than guessed.
#[test]
fn every_real_tensor_name_maps_and_unknown_ones_refuse() {
    let f = ModelFamily::Deepseek2;

    let expected_resident = [
        (
            "token_embd.weight",
            "language_model.model.embed_tokens.weight",
        ),
        ("output_norm.weight", "language_model.model.norm.weight"),
        ("output.weight", "language_model.lm_head.weight"),
        (
            "blk.0.attn_norm.weight",
            "language_model.model.layers.0.input_layernorm.weight",
        ),
        (
            "blk.0.attn_q.weight",
            "language_model.model.layers.0.self_attn.q_proj.weight",
        ),
        (
            "blk.0.attn_kv_a_mqa.weight",
            "language_model.model.layers.0.self_attn.kv_a_proj.weight",
        ),
        (
            "blk.0.attn_kv_a_norm.weight",
            "language_model.model.layers.0.self_attn.kv_a_norm.weight",
        ),
        (
            "blk.0.attn_kv_b.weight",
            "language_model.model.layers.0.self_attn.kv_b_proj.weight",
        ),
        (
            "blk.0.attn_output.weight",
            "language_model.model.layers.0.self_attn.o_proj.weight",
        ),
        (
            "blk.0.ffn_norm.weight",
            "language_model.model.layers.0.post_attention_layernorm.weight",
        ),
        // The dense lead's plain triple.
        (
            "blk.0.ffn_gate.weight",
            "language_model.model.layers.0.mlp.gate_proj.weight",
        ),
        (
            "blk.0.ffn_up.weight",
            "language_model.model.layers.0.mlp.up_proj.weight",
        ),
        (
            "blk.0.ffn_down.weight",
            "language_model.model.layers.0.mlp.down_proj.weight",
        ),
        // The fused shared expert (resident, one SwiGLU).
        (
            "blk.1.ffn_gate_shexp.weight",
            "language_model.model.layers.1.mlp.shared_expert.gate_proj.weight",
        ),
        (
            "blk.1.ffn_up_shexp.weight",
            "language_model.model.layers.1.mlp.shared_expert.up_proj.weight",
        ),
        (
            "blk.1.ffn_down_shexp.weight",
            "language_model.model.layers.1.mlp.shared_expert.down_proj.weight",
        ),
        (
            "blk.1.ffn_gate_inp.weight",
            "language_model.model.layers.1.mlp.gate.weight",
        ),
    ];
    for (gguf, canonical) in expected_resident {
        match map_gguf_name(gguf, f).unwrap_or_else(|e| panic!("{gguf}: {e}")) {
            GgufMapping::Resident(name) => assert_eq!(name, canonical, "{gguf}"),
            other => panic!("{gguf} mapped as {other:?}, expected resident"),
        }
    }

    for (gguf, role) in [
        ("blk.1.ffn_gate_exps.weight", "gate"),
        ("blk.1.ffn_up_exps.weight", "up"),
        ("blk.1.ffn_down_exps.weight", "down"),
    ] {
        match map_gguf_name(gguf, f).unwrap() {
            GgufMapping::Routed { layer, role: r } => {
                assert_eq!((layer, r), (1, role), "{gguf}");
            }
            other => panic!("{gguf} mapped as {other:?}, expected routed"),
        }
    }

    // Names the real file does not carry, refused by name: a qwen-style
    // per-head v or k tensor would silently map onto a table that has no
    // such row, and a rope_freqs tensor is the by-name refusal Gotcha 39
    // records.
    for gguf in [
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "rope_freqs.weight",
    ] {
        assert!(map_gguf_name(gguf, f).is_err(), "{gguf} should refuse");
    }
}

/// The two derived floats, pinned at BOTH ends: the baseline constant
/// against the formula, and the manifest round-trip against the constant.
/// The value is not a binary fraction, so this is Gotcha 24's trap
/// discharged rather than assumed; the scratch measurement said exact.
#[test]
fn the_derived_yarn_values_round_trip_through_the_manifest() {
    let arch = model_io::deepseek_v2_lite_16b();

    // HF's yarn_get_mscale(40, 0.707), applied once to the rope magnitude
    // and squared to the attention score.
    let mscale = 1.0 + 0.1 * 0.707 * 40.0f64.ln();
    assert_eq!(arch.rope_scaling.yarn_mscale(), mscale);
    assert_eq!(arch.attention_scale, mscale * mscale / 192.0f64.sqrt());
    assert_eq!(arch.attention_scale, 0.114_721_386_792_926_12);

    // The manifest's own write/read, which is what an install actually
    // goes through.
    let written = serde_json::to_string(&arch.attention_scale).unwrap();
    let read: f64 = serde_json::from_str(&written).unwrap();
    assert_eq!(
        read, arch.attention_scale,
        "attention_scale moved in the manifest"
    );
    let written = serde_json::to_string(&arch.rope_scaling.mscale).unwrap();
    let read: f64 = serde_json::from_str(&written).unwrap();
    assert_eq!(
        read, arch.rope_scaling.mscale,
        "mscale moved in the manifest"
    );
}

/// The registry row moved tables, and both of its old guarantees must
/// survive in their new form: the string resolves to a FAMILY (not a
/// refusal), and the family round-trips its wire string.
#[test]
fn the_registry_row_is_supported_and_the_wire_string_round_trips() {
    match gguf_arch_support("deepseek2") {
        Some(ArchSupport::Supported(f)) => {
            assert_eq!(f, ModelFamily::Deepseek2);
        }
        other => panic!("deepseek2 resolved as {other:?}, expected supported"),
    }
    assert_eq!(
        family_for_architecture("deepseek2"),
        Some(ModelFamily::Deepseek2)
    );
    assert_eq!(gguf_architecture(ModelFamily::Deepseek2), Some("deepseek2"));
    assert_eq!(ModelFamily::Deepseek2.as_str(), "deepseek2");
    assert_eq!(
        ModelFamily::parse("deepseek2"),
        Some(ModelFamily::Deepseek2)
    );
    // And the old message is gone: a supported architecture no longer
    // renders the planned refusal.
    assert!(!describe_gguf_architecture("deepseek2").contains("recognized but"));
}
