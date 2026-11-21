use super::*;

#[test]
fn steering_names_map_to_the_hf_install_names() {
    for (lamq, hf) in [
        (
            "model.layers.3.self_attn.q_proj.norm.scale",
            "model.llm.layers.3.attn.q_l2_clip.scale",
        ),
        (
            "model.layers.0.self_attn.relative_position_bias.norm.scale",
            "model.llm.layers.0.attn.rel_l2_clip.scale",
        ),
        (
            "model.layers.0.self_attn.relative_position_bias.proj",
            "model.llm.layers.0.attn.rel_logits_proj.proj",
        ),
        (
            "model.layers.5.self_attn.k_proj.sconv.static_weight",
            "model.llm.layers.5.attn.k_sconv.weight",
        ),
        (
            "model.layers.4.self_attn.v_proj.sconv.conv1d.weight",
            "model.llm.layers.4.attn.v_sconv.weight",
        ),
        (
            "model.layers.5.self_attn.k_proj.sconv.weight_proj.weight",
            "model.llm.layers.5.attn.k_sconv.weight_proj",
        ),
        (
            "model.embed_tokens.embed_norm.weight",
            "model.llm.embed_norm.weight",
        ),
        (
            "model.layers.1.attention_input.query",
            "model.llm.attn_res.attn_queries.0",
        ),
        (
            "model.layers.0.mlp_input.query",
            "model.llm.attn_res.mlp_queries.0",
        ),
        ("model.final_input.query", "model.llm.attn_res.output_query"),
        (
            "model.layers.7.mlp.gate.e_score_correction_bias",
            "model.llm.layers.7.mlp.gate.bias",
        ),
        (
            "model.layers.7.q_message_gate.forget_gate_proj.weight",
            "model.llm.layers.7.controllers.attn.forget_weight",
        ),
        (
            "model.layers.7.v_message_gate.message_gate_proj.weight",
            "model.llm.layers.7.controllers.v.gain_weight",
        ),
        (
            "model.layers.2.post_attention_convolution.conv1d.weight",
            "model.llm.layers.2.attn_sconv.weight",
        ),
        (
            "model.layers.2.self_attn.sink_proj.weight",
            "model.llm.layers.2.attn.attn_sink_proj.weight",
        ),
        ("model.norm.weight", "model.llm.norm.weight"),
    ] {
        assert_eq!(lamq_steering_to_hf(lamq).as_deref(), Some(hf), "{lamq}");
    }
    assert_eq!(lamq_steering_to_hf("model.message_input.query"), None);
    assert_eq!(
        lamq_steering_to_hf("lm_head.weight"),
        None,
        "not a steering name"
    );
}

#[test]
fn meta_parameters_must_agree_with_the_code() {
    let ok = serde_json::json!({"k": 2, "g": 4, "K": 512, "hash_mult": 2654435761u64});
    check_meta_code("t", &ok, LamqCode::Ht2).unwrap();
    for bad in [
        serde_json::json!({"k": 3, "g": 4, "K": 512, "hash_mult": 2654435761u64}),
        serde_json::json!({"k": 2, "g": 2, "K": 512, "hash_mult": 2654435761u64}),
        serde_json::json!({"k": 2, "g": 4, "K": 1024, "hash_mult": 2654435761u64}),
        serde_json::json!({"k": 2, "g": 4, "K": 512, "hash_mult": 0}),
    ] {
        assert!(check_meta_code("t", &bad, LamqCode::Ht2).is_err(), "{bad}");
    }
    let rtr = serde_json::json!({"k": 4, "g": 1, "K": 4096, "hash_mult": 0});
    check_meta_code("t", &rtr, LamqCode::Rtr4).unwrap();
    let hashed_rtr = serde_json::json!({"k": 4, "g": 1, "K": 4096, "hash_mult": 2654435761u64});
    assert!(check_meta_code("t", &hashed_rtr, LamqCode::Rtr4).is_err());
    check_meta_code("t", &serde_json::json!({"k": 6}), LamqCode::Lm6).unwrap();
    assert!(
        code_of("t", Some(&serde_json::json!("ht1"))).is_err(),
        "ht1 ships no table"
    );
}

/// The section deserializes and names the schemes `validate_quant` admits
/// (`model_io`'s `lamq_scheme_tests` validate the same shape).
#[test]
fn quant_section_names_the_admitted_schemes() {
    let q: model_io::ManifestQuant =
        serde_json::from_value(inkling2_lamq_manifest_quant()).unwrap();
    assert_eq!(q.embedding.scheme, LAMQ_SCHEME);
    assert_eq!(q.attention, q.embedding);
    assert_eq!(q.shared_expert, q.attention, "no shared expert: defaulted");
    assert_eq!(q.routed_expert.scheme, INKLING2_LAMQ_ROUTED_SCHEME);
    assert_eq!(
        (q.router.scheme.as_str(), q.router.weight_bits),
        ("raw", 32)
    );
    for s in [&q.embedding, &q.routed_expert] {
        assert_eq!(
            (
                s.weight_bits,
                s.scale_type.as_str(),
                s.bias_type.as_str(),
                s.group_size
            ),
            (0, "none", "none", 0)
        );
    }
}

use crate::npz_fixture::{entry, npy, zip};

fn tmp(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("turbospark-lamq-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn luts() -> Luts {
    Luts {
        tables: vec![(
            LamqCode::Rtr1,
            (0..4096).map(|i| i as f32 / 4096.0).collect(),
        )],
    }
}

fn u32s(n: usize) -> Vec<u8> {
    (0..n as u32)
        .flat_map(|v| v.wrapping_mul(2_654_435_761).to_le_bytes())
        .collect()
}

fn f16s(n: usize) -> Vec<u8> {
    (0..n).flat_map(|_| 0x3C00u16.to_le_bytes()).collect()
}

fn ones(n: usize) -> Vec<u8> {
    (0..n).map(|i| if i % 3 == 0 { 0xFF } else { 1 }).collect()
}

/// One rtr1 bucket of `rows` x 384 plus an order over `order`.
fn write_bucketed(dir: &Path, order: &[i32], rows: usize, extra: Option<(&str, Vec<u8>)>) {
    let meta = serde_json::json!([{"kb": 0, "opt": "rtr1", "rows": rows, "k": 1, "g": 1, "K": 4096, "hash_mult": 0}]);
    std::fs::write(dir.join("embed_meta.json"), meta.to_string()).unwrap();
    let ord: Vec<u8> = order.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut e = vec![
        entry(
            "order.npy",
            npy("<i4", &format!("({},)", order.len()), &ord, false),
        ),
        entry(
            "b0_codes.npy",
            npy("<u4", &format!("({rows}, 12)"), &u32s(rows * 12), false),
        ),
        entry(
            "b0_scales.npy",
            npy("<f2", &format!("({rows},)"), &f16s(rows), false),
        ),
        entry("b0_signs_in.npy", npy("|i1", "(384,)", &ones(384), false)),
    ];
    if let Some((name, data)) = extra {
        e.push(entry(name, data));
    }
    std::fs::write(dir.join("embed_packed.npz"), zip(&e)).unwrap();
}

fn run_bucketed(
    dir: &Path,
    positions: usize,
    vocab: usize,
) -> Result<Vec<ResidentEntrySpec>, String> {
    let mut out = Vec::new();
    bucketed(
        dir,
        "embed",
        "model.llm.embed.weight",
        384,
        positions,
        vocab,
        &luts(),
        &mut out,
    )
    .map(|_| out)
    .map_err(|e| e.0)
}

#[test]
fn bucketed_writes_an_order_and_its_buckets() {
    let d = tmp("bucket-ok");
    write_bucketed(&d, &[2, 0, 3], 3, None);
    let out = run_bucketed(&d, 3, 4).unwrap();
    let names: Vec<&str> = out
        .iter()
        .map(|e| match e {
            ResidentEntrySpec::Lamq(t) => t.name.as_str(),
            _ => panic!("only LAMQ entries"),
        })
        .collect();
    assert_eq!(
        names,
        ["model.llm.embed.weight", "model.llm.embed.weight.b0"]
    );
    let ResidentEntrySpec::Lamq(b0) = &out[1] else {
        unreachable!()
    };
    let aux: Vec<u8> = b0.biases.iter().flat_map(|v| v.to_le_bytes()).collect();
    let p = compute::lamq::parse_matrix_aux(&aux).unwrap();
    assert_eq!(
        (p.header.code, p.header.n, p.header.m),
        (Some(LamqCode::Rtr1), 384, 3)
    );
}

#[test]
fn bucketed_refuses_aliasing_orders_short_buckets_and_strays() {
    // (tag, order, bucket rows, extra member, expected refusal)
    type Case = (
        &'static str,
        Vec<i32>,
        usize,
        Option<(&'static str, Vec<u8>)>,
        &'static str,
    );
    let cases: [Case; 5] = [
        ("repeat", vec![2, 0, 2], 3, None, "out of range or repeated"),
        ("range", vec![2, 0, 4], 3, None, "out of range or repeated"),
        (
            "negative",
            vec![2, -1, 3],
            3,
            None,
            "out of range or repeated",
        ),
        ("short", vec![2, 0, 3], 2, None, "cover"),
        (
            "stray",
            vec![2, 0, 3],
            3,
            Some(("b9_codes.npy", npy("<u4", "(1,)", &[0; 4], false))),
            "unexpected members",
        ),
    ];
    for (tag, order, rows, extra, want) in cases {
        let d = tmp(&format!("bucket-{tag}"));
        write_bucketed(&d, &order, rows, extra);
        let e = run_bucketed(&d, 3, 4).unwrap_err();
        assert!(e.contains(want), "{tag}: {e}");
    }
    // signs_out on a bucket without out_rot is refused before the member count.
    let d = tmp("signs-out");
    write_bucketed(
        &d,
        &[2, 0, 3],
        3,
        Some(("b0_signs_out.npy", npy("|i1", "(3,)", &ones(3), false))),
    );
    assert!(run_bucketed(&d, 3, 4)
        .unwrap_err()
        .contains("signs_out present without out_rot"));
}

/// Two experts, rtr1 on both sides, hidden 384 and ffn 32.
fn write_experts(
    dir: &Path,
    gu_map: [i16; 4],
    extra: Option<(&str, Vec<u8>)>,
) -> std::path::PathBuf {
    let (h, f) = (384usize, 32usize);
    let map = |m: [i16; 4]| -> Vec<u8> { m.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let mut e = vec![
        entry("gu_signs_in.npy", npy("|i1", "(384,)", &ones(h), false)),
        entry("dn_signs_in.npy", npy("|i1", "(32,)", &ones(f), false)),
        entry(
            "gu_rtr1_codes.npy",
            npy("<u4", "(2, 64, 12)", &u32s(2 * 2 * f * h / 32), false),
        ),
        entry(
            "gu_rtr1_scales.npy",
            npy("<f2", "(2, 64)", &f16s(2 * 2 * f), false),
        ),
        entry("gu_map.npy", npy("<i2", "(2, 2)", &map(gu_map), false)),
        entry(
            "dn_rtr1_codes.npy",
            npy("<u4", "(2, 384, 1)", &u32s(2 * h), false),
        ),
        entry(
            "dn_rtr1_scales.npy",
            npy("<f2", "(2, 384)", &f16s(2 * h), false),
        ),
        entry(
            "dn_map.npy",
            npy("<i2", "(2, 2)", &map([3, 0, 3, 1]), false),
        ),
    ];
    if let Some((n, d)) = extra {
        e.push(entry(n, d));
    }
    let p = dir.join("layer.npz");
    std::fs::write(&p, zip(&e)).unwrap();
    p
}

fn run_experts(p: &Path) -> Result<(LayerBlobs, u64), String> {
    let meta = serde_json::json!({"layer": 0, "linears": {}, "experts": {"gu": {"rtr1": 2}, "dn": {"rtr1": 2}}});
    let mut z = NpzArchive::open(p).unwrap();
    layer_experts(&mut z, &meta, 0, 2, 384, 32)
        .map(|(b, _, n)| (b, n))
        .map_err(|e| e.0)
}

#[test]
fn expert_blobs_have_uniform_offsets_and_a_format_record() {
    let d = tmp("experts-ok");
    let (blobs, packed) = run_experts(&write_experts(&d, [3, 1, 3, 0], None)).unwrap();
    // Expert 0 is slot 1 of the gate-up slab.
    let e0 = &blobs.experts[0];
    let roles: Vec<&str> = e0.sub_tensors.iter().map(|s| s.role.as_str()).collect();
    assert_eq!(
        roles,
        [
            "lamq_fmt",
            "gate_up",
            "gate_up_scales",
            "down",
            "down_scales"
        ]
    );
    let fmt = &e0.sub_tensors[0].bytes;
    assert_eq!((fmt[0], fmt[4]), (LamqCode::Rtr1.id(), LamqCode::Rtr1.id()));
    assert_eq!(
        u32::from_le_bytes(fmt[8..12].try_into().unwrap()),
        48,
        "gate-up row bytes"
    );
    assert_eq!(
        u32::from_le_bytes(fmt[12..16].try_into().unwrap()),
        4,
        "down row bytes"
    );
    let span = 64 * 48;
    let slab = u32s(2 * 64 * 12);
    assert_eq!(
        e0.sub_tensors[1].bytes,
        slab[span..2 * span],
        "slot 1 of gate-up"
    );
    assert_eq!(
        blobs.experts[1].sub_tensors[1].bytes,
        slab[..span],
        "slot 0 of gate-up"
    );
    assert_eq!(packed, 2 * (span + 2 * 64 + 384 * 4 + 2 * 384) as u64);
}

#[test]
fn expert_maps_refuse_reuse_unknown_formats_and_strays() {
    for (tag, gu, want) in [
        ("reuse", [3, 0, 3, 0], "slot out of range or reused"),
        ("range", [3, 0, 3, 2], "slot out of range or reused"),
        ("ht1", [0, 0, 3, 1], "fid 0 refused"),
        ("noslab", [1, 0, 3, 1], "with no slab"),
    ] {
        let d = tmp(&format!("experts-{tag}"));
        let e = run_experts(&write_experts(&d, gu, None)).unwrap_err();
        assert!(e.contains(want), "{tag}: {e}");
    }
    let d = tmp("stray-expert");
    let e = run_experts(&write_experts(
        &d,
        [3, 0, 3, 1],
        Some(("gu_ht2_codes.npy", npy("<u4", "(1,)", &[0; 4], false))),
    ))
    .unwrap_err();
    assert!(e.contains("unexpected members"), "{e}");
}

#[test]
fn narrowing_rounds_to_nearest_even_and_refuses_non_finite() {
    let vals = [
        1.0f32,
        1.0 + 1.0 / 256.0,
        1.0 + 3.0 / 512.0,
        -2.5e-3,
        0.0,
        65504.0,
    ];
    let b: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let (out, worst) = narrow_f32_rne("t", &b).unwrap();
    let got: Vec<f32> = out
        .chunks_exact(2)
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
        .collect();
    // 1 + 1/256 is a tie between 1 and 1 + 1/128: even mantissa wins (1.0).
    assert_eq!(got[1], 1.0);
    // 1 + 3/512 rounds up to 1 + 1/128.
    assert_eq!(got[2], 1.0 + 1.0 / 128.0);
    assert_eq!((got[0], got[4]), (1.0, 0.0));
    assert!(
        worst > 0.0 && worst <= 1.0 / 256.0,
        "half a BF16 ulp at most: {worst}"
    );
    for bad in [f32::NAN, f32::INFINITY] {
        assert!(narrow_f32_rne("t", &bad.to_le_bytes()).is_err());
    }
    assert!(narrowed_to_bf16("model.llm.layers.0.attn.k_sconv.weight"));
    assert!(narrowed_to_bf16(
        "model.llm.layers.6.controllers.v.gain_weight"
    ));
    assert!(
        !narrowed_to_bf16("model.llm.layers.0.mlp.gate.weight"),
        "the router stays F32"
    );
    assert!(!narrowed_to_bf16("model.llm.attn_res.mlp_queries.3"));
}
