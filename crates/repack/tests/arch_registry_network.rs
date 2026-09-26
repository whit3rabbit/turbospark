//! The admission gate for `arch_registry`'s planned rows (ROADMAP Phase M
//! Stage 1): every key is re-read off the real published file it was taken
//! from.
//!
//! A registry of architecture strings is exactly the kind of table that
//! rots into folklore, because a wrong row costs nothing until someone
//! points a real checkpoint at it and gets "not in this port's registry"
//! for a model the port would in fact have recognized. `gguf_names.rs`
//! states the same rule for its name table; this is that rule made runnable.
//!
//! Cost is a header per row -- a few MB off files of 4 to 700 GB -- not a
//! download, and about a second each. `#[ignore]`d and opt-in like every
//! other `*_network` test.
//!
//! ```sh
//! cargo test -p turbospark-repack --test arch_registry_network --release -- --ignored --nocapture
//! ```

use model_io::ModelFamily;
use turbospark_repack::{
    arch_from_gguf, fetch_gguf_header, ggml_type_block, ggml_type_name, gguf_arch_support,
    map_gguf_name, planned_gguf_architectures, ArchSupport, GgufMapping, GgufSet, HttpRangeSource,
};

/// Mixtral, which is the reason ROADMAP Phase M2 does the `llama`
/// architecture MoE-first: it reports the SAME architecture string as a
/// dense Llama 3.1 and expresses its experts through `expert_count`. If
/// this ever stops being true, the plan that rests on it is wrong.
const MIXTRAL_8X7B: &str = "https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_0.gguf";
const SWIFT_QWEN38_IQ2_XS_SHARDS: [&str; 2] = [
    "https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/b22d729eae29b5796f76fb70f91aef549b9fc52c/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf",
    "https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/b22d729eae29b5796f76fb70f91aef549b9fc52c/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf",
];
const SWIFT_QWEN38_Q2_0_REVISION: &str = "b22d729eae29b5796f76fb70f91aef549b9fc52c";
const SWIFT_QWEN38_Q2_0_SHARDS: [(&str, u64); 2] = [
    (
        "Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
        39_799_117_984,
    ),
    (
        "Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf",
        26_750_834_816,
    ),
];

fn architecture_of(url: &str) -> String {
    let source = HttpRangeSource::new(url);
    let header = fetch_gguf_header(&source).expect("fetch GGUF header");
    header
        .architecture()
        .expect("general.architecture")
        .to_string()
}

#[test]
#[ignore = "network: reads a header per planned architecture"]
fn every_planned_row_still_matches_its_witness() {
    for (key, planned) in planned_gguf_architectures() {
        let found = architecture_of(planned.witness);
        println!("{key:<12} <- {found:<12} {}", planned.witness);
        assert_eq!(
            found, key,
            "registry row {key:?} disagrees with its witness {}",
            planned.witness
        );
    }
}

#[test]
#[ignore = "network: reads the Swift Qwen3.8 GGUF header, not its tensor payload"]
fn swift_qwen38_header_pins_the_gguf_conversion_gates() {
    let headers: Vec<_> = SWIFT_QWEN38_IQ2_XS_SHARDS
        .iter()
        .map(|url| fetch_gguf_header(&HttpRangeSource::new(*url)).expect("fetch shard header"))
        .collect();
    println!(
        "shard headers: {:?}",
        headers
            .iter()
            .map(|header| (
                header.architecture(),
                header.tensors.len(),
                header.metadata.len()
            ))
            .collect::<Vec<_>>()
    );
    assert_eq!(headers[0].architecture(), Some("qwen4exp"));
    assert_eq!(
        headers[1].architecture(),
        None,
        "continuation shard has split metadata only"
    );
    assert_eq!(headers[0].metadata.len(), 75);
    assert_eq!(
        headers.iter().map(|h| h.tensors.len()).sum::<usize>(),
        1_224
    );
    let metadata = &headers[0].metadata;
    assert_eq!(metadata["qwen4exp.block_count"].as_u64(), Some(48));
    assert_eq!(metadata["qwen4exp.expert_used_count"].as_u64(), Some(10));
    assert_eq!(
        gguf_arch_support("qwen4exp"),
        Some(ArchSupport::Supported(ModelFamily::Qwen4Exp))
    );

    let tensor = |name: &str| {
        headers
            .iter()
            .find_map(|header| header.tensors.get(name))
            .unwrap_or_else(|| panic!("missing witnessed tensor {name}"))
    };
    for name in headers.iter().flat_map(|header| header.tensors.keys()) {
        if name == "per_layer_token_embd.weight" {
            assert_eq!(
                map_gguf_name(name, ModelFamily::Qwen4Exp).unwrap(),
                GgufMapping::Ignored {
                    reason: "streamed to the Qwen4Exp IQ4_NL n-gram row store"
                }
            );
        } else {
            assert!(
                map_gguf_name(name, ModelFamily::Qwen4Exp).is_ok(),
                "unmapped published Qwen4Exp tensor {name}"
            );
        }
    }
    let out_proj = tensor("blk.0.ssm_out.weight");
    assert_eq!(ggml_type_name(out_proj.ggml_type), Some("IQ4_XS"));
    assert_eq!(out_proj.dims.as_slice(), [6_144, 2_560]);
    let (head_block_elems, _) = ggml_type_block(out_proj.ggml_type).expect("IQ4_XS block");
    assert_eq!(head_block_elems, 256);
    assert_eq!(6_144, 48 * 128, "48 128-wide V heads fill this axis");
    assert_ne!(128 % head_block_elems, 0, "a V head splits an IQ4_XS block");

    let ple = tensor("per_layer_token_embd.weight");
    assert_eq!(ggml_type_name(ple.ggml_type), Some("IQ4_NL"));
    assert_eq!(ple.dims.as_slice(), [160, 320_001_536]);

    let down = tensor("blk.0.ffn_down_exps.weight");
    assert_eq!(ggml_type_name(down.ggml_type), Some("Q2_0"));
    let gate = tensor("blk.0.ffn_gate_exps.weight");
    assert_eq!(ggml_type_name(gate.ggml_type), Some("IQ2_S"));
    let up = tensor("blk.0.ffn_up_exps.weight");
    assert_eq!(ggml_type_name(up.ggml_type), Some("IQ2_S"));
}

#[test]
#[ignore = "network: reads the pinned Q2_0 tier's two GGUF headers, not its tensor payload"]
fn swift_qwen38_q2_0_shards_derive_the_supported_qwen4exp_architecture() {
    let shards = SWIFT_QWEN38_Q2_0_SHARDS
        .iter()
        .map(|(name, bytes)| {
            let url = format!(
                "https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/{SWIFT_QWEN38_Q2_0_REVISION}/{name}"
            );
            let source = HttpRangeSource::new(url);
            let header = fetch_gguf_header(&source).expect("fetch Q2_0 shard header");
            (header, source, *bytes)
        })
        .collect();
    let set = GgufSet::new(shards).expect("complete Q2_0 shard set");
    assert_eq!(set.header.tensors.len(), 1_224);
    assert_eq!(set.header.architecture(), Some("qwen4exp"));
    let mut type_counts = std::collections::BTreeMap::new();
    let mut by_role = std::collections::BTreeMap::new();
    for (name, tensor) in &set.header.tensors {
        let kind = ggml_type_name(tensor.ggml_type)
            .unwrap_or_else(|| panic!("unknown ggml type {} on {name}", tensor.ggml_type));
        *type_counts.entry(kind).or_insert(0usize) += 1;
        let role = match map_gguf_name(name, ModelFamily::Qwen4Exp).expect("map tensor") {
            GgufMapping::Resident(_) => "resident",
            GgufMapping::Routed { .. } | GgufMapping::RoutedFusedGateUp { .. } => "routed",
            GgufMapping::Ignored { .. } => "ignored",
        };
        *by_role.entry((role, kind)).or_insert(0usize) += 1;
    }
    assert_eq!(type_counts["Q2_0"], 202);
    assert_eq!(type_counts["Q4_0"], 16);
    assert_eq!(type_counts["Q5_0"], 7);
    assert_eq!(type_counts["F16"], 1);
    assert_eq!(by_role.get(&("routed", "Q2_0")), Some(&144));
    assert_eq!(by_role.get(&("resident", "Q2_0")), Some(&58));
    assert_eq!(by_role.get(&("resident", "Q4_0")), Some(&16));
    assert_eq!(by_role.get(&("resident", "Q5_0")), Some(&7));
    assert_eq!(by_role.get(&("resident", "F16")), Some(&1));
    println!("Q2_0 tier type counts by role: {by_role:?}");
    let mut ssm_out_types = std::collections::BTreeMap::new();
    for layer in 0..48 {
        if let Some(tensor) = set
            .header
            .tensors
            .get(&format!("blk.{layer}.ssm_out.weight"))
        {
            assert_eq!(tensor.dims.as_slice(), [6_144, 2_560]);
            *ssm_out_types
                .entry(ggml_type_name(tensor.ggml_type).unwrap_or("unknown"))
                .or_insert(0usize) += 1;
        }
    }
    assert_eq!(
        ssm_out_types,
        std::collections::BTreeMap::from([
            ("IQ4_XS", 17),
            ("Q3_K", 3),
            ("Q4_K", 11),
            ("Q5_K", 3),
            ("Q6_K", 2),
        ])
    );

    let arch = arch_from_gguf(&set.header).expect("derive Qwen4Exp config from Q2_0 header");
    assert_eq!(arch.family, ModelFamily::Qwen4Exp);
    assert_eq!(arch.num_layers, 48);
    assert_eq!(arch.hidden_size, 2_560);
    assert_eq!(arch.num_experts, 512);
    assert_eq!(arch.top_k_experts, 10);
    assert_eq!(arch.ple.layer_ids, vec![2]);
    assert_eq!(arch.ple.layer_indices(), vec![1]);
    assert_eq!(
        map_gguf_name("blk.1.ple_key.weight", ModelFamily::Qwen4Exp).unwrap(),
        GgufMapping::Resident("language_model.model.layers.1.ple.key_proj.weight".to_string())
    );
    let multiplier_count = set.header.metadata["qwen4exp.ple.layer_multipliers"]
        .as_array()
        .expect("PLE hash multipliers are an array")
        .len();
    assert_eq!(multiplier_count, arch.ple.ngram_size as usize);

    for layer in 0..48 {
        let down = &set.header.tensors[&format!("blk.{layer}.ffn_down_exps.weight")];
        assert_eq!(ggml_type_name(down.ggml_type), Some("Q2_0"));
        for role in ["gate", "up"] {
            let tensor = &set.header.tensors[&format!("blk.{layer}.ffn_{role}_exps.weight")];
            let name = ggml_type_name(tensor.ggml_type)
                .unwrap_or_else(|| panic!("unknown routed type {}", tensor.ggml_type));
            assert_eq!(name, "Q2_0", "unexpected {role} type at layer {layer}");
        }
    }

    for name in set.header.tensors.keys() {
        assert!(
            map_gguf_name(name, ModelFamily::Qwen4Exp).is_ok(),
            "unmapped Q2_0-tier tensor {name}"
        );
    }
}

#[test]
#[ignore = "network: reads one header off a 26 GB remote checkpoint"]
fn mixtral_is_the_llama_architecture_with_experts() {
    let source = HttpRangeSource::new(MIXTRAL_8X7B);
    let header = fetch_gguf_header(&source).expect("fetch GGUF header");
    let arch = header.architecture().expect("general.architecture");
    assert_eq!(arch, "llama", "Mixtral is no longer the llama architecture");
    assert!(
        matches!(gguf_arch_support(arch), Some(ArchSupport::Supported(_))),
        "llama gained a decode flow in ROADMAP Phase M2"
    );

    // The MoE is in the metadata, not in the architecture string. Printed
    // rather than asserted on an exact count: the claim under test is that
    // an expert count EXISTS here and does not on a dense Llama.
    let experts = header
        .metadata
        .get("llama.expert_count")
        .and_then(|v| v.as_u64());
    println!("llama.expert_count = {experts:?}");
    assert!(
        experts.is_some_and(|n| n > 1),
        "a Mixtral header should carry an expert count"
    );
}
