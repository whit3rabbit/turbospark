//! GGUF manifest quantization spec generation.

use std::collections::BTreeMap;

use super::plan::Plan;
use super::types::ggml_scheme_name;
use crate::gguf_header::GgufHeader;

/// The `manifest.json -> quant` object for a GGUF-sourced install.
pub fn gguf_manifest_quant(header: &GgufHeader, plan: &Plan<'_>) -> serde_json::Value {
    let type_of = |suffixes: &[&str]| {
        header
            .tensors
            .iter()
            .find(|(name, _)| suffixes.iter().any(|s| name.ends_with(s)))
            .map(|(_, i)| ggml_scheme_name(i.ggml_type))
            .unwrap_or("absent")
    };
    let mut routed_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for sources in plan.routed.values() {
        for s in sources {
            *routed_counts
                .entry(ggml_scheme_name(header.tensors[s.name].ggml_type))
                .or_default() += 1;
        }
    }
    let mut routed_types: Vec<&str> = routed_counts.keys().copied().collect();
    routed_types.sort_by_key(|t| std::cmp::Reverse(routed_counts[t]));
    if routed_types.is_empty() {
        routed_types.push("absent");
    }
    let slot = |ggml: &str| {
        serde_json::json!({
            "weightBits": 0,
            "scheme": "gguf",
            "ggmlType": ggml,
            "scaleType": "inline",
            "biasType": "inline",
            "groupSize": 0,
        })
    };
    let routed_slot = {
        let mut v = slot(routed_types[0]);
        if routed_types.len() > 1 {
            v["ggmlTypes"] = serde_json::json!(routed_types);
        }
        v
    };
    let router_source = type_of(&["ffn_gate_inp.weight"]);
    let router = if router_source == "F32" {
        serde_json::json!({
            "weightBits": 8,
            "scheme": "affine",
            "scaleType": "bf16",
            "biasType": "bf16",
            "groupSize": 64,
        })
    } else {
        slot(router_source)
    };
    serde_json::json!({
        "embedding": slot(type_of(&["token_embd.weight"])),
        "attention": slot(type_of(&["attn_q.weight", "attn_qkv.weight"])),
        "router": router,
        "sharedExpert": slot(type_of(&["ffn_gate.weight", "ffn_gate_shexp.weight"])),
        "routedExpert": routed_slot,
    })
}
