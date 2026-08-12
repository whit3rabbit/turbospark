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
    // THE BIAS PLANES ARE COMPANIONS, NOT A QUANTIZATION SCHEME, and counting
    // them here put `F32` in the routed slot's type list and stopped a
    // perfectly runnable `gpt-oss` install at `validate_quant` (ROADMAP M5).
    //
    // The slot answers one question -- "does this type have the kernels its
    // slot needs" -- and a bias needs none: `moe_gguf.metal` reads it as a
    // plain `device const float*` off an offset the WEIGHT's kernel already
    // resolved. It is the same relationship the affine layout's `gate_scales`
    // has to its packed run, and those have never been counted either; they
    // only escaped notice because an affine blob's sources are not GGUF
    // tensors at all, so this loop never saw one.
    //
    // Keyed on the ROLE rather than on the dtype (`!= F32`) so that a future
    // checkpoint shipping BF16 biases, or an F32 weight, is classified by what
    // the sub-tensor IS rather than by what it happens to be stored as.
    let mut routed_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for sources in plan.routed.values() {
        for s in sources {
            if s.roles.iter().all(|r| r.ends_with("_biases")) {
                continue;
            }
            *routed_counts
                .entry(ggml_scheme_name(header.tensors[s.name].ggml_type))
                .or_default() += 1;
        }
    }
    let mut routed_types: Vec<&str> = routed_counts.keys().copied().collect();
    routed_types.sort_by_key(|t| std::cmp::Reverse(routed_counts[t]));
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
    let attention = type_of(&["attn_q.weight", "attn_qkv.weight"]);

    // EVERY SLOT NAMING A COMPONENT THE MODEL DOES NOT HAVE FALLS BACK TO
    // THE ATTENTION TYPE, and this is the rule rather than three patches.
    //
    // `manifest.quant` has five fixed slots and no architecture has all five.
    // A probe that finds nothing answers "absent", `validate_quant` refuses
    // "absent" because it is not a block type with a kernel, and a perfectly
    // runnable install fails to load with a message about a component it
    // never had. That has now happened three times, each found by the next
    // real checkpoint: the Qwen hybrid wrote `absent` for attention and
    // shared expert because it probed hardcoded `blk.0.` names on a model
    // whose layer 0 has no `attn_q` (AGENTS.md); Mixtral has no shared expert
    // at all; and ROADMAP M4's dense Mistral has neither a router nor routed
    // experts.
    //
    // Attention is the right fallback because it is executable exactly when
    // the install is, and because the resident core is what a reader would
    // take an inapplicable slot to describe. It is a defaulted statement, not
    // a measured one, so anything reading these slots for a DISPATCH must
    // read the resident index or `packed_experts/layout.json` instead
    // (crate Gotcha 10 already requires that).
    //
    // Worth knowing why the dense case surfaced only now: `is_production_arch`
    // keys on (num_layers, hidden_size), and Mistral 7B's (32, 4096) is
    // Mixtral 8x7B's exactly, so it is the first dense file held to the
    // production manifest rules at all. TinyLlama's (22, 2048) matches no
    // baseline and skips the check entirely.
    let or_attention = |found: &'static str| if found == "absent" { attention } else { found };

    let router_source = or_attention(type_of(&["ffn_gate_inp.weight"]));
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
    let shared = or_attention(type_of(&["ffn_gate.weight", "ffn_gate_shexp.weight"]));
    let routed_types: Vec<&str> = if routed_types.is_empty() {
        vec![attention]
    } else {
        routed_types
    };
    let routed_slot = {
        let mut v = slot(routed_types[0]);
        if routed_types.len() > 1 {
            v["ggmlTypes"] = serde_json::json!(routed_types);
        }
        v
    };
    serde_json::json!({
        "embedding": slot(type_of(&["token_embd.weight"])),
        "attention": slot(attention),
        "router": router,
        "sharedExpert": slot(shared),
        "routedExpert": routed_slot,
    })
}
