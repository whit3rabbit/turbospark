//! Pinned per-tensor role map for the dense GSQ-RCO IQ2_XS release.
//!
//! This is deliberately a checked-in text fixture rather than a copied list
//! in a test. The list proves the role assignment used to admit each new
//! resident dtype, including the last-layer exception, and makes a changed
//! upstream allocation visible as a diff.

use std::collections::BTreeMap;

const ALLOCATION: &str = include_str!("fixtures/qwen38_gsq_rco_iq2_xs_allocation.txt");

fn roles() -> BTreeMap<&'static str, &'static str> {
    ALLOCATION
        .lines()
        .filter_map(|line| line.split_once(": "))
        .filter(|(name, _)| !name.starts_with('#'))
        .collect()
}

#[test]
fn pinned_map_has_the_declared_dense_type_histogram() {
    assert!(ALLOCATION.contains("# Pinned revision: b71b542000a34ed0eb1e6a3a92f906e593728c65"));
    let mut counts = BTreeMap::<&str, usize>::new();
    for dtype in roles().into_values() {
        *counts.entry(dtype).or_default() += 1;
    }
    assert_eq!(counts["Q2_K"], 53);
    assert_eq!(counts["IQ2_XXS"], 70);
    assert_eq!(counts["IQ2_XS"], 53);
    assert_eq!(counts["IQ1_S"], 36);
    assert_eq!(counts["IQ3_S"], 47);
    assert_eq!(counts["IQ2_S"], 60);
    assert_eq!(counts["IQ1_M"], 31);
    assert_eq!(roles().len(), 851);
}

#[test]
fn resident_roles_cover_embedding_head_attention_and_last_layer_exception() {
    let map = roles();
    assert_eq!(map["token_embd.weight"], "IQ1_M");
    assert_eq!(map["output.weight"], "IQ4_XS");
    assert_eq!(map["blk.0.attn_qkv.weight"], "IQ3_S");
    assert_eq!(map["blk.0.attn_gate.weight"], "IQ3_S");
    assert_eq!(map["blk.0.ffn_gate.weight"], "IQ1_S");
    assert_eq!(map["blk.0.ffn_up.weight"], "IQ1_M");
    assert_eq!(map["blk.0.ffn_down.weight"], "IQ2_XXS");
    assert_eq!(map["blk.0.ssm_out.weight"], "IQ4_XS");
    assert_eq!(map["blk.45.ffn_down.weight"], "Q2_K");
}
