//! Tensor generators for synthetic `qwen4_exp` decode layers.

use std::collections::HashMap;

use crate::synthetic_tensors::{
    bf16_vector, deterministic_row, expert_int4_triple, int4_triple, int8_triple, u16_le, Tensor,
};

use super::arch::*;
use super::ngram::{ple_layer_tensors, PLE_LAYER};

/// One hyper-connection block's four tensors under `prefix` (either
/// `attn_hyper_connection` or `mlp_hyper_connection`, per layer).
fn hyper_connection_tensors(prefix: &str, seed: u64) -> Vec<Tensor> {
    let wide = wide_dim();
    let mut ts = vec![bf16_vector(
        &format!("{prefix}.hc_norm.weight"),
        wide,
        // Centered convention: weight is an OFFSET FROM UNITY, initialized
        // near zero in the real checkpoint.
        0.0,
        seed,
    )];
    ts.extend(int4_triple(
        &format!("{prefix}.input_mix_weight_down.weight"),
        HC_LOWRANK,
        wide,
        seed + 1,
    ));
    ts.extend(int4_triple(
        &format!("{prefix}.input_mix_weight_up.weight"),
        wide,
        HC_LOWRANK,
        seed + 2,
    ));
    ts.extend(int4_triple(
        &format!("{prefix}.block_inject_weight.weight"),
        HC_COUNT,
        wide,
        seed + 3,
    ));
    ts
}

/// The final `hyper_connection_mixer`: the same `hc_norm` +
/// `input_mix_weight_{down,up}` triple, with NO `block_inject_weight`
/// (`use_combine=False`).
fn final_mixer_tensors() -> Vec<Tensor> {
    let wide = wide_dim();
    let prefix = "language_model.model.hyper_connection_mixer";
    let mut ts = vec![bf16_vector(
        &format!("{prefix}.hc_norm.weight"),
        wide,
        0.0,
        8_000,
    )];
    ts.extend(int4_triple(
        &format!("{prefix}.input_mix_weight_down.weight"),
        HC_LOWRANK,
        wide,
        8_001,
    ));
    ts.extend(int4_triple(
        &format!("{prefix}.input_mix_weight_up.weight"),
        wide,
        HC_LOWRANK,
        8_002,
    ));
    ts
}

pub(super) fn conv1d_weight(name: &str, channels: usize, taps: usize, seed: u64) -> Tensor {
    let flat = bf16_vector(name, channels * taps, 0.0, seed);
    Tensor {
        name: flat.name,
        dtype: flat.dtype,
        shape: vec![channels as u64, taps as u64, 1],
        bytes: flat.bytes,
    }
}

fn gdn_layer_tensors(p: &str, seed: u64) -> Vec<Tensor> {
    let qkv_dim = 2 * LA_K_HEADS * LA_KEY_DIM + LA_V_HEADS * LA_VALUE_DIM;
    let value_dim = LA_V_HEADS * LA_VALUE_DIM;
    let mut ts = Vec::new();
    for (name, rows, cols, s) in [
        ("in_proj_qkv", qkv_dim, HIDDEN, 1u64),
        ("in_proj_z", value_dim, HIDDEN, 2),
        ("in_proj_a", LA_V_HEADS, HIDDEN, 3),
        ("in_proj_b", LA_V_HEADS, HIDDEN, 4),
        ("out_proj", HIDDEN, value_dim, 5),
    ] {
        ts.extend(int4_triple(
            &format!("{p}.linear_attn.{name}.weight"),
            rows,
            cols,
            seed + s,
        ));
    }
    ts.push(conv1d_weight(
        &format!("{p}.linear_attn.conv1d.weight"),
        qkv_dim,
        LA_CONV_K,
        seed + 10,
    ));
    // NO `.weight` suffix on these two, matching the real checkpoint
    // (`crates/repack/CLAUDE.md` Gotcha 15).
    ts.push(bf16_vector(
        &format!("{p}.linear_attn.A_log"),
        LA_V_HEADS,
        0.0,
        seed + 11,
    ));
    ts.push(bf16_vector(
        &format!("{p}.linear_attn.dt_bias"),
        LA_V_HEADS,
        0.0,
        seed + 12,
    ));
    // PLAIN (not centered) gated norm, per `docs/QWEN4_PHASE0.md` item 9's
    // one exception -- weight initialized near ONE, not zero.
    ts.push(bf16_vector(
        &format!("{p}.linear_attn.norm.weight"),
        LA_VALUE_DIM,
        1.0,
        seed + 13,
    ));
    ts
}

fn attention_layer_tensors(p: &str, seed: u64) -> Vec<Tensor> {
    let q_out = NUM_HEADS * HEAD_DIM;
    let kv_out = NUM_KV_HEADS * HEAD_DIM;
    let mut ts = Vec::new();
    ts.extend(int4_triple(
        &format!("{p}.self_attn.q_proj.weight"),
        2 * q_out,
        HIDDEN,
        seed + 1,
    ));
    for (i, role) in ["k_proj", "v_proj"].iter().enumerate() {
        ts.extend(int4_triple(
            &format!("{p}.self_attn.{role}.weight"),
            kv_out,
            HIDDEN,
            seed + 2 + i as u64,
        ));
    }
    ts.extend(int4_triple(
        &format!("{p}.self_attn.o_proj.weight"),
        HIDDEN,
        q_out,
        seed + 4,
    ));
    // CENTERED unconditionally -- this family has no plain-norm sibling
    // sharing this call (`families/qwen4/attn.rs`'s own doc).
    for (i, norm) in ["q_norm", "k_norm"].iter().enumerate() {
        ts.push(bf16_vector(
            &format!("{p}.self_attn.{norm}.weight"),
            HEAD_DIM,
            0.0,
            seed + 20 + i as u64,
        ));
    }
    // The QSA indexer (`docs/QWEN4_PHASE0.md` section 5): one packed
    // `[query heads; key head]` projection plus two per-head CENTERED norms,
    // the same three tensors the real install carries per QSA layer (INT4
    // projection, BF16 norms). Rows `[IDX_HEADS * IDX_HEAD_DIM ..)` are the
    // shared raw key the indexer cache stores.
    ts.extend(int4_triple(
        &format!("{p}.self_attn.indexer.index_qk_proj.weight"),
        (IDX_HEADS + IDX_KV_HEADS) * IDX_HEAD_DIM,
        HIDDEN,
        seed + 30,
    ));
    for (i, norm) in ["q_layernorm", "k_layernorm"].iter().enumerate() {
        ts.push(bf16_vector(
            &format!("{p}.self_attn.indexer.{norm}.weight"),
            IDX_HEAD_DIM,
            0.0,
            seed + 31 + i as u64,
        ));
    }
    ts
}

/// A raw (unquantized) BF16 router matrix, matching what the real
/// `qwen4_exp` checkpoint ships instead of the pre-packed INT8 [`int8_triple`]
/// produces. Router-shaped only: `int8_triple`'s `.weight`/`.scales`/`.biases`
/// triple is what `pass_through_packed` reads, and a raw tensor has no
/// companions to build.
fn bf16_matrix(name: &str, rows: usize, cols: usize, seed: u64) -> Tensor {
    let mut bits = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let row = deterministic_row(seed.wrapping_add(r as u64 * 97 + 1), cols);
        bits.extend(row.iter().map(|&v| compute::f32_to_bf16(v)));
    }
    Tensor {
        name: name.to_string(),
        dtype: "BF16",
        shape: vec![rows as u64, cols as u64],
        bytes: u16_le(&bits),
    }
}

/// `router_raw` ships BOTH of this family's small gating matrices -- the
/// router `mlp.gate.weight` and the shared expert's sigmoid gate
/// `mlp.shared_expert_gate.weight` -- as [`bf16_matrix`] instead of
/// pre-packed [`int8_triple`], matching the real REAP-288 checkpoint
/// (`crates/repack/CLAUDE.md` Gotcha 6's router-transcode pattern applied on
/// the safetensors side, `orchestrate.rs`'s `quantize_gating_matrix_int8`)
/// rather than every OTHER safetensors MoE checkpoint this walk has seen,
/// which ships them already `U32`-packed. **Both, not just the router**: an
/// earlier version of this fixture shipped only the router raw and kept the
/// shared-expert gate pre-packed, which passed every test here while the
/// real checkpoint still failed at a DIFFERENT dispatch site ("no dispatched
/// GEMV kernel" on `mlp.shared_expert_gate.weight`) after the router fix
/// alone. No `bits_overrides` entry is inserted for either, matching the
/// real checkpoint's `config.json`, which declares none for either.
fn moe_tensors(
    p: &str,
    seed: u64,
    overrides: &mut HashMap<String, u32>,
    router_raw: bool,
) -> Vec<Tensor> {
    let mut ts = Vec::new();
    if router_raw {
        ts.push(bf16_matrix(
            &format!("{p}.mlp.gate.weight"),
            NUM_EXPERTS,
            HIDDEN,
            seed + 50,
        ));
        ts.push(bf16_matrix(
            &format!("{p}.mlp.shared_expert_gate.weight"),
            1,
            HIDDEN,
            seed + 51,
        ));
    } else {
        ts.extend(int8_triple(
            &format!("{p}.mlp.gate.weight"),
            NUM_EXPERTS,
            HIDDEN,
            seed + 50,
        ));
        overrides.insert(format!("{p}.mlp.gate"), 8);
        ts.extend(int8_triple(
            &format!("{p}.mlp.shared_expert_gate.weight"),
            1,
            HIDDEN,
            seed + 51,
        ));
        overrides.insert(format!("{p}.mlp.shared_expert_gate"), 8);
    }
    for (i, role) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
        let (rows, cols) = if *role == "down_proj" {
            (HIDDEN, SHARED_INTER)
        } else {
            (SHARED_INTER, HIDDEN)
        };
        ts.extend(int8_triple(
            &format!("{p}.mlp.shared_expert.{role}.weight"),
            rows,
            cols,
            seed + 60 + i as u64,
        ));
        overrides.insert(format!("{p}.mlp.shared_expert.{role}"), 8);
        let (erows, ecols) = if *role == "down_proj" {
            (HIDDEN, MOE_INTER)
        } else {
            (MOE_INTER, HIDDEN)
        };
        ts.extend(expert_int4_triple(
            &format!("{p}.mlp.switch_mlp.{role}.weight"),
            NUM_EXPERTS,
            erows,
            ecols,
            seed + 70 + i as u64,
        ));
    }
    ts
}

pub(super) fn build_tensors(
    vocab_size: i64,
    router_raw: bool,
) -> (Vec<Tensor>, HashMap<String, u32>) {
    let vocab = vocab_size as usize;
    let mut overrides = HashMap::new();
    let mut ts = Vec::new();
    ts.extend(int4_triple(
        "language_model.model.embed_tokens.weight",
        vocab,
        HIDDEN,
        1001,
    ));
    ts.extend(int4_triple(
        "language_model.lm_head.weight",
        vocab,
        HIDDEN,
        1002,
    ));

    let mask = layer_mask();
    for (l, &m) in mask.iter().enumerate() {
        let p = format!("language_model.model.layers.{l}");
        let seed = 1000 * (l as u64 + 1);
        ts.extend(hyper_connection_tensors(
            &format!("{p}.attn_hyper_connection"),
            seed + 100,
        ));
        ts.extend(hyper_connection_tensors(
            &format!("{p}.mlp_hyper_connection"),
            seed + 200,
        ));
        if m == 1 {
            ts.extend(attention_layer_tensors(&p, seed));
        } else {
            ts.extend(gdn_layer_tensors(&p, seed));
        }
        ts.extend(moe_tensors(&p, seed, &mut overrides, router_raw));
        if l == PLE_LAYER {
            ts.extend(ple_layer_tensors(&p, vocab_size));
        }
    }
    ts.extend(final_mixer_tensors());
    (ts, overrides)
}
