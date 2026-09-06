//! PLE and n-gram table tensors and group-32 quantization for `qwen4_exp`.

use crate::synthetic_tensors::{bf16_vector, deterministic_row, int4_triple, u16_le, Tensor};

use super::arch::{wide_dim, HIDDEN};
use super::tensors::conv1d_weight;

/// `(ngram_size - 1) * heads_per_ngram` = `2 * 2`.
pub const NGRAM_HEADS: usize = 4;
/// One whole group and nothing more (`NGRAM_GROUP_SIZE` in
/// `gemma4_checkpoint::ngram`), so `ple_embed_dim` comes out to
/// `NGRAM_HEADS * NGRAM_HEAD_DIM` = 128 = [`HIDDEN`].
pub const NGRAM_HEAD_DIM: usize = 32;
pub const NGRAM_GROUP_SIZE: usize = 32;
pub const PLE_LAYER: usize = 1;
pub const NGRAM_EOS_TOKEN_ID: i64 = 999;
/// Small vocabulary base so the derived prime head-vocab sizes stay in the
/// tens rather than the tens of millions -- the fixture's table has to
/// physically hold every row the hash can address.
pub const NGRAM_VOCAB_BASE: i64 = 50;

/// One group's INT4-affine-at-[`NGRAM_GROUP_SIZE`] quantization, the
/// n-gram table's own layout -- distinct from
/// `compute::quantize_int4_affine`, which is fixed at group 64 and cannot
/// express this table's group 32. The inverse of
/// `compute::dequant_ngram_row`: same nibble order (low nibble first,
/// `byte_idx = elem / 2`, group-independent since every group here is a
/// whole number of bytes).
fn quantize_ngram_row(row: &[f32]) -> (Vec<u8>, Vec<u16>, Vec<u16>) {
    assert_eq!(row.len() % NGRAM_GROUP_SIZE, 0);
    let groups = row.len() / NGRAM_GROUP_SIZE;
    let mut packed = vec![0u8; row.len().div_ceil(2)];
    let mut scales = Vec::with_capacity(groups);
    let mut biases = Vec::with_capacity(groups);
    for g in 0..groups {
        let slice = &row[g * NGRAM_GROUP_SIZE..(g + 1) * NGRAM_GROUP_SIZE];
        let lo = slice.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let scale = ((hi - lo) / 15.0).max(1e-6);
        scales.push(compute::f32_to_bf16(scale));
        biases.push(compute::f32_to_bf16(lo));
        for (k, &v) in slice.iter().enumerate() {
            let q = (((v - lo) / scale).round().clamp(0.0, 15.0)) as u8;
            let elem = g * NGRAM_GROUP_SIZE + k;
            let byte_idx = elem / 2;
            if elem & 1 == 0 {
                packed[byte_idx] = (packed[byte_idx] & 0xF0) | q;
            } else {
                packed[byte_idx] = (packed[byte_idx] & 0x0F) | (q << 4);
            }
        }
    }
    (packed, scales, biases)
}

/// The n-gram table: ONE shard (the row/shard split is proven by
/// `synthetic_qwen::qwen4`'s own fixture; this one's job is REAL,
/// dequantizable data), with real quantized rows and prime-sized
/// per-head vocabularies small enough that the fixture's single shard can
/// address every row the hash reaches.
fn ngram_tensors(p: &str, vocab_size: i64) -> Vec<Tensor> {
    let head_vocab_sizes: Vec<i64> = (0..NGRAM_HEADS as i64)
        .map(|i| model_io::find_nth_prime_after(NGRAM_VOCAB_BASE - 1, i + 1))
        .collect();
    let mut running = 0i64;
    let head_offsets: Vec<i64> = head_vocab_sizes
        .iter()
        .map(|&s| {
            let o = running;
            running += s;
            o
        })
        .collect();
    let total_rows = running as usize;

    let mut weight = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for r in 0..total_rows {
        let row = deterministic_row(9_000 + r as u64, NGRAM_HEAD_DIM);
        let (pk, sc, bi) = quantize_ngram_row(&row);
        weight.extend_from_slice(&pk);
        scales.extend_from_slice(&u16_le(&sc));
        biases.extend_from_slice(&u16_le(&bi));
    }
    let weight_bytes_per_row = NGRAM_HEAD_DIM / 2;
    let groups_per_row = NGRAM_HEAD_DIM / NGRAM_GROUP_SIZE;

    let prefix = format!("{p}.ple.ple_embedding");
    let mut ts = vec![
        Tensor {
            name: format!("{prefix}.ngram_embedding.shard_0.weight"),
            dtype: "U32",
            shape: vec![total_rows as u64, (weight_bytes_per_row / 4).max(1) as u64],
            bytes: weight,
        },
        Tensor {
            name: format!("{prefix}.ngram_embedding.shard_0.scales"),
            dtype: "BF16",
            shape: vec![total_rows as u64, groups_per_row as u64],
            bytes: scales,
        },
        Tensor {
            name: format!("{prefix}.ngram_embedding.shard_0.biases"),
            dtype: "BF16",
            shape: vec![total_rows as u64, groups_per_row as u64],
            bytes: biases,
        },
    ];

    // The DERIVED multipliers, matching the frozen-constant cross-check
    // (`crates/model-io/src/ngram_hash.rs`): using the real derivation
    // here rather than arbitrary constants is what lets a test compute the
    // expected hash independently and compare.
    let multipliers = model_io::build_layer_multipliers(vocab_size, 3, 0, 1234);
    ts.push(i64_tensor(
        &format!("{prefix}.layer_multipliers"),
        &multipliers,
    ));
    ts.push(i64_tensor(
        &format!("{prefix}.ngram_heads_offsets"),
        &head_offsets,
    ));
    ts.push(i64_tensor(
        &format!("{prefix}.ngram_heads_vocab_sizes"),
        &head_vocab_sizes,
    ));
    ts
}

fn i64_tensor(name: &str, values: &[i64]) -> Tensor {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Tensor {
        name: name.to_string(),
        dtype: "I64",
        shape: vec![values.len() as u64],
        bytes,
    }
}

pub(super) fn ple_layer_tensors(p: &str, vocab_size: i64) -> Vec<Tensor> {
    let wide = wide_dim();
    let mut ts = Vec::new();
    ts.extend(int4_triple(
        &format!("{p}.ple.key_proj.weight"),
        wide,
        HIDDEN,
        9_200,
    ));
    ts.extend(int4_triple(
        &format!("{p}.ple.value_proj.weight"),
        HIDDEN,
        HIDDEN,
        9_201,
    ));
    ts.push(conv1d_weight(
        &format!("{p}.ple.conv1d.weight"),
        wide,
        4,
        9_202,
    ));
    for (i, norm) in ["norm_key", "norm_query", "norm_conv"].iter().enumerate() {
        ts.push(bf16_vector(
            &format!("{p}.ple.{norm}.weight"),
            wide,
            0.0,
            9_210 + i as u64,
        ));
    }
    ts.extend(ngram_tensors(p, vocab_size));
    ts
}
