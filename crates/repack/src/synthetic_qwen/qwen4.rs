//! Builds a tiny `qwen4_exp` install through the REAL checkpoint repack
//! pipeline, carrying `qwen4_exp`'s hashed n-gram PLE table
//! (`gemma4_checkpoint::ngram`).
//!
//! **THIS FIXTURE EXISTS TO PROVE THE N-GRAM TABLE'S WRITER WIRING, NOT
//! `qwen4_exp`'s DECODE FLOW.** It reuses `dense`'s per-layer tensor shapes
//! for the one ordinary layer it builds, and the PLE layer carries ONLY the
//! n-gram tensors -- no attention, no MLP, no `ple.{key_proj,...}` submodule
//! weights. Nothing here asserts on decode, and `arch.num_experts` stays 0
//! (dense-shaped): this walk emits no `.mlp.switch_mlp.` names, so
//! `plan.routed` is empty and the install writes through the zero-layer
//! dense path exactly as `qwen3_5`'s does. A fixture exercising the routed
//! half, the hyper-connections, or the QSA indexer belongs to a later phase.

use std::collections::HashMap;

use model_io::{ArchConfig, ModelFamily, PleConfig};

use super::dense_arch::{group_for, HEAD_DIM, HIDDEN, INTER, NUM_HEADS, NUM_KV_HEADS};
use super::dense_tensors::{f16_vector, packed_triple};
use crate::gemma4_checkpoint::{
    write_gemma4_install, write_gemma4_install_streamed, Gemma4Quant, Gemma4Shards,
};
use crate::ranged_download::MemoryRangeSource;
use crate::safetensors_header::parse_header;
use crate::synthetic_tensors::Tensor;

/// Shards in the PLE table this fixture builds. Deliberately not 1: `1` is
/// where a shard-count bug and a row-count bug read the same, since there is
/// nothing to be OUT of order with.
const NGRAM_SHARDS: usize = 3;
/// Rows per shard. Not equal to [`NGRAM_SHARDS`] and not a power of two, for
/// `qwen4_ngram_store.rs`'s reason: a writer confusing the two axes
/// disagrees with this fixture rather than coinciding with it.
const NGRAM_ROWS_PER_SHARD: usize = 2;
/// One whole group and nothing more: the smallest `head_dim`
/// `NgramTableSpec::validate` accepts at `NGRAM_GROUP_SIZE` (32).
const NGRAM_HEAD_DIM: usize = 32;
/// `(ngram_size - 1) * heads_per_ngram` = `2 * 2`.
const NGRAM_HEADS: usize = 4;
/// Which zero-based layer carries the table. `layer_ids` is one-based.
const PLE_LAYER: usize = 1;
/// Arbitrary but distinct from every real token id this fixture uses, so a
/// test can tell "the EOS-reset path fired" from "it read a real token".
const NGRAM_EOS_TOKEN_ID: i64 = 999;

/// A tiny `qwen4_exp`-shaped architecture with an active n-gram table.
///
/// Every field outside `family`, `num_experts` and `ple` takes
/// `tiny_qwen_gdn_dense_arch`'s own value -- this fixture is not exercising
/// the hyper-connection or QSA differences, only the ingest path every
/// family already shares.
pub fn tiny_qwen4_exp_arch(vocab_size: i64, num_layers: i64) -> ArchConfig {
    let mut arch = super::dense_arch::tiny_qwen_gdn_dense_arch(vocab_size, num_layers);
    arch.family = ModelFamily::Qwen4Exp;
    // Full attention throughout: this fixture's one ordinary layer does not
    // need to exercise the GDN branch, which `synthetic_qwen::dense`'s own
    // fixtures already cover.
    arch.full_attention_layer_mask = vec![1u8; num_layers as usize];
    arch.ple = PleConfig {
        ngram_size: 3,
        heads_per_ngram: 2,
        ngram_vocab_size_base: 1_000,
        make_divisible_by: 1,
        split_ngram_parts: NGRAM_SHARDS as i64,
        ple_embed_dim: (NGRAM_HEADS * NGRAM_HEAD_DIM) as i64,
        conv_kernel_size: 4,
        layer_ids: vec![PLE_LAYER as i64 + 1],
        seed: 1234,
        eos_token_id: NGRAM_EOS_TOKEN_ID,
    };
    arch
}

/// The n-gram table's tensors: [`NGRAM_SHARDS`] shards of `weight`/`scales`/
/// `biases`, plus the three hashing buffers.
///
/// Bytes are deterministic filler, not valid quantized data: `NgramTableWriter`
/// copies planes verbatim and never reads a value (see its module header), so
/// what this walk's tests can check is PLACEMENT -- which shard's bytes land
/// at which offset -- exactly as `qwen4_ngram_store.rs` does for the writer
/// alone. Each shard's weight plane is filled with its own shard index so a
/// misplaced shard is visible rather than merely a wrong number.
fn ngram_tensors() -> Vec<Tensor> {
    let weight_bytes = NGRAM_HEAD_DIM * 4 / 8; // bits = 4
    let groups = NGRAM_HEAD_DIM / 32; // NGRAM_GROUP_SIZE
    let companion_bytes = groups * 2; // one BF16 per group
    let prefix = format!("language_model.model.layers.{PLE_LAYER}.ple.ple_embedding");

    let mut ts = Vec::new();
    for shard in 0..NGRAM_SHARDS {
        let base = format!("{prefix}.ngram_embedding.shard_{shard}");
        ts.push(Tensor {
            name: format!("{base}.weight"),
            dtype: "U32",
            shape: vec![
                NGRAM_ROWS_PER_SHARD as u64,
                (weight_bytes / 4).max(1) as u64,
            ],
            bytes: vec![shard as u8; weight_bytes * NGRAM_ROWS_PER_SHARD],
        });
        for (role, fill) in [("scales", 0x50u8), ("biases", 0x60u8)] {
            ts.push(Tensor {
                name: format!("{base}.{role}"),
                dtype: "BF16",
                shape: vec![NGRAM_ROWS_PER_SHARD as u64, groups as u64],
                bytes: vec![fill.wrapping_add(shard as u8); companion_bytes * NGRAM_ROWS_PER_SHARD],
            });
        }
    }

    let total_rows = (NGRAM_SHARDS * NGRAM_ROWS_PER_SHARD) as i64;
    // Sizes/offsets underfill the table, as the real one's do
    // (`qwen4_ngram_store.rs`'s `write_table` records why).
    let head_size = 1i64;
    assert!(
        head_size * NGRAM_HEADS as i64 <= total_rows,
        "the fixture's heads must underfill its table"
    );
    let sizes: Vec<i64> = std::iter::repeat_n(head_size, NGRAM_HEADS).collect();
    let offsets: Vec<i64> = (0..NGRAM_HEADS as i64).map(|h| h * head_size).collect();
    ts.push(i64_tensor(
        &format!("{prefix}.layer_multipliers"),
        &[1, 3, 5],
    ));
    ts.push(i64_tensor(
        &format!("{prefix}.ngram_heads_offsets"),
        &offsets,
    ));
    ts.push(i64_tensor(
        &format!("{prefix}.ngram_heads_vocab_sizes"),
        &sizes,
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

/// One ordinary full-attention layer's tensors, at layer index 0. Shares
/// `synthetic_qwen::dense`'s widths so `packed_triple`'s group arithmetic
/// agrees with `Gemma4Quant`'s.
fn ordinary_layer_tensors(bits: u32) -> Vec<Tensor> {
    let p = "language_model.model.layers.0";
    let mut ts = Vec::new();
    for (i, norm) in ["input_layernorm", "post_attention_layernorm"]
        .iter()
        .enumerate()
    {
        ts.push(f16_vector(
            &format!("{p}.{norm}.weight"),
            HIDDEN,
            1.0,
            40 + i as u64,
        ));
    }
    ts.extend(packed_triple(
        &format!("{p}.self_attn.q_proj.weight"),
        2 * NUM_HEADS * HEAD_DIM,
        HIDDEN,
        1,
        bits,
    ));
    for (i, role) in ["k_proj", "v_proj"].iter().enumerate() {
        ts.extend(packed_triple(
            &format!("{p}.self_attn.{role}.weight"),
            NUM_KV_HEADS * HEAD_DIM,
            HIDDEN,
            2 + i as u64,
            bits,
        ));
    }
    ts.extend(packed_triple(
        &format!("{p}.self_attn.o_proj.weight"),
        HIDDEN,
        NUM_HEADS * HEAD_DIM,
        4,
        bits,
    ));
    for (i, norm) in ["q_norm", "k_norm"].iter().enumerate() {
        ts.push(f16_vector(
            &format!("{p}.self_attn.{norm}.weight"),
            HEAD_DIM,
            1.0,
            20 + i as u64,
        ));
    }
    for (i, role) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
        let (rows, cols) = if *role == "down_proj" {
            (HIDDEN, INTER)
        } else {
            (INTER, HIDDEN)
        };
        ts.extend(packed_triple(
            &format!("{p}.mlp.{role}.weight"),
            rows,
            cols,
            60 + i as u64,
            bits,
        ));
    }
    ts
}

fn build_tensors(vocab_size: i64, bits: u32) -> Vec<Tensor> {
    let vocab = vocab_size as usize;
    let mut ts = Vec::new();
    ts.extend(packed_triple(
        "language_model.model.embed_tokens.weight",
        vocab,
        HIDDEN,
        1001,
        bits,
    ));
    ts.extend(packed_triple(
        "language_model.lm_head.weight",
        vocab,
        HIDDEN,
        1002,
        bits,
    ));
    ts.extend(ordinary_layer_tensors(bits));
    ts.extend(ngram_tensors());
    ts.push(f16_vector(
        "language_model.model.norm.weight",
        HIDDEN,
        1.0,
        7,
    ));
    ts
}

fn quant(bits: u32) -> Gemma4Quant {
    Gemma4Quant {
        default_bits: bits,
        group_size: group_for(bits) as u32,
        bits_overrides: HashMap::new(),
    }
}

/// Writes a tiny `qwen4_exp` install through the NON-streamed writer
/// (`write_gemma4_install`) and returns the `ArchConfig` needed to open it.
pub fn build_synthetic_qwen4_exp_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    let arch = tiny_qwen4_exp_arch(vocab_size, num_layers);
    let bits = 4;
    let ts = build_tensors(vocab_size, bits);
    let blob = crate::synthetic_tensors::assemble_safetensors(&ts);
    let source = MemoryRangeSource::new(&blob);
    let header = parse_header(&blob, crate::safetensors_header::DEFAULT_MAX_HEADER_BYTES)?;
    write_gemma4_install(dir, &arch, model_id, &header, &source, &quant(bits))?;
    Ok(arch)
}

/// The same install through the STREAMED writer, which is the one every
/// real checkpoint takes -- and, per `gemma4_checkpoint::ngram`'s module
/// header, one of the two places the n-gram table is actually written.
pub fn build_synthetic_qwen4_exp_install_streamed(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    let arch = tiny_qwen4_exp_arch(vocab_size, num_layers);
    let bits = 4;
    let ts = build_tensors(vocab_size, bits);
    let blob = crate::synthetic_tensors::assemble_safetensors(&ts);
    let source = MemoryRangeSource::new(&blob);
    let header = parse_header(&blob, crate::safetensors_header::DEFAULT_MAX_HEADER_BYTES)?;
    let shards = Gemma4Shards::single(&header, &source);
    write_gemma4_install_streamed(dir, &arch, model_id, &shards, &quant(bits), |_| {})?;
    Ok(arch)
}
