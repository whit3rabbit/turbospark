//! Builds a DECODE-CAPABLE tiny `qwen4_exp` install: every layer carries
//! real hyper-connection, GDN-or-attention, and MoE weights, plus a fully
//! populated PLE layer (key/value/norm/conv1d projections, not just the
//! n-gram table `synthetic_qwen::qwen4`'s fixture stops at). This is what
//! `crates/runtime`'s `real_forward_qwen4.rs` exercises the decode flow
//! against -- that other fixture's own doc says explicitly it is not built
//! for this.
//!
//! Untrained, deterministic weights (the same xorshift scheme every other
//! synthetic fixture here uses), so nothing about the OUTPUT is meaningful;
//! what a test built on this can prove is that the flow runs, produces
//! finite logits, and reads the tensors it claims to (perturbation cases
//! plus a frozen digest, per `docs/NEW_MODEL.md` and AGENTS.md Gotcha 23).

mod arch;
mod ngram;
mod tensors;

pub use arch::{
    tiny_qwen4_exp_decode_arch, tiny_qwen4_exp_decode_arch_with_indexer_budget, HC_COUNT, HEAD_DIM,
    HIDDEN, NUM_EXPERTS, NUM_HEADS, NUM_KV_HEADS, NUM_LAYERS, TOP_K,
};
pub use ngram::{NGRAM_EOS_TOKEN_ID, PLE_LAYER};

use model_io::ArchConfig;

use crate::gemma4_checkpoint::{write_gemma4_install, Gemma4Quant, Gemma4Shards};
use crate::ranged_download::MemoryRangeSource;
use crate::safetensors_header::parse_header;
use crate::synthetic_tensors::assemble_safetensors;
use arch::IDX_BUDGET;
use tensors::build_tensors;

/// Writes a decode-capable `qwen4_exp` install through the real repack
/// pipeline and returns the `ArchConfig` needed to open it.
pub fn build_synthetic_qwen4_exp_decode_install(
    dir: &std::path::Path,
    vocab_size: i64,
    model_id: &str,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_qwen4_exp_decode_install_inner(dir, vocab_size, model_id, false, IDX_BUDGET)
}

/// [`build_synthetic_qwen4_exp_decode_install`] at a caller-chosen QSA
/// `index_budget`, so a test can cross it (and start dropping blocks) after
/// a handful of tokens instead of 2,049 of them. `index_top_k` follows as
/// `budget / IDX_COMPRESS`.
pub fn build_synthetic_qwen4_exp_decode_install_with_indexer_budget(
    dir: &std::path::Path,
    vocab_size: i64,
    model_id: &str,
    indexer_budget: i64,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_qwen4_exp_decode_install_inner(dir, vocab_size, model_id, false, indexer_budget)
}

/// [`build_synthetic_qwen4_exp_decode_install`], with the router shipped raw
/// (unquantized BF16) rather than pre-packed INT8, matching the real
/// REAP-288 checkpoint that exposed the router dtype bug
/// (`crates/runtime/src/families/qwen4/moe.rs`'s dtype-5 refusal). This is
/// the fixture that actually exercises `orchestrate.rs`'s
/// `quantize_router_int8`: the default fixture above ships the router
/// already `U32`-packed and takes the pre-existing `pass_through_packed`
/// branch, so it would pass identically whether or not that fix exists.
pub fn build_synthetic_qwen4_exp_decode_install_raw_router(
    dir: &std::path::Path,
    vocab_size: i64,
    model_id: &str,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_qwen4_exp_decode_install_inner(dir, vocab_size, model_id, true, IDX_BUDGET)
}

fn build_synthetic_qwen4_exp_decode_install_inner(
    dir: &std::path::Path,
    vocab_size: i64,
    model_id: &str,
    router_raw: bool,
    indexer_budget: i64,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    let arch = tiny_qwen4_exp_decode_arch_with_indexer_budget(vocab_size, indexer_budget);
    let (ts, bits_overrides) = build_tensors(vocab_size, router_raw);
    let blob = assemble_safetensors(&ts);
    let source = MemoryRangeSource::new(&blob);
    let header = parse_header(&blob, crate::safetensors_header::DEFAULT_MAX_HEADER_BYTES)?;
    let quant = Gemma4Quant {
        default_bits: 4,
        group_size: 64,
        bits_overrides,
    };
    let _ = Gemma4Shards::single(&header, &source);
    write_gemma4_install(dir, &arch, model_id, &header, &source, &quant)?;
    Ok(arch)
}
