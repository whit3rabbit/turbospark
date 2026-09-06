#![cfg(target_os = "macos")]
//! `--kv-bits` on `qwen4_exp`'s decode flow: the family with QSA
//! (query-sparse attention), the LAST of the ten (`docs/TRUBOQUANT.md`).
//! `attn::encode_linear_block` (GDN, mask-2) is untouched -- no KV at all --
//! and this file's real job is the FOURTH shape `crate::kv_write` does not
//! cover: the indexed-sparse kernel `attention_decode_indexed_partial_tq`
//! Phase 2 built alongside the three other families' dense kernels, wired
//! in `families/qwen4/attn.rs`'s own `match kv.layer_quant(layer)` (not
//! `encode_attention_any`, which only forks the two DENSE variants).
//!
//! Layer mask is `[GDN, GDN+PLE, attention, GDN]` (`synthetic_qwen::qwen4_decode`'s
//! own doc): the one full-attention layer (index 2) is neither first nor
//! last, so it is the only candidate and it quantizes -- `num_layers == 4 >
//! 2`'s last-layer rule excludes index 3, which is GDN and was never a
//! candidate anyway.
//!
//! `TINY_INDEXER_BUDGET`/`FIRST_SPARSE_POSITION`/`SPARSE_STEPS` are lifted
//! verbatim from `real_forward_qwen4.rs`'s own constants, so decoding
//! `SPARSE_STEPS` positions exercises BOTH attention kernels under
//! quantization: dense below position 19, indexed-sparse from it onward.
//!
//! Weights are deterministic but NOT trained (AGENTS.md Gotcha 12): only
//! token counts, finiteness, determinism and "the output moved" are
//! checked, never a specific value.

use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use model_io::{ExpertCacheSlots, KvQuant};
use turbospark_repack::{
    build_synthetic_qwen4_exp_decode_install,
    build_synthetic_qwen4_exp_decode_install_with_indexer_budget,
};
use turbospark_runtime::{
    ChunkedPrefillRunner, DraftPolicies, LogitProducer, RealForwardRunner, SteeringPolicy,
};

const VOCAB: i64 = 2048;
/// Comfortably under the default fixture's 2048-token QSA budget and
/// comfortably over anything this file decodes, matching
/// `real_forward_qwen4.rs`'s own `TEST_MAX_CONTEXT`.
const TEST_MAX_CONTEXT: usize = 512;
/// 16 tokens is 4 blocks of `IDX_COMPRESS = 4`, so `index_top_k` is 4 and
/// selection starts dropping blocks at the fifth complete one -- `visible
/// >= 20`, i.e. position 19 onward (`real_forward_qwen4.rs`'s own constant).
const TINY_INDEXER_BUDGET: i64 = 16;
/// Enough steps past position 19 to exercise every `visible % 4` tail phase
/// of block selection under quantization, matching
/// `real_forward_qwen4.rs`'s own constant.
const SPARSE_STEPS: usize = 28;

fn temp_dir(tag: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-qwen4-kv-quant-{tag}-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn open_runner(tag: &str, kv_quant: KvQuant) -> RealForwardRunner {
    let dir = temp_dir(tag);
    let arch = build_synthetic_qwen4_exp_decode_install_with_indexer_budget(
        &dir,
        VOCAB,
        tag,
        TINY_INDEXER_BUDGET,
    )
    .expect("write install");
    RealForwardRunner::open_with_kv_quant(
        &dir,
        arch,
        TEST_MAX_CONTEXT,
        ExpertCacheSlots::Fixed(4),
        DraftPolicies::off(),
        SteeringPolicy::off(),
        1,
        kv_quant,
    )
    .expect("qwen4_exp install opens")
}

fn greedy_decode(runner: &mut RealForwardRunner, steps: usize) -> Vec<Vec<f32>> {
    runner.reset();
    let mut token = 5i32;
    let mut out = Vec::new();
    for position in 0..steps {
        let mut head = vec![f16::from_f32(0.0); VOCAB as usize];
        runner
            .produce(token, position, &mut head)
            .unwrap_or_else(|e| panic!("produce failed at position {position}: {e}"));
        let logits: Vec<f32> = head.iter().map(|v| v.to_f32()).collect();
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "non-finite logit at position {position}"
        );
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as i32)
            .unwrap();
        out.push(logits);
        token = argmax;
    }
    out
}

#[test]
fn kv_quant_off_matches_the_original_open_entry_point() {
    let dir_a = temp_dir("open-a");
    let arch_a =
        build_synthetic_qwen4_exp_decode_install(&dir_a, VOCAB, "open-a").expect("write install");
    let mut runner_a =
        RealForwardRunner::open_with_max_context(&dir_a, arch_a, TEST_MAX_CONTEXT).expect("opens");
    let via_open = greedy_decode(&mut runner_a, 6);

    let dir_b = temp_dir("open-b");
    let arch_b =
        build_synthetic_qwen4_exp_decode_install(&dir_b, VOCAB, "open-b").expect("write install");
    let mut runner_b = RealForwardRunner::open_with_kv_quant(
        &dir_b,
        arch_b,
        TEST_MAX_CONTEXT,
        ExpertCacheSlots::Fixed(4),
        DraftPolicies::off(),
        SteeringPolicy::off(),
        1,
        KvQuant::Off,
    )
    .expect("opens");
    let via_kv_quant_off = greedy_decode(&mut runner_b, 6);

    assert_eq!(
        via_open, via_kv_quant_off,
        "KvQuant::Off must reproduce the original open path's logits exactly on the \
         QSA/GDN/PLE/MoE flow"
    );
}

#[test]
fn every_kv_bits_width_decodes_finite_and_deterministic_through_the_sparse_path() {
    for (k_bits, v_bits) in [(2, 2), (3, 3), (3, 4), (4, 4)] {
        let mut runner = open_runner("sweep-a", KvQuant::TurboQuant { k_bits, v_bits });
        let first = greedy_decode(&mut runner, SPARSE_STEPS);
        let mut runner2 = open_runner("sweep-b", KvQuant::TurboQuant { k_bits, v_bits });
        let second = greedy_decode(&mut runner2, SPARSE_STEPS);
        assert_eq!(
            first, second,
            "k_bits={k_bits} v_bits={v_bits}: two runs from a fresh install must agree exactly, \
             dense and indexed-sparse steps included"
        );
    }
}

#[test]
fn kv_bits_on_actually_moves_the_output_past_the_first_token() {
    let mut runner_off = open_runner("moves-off", KvQuant::Off);
    let off = greedy_decode(&mut runner_off, 10);

    let mut runner_on = open_runner(
        "moves-on",
        KvQuant::TurboQuant {
            k_bits: 3,
            v_bits: 4,
        },
    );
    let on = greedy_decode(&mut runner_on, 10);

    assert_eq!(off.len(), on.len());
    let moved = off
        .iter()
        .zip(&on)
        .skip(1)
        .any(|(a, b)| a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-3));
    assert!(
        moved,
        "K3/V4 quantization on the QSA install produced logits identical to FP16 past \
         token 0; the write path or the attention fork did not engage"
    );
}

/// `sparse_and_forced_dense_agree_below_budget_and_diverge_above` is
/// `real_forward_qwen4.rs`'s FP16 guard that the sparse path is a genuine
/// fork rather than a silent dense fallback. This is the same guard under
/// `--kv-bits`: forcing dense must move the logits somewhere past the
/// sparsity boundary, proving the indexed-TQ kernel is the one that
/// actually ran there rather than the dense-TQ kernel by accident.
#[test]
fn forcing_dense_past_the_sparsity_boundary_moves_the_output_under_kv_bits() {
    let mut sparse_runner = open_runner(
        "force-sparse",
        KvQuant::TurboQuant {
            k_bits: 3,
            v_bits: 4,
        },
    );
    let sparse = greedy_decode(&mut sparse_runner, SPARSE_STEPS);

    let mut dense_runner = open_runner(
        "force-dense",
        KvQuant::TurboQuant {
            k_bits: 3,
            v_bits: 4,
        },
    );
    dense_runner.set_qsa_force_dense(true);
    let forced_dense = greedy_decode(&mut dense_runner, SPARSE_STEPS);

    let moved = sparse
        .iter()
        .zip(&forced_dense)
        .any(|(a, b)| a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-3));
    assert!(
        moved,
        "forcing dense attention past the QSA sparsity boundary produced logits identical \
         to the sparse path under --kv-bits; the indexed-TQ kernel fork did not engage"
    );
}

/// The per-token CHUNKED prefill driver reuses the same sequential attention
/// functions this file already exercises (`crates/runtime/CLAUDE.md`'s
/// qwen4 chunked-prefill entry: "QSA needed NOTHING new ... calling it once
/// per token in increasing order reproduces that unchanged"), so it must
/// reproduce the sequential path exactly under `--kv-bits`, across the
/// sparsity boundary.
#[test]
fn chunked_prefill_matches_sequential_decode_under_kv_bits() {
    let prompt: Vec<i32> = (0..24).map(|i| (i * 7 + 3) % VOCAB as i32).collect();

    let mut seq_runner = open_runner(
        "chunk-seq",
        KvQuant::TurboQuant {
            k_bits: 3,
            v_bits: 4,
        },
    );
    seq_runner.reset();
    let mut sequential = Vec::new();
    for (position, &token) in prompt.iter().enumerate() {
        let mut head = vec![f16::from_f32(0.0); VOCAB as usize];
        seq_runner
            .produce(token, position, &mut head)
            .unwrap_or_else(|e| panic!("produce failed at position {position}: {e}"));
        sequential.push(head.iter().map(|v| v.to_f32()).collect::<Vec<_>>());
    }

    let mut chunked_runner = open_runner(
        "chunk-chunked",
        KvQuant::TurboQuant {
            k_bits: 3,
            v_bits: 4,
        },
    );
    let mut chunk_logits = vec![f16::from_f32(0.0); VOCAB as usize];
    chunked_runner
        .prefill_chunk(&prompt, 0, &mut chunk_logits)
        .expect("chunked prefill succeeds under --kv-bits, across the sparsity boundary");
    let chunked_last: Vec<f32> = chunk_logits.iter().map(|v| v.to_f32()).collect();

    let err = (sequential.last().unwrap().iter())
        .zip(&chunked_last)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        err < 1e-2,
        "chunked prefill's last-token logits diverge from sequential decode's under \
         --kv-bits: max abs diff {err}"
    );
}

/// Both `TURBOSPARK_BATCHED_GEMV` and `TURBOSPARK_ROUTED_BATCH` are refused
/// BY NAME on this family's chunked driver unconditionally
/// (`families/qwen4/prefill.rs`, independent of `--kv-bits`). Confirms that
/// pre-existing refusal still fires the same way on a TurboQuant-quantized
/// install.
#[test]
fn batched_gemv_is_still_refused_by_name_under_kv_bits() {
    let mut runner = open_runner(
        "batched-refusal",
        KvQuant::TurboQuant {
            k_bits: 3,
            v_bits: 4,
        },
    );
    runner.set_batched_gemv_prefill(true);
    let prompt: Vec<i32> = vec![5, 12, 3, 40];
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let Err(err) = runner.prefill_chunk(&prompt, 0, &mut logits) else {
        panic!("batched GEMV prefill must still be refused on this family under --kv-bits");
    };
    let message = err.to_string();
    assert!(
        message.contains("BATCHED_GEMV"),
        "expected the pre-existing batched-GEMV refusal, got: {message}"
    );
}
