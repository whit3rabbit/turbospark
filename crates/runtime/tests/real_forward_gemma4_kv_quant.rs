#![cfg(target_os = "macos")]
//! `--kv-bits` on the Gemma 4 decode flow: the first MIXED-attention
//! family wired (`docs/TRUBOQUANT.md`). Layers alternate SWA (mask 0) and
//! full (mask 1); TurboQuant only ever quantizes full layers, and with 4
//! layers the last-layer rule additionally excludes the last one (layer 3,
//! itself full-attention) from quantizing even though its mask permits it.
//! So this fixture exercises both exclusions in one install: mask alone
//! (layers 0, 2) and the last-layer rule on top of a permitting mask
//! (layer 3), leaving layer 1 as the only one that actually quantizes.
//!
//! Weights are deterministic but NOT trained (AGENTS.md Gotcha 12): only
//! token counts, finiteness, determinism and "the output moved" are
//! checked, never a specific value.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use model_io::{ExpertCacheSlots, KvQuant};
use turbospark_repack::build_synthetic_gemma4_real_install;
use turbospark_runtime::{
    ChunkedPrefillRunner, DraftPolicies, LogitProducer, RealForwardRunner, SteeringPolicy,
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

const VOCAB: i64 = 128;
const NUM_LAYERS: i64 = 4;
const NUM_EXPERTS: i64 = 2;
const TOP_K: i64 = 2;
const SLIDING_WINDOW: i64 = 8;
const MAX_CONTEXT: usize = 64;

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-real-forward-gemma4-kv-quant-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn open_runner(dir: &std::path::Path, kv_quant: KvQuant) -> RealForwardRunner {
    let arch = build_synthetic_gemma4_real_install(
        dir,
        VOCAB,
        NUM_LAYERS,
        NUM_EXPERTS,
        TOP_K,
        SLIDING_WINDOW,
        "tiny-gemma4-kv-quant",
    )
    .expect("real-naming install builds");
    RealForwardRunner::open_with_kv_quant(
        dir,
        arch,
        MAX_CONTEXT,
        ExpertCacheSlots::Fixed(4),
        DraftPolicies::off(),
        SteeringPolicy::off(),
        1,
        kv_quant,
    )
    .expect("real-naming install opens")
}

fn greedy_decode(runner: &mut RealForwardRunner, steps: usize) -> Vec<Vec<f32>> {
    runner.reset();
    let mut token = 5i32;
    let mut out = Vec::new();
    for position in 0..steps {
        let mut head = vec![f16::from_f32(0.0); VOCAB as usize];
        runner
            .produce(token, position, &mut head)
            .expect("produce succeeds");
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
    let dir_a = temp_dir();
    let arch_a = build_synthetic_gemma4_real_install(
        &dir_a,
        VOCAB,
        NUM_LAYERS,
        NUM_EXPERTS,
        TOP_K,
        SLIDING_WINDOW,
        "tiny-gemma4-kv-quant",
    )
    .expect("builds");
    let mut runner_a = RealForwardRunner::open(&dir_a, arch_a).expect("opens");
    let via_open = greedy_decode(&mut runner_a, 6);

    let dir_b = temp_dir();
    let mut runner_b = open_runner(&dir_b, KvQuant::Off);
    let via_kv_quant_off = greedy_decode(&mut runner_b, 6);

    assert_eq!(
        via_open, via_kv_quant_off,
        "KvQuant::Off must reproduce RealForwardRunner::open's logits exactly, \
         mixed-attention layers included"
    );
}

#[test]
fn every_kv_bits_width_decodes_finite_and_deterministic_past_the_swa_window() {
    // More steps than `SLIDING_WINDOW` so the SWA layers' ring wraps at
    // least once while the neighbouring full layers are quantized.
    for (k_bits, v_bits) in [(2, 2), (3, 3), (3, 4), (4, 4)] {
        let dir = temp_dir();
        let mut runner = open_runner(&dir, KvQuant::TurboQuant { k_bits, v_bits });
        let first = greedy_decode(&mut runner, 12);
        let second_dir = temp_dir();
        let mut runner2 = open_runner(&second_dir, KvQuant::TurboQuant { k_bits, v_bits });
        let second = greedy_decode(&mut runner2, 12);
        assert_eq!(
            first, second,
            "k_bits={k_bits} v_bits={v_bits}: two runs from a fresh install must agree exactly"
        );
    }
}

#[test]
fn kv_bits_on_actually_moves_the_output_past_the_first_token() {
    let dir_off = temp_dir();
    let mut runner_off = open_runner(&dir_off, KvQuant::Off);
    let off = greedy_decode(&mut runner_off, 10);

    let dir_on = temp_dir();
    let mut runner_on = open_runner(
        &dir_on,
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
        "K3/V4 quantization on the mixed-attention install produced logits identical \
         to FP16 past token 0; the write path or the attention fork did not engage"
    );
}

/// The per-token CHUNKED prefill driver reuses the same sequential
/// attention function this file already exercises
/// (`families/gemma4/prefill.rs` calls `encode_gemma4_layer_attn_and_router`),
/// so it must reproduce the sequential path exactly under `--kv-bits`.
#[test]
fn chunked_prefill_matches_sequential_decode_under_kv_bits() {
    let prompt: Vec<i32> = vec![5, 12, 3, 40, 7, 21, 9];

    let dir_seq = temp_dir();
    let mut seq_runner = open_runner(
        &dir_seq,
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
            .expect("produce succeeds");
        sequential.push(head.iter().map(|v| v.to_f32()).collect::<Vec<_>>());
    }

    let dir_chunked = temp_dir();
    let mut chunked_runner = open_runner(
        &dir_chunked,
        KvQuant::TurboQuant {
            k_bits: 3,
            v_bits: 4,
        },
    );
    let mut chunk_logits = vec![f16::from_f32(0.0); VOCAB as usize];
    chunked_runner
        .prefill_chunk(&prompt, 0, &mut chunk_logits)
        .expect("chunked prefill succeeds under --kv-bits");
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

/// `TURBOSPARK_BATCHED_GEMV`'s M-row K/V projection is NOT wired to
/// `--kv-bits` (`crates/runtime/src/kv_write.rs`'s module doc): it must
/// refuse a quantized layer by name rather than silently writing raw FP16
/// bytes into a buffer TurboQuant has sized for packed words.
#[test]
fn batched_gemv_refuses_a_quantized_layer_by_name() {
    let dir = temp_dir();
    let mut runner = open_runner(
        &dir,
        KvQuant::TurboQuant {
            k_bits: 3,
            v_bits: 4,
        },
    );
    runner.set_batched_gemv_prefill(true);
    let prompt: Vec<i32> = vec![5, 12, 3, 40];
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let Err(err) = runner.prefill_chunk(&prompt, 0, &mut logits) else {
        panic!("batched GEMV prefill on a quantized layer must be refused");
    };
    let message = err.to_string();
    assert!(
        message.contains("BATCHED_GEMV"),
        "expected the batched-GEMV kv-bits refusal, got: {message}"
    );
}
