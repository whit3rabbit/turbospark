#![cfg(target_os = "macos")]
//! `--kv-bits` on the `muse_glimmer` decode flow: the family with NoPE full
//! layers, a second Q scale after a no-scale per-head norm, and a separate
//! output gate applied to the ATTENTION OUTPUT rather than to K or V
//! (`docs/TRUBOQUANT.md`). None of those three touch K or V after the
//! write-path's norm/RoPE step, so this file's job is narrower than
//! gpt-oss's or gemma4's: prove the write path and the attention fork
//! engage on a family whose K/V side of the layer is otherwise the
//! simplest of the ten.
//!
//! Layer mask is `[0, 0, 0, 1]` repeating (`build_synthetic_muse_glimmer_install`'s
//! own fixture doc): layers 0-2 SWA, layer 3 full and NoPE. TurboQuant only
//! ever quantizes full layers, so of 8 layers, layers 3 and 7 are
//! candidates; with `num_layers == 8 > 2` the last-layer rule additionally
//! excludes layer 7, leaving layer 3 as the only one that quantizes.
//!
//! Weights are deterministic but NOT trained (AGENTS.md Gotcha 12): only
//! token counts, finiteness, determinism and "the output moved" are
//! checked, never a specific value.

use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use model_io::{ExpertCacheSlots, KvQuant};
use turbospark_repack::build_synthetic_muse_glimmer_install;
use turbospark_runtime::{
    ChunkedPrefillRunner, DraftPolicies, LogitProducer, RealForwardRunner, SteeringPolicy,
};

const VOCAB: i64 = 64;
/// A multiple of 4 so the `[0, 0, 0, 1]` window pattern is whole; see
/// `real_forward_muse.rs`'s identical constant for why 8 rather than 4.
const LAYERS: i64 = 8;

fn temp_dir(tag: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-muse-kv-quant-{tag}-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn open_runner(tag: &str, kv_quant: KvQuant) -> RealForwardRunner {
    let dir = temp_dir(tag);
    let arch = build_synthetic_muse_glimmer_install(&dir, VOCAB, LAYERS, "muse-kv-quant")
        .expect("the install writes");
    RealForwardRunner::open_with_kv_quant(
        &dir,
        arch,
        4096,
        ExpertCacheSlots::Fixed(4),
        DraftPolicies::off(),
        SteeringPolicy::off(),
        1,
        kv_quant,
    )
    .expect("muse_glimmer install opens")
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
    let dir_a = temp_dir("open-a");
    let arch_a = build_synthetic_muse_glimmer_install(&dir_a, VOCAB, LAYERS, "muse-kv-quant")
        .expect("the install writes");
    let mut runner_a = RealForwardRunner::open(&dir_a, arch_a).expect("opens");
    let via_open = greedy_decode(&mut runner_a, 6);

    let mut runner_b = open_runner("open-b", KvQuant::Off);
    let via_kv_quant_off = greedy_decode(&mut runner_b, 6);

    assert_eq!(
        via_open, via_kv_quant_off,
        "KvQuant::Off must reproduce RealForwardRunner::open's logits exactly on the \
         NoPE/output-gate flow"
    );
}

#[test]
fn every_kv_bits_width_decodes_finite_and_deterministic_past_the_swa_window() {
    // More steps than the fixture's 8-token sliding window, so the SWA
    // layers' ring wraps while the quantized full layer (3) keeps
    // accumulating unbounded history.
    for (k_bits, v_bits) in [(2, 2), (3, 3), (3, 4), (4, 4)] {
        let mut runner = open_runner("sweep-a", KvQuant::TurboQuant { k_bits, v_bits });
        let first = greedy_decode(&mut runner, 12);
        let mut runner2 = open_runner("sweep-b", KvQuant::TurboQuant { k_bits, v_bits });
        let second = greedy_decode(&mut runner2, 12);
        assert_eq!(
            first, second,
            "k_bits={k_bits} v_bits={v_bits}: two runs from a fresh install must agree exactly"
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
        "K3/V4 quantization on the NoPE/output-gate install produced logits identical to \
         FP16 past token 0; the write path or the attention fork did not engage"
    );
}

/// The per-token CHUNKED prefill driver reuses the same sequential attention
/// function this file already exercises (`families/museglimmer/prefill.rs`
/// calls `attn::encode_attention_block` unchanged), so it must reproduce the
/// sequential path exactly under `--kv-bits`.
#[test]
fn chunked_prefill_matches_sequential_decode_under_kv_bits() {
    let prompt: Vec<i32> = vec![5, 12, 3, 40, 7, 21, 9];

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
            .expect("produce succeeds");
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

/// `TURBOSPARK_BATCHED_GEMV` is refused BY NAME on this family's chunked
/// driver unconditionally (`families/museglimmer/prefill.rs`, independent
/// of `--kv-bits`: this family keeps attention per-token and unbatched
/// regardless). Confirms that pre-existing refusal still fires the same
/// way on a TurboQuant-quantized install.
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
