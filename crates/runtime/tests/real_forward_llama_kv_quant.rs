#![cfg(target_os = "macos")]
//! `--kv-bits` on the `llama` decode flow: the first family wired
//! (`docs/TRUBOQUANT.md`), chosen because every layer is full attention and
//! the flow has no per-head norms, no output gate and no sandwich norms to
//! interact with the write-path fork.
//!
//! Weights are deterministic but NOT trained (AGENTS.md Gotcha 12), so
//! nothing here asserts on generated TEXT or on a specific logit value --
//! only that `--kv-bits off` reproduces the exact FP16 path byte-for-byte,
//! that turning it on stays finite and deterministic, and that it actually
//! MOVES the output (proving the quantized path is exercised rather than
//! silently bypassed).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use model_io::{ExpertCacheSlots, KvQuant};
use turbospark_repack::build_synthetic_llama_real_install;
use turbospark_runtime::{DraftPolicies, LogitProducer, RealForwardRunner, SteeringPolicy};

static COUNTER: AtomicU64 = AtomicU64::new(0);

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;
const EXPERTS: i64 = 8;
const MAX_CONTEXT: usize = 64;

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-real-forward-llama-kv-quant-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn open_runner(dir: &std::path::Path, kv_quant: KvQuant) -> RealForwardRunner {
    let arch = build_synthetic_llama_real_install(dir, VOCAB, LAYERS, EXPERTS, "tiny-mixtral")
        .expect("llama install builds");
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
    .expect("llama install opens")
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

/// `--kv-bits off` through the new widest constructor must reproduce
/// exactly what `RealForwardRunner::open` (every pre-existing caller's
/// entry point) produces -- the whole point of `KvQuant::Off` being a thin
/// wrapper rather than a new code path.
#[test]
fn kv_quant_off_matches_the_original_open_entry_point() {
    let dir_a = temp_dir();
    let arch_a = build_synthetic_llama_real_install(&dir_a, VOCAB, LAYERS, EXPERTS, "tiny-mixtral")
        .expect("llama install builds");
    let mut runner_a = RealForwardRunner::open(&dir_a, arch_a).expect("opens");
    let via_open = greedy_decode(&mut runner_a, 6);

    let dir_b = temp_dir();
    let mut runner_b = open_runner(&dir_b, KvQuant::Off);
    let via_kv_quant_off = greedy_decode(&mut runner_b, 6);

    assert_eq!(
        via_open, via_kv_quant_off,
        "KvQuant::Off must reproduce RealForwardRunner::open's logits exactly"
    );
}

/// Every `(k_bits, v_bits)` combination `--kv-bits` exposes opens and
/// decodes to finite, deterministic logits over several tokens (past the
/// point the first-token softmax is trivially a single key).
#[test]
fn every_kv_bits_width_decodes_finite_and_deterministic() {
    for (k_bits, v_bits) in [(2, 2), (3, 3), (3, 4), (4, 4)] {
        let dir = temp_dir();
        let mut runner = open_runner(&dir, KvQuant::TurboQuant { k_bits, v_bits });
        let first = greedy_decode(&mut runner, 5);
        let second_dir = temp_dir();
        let mut runner2 = open_runner(&second_dir, KvQuant::TurboQuant { k_bits, v_bits });
        let second = greedy_decode(&mut runner2, 5);
        assert_eq!(
            first, second,
            "k_bits={k_bits} v_bits={v_bits}: two runs from a fresh install must agree exactly"
        );
    }
}

/// The quantized path must actually be EXERCISED, not silently equal to
/// FP16 because the commit or the attention fork never engaged: past the
/// first token (where a single-key softmax makes K's rotation invisible in
/// the score, and would make V's codebook error the only source of any
/// difference at all), quantization noise must move at least one logit.
#[test]
fn kv_bits_on_actually_moves_the_output_past_the_first_token() {
    let dir_off = temp_dir();
    let mut runner_off = open_runner(&dir_off, KvQuant::Off);
    let off = greedy_decode(&mut runner_off, 5);

    let dir_on = temp_dir();
    let mut runner_on = open_runner(
        &dir_on,
        KvQuant::TurboQuant {
            k_bits: 3,
            v_bits: 4,
        },
    );
    let on = greedy_decode(&mut runner_on, 5);

    assert_eq!(off.len(), on.len());
    let moved = off
        .iter()
        .zip(&on)
        .skip(1)
        .any(|(a, b)| a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-3));
    assert!(
        moved,
        "K3/V4 quantization produced logits identical to FP16 past token 0; \
         the write path or the attention fork did not engage"
    );
}

/// Refused by name on a head_dim TurboQuant cannot rotate -- this fixture's
/// widened head_dim (32) is exactly the floor, so shrinking it back below
/// 32 in a one-off `ArchConfig` clone proves the refusal fires rather than
/// silently opening with garbage rotation.
#[test]
fn kv_bits_is_refused_by_name_on_an_unsupported_head_dim() {
    let dir = temp_dir();
    let mut arch = build_synthetic_llama_real_install(&dir, VOCAB, LAYERS, EXPERTS, "tiny-mixtral")
        .expect("llama install builds");
    arch.head_dim = 16;
    arch.full_head_dim = 16;
    let Err(err) = RealForwardRunner::open_with_kv_quant(
        &dir,
        arch,
        MAX_CONTEXT,
        ExpertCacheSlots::Fixed(4),
        DraftPolicies::off(),
        SteeringPolicy::off(),
        1,
        KvQuant::TurboQuant {
            k_bits: 3,
            v_bits: 4,
        },
    ) else {
        panic!("a head_dim of 16 must be refused");
    };
    let message = err.to_string();
    assert!(
        message.contains("32..=512"),
        "expected the RHT-support refusal, got: {message}"
    );
}
