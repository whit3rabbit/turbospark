#![cfg(target_os = "macos")]
//! The Qwen 3.6 flow over an MLX mxfp4 install, the layout of
//! `mlx-community/MiniCPM-V-4.7-35B-A3B-mxfp4`: routed experts re-blocked to
//! GGUF MXFP4 and run by the MXFP4 pair with the PLAIN activation, resident
//! mxfp4 dequantized to BF16, BF16 embed/head/GDN projections, INT8 router.
//!
//! Untrained weights, so nothing asserts on text (AGENTS.md Gotcha 12). The
//! one assertion that compares against something computed before a change is
//! the frozen digest (Gotcha 51); every other test here is self-relative.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use turbospark_repack::build_synthetic_qwen_gdn_moe_mxfp4_install;
use turbospark_runtime::{LogitProducer, RealForwardRunner};

static COUNTER: AtomicU64 = AtomicU64::new(0);

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;
/// Twice top-k, so an 8-slot cache cannot hold every expert and the 8- and
/// 16-slot runs take different cache paths.
const EXPERTS: i64 = 16;
const MAX_CONTEXT: usize = 256;

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-real-forward-qwen-mxfp4-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn build(dir: &Path) -> model_io::ArchConfig {
    build_synthetic_qwen_gdn_moe_mxfp4_install(dir, VOCAB, LAYERS, EXPERTS, "tiny-qwen36-mxfp4")
        .expect("mxfp4 install builds")
}

fn open_runner(dir: &Path) -> RealForwardRunner {
    let arch = build(dir);
    RealForwardRunner::open(dir, arch).expect("mxfp4 install opens")
}

/// Argmax-fed greedy decode returning every step's logits, asserting the
/// logits contract on each: finite and not a probability distribution.
fn greedy(runner: &mut RealForwardRunner, steps: usize) -> (Vec<i32>, Vec<f16>) {
    runner.reset();
    let mut token = 5i32;
    let mut tokens = Vec::new();
    let mut head = vec![f16::from_f32(0.0); VOCAB as usize];
    for position in 0..steps {
        runner.produce(token, position, &mut head).expect("produce");
        assert!(
            head.iter().all(|v| v.to_f32().is_finite()),
            "position {position}"
        );
        let sum: f32 = head.iter().map(|v| v.to_f32()).sum();
        assert!(
            head.iter().any(|v| v.to_f32() < 0.0) || (sum - 1.0).abs() > 1e-2,
            "`produce` returned probabilities at position {position}"
        );
        token = head
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
            .map(|(i, _)| i as i32)
            .unwrap();
        tokens.push(token);
    }
    (tokens, head)
}

#[test]
fn decodes_deterministically_and_reset_rewinds_the_gdn_state() {
    let dir = temp_dir();
    let mut runner = open_runner(&dir);
    let first = greedy(&mut runner, 6);
    let _ = greedy(&mut runner, 11);
    let again = greedy(&mut runner, 6);
    assert_eq!(first, again);
}

#[test]
fn linear_layers_run_a_non_zero_recurrence() {
    let dir = temp_dir();
    let mut runner = open_runner(&dir);
    let _ = greedy(&mut runner, 3);
    // Odd layers are full attention in the tiny arch; even ones are GDN.
    for layer in (0..LAYERS as usize).step_by(2) {
        let max = runner.gdn_state_abs_max(layer).expect("linear layer");
        assert!(max > 0.0 && max.is_finite(), "layer {layer}: {max}");
    }
}

#[test]
fn decode_hot_path_allocates_no_gpu_buffers() {
    let dir = temp_dir();
    let mut runner = open_runner(&dir);
    let _ = greedy(&mut runner, 2);
    let before = runner.gpu_buffer_allocations();
    let _ = greedy(&mut runner, 4);
    assert_eq!(before, runner.gpu_buffer_allocations());
}

#[test]
fn prefill_then_decode_matches_all_produce() {
    let dir = temp_dir();
    let mut runner = open_runner(&dir);
    let prompt = [5i32, 9, 2, 7];
    let mut scratch = vec![f16::from_f32(0.0); VOCAB as usize];
    runner.reset();
    for (p, &t) in prompt.iter().enumerate() {
        runner.produce(t, p, &mut scratch).expect("produce");
    }
    let want = scratch.clone();
    runner.reset();
    for (p, &t) in prompt.iter().enumerate() {
        if p + 1 == prompt.len() {
            runner.produce(t, p, &mut scratch).expect("produce");
        } else {
            runner.produce_prefill(t, p, &mut scratch).expect("prefill");
        }
    }
    assert_eq!(scratch, want);
}

/// Routed slots are dispatched in router rank, so the cache size cannot
/// reach the reduction order (AGENTS.md Gotcha 27).
#[test]
fn the_slot_count_does_not_change_the_logits() {
    let dir = temp_dir();
    let arch = build(&dir);
    let run = |slots: usize| {
        let mut runner =
            RealForwardRunner::open_with_options(&dir, arch.clone(), MAX_CONTEXT, slots)
                .expect("opens");
        greedy(&mut runner, 8)
    };
    assert_eq!(run(8), run(16));
}

/// FNV-1a over FP16 bit patterns (the `real_forward_muse.rs` helper).
fn fnv1a(logits: &[f16]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in logits {
        for b in v.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    }
    h
}

/// The one assertion against a value computed before any later change: the
/// last of six greedy steps' logits. Mutation-checked by forcing
/// `gpt-oss`'s `swiglu_oai` onto this family (which a self-relative test
/// cannot see). A virtualized device moves FP16 roundings, so it is held to
/// the tolerant comparison `real_forward_muse.rs` documents instead.
#[test]
fn the_mxfp4_flows_arithmetic_is_frozen() {
    let dir = temp_dir();
    let mut runner = open_runner(&dir);
    let (tokens, head) = greedy(&mut runner, 6);
    if std::env::var_os("TURBOSPARK_PRINT_FROZEN").is_some() {
        println!("tokens {tokens:?}");
        println!("hash {}", fnv1a(&head));
        println!(
            "logits {:?}",
            head.iter().map(|v| v.to_f32()).collect::<Vec<_>>()
        );
    }
    let context = gpu::MetalContext::new().expect("Metal device");
    let virtualized = context.device().name().contains("Paravirtual");
    drop(context);
    if virtualized {
        assert_eq!(head.len(), FROZEN_LOGITS.len());
        for (i, (got, &want)) in head.iter().zip(FROZEN_LOGITS.iter()).enumerate() {
            let got = got.to_f32();
            let tol = 0.02f32.max(want.abs() * 0.02);
            assert!(
                (got - want).abs() <= tol,
                "logit {i}: got {got}, want {want}"
            );
        }
        return;
    }
    assert_eq!(
        tokens, FROZEN_TOKENS,
        "the mxfp4 flow's greedy tokens moved"
    );
    assert_eq!(
        fnv1a(&head),
        FROZEN_HASH,
        "the mxfp4 flow's arithmetic moved"
    );
}

/// Frozen 2026-10-13 on real Apple Silicon (Apple M4 Max), reproduced by two
/// fresh processes. Re-freeze only with the reason in the commit: a change
/// to the codec, the MXFP4 pair, the BF16 readers or the qwen flow moves it.
const FROZEN_TOKENS: [i32; 6] = [45, 21, 24, 47, 8, 2];
const FROZEN_HASH: u64 = 11_167_848_931_669_677_041;
/// The values `FROZEN_HASH` was taken over, for the virtualized-device path.
#[rustfmt::skip]
const FROZEN_LOGITS: [f32; VOCAB as usize] = [
    0.37719727, -0.16479492, 0.58691406, -0.32714844, -0.07739258, 0.3515625,
    -0.22680664, 0.10101318, 0.30273438, 0.32763672, -0.122802734, -0.11956787,
    0.22106934, -0.1385498, -0.1027832, 0.014045715, -0.19262695, -0.034729004,
    -0.014175415, 0.13879395, -0.37670898, 0.34277344, -0.04156494, -0.18334961,
    -0.067993164, 0.19592285, 0.091674805, 0.040924072, -0.13134766, 0.00039458275,
    -0.28173828, -0.14685059, 0.1303711, -0.00095129013, -0.07861328, -0.20166016,
    -0.2854004, 0.13586426, -0.2310791, 0.29492188, -0.13708496, 0.23718262,
    -0.23083496, 0.07421875, -0.015357971, -0.19909668, -0.1574707, 0.022750854,
    0.06903076, -0.056365967, 0.26293945, 0.29077148, 0.3244629, -0.04260254,
    -0.22375488, -0.35668945, -0.16625977, -0.031433105, 0.009048462, -0.37329102,
    0.25805664, 0.115234375, -0.1104126, -0.41796875, 0.052856445, -0.11920166,
    0.2746582, -0.0579834, 0.18005371, 0.13208008, 0.16149902, -0.23547363,
    -0.29516602, 0.20153809, 0.03515625, -0.31689453, 0.044036865, 0.23242188,
    -0.15966797, 3.9339066e-5, 0.11785889, 0.26000977, -0.10821533, -0.08746338,
    -0.17932129, 0.01927185, 0.13623047, -0.24658203, 0.052215576, 0.0022068024,
    -0.04257202, 0.087402344, 0.16845703, 0.07745361, -0.2290039, 0.017028809,
    -0.18676758, -0.47729492, -0.36499023, -0.12200928, -0.03677368, 0.109313965,
    -0.26049805, -0.2841797, 0.14831543, -0.14294434, -0.24230957, 0.18676758,
    0.2010498, -0.08905029, 0.064941406, 0.010192871, 0.0099487305, -0.1616211,
    0.055541992, -0.24267578, -0.026016235, -0.14465332, 0.3251953, -0.06903076,
    -0.1307373, 0.11517334, -0.3161621, -0.13220215, -0.105773926, -0.28344727,
    0.050689697, 0.04977417,
];
