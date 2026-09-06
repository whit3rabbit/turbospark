#![cfg(target_os = "macos")]
//! `--kv-bits` on the `gpt-oss` decode flow: the family with ATTENTION
//! SINKS (`docs/TRUBOQUANT.md`). Sinks join the softmax DENOMINATOR only
//! (`crate::kv_write::encode_attention_any`'s `sinks` parameter, threaded
//! through unchanged into `encode_attention_decode_tq`), so a quantized
//! layer's sink handling is the same dispatch as an FP16 layer's -- this
//! file exists to prove that pass-through actually happens rather than
//! being dropped on the quantized fork.
//!
//! The fixture's default `head_dim` (16) is below TurboQuant's 32..=512
//! floor (`model_io::kv_quant::rht_supported`), so this file widens it to
//! 32 explicitly -- `SyntheticGptOssShape`'s fields are independent knobs
//! (AGENTS.md Gotcha 33's llama precedent), and nothing here depends on the
//! stock 16.
//!
//! Layers alternate SWA (even) and full (odd) attention
//! (`SyntheticGptOssShape::default`'s doc); TurboQuant only ever quantizes
//! full layers, so with 2 layers only layer 1 is a candidate, and the
//! last-layer rule (`num_layers > 2`) does not apply at 2 layers, so layer
//! 1 (full, last) still quantizes.
//!
//! Weights are deterministic but NOT trained (AGENTS.md Gotcha 12): only
//! token counts, finiteness, determinism and "the output moved" are
//! checked, never a specific value.

use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use model_io::{ExpertCacheSlots, KvQuant};
use turbospark_repack::{
    build_synthetic_gpt_oss_gguf, parse_gguf_header, write_gguf_install_streamed,
    MemoryRangeSource, SyntheticGptOssShape, GGUF_DEFAULT_MAX_HEADER_BYTES,
};
use turbospark_runtime::{
    ChunkedPrefillRunner, DraftPolicies, LogitProducer, RealForwardRunner, SteeringPolicy,
};

fn shape() -> SyntheticGptOssShape {
    SyntheticGptOssShape {
        head_dim: 32,
        ..SyntheticGptOssShape::default()
    }
}

fn temp_dir(tag: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-gptoss-kv-quant-{tag}-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn build_install(dir: &std::path::Path) -> model_io::ArchConfig {
    let (bytes, _) = build_synthetic_gpt_oss_gguf(shape());
    let header = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).expect("parse");
    write_gguf_install_streamed(
        dir,
        &header,
        &MemoryRangeSource::new(&bytes),
        "gptoss-kv-quant",
        |_| {},
    )
    .expect("write install")
}

fn open_runner(tag: &str, kv_quant: KvQuant) -> RealForwardRunner {
    let dir = temp_dir(tag);
    let arch = build_install(&dir);
    RealForwardRunner::open_with_kv_quant(
        &dir,
        arch,
        4096,
        ExpertCacheSlots::Fixed(8),
        DraftPolicies::off(),
        SteeringPolicy::off(),
        1,
        kv_quant,
    )
    .expect("gpt-oss install opens")
}

fn greedy_decode(runner: &mut RealForwardRunner, vocab: usize, steps: usize) -> Vec<Vec<f32>> {
    runner.reset();
    let mut token = 5i32;
    let mut out = Vec::new();
    for position in 0..steps {
        let mut head = vec![f16::from_f32(0.0); vocab];
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
    let vocab = shape().vocab as usize;
    let dir_a = temp_dir("open-a");
    let arch_a = build_install(&dir_a);
    let mut runner_a = RealForwardRunner::open(&dir_a, arch_a).expect("opens");
    let via_open = greedy_decode(&mut runner_a, vocab, 6);

    let mut runner_b = open_runner("open-b", KvQuant::Off);
    let via_kv_quant_off = greedy_decode(&mut runner_b, vocab, 6);

    assert_eq!(
        via_open, via_kv_quant_off,
        "KvQuant::Off must reproduce RealForwardRunner::open's logits exactly on the \
         sinks-and-alternating-window flow"
    );
}

#[test]
fn every_kv_bits_width_decodes_finite_and_deterministic_past_the_swa_window() {
    let vocab = shape().vocab as usize;
    // Past `sliding_window` (8) so the even (SWA) layer's ring wraps while
    // the odd (full, quantized) layer keeps accumulating.
    for (k_bits, v_bits) in [(2, 2), (3, 3), (3, 4), (4, 4)] {
        let mut runner = open_runner("sweep-a", KvQuant::TurboQuant { k_bits, v_bits });
        let first = greedy_decode(&mut runner, vocab, 12);
        let mut runner2 = open_runner("sweep-b", KvQuant::TurboQuant { k_bits, v_bits });
        let second = greedy_decode(&mut runner2, vocab, 12);
        assert_eq!(
            first, second,
            "k_bits={k_bits} v_bits={v_bits}: two runs from a fresh install must agree exactly"
        );
    }
}

#[test]
fn kv_bits_on_actually_moves_the_output_past_the_first_token() {
    let vocab = shape().vocab as usize;
    let mut runner_off = open_runner("moves-off", KvQuant::Off);
    let off = greedy_decode(&mut runner_off, vocab, 10);

    let mut runner_on = open_runner(
        "moves-on",
        KvQuant::TurboQuant {
            k_bits: 3,
            v_bits: 4,
        },
    );
    let on = greedy_decode(&mut runner_on, vocab, 10);

    assert_eq!(off.len(), on.len());
    let moved = off
        .iter()
        .zip(&on)
        .skip(1)
        .any(|(a, b)| a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-3));
    assert!(
        moved,
        "K3/V4 quantization on the sinks-carrying install produced logits identical to \
         FP16 past token 0; the write path or the attention fork did not engage"
    );
}

/// The per-token CHUNKED prefill driver reuses the same sequential attention
/// function this file already exercises
/// (`families/gptoss/prefill.rs` calls `encode_gpt_oss_layer_attn_and_router`),
/// so it must reproduce the sequential path exactly under `--kv-bits`,
/// SINKS included.
#[test]
fn chunked_prefill_matches_sequential_decode_under_kv_bits() {
    let vocab = shape().vocab as usize;
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
        let mut head = vec![f16::from_f32(0.0); vocab];
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
    let mut chunk_logits = vec![f16::from_f32(0.0); vocab];
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
/// driver unconditionally (`families/gptoss/prefill.rs`, independent of
/// `--kv-bits`: this family keeps every resident GEMV per token regardless).
/// Confirms that pre-existing refusal still fires the same way on a
/// TurboQuant-quantized install.
#[test]
fn batched_gemv_is_still_refused_by_name_under_kv_bits() {
    let vocab = shape().vocab as usize;
    let mut runner = open_runner(
        "batched-refusal",
        KvQuant::TurboQuant {
            k_bits: 3,
            v_bits: 4,
        },
    );
    runner.set_batched_gemv_prefill(true);
    let prompt: Vec<i32> = vec![5, 12, 3, 40];
    let mut logits = vec![f16::from_f32(0.0); vocab];
    let Err(err) = runner.prefill_chunk(&prompt, 0, &mut logits) else {
        panic!("batched GEMV prefill must still be refused on this family under --kv-bits");
    };
    let message = err.to_string();
    assert!(
        message.contains("BATCHED_GEMV"),
        "expected the pre-existing batched-GEMV refusal, got: {message}"
    );
}
