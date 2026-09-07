#![cfg(target_os = "macos")]
//! `--kv-bits` on the DENSE half of the qwen linear-attention flow
//! (`qwenGdnDense`): the family with LINEAR-ATTENTION (GDN) layers that
//! carry no KV at all (`docs/TRUBOQUANT.md`). `attn::encode_linear_block`
//! (mask-2 layers) is untouched by this feature -- its whole history lives
//! in `GdnStateManager`'s recurrent state, never in `KvCacheManager` -- so
//! this file's job is to prove the write path engages on the mask-1 layers
//! ALONE, and that the surrounding GDN layers keep decoding exactly as
//! before.
//!
//! `LAYERS = 8` rather than the qwen35 suite's usual 4: `qwen_hybrid_layer_mask`
//! puts a full-attention layer at index 3 and again at 7 (`model-io`'s
//! `qwen_hybrid_layer_mask`), so at 4 layers the ONE full-attention layer is
//! also the LAST layer and the last-layer rule excludes every candidate --
//! nothing would ever quantize. At 8, layer 3 is full and not last, so it is
//! the only layer that quantizes; layer 7 is full but excluded by the
//! last-layer rule.
//!
//! Weights are deterministic but NOT trained (AGENTS.md Gotcha 12): only
//! token counts, finiteness, determinism and "the output moved" are
//! checked, never a specific value.

use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use model_io::{ExpertCacheSlots, KvQuant};
use turbospark_repack::build_synthetic_qwen_gdn_dense_install_at_bits;
use turbospark_runtime::{
    ChunkedPrefillRunner, DraftPolicies, LogitProducer, RealForwardRunner, SteeringPolicy,
};

const VOCAB: i64 = 128;
const LAYERS: i64 = 8;
/// The only width the batched-GEMV M-row GEMM has a kernel for
/// (`crates/runtime/CLAUDE.md` Gotcha 7's `TURBOSPARK_BATCHED_GEMV` bullet);
/// needed only by `batched_gemv_is_refused_by_name_under_kv_bits` below.
const INT4: u32 = 4;
/// The 1-bit default every other test here uses, matching
/// `real_forward_qwen35.rs`'s own fixtures.
const BITS_1: u32 = 1;

fn temp_dir(tag: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-qwen35-kv-quant-{tag}-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn open_runner(tag: &str, bits: u32, kv_quant: KvQuant) -> RealForwardRunner {
    let dir = temp_dir(tag);
    build_synthetic_qwen_gdn_dense_install_at_bits(&dir, VOCAB, LAYERS, tag, bits)
        .expect("dense qwen3_5 install builds");
    let peeked = turbospark_repack::peek_manifest_arch(&dir)
        .expect("a dense manifest peeks against the qwen3_5 baseline");
    RealForwardRunner::open_with_kv_quant(
        &dir,
        peeked,
        4096,
        ExpertCacheSlots::Fixed(4),
        DraftPolicies::off(),
        SteeringPolicy::off(),
        1,
        kv_quant,
    )
    .expect("dense qwen3_5 install opens")
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
    let arch_a =
        build_synthetic_qwen_gdn_dense_install_at_bits(&dir_a, VOCAB, LAYERS, "open-a", BITS_1)
            .expect("dense qwen3_5 install builds");
    let mut runner_a = RealForwardRunner::open(&dir_a, arch_a).expect("opens");
    let via_open = greedy_decode(&mut runner_a, 6);

    let mut runner_b = open_runner("open-b", BITS_1, KvQuant::Off);
    let via_kv_quant_off = greedy_decode(&mut runner_b, 6);

    assert_eq!(
        via_open, via_kv_quant_off,
        "KvQuant::Off must reproduce RealForwardRunner::open's logits exactly on the \
         GDN/full-attention hybrid flow"
    );
}

#[test]
fn every_kv_bits_width_decodes_finite_and_deterministic() {
    for (k_bits, v_bits) in [(2, 2), (3, 3), (3, 4), (4, 4)] {
        let mut runner = open_runner("sweep-a", BITS_1, KvQuant::TurboQuant { k_bits, v_bits });
        let first = greedy_decode(&mut runner, 12);
        let mut runner2 = open_runner("sweep-b", BITS_1, KvQuant::TurboQuant { k_bits, v_bits });
        let second = greedy_decode(&mut runner2, 12);
        assert_eq!(
            first, second,
            "k_bits={k_bits} v_bits={v_bits}: two runs from a fresh install must agree exactly"
        );
    }
}

#[test]
fn kv_bits_on_actually_moves_the_output_past_the_first_token() {
    let mut runner_off = open_runner("moves-off", BITS_1, KvQuant::Off);
    let off = greedy_decode(&mut runner_off, 10);

    let mut runner_on = open_runner(
        "moves-on",
        BITS_1,
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
        "K3/V4 quantization on the GDN/full-attention hybrid install produced logits \
         identical to FP16 past token 0; the write path or the attention fork did not engage"
    );
}

/// The dense driver's DEFAULT arm reuses the same sequential attention
/// functions this file already exercises (`families/qwen/prefill.rs`'s
/// module doc: "calls `attn::encode_full_attention_block` ... exactly as
/// the sequential flow"), so it must reproduce the sequential path exactly
/// under `--kv-bits`, GDN recurrence included.
#[test]
fn chunked_prefill_matches_sequential_decode_under_kv_bits() {
    let prompt: Vec<i32> = vec![5, 12, 3, 40, 7, 21, 9];

    let mut seq_runner = open_runner(
        "chunk-seq",
        BITS_1,
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
        BITS_1,
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

/// `TURBOSPARK_BATCHED_GEMV`'s M-row K/V projection
/// (`families/qwen/batched_layers.rs`) writes straight into the cache via a
/// GEMM and bypasses `kv_write_target`/`encode_kv_commit`; it must refuse a
/// quantized layer by name. Needs an INT4 install: at the 1-bit default the
/// WIDTH refusal fires first and this test would not reach the KV-specific
/// one at all.
#[test]
fn batched_gemv_refuses_a_quantized_layer_by_name() {
    let mut runner = open_runner(
        "batched-refusal",
        INT4,
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
