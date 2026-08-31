#![cfg(target_os = "macos")]
//! Mapped expert residency (`MFERENCE_EXPERT_RESIDENCY=mapped`,
//! `docs/EXPERT_RESIDENCY.md`) on the `llama` (Mixtral / `Qwen3Moe`) flow,
//! through the real `open` path.
//!
//! Sibling of `mapped_expert_residency.rs` (Gemma 4) and
//! `mapped_expert_residency_qwen.rs`. Own test binary for the same reason
//! both of those are: the seam is a process-global environment variable
//! read once at `open`.
//!
//! Two call sites are exercised, matching this family's two routed-MoE
//! drivers: the sequential per-token path (`produce`) and the pipelined
//! chunked-prefill driver (`prefill_chunk`, `encode_llama_layer_routed_moe_pipelined`).
//! Both have to read `mapped` correctly, since they are two separate
//! threadings of the same parameter rather than one function reused.

use half::f16;
use turbospark_repack::build_synthetic_llama_real_install;
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;
const EXPERTS: i64 = 8;
const PROMPT: [i32; 11] = [5, 9, 2, 7, 1, 3, 8, 4, 6, 0, 11];

#[test]
fn mapped_residency_opens_on_llama_and_serves_both_call_sites() {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-mapped-residency-llama-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let arch =
        build_synthetic_llama_real_install(&dir, VOCAB, LAYERS, EXPERTS, "tiny-mixtral-mapped")
            .expect("llama install builds");

    // Set BEFORE the open, because the mode is resolved there.
    std::env::set_var("MFERENCE_EXPERT_RESIDENCY", "mapped");

    let mut runner = RealForwardRunner::open_with_options(&dir, arch, 4096, 16)
        .expect("a llama install opens under mapped expert residency");

    let vocab = VOCAB as usize;

    // Call site 1: the sequential decode path (`families/llama/mod.rs`).
    let mut head = vec![f16::from_f32(0.0); vocab];
    runner
        .produce(5, 0, &mut head)
        .expect("sequential produce succeeds under mapped residency");
    assert!(
        head.iter().any(|v| v.to_f32() != 0.0),
        "the mapped path wrote no logits at all on the sequential decode path; \
         the experts were bound from an empty slot array rather than from their mappings"
    );
    assert!(
        head.iter().all(|v| v.to_f32().is_finite()),
        "a non-finite logit came out of the mapped path on the sequential decode path"
    );

    // Call site 2: the pipelined chunked-prefill driver
    // (`encode_llama_layer_routed_moe_pipelined`, `moe_prefill.rs`), a
    // second, independent threading of the same `mapped` parameter.
    runner.reset();
    let mut logits = vec![f16::from_f32(0.0); vocab];
    runner
        .prefill_chunk(&PROMPT, 0, &mut logits)
        .expect("chunked prefill runs under mapped residency");
    assert!(
        logits.iter().any(|v| v.to_f32() != 0.0),
        "the mapped path wrote no logits at all on the chunked-prefill driver"
    );
    assert!(
        logits.iter().all(|v| v.to_f32().is_finite()),
        "a non-finite logit came out of the mapped path on the chunked-prefill driver"
    );

    std::env::remove_var("MFERENCE_EXPERT_RESIDENCY");
    let _ = std::fs::remove_dir_all(&dir);
}
