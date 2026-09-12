#![cfg(target_os = "macos")]
//! Mapped expert residency (`TURBOSPARK_EXPERT_RESIDENCY=mapped`,
//! `docs/EXPERT_RESIDENCY.md`) on the `qwen` (`QwenGdnMoe`) flow, through
//! the real `open` path.
//!
//! Sibling of `mapped_expert_residency.rs` (Gemma 4's own test), and its
//! header explains why this is its own test binary: the seam is an
//! environment variable read once at `open`, and `set_var` is
//! process-global, so a file with any other `#[test]` would leak the mode
//! into whatever else happened to open a runner concurrently.
//!
//! Two claims, same reason as the Gemma 4 file: they need the same
//! expensive setup and the same poisoned environment.

use half::f16;
use turbospark_runtime::{
    DraftPolicies, ExpertCacheSlots, ExpertResidency, KvQuant, LogitProducer, MtpDraftPolicy,
    RealForwardRunner, SteeringPolicy,
};

#[test]
fn mapped_residency_opens_on_qwen_and_then_refuses_the_batched_verify_by_name() {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-mapped-residency-qwen-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    const VOCAB: i64 = 128;
    const LAYERS: i64 = 4;
    const EXPERTS: i64 = 8;

    // Set BEFORE the open, because the mode is resolved there: it decides
    // whether the slot cache is allocated at all.
    std::env::set_var("TURBOSPARK_EXPERT_RESIDENCY", "mapped");

    // The MTP-head variant is what makes the batched verify pass
    // reachable at all: `produce_batched` allocates its M-row scratch
    // beside a drafter, and a MoE install with no head has nowhere to
    // put it (`turbospark_repack::build_synthetic_qwen_gdn_moe_install_with_mtp`'s
    // own doc).
    let arch = turbospark_repack::build_synthetic_qwen_gdn_moe_install_with_mtp(
        &dir,
        VOCAB,
        LAYERS,
        EXPERTS,
        "tiny-qwen36-mapped",
    )
    .expect("a MoE qwen3_5 install with an MTP head builds");

    // EXPLICIT Mapped rather than the env seam: the measuring callers'
    // open entries pin `Streamed` (AGENTS.md Gotcha 35), so a test whose
    // subject is the mapped branch says so through the widest entry.
    let mut runner = RealForwardRunner::open_with_residency(
        &dir,
        arch,
        4096,
        ExpertCacheSlots::Fixed(16),
        DraftPolicies::mtp(MtpDraftPolicy::Fixed(3)),
        SteeringPolicy::off(),
        1,
        KvQuant::Off,
        ExpertResidency::Mapped,
    )
    .expect("a qwen MoE install opens under mapped expert residency");

    // The open is the load-bearing half: `open_expert_streamers` maps every
    // layer file, wraps each in a Metal buffer, and NULLS every `pread`
    // streamer, and nothing else in the fast suite reaches that branch for
    // this family at all.
    let vocab = VOCAB as usize;
    let mut head = vec![f16::from_f32(0.0); vocab];
    runner
        .produce(5, 0, &mut head)
        .expect("produce succeeds under mapped residency");
    assert!(
        head.iter().any(|v| v.to_f32() != 0.0),
        "the mapped path wrote no logits at all; the experts were bound from an \
         empty slot array rather than from their mappings"
    );
    assert!(
        head.iter().all(|v| v.to_f32().is_finite()),
        "a non-finite logit came out of the mapped path; the per-expert offset \
         into the layer mapping is reading the wrong bytes"
    );

    // The refusal is the half that rots: the batched verify pass binds one
    // buffer per CACHE SLOT, and mapped residency has no slot cache.
    runner.reset();
    let tokens = [5i32, 9, 2];
    let mut logits = vec![f16::from_f32(0.0); tokens.len() * vocab];
    let err = runner
        .produce_batched(&tokens, 0, &mut logits)
        .expect_err("batched verify under mapped residency must be refused by name");
    let text = err.to_string();
    assert!(
        text.contains("TURBOSPARK_EXPERT_RESIDENCY"),
        "the refusal must name the seam the caller set; got {text}"
    );

    std::env::remove_var("TURBOSPARK_EXPERT_RESIDENCY");
    let _ = std::fs::remove_dir_all(&dir);
}
