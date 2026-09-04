#![cfg(target_os = "macos")]
//! Mapped expert residency (`TURBOSPARK_EXPERT_RESIDENCY=mapped`,
//! `docs/EXPERT_RESIDENCY.md`) through the real `open` path, and the one
//! refusal that path cannot make.
//!
//! # Why this is its own test binary
//!
//! The seam is an ENVIRONMENT variable read at `open`, and a `set_var` is
//! process-global. Cargo runs a binary's `#[test]`s as threads of ONE
//! process, so setting it inside a file that has other cases would leak the
//! mode into whichever of them happened to open a runner concurrently -- and
//! the failure would be a flake, not a red line. A file with exactly one test
//! is a process with exactly one opinion about the environment.
//!
//! `crates/bench/tests/mapped_expert_probe.rs` is the other half and is a
//! different question: it measures what `phys_footprint` CHARGES for a
//! mapping, against a real 13 GB install. This one asks whether the open path
//! and the refusal work at all, on a synthetic fixture, in about a second.

use half::f16;
use turbospark_repack::build_synthetic_gemma4_real_install;
use turbospark_runtime::{ChunkedPrefillRunner, RealForwardRunner};

const VOCAB: i64 = 128;
const PROMPT: [i32; 11] = [5, 9, 2, 7, 1, 3, 8, 4, 6, 0, 11];

/// Mapped residency OPENS on this family, and the batched routed pair is then
/// refused BY NAME rather than running the streamed engine under the mapped
/// label.
///
/// Two claims in one test because they need the same expensive setup and the
/// same poisoned environment, and splitting them would mean two processes to
/// build one fixture twice.
///
/// **The open is the load-bearing half.** `open_expert_streamers` takes a
/// whole separate branch under this mode -- it maps every layer file, wraps
/// each in a Metal buffer, and NULLS every `pread` streamer -- and nothing
/// else in the fast suite reaches that branch at all. Without this, the first
/// thing to execute it would be a 13 GB install behind an `#[ignore]`.
///
/// **The refusal is the half that rots.** The batched routed pair binds one
/// buffer per CACHE SLOT (`MoePrefillRoute::slot` indexes that array) and
/// mapped residency has no slot cache, so the two cannot compose without
/// re-binding per sub-batch.
///
/// What an unrefused combination actually does was MEASURED by deleting the
/// refusal rather than reasoned about, and it is not the silent-ignore this
/// comment first claimed: `slot_buffers[layer]` is empty under this mode, so
/// the run trips `assert!(!blobs.is_empty() ...)` in
/// `gpu::moe_prefill_batch`'s argument encoder. That is a plain `assert!`, so
/// it aborts in RELEASE too -- a panic from a crate the caller never named,
/// mentioning neither seam it set. Loud rather than silent, and still the
/// wrong answer: the fix is a named error, not a crash with a good pedigree.
#[test]
fn mapped_residency_opens_and_then_refuses_the_batched_routed_pair_by_name() {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-mapped-residency-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let arch = build_synthetic_gemma4_real_install(&dir, VOCAB, 2, 16, 4, 8, "tiny-gemma4-mapped")
        .expect("real-naming install builds");

    // Set BEFORE the open, because the mode is resolved there: it decides
    // whether the slot cache is allocated at all, so it cannot be a knob
    // flipped afterwards the way `set_routed_batch_prefill` is.
    std::env::set_var("TURBOSPARK_EXPERT_RESIDENCY", "mapped");

    let mut runner = RealForwardRunner::open_with_options(&dir, arch, 4096, 16)
        .expect("a gemma4 install opens under mapped expert residency");

    // The batched routed pair, which is the combination that cannot work.
    runner.set_routed_batch_prefill(true);
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let err = runner
        .prefill_chunk(&PROMPT, 0, &mut logits)
        .expect_err("the batched routed pair must be refused under mapped residency");
    let text = err.to_string();
    // BOTH seams have to be named. A caller who set two environment variables
    // and got a message naming one of them cannot tell which to unset, and
    // this is the only place either is mentioned.
    assert!(
        text.contains("TURBOSPARK_ROUTED_BATCH"),
        "the refusal must name the batched seam; got {text}"
    );
    assert!(
        text.contains("TURBOSPARK_EXPERT_RESIDENCY"),
        "the refusal must name the residency seam; got {text}"
    );

    // And with that seam off, the SAME runner prefills: the refusal is about
    // the combination rather than about mapped residency being broken. Without
    // this arm the case above passes just as well against an open that mapped
    // nothing and a driver that refuses everything.
    runner.set_routed_batch_prefill(false);
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    runner
        .prefill_chunk(&PROMPT, 0, &mut logits)
        .expect("chunked prefill runs under mapped residency with the batched pair off");
    assert!(
        logits.iter().any(|v| v.to_f32() != 0.0),
        "the mapped path wrote no logits at all; the experts were bound from an \
         empty slot array rather than from their mappings"
    );
    assert!(
        logits.iter().all(|v| v.to_f32().is_finite()),
        "a non-finite logit came out of the mapped path; the per-expert offset \
         into the layer mapping is reading the wrong bytes"
    );

    std::env::remove_var("TURBOSPARK_EXPERT_RESIDENCY");
    let _ = std::fs::remove_dir_all(&dir);
}
