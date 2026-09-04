#![cfg(target_os = "macos")]
//! Mapped expert residency (`TURBOSPARK_EXPERT_RESIDENCY=mapped`,
//! `docs/EXPERT_RESIDENCY.md`) on the `gptoss` flow, through the real `open`
//! path.
//!
//! Sibling of `mapped_expert_residency.rs` (Gemma 4), `_qwen.rs` and
//! `_llama.rs`. Own test binary for the same reason all three are: the seam
//! is a process-global environment variable read once at `open`.
//!
//! This family's fixtures go through the raw GGUF builder rather than a
//! one-call "real install" helper, matching `real_forward_gptoss.rs` and
//! `real_forward_gptoss_chunked.rs`.
//!
//! Two claims, same reason as the other three files: they need the same
//! expensive setup and the same poisoned environment. The refusal half
//! mirrors Gemma 4's own test exactly, since `families/gptoss/moe_batch.rs`
//! carries the identical structural conflict (one buffer per CACHE SLOT
//! against mapped residency's one-buffer-per-layer-plus-offset shape).

use half::f16;
use turbospark_repack::{
    build_synthetic_gpt_oss_gguf, parse_gguf_header, write_gguf_install_streamed,
    MemoryRangeSource, SyntheticGptOssShape, GGUF_DEFAULT_MAX_HEADER_BYTES,
};
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};

fn build_install(dir: &std::path::Path) -> model_io::ArchConfig {
    let shape = SyntheticGptOssShape::default();
    let (bytes, _) = build_synthetic_gpt_oss_gguf(shape);
    let header = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).expect("parse");
    write_gguf_install_streamed(
        dir,
        &header,
        &MemoryRangeSource::new(&bytes),
        "gptoss-mapped",
        |_| {},
    )
    .expect("write install")
}

#[test]
fn mapped_residency_opens_on_gptoss_and_then_refuses_the_batched_routed_pair_by_name() {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-mapped-residency-gptoss-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let arch = build_install(&dir);
    let vocab = arch.vocab_size as usize;

    // Set BEFORE the open, because the mode is resolved there.
    std::env::set_var("TURBOSPARK_EXPERT_RESIDENCY", "mapped");

    let mut runner = RealForwardRunner::open_with_options(&dir, arch, 4096, 16)
        .expect("a gpt-oss install opens under mapped expert residency");

    // The open is the load-bearing half, same claim as the other three
    // files: nothing else in the fast suite reaches the mapped branch of
    // `open_expert_streamers` for this family.
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

    // The refusal is the half that rots: the MXFP4 batched routed pair binds
    // one buffer per CACHE SLOT, and mapped residency has no slot cache.
    runner.reset();
    runner.set_routed_batch_prefill(true);
    let prompt = [5i32, 9, 2, 7, 1, 3, 8, 4, 6, 0, 11];
    let mut logits = vec![f16::from_f32(0.0); vocab];
    let err = runner
        .prefill_chunk(&prompt, 0, &mut logits)
        .expect_err("the batched routed pair must be refused under mapped residency");
    let text = err.to_string();
    assert!(
        text.contains("TURBOSPARK_ROUTED_BATCH"),
        "the refusal must name the batched seam; got {text}"
    );
    assert!(
        text.contains("TURBOSPARK_EXPERT_RESIDENCY"),
        "the refusal must name the residency seam; got {text}"
    );

    // And with that seam off, the SAME runner prefills: the refusal is about
    // the combination rather than about mapped residency being broken.
    runner.set_routed_batch_prefill(false);
    let mut logits = vec![f16::from_f32(0.0); vocab];
    runner
        .prefill_chunk(&prompt, 0, &mut logits)
        .expect("chunked prefill runs under mapped residency with the batched pair off");
    assert!(
        logits.iter().any(|v| v.to_f32() != 0.0),
        "the mapped path wrote no logits at all on the chunked-prefill driver"
    );
    assert!(
        logits.iter().all(|v| v.to_f32().is_finite()),
        "a non-finite logit came out of the mapped path on the chunked-prefill driver"
    );

    std::env::remove_var("TURBOSPARK_EXPERT_RESIDENCY");
    let _ = std::fs::remove_dir_all(&dir);
}
