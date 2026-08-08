//! ROADMAP Phase G Stage 2's boundary, enforced rather than documented.
//!
//! Stage 1 refused every GGUF install, because no kernel in this port read a
//! block layout. Stage 2 moved that line rather than erasing it. Q8_0 and
//! Q4_K each have a resident GEMV, an embedding lookup and the routed-expert
//! decode pair behind them, and Q6_K has a resident GEMV, which is all any
//! real file asks of it; those three OPEN. Q4_0 has none and still refuses.
//!
//! Both directions are asserted here, and the refusal is checked twice over,
//! because a single gate is a single point of failure for a whole class of
//! silently-wrong numbers:
//!
//! 1. `load_manifest` refuses a `scheme: "gguf"` slot whose `ggmlType` is
//!    outside `model_io::EXECUTABLE_GGUF_TYPES`.
//! 2. `RealForwardRunner::open` refuses a resident-index dtype tag outside
//!    the same set, which is the backstop for an install whose manifest was
//!    edited to get past (1) -- exactly what someone trying to force one open
//!    would do. It reads the bytes rather than a claim about them.

#![cfg(target_os = "macos")]

use std::sync::atomic::{AtomicU64, Ordering};

use mrefrust_repack::{
    build_synthetic_gemma4_gguf, parse_gguf_header, write_gguf_install_streamed, MemoryRangeSource,
    SyntheticGgufShape, GGUF_DEFAULT_MAX_HEADER_BYTES,
};
use mrefrust_runtime::RealForwardRunner;

fn tempdir() -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "mrefrust-gguf-refused-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}

/// A shape whose every quantized row is a whole number of 32-element Q8_0
/// blocks, which is what real ggml files always are: the routed
/// `moe_intermediate` rows and the `hidden`-length rows both have to tile.
/// The default fixture shape does NOT (its `moe_intermediate` is 16), and it
/// stays that way because the repack-side tests want the smallest file.
fn executable_shape() -> SyntheticGgufShape {
    SyntheticGgufShape {
        moe_intermediate: 32,
        ..SyntheticGgufShape::default()
    }
}

/// Writes a GGUF-sourced install and returns its directory plus the arch it
/// declares.
fn gguf_install(shape: SyntheticGgufShape) -> (std::path::PathBuf, model_io::ArchConfig) {
    let (bytes, _) = build_synthetic_gemma4_gguf(shape);
    let header = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).expect("parse");
    let dir = tempdir();
    let arch = write_gguf_install_streamed(
        &dir,
        &header,
        &MemoryRangeSource::new(&bytes),
        "gguf-stage2",
        |_| {},
    )
    .expect("write install");
    (dir, arch)
}

/// Rewrites every resident-index entry carrying `from` to carry `to`, in
/// place, leaving the bytes those entries point at untouched. That is exactly
/// the state a hand-forged install would be in: a block type this port cannot
/// execute, claiming to be one it can, or the reverse.
fn retag_dtypes(dir: &std::path::Path, from: u8, to: u8) -> usize {
    let path = dir.join("model_weights.bin");
    let mut bytes = std::fs::read(&path).unwrap();
    let index = model_io::load_resident_index(&path).expect("index");
    let mut changed = 0;
    for i in 0..index.header.entry_count as usize {
        let at = model_io::HEADER_BYTES + i * model_io::ENTRY_BYTES + 6;
        if bytes[at] == from {
            bytes[at] = to;
            changed += 1;
        }
    }
    std::fs::write(&path, &bytes).unwrap();
    changed
}

/// The Stage 2 deliverable: a Q8_0 GGUF install opens, and the kernels behind
/// it are the ones this test is really about (the embedding lookup, the
/// resident GEMV, and the routed-expert decode pair).
#[test]
fn a_q8_0_gguf_install_opens() {
    let (dir, arch) = gguf_install(executable_shape());

    match RealForwardRunner::open(&dir, arch) {
        Ok(_) => {}
        Err(e) => panic!("a Q8_0 GGUF install must open now that its kernels exist: {e}"),
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// The K-quant mixture a real `Q4_K_M` carries, end to end: routed experts
/// and the embedding table at Q4_K, the attention projections at Q6_K, the
/// rest Q8_0. Three block types in one install, which is the case a
/// single-type fixture cannot make: every dispatch site has to pick from the
/// TENSOR rather than from one decision made at open.
#[test]
fn a_mixed_k_quant_gguf_install_opens() {
    let (dir, arch) = gguf_install(SyntheticGgufShape::k_quant());

    match RealForwardRunner::open(&dir, arch) {
        Ok(_) => {}
        Err(e) => panic!("a Q4_K/Q6_K GGUF install must open now that its kernels exist: {e}"),
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// The manifest gate, on a block type with no kernel. Q4_0 is the one the
/// parser knows and nothing decodes: it has no CPU reference, no GEMV and no
/// expert pair, so an install of one must not open and the refusal must say
/// which type.
#[test]
fn a_block_type_without_kernels_is_refused_by_the_manifest() {
    let (dir, arch) = gguf_install(executable_shape());

    let path = dir.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    for slot in ["embedding", "attention", "sharedExpert", "routedExpert"] {
        manifest["quant"][slot]["ggmlType"] = serde_json::json!("Q4_0");
    }
    std::fs::write(&path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();

    let text = match RealForwardRunner::open(&dir, arch) {
        Ok(_) => panic!("a Q4_0 install must not open"),
        Err(e) => e.to_string(),
    };
    assert!(
        text.contains("Q4_0") || text.contains("q4_0"),
        "the refusal must name the block type, got: {text}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The dtype backstop on its own: the manifest still says Q8_0, but the
/// bytes on disk are tagged Q4_0. `open` has to believe the index.
#[test]
fn the_dtype_backstop_fires_even_if_the_manifest_is_forged() {
    let (dir, arch) = gguf_install(executable_shape());

    let changed = retag_dtypes(&dir, 6, 9);
    assert!(changed > 0, "the fixture carries no Q8_0 resident tensors");

    let text = match RealForwardRunner::open(&dir, arch) {
        Ok(_) => panic!("forging the manifest must not make a Q4_0 install openable"),
        Err(e) => e.to_string(),
    };
    assert!(
        text.contains("GGUF block dtype"),
        "expected the resident-index backstop, got: {text}"
    );
    assert!(
        text.contains("Phase G"),
        "the refusal should point at the work that lifts it, got: {text}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// Opening is not running. This drives real decode steps through the whole
/// Q8_0 path on real Metal hardware: the embedding lookup, the attention and
/// shared-expert GEMVs, the routed-expert decode pair reading streamed
/// blobs, and the output head. The fixture's weights are deterministic
/// patterns rather than trained ones, so nothing about the TOKENS means
/// anything; what is asserted is that every logit is finite and that the
/// distribution is not degenerate, which a kernel reading a block layout
/// wrongly does not satisfy for long.
#[test]
fn a_q8_0_gguf_install_decodes() {
    decodes(executable_shape());
}

/// The same drive through the mixed K-quant install, which is where the Q4_K
/// routed pair, the Q4_K embedding lookup and the Q6_K resident GEMV all run
/// on real hardware inside one forward pass.
#[test]
fn a_mixed_k_quant_gguf_install_decodes() {
    decodes(SyntheticGgufShape::k_quant());
}

fn decodes(shape: SyntheticGgufShape) {
    use half::f16;
    use mrefrust_runtime::LogitProducer;

    let vocab = shape.vocab as usize;
    let (dir, arch) = gguf_install(shape);
    let mut runner = RealForwardRunner::open(&dir, arch).expect("opens");

    runner.reset();
    let mut token = 5i32;
    for position in 0..4usize {
        let mut logits = vec![f16::from_f32(0.0); vocab];
        runner
            .produce(token, position, &mut logits)
            .expect("produce succeeds");

        let bad: Vec<usize> = logits
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.to_f32().is_finite())
            .map(|(i, _)| i)
            .collect();
        assert!(
            bad.is_empty(),
            "non-finite logit at position {position}: {} of {vocab}, first {:?}",
            bad.len(),
            &bad[..bad.len().min(8)]
        );
        let first = logits[0].to_f32();
        assert!(
            logits.iter().any(|v| v.to_f32() != first),
            "every logit is {first} at position {position}: the head produced nothing"
        );

        token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
            .map(|(i, _)| i as i32)
            .unwrap();
    }

    std::fs::remove_dir_all(&dir).ok();
}
