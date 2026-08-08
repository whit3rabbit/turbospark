//! ROADMAP Phase G Stage 1 boundary, enforced rather than documented.
//!
//! The GGUF repack walk produces a real, well-formed `.gturbo` install whose
//! expert bytes are block-quantized (Q8_0 / Q4_K). No kernel in this port
//! reads a block layout yet, so opening one must FAIL, by name, rather than
//! produce plausible-looking garbage.
//!
//! Two independent refusals are checked, because a single one is a single
//! point of failure for a whole class of silently-wrong numbers:
//!
//! 1. `load_manifest` rejects `quant.scheme: "gguf"`.
//! 2. `RealForwardRunner::open` rejects the GGUF dtype tags in the resident
//!    index, which is the backstop for an install whose manifest was edited
//!    to get past (1) -- exactly what someone trying to force one open would
//!    do.
//!
//! Stage 2 flips these deliberately, once there are kernels behind them.

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

/// Writes a GGUF-sourced install and returns its directory plus the arch it
/// declares.
fn gguf_install() -> (std::path::PathBuf, model_io::ArchConfig) {
    let (bytes, _) = build_synthetic_gemma4_gguf(SyntheticGgufShape::default());
    let header = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).expect("parse");
    let dir = tempdir();
    let arch = write_gguf_install_streamed(
        &dir,
        &header,
        &MemoryRangeSource::new(&bytes),
        "gguf-stage1",
        |_| {},
    )
    .expect("write install");
    (dir, arch)
}

#[test]
fn opening_a_gguf_install_fails_and_says_why() {
    let (dir, arch) = gguf_install();

    let text = match RealForwardRunner::open(&dir, arch) {
        Ok(_) => panic!("a GGUF install must not open"),
        Err(e) => e.to_string(),
    };
    assert!(
        text.contains("quantization") || text.contains("GGUF"),
        "the refusal must name the cause, got: {text}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The dtype backstop on its own, reached by rewriting the manifest's quant
/// object to the affine one `load_manifest` accepts. The bytes on disk are
/// still Q8_0 blocks, so the second refusal is the one that has to fire.
#[test]
fn the_dtype_backstop_fires_even_if_the_manifest_is_forged() {
    let (dir, arch) = gguf_install();

    let path = dir.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let affine = serde_json::json!({
        "weightBits": 4,
        "scheme": "affine",
        "scaleType": "bf16",
        "biasType": "bf16",
        "groupSize": 64,
    });
    for slot in ["embedding", "attention", "sharedExpert", "routedExpert"] {
        manifest["quant"][slot] = affine.clone();
    }
    let mut router = affine.clone();
    router["weightBits"] = serde_json::json!(8);
    manifest["quant"]["router"] = router;
    std::fs::write(&path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();

    let text = match RealForwardRunner::open(&dir, arch) {
        Ok(_) => panic!("forging the manifest must not make a GGUF install openable"),
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
