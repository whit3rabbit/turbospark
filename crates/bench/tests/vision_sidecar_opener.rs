//! `open_model_runner_with_context_and_vision_sidecar` on synthetic installs
//! (vision memory sidecar, Part A6's opener, built in Part A4).
//!
//! Not a real-model gate: this proves the opener attaches a sidecar to a
//! text-only trunk correctly and returns a usable runner, the same way
//! `crates/runtime/tests/vision_sidecar_synthetic.rs` proves
//! `attach_vision_sidecar` itself. Part A6's real-install comparison against
//! `qwen38-27b-vision.gturbo` is a separate, later, `#[ignore]`d test this
//! session does not write.
#![cfg(target_os = "macos")]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use repack::{build_synthetic_qwen_gdn_dense_install, build_synthetic_vision_sidecar};
use turbospark_bench::real_model_open::open_model_runner_with_context_and_vision_sidecar;
use turbospark_vision_io::{GridThw, PreprocessParams, PreprocessedImage};

static COUNTER: AtomicU64 = AtomicU64::new(0);

const VOCAB: i64 = 256;
const LAYERS: i64 = 4;
const MAX_CONTEXT: u32 = 256;
const SLOTS: usize = 16;

fn temp_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-bench-vision-sidecar-opener-{tag}-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// None of the synthetic install builders in `crates/repack` write a
/// `tokenizer.json` -- they build only what `RealForwardRunner::open` reads
/// -- but `open_model_runner_with_context_and_vision_sidecar` also calls
/// `MfTokenizer::load_from_dir(model_dir)` (this crate's `open_with_arch`),
/// so a TRUNK directory needs one dropped in for that call to succeed at
/// all. The vendored ChatML fixture is unrelated to the synthetic arch's own
/// vocab size, which is fine here: nothing in these tests encodes text or
/// generates through it, only `has_vision_tower` / `encode_image`.
fn copy_tokenizer_fixture(dir: &std::path::Path) {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tokenizer/tests/fixtures/ChatMLTokenizer/tokenizer.json");
    std::fs::copy(&fixture, dir.join("tokenizer.json")).expect("copy the fixture tokenizer.json");
}

fn params() -> PreprocessParams {
    let v = repack::tiny_vision_config();
    PreprocessParams {
        patch_size: v.patch_size as usize,
        temporal_patch_size: v.temporal_patch_size as usize,
        merge_size: v.spatial_merge_size as usize,
        in_channels: v.in_channels as usize,
        min_pixels: 1,
        max_pixels: 1 << 20,
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
        rescale_factor: 1.0 / 255.0,
    }
}

fn image(p: &PreprocessParams) -> PreprocessedImage {
    let grid = GridThw::new(1, 4, 4);
    let dim = p.patch_dim();
    let patch_rows: Vec<f32> = (0..grid.patches() * dim)
        .map(|i| ((i as f32) * 0.0173).sin() * 0.8)
        .collect();
    PreprocessedImage {
        merged_tokens: grid.merged_tokens(p.merge_size),
        patch_rows,
        grid,
        resized: (4 * p.patch_size, 4 * p.patch_size),
    }
}

/// The opener attaches a compatible sidecar and hands back a runner that
/// actually serves an image through it -- not merely one that opens without
/// erroring.
#[test]
fn the_opener_attaches_a_compatible_sidecar_and_the_runner_serves_an_image() {
    let trunk_dir = temp_dir("trunk");
    build_synthetic_qwen_gdn_dense_install(&trunk_dir, VOCAB, LAYERS, "qwen35-opener-trunk")
        .expect("a text-only dense install builds");
    copy_tokenizer_fixture(&trunk_dir);

    let sidecar_dir = temp_dir("sidecar");
    build_synthetic_vision_sidecar(&sidecar_dir, "qwen35-opener-sidecar")
        .expect("a standalone sidecar builds");

    let (mut runner, tokenizer) = open_model_runner_with_context_and_vision_sidecar(
        &trunk_dir,
        SLOTS,
        MAX_CONTEXT,
        &sidecar_dir,
    )
    .expect("the opener should attach the sidecar and return a usable runner");

    assert!(
        runner.has_vision_tower(),
        "the opened runner must report a tower once the sidecar is attached"
    );
    assert_eq!(runner.vision_dir(), sidecar_dir.as_path());
    // The tokenizer is handed back too, exactly as every other opener in
    // this crate returns one -- a caller wiring `verify_image_markers`
    // ahead of an image reaches it through this same tuple.
    assert!(tokenizer.vocab_size > 0);

    let p = params();
    let img = image(&p);
    runner
        .encode_image(&img, &p)
        .expect("the sidecar-backed tower should actually run");
    assert_eq!(runner.vision_is_sidecar(), Some(true));

    let _ = std::fs::remove_dir_all(&trunk_dir);
    let _ = std::fs::remove_dir_all(&sidecar_dir);
}

/// A mismatched sidecar (wrong family) is refused by the opener, and the
/// refusal names both families -- the same message
/// `RealForwardRunner::attach_vision_sidecar` itself gives, since the opener
/// adds no wrapping of its own beyond the path in its error prefix.
///
/// `build_synthetic_vision_sidecar` always pairs with `QwenGdnDense`
/// (`crates/repack/src/synthetic_qwen/vision.rs`), so a Gemma 4 trunk is
/// enough to force the mismatch -- no separate "sidecar for family X"
/// builder is needed, matching
/// `crates/runtime/tests/vision_sidecar_synthetic.rs`'s own
/// `attach_vision_sidecar_refuses_a_family_mismatch` case.
#[test]
fn the_opener_refuses_a_family_mismatched_sidecar() {
    let trunk_dir = temp_dir("mismatch-trunk");
    let trunk_arch =
        repack::build_synthetic_gemma4_install(&trunk_dir, VOCAB, LAYERS, "gemma4-opener-mismatch")
            .expect("a synthetic Gemma 4 install builds");
    assert_eq!(trunk_arch.family, model_io::ModelFamily::Gemma4);
    copy_tokenizer_fixture(&trunk_dir);

    let sidecar_dir = temp_dir("mismatch-sidecar");
    build_synthetic_vision_sidecar(&sidecar_dir, "qwen35-opener-mismatch-sidecar")
        .expect("a standalone sidecar builds");

    // `expect_err` needs the Ok type to be `Debug`, and `RealForwardRunner`
    // is not (`crates/runtime/CLAUDE.md` Gotcha 17) -- destructure instead.
    let Err(err) = open_model_runner_with_context_and_vision_sidecar(
        &trunk_dir,
        SLOTS,
        MAX_CONTEXT,
        &sidecar_dir,
    ) else {
        panic!("a family-mismatched sidecar must not attach");
    };
    assert!(
        err.contains("gemma4") && err.contains("qwen35"),
        "unexpected error: {err}"
    );

    let _ = std::fs::remove_dir_all(&trunk_dir);
    let _ = std::fs::remove_dir_all(&sidecar_dir);
}
