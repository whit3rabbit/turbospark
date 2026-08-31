#![cfg(target_os = "macos")]
//! Mapped vision residency (`MFERENCE_VISION_RESIDENCY=mapped`,
//! `docs/EXPERT_RESIDENCY.md`) through the real `open` path, on a synthetic
//! install.
//!
//! # Why this is its own test binary
//!
//! Same reason as `mapped_expert_residency.rs`: the seam is an ENVIRONMENT
//! variable read at `VisionTower::open`, and `set_var` is process-global.
//! Cargo runs a binary's `#[test]`s as threads of one process, so a file with
//! any other case risks the mode leaking into a concurrently-opened runner.
//! A file with exactly one test is a process with exactly one opinion about
//! the environment.
//!
//! # Why this is a byte-identity check AND an engagement check, not either alone
//!
//! Byte-identity alone cannot distinguish "the mapped arm ran and produced
//! the same output" from "the mapped arm silently fell through to pread" --
//! both would pass parity trivially in the second case, which is exactly the
//! class of bug `docs/EXPERT_RESIDENCY.md` already documents for the routed
//! case. `RealForwardRunner::vision_residency_is_mapped` is the accessor that
//! closes the gap: it reports which arm the tower actually took, independent
//! of what the tower computed.

use turbospark_repack::build_synthetic_qwen_gdn_dense_install_with_vision_streamed;
use turbospark_runtime::RealForwardRunner;
use turbospark_vision_io::{GridThw, PreprocessParams, PreprocessedImage};

const VOCAB: i64 = 256;
const LAYERS: i64 = 4;
const BITS: u32 = 1;
const GRID_H: usize = 4;
const GRID_W: usize = 4;

fn temp_dir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-mapped-vision-residency-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn params() -> PreprocessParams {
    let v = turbospark_repack::tiny_vision_config();
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

/// A deterministic page, matching `vision_tower_synthetic.rs`'s own fixture
/// image so this file's numbers are directly comparable to that one's.
fn image(params: &PreprocessParams) -> PreprocessedImage {
    let grid = GridThw::new(1, GRID_H, GRID_W);
    let dim = params.patch_dim();
    let patch_rows: Vec<f32> = (0..grid.patches() * dim)
        .map(|i| ((i as f32) * 0.0173).sin() * 0.8)
        .collect();
    PreprocessedImage {
        merged_tokens: grid.merged_tokens(params.merge_size),
        patch_rows,
        grid,
        resized: (GRID_H * params.patch_size, GRID_W * params.patch_size),
    }
}

/// Both residency arms must produce byte-identical embeddings on the same
/// image, and the mapped arm must PROVE it actually engaged rather than
/// silently falling through to the pread streamer.
#[test]
fn mapped_and_pread_vision_residency_agree_and_the_mapped_arm_engages() {
    let dir = temp_dir();
    let arch = build_synthetic_qwen_gdn_dense_install_with_vision_streamed(
        &dir,
        VOCAB,
        LAYERS,
        "qwen35-vision-mapped-residency",
        BITS,
    )
    .expect("the streamed writer builds a dense install with a vision tower");

    let p = params();
    let img = image(&p);

    // The default arm: unset, and the ordinary pread streamer.
    std::env::remove_var("MFERENCE_VISION_RESIDENCY");
    let mut pread_runner =
        RealForwardRunner::open(&dir, arch.clone()).expect("the install opens (pread arm)");
    assert_eq!(
        pread_runner.vision_residency_is_mapped(),
        None,
        "the tower is lazy; it must not have opened before the first image"
    );
    let pread_embedding = pread_runner
        .encode_image(&img, &p)
        .expect("the tower runs under the default pread arm");
    assert_eq!(
        pread_runner.vision_residency_is_mapped(),
        Some(false),
        "the default arm must report itself as NOT mapped"
    );

    // The mapped arm, set BEFORE open: the mode is resolved at
    // `VisionTower::open`, which is lazy (first image), so it must be set
    // before that call rather than before `RealForwardRunner::open`.
    std::env::set_var("MFERENCE_VISION_RESIDENCY", "mapped");
    let mut mapped_runner =
        RealForwardRunner::open(&dir, arch).expect("the install opens (mapped arm)");
    let mapped_embedding = mapped_runner
        .encode_image(&img, &p)
        .expect("the tower runs under the mapped residency arm");
    assert_eq!(
        mapped_runner.vision_residency_is_mapped(),
        Some(true),
        "the mapped arm must report itself as mapped; if this reads Some(false) \
         the tower silently fell through to the pread streamer instead of \
         engaging MFERENCE_VISION_RESIDENCY=mapped"
    );
    std::env::remove_var("MFERENCE_VISION_RESIDENCY");

    // Sanity, matching mapped_expert_residency.rs's own checks: neither arm
    // produced silence or garbage.
    assert!(
        mapped_embedding.rows.iter().any(|&bits| bits != 0),
        "the mapped path wrote no rows at all"
    );
    assert!(
        mapped_embedding
            .rows
            .iter()
            .all(|&bits| half::f16::from_bits(bits).to_f32().is_finite()),
        "a non-finite value came out of the mapped path; the per-block offset \
         into the tower's mapping is reading the wrong bytes"
    );

    // The actual A/B: same image, same install, two residency arms, same
    // bytes. `base + roles.at(role)` reduces to `roles.at(role)` at base 0,
    // so the pread arm's own output cannot have moved either.
    assert_eq!(
        mapped_embedding.merged_tokens, pread_embedding.merged_tokens,
        "the two arms disagree on merged token count"
    );
    assert_eq!(
        mapped_embedding.out_hidden, pread_embedding.out_hidden,
        "the two arms disagree on hidden width"
    );
    assert_eq!(
        mapped_embedding.rows, pread_embedding.rows,
        "mapped and pread vision residency must produce byte-identical embeddings"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
