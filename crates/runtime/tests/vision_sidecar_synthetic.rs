//! `RealForwardRunner::attach_vision_sidecar` on synthetic installs (vision
//! memory sidecar, Part A2).
//!
//! # The headline claim
//!
//! A text-only trunk with a sidecar tower attached must produce
//! BYTE-IDENTICAL vision output to a combined install carrying the SAME
//! tower bytes (`build_synthetic_vision_sidecar` and
//! `build_synthetic_qwen_gdn_dense_install_with_vision*` both build their
//! tower from the same `vision_tower_tensors()`, per `crates/repack`'s own
//! module doc). If the sidecar path bound the wrong resident weights, or
//! read `packed_vision/` out of the wrong directory, this is where it would
//! show up -- not as an error, but as a plausible, differently-wrong
//! embedding.
//!
//! # `vision_is_sidecar()` and why it matters even though it looks provable
//! by construction here
//!
//! A TEXT-ONLY trunk cannot, on its own, ever have a combined tower to fall
//! through to, so it looks like the accessor is redundant on this fixture.
//! It is asserted anyway, per the module's own stated testing philosophy for
//! this class of engagement flag (`VisionTower::is_mapped_residency`'s
//! precedent): the fixture proves nothing on its own about a FUTURE session
//! that reuses this test's shape against an install that does carry a tower,
//! and the accessor is the whole reason a caller elsewhere does not have to
//! reason about that.
#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_repack::{
    build_synthetic_gemma4_install, build_synthetic_qwen_gdn_dense_install,
    build_synthetic_qwen_gdn_dense_install_with_vision_streamed, build_synthetic_vision_sidecar,
    tiny_vision_config,
};
use turbospark_runtime::RealForwardRunner;
use turbospark_vision_io::{GridThw, PreprocessParams, PreprocessedImage};

static COUNTER: AtomicU64 = AtomicU64::new(0);

const VOCAB: i64 = 256;
const LAYERS: i64 = 4;
const BITS: u32 = 1;
const GRID_H: usize = 4;
const GRID_W: usize = 4;

fn temp_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-vision-sidecar-{tag}-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn params() -> PreprocessParams {
    let v = tiny_vision_config();
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

/// A deterministic page, matching the shape `vision_tower_synthetic.rs` and
/// `mapped_vision_residency.rs` already use, so this file's numbers are
/// directly comparable to theirs.
fn image(p: &PreprocessParams) -> PreprocessedImage {
    let grid = GridThw::new(1, GRID_H, GRID_W);
    let dim = p.patch_dim();
    let patch_rows: Vec<f32> = (0..grid.patches() * dim)
        .map(|i| ((i as f32) * 0.0173).sin() * 0.8)
        .collect();
    PreprocessedImage {
        merged_tokens: grid.merged_tokens(p.merge_size),
        patch_rows,
        grid,
        resized: (GRID_H * p.patch_size, GRID_W * p.patch_size),
    }
}

fn open(dir: &Path, arch: model_io::ArchConfig) -> RealForwardRunner {
    RealForwardRunner::open(dir, arch).expect("the install opens")
}

/// Edits an already-built standalone sidecar's declared pairing hidden size,
/// in BOTH files it appears in (`vision_sidecar.json`'s own record and
/// `manifest.json`'s `arch.visionOutHiddenSize`/`arch.hiddenSize`, which
/// `sidecar_arch` always sets to the same value), so the two stay
/// self-consistent -- `model_io::load_vision_sidecar` cross-checks them
/// against EACH OTHER and would refuse a sidecar that disagreed with itself
/// before this test's own trunk-vs-sidecar check ever ran.
///
/// This is what lets the hidden-size-mismatch test below exercise
/// `RealForwardRunner::attach_vision_sidecar`'s OWN comparison against the
/// trunk, rather than `model_io::load_vision_sidecar`'s internal
/// record/manifest agreement check -- the two are different refusals and
/// this file wants to isolate the first.
fn rewrite_sidecar_hidden_size(sidecar_dir: &Path, new_size: i64) {
    let record_path = sidecar_dir.join("vision_sidecar.json");
    let mut record: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&record_path).unwrap()).unwrap();
    record["pairsWith"]["hiddenSize"] = serde_json::json!(new_size);
    std::fs::write(&record_path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();

    let manifest_path = sidecar_dir.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    manifest["arch"]["hiddenSize"] = serde_json::json!(new_size);
    manifest["arch"]["visionOutHiddenSize"] = serde_json::json!(new_size);
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
}

/// The headline test: a text-only trunk plus an attached sidecar must
/// produce byte-identical vision output (rows AND every captured stage) to
/// a combined install carrying the same tower bytes.
#[test]
fn a_sidecar_attached_to_a_text_only_trunk_matches_a_combined_install_byte_for_byte() {
    let combined_dir = temp_dir("combined");
    let combined_arch = build_synthetic_qwen_gdn_dense_install_with_vision_streamed(
        &combined_dir,
        VOCAB,
        LAYERS,
        "qwen35-combined",
        BITS,
    )
    .expect("the streamed writer builds a dense install with a vision tower");

    let trunk_dir = temp_dir("trunk");
    let trunk_arch =
        build_synthetic_qwen_gdn_dense_install(&trunk_dir, VOCAB, LAYERS, "qwen35-trunk-only")
            .expect("a text-only dense install builds");

    let sidecar_dir = temp_dir("sidecar");
    build_synthetic_vision_sidecar(&sidecar_dir, "qwen35-sidecar")
        .expect("a standalone sidecar builds from the same tower tensors");

    let mut combined = open(&combined_dir, combined_arch);
    let mut sidecar_runner = open(&trunk_dir, trunk_arch);

    // Sanity: the trunk really is headless before attach.
    assert!(!sidecar_runner.has_vision_tower());
    assert_eq!(sidecar_runner.vision_dir(), trunk_dir.as_path());

    sidecar_runner
        .attach_vision_sidecar(&sidecar_dir)
        .expect("a compatible sidecar attaches");
    assert!(sidecar_runner.has_vision_tower());
    assert_eq!(sidecar_runner.vision_dir(), sidecar_dir.as_path());

    let p = params();
    let img = image(&p);

    let (combined_embedding, combined_stages) = combined
        .encode_image_with_stages(&img, &p)
        .expect("the combined install's tower runs");
    let (sidecar_embedding, sidecar_stages) = sidecar_runner
        .encode_image_with_stages(&img, &p)
        .expect("the sidecar-attached trunk's tower runs");

    assert_eq!(
        combined.vision_is_sidecar(),
        Some(false),
        "the combined install's own tower must not report itself as sidecar-backed"
    );
    assert_eq!(
        sidecar_runner.vision_is_sidecar(),
        Some(true),
        "the attached tower must report itself as sidecar-backed"
    );

    assert_eq!(
        combined_embedding.merged_tokens, sidecar_embedding.merged_tokens,
        "the two paths disagree on merged token count"
    );
    assert_eq!(
        combined_embedding.out_hidden, sidecar_embedding.out_hidden,
        "the two paths disagree on hidden width"
    );
    assert_eq!(
        combined_embedding.rows, sidecar_embedding.rows,
        "a sidecar-attached trunk must produce byte-identical vision output to a combined \
         install built from the same tower bytes"
    );
    assert_eq!(
        combined_stages.patch_embed, sidecar_stages.patch_embed,
        "patch embedding stage diverged"
    );
    assert_eq!(
        combined_stages.block_first, sidecar_stages.block_first,
        "first block stage diverged"
    );
    assert_eq!(
        combined_stages.block_last, sidecar_stages.block_last,
        "last block stage diverged"
    );

    let _ = std::fs::remove_dir_all(&combined_dir);
    let _ = std::fs::remove_dir_all(&trunk_dir);
    let _ = std::fs::remove_dir_all(&sidecar_dir);
}

/// A text-only trunk with NO sidecar attached stays headless, and an image
/// is refused by the same "no tower" error a combined install with no
/// tower would give -- no partial state, no silent no-op.
#[test]
fn a_text_only_trunk_with_no_sidecar_refuses_an_image() {
    let dir = temp_dir("headless");
    let arch = build_synthetic_qwen_gdn_dense_install(&dir, VOCAB, LAYERS, "qwen35-headless")
        .expect("a text-only dense install builds");
    let mut runner = open(&dir, arch);

    assert!(!runner.has_vision_tower());
    assert!(!runner.vision_config().is_active());

    let p = params();
    let img = image(&p);
    let err = runner
        .encode_image(&img, &p)
        .expect_err("a headless session must refuse an image rather than silently no-op");
    assert!(
        err.to_string().contains("declares no vision tower"),
        "unexpected error: {err}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// `attach_vision_sidecar` refuses a sidecar whose declared family does not
/// match the trunk's, even though the sidecar is perfectly well-formed on
/// its own.
#[test]
fn attach_vision_sidecar_refuses_a_family_mismatch() {
    let trunk_dir = temp_dir("family-trunk");
    // A Gemma 4 trunk: a different family from the sidecar's QwenGdnDense
    // pairing, built through the older short-name synthetic generator since
    // this test never opens the tower or decodes anything -- it only needs
    // `RealForwardRunner::open` to succeed with `arch.family ==
    // ModelFamily::Gemma4`.
    let trunk_arch = build_synthetic_gemma4_install(&trunk_dir, VOCAB, LAYERS, "gemma4-trunk")
        .expect("a synthetic Gemma 4 install builds");
    assert_eq!(trunk_arch.family, model_io::ModelFamily::Gemma4);

    let sidecar_dir = temp_dir("family-sidecar");
    build_synthetic_vision_sidecar(&sidecar_dir, "qwen35-sidecar-family")
        .expect("a standalone sidecar builds");

    let mut runner = open(&trunk_dir, trunk_arch);
    let err = runner
        .attach_vision_sidecar(&sidecar_dir)
        .expect_err("a sidecar built for qwen35 must not attach to a gemma4 trunk");
    let msg = err.to_string();
    assert!(
        msg.contains("qwen35"),
        "error does not name the sidecar's family: {msg}"
    );
    assert!(
        msg.contains("gemma4"),
        "error does not name the trunk's family: {msg}"
    );
    assert!(!runner.has_vision_tower());

    let _ = std::fs::remove_dir_all(&trunk_dir);
    let _ = std::fs::remove_dir_all(&sidecar_dir);
}

/// `attach_vision_sidecar` refuses a sidecar whose declared hidden size
/// disagrees with the trunk's, with the FAMILY matching so the family check
/// above cannot be what is firing.
#[test]
fn attach_vision_sidecar_refuses_a_hidden_size_mismatch() {
    let trunk_dir = temp_dir("hidden-trunk");
    let trunk_arch =
        build_synthetic_qwen_gdn_dense_install(&trunk_dir, VOCAB, LAYERS, "qwen35-trunk-hidden")
            .expect("a text-only dense install builds");
    let trunk_hidden = trunk_arch.hidden_size;

    let sidecar_dir = temp_dir("hidden-sidecar");
    build_synthetic_vision_sidecar(&sidecar_dir, "qwen35-sidecar-hidden")
        .expect("a standalone sidecar builds");
    rewrite_sidecar_hidden_size(&sidecar_dir, trunk_hidden + 1);

    let mut runner = open(&trunk_dir, trunk_arch);
    let err = runner
        .attach_vision_sidecar(&sidecar_dir)
        .expect_err("a sidecar pairing with a different hidden size must not attach");
    let msg = err.to_string();
    assert!(
        msg.contains(&(trunk_hidden + 1).to_string()),
        "error does not name the sidecar's hidden size: {msg}"
    );
    assert!(
        msg.contains(&trunk_hidden.to_string()),
        "error does not name the trunk's hidden size: {msg}"
    );
    assert!(!runner.has_vision_tower());

    let _ = std::fs::remove_dir_all(&trunk_dir);
    let _ = std::fs::remove_dir_all(&sidecar_dir);
}

/// `attach_vision_sidecar` refuses when the trunk's own install already
/// carries a vision tower -- attaching a second one would be two towers for
/// one session.
#[test]
fn attach_vision_sidecar_refuses_when_the_trunk_already_has_its_own_tower() {
    let combined_dir = temp_dir("already-combined");
    let combined_arch = build_synthetic_qwen_gdn_dense_install_with_vision_streamed(
        &combined_dir,
        VOCAB,
        LAYERS,
        "qwen35-already-combined",
        BITS,
    )
    .expect("the streamed writer builds a dense install with a vision tower");

    let sidecar_dir = temp_dir("already-sidecar");
    build_synthetic_vision_sidecar(&sidecar_dir, "qwen35-sidecar-already")
        .expect("a standalone sidecar builds");

    let mut runner = open(&combined_dir, combined_arch);
    assert!(runner.has_vision_tower());

    let err = runner
        .attach_vision_sidecar(&sidecar_dir)
        .expect_err("a combined install must refuse a second, sidecar tower");
    assert!(
        err.to_string().contains("already declares a vision tower"),
        "unexpected error: {err}"
    );

    let _ = std::fs::remove_dir_all(&combined_dir);
    let _ = std::fs::remove_dir_all(&sidecar_dir);
}

/// `attach_vision_sidecar` refuses a second attach, both before any image
/// has been processed and after the first sidecar's tower has already
/// opened.
#[test]
fn attach_vision_sidecar_refuses_a_double_attach_before_and_after_the_tower_opens() {
    let trunk_dir = temp_dir("double-trunk");
    let trunk_arch =
        build_synthetic_qwen_gdn_dense_install(&trunk_dir, VOCAB, LAYERS, "qwen35-trunk-double")
            .expect("a text-only dense install builds");

    let sidecar_a = temp_dir("double-sidecar-a");
    build_synthetic_vision_sidecar(&sidecar_a, "qwen35-sidecar-a")
        .expect("the first standalone sidecar builds");
    let sidecar_b = temp_dir("double-sidecar-b");
    build_synthetic_vision_sidecar(&sidecar_b, "qwen35-sidecar-b")
        .expect("the second standalone sidecar builds");

    let mut runner = open(&trunk_dir, trunk_arch);
    runner
        .attach_vision_sidecar(&sidecar_a)
        .expect("the first attach succeeds");

    // Before any image: the tower has not opened yet, but the attach is
    // already recorded and must refuse a second one.
    let err_before = runner
        .attach_vision_sidecar(&sidecar_b)
        .expect_err("a second attach before the first image must still be refused");
    assert!(
        err_before.to_string().contains("already attached"),
        "unexpected error: {err_before}"
    );

    // Run an image, forcing the lazy open, then try again.
    let p = params();
    let img = image(&p);
    runner
        .encode_image(&img, &p)
        .expect("the first sidecar's tower runs");
    let err_after = runner
        .attach_vision_sidecar(&sidecar_b)
        .expect_err("a second attach after the tower has opened must also be refused");
    assert!(
        err_after.to_string().contains("already attached"),
        "unexpected error: {err_after}"
    );
    assert_eq!(runner.vision_dir(), sidecar_a.as_path());

    let _ = std::fs::remove_dir_all(&trunk_dir);
    let _ = std::fs::remove_dir_all(&sidecar_a);
    let _ = std::fs::remove_dir_all(&sidecar_b);
}

/// `RealForwardRunner::open` pointed directly at a sidecar directory refuses
/// by name rather than failing deep inside family-state construction with a
/// missing-trunk-tensor error.
#[test]
fn opening_a_sidecar_directory_as_a_model_is_refused_by_name() {
    let sidecar_dir = temp_dir("open-directly");
    build_synthetic_vision_sidecar(&sidecar_dir, "qwen35-sidecar-open-directly")
        .expect("a standalone sidecar builds");

    let arch = turbospark_repack::peek_manifest_arch(&sidecar_dir)
        .expect("the sidecar's own manifest peeks fine");
    let msg = match RealForwardRunner::open(&sidecar_dir, arch) {
        Ok(_) => panic!("opening a sidecar directory as a model must be refused"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("vision sidecar") && msg.contains("attach_vision_sidecar"),
        "unexpected error: {msg}"
    );

    let _ = std::fs::remove_dir_all(&sidecar_dir);
}

/// `release_vision_tower` frees the OPEN resources without forgetting the
/// sidecar ATTACHMENT (vision memory sidecar, Part C).
///
/// Built on a TEXT-ONLY trunk plus a standalone sidecar, deliberately, rather
/// than on a combined install: a text-only trunk has no tower of its own to
/// fall through to, so if release wrongly cleared `vision_sidecar_dir` too,
/// the second `encode_image` below would have nothing to silently produce a
/// plausible-but-wrong answer from -- it would refuse outright instead. A
/// combined-install fixture could not tell "released cleanly and reopened
/// the sidecar" apart from "forgot the sidecar and silently ran the trunk's
/// own tower instead", because both would pass a byte-identity check
/// trivially in the second case (`vision_is_sidecar`'s own module-doc
/// precedent, applied one feature over).
#[test]
fn release_vision_tower_frees_resources_but_preserves_a_sidecar_attachment() {
    let trunk_dir = temp_dir("release-trunk");
    let trunk_arch =
        build_synthetic_qwen_gdn_dense_install(&trunk_dir, VOCAB, LAYERS, "qwen35-trunk-release")
            .expect("a text-only dense install builds");

    let sidecar_dir = temp_dir("release-sidecar");
    build_synthetic_vision_sidecar(&sidecar_dir, "qwen35-sidecar-release")
        .expect("a standalone sidecar builds");

    let mut runner = open(&trunk_dir, trunk_arch);
    runner
        .attach_vision_sidecar(&sidecar_dir)
        .expect("a compatible sidecar attaches");

    let p = params();
    let img = image(&p);

    let first = runner
        .encode_image(&img, &p)
        .expect("the sidecar's tower runs for the first image");
    assert!(
        runner.vision_slot_bytes().is_some(),
        "the tower must report itself open after the first image"
    );
    assert_eq!(runner.vision_is_sidecar(), Some(true));

    runner.release_vision_tower();

    // Releasing frees the open resources and nothing else: the install still
    // declares the capability and the sidecar directory is still the one a
    // re-open should read from.
    assert!(
        runner.has_vision_tower(),
        "release must not un-declare the vision capability (arch.vision.is_active())"
    );
    assert_eq!(
        runner.vision_dir(),
        sidecar_dir.as_path(),
        "release must not forget which sidecar is attached"
    );
    assert!(
        runner.vision_slot_bytes().is_none(),
        "release must free the open tower's resources"
    );
    assert_eq!(
        runner.vision_is_sidecar(),
        None,
        "no tower is open right after release"
    );

    // A second image through the SAME runner must reopen from the sidecar and
    // reproduce byte-identical output, proving re-open rebuilds the exact
    // same tower rather than a subtly different one.
    let second = runner
        .encode_image(&img, &p)
        .expect("a second image reopens the sidecar's tower");
    assert_eq!(
        first.rows, second.rows,
        "re-opening after release must reproduce byte-identical vision output"
    );
    assert_eq!(first.merged_tokens, second.merged_tokens);
    assert_eq!(first.out_hidden, second.out_hidden);
    assert_eq!(
        runner.vision_is_sidecar(),
        Some(true),
        "the sidecar attachment must survive release and reopen"
    );

    let _ = std::fs::remove_dir_all(&trunk_dir);
    let _ = std::fs::remove_dir_all(&sidecar_dir);
}

/// Attaching a sidecar must not perturb a text-only run: the same trunk with
/// and without an attached (but never imaged) sidecar must produce
/// byte-identical text output. The module doc's lazy-open reasoning implies
/// this; nothing before this file stated it directly for the sidecar path.
#[test]
fn attaching_a_sidecar_does_not_perturb_a_text_only_generation() {
    let trunk_dir = temp_dir("noop-trunk");
    let trunk_arch =
        build_synthetic_qwen_gdn_dense_install(&trunk_dir, VOCAB, LAYERS, "qwen35-trunk-noop")
            .expect("a text-only dense install builds");

    let sidecar_dir = temp_dir("noop-sidecar");
    build_synthetic_vision_sidecar(&sidecar_dir, "qwen35-sidecar-noop")
        .expect("a standalone sidecar builds");

    let mut plain = open(&trunk_dir, trunk_arch.clone());
    let mut attached = open(&trunk_dir, trunk_arch);
    attached
        .attach_vision_sidecar(&sidecar_dir)
        .expect("a compatible sidecar attaches");

    // A handful of forward passes through the plain produce path (no image,
    // no injection map set) must read identically whether or not a sidecar
    // is attached -- the tower never opens because no image is processed.
    assert_eq!(plain.vocab_size(), attached.vocab_size());
    let vocab = plain.vocab_size();
    let prompt: Vec<i32> = vec![3, 7, 11, 13, 17, 19, 23, 29];
    for (position, &token) in prompt.iter().enumerate() {
        let mut plain_logits = vec![half::f16::from_f32(0.0); vocab];
        let mut attached_logits = vec![half::f16::from_f32(0.0); vocab];
        turbospark_runtime::LogitProducer::produce(&mut plain, token, position, &mut plain_logits)
            .expect("plain produce");
        turbospark_runtime::LogitProducer::produce(
            &mut attached,
            token,
            position,
            &mut attached_logits,
        )
        .expect("attached produce");
        assert_eq!(
            plain_logits, attached_logits,
            "attaching a sidecar changed text-only decode output at token {token}"
        );
    }
    assert_eq!(
        attached.vision_is_sidecar(),
        None,
        "no image was ever processed"
    );

    let _ = std::fs::remove_dir_all(&trunk_dir);
    let _ = std::fs::remove_dir_all(&sidecar_dir);
}
