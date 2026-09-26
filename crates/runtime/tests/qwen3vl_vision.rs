//! The `qwen3_vl` tower's DEEPSTACK injection against the real 4B install
//! (ROADMAP 9b's vision half).
//!
//! The synthetic fixtures prove the seams compose; they cannot prove the
//! composition is wired to the RIGHT bytes on an install whose weights are
//! trained. This file is the real-artifact half, `vision_tower_parity.rs`'s
//! relationship to `vision_tower_synthetic.rs` one family over, and it pins
//! three things the synthetic fixtures cannot:
//!
//! 1. the tower's deepstack mergers EMIT row sets on the real bytes, with
//!    the declared count and geometry, and the three sets are distinct
//!    functions of the image rather than one row set copied three times;
//! 2. the trunk's per-layer adds are WIRED -- perturbing one deepstack
//!    merger's resident bytes moves the prefill's logits, which no
//!    shape-only assertion can see (the qwen35 embedding lookup's stride
//!    lesson, Gotcha 48's shape, at the injection seam);
//! 3. the llama flow's two embed sites agree -- the chunked driver
//!    (production prefill) and the sequential walk (`produce`) produce the
//!    same last-position logits on the same image prompt, the equivalence
//!    `vision_chunked_synthetic.rs` holds for the qwen flow's two sites.
//!
//! # Setup
//!
//! ```sh
//! # The combined install (bytes-detected deepstack, 2.9 GiB).
//! turbospark-model pull --repo \
//!   "mlx-community/Qwen3-VL-4B-Instruct-4bit@2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b" \
//!   --alias qwen3vl-4b-vision
//!
//! # The deterministic page.
//! uv run --python 3.12 --with pillow scripts/make_vision_test_page.py \
//!   ~/.turbospark/vision-pages/page.png --size 1024 1280
//!
//! TURBOSPARK_QWEN3VL_VISION_INSTALL_DIR=~/.turbospark/models/text/qwen3vl-4b-vision.gturbo \
//! TURBOSPARK_QWEN3VL_VISION_PAGE=~/.turbospark/vision-pages/page.png \
//!   cargo test -p turbospark-runtime --test qwen3vl_vision --release -- \
//!   --ignored --nocapture
//! ```
//!
//! The qwen3vl install needs GPU + Metal, so like every real-install arm
//! this is `#[ignore]`d and opt-in.

use half::f16;
use std::path::{Path, PathBuf};
use turbospark_runtime::{ChunkedPrefillRunner, RealForwardRunner};

use turbospark_vision_io::{
    decode_image_file, mrope_position_triples, preprocess, GridThw, ImageSpan, MropePositions,
    PreprocessParams, VisionSpecialIds,
};

const IMAGE_PAD_ID: i32 = 151_655;
const VISION_START_ID: i32 = 151_652;

fn env_dir(key: &str) -> Option<PathBuf> {
    let raw = std::env::var(key).ok()?;
    let expanded = if let Some(rest) = raw.strip_prefix("~/") {
        PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(rest)
    } else {
        PathBuf::from(raw)
    };
    if expanded.is_dir() {
        Some(expanded)
    } else {
        None
    }
}

fn install_dir() -> Option<PathBuf> {
    env_dir("TURBOSPARK_QWEN3VL_VISION_INSTALL_DIR")
}

fn page_path() -> Option<PathBuf> {
    match std::env::var("TURBOSPARK_QWEN3VL_VISION_PAGE") {
        Ok(raw) if !raw.is_empty() => {
            let p = if let Some(rest) = raw.strip_prefix("~/") {
                PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(rest)
            } else {
                PathBuf::from(raw)
            };
            Some(p)
        }
        _ => None,
    }
}

fn params_from(v: &model_io::VisionConfig) -> PreprocessParams {
    PreprocessParams {
        patch_size: v.patch_size as usize,
        temporal_patch_size: v.temporal_patch_size as usize,
        merge_size: v.spatial_merge_size as usize,
        in_channels: v.in_channels as usize,
        min_pixels: 65_536,
        max_pixels: 16_777_216,
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
        rescale_factor: 1.0 / 255.0,
    }
}

/// Open the combined install and preprocess the page.
fn open_install(dir: &Path) -> RealForwardRunner {
    let arch = turbospark_repack::peek_manifest_arch(dir).expect("install manifest");
    assert!(arch.vision.is_active(), "the install declares no tower");
    assert_eq!(
        arch.vision.deepstack_visual_indexes,
        vec![5, 11, 17],
        "this file pins the 4B tower's deepstack config; a different install needs its \
         own expectations"
    );
    turbospark_runtime::RealForwardRunner::open(dir, arch).expect("open install")
}

fn preprocess_page(arch: &model_io::ArchConfig) -> turbospark_vision_io::PreprocessedImage {
    let page = page_path().unwrap_or_else(|| {
        panic!(
            "set TURBOSPARK_QWEN3VL_VISION_PAGE to the deterministic page \
             (scripts/make_vision_test_page.py --size 1024 1280)"
        )
    });
    let params = params_from(&arch.vision);
    let rgb = decode_image_file(&page).expect("decode the page");
    preprocess(&rgb, &params).expect("preprocess the page")
}

/// The ids of one image prompt: text, the vision start, the pads, text.
fn prompt_ids(merged_tokens: usize) -> Vec<i32> {
    let mut ids = vec![1000, 1001, 1002, VISION_START_ID];
    ids.extend(std::iter::repeat_n(IMAGE_PAD_ID, merged_tokens));
    ids.extend([2000, 2001]);
    ids
}

fn positions_for(ids: &[i32], grid: GridThw, merge: usize) -> MropePositions {
    mrope_position_triples(
        ids,
        &[grid],
        VisionSpecialIds {
            vision_start: VISION_START_ID,
            image_pad: IMAGE_PAD_ID,
        },
        merge,
    )
    .expect("the walk places the one image")
}

/// Walk a whole prompt token by token through `produce` (the sequential
/// site) and return the LAST position's logits.
fn walk(runner: &mut RealForwardRunner, ids: &[i32]) -> Vec<u16> {
    use turbospark_runtime::LogitProducer;
    let vocab = 151_936usize;
    let mut head = vec![f16::from_f32(0.0); vocab];
    for (position, &token) in ids.iter().enumerate() {
        runner
            .produce(token, position, &mut head)
            .expect("produces");
    }
    head.into_iter().map(|v| v.to_bits()).collect()
}

/// Prefill the whole prompt through the chunked driver (the production
/// site) and return its logits.
fn chunked_walk(runner: &mut RealForwardRunner, ids: &[i32]) -> Vec<u16> {
    let vocab = 151_936usize;
    let mut head = vec![f16::from_f32(0.0); vocab];
    let mut done = 0usize;
    while done < ids.len() {
        let take = (ids.len() - done).min(128);
        runner
            .prefill_chunk(&ids[done..done + take], done, &mut head)
            .expect("chunk prefills");
        done += take;
    }
    head.into_iter().map(|v| v.to_bits()).collect()
}

fn digest(logits: &[u16]) -> String {
    let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
    model_io::hash_data(&bytes)[..8].to_string()
}

/// Encode the page and hand the runner the map for `ids`.
fn inject(
    runner: &mut RealForwardRunner,
    arch: &model_io::ArchConfig,
    image: &turbospark_vision_io::PreprocessedImage,
    ids: &[i32],
) -> MropePositions {
    let params = params_from(&arch.vision);
    let embedding = runner.encode_image(image, &params).expect("the tower runs");
    let positions = positions_for(ids, image.grid, params.merge_size);
    runner
        .set_prompt_vision(std::slice::from_ref(&embedding), &positions, ids.len())
        .expect("the map validates");
    positions
}

/// The tower's deepstack mergers emit THREE distinct row sets on the real
/// bytes, each the declared geometry, none of them the main embedding.
///
/// A merger that returned its input's rows, another merger's rows, or a
/// zeroed buffer passes no shape check and reads as a slightly wrong
/// picture; pairwise distinctness is the cheapest assertion that can see
/// all three.
#[test]
#[ignore = "needs the real qwen3vl combined vision install and the deterministic page; \
            see this file's header"]
fn the_deepstack_mergers_emit_three_distinct_row_sets() {
    let Some(install) = install_dir() else {
        eprintln!("SKIP: set TURBOSPARK_QWEN3VL_VISION_INSTALL_DIR");
        return;
    };
    let arch = turbospark_repack::peek_manifest_arch(&install).expect("install manifest");
    let params = params_from(&arch.vision);
    let image = preprocess_page(&arch);

    let mut runner = open_install(&install);
    let embedding = runner
        .encode_image(&image, &params)
        .expect("the tower runs");

    let declared = arch.vision.deepstack_visual_indexes.len();
    assert_eq!(
        embedding.deepstack.len(),
        declared,
        "the tower emits one row set per declared deepstack index"
    );
    let expected = embedding.merged_tokens * embedding.out_hidden;
    assert_eq!(embedding.rows.len(), expected, "main rows are the geometry");
    for (k, ds) in embedding.deepstack.iter().enumerate() {
        assert_eq!(
            ds.rows.len(),
            expected,
            "deepstack row set {k} carries the same merge geometry as the embedding"
        );
        assert!(
            ds.rows.iter().any(|&v| v != 0),
            "deepstack row set {k} is entirely zero, which is an unloaded merger"
        );
    }
    // Every set is a DIFFERENT function of the image. Exact equality between
    // two sets would mean one merger's output is being injected twice.
    for a in 0..declared {
        for b in (a + 1)..declared {
            assert_ne!(
                embedding.deepstack[a].rows, embedding.deepstack[b].rows,
                "deepstack sets {a} and {b} are identical"
            );
        }
        assert_ne!(
            embedding.deepstack[a].rows, embedding.rows,
            "deepstack set {a} is a copy of the main merger's output"
        );
    }
}

/// Perturbing ONE deepstack merger's resident bytes moves the image
/// prompt's prefill logits, in BOTH prefill sites.
///
/// The add is encoded GPU-side between layer kernels, so no shape check
/// reaches it: an add wired to the wrong buffer, the wrong slot, or no
/// buffer at all leaves every row count correct and the logits untouched.
/// The patch is 32 bytes of ONE merger's `linear_fc2`, bitwise-flipped so
/// the perturbation cannot be a zero-fill, on an APFS-cloned copy of the
/// install (the original is never written).
#[test]
#[ignore = "needs the real qwen3vl combined vision install and the deterministic page; \
            see this file's header"]
fn the_deepstack_adds_move_the_trunks_logits() {
    let Some(install) = install_dir() else {
        eprintln!("SKIP: set TURBOSPARK_QWEN3VL_VISION_INSTALL_DIR");
        return;
    };
    let arch = turbospark_repack::peek_manifest_arch(&install).expect("install manifest");
    let image = preprocess_page(&arch);
    let ids = prompt_ids(image.merged_tokens);

    // The unperturbed run, sequential site.
    let mut runner = open_install(&install);
    inject(&mut runner, &arch, &image, &ids);
    let baseline = walk(&mut runner, &ids);
    drop(runner);

    // The clone. `packed_vision/` and `manifest.json` are hard-linked (never
    // patched); the resident bin is COPIED because that is where the
    // deepstack mergers live and what the patch writes.
    let clone = std::env::temp_dir().join(format!("qwen3vl-vision-perturb-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&clone);
    struct CleanupDir<'a>(&'a std::path::Path);
    impl<'a> Drop for CleanupDir<'a> {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(self.0);
        }
    }
    let _cleanup = CleanupDir(&clone);
    std::fs::create_dir_all(clone.join("packed_vision")).expect("clone dir");
    for file in ["manifest.json", "preprocessor_config.json", "config.json"] {
        let src = install.join(file);
        if src.is_file() {
            std::fs::copy(&src, clone.join(file)).expect("copy manifest");
        }
    }
    for entry in std::fs::read_dir(install.join("packed_vision")).expect("packed_vision") {
        let entry = entry.expect("packed_vision entry");
        std::fs::hard_link(
            entry.path(),
            clone.join("packed_vision").join(entry.file_name()),
        )
        .expect("hard-link packed_vision");
    }
    // The dense install still carries an (empty) packed_experts layout the
    // open path loads; hard-link it like packed_vision.
    let _ = std::fs::create_dir_all(clone.join("packed_experts"));
    if let Ok(entries) = std::fs::read_dir(install.join("packed_experts")) {
        for entry in entries.flatten() {
            let _ = std::fs::hard_link(
                entry.path(),
                clone.join("packed_experts").join(entry.file_name()),
            );
        }
    }
    std::fs::copy(
        install.join("model_weights.bin"),
        clone.join("model_weights.bin"),
    )
    .expect("copy the resident bin");

    // Find one deepstack merger tensor and flip 32 bytes of it.
    let index = model_io::load_resident_index(&clone.join("model_weights.bin"))
        .expect("clone resident index");
    let target = "vision.deepstack_merger_list.0.linear_fc2.weight";
    let entry = index
        .entries
        .get(target)
        .unwrap_or_else(|| panic!("the clone's index carries {target}"));
    let patch_at = entry.file_offset + entry.size_bytes / 2;
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(clone.join("model_weights.bin"))
        .expect("open clone for patch");
    {
        use std::io::{Seek, Write};
        let mut file = file;
        file.seek(std::io::SeekFrom::Start(patch_at)).expect("seek");
        let flipped: Vec<u8> = (0u8..32).map(|b| !b).collect();
        file.write_all(&flipped).expect("patch");
    }

    // The perturbed run must disagree.
    let mut runner = open_install(&clone);
    inject(&mut runner, &arch, &image, &ids);
    let perturbed = walk(&mut runner, &ids);
    assert_ne!(
        digest(&baseline),
        digest(&perturbed),
        "flipping 32 bytes of deepstack merger 0's fc2 did not move the logits; the \
         per-layer adds are not wired to the merger's rows"
    );
    drop(runner);

    // And the CHUNKED driver -- the site production prefill actually takes
    // -- must have been wired too, on the same clone.
    let mut runner = open_install(&clone);
    inject(&mut runner, &arch, &image, &ids);
    let perturbed_chunked = chunked_walk(&mut runner, &ids);
    assert_ne!(
        digest(&baseline),
        digest(&perturbed_chunked),
        "the chunked driver's deepstack adds are not wired"
    );
    drop(runner);
    let _ = std::fs::remove_dir_all(&clone);
}

/// The two embed sites agree on an image prompt: the chunked driver's
/// last-position logits are the sequential walk's, bit for bit.
///
/// The qwen flow holds this equivalence through
/// `vision_chunked_synthetic.rs`; the llama flow's sites were built
/// together here and the same rule applies -- a change made at one site and
/// not the other reddens instead of diverging quietly.
#[test]
#[ignore = "needs the real qwen3vl combined vision install and the deterministic page; \
            see this file's header"]
fn the_chunked_driver_and_the_sequential_walk_agree_on_an_image_prompt() {
    let Some(install) = install_dir() else {
        eprintln!("SKIP: set TURBOSPARK_QWEN3VL_VISION_INSTALL_DIR");
        return;
    };
    let arch = turbospark_repack::peek_manifest_arch(&install).expect("install manifest");
    let image = preprocess_page(&arch);
    let ids = prompt_ids(image.merged_tokens);

    let mut runner = open_install(&install);
    inject(&mut runner, &arch, &image, &ids);
    let sequential = walk(&mut runner, &ids);
    drop(runner);

    let mut runner = open_install(&install);
    inject(&mut runner, &arch, &image, &ids);
    let chunked = chunked_walk(&mut runner, &ids);

    assert_eq!(
        digest(&sequential),
        digest(&chunked),
        "the chunked driver and the sequential walk disagree on an image prompt; \
         one site's blit or deepstack add is not the other's"
    );
}

/// A span in the position table lands where the placeholder run is, and the
/// first pad's triple is degenerate or divergent per the reference's own
/// walk. Pins the qwen3vl mRoPE data shape so the dispatch rule above it is
/// read against the real table and not a fixture's.
#[test]
#[ignore = "needs the real qwen3vl combined vision install and the deterministic page; \
            see this file's header"]
fn the_position_walk_places_the_real_page() {
    let Some(install) = install_dir() else {
        eprintln!("SKIP: set TURBOSPARK_QWEN3VL_VISION_INSTALL_DIR");
        return;
    };
    let arch = turbospark_repack::peek_manifest_arch(&install).expect("install manifest");
    let image = preprocess_page(&arch);
    let ids = prompt_ids(image.merged_tokens);
    let positions = positions_for(&ids, image.grid, arch.vision.spatial_merge_size as usize);

    assert_eq!(positions.triples.len(), ids.len());
    assert_eq!(
        positions.spans,
        vec![ImageSpan {
            start: 4,
            len: image.merged_tokens
        }],
        "one span, at the pads, exactly the image's merged-token count"
    );
    // rope_delta is NEGATIVE: the image spends fewer positions than tokens.
    assert!(
        positions.rope_delta < 0,
        "an image must compress the clock, rope_delta = {}",
        positions.rope_delta
    );
    // The LAST text position's triple is (len - 1 + rope_delta) in all three
    // slots -- the degenerate form decode continues from.
    let last = *positions.triples.last().expect("non-empty");
    let p = (ids.len() as i32 - 1) + positions.rope_delta;
    assert_eq!(last, (p, p, p));
}
