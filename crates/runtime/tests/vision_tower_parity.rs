//! This port's vision tower against mlx-vlm, on the SAME checkpoint's bytes
//! and the SAME patch rows (ROADMAP M-V4, stage 2).
//!
//! `vision_tower_synthetic.rs` proves the tower composes what
//! `compute::vision` says it should. It cannot prove the composition is the
//! RIGHT one: its weights are untrained and its reference is this repo's own,
//! so a convention both sides share would agree there and disagree with the
//! model. This is the only instrument that can see that class of error, and
//! the conventions it reaches are the ones no fixture can -- the merge-window
//! patch order, the `(T, P_h, P_w, C)` row layout the repack copies verbatim,
//! the 2-D rope's height-then-width row, the endpoint-preserving position
//! interpolation, and the merger's pre-shuffle norm.
//!
//! # Four stages, not one
//!
//! A gap that reads 0.99 at the merger localizes immediately when the patch
//! embedding is exact and the last block is not. Bisecting it any other way
//! costs a run per stage.
//!
//! # It replays the reference's OWN patch rows
//!
//! The dump carries the rows it fed the tower and this test feeds those,
//! rather than preprocessing the image again. So a preprocessing difference
//! cannot present as a kernel gap -- the same discipline `logit_dump` plus
//! `kld_*.py` use for the text path, where the token IDS are replayed and
//! never the prose. `crates/vision-io`'s own golden fixtures are what hold
//! the preprocessing to the reference; this holds the TOWER.
//!
//! # Setup
//!
//! ```sh
//! # 1. The tower weights, at the revision the install was streamed from.
//! python3 scripts/fetch_vision_tower.py mlx-community/Qwen3.8-27B-4bit \
//!   ~/models/vision-probe-qwen38 3e6447f082e89cc7f0bc6e5441afd38dfce760ff
//! curl -sL "https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/resolve/3e6447f082e89cc7f0bc6e5441afd38dfce760ff/config.json" \
//!   -o ~/models/vision-probe-qwen38/config.json
//!
//! # 2. A deterministic page.
//! uv run --python 3.12 --with pillow scripts/make_vision_test_page.py \
//!   ~/models/vision-probe-qwen38/imgs/page.png --size 1024 1280
//!
//! # 3. The reference's per-stage dump.
//! uv run --python 3.12 --with mlx --with numpy --with pillow --with transformers -- \
//!   python scripts/vision_tower_probe.py --vendor-root ../mlx-v/mlx-vlm \
//!     --config ~/models/vision-probe-qwen38/config.json \
//!     --tower-dir ~/models/vision-probe-qwen38 \
//!     --image ~/models/vision-probe-qwen38/imgs/page.png \
//!     --mode dump --out-dir /tmp/vision-dump
//!
//! # 4. This test.
//! TURBOSPARK_QWEN38_VISION_INSTALL_DIR=~/models/qwen38-27b-vision.gturbo \
//! TURBOSPARK_VISION_DUMP_DIR=/tmp/vision-dump \
//!   cargo test -p turbospark-runtime --test vision_tower_parity --release -- \
//!   --ignored --nocapture
//!
//! # 5. The sidecar-attached arm (ROADMAP P1 item 1), same dump: a TEXT-ONLY
//!    trunk plus the standalone sidecar must clear the same bar.
//! TURBOSPARK_QWEN38_TRUNK_INSTALL_DIR=~/models/qwen38-27b.gturbo \
//! TURBOSPARK_VISION_SIDECAR_DIR=~/.turbospark/models/qwen38-vision-tower.gturbo-vision \
//! TURBOSPARK_VISION_DUMP_DIR=/tmp/vision-dump \
//!   cargo test -p turbospark-runtime --test vision_tower_parity --release -- \
//!   --ignored --nocapture the_sidecar
//! ```
//!
//! **The revision pin in step 1 is load-bearing.** The install was streamed
//! from a pinned revision; fetching the tower at `main` would compare this
//! port running one set of weights against the reference running another,
//! and the two towers have identical shapes, so nothing would fail loudly.
//!
//! # Measured, 2026-08-28, `mlx-community/Qwen3.8-27B-4bit`, 5,120 patches
//!
//! ```text
//! stage                 rms     absmax      worst  worst/max       cosine
//! patch_embed        0.5043     8.9453     0.0078   0.000873   0.99999995
//! block_0            0.7971    11.2500     0.0195   0.001736   0.99999970
//! block_26         116.6840  7904.0000   152.0000   0.019231   0.99999383
//! merger             0.6821   140.8750     1.4375   0.010204   0.99999334
//! ```
//!
//! **The merger reads 0.99999334 against the reference's own FP16-vs-FP32
//! cosine of 0.999993** (`docs/VISION_PHASE0.md` item 3, same tower, same page
//! size). So this port differs from mlx-vlm's FP16 by about what mlx-vlm's
//! FP16 differs from its own FP32 -- at the floor rather than above it, which
//! is the strongest statement available short of bit-identity.
//!
//! # What the bar is worth, per mutation on THIS install
//!
//! | mutation | patch_embed | block_0 | block_26 | merger |
//! |---|---|---|---|---|
//! | none | 0.99999995 | 0.99999970 | 0.99999383 | 0.99999334 |
//! | rope also rotates `v` | 0.99999995 | **0.805** | **0.794** | **0.765** |
//! | merger norms the wide row | 0.99999995 | 0.99999970 | 0.99999383 | **0.600** |
//!
//! Two things that table says and no prose could. A wiring error sits 0.2 to
//! 0.4 BELOW one, against a bar 0.0001 below it, so the factor is thousands
//! rather than the "comfortable margin" a tolerance usually buys. And the
//! four stages LOCALIZE: the rope mutation leaves `patch_embed` untouched
//! because it is upstream, and the merger mutation leaves all three upstream
//! stages bit-identical to a clean run. Neither is inferable from a
//! merger-only comparison.
#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};

use turbospark_runtime::RealForwardRunner;
use turbospark_vision_io::{GridThw, PreprocessParams, PreprocessedImage};

/// The bar every stage is held to: COSINE against the reference's stage.
///
/// # Why cosine, and why an RMS-relative bound is the wrong instrument here
///
/// The first draft gated on `worst_absolute / rms` and failed at block 26 on
/// a correct tower. That metric compares the WORST element's error against a
/// TYPICAL element's magnitude, which is fine on a flat tensor and
/// meaningless on this one: block 26's absmax is 7,904 against an RMS of 117,
/// a factor of 68, because this tower carries outlier features concentrated
/// in a few channels (`docs/VISION_PHASE0.md` item 3 measured exactly that,
/// a 31x step at block 8-9 and an 18x step at 25-26). The worst absolute
/// error naturally lands ON the outlier, where FP16's own quantum near 8,192
/// is 8. So the metric reported 1.30 while the cosine read 0.999994.
///
/// # The floor this bar sits above is MEASURED and is the reference's own
///
/// Phase 0 ran the SAME tower end to end in FP16 and in FP32 and read a
/// merger cosine of 0.999993 at this page size (0.999980 at the 64,516-patch
/// extreme). That is what FP16 accumulation costs inside ONE engine, so no
/// comparison BETWEEN two FP16 engines can resolve below it. This port reads
/// 0.999993 at the merger -- at the floor, not above it.
///
/// The bar is set a decade looser at 0.9999, and what says that is enough is
/// mutation rather than argument: rotating `v` reads 0.765 at the merger and
/// norming the wide row reads 0.600, against a clean 0.99999334. See the
/// module header's table.
const COSINE_BAR: f64 = 0.9999;

fn env_dir(key: &str) -> Option<PathBuf> {
    let raw = std::env::var(key).ok()?;
    if raw.trim().is_empty() {
        return None;
    }
    Some(PathBuf::from(shellexpand(&raw)))
}

/// `~` only. The oracles in `crates/bench` do the same; a full shell
/// expansion here would be a second, worse shell.
fn shellexpand(raw: &str) -> String {
    match raw.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", std::env::var("HOME").unwrap_or_default()),
        None => raw.to_string(),
    }
}

struct Dump {
    dir: PathBuf,
    grid: GridThw,
    shapes: std::collections::HashMap<String, Vec<usize>>,
}

impl Dump {
    fn load(dir: PathBuf) -> Self {
        let header: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("header.json")).expect("header.json"))
                .expect("header.json parses");
        let g = header["grid_thw"].as_array().expect("grid_thw");
        let grid = GridThw::new(
            g[0].as_u64().unwrap() as usize,
            g[1].as_u64().unwrap() as usize,
            g[2].as_u64().unwrap() as usize,
        );
        let mut shapes = std::collections::HashMap::new();
        for (name, meta) in header["stages"].as_object().expect("stages") {
            shapes.insert(
                name.clone(),
                meta["shape"]
                    .as_array()
                    .expect("shape")
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect(),
            );
        }
        Self { dir, grid, shapes }
    }

    fn stage(&self, name: &str) -> Vec<f32> {
        let bytes = std::fs::read(self.dir.join(format!("{name}.bin")))
            .unwrap_or_else(|e| panic!("dump is missing {name}.bin: {e}"));
        let want: usize = self.shapes[name].iter().product();
        assert_eq!(
            bytes.len(),
            want * 4,
            "{name}.bin is {} bytes against a declared shape of {:?}",
            bytes.len(),
            self.shapes[name]
        );
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

/// Preprocessing parameters read off the INSTALL, never restated.
///
/// The tower checks these against its own shape and refuses a mismatch, so
/// taking them from anywhere else would just move where the refusal happens.
/// Takes the VISION CONFIG directly rather than the whole arch so the
/// sidecar-attached arm can read it off the runner POST-ATTACH -- a
/// text-only trunk's peeked arch has an inactive vision field until the
/// attach mutates it.
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

fn to_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect()
}

struct Verdict {
    rms: f64,
    absmax: f64,
    worst: f64,
    cosine: f64,
}

/// Compare one stage, refusing a non-finite input BEFORE measuring anything.
///
/// Finiteness first is not defensive: NaN fails every comparison, so a
/// cosine over a NaN row reads `NaN` and a `worst < bar` test on it reads
/// FALSE -- which is a failure, but one whose message would be about a
/// tolerance rather than about an overflow. The tower runs at 13.8% of
/// FP16's ceiling with a factor of 7.3 in hand (`docs/VISION_PHASE0.md`), so
/// this is a real hazard and not a formality (AGENTS.md Gotchas 59 and 60).
fn compare(name: &str, got: &[f32], want: &[f32]) -> Verdict {
    assert_eq!(
        got.len(),
        want.len(),
        "{name}: this port produced {} values against the reference's {}",
        got.len(),
        want.len()
    );
    assert!(
        got.iter().all(|v| v.is_finite()),
        "{name}: this port produced a non-finite value"
    );
    assert!(
        want.iter().all(|v| v.is_finite()),
        "{name}: the reference dump carries a non-finite value"
    );

    // Accumulated in f64 throughout: at 5,120 patches a stage is ~5.9M
    // values, and an f32 sum of squares over that many terms loses enough of
    // the tail to move every statistic taken off it.
    let sq: f64 = want.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let rms = (sq / want.len() as f64).sqrt();
    let absmax = want.iter().fold(0.0f64, |m, v| m.max(f64::from(*v).abs()));
    let worst = got
        .iter()
        .zip(want)
        .map(|(a, b)| f64::from(a - b).abs())
        .fold(0.0f64, f64::max);
    let dot: f64 = got
        .iter()
        .zip(want)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let na: f64 = got
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt();
    Verdict {
        rms,
        absmax,
        worst,
        cosine: dot / (na * sq.sqrt()),
    }
}

/// The encode-plus-compare body both arms share, so the combined-install
/// comparison and the sidecar-attached one cannot drift apart into two
/// instruments that agree with mlx-vlm in different ways.
///
/// One helper rather than four per-stage cases, because all four stages come
/// out of ONE tower run: splitting them would re-run a 27-block forward pass
/// per stage for no added coverage, and the localization the four points buy
/// is in the REPORT rather than in which case reddens.
fn compare_stages_against_dump(
    runner: &mut RealForwardRunner,
    params: &PreprocessParams,
    depth: usize,
    dump: &Dump,
) {
    let patch_rows = dump.stage("patch_rows");
    let seq = dump.grid.patches();
    assert_eq!(
        patch_rows.len(),
        seq * params.patch_dim(),
        "the dump's patch rows do not match its own grid"
    );
    let image = PreprocessedImage {
        merged_tokens: dump.grid.merged_tokens(params.merge_size),
        patch_rows,
        grid: dump.grid,
        resized: (0, 0),
    };

    println!(
        "grid {:?}, {seq} patches, {} merged tokens, depth {depth}",
        (dump.grid.t, dump.grid.h, dump.grid.w),
        image.merged_tokens
    );

    let (embedding, stages) = runner
        .encode_image_with_stages(&image, params)
        .expect("encode");

    let last = format!("block_{}", depth - 1);
    let mut arms: Vec<(String, Vec<f32>)> = vec![
        ("patch_embed".to_string(), to_f32(&stages.patch_embed)),
        ("block_0".to_string(), to_f32(&stages.block_first)),
        (last, to_f32(&stages.block_last)),
        ("merger".to_string(), to_f32(&embedding.rows)),
    ];
    // The deepstack mergers, when the tower has them (`qwen3_vl`): compared
    // at the SAME bar, because a second injection seam that read the wrong
    // merger or the wrong block's rows is exactly as fluent as a wrong
    // main merger.
    for (k, ds) in stages.deepstack.iter().enumerate() {
        arms.push((format!("deepstack_merger_{k}"), to_f32(ds)));
    }

    // `worst/absmax` is REPORTED and not gated. It is the honest companion to
    // the cosine on a tensor with outlier features: the worst absolute error
    // lands on the largest element, so measuring it against that element's
    // own scale says what fraction of the biggest number is in dispute, where
    // measuring it against the RMS says how the biggest error compares to a
    // typical value and answers a question nobody asked.
    println!(
        "\n{:14} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "stage", "rms", "absmax", "worst", "worst/max", "cosine"
    );
    let mut failures = Vec::new();
    for (name, got) in &arms {
        let want = dump.stage(name);
        let v = compare(name, got, &want);
        println!(
            "{name:14} {:>10.4} {:>10.4} {:>10.4} {:>10.6} {:>12.8}",
            v.rms,
            v.absmax,
            v.worst,
            v.worst / v.absmax,
            v.cosine
        );
        // `< BAR` rather than `!(>= BAR)`: clippy objects to the negated
        // form on a partially ordered type, and a NaN cosine cannot reach
        // here anyway -- `compare` refuses a non-finite input before
        // computing one.
        if v.cosine < COSINE_BAR {
            failures.push(format!("{name}: cosine {} is below {COSINE_BAR}", v.cosine));
        }
    }
    assert!(
        failures.is_empty(),
        "stages disagree with mlx-vlm:\n  {}",
        failures.join("\n  ")
    );
}

/// The whole gate, on the COMBINED install.
#[test]
#[ignore = "needs a real vision install (TURBOSPARK_QWEN38_VISION_INSTALL_DIR) and an \
            mlx-vlm dump (TURBOSPARK_VISION_DUMP_DIR); see this file's header"]
fn the_tower_agrees_with_mlx_vlm_at_every_stage() {
    let Some(install) = env_dir("TURBOSPARK_QWEN38_VISION_INSTALL_DIR") else {
        eprintln!("SKIP: set TURBOSPARK_QWEN38_VISION_INSTALL_DIR to the vision install");
        return;
    };
    let Some(dump_dir) = env_dir("TURBOSPARK_VISION_DUMP_DIR") else {
        eprintln!("SKIP: set TURBOSPARK_VISION_DUMP_DIR to `--mode dump`'s output");
        return;
    };

    let dump = Dump::load(dump_dir);
    let arch =
        turbospark_repack::peek_manifest_arch(Path::new(&install)).expect("install manifest");
    assert!(
        arch.vision.is_active(),
        "this install declares no vision tower; point the var at the one written by \
         `repacks_the_real_qwen38_27b_checkpoint_with_its_vision_tower`"
    );
    let params = params_from(&arch.vision);
    let depth = arch.vision.depth as usize;

    let mut runner = RealForwardRunner::open(Path::new(&install), arch).expect("open install");
    compare_stages_against_dump(&mut runner, &params, depth, &dump);
}

/// The same four-stage gate THROUGH A SIDECAR-ATTACHED TRUNK (ROADMAP P1
/// item 1): a TEXT-ONLY trunk plus the standalone sidecar
/// (`~/.turbospark/models/qwen38-vision-tower.gturbo-vision`) must agree
/// with mlx-vlm at the same bar, which is the claim the CLI's byte-identity
/// A/B made indirectly -- same bytes as the combined install -- measured here
/// against the reference instrument itself.
///
/// # What this arm can see that the combined arm cannot
///
/// The sidecar's tower weights are read through `VisionTower::
/// open_with_sidecar` -- a SEPARATE resident-index read, a separate mmap, a
/// separate FP16 dtype backstop -- so a sidecar that bound the wrong
/// resident weights or mapped the wrong directory passes the synthetic
/// byte-identity fixture (same tower bytes by construction) and fails HERE,
/// against numbers that did not come from this port.
///
/// # Setup
///
/// The trunk is the text-only `~/models/qwen38-27b.gturbo`; the sidecar is
/// the standalone install above, streamed from the same pinned tower
/// revision the dump was (`docs/VISION.md`'s sidecar section). The dump is
/// the SAME `TURBOSPARK_VISION_DUMP_DIR` the combined arm uses.
#[test]
#[ignore = "needs a real text-only trunk (TURBOSPARK_QWEN38_TRUNK_INSTALL_DIR), the standalone \
            vision sidecar (TURBOSPARK_VISION_SIDECAR_DIR), and an mlx-vlm dump \
            (TURBOSPARK_VISION_DUMP_DIR); see this file's header"]
fn the_sidecar_attached_tower_agrees_with_mlx_vlm_at_every_stage() {
    let Some(trunk) = env_dir("TURBOSPARK_QWEN38_TRUNK_INSTALL_DIR") else {
        eprintln!("SKIP: set TURBOSPARK_QWEN38_TRUNK_INSTALL_DIR to the text-only trunk");
        return;
    };
    let Some(sidecar) = env_dir("TURBOSPARK_VISION_SIDECAR_DIR") else {
        eprintln!("SKIP: set TURBOSPARK_VISION_SIDECAR_DIR to the standalone sidecar");
        return;
    };
    let Some(dump_dir) = env_dir("TURBOSPARK_VISION_DUMP_DIR") else {
        eprintln!("SKIP: set TURBOSPARK_VISION_DUMP_DIR to `--mode dump`'s output");
        return;
    };

    let dump = Dump::load(dump_dir);
    let arch = turbospark_repack::peek_manifest_arch(Path::new(&trunk)).expect("trunk manifest");
    // ASSERT THE FIXTURE DISCRIMINATES: a trunk that already carries its own
    // tower would make this arm re-measure the combined install's path and
    // prove nothing about the sidecar's.
    assert!(
        !arch.vision.is_active(),
        "the trunk must be TEXT-ONLY; point the var at qwen38-27b.gturbo, not the \
         combined vision install"
    );

    let mut runner = RealForwardRunner::open(Path::new(&trunk), arch).expect("open trunk");
    runner
        .attach_vision_sidecar(Path::new(&sidecar))
        .expect("the sidecar pairs with this trunk's family and hidden size");
    assert!(
        runner.has_vision_tower(),
        "the attach must activate the vision capability before any comparison runs"
    );
    // The vision config now on the runner IS the sidecar's: params and depth
    // are read POST-ATTACH, which is the whole difference from the combined
    // arm's peeked arch.
    let params = params_from(runner.vision_config());
    let depth = runner.vision_config().depth as usize;

    compare_stages_against_dump(&mut runner, &params, depth, &dump);

    // Engagement, asserted after the comparison so a failure here reads as
    // "the wrong tower answered" rather than as a preamble: a text-only
    // trunk has no tower of its own to fall through to, but the flag is
    // what a future fixture reusing this shape should not have to re-derive.
    assert_eq!(
        runner.vision_is_sidecar(),
        Some(true),
        "the stages just compared must have come from the SIDECAR's tower"
    );
}
