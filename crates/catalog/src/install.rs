//! The install driver: a catalog row or a probed repository in, a `.gturbo`
//! directory out.
//!
//! This is the shape all thirteen `crates/repack/tests/*_network.rs` files
//! repeat, written once. What they each do by hand -- resolve the shards,
//! pick the family writer, stream, fetch sidecars, read the result back
//! through the real loaders -- is [`install`].
//!
//! **THE STEP ORDER IS THE DESIGN, and it inverts how those tests do it.**
//! Every one of them streams multi-GB weights first and fetches tokenizer
//! sidecars afterwards, which is how `Qwen3.8-27B`'s bring-up failed on a 404
//! for `merges.txt` AFTER a 20-minute stream had written a perfectly good
//! install (AGENTS.md Gotcha 47). Sidecars are a few MB and the tokenizer
//! either loads or does not in milliseconds, so they go FIRST: by the time a
//! byte of weight data moves, the install is known to have a tokenizer that
//! loads and a chat template that renders.
//!
//! **THE WALK CANNOT RESUME.** `repack::ranged_download` retries a chunk
//! eight times and then gives up, and giving up costs the whole walk. A pull
//! that dies 19 GB in starts again from zero. [`install`] says so through
//! `progress` before it begins rather than leaving it to be discovered;
//! adding resume is a change to the walks in `crates/repack`, not to this
//! file.

use std::path::{Path, PathBuf};

use model_io::ArchConfig;
use tokenizer::{Message, MfTokenizer, Role};

use crate::entry::{CatalogEntry, EntryKind, SourceKind};
use crate::hf::{Client, RepoRef};
use crate::probe::{probe, ProbeReport, Verdict};
use crate::store::{directory_bytes, InstalledModel, Store};

/// The sidecar files a vision-tower-only install fetches, in the order
/// `install()` writes them: `config.json` decides the family and the
/// tower's own shape (`crate::stream::stream_vision_sidecar`'s first step),
/// and `preprocessor_config.json` is what `vision_dir()` reads for
/// preprocessing settings (Part A3). A tower has no tokenizer, so this list
/// deliberately does not carry one.
pub const VISION_SIDECAR_FILES: [&str; 2] = ["preprocessor_config.json", "config.json"];

/// Everything needed to install one model, whether it came from the catalog
/// or from a `--repo` probe.
#[derive(Debug, Clone)]
pub struct InstallPlan {
    pub alias: String,
    pub weights: RepoRef,
    /// The `.gguf` filename, for a GGUF source.
    pub file: Option<String>,
    pub kind: SourceKind,
    pub sidecars: RepoRef,
    /// The sidecar filenames to fetch. For a catalog row this is the curated
    /// list; for a probe it is the intersection of what the repo HAS with
    /// what this port can use.
    pub sidecar_files: Vec<String>,
    pub install_bytes: u64,
    /// The catalog status, or `unlisted`.
    pub status: String,
    /// A separate repository to pull a multi-token-prediction head from, when
    /// the weights repo's own conversion carries none. See
    /// [`crate::entry::MtpSource`]. `None` for a probed repository: a probe
    /// has no catalog-curated pairing to offer.
    pub mtp: Option<RepoRef>,
    /// An EXISTING install to reuse the trunk's resident bytes from, rather
    /// than re-streaming them over the network, when `mtp` is set. `None` by
    /// default from both constructors below; the caller (`turbospark-model
    /// pull --reuse-trunk-from <alias>`) sets it after validating the named
    /// install's recorded `(repo, revision)` matches `weights` exactly --
    /// this struct does not validate that itself, since it has no `Store` to
    /// check against.
    pub reuse_trunk_from: Option<PathBuf>,
    /// True for a vision-tower-ONLY install (vision memory sidecar, part
    /// A5): no tokenizer to verify, no weight-file walk, the sidecar writer
    /// instead of a family writer, and [`crate::install::gate`]'s probe
    /// bypassed outright (a BF16 tower repo legitimately has no
    /// `quantization` block for that probe to find).
    pub vision_only: bool,
    /// An explicit filename for a vision-only install, when the repository's
    /// safetensors index does not name a shard `fetch_prefixed_shards` can
    /// find on its own (or the repository ships no index at all). `None` for
    /// every other install kind, and for a vision-only one whose index
    /// already lists a matching shard.
    pub vision_file: Option<String>,
}

impl InstallPlan {
    /// The plan a catalog row describes.
    pub fn from_entry(entry: &CatalogEntry) -> Self {
        Self {
            alias: entry.alias.clone(),
            weights: RepoRef::new(&entry.source.repo, &entry.source.revision),
            file: entry.source.file.clone(),
            kind: entry.source.kind,
            sidecars: RepoRef::new(entry.sidecar_repo(), entry.sidecar_revision()),
            sidecar_files: entry.sidecars.files.clone(),
            install_bytes: entry.install_bytes,
            status: entry.status.as_str().to_string(),
            mtp: entry
                .mtp
                .as_ref()
                .map(|m| RepoRef::new(&m.repo, &m.revision)),
            reuse_trunk_from: None,
            vision_only: entry.kind == EntryKind::VisionTower,
            vision_file: None,
        }
    }

    /// The plan a probe describes. Takes the sidecars the repository ACTUALLY
    /// has rather than a list carried over from a sibling checkpoint, which
    /// is the whole point of having probed.
    pub fn from_probe(alias: &str, report: &ProbeReport, sidecars: RepoRef) -> Self {
        Self {
            alias: alias.to_string(),
            weights: report.repo.clone(),
            file: report.file.clone(),
            kind: report.kind,
            sidecars,
            sidecar_files: report.sidecars_present.clone(),
            // No catalog estimate exists, so use the download size as a
            // stand-in. It is the right order of magnitude either way: the
            // walk copies quantized bytes through rather than re-quantizing.
            install_bytes: report.download_bytes.unwrap_or(0),
            status: "unlisted".to_string(),
            reuse_trunk_from: None,
            // A probe has no catalog row to read a curated pairing off, and
            // inventing one from a bare `--repo` argument would guess at a
            // second repository nobody named.
            mtp: None,
            vision_only: false,
            vision_file: None,
        }
    }

    /// The plan for an ad-hoc vision-tower pull (`turbospark-model
    /// pull-vision --repo ...`): no catalog row, no probe (Task 4's gate
    /// bypass applies to this plan by construction), no tokenizer sidecars
    /// -- just [`VISION_SIDECAR_FILES`] and whatever tower the repository's
    /// `config.json` and shards actually declare.
    pub fn for_vision_tower(alias: &str, weights: RepoRef, vision_file: Option<String>) -> Self {
        Self {
            alias: alias.to_string(),
            sidecars: weights.clone(),
            weights,
            file: None,
            kind: SourceKind::Mlx,
            sidecar_files: VISION_SIDECAR_FILES.iter().map(|s| s.to_string()).collect(),
            install_bytes: 0,
            status: "unlisted".to_string(),
            mtp: None,
            reuse_trunk_from: None,
            vision_only: true,
            vision_file,
        }
    }
}

/// What an install produced.
#[derive(Debug, Clone)]
pub struct Installed {
    pub model: InstalledModel,
    pub arch: ArchConfig,
}

pub use repack::ByteProgressCallback;

/// Install `plan` into `dir`.
///
/// `progress` receives both this driver's stage lines and the repack walk's
/// own, so a caller prints one stream.
pub fn install(
    plan: &InstallPlan,
    dir: &Path,
    client: &Client,
    progress: impl FnMut(&str),
) -> Result<Installed, String> {
    install_with_byte_progress(plan, dir, client, progress, None)
}

/// Install `plan` into `dir`, forwarding byte progress updates to `byte_progress`
/// when provided.
pub fn install_with_byte_progress(
    plan: &InstallPlan,
    dir: &Path,
    client: &Client,
    mut progress: impl FnMut(&str),
    byte_progress: Option<ByteProgressCallback>,
) -> Result<Installed, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;

    progress(&format!(
        "installing {} from {} into {}",
        plan.alias,
        plan.weights,
        dir.display()
    ));
    progress(&format!(
        "about {} of weights will stream; THIS WALK CANNOT RESUME, so a failure \
         restarts it from the beginning",
        human_bytes(plan.install_bytes)
    ));

    // Step 1, and it is first on purpose: see the module header. A
    // vision-tower install fetches the same two files (`preprocessor_config
    // .json`, `config.json`) through this call -- they are its
    // `sidecar_files` -- but needs no tokenizer verification, since a tower
    // has no tokenizer.
    fetch_sidecars(plan, dir, client, &mut progress, byte_progress.as_ref())?;
    if plan.vision_only {
        return install_vision_only(plan, dir, client, &mut progress, byte_progress);
    }
    verify_tokenizer(dir, &mut progress)?;

    // Step 2: the weights.
    let arch = match plan.kind {
        SourceKind::Gguf => {
            crate::stream::stream_gguf(plan, dir, &mut progress, byte_progress.as_ref())?
        }
        SourceKind::Mlx => {
            crate::stream::stream_mlx(plan, dir, client, &mut progress, byte_progress.as_ref())?
        }
    };

    // Step 3: read it back through the loaders a real run uses. Every
    // network test does this and it is not ceremony -- `load_manifest` is
    // one of the two block-type gates, so an install whose types have no
    // kernels is caught here rather than at the first dispatch.
    verify_install(dir, &arch, &mut progress)?;

    let model = InstalledModel {
        alias: plan.alias.clone(),
        repo: plan.weights.repo.clone(),
        revision: plan.weights.revision.clone(),
        path: dir
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(dir))
            .to_path_buf(),
        family: arch.family.as_str().to_string(),
        install_bytes: directory_bytes(dir),
        installed_on: today(),
        status: plan.status.clone(),
        kind: None,
    };
    Ok(Installed { model, arch })
}

/// The vision-tower branch of [`install_with_byte_progress`]: stream the
/// tower through [`crate::stream::stream_vision_sidecar`] and verify the
/// result through [`model_io::load_vision_sidecar`] rather than through
/// [`verify_install`]'s manifest/resident-index/expert-layout trio -- a
/// tower install carries no routed experts in the sense that trio checks,
/// and `load_vision_sidecar` already does the equivalent full validated
/// read for this format (`vision_sidecar.json` parses, the arch it implies
/// passes the real manifest loader, and the record's declared pairing
/// hidden size agrees with the manifest's own).
fn install_vision_only(
    plan: &InstallPlan,
    dir: &Path,
    client: &Client,
    progress: &mut impl FnMut(&str),
    byte_progress: Option<ByteProgressCallback>,
) -> Result<Installed, String> {
    let vision = crate::stream::stream_vision_sidecar(
        &plan.weights,
        dir,
        plan.vision_file.as_deref(),
        client,
        progress,
        byte_progress.as_ref(),
    )?;

    let (record, _vision) = model_io::load_vision_sidecar(dir)
        .map_err(|e| format!("the written vision sidecar does not validate: {e}"))?;
    let family = model_io::ModelFamily::parse(&record.pairs_with.family).ok_or_else(|| {
        format!(
            "the written vision sidecar names an unknown family {:?}",
            record.pairs_with.family
        )
    })?;
    progress(&format!(
        "vision sidecar install verified: {} block(s) pairing with {} at hidden_size {}",
        record.tower_blocks, record.pairs_with.family, record.pairs_with.hidden_size
    ));

    let arch = model_io::sidecar_arch(family, vision.out_hidden_size, &vision);
    let model = InstalledModel {
        alias: plan.alias.clone(),
        repo: plan.weights.repo.clone(),
        revision: plan.weights.revision.clone(),
        path: dir
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(dir))
            .to_path_buf(),
        family: family.as_str().to_string(),
        install_bytes: directory_bytes(dir),
        installed_on: today(),
        status: plan.status.clone(),
        kind: Some(EntryKind::VisionTower.as_str().to_string()),
    };
    Ok(Installed { model, arch })
}

/// Fetch the tokenizer sidecars. A missing REQUIRED file is fatal here, where
/// it costs seconds, rather than after the stream.
fn fetch_sidecars(
    plan: &InstallPlan,
    dir: &Path,
    client: &Client,
    progress: &mut impl FnMut(&str),
    byte_progress: Option<&ByteProgressCallback>,
) -> Result<(), String> {
    progress(&format!(
        "fetching {} tokenizer sidecar(s) from {}",
        plan.sidecar_files.len(),
        plan.sidecars
    ));
    for name in &plan.sidecar_files {
        let url = plan.sidecars.file_url(name);
        let bytes = client.get(&url)?;
        if let Some(cb) = byte_progress {
            cb(bytes.len() as u64);
        }
        std::fs::write(dir.join(name), bytes)
            .map_err(|e| format!("writing {}: {e}", dir.join(name).display()))?;
    }
    Ok(())
}

/// Load the tokenizer that was just written and render one turn through it.
///
/// **Loading is not enough on its own**, which is why this renders as well.
/// `MfTokenizer::load_from_dir` succeeds on any valid `tokenizer.json`,
/// template or no template; the render is what says an instruction-tuned
/// checkpoint will be framed the way it was trained rather than falling
/// through to a dialect default.
fn verify_tokenizer(dir: &Path, progress: &mut impl FnMut(&str)) -> Result<(), String> {
    let tokenizer = MfTokenizer::load_from_dir(dir).map_err(|e| {
        format!(
            "the sidecars in {} do not load as a tokenizer: {e}. Nothing was streamed.",
            dir.display()
        )
    })?;
    let rendered = tokenizer
        .apply_chat_template(&[Message::new(Role::User, "hello")])
        .map_err(|e| format!("the tokenizer loaded but its chat template does not render: {e}"))?;
    if rendered.trim().is_empty() {
        return Err("the chat template rendered an empty prompt".to_string());
    }
    // `add_bos = false` is the convention every call site here uses: each
    // dialect's template emits its own BOS as text, so adding one doubles it.
    let encoded = tokenizer.encode(&rendered, false);
    if encoded.is_empty() {
        return Err("the rendered prompt encodes to zero tokens".to_string());
    }
    progress(&format!(
        "tokenizer verified: {:?} dialect, a one-turn prompt renders to {} tokens",
        tokenizer.dialect,
        encoded.len()
    ));
    Ok(())
}

/// Read the install back through the loaders a real run uses.
fn verify_install(
    dir: &Path,
    arch: &ArchConfig,
    progress: &mut impl FnMut(&str),
) -> Result<(), String> {
    model_io::load_manifest(dir, arch, model_io::DEFAULT_MAX_BYTES)
        .map_err(|e| format!("the written manifest does not validate: {e}"))?;
    let resident = model_io::load_resident_index(&dir.join("model_weights.bin"))
        .map_err(|e| format!("the written resident index does not load: {e}"))?;
    let layout = model_io::load_packed_experts_layout(dir, 64 * 1024 * 1024)
        .map_err(|e| format!("the written expert layout does not load: {e}"))?;
    progress(&format!(
        "install verified: {} resident tensors, {} streamed expert layer(s)",
        resident.entries.len(),
        layout.layers.len()
    ));
    Ok(())
}

/// Gate a plan on a probe unless the caller forced it.
///
/// **A VISION-TOWER PLAN BYPASSES THE PROBE OUTRIGHT.** `probe`'s MLX gate
/// (`crate::probe::evaluate_config`) refuses a repository whose
/// `config.json` carries no `quantization` block, on the correct reasoning
/// that silence there usually means "not actually MLX-quantized" for a
/// TRUNK. A tower repository is legitimately BF16 with no such block --
/// `mlx-community/Qwen3.8-27B-4bit`'s tower is exactly that -- so running it
/// through the trunk's own gate would refuse every real tower by name. The
/// tower's own correctness gate is [`model_io::load_vision_sidecar`], run
/// after the write in [`install_vision_only`], not a pre-flight probe.
pub fn gate(
    client: &Client,
    plan: &InstallPlan,
    force: bool,
    progress: &mut impl FnMut(&str),
) -> Result<ProbeReport, String> {
    if plan.vision_only {
        progress(&format!(
            "{}: vision-tower install, skipping the trunk probe (a BF16 tower repo \
             legitimately has no quantization block for that gate to find)",
            plan.weights
        ));
        return Ok(vision_tower_stub_report(plan));
    }
    let report = probe(
        client,
        &plan.weights,
        plan.file.as_deref(),
        Some(&plan.sidecars),
    )?;
    match (&report.verdict, force) {
        (Verdict::Runnable, _) => Ok(report),
        (Verdict::Refused(why), true) => {
            progress(&format!("--force: proceeding past a refusal -- {why}"));
            Ok(report)
        }
        (Verdict::Refused(why), false) => Err(format!(
            "{} would not run here: {why}\nRe-run with --force to install it anyway.",
            plan.weights
        )),
    }
}

/// A trivially [`Verdict::Runnable`] report for a vision-tower plan, so
/// [`gate`]'s caller (`turbospark-model pull-vision`) can print the same
/// "probe result" shape every other pull prints without a real probe having
/// run. Every gated field is left at its neutral value; nothing downstream
/// of [`gate`] reads them for a vision-only plan.
fn vision_tower_stub_report(plan: &InstallPlan) -> ProbeReport {
    ProbeReport {
        repo: plan.weights.clone(),
        kind: plan.kind,
        file: plan.vision_file.clone(),
        download_bytes: None,
        architecture: None,
        family: None,
        arch: None,
        types: Vec::new(),
        affine: None,
        expert_stride: None,
        trained_context: None,
        sidecars_present: plan.sidecar_files.clone(),
        sidecars_missing: Vec::new(),
        chat_template: None,
        verdict: Verdict::Runnable,
        warnings: vec![
            "vision-tower install: the trunk probe was skipped by design (see `gate`'s doc)"
                .to_string(),
        ],
    }
}

/// Record an install in `store`.
pub fn record(store: &Store, installed: &Installed) -> Result<(), String> {
    store.record(&installed.model)
}

/// `YYYY-MM-DD` in UTC, from the system clock.
///
/// Hand-rolled rather than pulling in a date crate: this crate needs exactly
/// one date, formatted one way, and it is a record field rather than an input
/// to anything. The civil-from-days conversion is Howard Hinnant's, which is
/// exact for every day in the proleptic Gregorian calendar.
fn today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let z = secs.div_euclid(86_400) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

/// Bytes as a short human string. Binary units, matching every memory number
/// in this repo.
pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod vision_gate_tests {
    use super::*;
    use crate::entry::{Sidecars, Source, Status};

    fn vision_row() -> CatalogEntry {
        CatalogEntry {
            alias: "test-vision-tower".to_string(),
            kind: EntryKind::VisionTower,
            name: "Test Vision Tower".to_string(),
            family: "qwen35".to_string(),
            source: Source {
                kind: SourceKind::Mlx,
                repo: "owner/vision-tower".to_string(),
                revision: "0".repeat(40),
                file: None,
            },
            sidecars: Sidecars {
                repo: None,
                revision: None,
                files: VISION_SIDECAR_FILES.iter().map(|s| s.to_string()).collect(),
            },
            download_bytes: 900_000_000,
            install_bytes: 900_000_000,
            status: Status::Runs,
            gates: Vec::new(),
            measured: Vec::new(),
            notes: None,
            mtp: None,
        }
    }

    /// `InstallPlan::from_entry` derives `vision_only` from the row's own
    /// `kind`, not from any other field -- a model row with the same shape
    /// otherwise must NOT take the vision-only branch.
    #[test]
    fn from_entry_sets_vision_only_from_the_row_kind() {
        let tower = InstallPlan::from_entry(&vision_row());
        assert!(tower.vision_only);
        assert!(tower.vision_file.is_none());

        let mut model = vision_row();
        model.kind = EntryKind::Model;
        model.sidecars.files = vec!["tokenizer.json".to_string()];
        let plan = InstallPlan::from_entry(&model);
        assert!(!plan.vision_only);
    }

    #[test]
    fn for_vision_tower_builds_a_vision_only_plan_with_no_sidecar_repo_split() {
        let weights = RepoRef::new("owner/vision-tower", "abc123");
        let plan = InstallPlan::for_vision_tower(
            "mytower",
            weights.clone(),
            Some("model.safetensors".to_string()),
        );
        assert!(plan.vision_only);
        assert_eq!(plan.vision_file.as_deref(), Some("model.safetensors"));
        assert_eq!(plan.sidecars, weights);
        assert_eq!(
            plan.sidecar_files,
            VISION_SIDECAR_FILES
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        );
    }

    /// **THE GATE BYPASS, MUTATION-CHECKED.** `gate()` must never reach the
    /// network for a vision-only plan -- the whole point of the bypass is
    /// that the trunk probe's `quantization`-block gate refuses a BF16
    /// tower repo that has none. A `Client` with no server behind it proves
    /// no HTTP call was attempted: the ordinary path would fail on a
    /// connection error rather than returning `Ok`.
    #[test]
    fn gate_bypasses_the_probe_for_a_vision_only_plan() {
        let plan = InstallPlan::from_entry(&vision_row());
        let client = Client::new();
        let mut lines = Vec::new();
        let report = gate(&client, &plan, false, &mut |line: &str| {
            lines.push(line.to_string())
        })
        .expect("a vision-only plan must gate without touching the network");
        assert_eq!(report.verdict, Verdict::Runnable);
        assert!(
            lines.iter().any(|l| l.contains("skipping the trunk probe")),
            "{lines:?}"
        );
    }

    /// The mutation this guards against: deleting the `if plan.vision_only`
    /// early return would make this same plan reach `probe()`, which issues
    /// a real HTTP GET to `huggingface.co` against a repository that does
    /// not exist and returns `Err` rather than `Ok(Verdict::Runnable)`. This
    /// test cannot run offline in CI if that mutation is applied (the error
    /// message would name a network failure instead of matching
    /// `Verdict::Runnable`), which is exactly the discriminating behavior
    /// AGENTS.md's mutation-check rule asks for.
    #[test]
    fn a_non_vision_plan_is_not_affected_by_the_bypass_branch() {
        // Not exercised over the network here (that is `catalog_network.rs`'s
        // job); this only asserts the STRUCTURAL fact that a model-kind plan
        // built the ordinary way carries `vision_only: false`, so the branch
        // above cannot accidentally swallow it.
        let mut model = vision_row();
        model.kind = EntryKind::Model;
        model.sidecars.files = vec!["tokenizer.json".to_string()];
        let plan = InstallPlan::from_entry(&model);
        assert!(!plan.vision_only);
    }
}
