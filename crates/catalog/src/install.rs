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

use crate::entry::{CatalogEntry, SourceKind};
use crate::hf::{Client, RepoRef};
use crate::probe::{probe, ProbeReport, Verdict};
use crate::store::{directory_bytes, InstalledModel, Store};

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

    // Step 1, and it is first on purpose: see the module header.
    fetch_sidecars(plan, dir, client, &mut progress, byte_progress.as_ref())?;
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
pub fn gate(
    client: &Client,
    plan: &InstallPlan,
    force: bool,
    progress: &mut impl FnMut(&str),
) -> Result<ProbeReport, String> {
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
