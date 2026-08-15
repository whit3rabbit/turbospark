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

use model_io::{ArchConfig, ModelFamily};
use repack::{Gemma4Shards, HttpRangeSource, RangeSource};
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
        }
    }
}

/// What an install produced.
#[derive(Debug, Clone)]
pub struct Installed {
    pub model: InstalledModel,
    pub arch: ArchConfig,
}

/// Install `plan` into `dir`.
///
/// `progress` receives both this driver's stage lines and the repack walk's
/// own, so a caller prints one stream.
pub fn install(
    plan: &InstallPlan,
    dir: &Path,
    client: &Client,
    mut progress: impl FnMut(&str),
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
    fetch_sidecars(plan, dir, client, &mut progress)?;
    verify_tokenizer(dir, &mut progress)?;

    // Step 2: the weights.
    let arch = match plan.kind {
        SourceKind::Gguf => stream_gguf(plan, dir, &mut progress)?,
        SourceKind::Mlx => stream_mlx(plan, dir, client, &mut progress)?,
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
) -> Result<(), String> {
    progress(&format!(
        "fetching {} tokenizer sidecar(s) from {}",
        plan.sidecar_files.len(),
        plan.sidecars
    ));
    for name in &plan.sidecar_files {
        let url = plan.sidecars.file_url(name);
        let bytes = client.get(&url)?;
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

fn stream_gguf(
    plan: &InstallPlan,
    dir: &Path,
    progress: &mut impl FnMut(&str),
) -> Result<ArchConfig, String> {
    let file = plan
        .file
        .as_deref()
        .ok_or_else(|| "a gguf install needs a filename".to_string())?;
    let source = HttpRangeSource::new(plan.weights.file_url(file));
    let header = repack::fetch_gguf_header(&source)
        .map_err(|e| format!("reading the GGUF header of {file}: {e}"))?;
    let model_id = plan.weights.repo.clone();
    repack::write_gguf_install_streamed(dir, &header, &source, &model_id, |stage| {
        progress(&format!("[repack] {stage}"))
    })
    .map_err(|e| format!("streaming {file}: {e}"))
}

fn stream_mlx(
    plan: &InstallPlan,
    dir: &Path,
    client: &Client,
    progress: &mut impl FnMut(&str),
) -> Result<ArchConfig, String> {
    let config_text = String::from_utf8(client.get(&plan.weights.file_url("config.json"))?)
        .map_err(|e| format!("config.json is not UTF-8: {e}"))?;
    let config: serde_json::Value =
        serde_json::from_str(&config_text).map_err(|e| format!("parsing config.json: {e}"))?;
    let family = repack::config_json_family(&config)
        .ok_or_else(|| "config.json declares no model_type this port recognizes".to_string())?;

    let arch = match family {
        ModelFamily::Gemma4 => repack::parse_gemma4_config(&config_text).map_err(|e| e.to_string()),
        ModelFamily::QwenGdnMoe => {
            repack::parse_qwen_gdn_moe_config(&config_text).map_err(|e| e.to_string())
        }
        ModelFamily::QwenGdnDense => {
            repack::parse_qwen_gdn_dense_config(&config_text).map_err(|e| e.to_string())
        }
        other => Err(format!("{} has no safetensors intake here", other.as_str())),
    }?;
    let quant = repack::parse_gemma4_quantization(&config_text)
        .map_err(|e| format!("parsing the quantization block: {e}"))?;

    // The shard list. A single-file checkpoint may still ship an index, so
    // the index is consulted first and `model.safetensors` is the fallback
    // rather than the other way round.
    let shard_names = shard_names(plan, client)?;
    progress(&format!(
        "{} shard(s), family {}, {}-bit affine at group {}",
        shard_names.len(),
        arch.family.as_str(),
        quant.default_bits,
        quant.group_size
    ));

    let sources: Vec<HttpRangeSource> = shard_names
        .iter()
        .map(|name| HttpRangeSource::new(plan.weights.file_url(name)))
        .collect();
    let headers = sources
        .iter()
        .map(|s| repack::fetch_safetensors_header(s).map_err(|e| format!("shard header: {e}")))
        .collect::<Result<Vec<_>, _>>()?;
    let shards = Gemma4Shards::new(
        headers
            .iter()
            .zip(sources.iter())
            .map(|(h, s)| (h, s as &dyn RangeSource))
            .collect(),
    );

    let model_id = plan.weights.repo.clone();
    let report = |stage: &str| progress(&format!("[repack] {stage}"));
    match family {
        ModelFamily::Gemma4 => {
            repack::write_gemma4_install_streamed(dir, &arch, &model_id, &shards, &quant, report)
        }
        ModelFamily::QwenGdnMoe => repack::write_qwen_gdn_moe_install_streamed(
            dir, &arch, &model_id, &shards, &quant, report,
        ),
        ModelFamily::QwenGdnDense => repack::write_qwen_gdn_dense_install_streamed(
            dir, &arch, &model_id, &shards, &quant, report,
        ),
        other => Err(Box::<dyn std::error::Error>::from(format!(
            "{} has no safetensors writer here",
            other.as_str()
        ))),
    }
    .map_err(|e| format!("streaming {}: {e}", plan.weights))?;
    Ok(arch)
}

/// Shard filenames, from the index where there is one.
fn shard_names(plan: &InstallPlan, client: &Client) -> Result<Vec<String>, String> {
    let index_url = plan.weights.file_url("model.safetensors.index.json");
    if let Some(bytes) = client.get_optional(&index_url)? {
        let index: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| format!("parsing the shard index: {e}"))?;
        let map = index
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| "the shard index has no weight_map".to_string())?;
        let mut names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for value in map.values() {
            if let Some(name) = value.as_str() {
                names.insert(name.to_string());
            }
        }
        if !names.is_empty() {
            return Ok(names.into_iter().collect());
        }
    }
    Ok(vec!["model.safetensors".to_string()])
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
