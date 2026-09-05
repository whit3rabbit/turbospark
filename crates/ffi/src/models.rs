//! The model-management surface: browse, probe, install.
//!
//! Portable. Nothing here decodes, so a GUI can list the catalog and install
//! a model on a platform where it could not then run one -- which is the
//! honest behaviour, since the artifact is the same either way.
//!
//! The catalog and store rows are `crates/catalog`'s own `Serialize` types
//! passed straight through, so a GUI's rows and `turbospark-model list`'s
//! rows are the same data and cannot drift. `ProbeReport` is NOT
//! `Serialize` (it carries an `ArchConfig` and a `ModelFamily`), so its
//! projection below is written by hand -- deliberately narrow, since a GUI
//! needs the verdict and the arithmetic behind it rather than the whole
//! config.

use std::sync::Arc;

use catalog::{Catalog, Client, InstallPlan, RepoRef, Store, Verdict};
use serde::Serialize;
use serde_json::json;

/// The install-progress event kinds. Mirrored in `turbospark.h`.
pub const TS_INSTALL_STAGE: i32 = 0;
pub const TS_INSTALL_BYTES: i32 = 1;

#[derive(Serialize)]
struct CatalogRow<'a> {
    #[serde(flatten)]
    entry: &'a catalog::CatalogEntry,
    /// Whether this row is already installed, so a GUI needs one call rather
    /// than two and a join.
    installed: bool,
}

pub(crate) fn catalog_json() -> Result<String, String> {
    let catalog = Catalog::embedded()?;
    let installed = Store::default_store()
        .map(|s| s.installed())
        .unwrap_or_default();
    let rows: Vec<CatalogRow> = catalog
        .entries()
        .map(|entry| CatalogRow {
            installed: installed.contains_key(&entry.alias),
            entry,
        })
        .collect();
    serde_json::to_string(&rows).map_err(|e| e.to_string())
}

pub(crate) fn installed_json() -> Result<String, String> {
    let store = Store::default_store()?;
    let rows: Vec<_> = store.installed().into_values().collect();
    serde_json::to_string(&rows).map_err(|e| e.to_string())
}

/// The header-only probe: what this engine makes of an arbitrary repository,
/// reading kilobytes rather than gigabytes.
///
/// `repo` and `sidecarRepo` accept `owner/name` or `owner/name@revision`.
pub(crate) fn probe_json(
    repo: &str,
    file: Option<&str>,
    sidecar_repo: Option<&str>,
) -> Result<String, String> {
    let repo = parse_repo(repo)?;
    let sidecars = sidecar_repo.map(parse_repo).transpose()?;
    let report = catalog::probe(&Client::new(), &repo, file, sidecars.as_ref())?;

    // A hand-built projection rather than a derived one. The two omissions
    // are deliberate: the full `ArchConfig` is engine internals a GUI has no
    // use for, and `slot_cache_bytes` is included BECAUSE it is the number
    // that decides whether a model fits at all -- `slots x layers x stride`,
    // not the model's size.
    Ok(json!({
        "repo": repo.repo,
        "revision": repo.revision,
        "file": report.file,
        "downloadBytes": report.download_bytes,
        "architecture": report.architecture,
        "family": report.family.map(|f| f.as_str()),
        "runnable": report.verdict.is_runnable(),
        "refusedBecause": match &report.verdict {
            Verdict::Runnable => None,
            Verdict::Refused(why) => Some(why.clone()),
        },
        "types": report.types.iter().map(|t| json!({
            "name": t.name,
            "tensors": t.tensors,
            // Null rather than zero when this port cannot size the type,
            // which is a much louder statement: an unsized type is usually
            // the one carrying the model.
            "bytes": t.bytes,
            "executable": t.executable,
        })).collect::<Vec<_>>(),
        "affine": report.affine.map(|(bits, group)| json!({"bits": bits, "groupSize": group})),
        "expertStride": report.expert_stride,
        "slotCacheBytes": report.slot_cache_bytes().iter()
            .map(|(slots, bytes)| json!({"slots": slots, "bytes": bytes}))
            .collect::<Vec<_>>(),
        "sidecarsPresent": report.sidecars_present,
        "sidecarsMissing": report.sidecars_missing,
        "chatTemplate": report.chat_template,
        "warnings": report.warnings,
    })
    .to_string())
}

/// `owner/name` or `owner/name@revision`, defaulting to `main`.
fn parse_repo(spec: &str) -> Result<RepoRef, String> {
    let (repo, revision) = match spec.split_once('@') {
        Some((r, rev)) => (r, rev),
        None => (spec, "main"),
    };
    if repo.split('/').count() != 2 || repo.starts_with('/') || repo.ends_with('/') {
        return Err(format!(
            "repository must be owner/name (optionally @revision), got {spec:?}"
        ));
    }
    Ok(RepoRef::new(repo, revision))
}

/// Installs the catalog row named `alias` into the store.
///
/// **THE WALK CANNOT RESUME.** It streams 4 to 25 GB and a failure restarts
/// it from the beginning, which is why the first stage line says so and why
/// a GUI must surface that before starting rather than after failing.
///
/// **THE BYTE CALLBACK IS CALLED FROM WORKER THREADS.** `HttpRangeSource`
/// splits a large range into concurrent chunks, so byte progress arrives
/// concurrently and out of order while the STAGE lines arrive on this
/// thread. That is why `on_bytes` is a separate `Fn` bound rather than
/// another arm of the same `FnMut`.
pub(crate) fn install(
    alias: &str,
    mut on_stage: impl FnMut(&str),
    on_bytes: Arc<dyn Fn(u64) + Send + Sync>,
) -> Result<String, String> {
    let catalog = Catalog::embedded()?;
    let entry = catalog
        .get(alias)
        .ok_or_else(|| format!("no catalog row named {alias:?}"))?;
    let plan = InstallPlan::from_entry(entry);
    let store = Store::default_store()?;
    let dir = store.install_path(alias);

    on_stage(
        "this walk streams the checkpoint and CANNOT RESUME: a failure restarts it \
         from the beginning",
    );

    let installed = catalog::install_with_byte_progress(
        &plan,
        &dir,
        &Client::new(),
        |line| on_stage(line),
        Some(on_bytes),
    )?;
    catalog::record(&store, &installed)?;
    serde_json::to_string(&installed.model).map_err(|e| e.to_string())
}

/// The bytes an install of `alias` will read off the network, so a GUI can
/// show a determinate progress bar and a free-space warning before the walk
/// starts.
pub(crate) fn install_bytes(alias: &str) -> Result<(u64, u64), String> {
    let catalog = Catalog::embedded()?;
    let entry = catalog
        .get(alias)
        .ok_or_else(|| format!("no catalog row named {alias:?}"))?;
    Ok((entry.download_bytes, entry.install_bytes))
}

/// Deletes an installed model from the store and drops its directory.
pub(crate) fn delete(alias: &str) -> Result<(), String> {
    let store = Store::default_store()?;
    let path = store
        .resolve(alias)
        .ok_or_else(|| format!("model {alias:?} is not installed"))?;
    if path.exists() {
        std::fs::remove_dir_all(&path)
            .map_err(|e| format!("failed to remove {}: {e}", path.display()))?;
    }
    store.forget(alias)
}

/// Ranks curated models by hardware fit for this machine at `context`.
///
/// **`guard` MUST match what the host will OPEN with.** This ranking and
/// `ts_session_open`'s refusal share one budget by construction, which is
/// what makes a recommendation trustworthy; a hub ranking under `relaxed`
/// while its sessions open under `strict` promises a fit the loader then
/// refuses, in the one place the user cannot see the two disagree.
pub(crate) fn recommend_json(
    context: Option<u32>,
    guard: model_io::LoadGuard,
) -> Result<String, String> {
    let physical = runtime::physical_memory();
    if physical == 0 {
        return Err(
            "no physical memory probe available on this platform; cannot recommend models"
                .to_string(),
        );
    }
    let (working_set, chip) = match runtime::recommended_max_working_set() {
        Some((bytes, name)) => (Some(bytes), name),
        None => (None, String::new()),
    };
    let machine = catalog::Machine {
        physical_bytes: physical,
        working_set_bytes: working_set,
        load_guard: guard,
        chip,
    };
    let catalog = Catalog::embedded()?;
    let entries: Vec<&catalog::CatalogEntry> = catalog.entries().collect();
    let context_val = context.unwrap_or(4096);
    let recommendations = catalog::recommend_catalog(&entries, &machine, context_val);
    let rows: Vec<_> = recommendations
        .into_iter()
        .map(|r| {
            let alias = match &r.origin {
                catalog::Origin::Catalog(a) => a.clone(),
                catalog::Origin::Discovered { repo, .. } => repo.clone(),
            };
            json!({
                "alias": alias,
                "name": r.name,
                "family": r.family,
                "verdict": match r.fit.verdict {
                    catalog::FitVerdict::Resident => "resident",
                    catalog::FitVerdict::Streams => "streams",
                    catalog::FitVerdict::Tight => "tight",
                    catalog::FitVerdict::Refused => "refused",
                    catalog::FitVerdict::Unknown => "unknown",
                },
                "verdictSummary": r.fit.verdict.as_str(),
                "runs": r.fit.verdict.runs(),
                "countedBytes": r.fit.counted,
                "installBytes": r.fit.mapped,
                "slotCacheSlots": r.fit.slots,
                "largestContext": r.fit.largest_context,
                "notes": r.notes,
                "toksPerSecondMin": r.measured.as_ref().map(|m| m.decode_tok_s_min),
                "toksPerSecondMax": r.measured.as_ref().map(|m| m.decode_tok_s_max),
            })
        })
        .collect();
    serde_json::to_string(&rows).map_err(|e| e.to_string())
}

/// Probes and installs an arbitrary Hugging Face model repository.
pub(crate) fn install_repo(
    repo: &str,
    alias: &str,
    file: Option<&str>,
    sidecar_repo: Option<&str>,
    mut on_stage: impl FnMut(&str),
    on_bytes: Arc<dyn Fn(u64) + Send + Sync>,
) -> Result<String, String> {
    let weights = parse_repo(repo)?;
    let sidecars = match sidecar_repo {
        Some(text) => parse_repo(text)?,
        None => weights.clone(),
    };
    let client = Client::new();
    let report = catalog::probe(&client, &weights, file, Some(&sidecars))?;
    if !report.verdict.is_runnable() {
        return Err(match report.verdict {
            Verdict::Refused(why) => format!("model would not run here: {why}"),
            Verdict::Runnable => unreachable!(),
        });
    }
    let plan = InstallPlan::from_probe(alias, &report, sidecars);
    let store = Store::default_store()?;
    let dir = store.install_path(alias);
    if dir.join("manifest.json").is_file() {
        return Err(format!("{} already holds an install", dir.display()));
    }
    on_stage(
        "this walk streams the checkpoint and CANNOT RESUME: a failure restarts it \
         from the beginning",
    );
    let installed = catalog::install_with_byte_progress(
        &plan,
        &dir,
        &client,
        |line| on_stage(line),
        Some(on_bytes),
    )?;
    catalog::record(&store, &installed)?;
    serde_json::to_string(&installed.model).map_err(|e| e.to_string())
}

/// What a `.gguf` control vector declares, read from the file alone.
///
/// **A SHAPE MATCH IS NOT A SEMANTIC MATCH, and this reports the shape only.**
/// `SteeringSet::validate` refuses a width or layer-count mismatch, and refuses
/// nothing else: a vector extracted for a different checkpoint of the same
/// hidden size opens, steers, and changes behaviour in a direction nobody
/// asked for, silently (`docs/OBLITERATION.md`, "Running someone else's
/// vector"). A caller rendering these fields owes the user that sentence.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ControlVectorInfo {
    /// The hidden size every direction in the file declares. Must equal the
    /// install's own `arch.hiddenSize` or `open` refuses the set.
    hidden: usize,
    /// How many blocks actually carry a direction.
    covered_layers: usize,
    /// Lowest and highest block index covered, 0-based in THIS port's
    /// convention after `turbospark.layer_base` has been honoured. A file
    /// written by llama.cpp or by this port since 2026-08-24 starts at 1,
    /// because block 0 is not expressible (`crates/repack`'s
    /// `LayerZeroNotExpressible`).
    min_layer: Option<usize>,
    max_layer: Option<usize>,
    /// The span `layers` covers, which is `max_layer + 1` and is what the
    /// install's own `arch.numLayers` is compared against.
    spanned_layers: usize,
    /// The mode the file itself declares, used when the caller names none.
    declared_mode: Option<String>,
    /// The architecture string the file was extracted against, when it
    /// carries one. **Advisory only** -- nothing validates against it, which
    /// is exactly why it is worth showing.
    declared_arch: Option<String>,
}

/// Reads a control vector's header and reports what it covers, with no model
/// open and no session.
///
/// Goes through `repack::control_vector::load_control_vector`, the ONE parser
/// that decides what a vector means, rather than letting a GUI reimplement the
/// GGUF layout. A vector is ~1.3 MB, so this is milliseconds.
pub fn control_vector_info_json(path: &str) -> Result<String, String> {
    let set = repack::control_vector::load_control_vector(std::path::Path::new(path))
        .map_err(|e| format!("{e}"))?;
    // `covered_layers()` is a COUNT, so the indices come from the vector
    // itself. Both are wanted: the count says how much of the model is
    // steered and the range says WHICH part, and a user comparing a file
    // against an install needs the second (a 31-of-32 vector starting at
    // block 1 is llama.cpp's convention working correctly, not a gap).
    let covered: Vec<usize> = set
        .layers
        .iter()
        .enumerate()
        .filter_map(|(l, d)| d.as_ref().map(|_| l))
        .collect();
    let info = ControlVectorInfo {
        hidden: set.hidden,
        covered_layers: set.covered_layers(),
        min_layer: covered.first().copied(),
        max_layer: covered.last().copied(),
        spanned_layers: set.layers.len(),
        declared_mode: set.declared_mode.map(|m| m.as_str().to_string()),
        declared_arch: set.declared_arch.clone(),
    };
    serde_json::to_string(&info).map_err(|e| e.to_string())
}

#[cfg(test)]
mod control_vector_info_tests {
    use super::control_vector_info_json;
    use std::collections::BTreeMap;

    /// Writes a control vector through `repack`'s own writer, so the test
    /// reads what this engine really produces rather than a hand-rolled GGUF.
    fn write_vector(dir: &std::path::Path, hidden: usize, blocks: &[usize]) -> std::path::PathBuf {
        let mut directions: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
        for (n, &b) in blocks.iter().enumerate() {
            directions.insert(b, (0..hidden).map(|i| (i + n + 1) as f32).collect());
        }
        let bytes = repack::control_vector::write_control_vector(&directions, "test-arch", None)
            .expect("the writer should accept these directions");
        let path = dir.join("d.gguf");
        std::fs::write(&path, bytes).expect("write");
        path
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ts-cv-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// The reported shape is what the FILE carries, not what the caller
    /// hoped. A host compares `hidden` against its install's own
    /// `arch.hiddenSize`, so a wrong width here is a compatibility check that
    /// passes on a vector `open` will refuse.
    #[test]
    fn the_reported_width_and_coverage_come_from_the_file() {
        let dir = scratch("shape");
        let path = write_vector(&dir, 64, &[1, 2, 3, 7]);
        let json = control_vector_info_json(path.to_str().unwrap()).expect("readable");
        let v: serde_json::Value = serde_json::from_str(&json).expect("json");

        assert_eq!(v["hidden"], 64);
        assert_eq!(v["coveredLayers"], 4);
        assert_eq!(v["minLayer"], 1);
        assert_eq!(v["maxLayer"], 7);
        // The SPAN, not the count: a gapped vector covers 4 blocks across 8,
        // and it is the span that gets compared against the model's layer
        // count. Reporting only the count would call an 8-block vector on a
        // 5-layer model compatible.
        assert_eq!(v["spannedLayers"], 8);
        assert_eq!(v["declaredArch"], "test-arch");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// **BLOCK 0 IS NOT EXPRESSIBLE**, so a well-formed file starts at 1 and
    /// a host must not read "31 of 32" as a gap. Pinned here because the
    /// natural reading of a minLayer of 1 is that something is missing.
    #[test]
    fn a_vector_starts_at_block_one_because_block_zero_cannot_be_written() {
        let dir = scratch("base");
        let path = write_vector(&dir, 8, &[1, 2]);
        let json = control_vector_info_json(path.to_str().unwrap()).expect("readable");
        let v: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(v["minLayer"], 1);

        let mut zero: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
        zero.insert(0, vec![1.0; 8]);
        assert!(
            repack::control_vector::write_control_vector(&zero, "test-arch", None).is_err(),
            "the writer must refuse block 0, which is what makes minLayer 1 normal"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The same reading against a REAL vector, which a synthetic one cannot
    /// stand in for: the fixtures above are written by this port's own writer
    /// on this run, so they cannot catch a convention that drifted between
    /// the writer and a file already on disk.
    ///
    /// `#[ignore]`d and env-gated, like every other real-artifact test here.
    ///
    /// ```sh
    /// TURBOSPARK_STEERING_VECTOR=~/models/steering-vectors/ocean-legacy-layerbase0.gguf \
    ///   cargo test -p turbospark-ffi --lib -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn a_real_control_vector_reports_its_own_shape() {
        let Ok(path) = std::env::var("TURBOSPARK_STEERING_VECTOR") else {
            eprintln!("set TURBOSPARK_STEERING_VECTOR to run this");
            return;
        };
        let json = control_vector_info_json(&path).expect("a real vector should read");
        eprintln!("{json}");
        let v: serde_json::Value = serde_json::from_str(&json).expect("json");
        // Nothing here asserts a SPECIFIC width: the point is that the fields
        // are populated from the file rather than defaulted, and a hardcoded
        // 5120 would make this a test about one vector on one machine.
        assert!(v["hidden"].as_u64().unwrap_or(0) > 0, "{json}");
        assert!(v["coveredLayers"].as_u64().unwrap_or(0) > 0, "{json}");
        assert!(
            v["spannedLayers"].as_u64().unwrap_or(0) >= v["coveredLayers"].as_u64().unwrap_or(0),
            "the span can never be smaller than the count: {json}"
        );
    }

    /// An unreadable path is an ERROR naming itself, never an empty or
    /// default-looking reading. A host that got `{"hidden":0}` back would
    /// render a width mismatch against every model instead of "that file is
    /// not a control vector".
    #[test]
    fn a_path_that_is_not_a_control_vector_is_refused_rather_than_reported_empty() {
        let dir = scratch("bad");
        let path = dir.join("not-a-vector.gguf");
        std::fs::write(&path, b"this is not a gguf").expect("write");
        assert!(control_vector_info_json(path.to_str().unwrap()).is_err());
        assert!(control_vector_info_json(dir.join("missing.gguf").to_str().unwrap()).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
