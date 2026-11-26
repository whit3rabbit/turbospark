//! Hugging Face installs of the Inkling 2 Mini sources, which all need whole
//! files on disk: a LookAtMyQuant packed release repository, an Inkling 2
//! GGUF, the original checkpoint and an MLX conversion. None has a streamed
//! walk (the LAMQ intake reads npz members, the GGUF intake reads the file
//! end to end, and the checkpoint intakes read every expert of a layer to
//! write its blobs), so the route is:
//!
//! 1. pin the repository to the commit the Hub reports, so every file comes
//!    from one revision;
//! 2. download the release (or the one GGUF, plus the tokenizer files from
//!    the base repository, since a GGUF carries none) into a hidden folder
//!    beside the install's staging, skipping files already there at their
//!    listed size, so a retry resumes;
//! 3. hand the folder to [`crate::install_inkling2_local`], which writes,
//!    verifies, moves and records the install;
//! 4. remove the downloaded files once the install is recorded.
//!
//! The probe reports these repositories through [`probe_report`], so a host
//! that gates on the probe verdict sees them as runnable.

use std::path::{Path, PathBuf};

use crate::hf::{Client, RepoRef};
use crate::hub_validation::HubMetadataValidator;
use crate::inkling2_local::install_inkling2_local;
use crate::install::{human_bytes, CancelFlag, InstallTarget, Installed, INSTALL_CANCELLED};
use crate::library::LocationEnv;
use crate::probe::{ProbeReport, Verdict};
use crate::store::{Store, StoreError};
use crate::Inkling2LocalSource;

/// Where a GGUF install takes its tokenizer files when the caller names no
/// sidecar repository: the base checkpoint every Inkling 2 Mini GGUF and
/// LAMQ release is converted from.
pub const INKLING2_TOKENIZER_REPO: &str = "thinkingmachines/Inkling-2-Mini";

/// Which whole-file intake a repository takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inkling2RepoKind {
    Lamq,
    Gguf,
    /// The original checkpoint (`thinkingmachines/Inkling-2-Mini`).
    Hf,
    /// An MLX conversion of it.
    Mlx,
}

impl Inkling2RepoKind {
    /// Whether the install expands past the download (LAMQ padding, GGUF
    /// rewrite) rather than keeping the published expert bytes.
    fn packed(self) -> bool {
        matches!(self, Self::Lamq | Self::Gguf)
    }
}

/// A repository resolved for a whole-file Inkling install.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inkling2RepoSource {
    pub kind: Inkling2RepoKind,
    /// Pinned to the commit the Hub reported.
    pub repo: RepoRef,
    /// The weights files to download, with their listed sizes. For a LAMQ
    /// release this is the whole release, tokenizer files included.
    pub files: Vec<(String, u64)>,
    /// The architecture or format string the source declares:
    /// `general.architecture` for a GGUF, the manifest `format` for LAMQ.
    pub architecture: String,
    /// Pinned. The weights repository itself for a LAMQ release.
    pub tokenizer_repo: RepoRef,
    /// Tokenizer and chat files fetched from `tokenizer_repo` (GGUF only;
    /// a LAMQ release carries its own among `files`).
    pub tokenizer_files: Vec<String>,
}

impl Inkling2RepoSource {
    /// Total bytes of the listed weights files.
    pub fn download_bytes(&self) -> u64 {
        self.files.iter().map(|(_, n)| n).sum()
    }
}

/// True for the `general.architecture` strings the Inkling GGUF intake
/// takes.
pub(crate) fn is_inkling2_gguf_arch(arch: Option<&str>) -> bool {
    matches!(arch, Some("inkling2_loop" | "inkling2"))
}

/// A repository listing: each file and its size when the Hub gave one.
type Listing = Vec<(String, Option<u64>)>;

/// The repository's files and the commit they are at, through the same
/// validator the Hub search uses (path charset, sizes, revision shape).
fn pinned_listing(client: &Client, repo: &RepoRef) -> Result<(RepoRef, Listing), String> {
    let validator = HubMetadataValidator::default();
    let body = client.get_bounded(
        &repo.api_url_with_sizes(),
        validator.limits().max_response_bytes,
    )?;
    let report = validator
        .validate_repository_payload(&body)
        .map_err(|e| format!("{repo}: {e}"))?;
    if !report.valid.repo_id.eq_ignore_ascii_case(&repo.repo) {
        return Err(format!(
            "{repo}: the Hub answered for {:?}, a different repository",
            report.valid.repo_id
        ));
    }
    let files = report
        .valid
        .files
        .into_iter()
        .map(|f| (f.path, f.size_bytes))
        .collect();
    Ok((
        RepoRef::new(report.valid.repo_id, report.valid.revision),
        files,
    ))
}

/// The files of a LAMQ release worth downloading: top-level JSON, npz,
/// safetensors and Jinja files. README, licence, images and anything in a
/// subfolder are not read by the intake. Every one needs a listed size, so
/// the space check and resume are exact.
pub(crate) fn lamq_release_files(
    repo: &RepoRef,
    listing: &[(String, Option<u64>)],
) -> Result<Vec<(String, u64)>, String> {
    let mut out = Vec::new();
    for (name, size) in listing {
        let wanted = !name.contains('/')
            && [".json", ".npz", ".safetensors", ".jinja"]
                .iter()
                .any(|ext| name.ends_with(ext));
        if !wanted {
            continue;
        }
        let size = size.ok_or_else(|| format!("{repo}: the Hub listed no size for {name}"))?;
        out.push((name.clone(), size));
    }
    out.sort();
    Ok(out)
}

/// The Inkling tokenizer files `listing` holds; `tokenizer.json` is
/// required.
pub(crate) fn tokenizer_files_in(
    repo: &RepoRef,
    listing: &[(String, Option<u64>)],
) -> Result<Vec<String>, String> {
    let names: Vec<String> = repack::INKLING2_TOKENIZER_FILES
        .iter()
        .filter(|want| listing.iter().any(|(n, _)| n == *want))
        .map(|s| s.to_string())
        .collect();
    if !names.iter().any(|n| n == "tokenizer.json") {
        return Err(format!(
            "{repo} has no tokenizer.json; name a sidecar repository that has the Inkling 2 \
             tokenizer (such as {INKLING2_TOKENIZER_REPO})"
        ));
    }
    Ok(names)
}

/// Resolves `repo` for a whole-file Inkling install, or None when it is
/// neither a LAMQ release nor an Inkling GGUF (the caller then takes the
/// ordinary probe and walk). `want_file` picks the GGUF as `--file` does;
/// `tokenizer_repo` overrides [`INKLING2_TOKENIZER_REPO`] for a GGUF.
///
/// Requests: the repository listing, then `manifest.json` (a few KB) or the
/// GGUF header (a ranged read), then the tokenizer repository's listing.
pub fn detect_inkling2_repo(
    client: &Client,
    repo: &RepoRef,
    want_file: Option<&str>,
    tokenizer_repo: Option<&RepoRef>,
) -> Result<Option<Inkling2RepoSource>, String> {
    let (pinned, listing) = pinned_listing(client, repo)?;
    let ggufs: Vec<&(String, Option<u64>)> = listing
        .iter()
        .filter(|(n, _)| n.ends_with(".gguf"))
        .collect();
    let gguf = match want_file {
        Some(f) => ggufs.iter().find(|(n, _)| n == f).copied(),
        None if ggufs.len() == 1 => Some(ggufs[0]),
        None => None,
    };
    if let Some((file, size)) = gguf {
        let source = crate::stream::range_source(pinned.file_url(file), None, client, None, None);
        let header = repack::fetch_gguf_header(&source).map_err(|e| format!("{file}: {e}"))?;
        if !is_inkling2_gguf_arch(header.architecture()) {
            return Ok(None);
        }
        let arch = header.architecture().unwrap_or_default().to_string();
        let shards = repack::gguf_shard_names(file, &header)?;
        if shards.len() != 1 {
            return Err(format!(
                "{file} is one of {} split files; the Inkling GGUF intake reads one whole file. \
                 Merge the parts with llama.cpp's gguf-split, then use pull-local",
                shards.len()
            ));
        }
        let size = size.ok_or_else(|| format!("{pinned}: the Hub listed no size for {file}"))?;
        // Callers that name no sidecar often pass the weights repository
        // itself; a GGUF repository rarely has tokenizer.json, so that means
        // the base repository unless this one really carries it.
        let own_tokenizer = listing.iter().any(|(n, _)| n == "tokenizer.json");
        let (tok_pinned, tok_listing) = match tokenizer_repo {
            Some(t) if !t.repo.eq_ignore_ascii_case(&repo.repo) => pinned_listing(client, t)?,
            Some(_) if own_tokenizer => (pinned.clone(), listing.clone()),
            _ => pinned_listing(client, &RepoRef::new(INKLING2_TOKENIZER_REPO, "main"))?,
        };
        let tokenizer_files = tokenizer_files_in(&tok_pinned, &tok_listing)?;
        return Ok(Some(Inkling2RepoSource {
            kind: Inkling2RepoKind::Gguf,
            repo: pinned,
            files: vec![(file.clone(), size)],
            architecture: arch,
            tokenizer_repo: tok_pinned,
            tokenizer_files,
        }));
    }
    if want_file.is_some() || !ggufs.is_empty() {
        return Ok(None);
    }
    // A LookAtMyQuant release also ships an HF `config.json` declaring
    // `inkling2_loop` and a `steering.safetensors`, so its manifest is read
    // first; only a repository that is not a LAMQ release is tried as a
    // checkpoint.
    let manifest = if listing.iter().any(|(n, _)| n == "manifest.json") {
        client.get_optional(&pinned.file_url("manifest.json"))?
    } else {
        None
    };
    let format = manifest.as_ref().and_then(|m| {
        serde_json::from_slice::<serde_json::Value>(m)
            .ok()
            .and_then(|v| v.get("format").and_then(|f| f.as_str()).map(str::to_string))
    });
    if format.as_deref() != Some(repack::LAMQ_INKLING2_FORMAT) {
        let checkpoint = listing.iter().any(|(n, _)| n == "config.json")
            && listing.iter().any(|(n, _)| n.ends_with(".safetensors"));
        return if checkpoint {
            detect_checkpoint(client, &pinned, &listing)
        } else {
            Ok(None)
        };
    }
    let files = lamq_release_files(&pinned, &listing)?;
    // Checked here, before any download, rather than by the intake after it.
    tokenizer_files_in(&pinned, &listing)?;
    Ok(Some(Inkling2RepoSource {
        kind: Inkling2RepoKind::Lamq,
        repo: pinned.clone(),
        files,
        architecture: repack::LAMQ_INKLING2_FORMAT.to_string(),
        tokenizer_repo: pinned,
        tokenizer_files: Vec::new(),
    }))
}

/// The original checkpoint or an MLX conversion, from its `config.json`;
/// None for any other safetensors repository. Downloads the top-level JSON,
/// safetensors and Jinja files. The original checkpoint's DSpark block
/// drafter (`mtp.safetensors`, 636 MB) comes along: the intake copies it
/// into `<install>/dspark/`, and it is never counted as a weights file.
fn detect_checkpoint(
    client: &Client,
    pinned: &RepoRef,
    listing: &[(String, Option<u64>)],
) -> Result<Option<Inkling2RepoSource>, String> {
    let Some(config) = client.get_optional(&pinned.file_url("config.json"))? else {
        return Ok(None);
    };
    let Some(mlx) = std::str::from_utf8(&config)
        .ok()
        .and_then(crate::inkling2_local::inkling2_safetensors_kind)
    else {
        return Ok(None);
    };
    let files: Vec<(String, u64)> = lamq_release_files(pinned, listing)?
        .into_iter()
        .filter(|(n, _)| !n.ends_with(".npz"))
        .collect();
    let weights = files
        .iter()
        .filter(|(n, _)| n.ends_with(".safetensors") && n != "mtp.safetensors")
        .count();
    if !mlx && weights != 1 {
        return Err(format!(
            "{pinned} lists {weights} weights files; the Inkling 2 checkpoint intake reads one"
        ));
    }
    tokenizer_files_in(pinned, listing)?;
    Ok(Some(Inkling2RepoSource {
        kind: if mlx {
            Inkling2RepoKind::Mlx
        } else {
            Inkling2RepoKind::Hf
        },
        repo: pinned.clone(),
        files,
        architecture: "inkling2_loop".to_string(),
        tokenizer_repo: pinned.clone(),
        tokenizer_files: Vec::new(),
    }))
}

/// The probe's answer for a resolved source: runnable, Inkling 2 Mini, with
/// the whole download named, because nothing streams.
pub fn probe_report(source: &Inkling2RepoSource) -> ProbeReport {
    let (kind, file) = match source.kind {
        Inkling2RepoKind::Gguf => (
            crate::SourceKind::Gguf,
            source.files.first().map(|f| f.0.clone()),
        ),
        Inkling2RepoKind::Lamq => (crate::SourceKind::Lamq, None),
        Inkling2RepoKind::Hf | Inkling2RepoKind::Mlx => (crate::SourceKind::Mlx, None),
    };
    let tokenizer_present = match source.kind {
        Inkling2RepoKind::Gguf => source.tokenizer_files.clone(),
        Inkling2RepoKind::Lamq | Inkling2RepoKind::Hf | Inkling2RepoKind::Mlx => {
            repack::INKLING2_TOKENIZER_FILES
                .iter()
                .filter(|n| source.files.iter().any(|(f, _)| f == *n))
                .map(|s| s.to_string())
                .collect()
        }
    };
    let chat_template = tokenizer_present
        .iter()
        .any(|n| n == "chat_template.jinja")
        .then(|| format!("{}/chat_template.jinja", source.tokenizer_repo.repo));
    let bytes = source.download_bytes();
    ProbeReport {
        repo: source.repo.clone(),
        kind,
        file,
        download_bytes: Some(bytes),
        architecture: Some(source.architecture.clone()),
        family: Some(model_io::ModelFamily::Inkling2Loop),
        arch: Some(model_io::inkling2_mini()),
        types: Vec::new(),
        affine: None,
        mxfp4: false,
        expert_stride: None,
        trained_context: None,
        sidecars_present: tokenizer_present,
        sidecars_missing: Vec::new(),
        chat_template,
        verdict: Verdict::Runnable,
        adapter: None,
        speculation: None,
        warnings: vec![format!(
            "this installs by downloading the whole {} ({}) first and converting it on this Mac; \
             it needs about {} free on the install's drive while both exist",
            match source.kind {
                Inkling2RepoKind::Gguf => "GGUF",
                Inkling2RepoKind::Lamq => "LookAtMyQuant release",
                Inkling2RepoKind::Hf => "checkpoint",
                Inkling2RepoKind::Mlx => "MLX conversion",
            },
            human_bytes(bytes),
            human_bytes(bytes + inkling2_install_bytes(source.kind, bytes)),
        )],
    }
}

/// The estimate `crate::inkling2_local_required_bytes` makes from files on
/// disk, from listed sizes (the same function).
fn inkling2_install_bytes(kind: Inkling2RepoKind, source_bytes: u64) -> u64 {
    crate::inkling2_local::inkling2_install_estimate(kind.packed(), source_bytes)
}

/// The hidden download folder for `source`, named by repository and commit
/// so two releases never share one and a retry finds its own.
pub(crate) fn download_dir(staging_parent: &Path, source: &Inkling2RepoSource) -> PathBuf {
    let slug: String = format!("{}@{}", source.repo.repo, source.repo.revision)
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    staging_parent.join(format!(".inkling2-download-{slug}"))
}

/// Downloads `name` from `repo` to `dest` unless a file of `size` bytes is
/// already there. Reports the bytes either way, so a resumed install's bar
/// starts where the last one stopped.
///
/// Sized non-empty files use `repack::HttpRangeSource` (8-way concurrent 16 MiB
/// HTTP/1.1 ranges with resume cache on immutable revisions) so multi-GB
/// checkpoints saturate the link instead of crawling on one HTTP/2 stream.
fn fetch_file(
    client: &Client,
    repo: &RepoRef,
    name: &str,
    size: Option<u64>,
    dest: &Path,
    cache_dir: Option<&Path>,
    on_bytes: &mut dyn FnMut(u64),
    cancel: Option<&CancelFlag>,
) -> Result<bool, String> {
    if let (Some(size), Ok(meta)) = (size, std::fs::metadata(dest)) {
        if meta.is_file() && meta.len() == size {
            on_bytes(size);
            return Ok(false);
        }
    }
    if std::fs::symlink_metadata(dest).is_ok() {
        // A short or foreign leftover; this folder is this route's own.
        std::fs::remove_file(dest).map_err(|e| format!("replacing {}: {e}", dest.display()))?;
    }
    if let Some(expected_size) = size {
        if expected_size == 0 {
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("creating {}: {e}", parent.display()))?;
            }
            std::fs::File::create(dest).map_err(|e| format!("creating {}: {e}", dest.display()))?;
            return Ok(true);
        }
        download_sized_batch(
            client,
            repo,
            &[(name, dest, expected_size)],
            cache_dir,
            on_bytes,
            cancel,
        )?;
        return Ok(true);
    }
    let got = client.download_to_with(&repo.file_url(name), dest, on_bytes, cancel)?;
    let _ = got;
    Ok(true)
}

/// Downloads up to `HttpRangeSource::FILE_DOWNLOAD_WORKERS` sized files in
/// parallel through `HttpRangeSource::download_batch`, forwarding chunk byte
/// counts onto the caller's `on_bytes` callback.
fn download_sized_batch(
    client: &Client,
    repo: &RepoRef,
    files: &[(&str, &Path, u64)],
    cache_dir: Option<&Path>,
    on_bytes: &mut dyn FnMut(u64),
    cancel: Option<&CancelFlag>,
) -> Result<(), String> {
    if files.is_empty() {
        return Ok(());
    }
    enum Event {
        Bytes(u64),
        Finished(Result<(), (usize, repack::DownloadError)>),
    }
    let (sender, receiver) = std::sync::mpsc::channel();
    let sender = std::sync::Arc::new(sender);
    let mut sources = Vec::with_capacity(files.len());
    for &(name, dest, size) in files {
        let byte_sender = std::sync::Arc::downgrade(&sender);
        let mut range = repack::HttpRangeSource::with_progress(
            repo.file_url(name),
            std::sync::Arc::new(move |delta| {
                if let Some(sender) = byte_sender.upgrade() {
                    let _ = sender.send(Event::Bytes(delta));
                }
            }),
        )
        .with_optional_token(client.token());
        if let Some(cache) = cache_dir {
            range = range.with_cache_dir(cache);
        }
        if let Some(flag) = cancel {
            range = range.with_cancel(flag.clone());
        }
        sources.push((range, dest, size));
    }
    let result = std::thread::scope(|scope| {
        let downloads: Vec<_> = sources
            .iter()
            .map(|(source, dest, size)| (source, *dest, *size))
            .collect();
        scope.spawn(move || {
            let _ = sender.send(Event::Finished(repack::HttpRangeSource::download_batch(
                &downloads,
            )));
        });
        for event in receiver {
            match event {
                Event::Bytes(delta) => on_bytes(delta),
                Event::Finished(res) => return res,
            }
        }
        unreachable!("the batch worker owns the completion sender");
    });
    result.map_err(|(index, error)| match error {
        repack::DownloadError::Cancelled => INSTALL_CANCELLED.to_string(),
        other => format!("downloading {}: {other}", files[index].0),
    })
}

/// Downloads, installs and records `source` at `target`, then removes the
/// download. On failure the downloaded files are kept (the error says
/// where), so the next attempt skips them.
#[allow(clippy::too_many_arguments)]
pub fn install_inkling2_repo(
    source: &Inkling2RepoSource,
    alias: &str,
    target: &InstallTarget,
    client: &Client,
    mut progress: impl FnMut(&str),
    on_bytes: &mut dyn FnMut(u64),
    cancel: Option<&CancelFlag>,
    store: &Store,
    env: &LocationEnv,
) -> Result<Installed, StoreError> {
    let failed =
        |e: String| store.install_failure(&target.root, &target.root, &target.library_name, e);
    if std::fs::symlink_metadata(&target.dir).is_ok() {
        return Err(StoreError::Io(format!(
            "{} already exists; this never overwrites. Remove it or pick another alias",
            target.dir.display()
        )));
    }
    // Beside the install's staging, so the download, the install and its
    // final rename share one drive.
    let staging_parent = store.staging_parent_for(&target.root)?;
    let dl = download_dir(&staging_parent, source);
    std::fs::create_dir_all(&dl).map_err(|e| failed(format!("creating {}: {e}", dl.display())))?;

    let have: u64 = source
        .files
        .iter()
        .filter(|(n, size)| {
            std::fs::metadata(dl.join(local_name(n))).is_ok_and(|m| m.len() == *size)
        })
        .map(|(_, size)| size)
        .sum();
    let remaining = source.download_bytes() - have;
    let need = remaining + inkling2_install_bytes(source.kind, source.download_bytes());
    let available = model_io::available_bytes(&dl).map_err(|e| failed(e.to_string()))?;
    if available < need {
        return Err(failed(format!(
            "installing {} needs about {} free ({} still to download, then the install); \
             {} is available",
            source.repo,
            human_bytes(need),
            human_bytes(remaining),
            human_bytes(available)
        )));
    }
    progress(&format!(
        "downloading {} of {} from {} into {}",
        human_bytes(remaining),
        human_bytes(source.download_bytes()),
        source.repo,
        dl.display()
    ));

    let kept = |e: String| {
        failed(format!(
            "{e}. The downloaded files are kept in {} so a retry skips them; remove the folder \
             to free the space",
            dl.display()
        ))
    };
    let range_cache = crate::stream::immutable_download_cache(&dl, &source.repo);
    let mut pending: Vec<(&str, PathBuf, u64)> = Vec::new();
    for (name, size) in &source.files {
        if cancel.is_some_and(|c| c.checkpoint().is_err()) {
            return Err(kept(INSTALL_CANCELLED.to_string()));
        }
        let dest = dl.join(local_name(name));
        if let Ok(meta) = std::fs::metadata(&dest) {
            if meta.is_file() && meta.len() == *size {
                on_bytes(*size);
                continue;
            }
        }
        if std::fs::symlink_metadata(&dest).is_ok() {
            std::fs::remove_file(&dest)
                .map_err(|e| kept(format!("replacing {}: {e}", dest.display())))?;
        }
        if *size == 0 {
            std::fs::File::create(&dest)
                .map_err(|e| kept(format!("creating {}: {e}", dest.display())))?;
            progress(&format!("downloaded {name} ({})", human_bytes(*size)));
            continue;
        }
        pending.push((name.as_str(), dest, *size));
    }
    for batch in pending.chunks(repack::HttpRangeSource::FILE_DOWNLOAD_WORKERS) {
        if cancel.is_some_and(|c| c.checkpoint().is_err()) {
            return Err(kept(INSTALL_CANCELLED.to_string()));
        }
        let entries: Vec<(&str, &Path, u64)> = batch
            .iter()
            .map(|(name, dest, size)| (*name, dest.as_path(), *size))
            .collect();
        download_sized_batch(
            client,
            &source.repo,
            &entries,
            range_cache.as_deref(),
            on_bytes,
            cancel,
        )
        .map_err(&kept)?;
        for (name, _, size) in batch {
            progress(&format!("downloaded {name} ({})", human_bytes(*size)));
        }
    }
    let tokenizer_dir = dl.join("tokenizer");
    if source.kind == Inkling2RepoKind::Gguf {
        std::fs::create_dir_all(&tokenizer_dir)
            .map_err(|e| kept(format!("creating {}: {e}", tokenizer_dir.display())))?;
        for name in &source.tokenizer_files {
            fetch_file(
                client,
                &source.tokenizer_repo,
                name,
                None,
                &tokenizer_dir.join(name),
                None,
                &mut |_| {},
                cancel,
            )
            .map_err(&kept)?;
        }
        progress(&format!(
            "fetched the tokenizer files from {}",
            source.tokenizer_repo
        ));
    }

    let local = match source.kind {
        Inkling2RepoKind::Lamq => Inkling2LocalSource::Lamq(dl.clone()),
        Inkling2RepoKind::Hf => Inkling2LocalSource::Hf(dl.clone()),
        Inkling2RepoKind::Mlx => Inkling2LocalSource::Mlx(dl.clone()),
        Inkling2RepoKind::Gguf => {
            Inkling2LocalSource::Gguf(dl.join(local_name(&source.files[0].0)))
        }
    };
    // Measured again: the download just took its share of the drive.
    let available = model_io::available_bytes(target.dir.parent().unwrap_or(&target.dir))
        .map_err(|e| kept(e.to_string()))?;
    let installed = install_inkling2_local(
        &local,
        alias,
        target,
        (source.kind == Inkling2RepoKind::Gguf).then_some(tokenizer_dir.as_path()),
        &source.repo.repo,
        &source.repo.revision,
        available,
        &mut progress,
        store,
        env,
    )
    .map_err(|e| match e {
        StoreError::Io(msg) => StoreError::Io(format!(
            "{msg}. The downloaded files are kept in {} so a retry skips the download",
            dl.display()
        )),
        other => other,
    })?;
    // Recorded; the download is no longer needed. This folder is this
    // route's own, named for this repository and commit.
    if let Err(e) = std::fs::remove_dir_all(&dl) {
        progress(&format!(
            "could not remove the download {}: {e}",
            dl.display()
        ));
    }
    Ok(installed)
}

/// True when a catalog row installs through this route (every
/// `inkling2_loop` row: none of its sources has a streamed walk).
pub fn is_inkling2_row(entry: &crate::CatalogEntry) -> bool {
    entry.family == model_io::ModelFamily::Inkling2Loop.as_str()
}

/// Installs the Inkling 2 catalog row `entry` at `target`: resolves its
/// repository at the row's pinned revision, refuses a Hub answer for any
/// other commit, installs through [`install_inkling2_repo`], and records
/// the install under the row's own status rather than `unlisted`.
#[allow(clippy::too_many_arguments)]
pub fn install_inkling2_row(
    entry: &crate::CatalogEntry,
    target: &InstallTarget,
    client: &Client,
    progress: impl FnMut(&str),
    on_bytes: &mut dyn FnMut(u64),
    cancel: Option<&CancelFlag>,
    store: &Store,
    env: &LocationEnv,
) -> Result<Installed, StoreError> {
    if !is_inkling2_row(entry) {
        return Err(StoreError::Io(format!(
            "{} is a {} row, not an Inkling 2 row",
            entry.alias, entry.family
        )));
    }
    let repo = RepoRef::new(&entry.source.repo, &entry.source.revision);
    let tokenizer = entry
        .sidecars
        .repo
        .as_ref()
        .map(|r| RepoRef::new(r, entry.sidecar_revision()));
    let source = detect_inkling2_repo(
        client,
        &repo,
        entry.source.file.as_deref(),
        tokenizer.as_ref(),
    )
    .map_err(StoreError::Io)?
    .ok_or_else(|| {
        StoreError::Io(format!(
            "{repo}: the catalog row {} no longer resolves to an Inkling 2 source",
            entry.alias
        ))
    })?;
    if pinned_to_other_commit(&entry.source.revision, &source.repo.revision) {
        return Err(StoreError::Io(format!(
            "{repo}: the Hub answered for commit {}, not the row's pinned {}",
            source.repo.revision, entry.source.revision
        )));
    }
    let mut progress = progress;
    if let Some(mtp) = &entry.mtp {
        // The row names the repository its DSpark drafter comes from (an MLX
        // conversion drops `mtp.safetensors`). Fetched into the download
        // folder's `dspark/`, where the intake copies it from; the folder is
        // removed with the rest of the download once the install is
        // recorded.
        let staging_parent = store.staging_parent_for(&target.root)?;
        let into = download_dir(&staging_parent, &source).join("dspark");
        fetch_drafter(
            client,
            &RepoRef::new(&mtp.repo, &mtp.revision),
            ["mtp.safetensors", "config.json"],
            &into,
            cancel,
        )
        .map_err(|e| store.install_failure(&target.root, &target.root, &target.library_name, e))?;
        progress(&format!(
            "fetched the DSpark drafter from {}@{}",
            mtp.repo, mtp.revision
        ));
    }
    if let Some(draft) = &entry.dflash2 {
        // As above, into the `dflash2/` the intake copies from.
        let staging_parent = store.staging_parent_for(&target.root)?;
        let into = download_dir(&staging_parent, &source).join("dflash2");
        fetch_drafter(
            client,
            &RepoRef::new(&draft.repo, &draft.revision),
            ["model.safetensors", "config.json"],
            &into,
            cancel,
        )
        .map_err(|e| store.install_failure(&target.root, &target.root, &target.library_name, e))?;
        progress(&format!(
            "fetched the DFlash2 drafter from {}@{}",
            draft.repo, draft.revision
        ));
    }
    let installed = install_inkling2_repo(
        &source,
        &entry.alias,
        target,
        client,
        progress,
        on_bytes,
        cancel,
        store,
        env,
    )?;
    let mut model = installed.model;
    model.status = entry.status.as_str().to_string();
    let model = crate::record_into(store, &model, target, env)?;
    Ok(Installed {
        model,
        arch: installed.arch,
    })
}

/// Downloads a drafter repository's weights and the `config.json` the runtime
/// reads (`names`) into `into`, at the pinned commit.
fn fetch_drafter(
    client: &Client,
    repo: &RepoRef,
    names: [&str; 2],
    into: &Path,
    cancel: Option<&CancelFlag>,
) -> Result<(), String> {
    let (pinned, listing) = pinned_listing(client, repo)?;
    if pinned_to_other_commit(&repo.revision, &pinned.revision) {
        return Err(format!(
            "{repo}: the Hub answered for commit {}, not the drafter's pinned {}",
            pinned.revision, repo.revision
        ));
    }
    std::fs::create_dir_all(into).map_err(|e| format!("creating {}: {e}", into.display()))?;
    let range_cache = crate::stream::immutable_download_cache(into, &pinned);
    for name in names {
        let size = listing
            .iter()
            .find(|(n, _)| n == name)
            .ok_or_else(|| format!("{pinned} has no {name}; it is not a drafter source"))?
            .1;
        fetch_file(
            client,
            &pinned,
            name,
            size,
            &into.join(name),
            range_cache.as_deref(),
            &mut |_| {},
            cancel,
        )?;
    }
    Ok(())
}

/// A row pinned to a full commit id must install that commit; a row that
/// floats (`main`) takes whatever the Hub reports.
fn pinned_to_other_commit(row_revision: &str, got: &str) -> bool {
    let pinned = row_revision.len() == 40 && row_revision.bytes().all(|b| b.is_ascii_hexdigit());
    pinned && !row_revision.eq_ignore_ascii_case(got)
}

/// The local file name of a repository path: its last component, since a
/// GGUF may sit in a quantization subfolder.
fn local_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo() -> RepoRef {
        RepoRef::new("o/r", "0123456789abcdef0123456789abcdef01234567")
    }

    #[test]
    fn a_lamq_release_downloads_its_top_level_data_files_only() {
        let listing = vec![
            ("README.md".to_string(), Some(10)),
            (".gitattributes".to_string(), Some(1)),
            ("manifest.json".to_string(), Some(100)),
            ("layer_00_packed.npz".to_string(), Some(1000)),
            ("steering.safetensors".to_string(), Some(50)),
            ("chat_template.jinja".to_string(), Some(5)),
            ("images/plot.png".to_string(), Some(7)),
            ("extra/layer_00_packed.npz".to_string(), Some(9)),
        ];
        let files = lamq_release_files(&repo(), &listing).unwrap();
        let names: Vec<&str> = files.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            [
                "chat_template.jinja",
                "layer_00_packed.npz",
                "manifest.json",
                "steering.safetensors"
            ]
        );
        assert_eq!(files.iter().map(|f| f.1).sum::<u64>(), 1155);
        let no_size = vec![("layer_00_packed.npz".to_string(), None)];
        assert!(lamq_release_files(&repo(), &no_size).is_err());
    }

    #[test]
    fn tokenizer_files_need_tokenizer_json() {
        let listing = vec![
            ("tokenizer.json".to_string(), Some(1)),
            ("chat_template.jinja".to_string(), Some(1)),
            ("model.safetensors".to_string(), Some(1)),
        ];
        assert_eq!(
            tokenizer_files_in(&repo(), &listing).unwrap(),
            ["tokenizer.json", "chat_template.jinja"]
        );
        assert!(tokenizer_files_in(&repo(), &listing[1..]).is_err());
    }

    #[test]
    fn only_the_inkling_arch_strings_take_this_route() {
        assert!(is_inkling2_gguf_arch(Some("inkling2_loop")));
        assert!(is_inkling2_gguf_arch(Some("inkling2")));
        assert!(!is_inkling2_gguf_arch(Some("inkling")));
        assert!(!is_inkling2_gguf_arch(Some("llama")));
        assert!(!is_inkling2_gguf_arch(None));
    }

    #[test]
    fn a_pinned_row_refuses_any_other_commit_and_a_floating_one_takes_any() {
        let pin = "9af870917c5f3a2e26c6a305f82f27b4d350e9e1";
        assert!(!pinned_to_other_commit(pin, pin));
        assert!(!pinned_to_other_commit(pin, &pin.to_uppercase()));
        assert!(pinned_to_other_commit(pin, &"0".repeat(40)));
        assert!(!pinned_to_other_commit("main", &"0".repeat(40)));
    }

    #[test]
    fn the_download_folder_is_hidden_and_names_repo_and_commit() {
        let src = Inkling2RepoSource {
            kind: Inkling2RepoKind::Lamq,
            repo: RepoRef::new("a/b", "c0ffee"),
            files: Vec::new(),
            architecture: String::new(),
            tokenizer_repo: RepoRef::new("a/b", "c0ffee"),
            tokenizer_files: Vec::new(),
        };
        assert_eq!(
            download_dir(Path::new("/s"), &src),
            Path::new("/s/.inkling2-download-a_b_c0ffee")
        );
        assert_eq!(local_name("UD-Q2/m.gguf"), "m.gguf");
        assert_eq!(local_name("m.gguf"), "m.gguf");
    }

    #[test]
    fn checkpoint_sources_probe_as_mlx_kind_and_size_their_install_at_the_source() {
        for (kind, label) in [
            (Inkling2RepoKind::Mlx, "MLX conversion"),
            (Inkling2RepoKind::Hf, "checkpoint"),
        ] {
            let src = Inkling2RepoSource {
                kind,
                repo: repo(),
                files: vec![
                    ("config.json".to_string(), 1),
                    ("tokenizer.json".to_string(), 1),
                    ("model.safetensors".to_string(), 10 << 30),
                ],
                architecture: "inkling2_loop".to_string(),
                tokenizer_repo: repo(),
                tokenizer_files: Vec::new(),
            };
            let r = probe_report(&src);
            assert!(r.verdict.is_runnable());
            assert_eq!(r.kind, crate::SourceKind::Mlx);
            assert_eq!(r.file, None);
            assert!(r.sidecars_present.iter().any(|n| n == "tokenizer.json"));
            assert!(r.warnings[0].contains(label), "{:?}", r.warnings);
            // About the source's size plus 1 GiB, not the packed 1.5x.
            let need = inkling2_install_bytes(kind, src.download_bytes());
            assert_eq!(need, src.download_bytes() + (1 << 30));
        }
        let packed = inkling2_install_bytes(Inkling2RepoKind::Lamq, 10 << 30);
        assert_eq!(packed, (15 << 30) + (1 << 30));
    }

    #[test]
    fn the_probe_answer_is_runnable_inkling_and_names_the_download() {
        let src = Inkling2RepoSource {
            kind: Inkling2RepoKind::Gguf,
            repo: repo(),
            files: vec![("m.gguf".to_string(), 3 << 30)],
            architecture: "inkling2_loop".to_string(),
            tokenizer_repo: RepoRef::new(INKLING2_TOKENIZER_REPO, "f".repeat(40)),
            tokenizer_files: vec!["tokenizer.json".into(), "chat_template.jinja".into()],
        };
        let r = probe_report(&src);
        assert!(r.verdict.is_runnable());
        assert_eq!(r.family, Some(model_io::ModelFamily::Inkling2Loop));
        assert_eq!(r.kind, crate::SourceKind::Gguf);
        assert_eq!(r.file.as_deref(), Some("m.gguf"));
        assert_eq!(r.download_bytes, Some(3 << 30));
        assert!(r.chat_template.is_some());
        assert!(r.warnings[0].contains("downloading the whole GGUF"));
    }
}
