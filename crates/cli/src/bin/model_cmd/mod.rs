//! The eight subcommands, each a print of something `turbospark-catalog`
//! computed.
//!
//! **Nothing here decides anything.** The catalog crate resolves rows, the
//! probe reaches a verdict and the install driver runs the walk; this module
//! chooses column widths. That split is the same one `turbospark-check` has
//! with `turbospark-invocation`, and it is what lets the verdict logic be
//! tested without a terminal.

mod audio;
mod auth;
mod destination;
mod drama3;
mod image_source;
mod library;
mod progress;
mod render;
mod transfer;

use catalog::{Catalog, Client, InstallKind, InstallPlan, RepoRef, Store, Verdict};
use destination::Destination;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::{Error, Options};

/// Set by Ctrl-C once [`install_sigint_handler`] runs. An audio pull and a
/// move each clear it before they start and stop when it is set.
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

pub use audio::{list_audio, pull_audio};
pub use auth::auth;
pub use drama3::pull_drama3;
pub use library::library;
pub use render::human_bytes;
pub use transfer::move_model;

/// Curated rows, marking which are installed.
/// The bundled speech table. Speech entries ride their own embedded table
/// (`speech_models.json`), pinned like the decoder rows.
pub fn list_speech(store: &Store) {
    let entries = catalog::embedded_speech_entries().expect("embedded speech catalog");
    render::speech_list_header(entries.len());
    println!();
    for entry in &entries {
        render::speech_entry(entry, store);
        println!();
    }
}

fn install_sigint_handler() {
    #[cfg(unix)]
    unsafe {
        libc::signal(
            libc::SIGINT,
            handle_sigint as *const () as libc::sighandler_t,
        );
    }
}

extern "C" fn handle_sigint(_: libc::c_int) {
    INTERRUPTED.store(true, Ordering::Release);
}

/// The curated rows, marking the ones installed, then every install with
/// its library and whether it can be opened now (10.2).
pub fn list(catalog: &Catalog, store: &Store, filter: Option<&str>) {
    let installed = store.installed();
    let rows: Vec<_> = match filter {
        Some(needle) => catalog.find(needle),
        None => catalog.entries().collect(),
    };
    let installs = installs_matching(store, filter);
    if rows.is_empty() && installs.is_empty() {
        println!("no models match");
        return;
    }
    if !rows.is_empty() {
        catalog_table(&rows, &installed);
    }
    if !installs.is_empty() {
        installs_table(&installs);
    }
}

/// Every recorded install whose alias contains `filter`, ignoring case.
/// Listing never fails (3.4): an environment that cannot be read leaves the
/// installs out with a warning, and the catalog is still listed.
fn installs_matching(store: &Store, filter: Option<&str>) -> Vec<catalog::InstalledEntry> {
    let env = match catalog::library::LocationEnv::system(store.root()) {
        Ok(env) => env,
        Err(error) => {
            eprintln!("warning: cannot list installs by library: {error}");
            return Vec::new();
        }
    };
    let needle = filter.map(str::to_lowercase);
    store
        .installed_with_availability(&env)
        .into_iter()
        .filter(|entry| {
            needle
                .as_deref()
                .is_none_or(|needle| entry.row.alias.to_lowercase().contains(needle))
        })
        .collect()
}

/// Installs with LOCATION (the library's name, "This Mac", or "custom
/// location") and STATUS. Names and aliases can come from a drive, so they
/// are escaped and capped as `library list` does.
fn installs_table(installs: &[catalog::InstalledEntry]) {
    use library::{clean, NAME_LIMIT};
    let rows: Vec<[String; 4]> = installs
        .iter()
        .map(|entry| {
            let kind = entry
                .row
                .kind
                .clone()
                .unwrap_or_else(|| entry.row.effective_modality().as_str().to_string());
            let status = match entry.availability {
                catalog::library::Availability::Available => "available",
                catalog::library::Availability::NotConnected { .. } => "not connected",
                catalog::library::Availability::Missing => "missing",
            };
            [
                clean(&entry.row.alias, NAME_LIMIT),
                clean(&kind, NAME_LIMIT),
                clean(&entry.library_name, NAME_LIMIT),
                status.to_string(),
            ]
        })
        .collect();
    let width = |column: usize, header: &str| {
        rows.iter()
            .map(|row| row[column].chars().count())
            .chain(std::iter::once(header.len()))
            .max()
            .unwrap_or(0)
    };
    let (alias_width, kind_width, location_width) =
        (width(0, "ALIAS"), width(1, "KIND"), width(2, "LOCATION"));
    println!("\ninstalled:");
    println!(
        "  {:<alias_width$}  {:<kind_width$}  {:<location_width$}  STATUS",
        "ALIAS", "KIND", "LOCATION"
    );
    for [alias, kind, location, status] in &rows {
        println!(
            "  {alias:<alias_width$}  {kind:<kind_width$}  {location:<location_width$}  {status}"
        );
    }
}

fn catalog_table(
    rows: &[&catalog::CatalogEntry],
    installed: &std::collections::BTreeMap<String, catalog::InstalledModel>,
) {
    let width = rows.iter().map(|e| e.alias.len()).max().unwrap_or(5).max(5);
    println!(
        "  {:<width$}  {:<9}  {:<12}  {:>9}  MODEL",
        "ALIAS",
        "STATUS",
        "KIND",
        "DOWNLOAD",
        width = width
    );
    for entry in rows {
        let mark = if installed.contains_key(&entry.alias) {
            "*"
        } else {
            " "
        };
        println!(
            "{mark} {:<width$}  {:<9}  {:<12}  {:>9}  {}",
            entry.alias,
            entry.status.as_str(),
            entry.kind.as_str(),
            human_bytes(entry.download_bytes),
            entry.name,
            width = width
        );
    }
    println!(
        "\n* installed. {} row(s). `info <alias>` for the full record, \
         `probe <repo>` for anything not listed.",
        rows.len()
    );
    println!(
        "status: verified = has a frozen gate row in docs/BENCHMARKS.md; \
         runs = generated coherent text here; caveat = read the notes first."
    );
}

/// One row in full.
pub fn info(catalog: &Catalog, store: &Store, alias: &str) -> Result<(), Error> {
    let entry = catalog
        .get(alias)
        .ok_or_else(|| Error::Failed(unknown_alias(catalog, alias)))?;
    render::entry(entry, catalog.is_user_row(alias), store);
    Ok(())
}

/// Header-only verdict for an arbitrary repository.
pub fn probe(
    client: &Client,
    repo: &RepoRef,
    file: Option<&str>,
    sidecars: Option<&RepoRef>,
) -> Result<(), Error> {
    let pb = progress::spinner(format!("probing {repo}..."));
    let report_res = catalog::probe(client, repo, file, sidecars);
    pb.finish_and_clear();
    let report = report_res.map_err(Error::Failed)?;
    render::report(&report);
    match &report.verdict {
        Verdict::Runnable => Ok(()),
        // A refusal is a successful probe of an unusable model, and the exit
        // code says the model is unusable rather than that the probe broke.
        // The distinction matters for scripting: `probe X && pull X`.
        Verdict::Refused(why) => Err(Error::Failed(format!("would not run here: {why}"))),
    }
}

/// Print an install directory, or fail so `$(...)` does not expand to
/// nothing and silently produce a `--model ''`.
pub fn path(store: &Store, alias: &str) -> Result<(), Error> {
    // A text alias wins; an audio alias answers when no text install does.
    match store
        .resolve(alias)
        .or_else(|| audio::path_audio(store, alias))
    {
        Some(path) => {
            println!("{}", path.display());
            Ok(())
        }
        None => Err(Error::Failed(format!(
            "{alias} is not installed. `turbospark-model pull {alias}` (or \
             `pull-audio {alias}` for an audio model) first."
        ))),
    }
}

/// Delete an install directory.
pub fn remove(store: &Store, alias: &str, yes: bool) -> Result<(), Error> {
    let Some(path) = store.resolve(alias) else {
        // Not a text install: an audio alias has its own ownership-checked delete.
        return audio::remove_audio(store, alias, yes);
    };
    let bytes = catalog::directory_bytes(&path);
    if !yes {
        // The re-download cost is the thing a user actually wants to weigh,
        // so it is in the prompt rather than left to be remembered.
        eprintln!(
            "delete {} ({})? This cannot be undone and re-installing means \
             streaming the checkpoint again.",
            path.display(),
            human_bytes(bytes)
        );
        eprint!("type the alias to confirm: ");
        use std::io::Write;
        let _ = std::io::stderr().flush();
        let mut typed = String::new();
        std::io::stdin()
            .read_line(&mut typed)
            .map_err(|e| Error::Failed(format!("reading confirmation: {e}")))?;
        if typed.trim() != alias {
            return Err(Error::Failed("not confirmed, nothing deleted".to_string()));
        }
    }
    let pb = progress::spinner(format!("removing {}...", path.display()));
    let rm_res = std::fs::remove_dir_all(&path);
    pb.finish_and_clear();
    rm_res.map_err(|e| Error::Failed(format!("removing {}: {e}", path.display())))?;
    store.forget(alias).map_err(Error::Failed)?;
    println!("removed {} ({})", path.display(), human_bytes(bytes));
    Ok(())
}

/// Where a text or vision pull of `alias` lands: `--out`, `--library`, or
/// the default library, settled before any network request.
///
/// Through the same resolver a library install uses either way, so an alias
/// another library holds is refused before anything is downloaded. That
/// includes `--out`, whose row `record` would otherwise put over the other
/// library's, orphaning its folder while its marker keeps listing it.
fn text_destination(
    store: &Store,
    kind: InstallKind,
    alias: &str,
    options: &Options,
) -> Result<Destination, Error> {
    let destination = destination::destination(store, kind, alias, options)?;
    if let Destination::Out(_) = destination {
        this_mac_destination(store, kind, alias)?;
    }
    Ok(destination)
}

/// The install folder in This Mac for `alias`, through the same resolver a
/// library install uses.
fn this_mac_destination(
    store: &Store,
    kind: catalog::InstallKind,
    alias: &str,
) -> Result<std::path::PathBuf, Error> {
    let env = catalog::library::LocationEnv::system(store.root()).map_err(Error::Failed)?;
    catalog::install_destination(store, None, kind, alias, &env)
        .map(|target| target.dir)
        .map_err(|error| Error::Failed(error.to_string()))
}

/// Walks `plan` into `destination` and records it. Into a library through
/// `catalog::install_text_into`, which records the row with its library and
/// in the library's marker. Into This Mac or `--out` exactly as before
/// libraries.
fn walk_and_record(
    plan: &InstallPlan,
    destination: &Destination,
    store: &Store,
    client: &Client,
    progress: impl FnMut(&str),
    byte_progress: catalog::ByteProgressCallback,
) -> Result<catalog::Installed, Error> {
    // The CLI has no cancel surface: a pull runs until it finishes.
    if let Destination::Target(target, env) = destination {
        if target.library_id.is_some() {
            return catalog::install_text_into(
                plan,
                target,
                client,
                progress,
                Some(byte_progress),
                None,
                store,
                env,
            )
            .map_err(destination::failure);
        }
    }
    let installed = catalog::install_with_byte_progress(
        plan,
        destination.dir(),
        client,
        progress,
        Some(byte_progress),
        None,
    )
    .map_err(Error::Failed)?;
    catalog::record(store, &installed).map_err(Error::Failed)?;
    Ok(installed)
}

/// `pull`'s two argument forms, read without any network request, so the
/// destination is settled before the probe a `--repo` pull makes.
enum PullSource {
    /// A curated row's plan.
    Row(InstallPlan),
    /// `--repo`, planned from its probe.
    Repo {
        alias: String,
        weights: RepoRef,
        sidecars: RepoRef,
    },
}

impl PullSource {
    fn alias(&self) -> &str {
        match self {
            Self::Row(plan) => &plan.alias,
            Self::Repo { alias, .. } => alias,
        }
    }

    /// The plan, probing a `--repo` source first.
    fn plan(self, client: &Client, options: &Options) -> Result<InstallPlan, Error> {
        match self {
            Self::Row(plan) => Ok(plan),
            Self::Repo {
                alias,
                weights,
                sidecars,
            } => {
                let report =
                    catalog::probe(client, &weights, options.file.as_deref(), Some(&sidecars))
                        .map_err(Error::Failed)?;
                Ok(InstallPlan::from_probe(&alias, &report, sidecars))
            }
        }
    }
}

/// Install a curated row, or any repository the probe accepts.
pub fn pull(
    catalog: &Catalog,
    store: &Store,
    client: &Client,
    positionals: &[String],
    options: &Options,
) -> Result<(), Error> {
    let source = pull_source(catalog, positionals, options)?;
    // Before the probe, a `--repo` pull's first network request (10.4).
    let destination = text_destination(store, InstallKind::Text, source.alias(), options)?;
    let mut plan = source.plan(client, options)?;
    if let Some(alias) = &options.reuse_trunk_from {
        plan.reuse_trunk_from = Some(resolve_reuse_trunk_from(store, &plan, alias)?);
    }
    let dir = destination.dir();
    if dir.join("manifest.json").is_file() {
        return Err(Error::Failed(format!(
            "{} already holds an install. Remove it first, or pass --out.",
            dir.display()
        )));
    }

    let pb = progress::byte_progress_bar(plan.install_bytes);
    let pb_for_msg = pb.clone();
    let mut progress = move |stage: &str| {
        pb_for_msg.set_message(stage.to_string());
        pb_for_msg.println(format!("[pull] {stage}"));
    };
    let report = catalog::gate(client, &plan, options.force, &mut progress).map_err(|e| {
        pb.finish_and_clear();
        Error::Failed(e)
    })?;
    pb.suspend(|| {
        render::report(&report);
    });

    let pb_for_bytes = pb.clone();
    let byte_callback = std::sync::Arc::new(move |bytes: u64| {
        pb_for_bytes.inc(bytes);
    });

    let installed = walk_and_record(
        &plan,
        &destination,
        store,
        client,
        &mut progress,
        byte_callback,
    )
    .inspect_err(|_| pb.finish_and_clear())?;
    pb.finish_and_clear();

    println!(
        "\ninstalled {} ({}) to {}",
        installed.model.alias,
        human_bytes(installed.model.install_bytes),
        installed.model.path.display()
    );
    println!(
        "run it: turbospark-check --model {} --messages-file /tmp/p.json",
        installed.model.alias
    );
    Ok(())
}

/// Package a pinned local Diffusers image export. This deliberately bypasses
/// the text catalog and its Mlx/GGUF probe: an image install has five
/// components, a different manifest, and a different runtime capability.
pub fn pull_image(
    _catalog: &Catalog,
    store: &Store,
    client: &Client,
    positionals: &[String],
    options: &Options,
) -> Result<(), Error> {
    let explicit_remote_repo = positionals
        .iter()
        .find_map(|value| value.strip_prefix("--repo="));
    if positionals
        .iter()
        .any(|value| !value.starts_with("--repo="))
        || positionals
            .iter()
            .filter(|value| value.starts_with("--repo="))
            .count()
            > 1
    {
        return Err(Error::Usage(
            "pull-image takes --alias <NAME> and optional --repo <REPO>@<REV>".to_string(),
        ));
    }
    let alias = options
        .alias
        .as_deref()
        .ok_or_else(|| Error::Usage("pull-image needs --alias <NAME>".to_string()))?;
    if alias.is_empty()
        || !alias
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(Error::Usage(
            "--alias must contain only ASCII letters, digits, '.', '_' or '-'".to_string(),
        ));
    }
    // Where the install lands, settled before any download (10.4). Into a
    // library the source and the build stage on its drive and publish with
    // one rename, so a partial build never shows there. This Mac and `--out`
    // build in place, as before. Image installs are not recorded.
    let destination = destination::destination(store, InstallKind::Image, alias, options)?;
    let library = match &destination {
        Destination::Target(target, _) if target.library_id.is_some() => {
            // Publishing never replaces an install, so refuse before the
            // download rather than after it.
            if std::fs::symlink_metadata(&target.dir).is_ok() {
                return Err(Error::Failed(format!(
                    "{} already holds an image install; remove it first",
                    target.dir.display()
                )));
            }
            let staging = store.staging_parent_for(&target.root).map_err(|error| {
                destination::failure(store.install_failure(
                    &target.root,
                    &target.root,
                    &target.library_name,
                    error.to_string(),
                ))
            })?;
            Some((target, staging))
        }
        _ => None,
    };
    // Classified by the store's one rule, so a library step that failed
    // because the drive went away reads as a disconnect.
    let failed = |error: String| match &library {
        Some((target, staging)) => destination::failure(store.install_failure(
            staging,
            &target.root,
            &target.library_name,
            error,
        )),
        None => Error::Failed(error),
    };
    let output = match &library {
        Some((_, staging)) => staging.join(format!("{alias}.gturbo")),
        None => destination.dir().to_path_buf(),
    };
    let catalog_repo = if explicit_remote_repo.is_none() && options.source.is_none() {
        catalog::ImageCatalog::embedded()
            .map_err(Error::Failed)?
            .get(alias)
            .map(|entry| format!("{}@{}", entry.model_id, entry.revision))
    } else {
        None
    };
    let remote_repo = explicit_remote_repo.or(catalog_repo.as_deref());
    let (source, model_id, model_revision, cleanup_source) = if let Some(repo_text) = remote_repo {
        if options.source.is_some()
            || options.model_id.is_some()
            || options.model_revision.is_some()
        {
            return Err(Error::Usage(
                "pull-image --repo or a catalog image cannot be combined with --source, --model-id or --model-revision"
                    .to_string(),
            ));
        }
        let repo = RepoRef::parse(repo_text).map_err(Error::Usage)?;
        if repo.revision == "main" {
            return Err(Error::Usage(
                "pull-image --repo needs an immutable @<40-hex-commit> revision".to_string(),
            ));
        }
        let output_parent = output
            .parent()
            .map(std::path::Path::to_path_buf)
            .unwrap_or_else(|| {
                store
                    .image_install_path(alias)
                    .parent()
                    .unwrap()
                    .to_path_buf()
            });
        // A library's staging folder exists already. Not recreated, which
        // with its drive gone would rebuild it under the mount point.
        if library.is_none() {
            std::fs::create_dir_all(&output_parent).map_err(|error| {
                Error::Failed(format!(
                    "creating image install parent {}: {error}",
                    output_parent.display()
                ))
            })?;
        }
        let source = output_parent.join(format!(".{alias}.image-source-{}", std::process::id()));
        // A drive's staging folder is untrusted: whatever already sits at
        // this name, a link included, is removed without being followed, so
        // the download cannot be redirected off the drive. Neither call
        // follows a link; one of them fails harmlessly.
        if library.is_some() {
            let _ = std::fs::remove_dir_all(&source);
            let _ = std::fs::remove_file(&source);
        }
        let downloaded = image_source::materialize(client, &repo, &source).map_err(|error| {
            let _ = std::fs::remove_dir_all(&source);
            failed(error)
        })?;
        println!("downloaded {} image source bytes", downloaded);
        (source.clone(), repo.repo, repo.revision, Some(source))
    } else {
        if !positionals.is_empty() {
            return Err(Error::Usage(
                "pull-image takes --alias <NAME>, not a positional alias".to_string(),
            ));
        }
        let source = options.source.as_deref().ok_or_else(|| {
            Error::Usage("pull-image needs --source <DIR> or --repo <REPO>@<REV>".to_string())
        })?;
        let model_id = options
            .model_id
            .as_deref()
            .ok_or_else(|| Error::Usage("pull-image needs --model-id <ID>".to_string()))?;
        let model_revision = options
            .model_revision
            .as_deref()
            .ok_or_else(|| Error::Usage("pull-image needs --model-revision <REV>".to_string()))?;
        if model_revision.len() != 40
            || !model_revision.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(Error::Usage(
                "--model-revision must be an immutable 40-hex commit revision".to_string(),
            ));
        }
        (
            std::path::PathBuf::from(source),
            model_id.to_string(),
            model_revision.to_string(),
            None,
        )
    };
    match (&library, output.parent()) {
        // A build an earlier attempt left unpublished in staging. A link
        // there is removed, not followed.
        (Some(_), _) => {
            let _ = std::fs::remove_dir_all(&output);
        }
        (None, Some(parent)) => {
            std::fs::create_dir_all(parent).map_err(|error| {
                Error::Failed(format!(
                    "creating image install parent {}: {error}",
                    parent.display()
                ))
            })?;
        }
        (None, None) => {}
    }
    let report = image::build_image_install(&image::ImageInstallSpec {
        source_root: source,
        output_root: output,
        model_id: model_id.clone(),
        model_revision: model_revision.clone(),
    });
    let _ = cleanup_source.as_deref().map(std::fs::remove_dir_all);
    let report = report.map_err(failed)?;
    let path = match &library {
        Some((target, _)) => {
            if let Err(error) =
                store.publish_staged(&report.output_root, &target.dir, &target.library_name)
            {
                let _ = std::fs::remove_dir_all(&report.output_root);
                return Err(destination::failure(error));
            }
            target.dir.clone()
        }
        None => report
            .output_root
            .canonicalize()
            .unwrap_or_else(|_| report.output_root.clone()),
    };
    println!(
        "installed image {} ({} bytes) to {}",
        alias,
        report.total_bytes,
        path.display()
    );
    Ok(())
}

/// Install a curated vision-tower row, or an ad-hoc `--repo` naming one
/// directly (vision memory sidecar, part A5). Mirrors [`pull`], with two
/// differences: the default install directory carries the `-vision` suffix
/// (`Store::vision_install_path`), and the printed hint pairs the tower with
/// a trunk (`--vision-sidecar`) rather than opening it as a session.
pub fn pull_vision(
    catalog: &Catalog,
    store: &Store,
    client: &Client,
    positionals: &[String],
    options: &Options,
) -> Result<(), Error> {
    let plan = resolve_vision_plan(catalog, positionals, options)?;
    let destination = text_destination(store, InstallKind::VisionTower, &plan.alias, options)?;
    let dir = destination.dir();
    if dir.join("manifest.json").is_file() {
        return Err(Error::Failed(format!(
            "{} already holds an install. Remove it first, or pass --out.",
            dir.display()
        )));
    }

    let pb = progress::byte_progress_bar(plan.install_bytes);
    let pb_for_msg = pb.clone();
    let mut progress = move |stage: &str| {
        pb_for_msg.set_message(stage.to_string());
        pb_for_msg.println(format!("[pull-vision] {stage}"));
    };
    let report = catalog::gate(client, &plan, options.force, &mut progress).map_err(|e| {
        pb.finish_and_clear();
        Error::Failed(e)
    })?;
    pb.suspend(|| {
        render::report(&report);
    });

    let pb_for_bytes = pb.clone();
    let byte_callback = std::sync::Arc::new(move |bytes: u64| {
        pb_for_bytes.inc(bytes);
    });

    let installed = walk_and_record(
        &plan,
        &destination,
        store,
        client,
        &mut progress,
        byte_callback,
    )
    .inspect_err(|_| pb.finish_and_clear())?;
    pb.finish_and_clear();

    println!(
        "\ninstalled vision tower {} ({}) to {}",
        installed.model.alias,
        human_bytes(installed.model.install_bytes),
        installed.model.path.display()
    );
    println!(
        "pair it with a trunk: turbospark-check --model <trunk-alias-or-path> \
         --vision-sidecar {}",
        installed.model.path.display()
    );
    Ok(())
}

/// Turn `pull-vision`'s two argument forms into one plan.
///
/// Same shape as [`resolve_plan`], with two differences: an alias must name
/// a [`catalog::EntryKind::VisionTower`] row (a model row is refused by
/// name, pointing at `pull` instead), and the `--repo` form builds an
/// [`InstallPlan`] directly rather than probing first --
/// [`catalog::install::gate`]'s bypass is what makes that safe.
fn resolve_vision_plan(
    catalog: &Catalog,
    positionals: &[String],
    options: &Options,
) -> Result<InstallPlan, Error> {
    let repo_flag = positionals
        .iter()
        .find_map(|p| p.strip_prefix("--repo=").map(str::to_string));
    let aliases: Vec<&String> = positionals
        .iter()
        .filter(|p| !p.starts_with("--repo="))
        .collect();

    match (repo_flag, aliases.len()) {
        (Some(_), n) if n > 0 => Err(Error::Usage(
            "pull-vision takes either an alias or --repo, not both".to_string(),
        )),
        (Some(repo), _) => {
            let alias = options.alias.clone().ok_or_else(|| {
                Error::Usage(
                    "--repo needs --alias <name>, which is what the install will be \
                     called locally"
                        .to_string(),
                )
            })?;
            let weights = RepoRef::parse(&repo).map_err(Error::Usage)?;
            Ok(InstallPlan::for_vision_tower(
                &alias,
                weights,
                options.file.clone(),
            ))
        }
        (None, 1) => {
            let alias = aliases[0];
            let entry = catalog
                .get(alias)
                .ok_or_else(|| Error::Failed(unknown_alias(catalog, alias)))?;
            if entry.kind != catalog::EntryKind::VisionTower {
                return Err(Error::Failed(format!(
                    "{alias} is a model row, not a vision-tower row. \
                     `turbospark-model pull {alias}` installs it."
                )));
            }
            Ok(InstallPlan::from_entry(entry))
        }
        (None, 0) => Err(Error::Usage(
            "pull-vision needs <ALIAS> or --repo <REPO> --alias <NAME>".to_string(),
        )),
        (None, n) => Err(Error::Usage(format!(
            "pull-vision takes one alias, got {n}"
        ))),
    }
}

/// Read `pull`'s two argument forms, without the probe a `--repo` pull
/// plans from.
fn pull_source(
    catalog: &Catalog,
    positionals: &[String],
    options: &Options,
) -> Result<PullSource, Error> {
    let repo_flag = positionals
        .iter()
        .find_map(|p| p.strip_prefix("--repo=").map(str::to_string));
    let aliases: Vec<&String> = positionals
        .iter()
        .filter(|p| !p.starts_with("--repo="))
        .collect();

    match (repo_flag, aliases.len()) {
        (Some(_), n) if n > 0 => Err(Error::Usage(
            "pull takes either an alias or --repo, not both".to_string(),
        )),
        (Some(repo), _) => {
            let alias = options.alias.clone().ok_or_else(|| {
                Error::Usage(
                    "--repo needs --alias <name>, which is what the install will be \
                     called locally"
                        .to_string(),
                )
            })?;
            let weights = RepoRef::parse(&repo).map_err(Error::Usage)?;
            let sidecars = match &options.sidecar_repo {
                Some(text) => RepoRef::parse(text).map_err(Error::Usage)?,
                None => weights.clone(),
            };
            Ok(PullSource::Repo {
                alias,
                weights,
                sidecars,
            })
        }
        (None, 1) => {
            let alias = aliases[0];
            catalog
                .get(alias)
                .map(|entry| PullSource::Row(InstallPlan::from_entry(entry)))
                .ok_or_else(|| Error::Failed(unknown_alias(catalog, alias)))
        }
        (None, 0) => Err(Error::Usage(
            "pull needs <ALIAS> or --repo <REPO> --alias <NAME>".to_string(),
        )),
        (None, n) => Err(Error::Usage(format!("pull takes one alias, got {n}"))),
    }
}

/// Validates `--reuse-trunk-from <alias>` before letting the walk trust the
/// named install's bytes: it must actually exist, and it must have come from
/// the EXACT repository and revision `plan` is about to pull, or the reused
/// resident entries are bytes from a different checkpoint entirely.
fn resolve_reuse_trunk_from(
    store: &Store,
    plan: &InstallPlan,
    alias: &str,
) -> Result<std::path::PathBuf, Error> {
    if plan.mtp.is_none() {
        return Err(Error::Usage(format!(
            "--reuse-trunk-from only applies to a row naming an mtp source; {} names none",
            plan.alias
        )));
    }
    let installed = store.installed();
    let record = installed.get(alias).ok_or_else(|| {
        Error::Failed(format!(
            "--reuse-trunk-from {alias:?}: no install by that name. \
             `turbospark-model pull {alias}` first, or check `turbospark-model list`."
        ))
    })?;
    if record.repo != plan.weights.repo || record.revision != plan.weights.revision {
        return Err(Error::Failed(format!(
            "--reuse-trunk-from {alias:?} came from {}@{}, but {} needs {}; \
             refusing to graft a head onto a different checkpoint's trunk",
            record.repo, record.revision, plan.alias, plan.weights
        )));
    }
    Ok(record.path.clone())
}

/// An unknown alias, with the nearest matches. Cheap, and it turns the most
/// common typo into a one-line fix instead of a trip to `list`.
fn unknown_alias(catalog: &Catalog, alias: &str) -> String {
    let near: Vec<&str> = catalog
        .find(alias)
        .iter()
        .map(|e| e.alias.as_str())
        .collect();
    if near.is_empty() {
        format!("no model named {alias:?}. `turbospark-model list` shows the catalog.")
    } else {
        format!(
            "no model named {alias:?}. Did you mean: {}?",
            near.join(", ")
        )
    }
}

/// What this machine should run, ranked.
///
/// Three sources of shape, cheapest first, and the output says which one each
/// row used. Offline, a curated row is described by its size and by the
/// `measured` block in `models.json` if this chip has one. `--probe` reads
/// every row's header, which is what turns the slot cache and the KV from
/// unknowns into arithmetic. `--discover` adds Hugging Face at large, filtered
/// through the same probe.
pub fn recommend(catalog: &Catalog, client: &Client, options: &Options) -> Result<(), Error> {
    let machine = machine(options);
    if machine.physical_bytes == 0 {
        return Err(Error::Failed(
            "no memory probe on this platform and no --budget given, so there is no \
             machine to fit against. Pass --budget 36GiB."
                .to_string(),
        ));
    }
    let context = options.context.unwrap_or(DEFAULT_RECOMMEND_CONTEXT);
    // This subcommand has no slot flag, so it fits against what `open()`
    // would pick here. `DiscoverOptions::default()` below carries the same
    // policy, and the two must not drift: a discovered row ranked at one slot
    // count beside a curated row ranked at another is not one table.
    const SLOT_POLICY: model_io::ExpertCacheSlots = model_io::ExpertCacheSlots::Auto;

    // A vision-tower row is not a fit candidate (no tokenizer, cannot be
    // opened as a session on its own), so it is filtered out of both arms
    // below -- `recommend_catalog` already does the same filter for the
    // offline arm, but this arm builds its own rows directly and would
    // otherwise rank a tower beside the trunks it attaches to.
    let entries: Vec<&catalog::CatalogEntry> = catalog
        .entries()
        .filter(|e| e.kind == catalog::EntryKind::Model)
        .collect();
    let mut rows: Vec<catalog::Recommendation> = if options.probe {
        let pb = progress::count_progress_bar(entries.len() as u64, "probing curated models...");
        let results = catalog::recommend_catalog_probed_with_progress(
            &entries,
            client,
            &machine,
            context,
            SLOT_POLICY,
            |done, total| {
                pb.set_length(total as u64);
                pb.set_position(done as u64);
            },
        );
        pb.finish_and_clear();
        results?
    } else {
        catalog::recommend_catalog(&entries, &machine, context, SLOT_POLICY)
    };

    if let Some(scan) = options.discover {
        let pb = progress::spinner(format!(
            "scanning the {scan} most-downloaded GGUF repositories on Hugging Face..."
        ));
        let found = catalog::discover(
            client,
            &machine,
            &catalog::DiscoverOptions {
                scan,
                context,
                ..Default::default()
            },
        );
        pb.finish_and_clear();
        rows.extend(found?);
    }
    // **Rank over the final concatenation.** The catalog helpers also return
    // their standalone rows ranked for FFI callers, but only this final pass
    // can interleave a discovered row with a curated one. Without discovery
    // it is an intentionally harmless repeat of the same deterministic sort.
    catalog::rank_recommendations(&mut rows);
    render::recommendations(&rows, &machine, context);
    Ok(())
}

/// The protocol's shared window, and what `recommend` fits against unless
/// told otherwise. Deliberately not `MaxContext::Auto`'s answer: `auto`
/// resolves per install, and this has to compare rows against ONE window or
/// the column means something different in every line.
const DEFAULT_RECOMMEND_CONTEXT: u32 = 4096;

/// Read the machine, or take `--budget` for it.
///
/// **THE CHIP COMES FROM THE METAL DEVICE NAME AND THE ORACLES TAKE IT FROM
/// `sysctl machdep.cpu.brand_string`.** Two probes for one fact, which is
/// worth stating because it looks like an oversight. On Apple silicon both
/// answer the chip's marketing name ("Apple M4 Max"), and `measured_for`
/// matches by SUBSTRING, so the two agree for every row in `models.json`.
/// The alternative was a dependency on `turbospark-bench` -- a benchmark
/// harness -- from the model-management binary, to reach one `sysctl` call.
/// If they ever disagree, the symptom is a measured row not matching, which
/// prints as `unknown` rather than as a wrong number.
///
/// **`TURBOSPARK_TEST_CHIP` overrides the probed name, for the same reason
/// `tests/model_cli.rs` gives itself a private `TURBOSPARK_HOME`: a
/// developer's real hardware must not be what makes this pass or fail.**
/// The Metal device on a virtualized CI runner reports its own name (e.g.
/// "Apple Paravirtual device"), which matches no chip any measured row in
/// `models.json` was ever taken on -- so `measured_for` correctly answers
/// `None` for every row there, and a test asserting a measured row's text
/// is asserting something only a specific developer's real chip can produce.
/// Unset in every non-test invocation, so nothing about resolving `recommend`
/// for a real user reads this variable at all.
#[cfg(target_os = "macos")]
fn machine(options: &Options) -> catalog::Machine {
    let (working_set, probed_chip) = match runtime::recommended_max_working_set() {
        Some((bytes, name)) => (Some(bytes), name),
        None => (None, String::new()),
    };
    let chip = std::env::var("TURBOSPARK_TEST_CHIP").unwrap_or(probed_chip);
    catalog::Machine {
        physical_bytes: options.budget.unwrap_or_else(runtime::physical_memory),
        working_set_bytes: working_set,
        load_guard: options.load_guard,
        chip,
    }
}

#[cfg(not(target_os = "macos"))]
fn machine(options: &Options) -> catalog::Machine {
    catalog::Machine {
        physical_bytes: options.budget.unwrap_or(0),
        working_set_bytes: None,
        load_guard: options.load_guard,
        chip: std::env::var("TURBOSPARK_TEST_CHIP").unwrap_or_default(),
    }
}
