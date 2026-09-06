//! The eight subcommands, each a print of something `turbospark-catalog`
//! computed.
//!
//! **Nothing here decides anything.** The catalog crate resolves rows, the
//! probe reaches a verdict and the install driver runs the walk; this module
//! chooses column widths. That split is the same one `turbospark-check` has
//! with `turbospark-invocation`, and it is what lets the verdict logic be
//! tested without a terminal.

mod auth;
mod progress;
mod render;

use catalog::{Catalog, Client, InstallPlan, RepoRef, Store, Verdict};

use crate::{Error, Options};

pub use auth::auth;
pub use render::human_bytes;

/// Curated rows, marking which are installed.
pub fn list(catalog: &Catalog, store: &Store, filter: Option<&str>) {
    let installed = store.installed();
    let rows: Vec<_> = match filter {
        Some(needle) => catalog.find(needle),
        None => catalog.entries().collect(),
    };
    if rows.is_empty() {
        println!("no models match");
        return;
    }
    let width = rows.iter().map(|e| e.alias.len()).max().unwrap_or(5).max(5);
    println!(
        "  {:<width$}  {:<9}  {:<12}  {:>9}  MODEL",
        "ALIAS",
        "STATUS",
        "KIND",
        "DOWNLOAD",
        width = width
    );
    for entry in &rows {
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
    match store.resolve(alias) {
        Some(path) => {
            println!("{}", path.display());
            Ok(())
        }
        None => Err(Error::Failed(format!(
            "{alias} is not installed. `turbospark-model pull {alias}` first."
        ))),
    }
}

/// Delete an install directory.
pub fn remove(store: &Store, alias: &str, yes: bool) -> Result<(), Error> {
    let Some(path) = store.resolve(alias) else {
        return Err(Error::Failed(format!("{alias} is not installed")));
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

/// Install a curated row, or any repository the probe accepts.
pub fn pull(
    catalog: &Catalog,
    store: &Store,
    client: &Client,
    positionals: &[String],
    options: &Options,
) -> Result<(), Error> {
    let mut plan = resolve_plan(catalog, client, positionals, options)?;
    if let Some(alias) = &options.reuse_trunk_from {
        plan.reuse_trunk_from = Some(resolve_reuse_trunk_from(store, &plan, alias)?);
    }
    let dir = match &options.out {
        Some(out) => std::path::PathBuf::from(out),
        None => store.install_path(&plan.alias),
    };
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

    let installed = catalog::install_with_byte_progress(
        &plan,
        &dir,
        client,
        &mut progress,
        Some(byte_callback),
    )
    .map_err(|e| {
        pb.finish_and_clear();
        Error::Failed(e)
    })?;
    catalog::record(store, &installed).map_err(Error::Failed)?;
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
    let dir = match &options.out {
        Some(out) => std::path::PathBuf::from(out),
        None => store.vision_install_path(&plan.alias),
    };
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

    let installed = catalog::install_with_byte_progress(
        &plan,
        &dir,
        client,
        &mut progress,
        Some(byte_callback),
    )
    .map_err(|e| {
        pb.finish_and_clear();
        Error::Failed(e)
    })?;
    catalog::record(store, &installed).map_err(Error::Failed)?;
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

/// Turn `pull`'s two argument forms into one plan.
fn resolve_plan(
    catalog: &Catalog,
    client: &Client,
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
            let report = catalog::probe(client, &weights, options.file.as_deref(), Some(&sidecars))
                .map_err(Error::Failed)?;
            Ok(InstallPlan::from_probe(&alias, &report, sidecars))
        }
        (None, 1) => {
            let alias = aliases[0];
            catalog
                .get(alias)
                .map(InstallPlan::from_entry)
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
        let results = entries
            .iter()
            .map(|entry| {
                pb.set_message(format!("probing {}...", entry.alias));
                // A probe failure is not a refusal: the row still has its
                // size and its evidence, and losing it entirely because a
                // header read timed out would be the worse answer.
                let report = catalog::probe_entry(client, entry).ok();
                pb.inc(1);
                catalog::from_entry(entry, &machine, context, SLOT_POLICY, report.as_ref())
            })
            .collect();
        pb.finish_and_clear();
        results
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
    // **Ranked ONCE, here, over everything.** Ranking inside each arm is what
    // the first draft did and it left `--probe` unsorted entirely, because
    // that arm builds its rows with a `map` and only the offline arm went
    // through `recommend_catalog`. One call over the concatenation is also
    // the only ordering that can interleave a discovered row with a curated
    // one, which is the whole point of the evidence tier.
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
