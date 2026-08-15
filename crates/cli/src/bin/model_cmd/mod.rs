//! The six subcommands, each a print of something `turbospark-catalog`
//! computed.
//!
//! **Nothing here decides anything.** The catalog crate resolves rows, the
//! probe reaches a verdict and the install driver runs the walk; this module
//! chooses column widths. That split is the same one `turbospark-check` has
//! with `turbospark-invocation`, and it is what lets the verdict logic be
//! tested without a terminal.

mod render;

use catalog::{Catalog, Client, InstallPlan, RepoRef, Store, Verdict};

use crate::{Error, Options};

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
        "  {:<width$}  {:<9}  {:>9}  MODEL",
        "ALIAS",
        "STATUS",
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
            "{mark} {:<width$}  {:<9}  {:>9}  {}",
            entry.alias,
            entry.status.as_str(),
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
    let report = catalog::probe(client, repo, file, sidecars).map_err(Error::Failed)?;
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
    std::fs::remove_dir_all(&path)
        .map_err(|e| Error::Failed(format!("removing {}: {e}", path.display())))?;
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
    let plan = resolve_plan(catalog, positionals, options)?;
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

    let mut progress = |stage: &str| eprintln!("[pull] {stage}");
    let report =
        catalog::gate(client, &plan, options.force, &mut progress).map_err(Error::Failed)?;
    render::report(&report);

    let installed = catalog::install(&plan, &dir, client, &mut progress).map_err(Error::Failed)?;
    catalog::record(store, &installed).map_err(Error::Failed)?;

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

/// Turn `pull`'s two argument forms into one plan.
fn resolve_plan(
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
            let client = Client::new();
            let report =
                catalog::probe(&client, &weights, options.file.as_deref(), Some(&sidecars))
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
