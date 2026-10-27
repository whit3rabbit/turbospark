//! The user views behind `turbospark` (the home screen), `ls`, `search`,
//! `show` and `doctor`.
//!
//! **New verbs, not new output for old ones.** `list`, `info` and
//! `recommend` keep their exact text: docs, CI and the app bundle scripts
//! quote it. These views speak to a user rather than a maintainer (`tested`
//! rather than "has a frozen gate row"), go through `crate::ui` for color,
//! and each has a `--json` form built from the same rows, so nothing has to
//! scrape the styled table.
//!
//! Like the rest of this module, nothing here decides anything: the catalog
//! resolves, `recommend_catalog` ranks, and this chooses words and columns.

use catalog::{
    library::Availability, Catalog, CatalogEntry, CountedSource, EntryKind, FitVerdict,
    InstalledEntry, Recommendation, Status, Store,
};
use serde_json::{json, Value};

use super::{human_bytes, installs_matching, machine, DEFAULT_RECOMMEND_CONTEXT};
use crate::ui::{self, Cell, CellStyle, Paint, Table};
use crate::{Error, Options};

/// What a user reads for a catalog status. The raw value stays in `--json`
/// and in `info`.
pub fn status_word(status: Status) -> &'static str {
    match status {
        Status::Verified => "tested",
        Status::Runs => "works",
        Status::Caveat => "see notes",
    }
}

fn status_cell(status: Status) -> Cell {
    let style = match status {
        Status::Verified => CellStyle::Accent,
        Status::Runs => CellStyle::Plain,
        Status::Caveat => CellStyle::Warn,
    };
    Cell::styled(status_word(status), style)
}

/// The name a user types for `entry`: its shortest tag when it has one
/// that reads better than the alias, else the alias.
fn display_tag(catalog: &Catalog, alias: &str) -> String {
    catalog
        .tags_for(alias)
        .first()
        .map(|t| t.to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn print_json(value: &Value) {
    println!(
        "{}",
        serde_json::to_string_pretty(value).expect("a serde_json::Value always serializes")
    );
}

fn availability_word(availability: &Availability) -> &'static str {
    match availability {
        Availability::Available => "available",
        Availability::NotConnected { .. } => "not connected",
        Availability::Missing => "missing",
    }
}

fn kind_of(entry: &InstalledEntry) -> String {
    entry
        .row
        .kind
        .clone()
        .unwrap_or_else(|| entry.row.effective_modality().as_str().to_string())
}

// ---------------------------------------------------------------- ls

/// `turbospark ls`: what is on disk, and nothing else. The catalog is
/// `search`.
pub fn ls(catalog: &Catalog, store: &Store, options: &Options) -> Result<(), Error> {
    let installs = installs_matching(store, options.filter.as_deref());
    if options.json {
        let rows: Vec<Value> = installs
            .iter()
            .map(|entry| {
                json!({
                    "name": entry.row.alias,
                    "tags": catalog.tags_for(&entry.row.alias),
                    "kind": kind_of(entry),
                    "family": entry.row.family,
                    "size_bytes": entry.row.install_bytes,
                    "library": entry.library_name,
                    "availability": availability_word(&entry.availability),
                    "path": entry.row.path,
                    "installed_on": entry.row.installed_on,
                    "repo": entry.row.repo,
                    "revision": entry.row.revision,
                })
            })
            .collect();
        print_json(&Value::Array(rows));
        return Ok(());
    }

    let paint = Paint::out();
    if installs.is_empty() {
        println!("  No models installed yet.");
        println!();
        println!(
            "  {}  {}",
            paint.bold("turbospark search"),
            paint.dim("browse the catalog")
        );
        println!(
            "  {}  {}",
            paint.bold("turbospark pull qwen3.6"),
            paint.dim("download one")
        );
        return Ok(());
    }
    let mut table = Table::new(&["NAME", "TAG", "KIND", "SIZE", "WHERE", "ADDED"]).right(3);
    let mut total = 0u64;
    for entry in &installs {
        total += entry.row.install_bytes;
        let place = match entry.availability {
            Availability::Available => Cell::new(entry.library_name.clone()),
            _ => Cell::styled(
                format!(
                    "{} ({})",
                    entry.library_name,
                    availability_word(&entry.availability)
                ),
                CellStyle::Warn,
            ),
        };
        table.row(vec![
            Cell::styled(entry.row.alias.clone(), CellStyle::Bold),
            Cell::styled(display_tag(catalog, &entry.row.alias), CellStyle::Dim),
            Cell::new(kind_of(entry)),
            Cell::new(human_bytes(entry.row.install_bytes)),
            place,
            Cell::styled(entry.row.installed_on.clone(), CellStyle::Dim),
        ]);
    }
    print!("{}", table.render(paint, ui::columns()));
    println!(
        "\n  {}",
        paint.dim(&format!(
            "{} model(s), {} on disk. `turbospark rm NAME` frees one.",
            installs.len(),
            human_bytes(total)
        ))
    );
    Ok(())
}

// ---------------------------------------------------------------- search

/// `turbospark search [TEXT]`: the curated catalog, installed ones marked.
pub fn search(
    catalog: &Catalog,
    store: &Store,
    needle: Option<&str>,
    options: &Options,
) -> Result<(), Error> {
    let installed = store.installed();
    let rows: Vec<&CatalogEntry> = match needle {
        Some(needle) => catalog.find(needle),
        None => catalog.entries().collect(),
    };
    if options.json {
        let rows: Vec<Value> = rows
            .iter()
            .map(|e| {
                json!({
                    "name": e.alias,
                    "tags": catalog.tags_for(&e.alias),
                    "description": e.name,
                    "kind": e.kind.as_str(),
                    "family": e.family,
                    "status": e.status.as_str(),
                    "download_bytes": e.download_bytes,
                    "install_bytes": e.install_bytes,
                    "installed": installed.contains_key(&e.alias),
                    "repo": e.source.repo,
                    "revision": e.source.revision,
                })
            })
            .collect();
        print_json(&Value::Array(rows));
        return Ok(());
    }

    let paint = Paint::out();
    if rows.is_empty() {
        println!(
            "  Nothing in the catalog matches {:?}.",
            needle.unwrap_or_default()
        );
        println!(
            "  {}",
            paint.dim("`turbospark model probe OWNER/REPO` checks any Hugging Face repo.")
        );
        return Ok(());
    }
    let mut table = Table::new(&["", "NAME", "TAG", "SIZE", "STATUS", "MODEL"]).right(3);
    for e in &rows {
        let have = installed.contains_key(&e.alias);
        table.row(vec![
            if have {
                Cell::styled("*", CellStyle::Accent)
            } else {
                Cell::new("")
            },
            Cell::styled(e.alias.clone(), CellStyle::Bold),
            Cell::styled(display_tag(catalog, &e.alias), CellStyle::Dim),
            Cell::new(human_bytes(e.download_bytes)),
            status_cell(e.status),
            Cell::new(e.name.clone()),
        ]);
    }
    print!("{}", table.render(paint, ui::columns()));
    println!();
    println!(
        "  {} {}   {}",
        paint.accent("*"),
        paint.dim("installed"),
        paint.dim(
            "tested: measured on real hardware   works: ran here   see notes: `turbospark show NAME`"
        )
    );
    Ok(())
}

// ---------------------------------------------------------------- show

/// The row a user-typed name means, or the error a user should read.
pub fn resolve<'a>(catalog: &'a Catalog, name: &str) -> Result<&'a CatalogEntry, Error> {
    if let Some(entry) = catalog.resolve_name(name) {
        return Ok(entry);
    }
    let near: Vec<String> = catalog
        .find(name.split(':').next().unwrap_or(name))
        .iter()
        .map(|e| e.alias.clone())
        .take(6)
        .collect();
    Err(Error::Failed(if near.is_empty() {
        format!("no model named {name:?}. `turbospark search` lists the catalog.")
    } else {
        format!(
            "no model named {name:?}. Close: {}. `turbospark search` lists the catalog.",
            near.join(", ")
        )
    }))
}

/// `turbospark show NAME`: one model in user terms. `--verbose` adds the
/// provenance `info` prints.
pub fn show(catalog: &Catalog, store: &Store, name: &str, options: &Options) -> Result<(), Error> {
    let entry = resolve(catalog, name)?;
    let installed = store.installed();
    let install = installed.get(&entry.alias);
    if options.json {
        let mut value = serde_json::to_value(entry)
            .map_err(|e| Error::Failed(format!("serializing {}: {e}", entry.alias)))?;
        value["tags"] = json!(catalog.tags_for(&entry.alias));
        value["installed"] = match install {
            Some(row) => json!({
                "path": row.path,
                "size_bytes": row.install_bytes,
                "installed_on": row.installed_on,
            }),
            None => Value::Null,
        };
        print_json(&value);
        return Ok(());
    }

    let paint = Paint::out();
    let field = |label: &str, value: String| {
        println!("  {:<10}  {value}", paint.dim(label));
    };
    println!();
    println!("  {}  {}", paint.bold(&entry.alias), paint.dim(&entry.name));
    println!();
    let tags = catalog.tags_for(&entry.alias);
    if !tags.is_empty() {
        field("also", tags.join(", "));
    }
    field("family", entry.family.clone());
    field(
        "status",
        match entry.status {
            Status::Verified => paint.accent("tested"),
            Status::Runs => "works".to_string(),
            Status::Caveat => paint.warn("see notes"),
        },
    );
    field(
        "size",
        format!(
            "{} download, about {} on disk",
            human_bytes(entry.download_bytes),
            human_bytes(entry.install_bytes)
        ),
    );
    let revision = if options.verbose {
        entry.source.revision.clone()
    } else {
        entry.source.revision.chars().take(12).collect()
    };
    field(
        "source",
        format!(
            "{} @ {revision} ({})",
            entry.source.repo,
            entry.source.kind.as_str()
        ),
    );
    match install {
        Some(row) => field(
            "installed",
            format!(
                "{} {}",
                paint.accent("yes"),
                paint.dim(&format!(
                    "{}, {}",
                    row.installed_on,
                    human_bytes(row.install_bytes)
                ))
            ),
        ),
        None => field(
            "installed",
            format!(
                "no  {}",
                paint.dim(&format!("turbospark pull {}", entry.alias))
            ),
        ),
    }
    if options.verbose {
        if let Some(file) = &entry.source.file {
            field("file", file.clone());
        }
        if let Some(mtp) = &entry.mtp {
            field("mtp head", format!("{} @ {}", mtp.repo, mtp.revision));
        }
        field(
            "sidecars",
            format!(
                "{} @ {}: {}",
                entry.sidecar_repo(),
                entry.sidecar_revision(),
                entry.sidecars.files.join(", ")
            ),
        );
        if !entry.gates.is_empty() {
            field("gates", entry.gates.join(", "));
        }
        if let Some(row) = install {
            field("path", row.path.display().to_string());
        }
    }
    if let Some(notes) = &entry.notes {
        println!();
        for line in wrap(notes, ui::columns().saturating_sub(4).max(40)) {
            println!("  {line}");
        }
    }
    if !options.verbose {
        println!();
        println!(
            "  {}",
            paint.dim("--verbose adds the full revision, sidecars, gates and install path.")
        );
    }
    Ok(())
}

/// Greedy word wrap, for notes that were written as one paragraph.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for paragraph in text.split('\n') {
        let mut line = String::new();
        for word in paragraph.split_whitespace() {
            if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > width {
                lines.push(std::mem::take(&mut line));
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
        lines.push(line);
    }
    lines
}

// ---------------------------------------------------------------- doctor

fn verdict_word(verdict: FitVerdict) -> &'static str {
    match verdict {
        FitVerdict::Resident => "fits",
        FitVerdict::Streams => "fits, streams experts",
        FitVerdict::Tight => "tight",
        FitVerdict::Refused => "too big",
        FitVerdict::Unknown => "not planned",
    }
}

fn verdict_cell(row: &Recommendation) -> Cell {
    if row.suspicious {
        return Cell::styled("smaller than its name claims", CellStyle::Warn);
    }
    let style = match row.fit.verdict {
        FitVerdict::Resident | FitVerdict::Streams => CellStyle::Accent,
        FitVerdict::Tight | FitVerdict::Refused => CellStyle::Warn,
        FitVerdict::Unknown => CellStyle::Dim,
    };
    Cell::styled(verdict_word(row.fit.verdict), style)
}

fn rank_rows(
    catalog: &Catalog,
    options: &Options,
    machine: &catalog::Machine,
) -> Vec<Recommendation> {
    let context = options.context.unwrap_or(DEFAULT_RECOMMEND_CONTEXT);
    let entries: Vec<&CatalogEntry> = catalog
        .entries()
        .filter(|e| e.kind == EntryKind::Model)
        .collect();
    let mut rows =
        catalog::recommend_catalog(&entries, machine, context, model_io::ExpertCacheSlots::Auto);
    catalog::rank_recommendations(&mut rows);
    rows
}

/// The best model for this machine, as the ranking puts it: the first row
/// that runs. `None` when nothing can be planned without reading headers.
fn best(rows: &[Recommendation]) -> Option<&Recommendation> {
    rows.iter().find(|r| r.fit.verdict.runs() && !r.suspicious)
}

fn alias_of(row: &Recommendation) -> String {
    match &row.origin {
        catalog::Origin::Catalog(alias) => alias.clone(),
        other => other.install_target(),
    }
}

fn machine_line(paint: Paint, machine: &catalog::Machine) -> String {
    let mut parts = Vec::new();
    if !machine.chip.is_empty() {
        parts.push(machine.chip.clone());
    }
    parts.push(format!("{} memory", human_bytes(machine.physical_bytes)));
    if let Some(ws) = machine.working_set_bytes {
        parts.push(format!("{} for the GPU", human_bytes(ws)));
    }
    paint.dim(&parts.join("  |  "))
}

/// `turbospark doctor`: this machine, and which curated models fit it.
/// The same ranking `recommend` prints, in user words.
pub fn doctor(catalog: &Catalog, store: &Store, options: &Options) -> Result<(), Error> {
    let machine = machine(options);
    if machine.physical_bytes == 0 {
        return Err(Error::Failed(
            "no memory probe on this platform; pass --budget 36GiB".to_string(),
        ));
    }
    let context = options.context.unwrap_or(DEFAULT_RECOMMEND_CONTEXT);
    let rows = rank_rows(catalog, options, &machine);
    let installed = store.installed();

    if options.json {
        let best_alias = best(&rows).map(alias_of);
        let rows: Vec<Value> = rows
            .iter()
            .map(|r| {
                json!({
                    "name": alias_of(r),
                    "description": r.name,
                    "status": r.evidence.as_str(),
                    "verdict": verdict_word(r.fit.verdict),
                    "runs": r.fit.verdict.runs(),
                    "memory_bytes": match r.fit.counted_source {
                        CountedSource::Unknown => Value::Null,
                        _ => json!(r.fit.counted),
                    },
                    "memory_source": match r.fit.counted_source {
                        CountedSource::Measured => "measured",
                        CountedSource::Estimated => "estimated",
                        CountedSource::Unknown => "unknown",
                    },
                    "disk_bytes": r.fit.mapped,
                    "tok_s": r.measured.as_ref().map(|m| json!([m.decode_tok_s_min, m.decode_tok_s_max])),
                    "installed": installed.contains_key(&alias_of(r)),
                })
            })
            .collect();
        print_json(&json!({
            "machine": {
                "chip": machine.chip,
                "memory_bytes": machine.physical_bytes,
                "gpu_working_set_bytes": machine.working_set_bytes,
                "load_guard": machine.load_guard.as_str(),
            },
            "context": context,
            "best": best_alias,
            "models": rows,
        }));
        return Ok(());
    }

    let paint = Paint::out();
    println!();
    println!("{}", ui::section(paint, "This machine"));
    println!("  {}", machine_line(paint, &machine));
    println!(
        "  {}",
        paint.dim(&format!("planned for one {context}-token conversation"))
    );
    println!();
    println!("{}", ui::section(paint, "Models"));
    let best_alias = best(&rows).map(alias_of);
    let mut table = Table::new(&["", "NAME", "STATUS", "MEMORY", "DISK", "TOK/S", "FIT"])
        .right(3)
        .right(4)
        .right(5);
    for r in &rows {
        let alias = alias_of(r);
        let mark = if Some(&alias) == best_alias.as_ref() {
            Cell::styled("top", CellStyle::Accent)
        } else if installed.contains_key(&alias) {
            Cell::styled("*", CellStyle::Accent)
        } else {
            Cell::new("")
        };
        let memory = match r.fit.counted_source {
            CountedSource::Unknown => Cell::styled("?", CellStyle::Dim),
            CountedSource::Measured => Cell::new(human_bytes(r.fit.counted)),
            CountedSource::Estimated => Cell::new(format!("~{}", human_bytes(r.fit.counted))),
        };
        let tok_s = match &r.measured {
            Some(m) => Cell::new(format!(
                "{:.0}-{:.0}",
                m.decode_tok_s_min, m.decode_tok_s_max
            )),
            None => Cell::styled("-", CellStyle::Dim),
        };
        table.row(vec![
            mark,
            Cell::styled(alias, CellStyle::Bold),
            Cell::new(match r.evidence {
                catalog::Evidence::Verified => "tested",
                catalog::Evidence::Runs => "works",
                catalog::Evidence::Caveat => "see notes",
                catalog::Evidence::Discovered => "unmeasured",
            }),
            memory,
            Cell::new(human_bytes(r.fit.mapped)),
            tok_s,
            verdict_cell(r),
        ]);
    }
    print!("{}", table.render(paint, ui::columns()));
    println!();
    println!(
        "  {}",
        paint.dim(
            "memory: what the engine allocates (~ estimated, ? not read yet). disk: the whole \
             install; experts stream, so it need not fit. tok/s: measured on this chip."
        )
    );
    let unknowns = rows
        .iter()
        .filter(|r| r.fit.verdict == FitVerdict::Unknown)
        .count();
    if unknowns > 0 {
        println!(
            "  {}",
            paint.dim(&format!(
                "{unknowns} not planned yet: `turbospark model recommend --probe` reads their \
                 headers (one request each)."
            ))
        );
    }
    Ok(())
}

// ---------------------------------------------------------------- home

/// Bare `turbospark`: who this is, what is installed, what to type next.
pub fn home(catalog: &Catalog, store: &Store, options: &Options) -> Result<(), Error> {
    let paint = Paint::out();
    let machine = machine(options);
    println!();
    println!(
        "  {} {}",
        paint.bold("turbospark"),
        paint.dim(env!("CARGO_PKG_VERSION"))
    );
    if machine.physical_bytes > 0 {
        println!("  {}", machine_line(paint, &machine));
    }
    let rows = if machine.physical_bytes > 0 {
        rank_rows(catalog, options, &machine)
    } else {
        Vec::new()
    };
    let best_alias = best(&rows).map(alias_of);

    let installs = installs_matching(store, None);
    println!();
    println!("{}", ui::section(paint, "Installed"));
    if installs.is_empty() {
        println!("  {}", paint.dim("nothing yet"));
    } else {
        let mut table = Table::new(&["", "NAME", "SIZE", "MODEL"]).right(2);
        for entry in &installs {
            let alias = &entry.row.alias;
            let mark = if Some(alias) == best_alias.as_ref() {
                Cell::styled("top", CellStyle::Accent)
            } else if entry.availability != Availability::Available {
                Cell::styled("!", CellStyle::Warn)
            } else {
                Cell::new("")
            };
            let about = catalog
                .get(alias)
                .map(|e| e.name.clone())
                .unwrap_or_else(|| kind_of(entry));
            table.row(vec![
                mark,
                Cell::styled(alias.clone(), CellStyle::Bold),
                Cell::new(human_bytes(entry.row.install_bytes)),
                Cell::styled(about, CellStyle::Dim),
            ]);
        }
        // The table's own header row is noise on a home screen.
        let text = table.render(paint, ui::columns());
        for line in text.lines().skip(1) {
            println!("{line}");
        }
    }
    if let Some(best) = &best_alias {
        if !installs.iter().any(|e| &e.row.alias == best) {
            let tag = catalog
                .tags_for(best)
                .first()
                .map(|t| t.to_string())
                .unwrap_or_else(|| best.clone());
            println!(
                "  {} {}",
                paint.accent("fastest tested model here:"),
                paint.bold(&format!("turbospark pull {tag}"))
            );
        }
    }

    let first = installs
        .iter()
        .find(|e| e.availability == Availability::Available)
        .map(|e| e.row.alias.clone())
        .or(best_alias)
        .unwrap_or_else(|| "qwen3.6".to_string());
    println!();
    println!("{}", ui::section(paint, "Commands"));
    let commands = [
        (format!("turbospark run {first}"), "chat in the terminal"),
        (
            format!("turbospark run {first} \"hi\""),
            "one answer, then exit",
        ),
        ("turbospark pull NAME".to_string(), "download a model"),
        ("turbospark search".to_string(), "browse the catalog"),
        (
            "turbospark serve".to_string(),
            "OpenAI, Anthropic and Ollama APIs",
        ),
        (
            format!("turbospark launch claude --model {first}"),
            "a coding agent on it",
        ),
        ("turbospark doctor".to_string(), "what fits this machine"),
    ];
    let width = commands.iter().map(|(c, _)| c.len()).max().unwrap_or(0);
    for (command, about) in &commands {
        println!(
            "  {}{}  {}",
            paint.bold(command),
            " ".repeat(width - command.len()),
            paint.dim(about)
        );
    }
    println!();
    println!(
        "  {}",
        paint.dim("turbospark --help for every command, turbospark COMMAND --help for its options")
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_words_are_user_words() {
        assert_eq!(status_word(Status::Verified), "tested");
        assert_eq!(status_word(Status::Runs), "works");
        assert_eq!(status_word(Status::Caveat), "see notes");
    }

    #[test]
    fn wrap_keeps_lines_inside_the_width() {
        let lines = wrap("one two three four five six seven", 10);
        assert!(lines.iter().all(|l| l.chars().count() <= 10), "{lines:?}");
        assert_eq!(lines.join(" "), "one two three four five six seven");
    }

    #[test]
    fn resolve_takes_a_tag_and_names_close_rows_on_a_miss() {
        let catalog = Catalog::embedded().unwrap();
        let Ok(entry) = resolve(&catalog, "qwen3.6:35b") else {
            panic!("a tag must resolve");
        };
        assert_eq!(entry.alias, "qwen36");
        let Err(Error::Failed(message)) = resolve(&catalog, "qwen3.6:99b") else {
            panic!("a typo must not resolve");
        };
        assert!(message.contains("qwen36"), "{message}");
    }
}
