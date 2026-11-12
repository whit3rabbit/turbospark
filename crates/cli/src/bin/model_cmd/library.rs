//! `library list|add|remove|default`: the places that hold model files.
//!
//! `LibraryRegistry` decides every rule: which locations may host a library,
//! what adding adopts, and which library is the default. This module
//! resolves the name typed and prints the outcome, with one sentence per
//! refusal so the user learns which rule the location broke.
//!
//! Library names, aliases, and paths can come from a drive anyone could
//! have written (an adopted library keeps the name its marker carries), so
//! everything printed from them is escaped and capped.

use std::path::Path;

use catalog::library::{
    AddedLibrary, LibraryError, LibraryRecord, LibraryRegistry, LibraryView, LoadReport,
    LocationEnv, LocationRefusal, Refreshed,
};
use catalog::Store;

use super::render::truncate;
use crate::{one_positional, Error, Options};

/// The most characters printed of a library name or an alias.
pub(super) const NAME_LIMIT: usize = 64;
/// The most characters printed of an error reason. Marker errors echo
/// untrusted values up to the marker's size cap.
pub(super) const MESSAGE_LIMIT: usize = 300;

pub fn library(store: &Store, positionals: &[String], options: &Options) -> Result<(), Error> {
    let Some((command, arguments)) = positionals.split_first() else {
        return Err(Error::Usage(
            "library needs list, add, remove, or default".to_string(),
        ));
    };
    match command.as_str() {
        "list" => {
            options.reject_unused(&[])?;
            if !arguments.is_empty() {
                return Err(Error::Usage("library list takes no arguments".to_string()));
            }
            list(store)
        }
        "add" => {
            options.reject_unused(&["name"])?;
            let path = one_positional(arguments, "library add", "<PATH>")?;
            add(store, path, options.name.as_deref())
        }
        "remove" => {
            options.reject_unused(&["yes"])?;
            let target = one_positional(arguments, "library remove", "<NAME>")?;
            remove(store, target, options.yes)
        }
        "default" => {
            options.reject_unused(&[])?;
            let target = one_positional(arguments, "library default", "<NAME>")?;
            set_default(store, target)
        }
        other => Err(Error::Usage(format!("unknown library command {other:?}"))),
    }
}

/// Every library, This Mac first, after a refresh that locates each one.
///
/// Bookkeeping trouble is a warning, never a failure (3.4): the refresh
/// returns current views even when it could not record them, and a list it
/// could not read still lists This Mac.
fn list(store: &Store) -> Result<(), Error> {
    let env = location_env(store)?;
    let mut registry = LibraryRegistry::load(store.root());
    // `load` may have set an unparseable list aside, which the refresh's own
    // re-read would no longer report.
    let mut warnings = report_warnings(registry.load_report());
    let Refreshed {
        views,
        persist_error,
    } = registry.refresh(&env);
    for warning in report_warnings(registry.load_report()) {
        if !warnings.contains(&warning) {
            warnings.push(warning);
        }
    }
    if let Some(error) = persist_error {
        warnings.push(format!(
            "could not record when each library was last seen: {}",
            reason(&error)
        ));
    }
    for warning in &warnings {
        eprintln!("warning: {warning}");
    }

    let rows: Vec<[String; 3]> = views
        .iter()
        .map(|view| {
            [
                clean(&view.record.name, NAME_LIMIT),
                location(view),
                status(view),
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
    let (name_width, location_width, status_width) =
        (width(0, "NAME"), width(1, "LOCATION"), width(2, "STATUS"));
    println!(
        "  {:<name_width$}  {:<location_width$}  {:<status_width$}  MODELS",
        "NAME", "LOCATION", "STATUS"
    );
    for (view, [name, location, status]) in views.iter().zip(&rows) {
        let default = if view.record.default_for_downloads {
            "  (default)"
        } else {
            ""
        };
        // Only for this refresh: the next one finds the library at its
        // recorded root, so it is not a lasting state.
        let note = if view.duplicate {
            "  note: another connected folder carries this library too; this one is used"
        } else {
            ""
        };
        println!(
            "  {name:<name_width$}  {location:<location_width$}  {status:<status_width$}  {:>6}{default}{note}",
            view.model_count
        );
    }
    Ok(())
}

/// Adds the folder at `path` as a library, or adopts the library already
/// there with its models (1.3, 1.6).
fn add(store: &Store, path: &str, name: Option<&str>) -> Result<(), Error> {
    let env = location_env(store)?;
    let mut registry = load(store);
    let listed: Vec<String> = registry
        .libraries()
        .into_iter()
        .map(|record| record.id)
        .collect();
    let AddedLibrary { view, skipped } = registry
        .add(Path::new(path), name, &env)
        .map_err(|error| Error::Failed(add_failure(path, &error)))?;
    let name = clean(&view.record.name, NAME_LIMIT);
    let location = location(&view);
    if listed.contains(&view.record.id) {
        println!("{location} is already the library {name}");
        return Ok(());
    }
    println!(
        "added the library {name} at {location} with {}",
        models(view.model_count)
    );
    if !skipped.is_empty() {
        let aliases: Vec<String> = skipped
            .iter()
            .map(|alias| clean(alias, NAME_LIMIT))
            .collect();
        println!(
            "not listed, because another library already has a model by that name \
             (or the drive lists it twice): {}",
            aliases.join(", ")
        );
    }
    Ok(())
}

/// Stops listing a library and its models. Never touches a file (1.8).
fn remove(store: &Store, target: &str, yes: bool) -> Result<(), Error> {
    let mut registry = load(store);
    let record = resolve(&registry, target)?;
    if record.is_internal() {
        return Err(Error::Failed(
            "This Mac cannot be removed: it is the library in the TurboSpark home folder"
                .to_string(),
        ));
    }
    let name = clean(&record.name, NAME_LIMIT);
    if !yes {
        // As `rm` does: the argument typed back, unless --yes.
        eprintln!(
            "remove the library {name} (last seen at {})? Its models stop being listed. \
             The files on the drive stay untouched, and adding the folder again lists them again.",
            clean_path(&record.last_mount_path.join(&record.volume_relative_path))
        );
        eprint!("type \"{}\" to confirm: ", clean(target, MESSAGE_LIMIT));
        use std::io::Write;
        let _ = std::io::stderr().flush();
        let mut typed = String::new();
        std::io::stdin()
            .read_line(&mut typed)
            .map_err(|e| Error::Failed(format!("reading confirmation: {e}")))?;
        if typed.trim() != target {
            return Err(Error::Failed("not confirmed, nothing removed".to_string()));
        }
    }
    registry.remove(&record.id).map_err(|error| {
        Error::Failed(format!(
            "could not remove the library {name}: {}",
            reason(&error)
        ))
    })?;
    println!("removed the library {name}; the files on its drive are untouched");
    Ok(())
}

/// Makes one library the default for new downloads (1.9).
fn set_default(store: &Store, target: &str) -> Result<(), Error> {
    let mut registry = load(store);
    let record = resolve(&registry, target)?;
    let name = clean(&record.name, NAME_LIMIT);
    registry.set_default(&record.id).map_err(|error| {
        Error::Failed(format!(
            "could not make {name} the default: {}",
            reason(&error)
        ))
    })?;
    println!("new downloads go to {name} by default");
    Ok(())
}

fn location_env(store: &Store) -> Result<LocationEnv, Error> {
    LocationEnv::system(store.root()).map_err(Error::Failed)
}

/// The library list, with whatever reading it set aside printed as
/// warnings. Never fails: a list that cannot be read is only This Mac.
pub(super) fn load(store: &Store) -> LibraryRegistry {
    let registry = LibraryRegistry::load(store.root());
    for warning in report_warnings(registry.load_report()) {
        eprintln!("warning: {warning}");
    }
    registry
}

fn report_warnings(report: &LoadReport) -> Vec<String> {
    let mut warnings = Vec::new();
    if let Some(path) = &report.quarantined {
        warnings.push(format!(
            "the library list could not be parsed and was set aside as {}",
            clean_path(path)
        ));
    }
    for dropped in &report.dropped {
        warnings.push(format!(
            "ignored a library record: {}",
            clean(dropped, MESSAGE_LIMIT)
        ));
    }
    if let Some(unreadable) = &report.unreadable {
        warnings.push(format!(
            "could not read the library list: {}",
            clean(unreadable, MESSAGE_LIMIT)
        ));
    }
    warnings
}

/// The library `target` names: an id exactly, else a name ignoring case.
/// Two libraries can share a name (an adopted library keeps the name its
/// drive gives it), so a name that matches several is refused with their
/// ids rather than guessed. Pulls name their `--library` the same way.
pub(super) fn resolve(registry: &LibraryRegistry, target: &str) -> Result<LibraryRecord, Error> {
    let records = registry.libraries();
    if let Some(record) = records.iter().find(|record| record.id == target) {
        return Ok(record.clone());
    }
    let wanted = target.to_lowercase();
    let matches: Vec<&LibraryRecord> = records
        .iter()
        .filter(|record| record.name.to_lowercase() == wanted)
        .collect();
    let shown = clean(target, NAME_LIMIT);
    match matches.as_slice() {
        [record] => Ok((*record).clone()),
        [] => Err(Error::Failed(format!(
            "no library named \"{shown}\"; `turbospark-model library list` shows them"
        ))),
        several => {
            let ids: Vec<String> = several
                .iter()
                .map(|record| {
                    format!(
                        "{} (last seen at {})",
                        clean(&record.id, MESSAGE_LIMIT),
                        clean_path(&record.last_mount_path.join(&record.volume_relative_path))
                    )
                })
                .collect();
            Err(Error::Failed(format!(
                "{} libraries are named \"{shown}\"; name one by its id: {}",
                several.len(),
                ids.join(", ")
            )))
        }
    }
}

/// The message for a failed add: one sentence per refusal, else the reason.
fn add_failure(path: &str, error: &LibraryError) -> String {
    let path = clean(path, MESSAGE_LIMIT);
    match error {
        LibraryError::Refused(refusal) => refusal_sentence(&path, refusal),
        other => format!("could not add {path} as a library: {}", reason(other)),
    }
}

/// Why `path` cannot be a library, naming the rule it broke (1.4, 1.5,
/// 1.7, 1.10).
fn refusal_sentence(path: &str, refusal: &LocationRefusal) -> String {
    match refusal {
        LocationRefusal::SameDiskAsHome => {
            format!("{path} is on the same disk as This Mac, which already covers it.")
        }
        LocationRefusal::Network => format!(
            "{path} is on a network share; libraries must be on a directly connected drive."
        ),
        LocationRefusal::SmallFileLimit { fs_type } => format!(
            "{path} is on a drive formatted as {}, which cannot store a single file of 4 GiB \
             or larger.",
            clean(fs_type, NAME_LIMIT)
        ),
        LocationRefusal::ReadOnly => format!(
            "{path} is on a read-only drive; a library needs a drive TurboSpark can write to."
        ),
        LocationRefusal::NotADirectory => format!("{path} is not an existing folder."),
        // Also an unreadable mount table and a mount nested in another
        // volume: in each case the drive's kind is unknown.
        LocationRefusal::UnknownVolume => format!(
            "TurboSpark could not identify this drive, so it cannot check that {path} is on a \
             directly connected drive that can hold large files."
        ),
        LocationRefusal::NestedIn { library } => format!(
            "{path} overlaps the library {}, which already covers it.",
            clean(library, NAME_LIMIT)
        ),
        LocationRefusal::InsideHome => {
            format!("{path} overlaps the TurboSpark home folder, which This Mac already covers.")
        }
    }
}

fn reason(error: &LibraryError) -> String {
    let text = match error {
        LibraryError::Refused(refusal) => refusal_sentence("the location", refusal),
        LibraryError::UnknownLibrary(id) => format!("no library has the id {id}"),
        LibraryError::Internal => "This Mac cannot be removed".to_string(),
        LibraryError::Marker(error) => error.to_string(),
        LibraryError::NotConnected(path) => format!("{} is not connected", path.display()),
        LibraryError::Io(reason) => reason.clone(),
    };
    clean(&text, MESSAGE_LIMIT)
}

/// The root, or where the library was last seen when not connected.
fn location(view: &LibraryView) -> String {
    match &view.root {
        Some(root) => clean_path(root),
        None => clean_path(
            &view
                .record
                .last_mount_path
                .join(&view.record.volume_relative_path),
        ),
    }
}

fn status(view: &LibraryView) -> String {
    match (view.available, &view.record.last_seen) {
        (true, _) => "connected".to_string(),
        (false, Some(date)) => format!("not connected, last seen {}", clean(date, NAME_LIMIT)),
        (false, None) => "not connected".to_string(),
    }
}

fn models(count: usize) -> String {
    if count == 1 {
        "1 model".to_string()
    } else {
        format!("{count} models")
    }
}

/// A path printed in full, with only the escaping of [`clean`]: a cut path
/// could not be found again.
pub(super) fn clean_path(path: &Path) -> String {
    clean(&path.display().to_string(), usize::MAX)
}

/// `text` with control and bidirectional formatting characters escaped,
/// cut to `limit` characters. Printed raw, a name from a drive could move
/// the cursor, recolor the terminal, or reorder what the user reads.
pub(super) fn clean(text: &str, limit: usize) -> String {
    let escaped: String = text
        .chars()
        .map(|c| {
            if c.is_control() || is_bidi_control(c) {
                c.escape_default().to_string()
            } else {
                c.to_string()
            }
        })
        .collect();
    truncate(&escaped, limit)
}

/// The explicit direction marks, embeddings, overrides, and isolates.
fn is_bidi_control(c: char) -> bool {
    matches!(
        c,
        '\u{061c}' | '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_escapes_control_and_direction_characters_and_caps_length() {
        assert_eq!(clean("T7\u{1b}[31m", NAME_LIMIT), "T7\\u{1b}[31m");
        assert_eq!(clean("a\nb", NAME_LIMIT), "a\\nb");
        assert_eq!(clean("abc\u{202e}fed", NAME_LIMIT), "abc\\u{202e}fed");
        let long = "x".repeat(100);
        let cut = clean(&long, NAME_LIMIT);
        assert_eq!(cut.chars().count(), NAME_LIMIT);
        assert!(cut.ends_with("..."));
        assert_eq!(clean("Fast", NAME_LIMIT), "Fast");
        assert_eq!(clean("caf\u{e9}", NAME_LIMIT), "caf\u{e9}");
    }

    /// Every refusal has its own sentence, so the user can tell which rule
    /// the location broke.
    #[test]
    fn each_refusal_has_a_distinct_sentence() {
        let refusals = [
            LocationRefusal::SameDiskAsHome,
            LocationRefusal::Network,
            LocationRefusal::SmallFileLimit {
                fs_type: "msdos".to_string(),
            },
            LocationRefusal::ReadOnly,
            LocationRefusal::NotADirectory,
            LocationRefusal::UnknownVolume,
            LocationRefusal::NestedIn {
                library: "Fast".to_string(),
            },
            LocationRefusal::InsideHome,
        ];
        let sentences: Vec<String> = refusals
            .iter()
            .map(|refusal| refusal_sentence("/Volumes/X", refusal))
            .collect();
        for (index, sentence) in sentences.iter().enumerate() {
            assert!(sentence.contains("/Volumes/X"), "{sentence}");
            assert!(!sentences[index + 1..].contains(sentence), "{sentence}");
        }
        assert!(sentences[2].contains("4 GiB"), "{}", sentences[2]);
        assert!(sentences[6].contains("Fast"), "{}", sentences[6]);
    }
}
