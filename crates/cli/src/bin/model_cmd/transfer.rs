//! `move <ALIAS> --to <LIBRARY>`: one install from one library to another
//! (10.6).
//!
//! `catalog::library::move_install` applies every transfer rule: copy,
//! verify, publish, record, and only then delete the original, with a
//! failure or cancel before the record leaving the original as it was (7.1,
//! 7.3). This module finds the install the alias names, measures the
//! destination's free space with `df`, shows progress, turns Ctrl-C into a
//! cancel, and prints one sentence per outcome.
//!
//! Busy checks are the caller's (7.5), and a command has no view of other
//! processes: nothing stops an install or a session in another process from
//! using the model while it moves. The help text says so.

use std::cell::{Cell, OnceCell};
use std::io::IsTerminal;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use catalog::library::{
    move_install, Availability, LibraryRecord, LibraryRegistry, LocationEnv, MoveError,
    MoveOutcome, INTERNAL_LIBRARY_ID,
};
use catalog::{human_bytes, EntryKind, ModelModality, Store};
use indicatif::ProgressBar;

use super::library::{clean, clean_path, load, resolve, MESSAGE_LIMIT, NAME_LIMIT};
use super::{install_sigint_handler, progress, INTERRUPTED};
use crate::{Error, Options};

/// What `--kind` takes. A vision tower sits in the text folder and is told
/// apart from a text model by its row's kind.
const KINDS: [&str; 4] = ["text", "vision", "audio", "image"];

/// macOS keeps `df` in `/bin`. An absolute path, so `PATH` cannot swap in
/// another program.
const DF: &str = "/bin/df";

/// One install the alias names.
struct Found {
    kind: &'static str,
    install_id: String,
    library_name: String,
    /// The library's name when it is not connected.
    not_connected: Option<String>,
}

pub fn move_model(store: &Store, alias: &str, options: &Options) -> Result<(), Error> {
    let target = options
        .to
        .as_deref()
        .ok_or_else(|| Error::Usage("move needs --to <LIBRARY>".to_string()))?;
    let kind = options.kind.as_deref();
    if let Some(kind) = kind {
        if !KINDS.contains(&kind) {
            return Err(Error::Usage(format!(
                "--kind is text, vision, audio, or image, not \"{}\"",
                clean(kind, NAME_LIMIT)
            )));
        }
    }
    // The alias is joined into paths below, so it must be a plain name, as
    // the store requires of every install alias.
    let shown = clean(alias, NAME_LIMIT);
    if alias.is_empty() || alias.starts_with('.') || alias.contains('/') {
        return Err(Error::Usage(format!(
            "\"{shown}\" is not a valid install alias"
        )));
    }

    let env = LocationEnv::system(store.root()).map_err(Error::Failed)?;
    let source = find_install(store, alias, kind, &env)?;
    if let Some(library) = &source.not_connected {
        return Err(Error::Failed(format!(
            "{shown} is in the library \"{}\", which is not connected; connect its drive and \
             try again",
            clean(library, NAME_LIMIT)
        )));
    }
    let registry = load(store);
    let destination = destination(store, &registry, target)?;
    let library = clean(&destination.name, NAME_LIMIT);
    let failed = |error: MoveError| Error::Failed(failure(&error, &shown, &library));
    // Settles that the destination is connected and writable before `df`
    // reads it.
    let dest_id = (!destination.is_internal()).then_some(destination.id.as_str());
    let root = store
        .library_root_for_install(dest_id, &env)
        .map_err(|error| failed(error.into()))?;
    let free = free_bytes(&root).map_err(|reason| {
        Error::Failed(format!(
            "could not measure the free space in {library}, so nothing was moved: {}",
            clean(&reason, MESSAGE_LIMIT)
        ))
    })?;

    INTERRUPTED.store(false, Ordering::Release);
    install_sigint_handler();
    let reporter = Reporter::new();
    let outcome = move_install(
        store,
        &source.install_id,
        &destination.id,
        &env,
        free,
        &|copied, total| reporter.report(copied, total),
        &INTERRUPTED,
    );
    reporter.finish();
    // Exit 0 whenever the install is in its destination: the registry
    // points there, so what is left over is a warning, not a failed move.
    let outcome = outcome.map_err(failed)?;
    println!("moved {shown} to {library}");
    let stale = |name: &str| {
        eprintln!(
            "warning: the marker of the library \"{}\" still lists {shown}, which is harmless",
            clean(name, NAME_LIMIT)
        );
    };
    match outcome {
        MoveOutcome::Moved => {}
        MoveOutcome::SourceMarkerStale(name) => stale(&name),
        MoveOutcome::SourceCleanupFailed { path, stale_marker } => {
            eprintln!(
                "warning: the move succeeded, but the original at {} could not be deleted; \
                 delete it by hand",
                clean_path(&path)
            );
            if let Some(name) = stale_marker {
                stale(&name);
            }
        }
    }
    Ok(())
}

/// The install `alias` names: its registry row, or for an image, which is
/// not recorded, its folder in the one connected library holding it. An
/// alias of several kinds needs `kind`.
fn find_install(
    store: &Store,
    alias: &str,
    kind: Option<&str>,
    env: &LocationEnv,
) -> Result<Found, Error> {
    let mut found: Vec<Found> = store
        .installed_with_availability(env)
        .into_iter()
        .filter(|entry| entry.row.alias == alias)
        .map(|entry| Found {
            kind: kind_of(&entry.row),
            not_connected: match entry.availability {
                Availability::NotConnected { library } => Some(library),
                _ => None,
            },
            install_id: entry.install_id,
            library_name: entry.library_name,
        })
        .collect();
    if !found.iter().any(|install| install.kind == "image") {
        found.extend(image_folders(store, alias, env));
    }
    let shown = clean(alias, NAME_LIMIT);
    let described = |installs: &[Found]| {
        installs
            .iter()
            .map(|install| {
                format!(
                    "{} in {}",
                    install.kind,
                    clean(&install.library_name, NAME_LIMIT)
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    let (mut matching, others): (Vec<Found>, Vec<Found>) = found
        .into_iter()
        .partition(|install| kind.is_none_or(|kind| install.kind == kind));
    match matching.as_slice() {
        [_] => Ok(matching.remove(0)),
        [] if !others.is_empty() => Err(Error::Failed(format!(
            "no {} install is named {shown}; there is {}",
            kind.unwrap_or_default(),
            described(&others)
        ))),
        // Embedding and OCR installs have no row and only one place.
        [] if kind.is_none() && home_only_install(store, alias) => {
            Err(Error::Failed(failure(&MoveError::HomeOnly, &shown, "")))
        }
        [] => Err(Error::Failed(format!(
            "{shown} is not installed; `turbospark-model list` shows installs"
        ))),
        several
            if several
                .iter()
                .all(|install| install.kind == several[0].kind) =>
        {
            // Only image folders repeat: aliases are unique per kind in the
            // registry, but a folder can be copied to another drive by hand.
            Err(Error::Usage(format!(
                "the {} {shown} is in more than one library ({}); keep one copy and remove \
                 the others first",
                several[0].kind,
                described(several)
            )))
        }
        several => {
            let mut kinds: Vec<&str> = several.iter().map(|install| install.kind).collect();
            kinds.sort_unstable();
            kinds.dedup();
            Err(Error::Usage(format!(
                "{shown} names installs of more than one kind ({}); pass --kind {}",
                described(several),
                kinds.join(" or --kind ")
            )))
        }
    }
}

/// The `--kind` word for a registry row.
fn kind_of(row: &catalog::InstalledModel) -> &'static str {
    match row.effective_modality() {
        ModelModality::Text if row.kind.as_deref() == Some(EntryKind::VisionTower.as_str()) => {
            "vision"
        }
        ModelModality::Text => "text",
        ModelModality::Image => "image",
        ModelModality::Audio => "audio",
    }
}

/// The image folders of `alias` in This Mac and every connected library,
/// as `move_install` names them: `<library id>:image/<alias>`. Anything at
/// the folder's name counts, so a link there is refused by the move with
/// its reason rather than reported as not installed.
fn image_folders(store: &Store, alias: &str, env: &LocationEnv) -> Vec<Found> {
    // One mount table read for every library. A bookkeeping write the
    // refresh cannot make is no concern of a move (3.4).
    let views = LibraryRegistry::load(store.root()).refresh(env).views;
    views
        .into_iter()
        .filter_map(|view| {
            let folder = Store::image_install_path_in(view.root.as_deref()?, alias).ok()?;
            std::fs::symlink_metadata(folder).ok()?;
            Some(Found {
                kind: "image",
                install_id: format!("{}:image/{alias}", view.record.id),
                library_name: view.record.name,
                not_connected: None,
            })
        })
        .collect()
}

/// Whether This Mac holds an embedding or OCR install of `alias`, the kinds
/// that stay there (2.5).
fn home_only_install(store: &Store, alias: &str) -> bool {
    let models = store.models_root();
    [
        store.embedding_install_path(alias),
        models.join("ocr").join(alias),
    ]
    .iter()
    .any(|folder| std::fs::symlink_metadata(folder).is_ok())
}

/// The library `target` names. "This Mac", in any case, and "internal"
/// always mean the internal library, even beside a drive's library that
/// took the same name.
fn destination(
    store: &Store,
    registry: &LibraryRegistry,
    target: &str,
) -> Result<LibraryRecord, Error> {
    if target.eq_ignore_ascii_case("this mac") || target == INTERNAL_LIBRARY_ID {
        return Ok(LibraryRecord::internal(store.root()));
    }
    resolve(registry, target)
}

/// One sentence per way a move ends without moving. `alias` and `library`
/// (the destination's name) are already escaped.
fn failure(error: &MoveError, alias: &str, library: &str) -> String {
    let name = |text: &str| clean(text, NAME_LIMIT);
    match error {
        MoveError::Busy(reason) => format!(
            "{alias} is in use, so it was not moved: {}",
            clean(reason, MESSAGE_LIMIT)
        ),
        MoveError::InsufficientSpace { needed, available } => format!(
            "{library} has {} free and {alias} needs {}, so nothing was moved",
            human_bytes(*available),
            human_bytes(*needed)
        ),
        MoveError::NotConnected(library) => format!(
            "the library \"{}\" is not connected; connect its drive and try again",
            name(library)
        ),
        MoveError::LibraryReadOnly(library) => format!(
            "the library \"{}\" was written by a newer TurboSpark and is read-only here, so \
             nothing was moved",
            name(library)
        ),
        MoveError::HomeOnly => {
            format!("embedding and OCR models stay on This Mac, so {alias} cannot be moved")
        }
        MoveError::SameLibrary => format!("{alias} is already in {library}"),
        MoveError::UnknownInstall(_) => {
            format!("{alias} is not installed; `turbospark-model list` shows installs")
        }
        MoveError::DestinationTaken(library) => format!(
            "the library \"{}\" already holds a model named {alias}, so nothing was moved",
            name(library)
        ),
        MoveError::Cancelled => "the move was cancelled; nothing was changed".to_string(),
        MoveError::VerifyFailed {
            expected,
            found,
            expected_files,
            found_files,
        } => format!(
            "the copy did not match the original ({found} of {expected} bytes, {found_files} \
             of {expected_files} files); nothing was changed"
        ),
        MoveError::Io(reason) => {
            format!("could not move {alias}: {}", clean(reason, MESSAGE_LIMIT))
        }
    }
}

/// The free space at `root`, from `df`. A report that cannot be read is an
/// error, never a guess: the move's only free-space check depends on it.
fn free_bytes(root: &Path) -> Result<u64, String> {
    if let Some(forced) = free_bytes_override() {
        return forced;
    }
    // This Mac's models folder may not exist yet, and is made on the disk
    // of the nearest folder that does.
    let existing = root
        .ancestors()
        .find(|folder| folder.exists())
        .ok_or_else(|| format!("{} does not exist", root.display()))?;
    let output = Command::new(DF)
        .arg("-Pk")
        .arg("--")
        .arg(existing)
        .env("LC_ALL", "C")
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("running {DF}: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "{DF} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    available_bytes(&String::from_utf8_lossy(&output.stdout))
        .ok_or_else(|| format!("{DF} printed a report this command cannot read"))
}

/// Debug builds only: `TURBOSPARK_TEST_FREE_BYTES` stands in for `df`, as
/// the other `TURBOSPARK_TEST_*` overrides stand in for the system. Release
/// builds ignore it, so a shipped binary always measures.
#[cfg(debug_assertions)]
fn free_bytes_override() -> Option<Result<u64, String>> {
    let value = std::env::var_os("TURBOSPARK_TEST_FREE_BYTES").filter(|value| !value.is_empty())?;
    Some(
        value
            .to_str()
            .and_then(|text| text.trim().parse().ok())
            .ok_or_else(|| format!("TURBOSPARK_TEST_FREE_BYTES is not a byte count: {value:?}")),
    )
}

#[cfg(not(debug_assertions))]
fn free_bytes_override() -> Option<Result<u64, String>> {
    None
}

/// The available bytes in a `df -Pk` report: a header naming 1024-byte
/// blocks, then one filesystem's line, whose fourth column is the available
/// blocks. The numbers are found as the first run of three counts and a
/// percentage (total, used, available, capacity) rather than by column:
/// a filesystem name can hold spaces (`map auto_home`) or, when long, sit
/// on a line of its own, and either shifts the columns.
fn available_bytes(report: &str) -> Option<u64> {
    let mut lines = report.lines();
    // `-k` makes the blocks 1024 bytes, which the header names.
    if !lines
        .next()?
        .split_whitespace()
        .any(|word| word == "1024-blocks")
    {
        return None;
    }
    let words: Vec<&str> = lines.flat_map(str::split_whitespace).collect();
    let count = |word: &str| word.parse::<u64>().ok();
    let percentage = |word: &str| word.strip_suffix('%').and_then(count).is_some();
    let at = words
        .windows(4)
        .position(|run| run[..3].iter().all(|word| count(word).is_some()) && percentage(run[3]))?;
    // A filesystem name comes first.
    if at == 0 {
        return None;
    }
    count(words[at + 2])?.checked_mul(1024)
}

/// Move progress on stderr: a bar on a terminal, else a plain line each
/// time another 5% is copied, at most once a second, and one at the end.
struct Reporter {
    terminal: bool,
    bar: OnceCell<ProgressBar>,
    /// When the last plain line was printed, and the percentage it showed.
    last: Cell<Option<(Instant, u64)>>,
}

impl Reporter {
    fn new() -> Self {
        Self {
            terminal: std::io::stderr().is_terminal(),
            bar: OnceCell::new(),
            last: Cell::new(None),
        }
    }

    fn report(&self, copied: u64, total: u64) {
        if self.terminal {
            self.bar
                .get_or_init(|| progress::byte_progress_bar(total))
                .set_position(copied);
            return;
        }
        let percent = if total == 0 {
            100
        } else {
            (u128::from(copied.min(total)) * 100 / u128::from(total)) as u64
        };
        let now = Instant::now();
        let due = match self.last.get() {
            None => true,
            Some((_, shown)) if percent == 100 => shown < 100,
            Some((at, shown)) => {
                percent >= shown + 5 && now.duration_since(at) >= Duration::from_secs(1)
            }
        };
        if due {
            eprintln!(
                "[move] {percent}% ({} of {})",
                human_bytes(copied),
                human_bytes(total)
            );
            self.last.set(Some((now, percent)));
        }
    }

    fn finish(&self) {
        if let Some(bar) = self.bar.get() {
            bar.finish_and_clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What `/bin/df -Pk /` prints on macOS.
    const MACOS: &str = "\
Filesystem     1024-blocks      Used Available Capacity  Mounted on
/dev/disk3s1s1   971350180  12344364  56034448    19%    /
";

    #[test]
    fn the_available_column_of_a_df_report_is_read_as_bytes() {
        assert_eq!(available_bytes(MACOS), Some(56034448 * 1024));
        // A filesystem name with a space, as macOS prints the automounter.
        let spaced = "\
Filesystem    1024-blocks Used Available Capacity  Mounted on
map auto_home           0    0         7   100%    /System/Volumes/Data/home
";
        assert_eq!(available_bytes(spaced), Some(7 * 1024));
        // A long name on a line of its own, the numbers on the next, and a
        // mount point whose own words look like the numbers: the first run
        // is the one read.
        let wrapped = "\
Filesystem 1024-blocks Used Available Capacity Mounted on
/dev/mapper/a-volume-group-name-long-enough-to-wrap-onto-its-own-line
                 971350180 12344364 56034448 19% /Volumes/Disk 1 2 3 4%
";
        assert_eq!(available_bytes(wrapped), Some(56034448 * 1024));
    }

    #[test]
    fn a_df_report_that_cannot_be_read_is_refused() {
        for report in [
            "",
            "garbage",
            // Header only.
            "Filesystem 1024-blocks Used Available Capacity Mounted on\n",
            // Blocks of another size.
            "Filesystem 512-blocks Used Available Capacity Mounted on\n\
             /dev/disk3s1 100 50 40 50% /\n",
            // A column that is not a count.
            "Filesystem 1024-blocks Used Available Capacity Mounted on\n\
             /dev/disk3s1 100 50 lots 50% /\n",
            // No filesystem name before the numbers.
            "Filesystem 1024-blocks Used Available Capacity Mounted on\n\
             100 50 40 50% /\n",
        ] {
            assert_eq!(available_bytes(report), None, "{report:?}");
        }
    }

    #[test]
    fn a_df_count_too_large_for_bytes_is_refused() {
        let header = "Filesystem 1024-blocks Used Available Capacity Mounted on\n";
        // Fits in u64 as blocks, overflows as bytes.
        let blocks = u64::MAX / 1024 + 1;
        let report = format!("{header}/dev/disk3s1 {blocks} 0 {blocks} 0% /\n");
        assert_eq!(available_bytes(&report), None);
        let fits = u64::MAX / 1024;
        let report = format!("{header}/dev/disk3s1 {fits} 0 {fits} 0% /\n");
        assert_eq!(available_bytes(&report), Some(fits * 1024));
        // Past u64 even as blocks.
        let report = format!("{header}/dev/disk3s1 1 0 99999999999999999999999 0% /\n");
        assert_eq!(available_bytes(&report), None);
    }

    /// Every way a move fails has its own sentence, naming what the user
    /// can act on.
    #[test]
    fn each_move_failure_has_a_distinct_sentence() {
        let errors = [
            MoveError::Busy("loaded".to_string()),
            MoveError::InsufficientSpace {
                needed: 5 * 1024 * 1024,
                available: 10,
            },
            MoveError::NotConnected("T7".to_string()),
            MoveError::LibraryReadOnly("T7".to_string()),
            MoveError::HomeOnly,
            MoveError::SameLibrary,
            MoveError::UnknownInstall("internal:text/qwen".to_string()),
            MoveError::DestinationTaken("T9".to_string()),
            MoveError::Cancelled,
            MoveError::VerifyFailed {
                expected: 10,
                found: 9,
                expected_files: 2,
                found_files: 2,
            },
            MoveError::Io("disk on fire".to_string()),
        ];
        let sentences: Vec<String> = errors
            .iter()
            .map(|error| failure(error, "qwen", "Fast"))
            .collect();
        for (index, sentence) in sentences.iter().enumerate() {
            assert!(!sentence.contains('\n'), "{sentence}");
            assert!(!sentences[index + 1..].contains(sentence), "{sentence}");
        }
        assert!(sentences[1].contains("5.0 MiB"), "{}", sentences[1]);
        assert!(sentences[1].contains("10 B"), "{}", sentences[1]);
        assert!(
            sentences[2].contains("connect its drive"),
            "{}",
            sentences[2]
        );
        assert!(
            sentences[4].contains("stay on This Mac"),
            "{}",
            sentences[4]
        );
        assert!(
            sentences[8].contains("nothing was changed"),
            "{}",
            sentences[8]
        );
        assert!(
            sentences[9].contains("nothing was changed"),
            "{}",
            sentences[9]
        );
        assert!(sentences[10].contains("disk on fire"), "{}", sentences[10]);
        // A name from a drive is escaped.
        let hostile = failure(
            &MoveError::NotConnected("T7\u{1b}[31m".to_string()),
            "q",
            "",
        );
        assert!(!hostile.contains('\u{1b}'), "{hostile}");
    }
}
