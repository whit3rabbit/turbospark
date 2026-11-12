//! Where a pull lands: `--out`, the library `--library` names, or the
//! default library, settled before any network request (10.4).
//!
//! `catalog::install_destination` decides every rule (not connected,
//! read-only, an invalid alias, an alias another library holds). This
//! module only picks which library to ask it about, and falls back to This
//! Mac, with a note, when the default library is the one not connected
//! (2.2). A library the user named is never swapped for another.

use std::path::{Path, PathBuf};

use catalog::library::{LibraryRecord, LocationEnv};
use catalog::{InstallKind, InstallTarget, Store, StoreError};

use super::library::{clean, load, resolve, NAME_LIMIT};
use super::render::truncate;
use crate::{Error, Options};

pub(super) enum Destination {
    /// `--out <DIR>`: a custom location, recorded with no library.
    Out(PathBuf),
    /// This Mac or a library, with the environment that located it, which
    /// recording into the library needs again.
    Target(InstallTarget, LocationEnv),
}

impl Destination {
    /// The install folder.
    pub(super) fn dir(&self) -> &Path {
        match self {
            Self::Out(dir) => dir,
            Self::Target(target, _) => &target.dir,
        }
    }
}

/// Where a pull of `alias` as `kind` lands. Call it before anything that
/// reaches the network.
pub(super) fn destination(
    store: &Store,
    kind: InstallKind,
    alias: &str,
    options: &Options,
) -> Result<Destination, Error> {
    if let Some(out) = &options.out {
        if options.library.is_some() {
            return Err(Error::Usage(
                "--out and --library each name where the install goes; pass one".to_string(),
            ));
        }
        let dir = PathBuf::from(out);
        check_out_dir(&dir)?;
        return Ok(Destination::Out(dir));
    }
    let (target, env) = library_target(store, kind, alias, options.library.as_deref())?;
    Ok(Destination::Target(target, env))
}

/// Top-level names a pull writes into its folder besides the sidecars:
/// the weights, the per-family blob dirs, the install receipt, and the range
/// cache an interrupted download leaves. `.DS_Store` is Finder's, not the
/// user's data.
const OWNED_INSTALL_NAMES: [&str; 7] = [
    "model_weights.bin",
    model_io::PACKED_EXPERTS_DIR,
    model_io::PACKED_VISION_DIR,
    model_io::NGRAM_TABLE_DIR,
    "verified-install.json",
    ".download-cache",
    ".DS_Store",
];

/// Refuses `--out` into a folder holding anything TurboSpark did not put
/// there, before any network request. Some writers publish by swapping a
/// staged folder in for `dir` and deleting the old one whole (the Gemma4
/// streamed path), so a user's own files or subfolders in `dir` would be
/// lost. Allowed: a missing or empty folder, a finished install (it has a
/// `manifest.json`, and replacing one is what a re-pull means), or the
/// leftovers of an interrupted pull (only owned names). Anything else is
/// refused rather than guessed at.
fn check_out_dir(dir: &Path) -> Result<(), Error> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(Error::Failed(format!("reading {}: {error}", dir.display()))),
    };
    let mut foreign = Vec::new();
    for entry in entries {
        let entry =
            entry.map_err(|error| Error::Failed(format!("reading {}: {error}", dir.display())))?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name == "manifest.json" {
            return Ok(());
        }
        if !catalog::OWNED_SIDECAR_NAMES.contains(&name.as_ref())
            && !OWNED_INSTALL_NAMES.contains(&name.as_ref())
        {
            foreign.push(name.into_owned());
        }
    }
    if foreign.is_empty() {
        return Ok(());
    }
    foreign.sort();
    let shown = foreign.len().min(3);
    let more = if foreign.len() > shown {
        format!(" and {} more", foreign.len() - shown)
    } else {
        String::new()
    };
    Err(Error::Usage(format!(
        "--out {} already holds files TurboSpark did not write ({}{more}); an install \
         can replace that folder whole. Pass a new or empty folder.",
        dir.display(),
        foreign[..shown].join(", ")
    )))
}

/// The install target in the library `library` names (a name or an id, as
/// `library default` takes), else in the default library. A default library
/// that is not connected gives way to This Mac, with a note naming it.
pub(super) fn library_target(
    store: &Store,
    kind: InstallKind,
    alias: &str,
    library: Option<&str>,
) -> Result<(InstallTarget, LocationEnv), Error> {
    let env = LocationEnv::system(store.root()).map_err(Error::Failed)?;
    let registry = load(store);
    let (record, named) = match library {
        Some(target) => (resolve(&registry, target)?, true),
        None => (
            registry
                .libraries()
                .into_iter()
                .find(|record| record.default_for_downloads)
                .unwrap_or_else(|| LibraryRecord::internal(store.root())),
            false,
        ),
    };
    let id = (!record.is_internal()).then_some(record.id.as_str());
    let target = match catalog::install_destination(store, id, kind, alias, &env) {
        Err(StoreError::NotConnected { .. }) if !named => {
            eprintln!(
                "note: the default library \"{}\" is not connected; installing into This Mac \
                 instead",
                clean(&record.name, NAME_LIMIT)
            );
            catalog::install_destination(store, None, kind, alias, &env)
        }
        other => other,
    };
    target.map(|target| (target, env)).map_err(failure)
}

/// A failed install step as the command's error, with the library's name,
/// which an adopted library takes from its drive, capped. The catalog's
/// sentences quote it with escapes.
pub(super) fn failure(error: StoreError) -> Error {
    let cap = |library: String| truncate(&library, NAME_LIMIT);
    let error = match error {
        StoreError::NotConnected { library } => StoreError::NotConnected {
            library: cap(library),
        },
        StoreError::LibraryDisconnected { library } => StoreError::LibraryDisconnected {
            library: cap(library),
        },
        StoreError::LibraryReadOnly { library } => StoreError::LibraryReadOnly {
            library: cap(library),
        },
        other => other,
    };
    Error::Failed(error.to_string())
}
