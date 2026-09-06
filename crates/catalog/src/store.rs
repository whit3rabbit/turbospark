//! Where pulled installs live, what is recorded about them, and how a bare
//! alias becomes a path.
//!
//! **`resolve` prefers an existing DIRECTORY over an alias, and the order is
//! load-bearing rather than a tie-break.** `turbospark-check --model` has
//! always taken a path, every gate env var in this repo points at one, and a
//! path that silently resolved to an alias of the same name would run a
//! DIFFERENT model than the one named on the command line -- fluently, with
//! no error. A path wins; an alias is the fallback for a string that is not a
//! directory.
//!
//! Nothing here writes into an install directory. The store owns
//! `installed.json` beside the models, so deleting an install by hand leaves
//! a stale row rather than a corrupt one, and [`Store::installed`] filters
//! rows whose directory has gone.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// One recorded install.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledModel {
    pub alias: String,
    /// The weights repository, `owner/name`.
    pub repo: String,
    /// The revision streamed, which for a floating row is the literal
    /// `main` and therefore says less than it looks like it does.
    pub revision: String,
    /// Where the `.gturbo` directory is. Absolute.
    pub path: PathBuf,
    /// The `.gturbo` manifest family the walk resolved, which is evidence
    /// about the artifact rather than a restatement of the catalog row.
    pub family: String,
    /// Bytes on disk after the walk, measured rather than estimated.
    pub install_bytes: u64,
    /// When it was written, as `YYYY-MM-DD`. Whole days only: the field
    /// exists to answer "how old is this", and this crate has no clock
    /// dependency worth adding for more precision.
    pub installed_on: String,
    /// The catalog status at install time, or `unlisted` for a `--repo`
    /// pull. Recorded rather than looked up later, because a row's status
    /// can change under an install that did not.
    pub status: String,
}

/// The `~/.turbospark` tree.
#[derive(Debug, Clone)]
pub struct Store {
    root: PathBuf,
}

/// The default store root: `$TURBOSPARK_HOME`, else `$HOME/.turbospark`.
///
/// Returns `None` when neither is set, which is a real condition in a
/// stripped test or CI environment and is reported rather than defaulted to
/// a relative path that would scatter multi-GB installs into whatever the
/// working directory happened to be.
pub fn default_root() -> Option<PathBuf> {
    if let Some(home) = std::env::var_os("TURBOSPARK_HOME") {
        return Some(PathBuf::from(home));
    }
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".turbospark"))
}

impl Store {
    /// Creates a new model store rooted at the given path.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The store at [`default_root`].
    pub fn default_store() -> Result<Self, String> {
        default_root().map(Self::new).ok_or_else(|| {
            "neither TURBOSPARK_HOME nor HOME is set, so there is no default store; \
             pass --out <dir>"
                .to_string()
        })
    }

    /// The root directory of this store.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Where a pull of `alias` lands by default.
    pub fn install_path(&self, alias: &str) -> PathBuf {
        self.root.join("models").join(format!("{alias}.gturbo"))
    }

    fn record_path(&self) -> PathBuf {
        self.root.join("installed.json")
    }

    /// Where the stored Hugging Face token lives.
    pub fn hf_token_path(&self) -> PathBuf {
        self.root.join("hf_token")
    }

    /// Read the stored Hugging Face token, if present and non-empty.
    pub fn get_hf_token(&self) -> Option<String> {
        let path = self.hf_token_path();
        std::fs::read_to_string(&path)
            .ok()
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
    }

    /// Write the Hugging Face token to this store's `hf_token` file.
    pub fn set_hf_token(&self, token: &str) -> Result<(), String> {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            return self.clear_hf_token();
        }
        std::fs::create_dir_all(&self.root)
            .map_err(|e| format!("creating store directory {}: {e}", self.root.display()))?;
        let path = self.hf_token_path();
        std::fs::write(&path, format!("{trimmed}\n"))
            .map_err(|e| format!("writing Hugging Face token to {}: {e}", path.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
        }
        Ok(())
    }

    /// Delete the stored Hugging Face token file.
    pub fn clear_hf_token(&self) -> Result<(), String> {
        let path = self.hf_token_path();
        if path.exists() {
            std::fs::remove_file(&path)
                .map_err(|e| format!("removing token file {}: {e}", path.display()))?;
        }
        Ok(())
    }

    /// Every recorded install whose directory still exists, alias order.
    ///
    /// A missing or unreadable record is an EMPTY list rather than an error:
    /// the record is a convenience index over directories that are
    /// self-describing, so a corrupt one must not stop `pull` from writing a
    /// new install.
    pub fn installed(&self) -> BTreeMap<String, InstalledModel> {
        let text = match std::fs::read_to_string(self.record_path()) {
            Ok(t) => t,
            Err(_) => return BTreeMap::new(),
        };
        let rows: Vec<InstalledModel> = serde_json::from_str(&text).unwrap_or_default();
        rows.into_iter()
            .filter(|r| r.path.is_dir())
            .map(|r| (r.alias.clone(), r))
            .collect()
    }

    /// Add or replace a row, then rewrite the record.
    pub fn record(&self, model: &InstalledModel) -> Result<(), String> {
        let mut rows = self.installed();
        rows.insert(model.alias.clone(), model.clone());
        self.write_rows(&rows)
    }

    /// Drop a row. Does not touch the install directory; deleting that is
    /// the caller's decision and its confirmation prompt.
    pub fn forget(&self, alias: &str) -> Result<(), String> {
        let mut rows = self.installed();
        rows.remove(alias);
        self.write_rows(&rows)
    }

    fn write_rows(&self, rows: &BTreeMap<String, InstalledModel>) -> Result<(), String> {
        std::fs::create_dir_all(&self.root)
            .map_err(|e| format!("creating {}: {e}", self.root.display()))?;
        let list: Vec<&InstalledModel> = rows.values().collect();
        let text = serde_json::to_string_pretty(&list)
            .map_err(|e| format!("serializing the install record: {e}"))?;
        std::fs::write(self.record_path(), text)
            .map_err(|e| format!("writing {}: {e}", self.record_path().display()))
    }

    /// Turn `name` into an install directory: an existing directory wins,
    /// then a recorded alias, then a default install path that happens to
    /// exist (so a store populated by hand still resolves).
    ///
    /// See the module header for why the first arm is first.
    pub fn resolve(&self, name: &str) -> Option<PathBuf> {
        let as_path = PathBuf::from(name);
        if as_path.is_dir() {
            return Some(as_path);
        }
        if let Some(row) = self.installed().get(name) {
            return Some(row.path.clone());
        }
        let default = self.install_path(name);
        default.is_dir().then_some(default)
    }
}

/// [`Store::resolve`] against the default store, falling back to the raw
/// string when there is no store at all.
///
/// This is what `turbospark-check` and `turbospark-server` both call -- one
/// resolution for both binaries, so an install answers to one name whichever
/// opens it -- and it must NEVER fail a run that used to work: a machine with
/// no `HOME` still has paths, so an unresolvable name comes back unchanged
/// and the caller reports the same "no such install" it always did.
pub fn resolve_model_arg(name: &str) -> PathBuf {
    match Store::default_store() {
        Ok(store) => store.resolve(name).unwrap_or_else(|| PathBuf::from(name)),
        Err(_) => PathBuf::from(name),
    }
}

/// Total bytes of a directory tree, following no symlinks and counting file
/// lengths rather than block usage.
///
/// Used to record what an install actually cost, which is the only honest
/// version of that number: the catalog's `install_bytes` is an estimate kept
/// deliberately generous for the free-space check.
pub fn directory_bytes(dir: &Path) -> u64 {
    let mut total = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(at) = stack.pop() {
        let entries = match std::fs::read_dir(&at) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            match entry.file_type() {
                Ok(t) if t.is_dir() => stack.push(entry.path()),
                Ok(t) if t.is_file() => {
                    total += entry.metadata().map(|m| m.len()).unwrap_or(0);
                }
                _ => {}
            }
        }
    }
    total
}
