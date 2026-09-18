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
    /// [`crate::EntryKind::as_str`]'s spelling ("vision-tower"), or `None`.
    ///
    /// **Absence means a regular trunk model**, which is AGENTS.md Gotcha
    /// 39's rule: every `installed.json` row written before this field
    /// existed is exactly that, and `#[serde(default)]` is what lets those
    /// rows keep deserializing rather than failing to parse the whole file.
    #[serde(default)]
    pub kind: Option<String>,
    /// The runtime modality. Missing means text for rows written before the
    /// modality-aware store existed. A legacy image row is recognized from
    /// `kind == "image"` by [`Self::effective_modality`].
    #[serde(default)]
    pub modality: ModelModality,
}

/// The user-facing model namespace under the shared TurboSpark store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ModelModality {
    /// Chat, completion, embedding, and vision-enabled text models.
    #[default]
    Text,
    /// Diffusion and other image-generation artifacts.
    Image,
    /// Audio models, reserved for the transcription runtime.
    Audio,
}

impl ModelModality {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Audio => "audio",
        }
    }
}

impl InstalledModel {
    /// Interprets rows written before `modality` existed.
    pub fn effective_modality(&self) -> ModelModality {
        if self.kind.as_deref() == Some("image") {
            ModelModality::Image
        } else {
            self.modality
        }
    }
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

    /// The shared parent for all modality-specific model directories.
    pub fn models_root(&self) -> PathBuf {
        self.root.join("models")
    }

    /// The canonical directory for one modality.
    pub fn modality_root(&self, modality: ModelModality) -> PathBuf {
        self.models_root().join(modality.as_str())
    }

    /// Where a text-model pull of `alias` lands by default.
    pub fn install_path(&self, alias: &str) -> PathBuf {
        self.modality_root(ModelModality::Text)
            .join(format!("{alias}.gturbo"))
    }

    /// Where an image install lands by default. The modality directory, not a
    /// filename suffix, keeps image and text aliases independent.
    pub fn image_install_path(&self, alias: &str) -> PathBuf {
        self.modality_root(ModelModality::Image)
            .join(format!("{alias}.gturbo"))
    }

    /// Where a future audio install lands by default.
    pub fn audio_install_path(&self, alias: &str) -> PathBuf {
        self.modality_root(ModelModality::Audio)
            .join(format!("{alias}.gturbo"))
    }

    /// Where a vision-tower sidecar pull of `alias` lands by default. Towers
    /// are text-model accessories, so they stay in the text namespace.
    pub fn vision_install_path(&self, alias: &str) -> PathBuf {
        self.modality_root(ModelModality::Text)
            .join(format!("{alias}.gturbo-vision"))
    }

    fn legacy_install_path(&self, alias: &str) -> PathBuf {
        self.models_root().join(format!("{alias}.gturbo"))
    }

    fn legacy_image_install_path(&self, alias: &str) -> PathBuf {
        self.models_root().join(format!("{alias}.image.gturbo"))
    }

    fn legacy_vision_install_path(&self, alias: &str) -> PathBuf {
        self.models_root().join(format!("{alias}.gturbo-vision"))
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

    /// Write the Hugging Face token to this store's `hf_token` file, at mode
    /// 0600 from the moment it exists (Unix): the file is created with that
    /// mode already set, rather than written at the process umask and
    /// tightened afterward, so there is no window -- however brief -- where
    /// a secret sits world-readable on a shared machine. A failure to apply
    /// the mode is a hard error rather than an ignored one, since silently
    /// leaving the file at the default mode is exactly the outcome this
    /// exists to prevent.
    pub fn set_hf_token(&self, token: &str) -> Result<(), String> {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            return self.clear_hf_token();
        }
        std::fs::create_dir_all(&self.root)
            .map_err(|e| format!("creating store directory {}: {e}", self.root.display()))?;
        let path = self.hf_token_path();
        #[cfg(unix)]
        {
            use std::io::Write;
            use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(&path)
                .map_err(|e| format!("opening {} at mode 0600: {e}", path.display()))?;
            // `mode()` above only takes effect when this call is what
            // CREATES the file; a token file left over from a build before
            // this fix would otherwise keep its old, more permissive mode
            // untouched. Setting it again here operates on the already-open
            // handle rather than the path, so there is no reopen and no
            // window between checking and acting on it either.
            file.set_permissions(std::fs::Permissions::from_mode(0o600))
                .map_err(|e| format!("restricting {} to mode 0600: {e}", path.display()))?;
            file.write_all(format!("{trimmed}\n").as_bytes())
                .map_err(|e| format!("writing Hugging Face token to {}: {e}", path.display()))?;
        }
        #[cfg(not(unix))]
        {
            std::fs::write(&path, format!("{trimmed}\n"))
                .map_err(|e| format!("writing Hugging Face token to {}: {e}", path.display()))?;
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
    fn read_rows(&self) -> Vec<InstalledModel> {
        let text = match std::fs::read_to_string(self.record_path()) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        serde_json::from_str(&text).unwrap_or_default()
    }

    /// Every recorded text install whose directory still exists, alias order.
    /// Image rows are deliberately excluded even if an older CLI wrote one
    /// into this shared record.
    pub fn installed(&self) -> BTreeMap<String, InstalledModel> {
        self.read_rows()
            .into_iter()
            .filter(|r| r.path.is_dir() && r.effective_modality() == ModelModality::Text)
            .map(|r| (r.alias.clone(), r))
            .collect()
    }

    /// Add or replace a row, then rewrite the record.
    pub fn record(&self, model: &InstalledModel) -> Result<(), String> {
        let modality = model.effective_modality();
        let mut rows: Vec<InstalledModel> = self
            .read_rows()
            .into_iter()
            .filter(|row| {
                row.path.is_dir()
                    && !(row.alias == model.alias && row.effective_modality() == modality)
            })
            .collect();
        rows.push(model.clone());
        self.write_rows(&rows)
    }

    /// Drop a row. Does not touch the install directory; deleting that is
    /// the caller's decision and its confirmation prompt.
    pub fn forget(&self, alias: &str) -> Result<(), String> {
        let rows: Vec<InstalledModel> = self
            .read_rows()
            .into_iter()
            .filter(|row| !(row.alias == alias && row.effective_modality() == ModelModality::Text))
            .collect();
        self.write_rows(&rows)
    }

    fn write_rows(&self, rows: &[InstalledModel]) -> Result<(), String> {
        std::fs::create_dir_all(&self.root)
            .map_err(|e| format!("creating {}: {e}", self.root.display()))?;
        let text = serde_json::to_string_pretty(rows)
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
        self.resolve_text(name)
    }

    /// Resolve a text alias or explicit path. Image and audio aliases are
    /// intentionally not considered.
    pub fn resolve_text(&self, name: &str) -> Option<PathBuf> {
        let as_path = PathBuf::from(name);
        if as_path.is_dir() {
            return Some(as_path);
        }
        if let Some(row) = self.installed().get(name) {
            return Some(row.path.clone());
        }
        let default = self.install_path(name);
        if default.is_dir() {
            return Some(default);
        }
        let vision_default = self.vision_install_path(name);
        if vision_default.is_dir() {
            return Some(vision_default);
        }
        let legacy = self.legacy_install_path(name);
        if legacy.is_dir() {
            return Some(legacy);
        }
        let legacy_vision = self.legacy_vision_install_path(name);
        legacy_vision.is_dir().then_some(legacy_vision)
    }

    /// Resolve an image alias or explicit path. Text and audio aliases are
    /// intentionally not considered.
    pub fn resolve_image(&self, name: &str) -> Option<PathBuf> {
        let as_path = PathBuf::from(name);
        if as_path.is_dir() {
            return Some(as_path);
        }
        let canonical = self.image_install_path(name);
        if canonical.is_dir() {
            return Some(canonical);
        }
        let legacy = self.legacy_image_install_path(name);
        legacy.is_dir().then_some(legacy)
    }

    /// Resolve an audio alias or explicit path. Audio loading is reserved for
    /// the transcription runtime, but its namespace is defined now.
    pub fn resolve_audio(&self, name: &str) -> Option<PathBuf> {
        let as_path = PathBuf::from(name);
        if as_path.is_dir() {
            return Some(as_path);
        }
        let canonical = self.audio_install_path(name);
        canonical.is_dir().then_some(canonical)
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
        Ok(store) => store
            .resolve_text(name)
            .unwrap_or_else(|| PathBuf::from(name)),
        Err(_) => PathBuf::from(name),
    }
}

/// Resolve an image alias against the default store, preserving explicit
/// paths and the old flat image location.
pub fn resolve_image_arg(name: &str) -> PathBuf {
    match Store::default_store() {
        Ok(store) => store
            .resolve_image(name)
            .unwrap_or_else(|| PathBuf::from(name)),
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
