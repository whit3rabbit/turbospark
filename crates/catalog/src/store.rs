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
use std::fs;
use std::io::Write;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::library::{
    locate, LibraryMarker, LibraryRegistry, Located, LocationEnv, INTERNAL_LIBRARY_ID,
};

/// The folder in an external library root that its installs stage in. It is
/// on the destination's drive because publication is a same-volume rename.
const LIBRARY_STAGING_DIR: &str = ".staging";

/// Why a library-aware store operation failed. The store functions that
/// predate libraries keep their `String` errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreError {
    /// The library's drive is not connected. `library` is its name.
    NotConnected {
        library: String,
    },
    /// The library's drive went away during an install, taking the staged
    /// install or the library root with it. Nothing was recorded (2.4).
    LibraryDisconnected {
        library: String,
    },
    /// A newer TurboSpark wrote the library's marker, so this build installs
    /// nothing into it.
    LibraryReadOnly {
        library: String,
    },
    /// Embedding and OCR models install only into This Mac (2.5).
    HomeOnly,
    Io(String),
}

impl std::fmt::Display for StoreError {
    // Names are quoted with escapes: an adopted library's name comes from
    // an untrusted marker and must not put control characters on a terminal.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotConnected { library } => write!(
                f,
                "the library {library:?} is not connected; connect its drive and try again"
            ),
            Self::LibraryDisconnected { library } => write!(
                f,
                "the drive of the library {library:?} was disconnected; the install stopped"
            ),
            Self::LibraryReadOnly { library } => write!(
                f,
                "the library {library:?} was written by a newer TurboSpark and is read-only here"
            ),
            Self::HomeOnly => write!(f, "embedding and OCR models install only into This Mac"),
            Self::Io(reason) => write!(f, "{reason}"),
        }
    }
}

impl std::error::Error for StoreError {}

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
    /// Canonical GGUF quantization label selected for a variant install.
    /// Absent for legacy installs and installs that did not select a live
    /// variant, preserving the earlier serialized shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
    /// The id of the library holding this install. Absent means the internal
    /// library ([`crate::library::INTERNAL_LIBRARY_ID`]), which is what every
    /// row written before libraries existed is. Skipped when absent so an
    /// internal row writes exactly as it did before, for older readers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub library: Option<String>,
    /// The install's path inside its library root. It survives a drive
    /// mounting under a new name, which `path` (the last resolved absolute
    /// path, kept for older readers) does not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relative_path: Option<PathBuf>,
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
    /// Audio models, including speech recognition and music generation.
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

    /// `<library>:<modality>/<alias>`, the key app metadata follows when the
    /// install's location changes. Modality is the effective one so a legacy
    /// image row gets the same id every reader already treats it as having.
    pub fn install_id(&self) -> String {
        format!(
            "{}:{}/{}",
            self.library
                .as_deref()
                .unwrap_or(crate::library::INTERNAL_LIBRARY_ID),
            self.effective_modality().as_str(),
            self.alias
        )
    }
}

/// The `~/.turbospark` tree.
#[derive(Debug, Clone)]
pub struct Store {
    root: PathBuf,
}

static DEFAULT_ROOT_OVERRIDE: RwLock<Option<PathBuf>> = RwLock::new(None);

/// Overrides the process-local default store used by library and FFI callers.
/// The CLI does not call this, so it continues to resolve from the environment.
pub fn set_default_root(root: Option<PathBuf>) -> Result<(), String> {
    let mut slot = DEFAULT_ROOT_OVERRIDE
        .write()
        .map_err(|_| "the model store root lock is poisoned".to_string())?;
    *slot = root;
    Ok(())
}

/// The default store root: `$TURBOSPARK_HOME`, else `$HOME/.turbospark`.
///
/// Returns `None` when neither is set, which is a real condition in a
/// stripped test or CI environment and is reported rather than defaulted to
/// a relative path that would scatter multi-GB installs into whatever the
/// working directory happened to be.
pub fn default_root() -> Option<PathBuf> {
    if let Ok(override_root) = DEFAULT_ROOT_OVERRIDE.read() {
        if let Some(root) = override_root.as_ref() {
            return Some(root.clone());
        }
    }
    if let Some(home) = std::env::var_os("TURBOSPARK_HOME") {
        return Some(PathBuf::from(home));
    }
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".turbospark"))
}

/// Result of moving a managed store to another root.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreRelocation {
    pub source: PathBuf,
    pub destination: PathBuf,
    pub bytes: u64,
    pub files: u64,
}

fn copy_entry(
    source: &Path,
    destination: &Path,
    copied_bytes: &mut u64,
    total_bytes: u64,
    on_progress: &mut impl FnMut(u64, u64),
) -> Result<u64, String> {
    let metadata = std::fs::symlink_metadata(source)
        .map_err(|e| format!("reading {}: {e}", source.display()))?;
    if metadata.is_dir() {
        std::fs::create_dir(destination)
            .map_err(|e| format!("creating {}: {e}", destination.display()))?;
        let mut files = 0;
        for entry in
            std::fs::read_dir(source).map_err(|e| format!("reading {}: {e}", source.display()))?
        {
            let entry = entry.map_err(|e| format!("reading a store entry: {e}"))?;
            files += copy_entry(
                &entry.path(),
                &destination.join(entry.file_name()),
                copied_bytes,
                total_bytes,
                on_progress,
            )?;
        }
        Ok(files)
    } else if metadata.is_file() {
        let mut input = std::fs::File::open(source)
            .map_err(|e| format!("opening {}: {e}", source.display()))?;
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(destination)
            .map_err(|e| {
                format!(
                    "creating destination file {} without replacing an existing entry: {e}",
                    destination.display()
                )
            })?;
        std::io::copy(&mut input, &mut output).map_err(|e| {
            format!(
                "copying {} to {}: {e}",
                source.display(),
                destination.display()
            )
        })?;
        output
            .set_permissions(metadata.permissions())
            .map_err(|e| format!("setting permissions on {}: {e}", destination.display()))?;
        *copied_bytes += metadata.len();
        on_progress(*copied_bytes, total_bytes);
        Ok(1)
    } else {
        Err(format!(
            "unsupported store entry type: {}",
            source.display()
        ))
    }
}

/// Creates `path` at mode 0700 from the start, like the transfer staging
/// folders, so staged weights are never readable by other local users.
fn create_private_dir(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    fs::DirBuilder::new().mode(0o700).create(path)
}

fn managed_entries(root: &Path) -> Vec<PathBuf> {
    // Credentials and authenticated Hub metadata are deliberately not part of
    // model-store relocation. A destination may be removable or shared
    // storage whose access controls are weaker than their private source
    // directory. Hub metadata is a disposable cache and will be rebuilt.
    ["models", "installed.json", "models.json"]
        .into_iter()
        .map(|name| root.join(name))
        .filter(|path| path.exists())
        .collect()
}

fn managed_bytes(root: &Path) -> u64 {
    managed_entries(root)
        .into_iter()
        .map(|path| entry_bytes(&path))
        .sum()
}

fn entry_bytes(path: &Path) -> u64 {
    if path.is_dir() {
        directory_bytes(path)
    } else {
        std::fs::metadata(path)
            .map(|metadata| metadata.len())
            .unwrap_or(0)
    }
}

/// Copies the managed store, verifies its model payload, rewrites in-root
/// install paths, removes the old managed entries, and activates the new root.
/// The destination must be empty so the operation never merges unrelated stores.
pub fn relocate_default_store(
    destination: &Path,
    mut on_progress: impl FnMut(u64, u64),
) -> Result<StoreRelocation, String> {
    let source = default_root().ok_or_else(|| {
        "neither TURBOSPARK_HOME nor HOME is set, so there is no source store".to_string()
    })?;
    let source = source.canonicalize().unwrap_or_else(|_| source.clone());
    let destination = if destination.is_absolute() {
        destination.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| format!("resolving destination: {e}"))?
            .join(destination)
    };
    let destination = destination
        .canonicalize()
        .unwrap_or_else(|_| destination.clone());

    if source == destination {
        return Err("the destination is already the active TurboSpark store".to_string());
    }
    if destination.starts_with(&source) || source.starts_with(&destination) {
        return Err("the source and destination stores cannot contain one another".to_string());
    }
    if destination.exists() {
        let non_empty = std::fs::read_dir(&destination)
            .map_err(|e| format!("reading destination {}: {e}", destination.display()))?
            .next()
            .transpose()
            .map_err(|e| format!("reading destination {}: {e}", destination.display()))?
            .is_some();
        if non_empty {
            return Err(format!(
                "destination {} is not empty",
                destination.display()
            ));
        }
    } else if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("creating destination parent {}: {e}", parent.display()))?;
        std::fs::create_dir(&destination)
            .map_err(|e| format!("creating destination {}: {e}", destination.display()))?;
    }

    // The copy below deliberately creates every child instead of replacing
    // one. Restricting the root first also prevents a different local user
    // from adding entries after the emptiness check. Together these make an
    // attacker-created destination symlink an error, never a write target.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&destination, std::fs::Permissions::from_mode(0o700))
            .map_err(|e| format!("restricting destination {}: {e}", destination.display()))?;
    }

    let total_bytes = managed_bytes(&source);
    let mut copied_bytes = 0;
    let mut source_cleanup_started = false;
    let result = (|| {
        let mut files = 0;
        for entry in managed_entries(&source) {
            let name = entry
                .file_name()
                .ok_or_else(|| format!("invalid store entry {}", entry.display()))?;
            files += copy_entry(
                &entry,
                &destination.join(name),
                &mut copied_bytes,
                total_bytes,
                &mut on_progress,
            )?;
        }

        let source_models = source.join("models");
        let destination_models = destination.join("models");
        if entry_bytes(&source_models) != entry_bytes(&destination_models) {
            return Err("model bytes did not verify after copying".to_string());
        }

        let source_rows = Store::new(&source).read_rows();
        let rewritten: Vec<_> = source_rows
            .into_iter()
            .map(|mut row| {
                let row_path = row.path.canonicalize().unwrap_or_else(|_| row.path.clone());
                if row_path.starts_with(&source) {
                    if let Ok(relative) = row_path.strip_prefix(&source) {
                        row.path = destination.join(relative);
                    }
                }
                row
            })
            .collect();
        if !rewritten.is_empty() {
            let record_path = destination.join("installed.json");
            let text = serde_json::to_string_pretty(&rewritten)
                .map_err(|e| format!("serializing the install record: {e}"))?;
            let mut record = std::fs::OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(&record_path)
                .map_err(|e| format!("opening {} for rewrite: {e}", record_path.display()))?;
            record
                .write_all(text.as_bytes())
                .map_err(|e| format!("writing {}: {e}", record_path.display()))?;
        }
        for row in &rewritten {
            if row.path.starts_with(&destination) && !row.path.is_dir() {
                return Err(format!(
                    "relocated install is missing: {}",
                    row.path.display()
                ));
            }
        }

        // Activate the verified destination before deleting the source. If a
        // source cleanup fails, the new store remains usable and the old
        // partial tree remains available for recovery.
        set_default_root(Some(destination.clone()))?;
        source_cleanup_started = true;
        for entry in managed_entries(&source) {
            if entry.is_dir() {
                std::fs::remove_dir_all(&entry)
                    .map_err(|e| format!("removing {}: {e}", entry.display()))?;
            } else {
                std::fs::remove_file(&entry)
                    .map_err(|e| format!("removing {}: {e}", entry.display()))?;
            }
        }
        let _ = std::fs::remove_dir(&source);
        Ok(StoreRelocation {
            source,
            destination: destination.clone(),
            bytes: total_bytes,
            files,
        })
    })();

    if result.is_err() && !source_cleanup_started {
        let _ = std::fs::remove_dir_all(&destination);
    }
    result
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

    /// `<root>/<modality>/<alias><suffix>`, the layout every library root
    /// shares, This Mac's `<home>/models` included. Unchecked, so the
    /// internal paths keep accepting every alias they always did.
    fn modality_path(root: &Path, modality: ModelModality, alias: &str, suffix: &str) -> PathBuf {
        root.join(modality.as_str())
            .join(format!("{alias}{suffix}"))
    }

    /// [`Self::modality_path`] under a caller-chosen library root, after
    /// refusing an alias that could name a folder outside it (empty, a
    /// leading `.`, or a `/`), the rule marker rows meet. Every path joined
    /// onto an external root goes through here.
    fn checked_modality_path(
        root: &Path,
        modality: ModelModality,
        alias: &str,
        suffix: &str,
    ) -> Result<PathBuf, StoreError> {
        crate::library::validate_alias(alias)
            .map_err(|_| StoreError::Io(format!("{alias:?} is not a valid install alias")))?;
        Ok(Self::modality_path(root, modality, alias, suffix))
    }

    /// Where a text-model pull of `alias` lands by default.
    pub fn install_path(&self, alias: &str) -> PathBuf {
        Self::modality_path(&self.models_root(), ModelModality::Text, alias, ".gturbo")
    }

    /// Where a text-model install of `alias` lands in the library rooted at
    /// `root`, as [`Self::library_root_for_install`] returns it.
    pub fn install_path_in(root: &Path, alias: &str) -> Result<PathBuf, StoreError> {
        Self::checked_modality_path(root, ModelModality::Text, alias, ".gturbo")
    }

    /// Where an image install lands by default. The modality directory, not a
    /// filename suffix, keeps image and text aliases independent.
    pub fn image_install_path(&self, alias: &str) -> PathBuf {
        Self::modality_path(&self.models_root(), ModelModality::Image, alias, ".gturbo")
    }

    /// [`Self::image_install_path`] in the library rooted at `root`.
    pub fn image_install_path_in(root: &Path, alias: &str) -> Result<PathBuf, StoreError> {
        Self::checked_modality_path(root, ModelModality::Image, alias, ".gturbo")
    }

    /// The managed namespace for task-aware audio profiles.
    pub fn audio_models_dir(&self) -> PathBuf {
        self.modality_root(ModelModality::Audio)
    }

    /// Where an audio model install lands by default.
    pub fn audio_install_path(&self, alias: &str) -> PathBuf {
        Self::modality_path(&self.models_root(), ModelModality::Audio, alias, ".gturbo")
    }

    /// [`Self::audio_install_path`] in the library rooted at `root`.
    pub fn audio_install_path_in(root: &Path, alias: &str) -> Result<PathBuf, StoreError> {
        Self::checked_modality_path(root, ModelModality::Audio, alias, ".gturbo")
    }

    /// The namespace for pinned embedding installs (`crate::embedding`).
    /// Not a [`ModelModality`]: embedding installs have no registry row.
    pub fn embedding_models_dir(&self) -> PathBuf {
        self.models_root().join("embeddings")
    }

    /// Where an embedding install of `alias` lands.
    pub fn embedding_install_path(&self, alias: &str) -> PathBuf {
        self.embedding_models_dir().join(alias)
    }

    /// Where a vision-tower sidecar pull of `alias` lands by default. Towers
    /// are text-model accessories, so they stay in the text namespace.
    pub fn vision_install_path(&self, alias: &str) -> PathBuf {
        Self::modality_path(
            &self.models_root(),
            ModelModality::Text,
            alias,
            ".gturbo-vision",
        )
    }

    /// [`Self::vision_install_path`] in the library rooted at `root`.
    pub fn vision_install_path_in(root: &Path, alias: &str) -> Result<PathBuf, StoreError> {
        Self::checked_modality_path(root, ModelModality::Text, alias, ".gturbo-vision")
    }

    /// Refuses a destination library for the kinds that install only into
    /// This Mac, embedding and OCR models (2.5). Their paths stay home-only,
    /// so a caller that takes a destination checks it here before anything
    /// else, and an unlisted id is refused the same as a listed one.
    pub fn home_only(library: Option<&str>) -> Result<(), StoreError> {
        match library {
            None | Some(INTERNAL_LIBRARY_ID) => Ok(()),
            Some(_) => Err(StoreError::HomeOnly),
        }
    }

    /// The library root an install into `library` goes under: This Mac's
    /// `<home>/models` for `None` or [`INTERNAL_LIBRARY_ID`], else the
    /// canonical root of the listed library with that id. Choosing the
    /// default library is the caller's business.
    ///
    /// `env` is a parameter rather than a field of `Store` (a refinement of
    /// the design, which had the store carry it): the store is built on
    /// server request paths that must never read the mount table, and only
    /// this lookup needs it.
    ///
    /// A library that is not connected, or whose marker at its canonical
    /// root is gone, unreadable, or another library's, is
    /// [`StoreError::NotConnected`]. One a newer TurboSpark wrote is
    /// [`StoreError::LibraryReadOnly`].
    pub fn library_root_for_install(
        &self,
        library: Option<&str>,
        env: &LocationEnv,
    ) -> Result<PathBuf, StoreError> {
        let id = match library {
            None | Some(INTERNAL_LIBRARY_ID) => return Ok(self.models_root()),
            Some(id) => id,
        };
        let record = LibraryRegistry::load(&self.root)
            .libraries()
            .into_iter()
            .find(|record| record.id == id)
            .ok_or_else(|| StoreError::Io(format!("no library with the id {id:?} is listed")))?;
        let not_connected = || StoreError::NotConnected {
            library: record.name.clone(),
        };
        let located = match locate(&record, env) {
            Located::Available { root, .. } => root,
            Located::NotConnected => return Err(not_connected()),
        };
        // `locate` builds the root as the volume joined with the relative
        // path, so the volume is that many levels up.
        let depth = record.volume_relative_path.components().count();
        let volume = located
            .ancestors()
            .nth(depth)
            .and_then(|volume| volume.canonicalize().ok())
            .ok_or_else(not_connected)?;
        let root = located.canonicalize().map_err(|_| not_connected())?;
        // Adding a library refuses a linked folder, but a drive anyone could
        // have written may have swapped one in since. Installs never follow
        // a root off its drive.
        let device = |path: &Path| fs::metadata(path).map(|found| found.dev()).ok();
        if !root.starts_with(&volume) || device(&root) != device(&volume) {
            return Err(StoreError::Io(format!(
                "the library {:?} at {} leads off its drive",
                record.name,
                located.display()
            )));
        }
        // The marker is read again at the canonical root, so the path an
        // install joins onto is the one that carries this library's id.
        match LibraryMarker::read(&root) {
            Ok(Some(marker)) if marker.id == record.id => {
                if marker.is_read_only() {
                    return Err(StoreError::LibraryReadOnly {
                        library: record.name.clone(),
                    });
                }
                Ok(root)
            }
            // Gone or replaced since it was located, or unreadable, as a
            // drive being unplugged can read.
            _ => Err(not_connected()),
        }
    }

    /// Whether `root` is This Mac's `<home>/models`, by spelling or, when
    /// both exist, by identity.
    fn is_internal_root(&self, root: &Path) -> bool {
        let internal = self.models_root();
        root == internal
            || matches!(
                (root.canonicalize(), internal.canonicalize()),
                (Ok(given), Ok(internal)) if given == internal
            )
    }

    /// Whether an install into `library_root` has lost its drive: the
    /// external root, or `staging` (the staged install or staging folder it
    /// was using), can no longer be examined. Anything that cannot be
    /// examined counts as gone, since a drive being unplugged can answer
    /// with an IO error before it disappears. Never true for This Mac: the
    /// home does not unplug, so a folder missing there is an IO error.
    fn library_gone(&self, staging: &Path, library_root: &Path) -> bool {
        let vanished = |path: &Path| fs::symlink_metadata(path).is_err();
        !self.is_internal_root(library_root) && (vanished(staging) || vanished(library_root))
    }

    /// The error for an install step into `library_root` that failed with
    /// `error`, where `staging` is the staged install or staging folder the
    /// step used and `library` names the library:
    /// [`StoreError::LibraryDisconnected`] when the staging or the external
    /// root is gone (see [`Self::library_gone`]), else [`StoreError::Io`].
    ///
    /// The one rule for every install step, so a disconnect reads the same
    /// whichever step saw it: [`Self::publish_staged`] applies it, and the
    /// library installers apply it to a failed transfer
    /// (`HubError::Staging`) and a failed [`Self::staging_parent_for`].
    pub(crate) fn install_failure(
        &self,
        staging: &Path,
        library_root: &Path,
        library: &str,
        error: impl Into<String>,
    ) -> StoreError {
        if self.library_gone(staging, library_root) {
            StoreError::LibraryDisconnected {
                library: library.to_string(),
            }
        } else {
            StoreError::Io(error.into())
        }
    }

    /// The folder that pinned transfers into the library root
    /// `library_root`, and every install into an external library, stage
    /// under, with `library_root` as [`Self::library_root_for_install`]
    /// returns it.
    ///
    /// This Mac's is the home, the parent internal pinned transfers have
    /// always staged under. Internal installs keep their current staging
    /// locations, including image sources under `<home>/models/image`, so
    /// they do not change. An external library's is `<root>/.staging`,
    /// created with mode 0700 on first use, so staging shares the
    /// destination's volume and publication is a rename. A `.staging` that
    /// is a symlink, not a folder, or on another device is refused rather
    /// than followed: a drive anyone could have written must not be able to
    /// point an install's writes elsewhere.
    pub fn staging_parent_for(&self, library_root: &Path) -> Result<PathBuf, StoreError> {
        if self.is_internal_root(library_root) {
            return Ok(self.root.clone());
        }
        let io = |action: &str, path: &Path, e: std::io::Error| {
            StoreError::Io(format!("{action} {}: {e}", path.display()))
        };
        // Canonical, so a `.staging` that is no link is inside the library
        // by construction.
        let root = library_root
            .canonicalize()
            .map_err(|e| io("resolving the library root", library_root, e))?;
        let staging = root.join(LIBRARY_STAGING_DIR);
        match create_private_dir(&staging) {
            Err(e) if e.kind() != std::io::ErrorKind::AlreadyExists => {
                return Err(io("creating", &staging, e));
            }
            _ => {}
        }
        // Examined after creating, so a link planted in between is caught.
        let found = fs::symlink_metadata(&staging).map_err(|e| io("reading", &staging, e))?;
        if !found.file_type().is_dir() {
            return Err(StoreError::Io(format!(
                "{} is not a plain folder",
                staging.display()
            )));
        }
        let root_device = fs::metadata(&root)
            .map_err(|e| io("reading", &root, e))?
            .dev();
        if found.dev() != root_device {
            return Err(StoreError::Io(format!(
                "{} is on another drive than its library",
                staging.display()
            )));
        }
        Ok(staging)
    }

    /// Moves the staged install `staged` to `destination`, an install path
    /// from one of the `*_path_in` helpers, creating its modality folder if
    /// needed. `library` names the library for the error.
    ///
    /// Records nothing: the caller records the row only after this succeeds,
    /// so an install that fails here leaves no row (2.4). Failures are
    /// classified by [`Self::install_failure`]: a missing `staged` or a
    /// vanished root of an external library is
    /// [`StoreError::LibraryDisconnected`], since an unplugged drive takes
    /// both with it, and the root is never recreated. In This Mac a missing
    /// `staged` is an IO error. An existing `destination`, even an empty
    /// folder that a rename would quietly replace, and a modality folder
    /// that is a link are refused.
    pub fn publish_staged(
        &self,
        staged: &Path,
        destination: &Path,
        library: &str,
    ) -> Result<(), StoreError> {
        fn parent(path: &Path) -> Option<&Path> {
            path.parent().filter(|p| !p.as_os_str().is_empty())
        }
        let Some((modality_dir, library_root)) = parent(destination)
            .and_then(|modality_dir| Some((modality_dir, parent(modality_dir)?)))
        else {
            return Err(StoreError::Io(format!(
                "{} is not an install path",
                destination.display()
            )));
        };
        // This Mac's root is on the Mac and may not exist yet in a fresh
        // home. Only an external root can vanish with its drive.
        let internal = self.is_internal_root(library_root);
        let failed = |message: String| self.install_failure(staged, library_root, library, message);
        // Before anything is created, so a drive that is not the one the
        // install started on, remounted under the same name, is left as it is.
        if fs::symlink_metadata(staged).is_err() || self.library_gone(staged, library_root) {
            return Err(failed(format!(
                "the staged install {} cannot be read",
                staged.display()
            )));
        }
        if fs::symlink_metadata(destination).is_ok() {
            return Err(StoreError::Io(format!(
                "{} already exists, and publishing never replaces an install",
                destination.display()
            )));
        }
        // Not `create_dir_all` for an external root: with the drive gone it
        // would rebuild the library root on the disk under the mount point.
        let created = if internal {
            fs::create_dir_all(modality_dir)
        } else {
            fs::create_dir(modality_dir)
        };
        match created {
            Err(e) if e.kind() != std::io::ErrorKind::AlreadyExists => {
                return Err(failed(format!("creating {}: {e}", modality_dir.display())));
            }
            _ => {}
        }
        // Not followed: a linked modality folder would put the install
        // outside the library.
        if !fs::symlink_metadata(modality_dir).is_ok_and(|found| found.file_type().is_dir()) {
            return Err(failed(format!(
                "{} is not a plain folder",
                modality_dir.display()
            )));
        }
        fs::rename(staged, destination).map_err(|e| {
            failed(format!(
                "moving {} to {}: {e}",
                staged.display(),
                destination.display()
            ))
        })
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
    pub(crate) fn read_rows(&self) -> Vec<InstalledModel> {
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

    /// Raw Audio records preserve their original paths for explicit ownership validation.
    pub fn audio_records(&self) -> Vec<InstalledModel> {
        self.read_rows()
            .into_iter()
            .filter(|row| row.effective_modality() == ModelModality::Audio)
            .collect()
    }

    /// Recorded audio installs under this root, including after relocation.
    pub fn installed_audio(&self) -> BTreeMap<String, InstalledModel> {
        self.read_rows()
            .into_iter()
            .filter(|row| row.effective_modality() == ModelModality::Audio)
            .filter_map(|mut row| {
                row.path = self.audio_install_path(&row.alias);
                row.path.is_dir().then_some((row.alias.clone(), row))
            })
            .collect()
    }

    /// Forget one audio row without touching text or image aliases.
    pub fn forget_audio(&self, alias: &str) -> Result<(), String> {
        let rows: Vec<_> = self
            .read_rows()
            .into_iter()
            .filter(|row| !(row.alias == alias && row.effective_modality() == ModelModality::Audio))
            .collect();
        self.write_rows(&rows)
    }

    /// Add or replace a row, then rewrite the record.
    ///
    /// Only the row with the same alias and modality is replaced. A row whose
    /// directory is missing is kept: it may sit on an unmounted external
    /// volume, and pruning it here would forget that install for good the
    /// next time anything else is recorded. Readers such as
    /// [`Self::installed`] hide missing rows instead.
    pub fn record(&self, model: &InstalledModel) -> Result<(), String> {
        self.record_all(std::slice::from_ref(model))
    }

    /// [`Self::record`] for several rows in one write, so either every row
    /// lands or none does, as adopting a library's rows needs.
    pub(crate) fn record_all(&self, models: &[InstalledModel]) -> Result<(), String> {
        let mut rows = self.read_rows();
        for model in models {
            let modality = model.effective_modality();
            rows.retain(|row| !(row.alias == model.alias && row.effective_modality() == modality));
            rows.push(model.clone());
        }
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

    /// Drop every row of the library `library`, as removing that library
    /// does. Its files and marker are left alone, so re-adding the library
    /// restores the rows from the marker.
    pub(crate) fn forget_library_rows(&self, library: &str) -> Result<(), String> {
        let rows = self.read_rows();
        let kept: Vec<InstalledModel> = rows
            .iter()
            .filter(|row| row.library.as_deref() != Some(library))
            .cloned()
            .collect();
        // No rewrite when nothing changed, so removing an empty library never
        // creates or touches the record.
        if kept.len() == rows.len() {
            return Ok(());
        }
        self.write_rows(&kept)
    }

    /// Replaces the record through a temp file and a rename. A corrupt record
    /// reads as empty, so a write a crash cut short would make the next
    /// `record` drop every other install (6.4).
    fn write_rows(&self, rows: &[InstalledModel]) -> Result<(), String> {
        std::fs::create_dir_all(&self.root)
            .map_err(|e| format!("creating {}: {e}", self.root.display()))?;
        let text = serde_json::to_string_pretty(rows)
            .map_err(|e| format!("serializing the install record: {e}"))?;
        crate::library::replace_file(&self.record_path(), text.as_bytes())
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

    /// Resolve a pinned embedding alias to its install directory. Only a
    /// plain alias is considered, so a path never lands here.
    pub fn resolve_embedding(&self, name: &str) -> Option<PathBuf> {
        if name.is_empty() || name.contains('/') || name.starts_with('.') {
            return None;
        }
        let dir = self.embedding_install_path(name);
        dir.join(crate::embedding::RECEIPT_FILENAME)
            .is_file()
            .then_some(dir)
    }

    /// Resolve an audio alias or explicit path.
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
            .or_else(|| store.resolve_embedding(name))
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
