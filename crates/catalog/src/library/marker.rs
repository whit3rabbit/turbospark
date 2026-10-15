//! The library marker, `<library root>/library.json`.
//!
//! The marker makes a library self-describing: it carries the library's id
//! and name and a portable copy of its install rows, so a drive that is
//! renamed, remounted, or connected to another Mac is recognized and its
//! models restored without the home registry.
//!
//! A marker sits on a removable drive anyone could have written, so it is
//! untrusted input. Reading caps its size, refuses anything but a regular
//! file, and rejects ids, aliases, and relative paths that could escape the
//! library root or collide with the internal library. Nothing here opens a
//! path a row names.

use super::INTERNAL_LIBRARY_ID;
use crate::{InstalledModel, ModelModality};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// The marker's file name inside a library root.
pub const MARKER_FILE: &str = "library.json";

/// The format this build reads and writes. A marker with a greater format
/// is read-only here, because rewriting it would drop the fields this build
/// does not know.
const MARKER_FORMAT: u32 = 1;

/// The largest marker read or written. Thousands of rows fit well below it,
/// and it bounds what a hostile drive can make the reader allocate.
const MAX_MARKER_BYTES: u64 = 1024 * 1024;

/// Keeps concurrent writers in one process off the same temp file.
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// One install as a marker stores it: an [`InstalledModel`] without the
/// absolute `path` and the `library` id, which the marker's location and id
/// supply when the row is restored.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PortableRow {
    pub alias: String,
    pub repo: String,
    pub revision: String,
    /// The install's path inside the library root, for example
    /// `text/qwen.gturbo`. Required here, unlike on a registry row.
    pub relative_path: PathBuf,
    pub family: String,
    pub install_bytes: u64,
    pub installed_on: String,
    pub status: String,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub modality: ModelModality,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
}

/// A parsed `library.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LibraryMarker {
    /// Private so a read-only marker cannot be relabeled as the current
    /// format and rewritten.
    format: u32,
    pub id: String,
    pub name: String,
    /// `YYYY-MM-DD`, whole days like `installed_on`.
    pub created_on: String,
    pub models: Vec<PortableRow>,
}

/// Why a marker could not be read, written, or converted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MarkerError {
    /// Over the 1 MiB cap. For a read, `bytes` is the file's length, or one
    /// byte past the cap when the file grew after its stat.
    TooLarge {
        bytes: u64,
    },
    /// `library.json` is a symlink, folder, or other non-regular file.
    NotAFile,
    /// Not marker JSON, or format 0.
    Malformed(String),
    /// Empty, the internal library's id, or containing `:` or `/`. The id is
    /// embedded in `install_id` as `<library>:<modality>/<alias>`.
    InvalidId(String),
    /// Empty, starting with `.`, or containing `/`. Store paths are built
    /// from aliases, so such an alias could name a folder outside the
    /// library.
    InvalidAlias(String),
    /// A relative path that is empty or has a root, prefix, `.`, or `..`
    /// component, or an install path outside the library root.
    UnsafePath(PathBuf),
    /// Rewriting a newer-format marker would downgrade it.
    ReadOnly {
        format: u32,
    },
    Io(String),
}

impl std::fmt::Display for MarkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLarge { bytes } => write!(
                f,
                "the library marker is {bytes} bytes, over the {MAX_MARKER_BYTES} byte limit"
            ),
            Self::NotAFile => write!(f, "the library marker is not a regular file"),
            Self::Malformed(reason) => write!(f, "the library marker is malformed: {reason}"),
            Self::InvalidId(id) => write!(f, "the library marker has an invalid id {id:?}"),
            Self::InvalidAlias(alias) => {
                write!(f, "the library marker has an invalid alias {alias:?}")
            }
            Self::UnsafePath(path) => write!(
                f,
                "the install path {} is not inside the library",
                path.display()
            ),
            Self::ReadOnly { format } => write!(
                f,
                "the library marker is format {format}, written by a newer TurboSpark"
            ),
            Self::Io(reason) => write!(f, "{reason}"),
        }
    }
}

impl LibraryMarker {
    /// A current-format marker with no models, created today.
    pub fn new(id: String, name: String) -> Self {
        Self {
            format: MARKER_FORMAT,
            id,
            name,
            created_on: crate::install::today(),
            models: Vec::new(),
        }
    }

    pub fn format(&self) -> u32 {
        self.format
    }

    /// Whether a newer TurboSpark wrote this marker. It can be listed and
    /// its models restored, but [`Self::write`] refuses it.
    pub fn is_read_only(&self) -> bool {
        self.format > MARKER_FORMAT
    }

    /// The marker in `root`, or `Ok(None)` when there is none, so a caller
    /// can tell an unmarked folder from a marker it must not overwrite.
    pub fn read(root: &Path) -> Result<Option<Self>, MarkerError> {
        let path = root.join(MARKER_FILE);
        // Not followed: a symlink could point out of the library, and a FIFO
        // would block the open forever.
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(io_error("reading", &path, e)),
        };
        if !metadata.file_type().is_file() {
            return Err(MarkerError::NotAFile);
        }
        // Refuses an oversized marker without reading it.
        if metadata.len() > MAX_MARKER_BYTES {
            return Err(MarkerError::TooLarge {
                bytes: metadata.len(),
            });
        }
        let file = File::open(&path).map_err(|e| io_error("reading", &path, e))?;
        let bytes = read_bounded(file)?;
        let marker: Self =
            serde_json::from_slice(&bytes).map_err(|e| MarkerError::Malformed(e.to_string()))?;
        marker.validate()?;
        Ok(Some(marker))
    }

    /// Writes `<root>/library.json` through a temp file and a rename inside
    /// `root`. Does not create `root`.
    pub fn write(&self, root: &Path) -> Result<(), MarkerError> {
        if self.is_read_only() {
            return Err(MarkerError::ReadOnly {
                format: self.format,
            });
        }
        // Never write a marker this build would refuse to read back.
        self.validate()?;
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|e| MarkerError::Io(format!("serializing the library marker: {e}")))?;
        if bytes.len() as u64 > MAX_MARKER_BYTES {
            return Err(MarkerError::TooLarge {
                bytes: bytes.len() as u64,
            });
        }
        let path = root.join(MARKER_FILE);
        replace_file(&path, &bytes).map_err(|e| io_error("writing", &path, e))
    }

    /// Applies to every format, since a newer marker's rows are restored
    /// into the registry too.
    fn validate(&self) -> Result<(), MarkerError> {
        if self.format == 0 {
            return Err(MarkerError::Malformed("format 0".to_string()));
        }
        validate_id(&self.id)?;
        for row in &self.models {
            validate_alias(&row.alias)?;
            validate_relative_path(&row.relative_path)?;
        }
        Ok(())
    }
}

impl PortableRow {
    /// The portable form of `row`, an install inside the library at
    /// `library_root`. The row's `relative_path` wins over its `path`,
    /// which goes stale when the drive mounts under a new name.
    pub fn from_installed(row: &InstalledModel, library_root: &Path) -> Result<Self, MarkerError> {
        let relative_path = match &row.relative_path {
            Some(relative) => relative.clone(),
            None => row
                .path
                .strip_prefix(library_root)
                .map_err(|_| MarkerError::UnsafePath(row.path.clone()))?
                .to_path_buf(),
        };
        validate_alias(&row.alias)?;
        validate_relative_path(&relative_path)?;
        Ok(Self {
            alias: row.alias.clone(),
            repo: row.repo.clone(),
            revision: row.revision.clone(),
            relative_path,
            family: row.family.clone(),
            install_bytes: row.install_bytes,
            installed_on: row.installed_on.clone(),
            status: row.status.clone(),
            kind: row.kind.clone(),
            modality: row.modality,
            variant: row.variant.clone(),
        })
    }

    /// The registry row for this install in library `library_id`, now at
    /// `library_root`.
    pub fn to_installed(
        &self,
        library_root: &Path,
        library_id: &str,
    ) -> Result<InstalledModel, MarkerError> {
        validate_id(library_id)?;
        validate_alias(&self.alias)?;
        let path = join_relative(library_root, &self.relative_path)?;
        Ok(InstalledModel {
            alias: self.alias.clone(),
            repo: self.repo.clone(),
            revision: self.revision.clone(),
            path,
            family: self.family.clone(),
            install_bytes: self.install_bytes,
            installed_on: self.installed_on.clone(),
            status: self.status.clone(),
            kind: self.kind.clone(),
            modality: self.modality,
            variant: self.variant.clone(),
            library: Some(library_id.to_string()),
            relative_path: Some(self.relative_path.clone()),
        })
    }
}

/// `root` joined with `relative` after refusing any component that could
/// lexically leave `root`. Every join of an untrusted relative path goes
/// through here. Symlinked directories inside the root are not detected
/// here, so callers that open, copy, or delete the joined path must check
/// containment.
pub fn join_relative(root: &Path, relative: &Path) -> Result<PathBuf, MarkerError> {
    validate_relative_path(relative)?;
    Ok(root.join(relative))
}

/// A fresh library id: a random (version 4) UUID in lowercase hex.
///
/// Reads the kernel's random source directly: the crate has no uuid or rand
/// dependency, and one id per added library does not justify adding one.
pub fn new_library_id() -> io::Result<String> {
    let mut bytes = [0u8; 16];
    File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    // RFC 9562 version 4 and variant bits, so the id reads as what it is.
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    Ok(format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    ))
}

/// Reads at most one byte past the cap. The length check alone is not
/// enough, because the file can grow between the stat and the read.
fn read_bounded(reader: impl Read) -> Result<Vec<u8>, MarkerError> {
    let mut bytes = Vec::new();
    reader
        .take(MAX_MARKER_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| MarkerError::Io(format!("reading the library marker: {e}")))?;
    if bytes.len() as u64 > MAX_MARKER_BYTES {
        return Err(MarkerError::TooLarge {
            bytes: bytes.len() as u64,
        });
    }
    Ok(bytes)
}

/// Replaces `path` with `bytes` through a temp file beside it and a rename,
/// so a reader sees the old file or the new one, never a partial write. The
/// temp file is removed when the write or rename fails.
pub(crate) fn replace_file(path: &Path, bytes: &[u8]) -> io::Result<()> {
    // The pid keeps processes apart and the counter keeps threads apart, so
    // no writer renames another's half-written file.
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(
        ".tmp-{}-{}",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let temporary = path.with_file_name(name);
    let written = write_new(&temporary, bytes).and_then(|()| fs::rename(&temporary, path));
    if written.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    written
}

/// Creates `path` fresh, so a planted symlink at the temp name is never
/// written through.
fn write_new(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let open = || OpenOptions::new().write(true).create_new(true).open(path);
    let mut file = match open() {
        // Left by a crashed writer whose pid was reused. Removing it removes
        // a link itself, never its target.
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
            fs::remove_file(path)?;
            open()?
        }
        opened => opened?,
    };
    file.write_all(bytes)?;
    file.sync_all()
}

fn io_error(action: &str, path: &Path, error: io::Error) -> MarkerError {
    MarkerError::Io(format!("{action} {}: {error}", path.display()))
}

pub(super) fn validate_id(id: &str) -> Result<(), MarkerError> {
    // The internal id would make a drive's library pose as This Mac.
    if id.is_empty() || id == INTERNAL_LIBRARY_ID || id.contains([':', '/']) {
        return Err(MarkerError::InvalidId(id.to_string()));
    }
    Ok(())
}

/// The same rule `Store::resolve_embedding` applies to a bare alias. The
/// store applies it too before joining an alias onto a library root.
pub(crate) fn validate_alias(alias: &str) -> Result<(), MarkerError> {
    if alias.is_empty() || alias.starts_with('.') || alias.contains('/') {
        return Err(MarkerError::InvalidAlias(alias.to_string()));
    }
    Ok(())
}

/// Only plain names, so the joined path cannot lexically leave the root. An
/// empty path is refused too: it names the library root itself. This is a
/// lexical check: a symlinked directory inside the root is not detected.
pub(super) fn validate_relative_path(relative: &Path) -> Result<(), MarkerError> {
    let plain = relative
        .components()
        .all(|component| matches!(component, Component::Normal(_)));
    if relative.as_os_str().is_empty() || !plain {
        return Err(MarkerError::UnsafePath(relative.to_path_buf()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ts-library-marker-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn entries(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    fn row(alias: &str, relative: &str, modality: ModelModality) -> PortableRow {
        PortableRow {
            alias: alias.to_string(),
            repo: format!("owner/{alias}"),
            revision: "main".to_string(),
            relative_path: PathBuf::from(relative),
            family: "qwen3".to_string(),
            install_bytes: 4_096,
            installed_on: "2026-10-14".to_string(),
            status: "verified".to_string(),
            kind: None,
            modality,
            variant: None,
        }
    }

    fn marker_json(format: u32, id: &str, relative: &str) -> String {
        format!(
            r#"{{
                "format": {format},
                "id": "{id}",
                "name": "T7",
                "created_on": "2026-10-14",
                "models": [{{
                    "alias": "qwen",
                    "repo": "owner/qwen",
                    "revision": "main",
                    "relative_path": "{relative}",
                    "family": "qwen3",
                    "install_bytes": 1,
                    "installed_on": "2026-10-14",
                    "status": "verified"
                }}]
            }}"#
        )
    }

    const ID: &str = "0f6c2a3e-1b7d-4c55-9e0a-7d2f1c3b4a59";

    #[test]
    fn a_marker_round_trips_through_its_file_and_leaves_no_temp_file() {
        let root = scratch("round-trip");
        let mut marker = LibraryMarker::new(ID.to_string(), "T7".to_string());
        assert_eq!(marker.format(), 1);
        assert!(!marker.is_read_only());
        assert_eq!(marker.created_on, crate::install::today());
        marker
            .models
            .push(row("qwen", "text/qwen.gturbo", ModelModality::Text));
        let mut audio = row("whisper", "audio/whisper.gturbo", ModelModality::Audio);
        audio.kind = Some("speech".to_string());
        audio.variant = Some("Q4_K_M".to_string());
        marker.models.push(audio);

        marker.write(&root).unwrap();
        assert_eq!(entries(&root), vec![MARKER_FILE.to_string()]);
        let read = LibraryMarker::read(&root).unwrap().unwrap();
        assert_eq!(read, marker);

        // The file carries the documented shape, and a portable row never
        // carries an absolute path or a library id.
        let json: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join(MARKER_FILE)).unwrap()).unwrap();
        assert_eq!(json["format"], 1);
        assert_eq!(json["id"], ID);
        assert_eq!(json["name"], "T7");
        assert_eq!(json["models"][0]["relative_path"], "text/qwen.gturbo");
        assert!(json["models"][0].get("path").is_none());
        assert!(json["models"][0].get("library").is_none());

        // A rewrite replaces the marker in place.
        marker.name = "Renamed".to_string();
        marker.models.clear();
        marker.write(&root).unwrap();
        assert_eq!(entries(&root), vec![MARKER_FILE.to_string()]);
        assert_eq!(LibraryMarker::read(&root).unwrap().unwrap(), marker);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn an_unmarked_folder_reads_as_none_and_a_non_file_marker_is_refused() {
        let root = scratch("absent");
        assert_eq!(LibraryMarker::read(&root), Ok(None));

        fs::create_dir(root.join(MARKER_FILE)).unwrap();
        assert_eq!(LibraryMarker::read(&root), Err(MarkerError::NotAFile));
        fs::remove_dir(root.join(MARKER_FILE)).unwrap();

        // A symlink is refused rather than followed out of the library, even
        // when its target is a valid marker.
        let outside = scratch("absent-outside");
        fs::write(
            outside.join(MARKER_FILE),
            marker_json(1, ID, "text/qwen.gturbo"),
        )
        .unwrap();
        std::os::unix::fs::symlink(outside.join(MARKER_FILE), root.join(MARKER_FILE)).unwrap();
        assert_eq!(LibraryMarker::read(&root), Err(MarkerError::NotAFile));
        let _ = fs::remove_dir_all(&root);
        let _ = fs::remove_dir_all(&outside);
    }

    #[test]
    fn a_marker_over_one_mib_is_refused_and_one_at_the_cap_is_read() {
        let root = scratch("oversize");
        let body = marker_json(1, ID, "text/qwen.gturbo");
        let pad = MAX_MARKER_BYTES as usize - body.len();
        // JSON allows trailing whitespace, so this is a valid marker of
        // exactly the cap.
        fs::write(root.join(MARKER_FILE), format!("{body}{}", " ".repeat(pad))).unwrap();
        assert!(LibraryMarker::read(&root).unwrap().is_some());

        fs::write(
            root.join(MARKER_FILE),
            format!("{body}{}", " ".repeat(pad + 1)),
        )
        .unwrap();
        assert_eq!(
            LibraryMarker::read(&root),
            Err(MarkerError::TooLarge {
                bytes: MAX_MARKER_BYTES + 1
            })
        );

        // A large marker is refused from its length alone. A read would
        // stop at the cap and report one byte past it instead.
        File::create(root.join(MARKER_FILE))
            .unwrap()
            .set_len(8 * MAX_MARKER_BYTES)
            .unwrap();
        assert_eq!(
            LibraryMarker::read(&root),
            Err(MarkerError::TooLarge {
                bytes: 8 * MAX_MARKER_BYTES
            })
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn the_bounded_read_stops_one_byte_past_the_cap() {
        // An endless reader stands in for a file that grows after its
        // length was checked. Without the cap this never returns.
        assert_eq!(
            read_bounded(io::repeat(b' ')),
            Err(MarkerError::TooLarge {
                bytes: MAX_MARKER_BYTES + 1
            })
        );
        assert_eq!(read_bounded(&b"{}"[..]), Ok(b"{}".to_vec()));
    }

    #[test]
    fn a_marker_that_would_be_refused_on_read_is_not_written() {
        let root = scratch("oversize-write");
        let mut marker = LibraryMarker::new(ID.to_string(), "T7".to_string());
        let alias = "a".repeat(MAX_MARKER_BYTES as usize);
        marker
            .models
            .push(row(&alias, "text/a.gturbo", ModelModality::Text));
        assert!(matches!(
            marker.write(&root),
            Err(MarkerError::TooLarge { .. })
        ));
        assert!(entries(&root).is_empty());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn traversal_paths_are_refused_and_touch_nothing_outside_the_library() {
        let scratch = scratch("traversal");
        let root = scratch.join("TurboSpark");
        let outside = scratch.join("outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();

        for relative in [
            "../outside/evil.gturbo",
            "text/../../outside/evil.gturbo",
            "/Users/someone/evil.gturbo",
            "./text/qwen.gturbo",
            "",
        ] {
            fs::write(root.join(MARKER_FILE), marker_json(1, ID, relative)).unwrap();
            assert_eq!(
                LibraryMarker::read(&root),
                Err(MarkerError::UnsafePath(PathBuf::from(relative))),
                "{relative:?}"
            );
            assert!(join_relative(&root, Path::new(relative)).is_err());

            // The writer refuses the same row before creating anything.
            fs::remove_file(root.join(MARKER_FILE)).unwrap();
            let mut marker = LibraryMarker::new(ID.to_string(), "T7".to_string());
            marker
                .models
                .push(row("qwen", relative, ModelModality::Text));
            assert_eq!(
                marker.write(&root),
                Err(MarkerError::UnsafePath(PathBuf::from(relative)))
            );
            assert!(entries(&root).is_empty(), "{relative:?}");
            assert_eq!(
                marker.models[0].to_installed(&root, ID),
                Err(MarkerError::UnsafePath(PathBuf::from(relative)))
            );
        }

        assert!(entries(&outside).is_empty());
        assert_eq!(
            entries(&scratch),
            vec!["TurboSpark".to_string(), "outside".to_string()]
        );
        assert_eq!(
            join_relative(&root, Path::new("text/qwen.gturbo")),
            Ok(root.join("text").join("qwen.gturbo"))
        );
        let _ = fs::remove_dir_all(&scratch);
    }

    #[test]
    fn a_newer_format_marker_is_listed_but_never_rewritten() {
        let root = scratch("newer");
        // A newer writer adds fields this build does not know.
        let text = marker_json(2, ID, "text/qwen.gturbo").replacen(
            "\"name\"",
            "\"shelves\": [1, 2], \"name\"",
            1,
        );
        fs::write(root.join(MARKER_FILE), &text).unwrap();

        let marker = LibraryMarker::read(&root).unwrap().unwrap();
        assert_eq!(marker.format(), 2);
        assert!(marker.is_read_only());
        assert_eq!(marker.id, ID);
        assert_eq!(marker.models.len(), 1);

        assert_eq!(
            marker.write(&root),
            Err(MarkerError::ReadOnly { format: 2 })
        );
        assert_eq!(fs::read_to_string(root.join(MARKER_FILE)).unwrap(), text);
        assert_eq!(entries(&root), vec![MARKER_FILE.to_string()]);

        // Unknown fields alone do not make a current-format marker
        // read-only.
        let current = marker_json(1, ID, "text/qwen.gturbo").replacen(
            "\"name\"",
            "\"shelves\": [1, 2], \"name\"",
            1,
        );
        fs::write(root.join(MARKER_FILE), current).unwrap();
        assert!(!LibraryMarker::read(&root).unwrap().unwrap().is_read_only());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn malformed_markers_and_unusable_ids_and_aliases_are_refused() {
        let root = scratch("malformed");
        for text in [
            "not json".to_string(),
            marker_json(0, ID, "text/qwen.gturbo"),
        ] {
            fs::write(root.join(MARKER_FILE), text).unwrap();
            assert!(matches!(
                LibraryMarker::read(&root),
                Err(MarkerError::Malformed(_))
            ));
        }

        for id in ["", "a:b", "a/b", INTERNAL_LIBRARY_ID] {
            fs::write(
                root.join(MARKER_FILE),
                marker_json(1, id, "text/qwen.gturbo"),
            )
            .unwrap();
            assert_eq!(
                LibraryMarker::read(&root),
                Err(MarkerError::InvalidId(id.to_string())),
                "{id:?}"
            );
            assert_eq!(
                row("qwen", "text/qwen.gturbo", ModelModality::Text).to_installed(&root, id),
                Err(MarkerError::InvalidId(id.to_string()))
            );
        }

        for alias in ["", "..", "../evil", "a/b", ".hidden"] {
            let text = marker_json(1, ID, "text/qwen.gturbo")
                .replace("\"alias\": \"qwen\"", &format!("\"alias\": \"{alias}\""));
            fs::write(root.join(MARKER_FILE), text).unwrap();
            assert_eq!(
                LibraryMarker::read(&root),
                Err(MarkerError::InvalidAlias(alias.to_string())),
                "{alias:?}"
            );
        }
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_planted_temp_symlink_is_replaced_rather_than_written_through() {
        let scratch = scratch("planted");
        let root = scratch.join("TurboSpark");
        let outside = scratch.join("outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        let victim = outside.join("victim");
        fs::write(&victim, "untouched").unwrap();
        let temporary = root.join(format!("{MARKER_FILE}.tmp-1-0"));
        std::os::unix::fs::symlink(&victim, &temporary).unwrap();

        write_new(&temporary, b"{}").unwrap();
        assert_eq!(fs::read_to_string(&victim).unwrap(), "untouched");
        assert!(fs::symlink_metadata(&temporary)
            .unwrap()
            .file_type()
            .is_file());
        assert_eq!(fs::read(&temporary).unwrap(), b"{}");
        let _ = fs::remove_dir_all(&scratch);
    }

    #[test]
    fn writing_into_a_missing_root_fails_without_creating_it() {
        let scratch = scratch("missing-root");
        let root = scratch.join("TurboSpark");
        let marker = LibraryMarker::new(ID.to_string(), "T7".to_string());
        assert!(matches!(marker.write(&root), Err(MarkerError::Io(_))));
        assert!(entries(&scratch).is_empty());
        let _ = fs::remove_dir_all(&scratch);
    }

    #[test]
    fn a_portable_row_converts_to_and_from_a_registry_row() {
        let old_root = Path::new("/Volumes/T7/TurboSpark");
        let new_root = Path::new("/Volumes/T7 1/TurboSpark");
        let installed = InstalledModel {
            alias: "qwen".to_string(),
            repo: "owner/qwen".to_string(),
            revision: "main".to_string(),
            path: old_root.join("text/qwen.gturbo"),
            family: "qwen3".to_string(),
            install_bytes: 4_096,
            installed_on: "2026-10-14".to_string(),
            status: "verified".to_string(),
            kind: Some("vision-tower".to_string()),
            modality: ModelModality::Text,
            variant: Some("Q4_K_M".to_string()),
            library: None,
            relative_path: None,
        };

        // An older writer dropped `relative_path`, so it comes from `path`.
        let portable = PortableRow::from_installed(&installed, old_root).unwrap();
        assert_eq!(portable.relative_path, PathBuf::from("text/qwen.gturbo"));

        // Restoring after the volume was renamed resolves under the new root.
        let restored = portable.to_installed(new_root, ID).unwrap();
        assert_eq!(
            restored,
            InstalledModel {
                path: new_root.join("text/qwen.gturbo"),
                library: Some(ID.to_string()),
                relative_path: Some(PathBuf::from("text/qwen.gturbo")),
                ..installed.clone()
            }
        );

        // A recorded `relative_path` wins over a stale absolute path.
        assert_eq!(
            PortableRow::from_installed(&restored, old_root).unwrap(),
            portable
        );

        // A row outside the library has no portable form.
        let outside = InstalledModel {
            path: PathBuf::from("/Users/someone/.turbospark/models/text/qwen.gturbo"),
            ..installed.clone()
        };
        assert_eq!(
            PortableRow::from_installed(&outside, old_root),
            Err(MarkerError::UnsafePath(outside.path.clone()))
        );

        // Both conversions re-check the alias rather than trusting that the
        // row came through a marker read, since store paths are built from
        // aliases.
        let bad_alias = InstalledModel {
            alias: "../x".to_string(),
            ..installed.clone()
        };
        assert_eq!(
            PortableRow::from_installed(&bad_alias, old_root),
            Err(MarkerError::InvalidAlias("../x".to_string()))
        );
        let bad_portable = PortableRow {
            alias: "../x".to_string(),
            ..portable.clone()
        };
        assert_eq!(
            bad_portable.to_installed(new_root, ID),
            Err(MarkerError::InvalidAlias("../x".to_string()))
        );

        let escaping = InstalledModel {
            relative_path: Some(PathBuf::from("../escape.gturbo")),
            ..installed
        };
        assert_eq!(
            PortableRow::from_installed(&escaping, old_root),
            Err(MarkerError::UnsafePath(PathBuf::from("../escape.gturbo")))
        );
    }

    #[test]
    fn new_library_ids_are_random_lowercase_v4_uuids() {
        let first = new_library_id().unwrap();
        let second = new_library_id().unwrap();
        assert_ne!(first, second);
        for id in [&first, &second] {
            let groups: Vec<&str> = id.split('-').collect();
            assert_eq!(
                groups.iter().map(|g| g.len()).collect::<Vec<_>>(),
                vec![8, 4, 4, 4, 12],
                "{id}"
            );
            assert!(id
                .chars()
                .all(|c| c == '-' || c.is_ascii_digit() || ('a'..='f').contains(&c)));
            assert!(groups[2].starts_with('4'), "{id}");
            assert!(matches!(
                groups[3].chars().next(),
                Some('8' | '9' | 'a' | 'b')
            ));
            // A fresh id is one a marker read accepts.
            let marker = LibraryMarker::new(id.clone(), "T7".to_string());
            assert_eq!(marker.validate(), Ok(()));
        }
    }
}
