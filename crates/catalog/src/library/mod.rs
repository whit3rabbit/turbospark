//! Model libraries: places that hold model files, apart from the TurboSpark
//! home that holds the registry and settings.
//!
//! "This Mac" is the implicit internal library rooted at `<home>/models`.
//! Every other library is a folder on a directly connected drive, recorded in
//! `<home>/libraries.json` and described on its drive by a marker, so it can
//! be recognized after the volume is renamed or remounted elsewhere.

mod locate;
mod marker;
mod mount_table;
mod registry;

pub use locate::{
    discover_at, locate, volume_relative_path, volume_root_of, Located, STANDARD_LIBRARY_FOLDER,
};
pub use marker::{
    join_relative, new_library_id, LibraryMarker, MarkerError, PortableRow, MARKER_FILE,
};
pub use mount_table::{
    parse_mount_table, validate_location, LocationEnv, LocationRefusal, MountEntry, MountSource,
};
pub use registry::{AddedLibrary, LibraryRegistry, LoadReport, LIBRARIES_FILE};
// The store writes `installed.json` the same way the library files are written.
pub(crate) use marker::replace_file;

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The id of "This Mac". Never written as a removable record, and the
/// library a row without a `library` field belongs to.
pub const INTERNAL_LIBRARY_ID: &str = "internal";

/// One library as `libraries.json` stores it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LibraryRecord {
    /// [`INTERNAL_LIBRARY_ID`], or the uuid the library's marker carries.
    pub id: String,
    pub name: String,
    /// The library root relative to its volume root, for example
    /// `TurboSpark`. Kept apart from the mount path so a renamed volume is
    /// searched at the same relative place.
    pub volume_relative_path: PathBuf,
    /// The volume root where the library was last found. The library root
    /// is this joined with [`Self::volume_relative_path`].
    pub last_mount_path: PathBuf,
    pub default_for_downloads: bool,
    /// `YYYY-MM-DD`, whole days like `installed_on`.
    pub added_on: String,
    /// `YYYY-MM-DD` of the last refresh that found the library connected.
    /// `None` until a refresh sees it.
    #[serde(default)]
    pub last_seen: Option<String>,
}

impl LibraryRecord {
    /// "This Mac" for the home at `home`.
    ///
    /// The root is split as `home` plus `models` so the same join that
    /// resolves an external library yields today's `<home>/models` layout,
    /// which keeps every existing row resolving unchanged. It starts as the
    /// default for downloads (1.9). It is implicit and never persisted, so
    /// it has no add date.
    pub fn internal(home: &Path) -> Self {
        Self {
            id: INTERNAL_LIBRARY_ID.to_string(),
            name: "This Mac".to_string(),
            volume_relative_path: PathBuf::from("models"),
            last_mount_path: home.to_path_buf(),
            default_for_downloads: true,
            added_on: String::new(),
            last_seen: None,
        }
    }

    /// Whether this is "This Mac", which cannot be removed (1.2).
    pub fn is_internal(&self) -> bool {
        self.id == INTERNAL_LIBRARY_ID
    }
}

/// A library as listing surfaces see it after locating.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibraryView {
    pub record: LibraryRecord,
    /// The current library root, `None` when not connected.
    pub root: Option<PathBuf>,
    pub available: bool,
    /// More than one connected folder carries this library's marker id.
    pub duplicate: bool,
    pub model_count: usize,
}

/// Why a library list operation failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LibraryError {
    Refused(LocationRefusal),
    UnknownLibrary(String),
    /// The operation is not allowed on "This Mac", such as removing it.
    Internal,
    /// A library marker could not be read or written. A marker that exists
    /// but cannot be read is never overwritten, because it may describe a
    /// library and its models.
    Marker(MarkerError),
    Io(String),
}

/// Whether an install's files can be opened right now. Not connected is a
/// state rather than an error, so offline rows stay listed without raising.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Availability {
    Available,
    /// The row's library is not connected. `library` is its name.
    NotConnected {
        library: String,
    },
    /// The library is connected but the install folder is gone.
    Missing,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Store;
    use std::path::{Path, PathBuf};

    #[test]
    fn internal_record_is_this_mac_rooted_at_the_store_models_folder() {
        let home = Path::new("/Users/someone/.turbospark");
        let record = LibraryRecord::internal(home);

        assert_eq!(record.id, INTERNAL_LIBRARY_ID);
        assert_eq!(INTERNAL_LIBRARY_ID, "internal");
        assert!(record.is_internal());
        assert_eq!(record.name, "This Mac");
        // The internal root must stay today's layout so existing rows keep
        // resolving without a migration.
        assert_eq!(
            record.last_mount_path.join(&record.volume_relative_path),
            Store::new(home).models_root()
        );
        // This Mac is the initial default for downloads (1.9).
        assert!(record.default_for_downloads);
        assert_eq!(record.last_seen, None);

        let external = LibraryRecord {
            id: "0f6c2a3e-1b7d-4c55-9e0a-7d2f1c3b4a59".to_string(),
            ..record
        };
        assert!(!external.is_internal());
    }

    #[test]
    fn library_record_json_uses_field_names_and_defaults_last_seen() {
        let json = r#"{
            "id": "0f6c2a3e-1b7d-4c55-9e0a-7d2f1c3b4a59",
            "name": "T7",
            "volume_relative_path": "TurboSpark",
            "last_mount_path": "/Volumes/T7",
            "default_for_downloads": false,
            "added_on": "2026-10-14"
        }"#;
        let record: LibraryRecord = serde_json::from_str(json).unwrap();
        assert_eq!(record.last_seen, None);
        assert_eq!(record.volume_relative_path, PathBuf::from("TurboSpark"));
        assert_eq!(record.last_mount_path, PathBuf::from("/Volumes/T7"));

        let seen = LibraryRecord {
            last_seen: Some("2026-10-14".to_string()),
            ..record
        };
        let round: LibraryRecord =
            serde_json::from_str(&serde_json::to_string(&seen).unwrap()).unwrap();
        assert_eq!(round, seen);
    }

    #[test]
    fn injected_location_env_carries_its_three_inputs() {
        let entries = vec![
            MountEntry {
                mount_point: PathBuf::from("/"),
                fs_type: "apfs".to_string(),
                local: true,
                read_only: true,
            },
            MountEntry {
                mount_point: PathBuf::from("/Volumes/Embedding Gemma Transcribe"),
                fs_type: "exfat".to_string(),
                local: true,
                read_only: false,
            },
        ];
        let volumes_root = std::env::temp_dir().join("catalog-library-volumes");
        let env = LocationEnv {
            mount_source: MountSource::Fixed(entries.clone()),
            volumes_root: volumes_root.clone(),
            home_device: 16_777_230,
        };

        match &env.mount_source {
            MountSource::Fixed(fixed) => assert_eq!(fixed, &entries),
            MountSource::System => panic!("an injected env must not use the system table"),
        }
        assert_eq!(env.volumes_root, volumes_root);
        assert_eq!(env.home_device, 16_777_230);
        assert_ne!(
            env,
            LocationEnv {
                mount_source: MountSource::System,
                ..env.clone()
            }
        );
    }
}
