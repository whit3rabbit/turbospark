//! The inputs and outcomes of deciding whether a path may host a library.
//!
//! Every OS-derived fact arrives through [`LocationEnv`], so tests decide
//! locations against a fixed mount table and a temp volumes folder without
//! spawning a process.

use std::path::PathBuf;

/// One line of the mount table: `<device> on <mount point> (<fstype>, ...)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MountEntry {
    pub mount_point: PathBuf,
    pub fs_type: String,
    /// The `local` flag. Its absence marks a network mount.
    pub local: bool,
    pub read_only: bool,
}

/// Why a location cannot host a library. Each kind maps to its own sentence
/// in the CLI and the app, so a refusal names the limit it hit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocationRefusal {
    /// This Mac already covers the home folder's disk (1.10).
    SameDiskAsHome,
    /// Libraries must be on a directly connected drive (1.4).
    Network,
    /// The filesystem cannot hold a single file of 4 GiB or more (1.5).
    SmallFileLimit {
        fs_type: String,
    },
    ReadOnly,
    NotADirectory,
    /// Inside the named library (1.7).
    NestedIn {
        library: String,
    },
    /// Inside the TurboSpark home folder (1.7).
    InsideHome,
}

/// Where mount entries come from. `System` runs `/sbin/mount` only when
/// `validate_location` asks, never at construction, because the default
/// store is built on server request paths and must not spawn or fail there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MountSource {
    System,
    Fixed(Vec<MountEntry>),
}

/// Every OS-derived input to location decisions, injectable for tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocationEnv {
    pub mount_source: MountSource,
    /// `/Volumes` on a live system, a temp folder of fake volumes in tests.
    pub volumes_root: PathBuf,
    /// `st_dev` of the home folder, for the same-disk refusal.
    pub home_device: u64,
}
