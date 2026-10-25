//! Fish Audio Drama 3 installs by conversion, not by copy.
//!
//! The catalog's other audio installs download pinned files and check them
//! against pinned hashes. Drama 3's runtime layout (a headless trunk, the
//! tower and head weights, the codec) is made from the upstream MLX
//! checkpoint by `repack::write_fish_drama3_install_streamed`, so most of the
//! installed bytes exist nowhere upstream. This module owns that walk for the
//! CLI, the FFI and the app:
//!
//! - the five small root files download whole, each against a pinned size and
//!   sha256 (they are the catalog profile's assets);
//! - the six shards stream through the converter by range read and are never
//!   stored whole, so the 20 GB checkpoint is not on disk twice. A range read
//!   never sees a whole file, so the shards are not hash-checked. Each
//!   shard's header must stay inside its pinned size, which ties the walk to
//!   the pinned revision;
//! - the install is built in a sibling staging directory and renamed into
//!   place once every file is present, so a failed or interrupted run never
//!   leaves a half-built destination.
//!
//! Pins and license terms: `docs/FISH_DRAMA3_PHASE0.md`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use repack::{
    canonicalize_drama3_header, fetch_safetensors_header, write_fish_drama3_install_streamed,
    ByteProgressCallback, Gemma4Shards, HttpRangeSource, RangeSource, DRAMA3_SIDE_FILE,
};

use crate::audio_catalog::AudioAsset;
use crate::{CancelFlag, RepoRef};

/// The catalog alias of the one pinned Drama 3 profile.
pub const ALIAS: &str = "fish-drama3-4bit";
pub const REPO: &str = "mlx-community/fish-audio-drama-3-4bit";
pub const REVISION: &str = "cc639e27aba7edb73429d3e3806f471dcd79d316";
/// The manifest's name for this conversion. A profile that declares it is a
/// derived install.
pub const TOOL: &str = "fish_drama3_intake";
/// Bumped when the converter's output changes, so an older receipt is
/// recognised instead of trusted.
pub const TOOL_VERSION: u32 = 1;

const GIB: u64 = 1 << 30;
/// Free space the walk needs: the install plus the trunk writer's own
/// staging copy. Measured on the real install (22 GiB result).
pub const WORKING_BYTES: u64 = 25 * GIB;
/// What the finished install occupies.
pub const RESULT_BYTES: u64 = 22 * GIB;

/// The pinned shard sizes, in order, from `docs/FISH_DRAMA3_PHASE0.md`.
const SHARD_BYTES: [u64; 6] = [
    4_354_661_374,
    4_360_386_773,
    4_304_379_853,
    4_407_528_323,
    4_425_623_733,
    335_374_378,
];

/// The converter's outputs a receipt must record: the layout files that exist
/// nowhere upstream.
pub const REQUIRED_OUTPUTS: [&str; 3] = [
    DRAMA3_SIDE_FILE,
    "trunk/manifest.json",
    "trunk/trunk_role.json",
];

/// The files, relative to the install, a Drama 3 install must hold to open.
const REQUIRED_FILES: [&str; 8] = [
    "config.json",
    "tokenizer.json",
    "codec.safetensors",
    "LICENSE",
    "NOTICE",
    DRAMA3_SIDE_FILE,
    "trunk/manifest.json",
    "trunk/trunk_role.json",
];

/// One small root file: downloaded whole, hashed after.
#[derive(Debug, Clone)]
pub struct Pin {
    pub name: String,
    pub bytes: u64,
    /// Lowercase hex sha256, checked after the download.
    pub sha256: String,
}

#[derive(Debug, Clone)]
pub struct Shard {
    pub name: String,
    pub bytes: u64,
}

#[derive(Debug, Clone)]
pub struct Pins {
    pub shards: Vec<Shard>,
    pub config: Pin,
    /// LICENSE, NOTICE and the tokenizer: small, fetched before the walk.
    pub small: Vec<Pin>,
    /// The codec, fetched last so a failed walk wastes no 1.5 GB download.
    pub codec: Pin,
}

impl Pins {
    /// The root pins come from the catalog profile's assets, so the manifest
    /// is the one place they are written down.
    pub fn from_assets(assets: &[AudioAsset]) -> Result<Self, String> {
        let pin = |name: &str| -> Result<Pin, String> {
            assets
                .iter()
                .find(|a| a.path == name)
                .map(|a| Pin {
                    name: a.path.clone(),
                    bytes: a.size,
                    sha256: a.sha256.clone(),
                })
                .ok_or_else(|| format!("the Drama 3 profile has no pin for {name}"))
        };
        Ok(Self {
            shards: SHARD_BYTES
                .iter()
                .enumerate()
                .map(|(i, &bytes)| Shard {
                    name: format!("model-{:05}.safetensors", i + 1),
                    bytes,
                })
                .collect(),
            config: pin("config.json")?,
            small: vec![pin("LICENSE")?, pin("NOTICE")?, pin("tokenizer.json")?],
            codec: pin("codec.safetensors")?,
        })
    }

    /// `mlx-community/fish-audio-drama-3-4bit` at [`REVISION`], from the
    /// embedded catalog.
    pub fn upstream() -> Result<Self, String> {
        let catalog = crate::AudioCatalog::embedded()?;
        let profile = catalog
            .get(ALIAS)
            .ok_or("the embedded audio catalog has no Drama 3 profile")?;
        Self::from_assets(&profile.assets)
    }

    pub fn shard_bytes(&self) -> u64 {
        self.shards.iter().map(|s| s.bytes).sum()
    }

    /// Every byte the install reads from upstream.
    pub fn total_bytes(&self) -> u64 {
        self.shard_bytes()
            + self.config.bytes
            + self.small.iter().map(|p| p.bytes).sum::<u64>()
            + self.codec.bytes
    }
}

/// Upstream bytes the walk reads, for the profile's `download_bytes`.
pub fn shard_total_bytes() -> u64 {
    SHARD_BYTES.iter().sum()
}

/// Where the bytes come from. The real one reads Hugging Face by range; tests
/// read memory.
pub trait Source {
    fn shard(&self, name: &str) -> Box<dyn RangeSource + '_>;
    /// Writes a whole root file to `dest`, which must not exist.
    fn root_file(&self, pin: &Pin, dest: &Path) -> Result<(), String>;
    /// Bytes the shard readers have delivered so far, for progress.
    fn bytes_read(&self) -> u64 {
        0
    }
}

/// Hugging Face at the pinned revision. Cancelling `cancel` aborts the walk
/// at its next chunk read, even inside the 27 minute conversion.
pub struct HubSource {
    repo: RepoRef,
    token: Option<String>,
    cancel: CancelFlag,
    bytes: Arc<AtomicU64>,
}

impl HubSource {
    pub fn new(token: Option<&str>, cancel: &CancelFlag) -> Self {
        Self {
            repo: RepoRef::new(REPO, REVISION),
            token: token.map(str::to_string),
            cancel: cancel.clone(),
            bytes: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl Source for HubSource {
    fn shard(&self, name: &str) -> Box<dyn RangeSource + '_> {
        let bytes = Arc::clone(&self.bytes);
        let count: ByteProgressCallback = Arc::new(move |n| {
            bytes.fetch_add(n, Ordering::Relaxed);
        });
        // No cache directory: the walk reads each shard once, and a cache
        // would store the 20 GB a second time.
        Box::new(
            HttpRangeSource::with_progress(self.repo.file_url(name), count)
                .with_optional_token(self.token.clone())
                .with_cancel(self.cancel.clone()),
        )
    }

    fn root_file(&self, pin: &Pin, dest: &Path) -> Result<(), String> {
        HttpRangeSource::new(self.repo.file_url(&pin.name))
            .with_optional_token(self.token.clone())
            .with_cancel(self.cancel.clone())
            .download_to(dest, pin.bytes)
            .map(|_| ())
            .map_err(|e| format!("downloading {}: {e}", pin.name))
    }

    fn bytes_read(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }
}

/// One progress report. `completed_bytes` counts upstream bytes read out of
/// `total_bytes`; the hashing that follows reports its own bytes.
pub struct Progress<'a> {
    pub stage: &'a str,
    pub completed_bytes: u64,
    pub total_bytes: u64,
}

struct Report<'a, 'b> {
    total: u64,
    roots: u64,
    source: &'a dyn Source,
    sink: &'b mut dyn FnMut(&Progress<'_>),
}

impl Report<'_, '_> {
    fn emit(&mut self, stage: &str) {
        let done = (self.roots + self.source.bytes_read()).min(self.total);
        (self.sink)(&Progress {
            stage,
            completed_bytes: done,
            total_bytes: self.total,
        });
    }
}

/// The install, every outside effect passed in. Returns once `dest` exists.
pub fn install(
    dest: &Path,
    pins: &Pins,
    source: &dyn Source,
    available: u64,
    cancel: &CancelFlag,
    progress: &mut dyn FnMut(&Progress<'_>),
) -> Result<(), String> {
    install_with(dest, pins, source, available, cancel, progress, |_, _| {
        Ok(())
    })
}

/// [`install`] with a step that runs on the finished staging directory
/// before it is renamed into place, such as the catalog's receipt. An error
/// from it removes the staging directory like any other failure.
pub fn install_with(
    dest: &Path,
    pins: &Pins,
    source: &dyn Source,
    available: u64,
    cancel: &CancelFlag,
    progress: &mut dyn FnMut(&Progress<'_>),
    finish: impl FnOnce(&Path, &mut dyn FnMut(&Progress<'_>)) -> Result<(), String>,
) -> Result<(), String> {
    refuse_existing(dest)?;
    check_free_space(available, WORKING_BYTES, dest)?;
    cancel.checkpoint().map_err(|e| e.to_string())?;
    let staging = staging_dir(dest)?;
    if std::fs::symlink_metadata(&staging).is_ok() {
        return Err(format!(
            "{} is left over from an interrupted install; remove it and run again",
            staging.display()
        ));
    }
    std::fs::create_dir_all(&staging)
        .map_err(|e| format!("creating {}: {e}", staging.display()))?;
    let built = build(&staging, pins, source, cancel, progress)
        .and_then(|()| verify_layout(&staging))
        .and_then(|()| finish(&staging, progress))
        .and_then(|()| {
            cancel.checkpoint().map_err(|e| e.to_string())?;
            // Last look: rename onto an empty directory would succeed.
            refuse_existing(dest)?;
            std::fs::rename(&staging, dest)
                .map_err(|e| format!("moving the install into place: {e}"))
        });
    if built.is_err() {
        // The staging directory is this run's own; nothing else writes it.
        let _ = std::fs::remove_dir_all(&staging);
    }
    built
}

fn build(
    staging: &Path,
    pins: &Pins,
    source: &dyn Source,
    cancel: &CancelFlag,
    sink: &mut dyn FnMut(&Progress<'_>),
) -> Result<(), String> {
    let mut report = Report {
        total: pins.total_bytes(),
        roots: 0,
        source,
        sink,
    };
    fetch_verified(source, &pins.config, staging, cancel, &mut report)?;
    for pin in &pins.small {
        fetch_verified(source, pin, staging, cancel, &mut report)?;
    }
    let config_text = std::fs::read_to_string(staging.join(&pins.config.name))
        .map_err(|e| format!("reading config.json: {e}"))?;

    let shard_sources: Vec<_> = pins.shards.iter().map(|s| source.shard(&s.name)).collect();
    let mut headers = Vec::with_capacity(pins.shards.len());
    for (shard, reader) in pins.shards.iter().zip(&shard_sources) {
        cancel.checkpoint().map_err(|e| e.to_string())?;
        let mut header = fetch_safetensors_header(reader.as_ref())
            .map_err(|e| format!("reading the header of {}: {e}", shard.name))?;
        check_shard_extent(&header, shard)?;
        canonicalize_drama3_header(&mut header).map_err(|e| format!("{}: {e}", shard.name))?;
        headers.push(header);
    }
    let shards = Gemma4Shards::new(
        headers
            .iter()
            .zip(&shard_sources)
            .map(|(h, s)| (h, s.as_ref()))
            .collect(),
    )
    .map_err(|e| format!("assembling the shards: {e}"))?;
    let repo = RepoRef::new(REPO, REVISION);
    let model_id = repo.file_url("").trim_end_matches('/').to_string();
    let arch = write_fish_drama3_install_streamed(staging, &config_text, &model_id, &shards, |s| {
        report.emit(s)
    })
    .map_err(|e| format!("converting the checkpoint: {e}"))?;
    report.emit(&format!(
        "converted {} layers, {} experts, hidden {}",
        arch.num_layers, arch.num_experts, arch.hidden_size
    ));
    fetch_verified(source, &pins.codec, staging, cancel, &mut report)
}

/// A shard whose header declares tensor data past the pinned file size is not
/// the pinned file.
fn check_shard_extent(header: &repack::SafetensorsHeader, shard: &Shard) -> Result<(), String> {
    let end = header
        .tensors
        .values()
        .map(|t| t.data_offsets.1)
        .max()
        .unwrap_or(0);
    let declared = header.data_region_start().saturating_add(end);
    if declared > shard.bytes {
        return Err(format!(
            "{} declares {declared} bytes but the pin is {}; it is not the pinned file",
            shard.name, shard.bytes
        ));
    }
    Ok(())
}

fn fetch_verified(
    source: &dyn Source,
    pin: &Pin,
    dir: &Path,
    cancel: &CancelFlag,
    report: &mut Report<'_, '_>,
) -> Result<(), String> {
    cancel.checkpoint().map_err(|e| e.to_string())?;
    report.emit(&format!(
        "downloading {} ({})",
        pin.name,
        human_bytes(pin.bytes)
    ));
    let dest = dir.join(&pin.name);
    source.root_file(pin, &dest)?;
    let actual =
        model_io::hash_file(&dest, 1 << 20).map_err(|e| format!("hashing {}: {e}", pin.name))?;
    if actual != pin.sha256 {
        let _ = std::fs::remove_file(&dest);
        return Err(format!(
            "{} has sha256 {actual}, expected {}",
            pin.name, pin.sha256
        ));
    }
    report.roots += pin.bytes;
    report.emit(&format!("downloaded {}", pin.name));
    Ok(())
}

/// What `fish_drama3_say` and `turbospark-speak` open.
pub fn verify_layout(dir: &Path) -> Result<(), String> {
    for relative in REQUIRED_FILES {
        if !dir.join(relative).is_file() {
            return Err(format!("the install is missing {relative}"));
        }
    }
    Ok(())
}

/// The converter's outputs, which exist nowhere upstream: every regular file
/// under `dir` that is not one of the pinned root files and not the catalog's
/// receipt. Each is recorded with its size and sha256, so a later check can
/// tell a damaged install from an intact one. Symlinks and special files are
/// refused. `progress(file, done, total)` counts bytes hashed.
pub fn hash_outputs(
    dir: &Path,
    pinned: &BTreeSet<String>,
    skip: &str,
    cancel: &CancelFlag,
    progress: &mut dyn FnMut(&str, u64, u64),
) -> Result<BTreeMap<String, model_io::speech_receipt::FileEntry>, String> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, u64)>) -> Result<(), String> {
        for entry in
            std::fs::read_dir(dir).map_err(|e| format!("reading {}: {e}", dir.display()))?
        {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            let kind = entry.file_type().map_err(|e| e.to_string())?;
            if kind.is_dir() {
                walk(root, &path, out)?;
            } else if kind.is_file() {
                let relative = path
                    .strip_prefix(root)
                    .map_err(|e| e.to_string())?
                    .to_str()
                    .ok_or("a non-UTF8 file name in the install")?
                    .replace(std::path::MAIN_SEPARATOR, "/");
                let size = entry.metadata().map_err(|e| e.to_string())?.len();
                out.push((relative, size));
            } else {
                return Err(format!(
                    "{} is a symlink or special file; the install holds regular files only",
                    path.display()
                ));
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(dir, dir, &mut files)?;
    files.retain(|(name, _)| !pinned.contains(name) && name != skip);
    files.sort();
    let total: u64 = files.iter().map(|(_, size)| size).sum();
    let mut finished = 0u64;
    let mut out = BTreeMap::new();
    for (name, size) in files {
        cancel.checkpoint().map_err(|e| e.to_string())?;
        let mut reported = 0u64;
        let base = finished;
        let digest = model_io::hash_file_observed(&dir.join(&name), 1 << 20, |done| {
            if cancel.is_cancelled() {
                return false;
            }
            // Reporting every 1 MiB chunk would call the host thousands of
            // times for the 17 GB of expert files.
            if done - reported >= 16 << 20 {
                reported = done;
                progress(&name, base + done, total);
            }
            true
        })
        .map_err(|e| format!("hashing {name}: {e}"))?
        .ok_or_else(|| "cancelled by the caller".to_string())?;
        finished += size;
        progress(&name, finished, total);
        out.insert(
            name,
            model_io::speech_receipt::FileEntry {
                size,
                sha256: digest,
            },
        );
    }
    Ok(out)
}

pub fn refuse_existing(dest: &Path) -> Result<(), String> {
    if std::fs::symlink_metadata(dest).is_ok() {
        return Err(format!(
            "{} already exists; this never overwrites. Remove it or pick another destination",
            dest.display()
        ));
    }
    Ok(())
}

pub fn check_free_space(available: u64, required: u64, dest: &Path) -> Result<(), String> {
    if available < required {
        return Err(format!(
            "installing Drama 3 needs about {} free ({} for the install, the rest working space); \
             {} is available where {} will be created",
            human_bytes(required),
            human_bytes(RESULT_BYTES),
            human_bytes(available),
            dest.display()
        ));
    }
    Ok(())
}

/// A sibling of `dest`, so the final rename stays on one filesystem.
pub fn staging_dir(dest: &Path) -> Result<PathBuf, String> {
    let name = dest
        .file_name()
        .ok_or_else(|| format!("{} has no directory name", dest.display()))?;
    let parent = dest.parent().filter(|p| !p.as_os_str().is_empty());
    let mut staging_name = std::ffi::OsString::from(".");
    staging_name.push(name);
    staging_name.push(".partial");
    Ok(parent.unwrap_or_else(|| Path::new(".")).join(staging_name))
}

pub fn human_bytes(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / GIB as f64)
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    use repack::{
        synthetic_fish_drama3_checkpoint, Drama3NormStorage, MemoryRangeSource, SyntheticDrama3,
    };

    pub(crate) fn tempdir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("drama3_install_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The synthetic checkpoint behind a [`Source`], with tiny root files.
    pub(crate) struct Memory {
        pub(crate) checkpoint: SyntheticDrama3,
        pub(crate) files: Vec<(String, Vec<u8>)>,
        pub(crate) root_calls: AtomicUsize,
        /// Cancels this flag when the Nth root file is requested.
        pub(crate) cancel_on_root: Option<(usize, CancelFlag)>,
    }

    impl Source for Memory {
        fn shard(&self, _name: &str) -> Box<dyn RangeSource + '_> {
            Box::new(MemoryRangeSource::new(&self.checkpoint.blob))
        }

        fn root_file(&self, pin: &Pin, dest: &Path) -> Result<(), String> {
            let call = self.root_calls.fetch_add(1, Ordering::SeqCst) + 1;
            if let Some((n, flag)) = &self.cancel_on_root {
                if call == *n {
                    flag.cancel();
                }
            }
            let bytes = if pin.name == "config.json" {
                self.checkpoint.config_text.as_bytes().to_vec()
            } else {
                self.files
                    .iter()
                    .find(|(name, _)| *name == pin.name)
                    .map(|(_, bytes)| bytes.clone())
                    .ok_or_else(|| format!("no fixture for {}", pin.name))?
            };
            std::fs::write(dest, bytes).map_err(|e| e.to_string())
        }
    }

    pub(crate) fn fixture() -> (Memory, Pins) {
        let checkpoint = synthetic_fish_drama3_checkpoint(256, 4, Drama3NormStorage::Shifted);
        let files: Vec<(String, Vec<u8>)> = [
            ("LICENSE", b"license text".to_vec()),
            ("NOTICE", b"notice text".to_vec()),
            ("tokenizer.json", b"{}".to_vec()),
            ("codec.safetensors", b"codec bytes".to_vec()),
        ]
        .into_iter()
        .map(|(n, b)| (n.to_string(), b))
        .collect();
        let pin_of = |name: &str, bytes: &[u8]| Pin {
            name: name.to_string(),
            bytes: bytes.len() as u64,
            sha256: model_io::hash_data(bytes),
        };
        let find = |name: &str| files.iter().find(|(n, _)| n == name).unwrap().1.clone();
        let pins = Pins {
            shards: vec![Shard {
                name: "model-00001.safetensors".to_string(),
                bytes: checkpoint.blob.len() as u64,
            }],
            config: pin_of("config.json", checkpoint.config_text.as_bytes()),
            small: ["LICENSE", "NOTICE", "tokenizer.json"]
                .iter()
                .map(|n| pin_of(n, &find(n)))
                .collect(),
            codec: pin_of("codec.safetensors", &find("codec.safetensors")),
        };
        (
            Memory {
                checkpoint,
                files,
                root_calls: AtomicUsize::new(0),
                cancel_on_root: None,
            },
            pins,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{fixture, Memory};
    use super::*;

    use super::test_support::tempdir;

    const PLENTY: u64 = u64::MAX;

    fn run(dest: &Path, pins: &Pins, source: &Memory, available: u64) -> Result<(), String> {
        install(
            dest,
            pins,
            source,
            available,
            &CancelFlag::new(),
            &mut |_| {},
        )
    }

    #[test]
    fn installs_a_layout_the_runtime_opens() {
        let root = tempdir("ok");
        let dest = root.join("drama3");
        let (source, pins) = fixture();
        let mut stages = Vec::new();
        let mut last = 0u64;
        install(
            &dest,
            &pins,
            &source,
            PLENTY,
            &CancelFlag::new(),
            &mut |p| {
                stages.push(p.stage.to_string());
                // Progress never goes backwards and never passes its total.
                assert!(p.completed_bytes >= last, "{}", p.stage);
                assert!(p.completed_bytes <= p.total_bytes, "{}", p.stage);
                last = p.completed_bytes;
            },
        )
        .unwrap();
        verify_layout(&dest).unwrap();
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dest.join("config.json")).unwrap()).unwrap();
        assert_eq!(config["model_type"], "fish_drama3");
        // LICENSE and NOTICE travel with the install, byte for byte.
        assert_eq!(
            std::fs::read(dest.join("LICENSE")).unwrap(),
            b"license text"
        );
        assert_eq!(std::fs::read(dest.join("NOTICE")).unwrap(), b"notice text");
        // The staging directory is gone and the walk reported its stages.
        assert!(!staging_dir(&dest).unwrap().exists());
        assert!(stages.iter().any(|s| s.contains("codec.safetensors")));
        assert_eq!(
            last,
            pins.config.bytes + pins.small.iter().map(|p| p.bytes).sum::<u64>() + pins.codec.bytes
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn refuses_an_existing_destination_before_any_download() {
        let root = tempdir("exists");
        let dest = root.join("drama3");
        std::fs::create_dir_all(&dest).unwrap();
        std::fs::write(dest.join("keep.txt"), b"mine").unwrap();
        let (source, pins) = fixture();
        let err = run(&dest, &pins, &source, PLENTY).unwrap_err();
        assert!(err.contains("already exists"), "{err}");
        assert_eq!(source.root_calls.load(Ordering::SeqCst), 0);
        assert_eq!(std::fs::read(dest.join("keep.txt")).unwrap(), b"mine");
        assert!(!staging_dir(&dest).unwrap().exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn refuses_too_little_disk_before_any_download() {
        let root = tempdir("disk");
        let dest = root.join("drama3");
        let (source, pins) = fixture();
        let err = run(&dest, &pins, &source, WORKING_BYTES - 1).unwrap_err();
        assert!(err.contains("free"), "{err}");
        assert_eq!(source.root_calls.load(Ordering::SeqCst), 0);
        assert!(!dest.exists() && !staging_dir(&dest).unwrap().exists());
        // Exactly the requirement is enough.
        assert!(check_free_space(WORKING_BYTES, WORKING_BYTES, &dest).is_ok());
        assert!(check_free_space(WORKING_BYTES - 1, WORKING_BYTES, &dest).is_err());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_wrong_hash_stops_the_install_and_leaves_no_directory() {
        let root = tempdir("hash");
        let dest = root.join("drama3");
        let (source, mut pins) = fixture();
        pins.small[0].sha256 = model_io::hash_data(b"a different license");
        let err = run(&dest, &pins, &source, PLENTY).unwrap_err();
        assert!(err.contains("LICENSE has sha256"), "{err}");
        assert!(!dest.exists() && !staging_dir(&dest).unwrap().exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_wrong_codec_hash_fails_after_the_walk_and_cleans_up() {
        let root = tempdir("codec");
        let dest = root.join("drama3");
        let (source, mut pins) = fixture();
        pins.codec.sha256 = model_io::hash_data(b"not the codec");
        let err = run(&dest, &pins, &source, PLENTY).unwrap_err();
        assert!(err.contains("codec.safetensors has sha256"), "{err}");
        assert!(!dest.exists() && !staging_dir(&dest).unwrap().exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_shard_larger_than_its_pin_is_refused() {
        let root = tempdir("extent");
        let dest = root.join("drama3");
        let (source, mut pins) = fixture();
        pins.shards[0].bytes -= 1;
        let err = run(&dest, &pins, &source, PLENTY).unwrap_err();
        assert!(err.contains("not the pinned file"), "{err}");
        assert!(!dest.exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_leftover_staging_directory_is_reported_and_kept() {
        let root = tempdir("stale");
        let dest = root.join("drama3");
        let stale = staging_dir(&dest).unwrap();
        std::fs::create_dir_all(&stale).unwrap();
        let (source, pins) = fixture();
        let err = run(&dest, &pins, &source, PLENTY).unwrap_err();
        assert!(err.contains("interrupted install"), "{err}");
        assert!(
            stale.exists(),
            "another run's staging directory is not ours to delete"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// A stop requested mid-walk ends the install, and nothing is left.
    #[test]
    fn a_cancel_between_files_removes_the_staging_directory() {
        let root = tempdir("cancel");
        let dest = root.join("drama3");
        let (mut source, pins) = fixture();
        let flag = CancelFlag::new();
        source.cancel_on_root = Some((2, flag.clone()));
        let err = install(&dest, &pins, &source, PLENTY, &flag, &mut |_| {}).unwrap_err();
        assert!(err.contains("cancelled"), "{err}");
        assert!(!dest.exists() && !staging_dir(&dest).unwrap().exists());
        // Stopped before the third root file was asked for.
        assert_eq!(source.root_calls.load(Ordering::SeqCst), 2);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_failing_finish_step_removes_the_staging_directory() {
        let root = tempdir("finish");
        let dest = root.join("drama3");
        let (source, pins) = fixture();
        let err = install_with(
            &dest,
            &pins,
            &source,
            PLENTY,
            &CancelFlag::new(),
            &mut |_| {},
            |staging, _| {
                assert!(staging.join("trunk").is_dir(), "the layout is built first");
                Err("receipt refused".to_string())
            },
        )
        .unwrap_err();
        assert_eq!(err, "receipt refused");
        assert!(!dest.exists() && !staging_dir(&dest).unwrap().exists());
        let _ = std::fs::remove_dir_all(root);
    }

    /// The layout names are written out above, so they must not drift from
    /// what the converter writes.
    #[test]
    fn the_required_files_follow_the_converter_constants() {
        assert_eq!(repack::DRAMA3_TRUNK_DIR, "trunk");
        assert!(REQUIRED_FILES.contains(&DRAMA3_SIDE_FILE));
        for output in REQUIRED_OUTPUTS {
            assert!(REQUIRED_FILES.contains(&output), "{output}");
        }
        assert!(REQUIRED_FILES
            .iter()
            .any(|f| *f == format!("{}/manifest.json", repack::DRAMA3_TRUNK_DIR)));
    }

    #[test]
    fn the_layout_check_names_a_missing_file() {
        let root = tempdir("layout");
        assert!(verify_layout(&root).unwrap_err().contains("config.json"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn staging_is_a_sibling_of_the_destination() {
        assert_eq!(
            staging_dir(Path::new("/a/b/drama3")).unwrap(),
            Path::new("/a/b/.drama3.partial")
        );
        assert_eq!(
            staging_dir(Path::new("drama3")).unwrap(),
            Path::new("./.drama3.partial")
        );
    }

    #[test]
    fn upstream_pins_match_the_recorded_table() {
        let pins = Pins::upstream().unwrap();
        assert_eq!(pins.shards.len(), 6);
        assert_eq!(pins.shards[0].name, "model-00001.safetensors");
        assert_eq!(pins.shard_bytes(), 22_187_954_434);
        assert_eq!(shard_total_bytes(), pins.shard_bytes());
        assert_eq!(REVISION.len(), 40);
        for pin in pins.small.iter().chain([&pins.config, &pins.codec]) {
            assert_eq!(pin.sha256.len(), 64, "{}", pin.name);
        }
        // The profile pins the same revision the walk reads.
        let catalog = crate::AudioCatalog::embedded().unwrap();
        let profile = catalog.get(ALIAS).unwrap();
        assert_eq!(profile.identity.repository, REPO);
        assert_eq!(profile.identity.revision, REVISION);
    }

    #[test]
    fn hash_outputs_covers_what_upstream_does_not_pin() {
        let root = tempdir("outputs");
        std::fs::create_dir_all(root.join("trunk/packed_experts")).unwrap();
        std::fs::write(root.join("config.json"), b"pinned").unwrap();
        std::fs::write(root.join("tts_side.safetensors"), b"side").unwrap();
        std::fs::write(root.join("trunk/manifest.json"), b"{}").unwrap();
        std::fs::write(
            root.join("trunk/packed_experts/layer_00.bin"),
            vec![7u8; 5000],
        )
        .unwrap();
        std::fs::write(root.join("audio-receipt.json"), b"skipped").unwrap();
        let pinned: BTreeSet<String> = ["config.json".to_string()].into();
        let mut last_total = 0;
        let outputs = hash_outputs(
            &root,
            &pinned,
            "audio-receipt.json",
            &CancelFlag::new(),
            &mut |_, _, t| last_total = t,
        )
        .unwrap();
        let names: Vec<_> = outputs.keys().map(String::as_str).collect();
        assert_eq!(
            names,
            [
                "trunk/manifest.json",
                "trunk/packed_experts/layer_00.bin",
                "tts_side.safetensors"
            ]
        );
        assert_eq!(outputs["tts_side.safetensors"].size, 4);
        assert_eq!(
            outputs["tts_side.safetensors"].sha256,
            model_io::hash_data(b"side")
        );
        assert_eq!(outputs["trunk/packed_experts/layer_00.bin"].size, 5000);
        assert_eq!(last_total, 4 + 2 + 5000);
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn hash_outputs_refuses_a_symlink() {
        let root = tempdir("symlink");
        std::fs::write(root.join("real"), b"x").unwrap();
        std::os::unix::fs::symlink(root.join("real"), root.join("link")).unwrap();
        let err = hash_outputs(
            &root,
            &BTreeSet::new(),
            "",
            &CancelFlag::new(),
            &mut |_, _, _| {},
        )
        .unwrap_err();
        assert!(err.contains("symlink"), "{err}");
        let _ = std::fs::remove_dir_all(root);
    }
}
