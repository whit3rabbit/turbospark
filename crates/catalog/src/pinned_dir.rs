//! Install machinery shared by pinned directory profiles (embedding and OCR).
//!
//! A pinned directory is a set of files from one Hugging Face repository at
//! an immutable revision, each pinned by path, size and SHA-256. The
//! transfer verifies every byte before the directory is published with a
//! rename, and a receipt in the directory records the pin so later checks
//! can tell a current install from an older pin without re-hashing.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{
    CancelFlag, Catalog, Client, DownloadReceipt, HubClient, HubDownloadProgress, Machine,
    PinnedArtifactPlan, PinnedSourceFile, PinnedSourceGroup, RepoRef, Store,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PinnedAsset {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

/// Written into the install after every file is verified. The field set is
/// the on-disk format of existing installs; changing it orphans them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PinnedDirReceipt {
    pub schema_version: u32,
    pub alias: String,
    pub repository: String,
    pub revision: String,
    pub family: String,
    pub files: BTreeMap<String, model_io::speech_receipt::FileEntry>,
}

/// What already sits at an install destination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Existing {
    Absent,
    /// A receipt for this exact pin with every file at its pinned size.
    Current,
    /// This alias and repository, but an older pin or a damaged install.
    /// Safe to replace: the receipt proves TurboSpark wrote it.
    Replaceable(String),
    /// Not provably ours (no receipt, another model, a symlink, a file).
    /// Never replaced.
    Foreign(String),
}

/// A borrowed view of one pinned profile.
pub(crate) struct PinnedDir<'a> {
    /// `embedding` or `ocr`: the owner-id prefix and the source role.
    pub kind: &'static str,
    pub receipt_name: &'static str,
    pub alias: &'a str,
    pub repository: &'a str,
    pub revision: &'a str,
    pub family: &'a str,
    pub assets: &'a [PinnedAsset],
}

fn safe_relative(path: &str) -> bool {
    !path.is_empty()
        && Path::new(path)
            .components()
            .all(|c| matches!(c, std::path::Component::Normal(_)))
}

fn hex64(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

pub(crate) fn valid_alias(alias: &str) -> bool {
    !alias.is_empty()
        && alias
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

impl PinnedDir<'_> {
    fn owner_id(&self) -> String {
        format!("{}:{}", self.kind, self.alias)
    }

    /// Per-profile manifest checks. Alias uniqueness is the catalog's job.
    pub fn validate(&self) -> Result<(), String> {
        if !valid_alias(self.alias) {
            return Err(format!("invalid {} alias {:?}", self.kind, self.alias));
        }
        if self.revision.len() != 40 || !self.revision.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!("{} does not pin an immutable revision", self.alias));
        }
        if self.assets.is_empty() {
            return Err(format!("{} has no assets", self.alias));
        }
        let mut paths = HashSet::new();
        for a in self.assets {
            if !safe_relative(&a.path) || a.size == 0 || !hex64(&a.sha256) || !paths.insert(&a.path)
            {
                return Err(format!("{} has an invalid asset {:?}", self.alias, a.path));
            }
            if a.path == self.receipt_name {
                return Err(format!("{} pins the receipt name as an asset", self.alias));
            }
        }
        Ok(())
    }

    pub fn plan(&self) -> PinnedArtifactPlan {
        PinnedArtifactPlan {
            owner_id: self.owner_id(),
            sources: vec![PinnedSourceGroup {
                role: self.kind.to_string(),
                repo: RepoRef::new(self.repository, self.revision),
                files: self
                    .assets
                    .iter()
                    .map(|a| PinnedSourceFile {
                        path: a.path.clone(),
                        expected_size: Some(a.size),
                        expected_sha256: Some(a.sha256.clone()),
                    })
                    .collect(),
            }],
        }
    }

    fn read_receipt(&self, dir: &Path) -> Result<PinnedDirReceipt, String> {
        let path = dir.join(self.receipt_name);
        let meta = std::fs::symlink_metadata(&path)
            .map_err(|e| format!("{}: no {} receipt ({e})", dir.display(), self.kind))?;
        if !meta.is_file() {
            return Err(format!("{} is not a regular file", path.display()));
        }
        let bytes = std::fs::read(&path)
            .map_err(|e| format!("{}: no {} receipt ({e})", dir.display(), self.kind))?;
        serde_json::from_slice(&bytes).map_err(|e| format!("{} receipt: {e}", self.kind))
    }

    /// The install holds a receipt for this exact pin, and every pinned file
    /// is a regular file (no symlinks) at its recorded size. Extra files,
    /// such as Core ML's `.compiled` cache, are allowed. A size check, not a
    /// re-hash: the transfer hashed every byte before publishing.
    pub fn verify(&self, dir: &Path) -> Result<(), String> {
        let r = self.read_receipt(dir)?;
        if r.alias != self.alias
            || r.revision != self.revision
            || !r.repository.eq_ignore_ascii_case(self.repository)
        {
            return Err(format!("{} receipt names a different pin", self.kind));
        }
        for a in self.assets {
            let meta = std::fs::symlink_metadata(dir.join(&a.path))
                .map_err(|e| format!("{}: {e}", a.path))?;
            if !meta.is_file() || meta.len() != a.size {
                return Err(format!("{} is missing or the wrong size", a.path));
            }
            if r.files.get(&a.path).map(|f| &f.sha256) != Some(&a.sha256) {
                return Err(format!("receipt does not record {}", a.path));
            }
        }
        Ok(())
    }

    pub fn existing(&self, dest: &Path) -> Existing {
        let meta = match std::fs::symlink_metadata(dest) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Existing::Absent,
            Err(e) => return Existing::Foreign(format!("{}: {e}", dest.display())),
            Ok(m) => m,
        };
        if !meta.is_dir() {
            return Existing::Foreign(format!("{} is not a directory", dest.display()));
        }
        let r = match self.read_receipt(dest) {
            Ok(r) => r,
            Err(e) => return Existing::Foreign(e),
        };
        if r.alias != self.alias || !r.repository.eq_ignore_ascii_case(self.repository) {
            return Existing::Foreign(format!(
                "its receipt names {} from {}",
                r.alias, r.repository
            ));
        }
        match self.verify(dest) {
            Ok(()) => Existing::Current,
            Err(e) => Existing::Replaceable(e),
        }
    }

    /// Checks the transfer returned exactly this profile's pinned files.
    fn check_receipt(&self, receipt: &DownloadReceipt) -> Result<PathBuf, String> {
        let kind = self.kind;
        if receipt.owner_id != self.owner_id() || receipt.sources.len() != 1 {
            return Err(format!(
                "{kind} transfer returned an unexpected owner or source count"
            ));
        }
        let group = &receipt.sources[0];
        if group.role != kind
            || !group.repo.repo.eq_ignore_ascii_case(self.repository)
            || group.repo.revision != self.revision
        {
            return Err(format!(
                "{kind} transfer does not match the pinned repository"
            ));
        }
        let root = receipt.staging_root.join("source-0000");
        let mut remaining: HashSet<&str> = self.assets.iter().map(|a| a.path.as_str()).collect();
        for f in &group.files {
            let asset = self
                .assets
                .iter()
                .find(|a| a.path == f.path)
                .ok_or_else(|| format!("{kind} transfer returned a foreign file {:?}", f.path))?;
            if !remaining.remove(f.path.as_str())
                || f.staged_path != root.join(&f.path)
                || f.size != asset.size
                || f.sha256 != asset.sha256
            {
                return Err(format!(
                    "{kind} transfer metadata does not match {}",
                    f.path
                ));
            }
        }
        if !remaining.is_empty() {
            return Err(format!("{kind} transfer omitted pinned files"));
        }
        Ok(root)
    }

    pub fn write_receipt(&self, dir: &Path) -> Result<(), String> {
        let receipt = PinnedDirReceipt {
            schema_version: 1,
            alias: self.alias.to_string(),
            repository: self.repository.to_string(),
            revision: self.revision.to_string(),
            family: self.family.to_string(),
            files: self
                .assets
                .iter()
                .map(|a| {
                    (
                        a.path.clone(),
                        model_io::speech_receipt::FileEntry {
                            size: a.size,
                            sha256: a.sha256.clone(),
                        },
                    )
                })
                .collect(),
        };
        let json = serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?;
        std::fs::write(dir.join(self.receipt_name), json)
            .map_err(|e| format!("writing receipt: {e}"))
    }

    /// Refuses a revision the caller did not see in the catalog, a current
    /// install, and a folder that is not provably ours, all before any
    /// download. Replaces an older pin of the same alias.
    #[allow(clippy::too_many_arguments)]
    pub fn install(
        &self,
        client: &Client,
        catalog: &Catalog,
        store: &Store,
        destination: &Path,
        requested_revision: &str,
        progress: &mut dyn FnMut(HubDownloadProgress),
        cancel: &CancelFlag,
    ) -> Result<PathBuf, String> {
        self.check_requested_revision(requested_revision)?;
        let replacing = match self.existing(destination) {
            Existing::Absent => false,
            Existing::Current => {
                return Err(format!(
                    "{} is already installed at this revision",
                    destination.display()
                ))
            }
            Existing::Foreign(why) => {
                return Err(format!(
                "{} is not a TurboSpark install of {} ({why}); move it aside before downloading",
                destination.display(),
                self.alias
            ))
            }
            Existing::Replaceable(_) => true,
        };
        let parent = destination
            .parent()
            .ok_or_else(|| format!("{} has no parent", destination.display()))?;
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("creating {} store: {e}", self.kind))?;
        sweep_tombstones(parent, self.alias);
        let hub = HubClient::new(
            client,
            catalog.clone(),
            Machine::default(),
            4096,
            model_io::ExpertCacheSlots::Auto,
        )
        .with_store(store.clone());
        let receipt = hub
            .download_pinned_artifacts(&self.plan(), store.root(), progress, cancel)
            .map_err(|e| e.to_string())?;
        let result = (|| {
            cancel.checkpoint().map_err(|e| e.to_string())?;
            let staged = self.check_receipt(&receipt)?;
            self.write_receipt(&staged)?;
            let tombstone = replacing.then(|| tombstone_path(parent, self.alias));
            publish(&staged, destination, tombstone.as_deref())
        })();
        let _ = std::fs::remove_dir_all(&receipt.staging_root);
        result?;
        // Best effort: the published install holds these bytes now, and a
        // leftover cache costs disk, not correctness.
        let _ = receipt.discard_download_cache();
        Ok(destination.to_path_buf())
    }

    pub fn check_requested_revision(&self, requested: &str) -> Result<(), String> {
        if requested != self.revision {
            return Err(format!(
                "{} is pinned at {}, not {requested}; refresh the model list",
                self.alias, self.revision
            ));
        }
        Ok(())
    }
}

/// Aliases are `[a-z0-9-]`, so the dot-separated prefix cannot match
/// another alias's tombstones.
fn tombstone_prefix(alias: &str) -> String {
    format!(".replace.{alias}.")
}

fn tombstone_path(parent: &Path, alias: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    parent.join(format!(
        "{}{}.{}",
        tombstone_prefix(alias),
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ))
}

/// Removes old-pin directories a crashed replace left behind. Best effort:
/// they hold superseded files only.
fn sweep_tombstones(parent: &Path, alias: &str) {
    let prefix = tombstone_prefix(alias);
    let Ok(entries) = std::fs::read_dir(parent) else {
        return;
    };
    for entry in entries.flatten() {
        let is_ours = entry
            .file_name()
            .to_str()
            .is_some_and(|n| n.starts_with(&prefix));
        let is_dir = entry.file_type().is_ok_and(|t| t.is_dir());
        if is_ours && is_dir {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

/// Publishes `staged` at `dest`. When replacing, the old install moves to
/// `tombstone` first and comes back if the publish rename fails, so a
/// failure never leaves `dest` empty.
fn publish(staged: &Path, dest: &Path, tombstone: Option<&Path>) -> Result<(), String> {
    let Some(tombstone) = tombstone else {
        return std::fs::rename(staged, dest)
            .map_err(|e| format!("publishing {}: {e}", dest.display()));
    };
    std::fs::rename(dest, tombstone).map_err(|e| format!("moving the older install aside: {e}"))?;
    if let Err(e) = std::fs::rename(staged, dest) {
        return match std::fs::rename(tombstone, dest) {
            Ok(()) => Err(format!("publishing {}: {e}", dest.display())),
            Err(rollback) => Err(format!(
                "publishing {}: {e}; restoring the older install failed: {rollback}",
                dest.display()
            )),
        };
    }
    let _ = std::fs::remove_dir_all(tombstone);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Tmp(PathBuf);
    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn tmp(tag: &str) -> Tmp {
        let d = std::env::temp_dir().join(format!(
            "ts-pinned-dir-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        Tmp(d)
    }

    fn assets() -> Vec<PinnedAsset> {
        vec![PinnedAsset {
            path: "pkg/w.bin".into(),
            size: 3,
            sha256: "b".repeat(64),
        }]
    }

    fn view<'a>(alias: &'a str, revision: &'a str, assets: &'a [PinnedAsset]) -> PinnedDir<'a> {
        PinnedDir {
            kind: "embedding",
            receipt_name: "embedding-receipt.json",
            alias,
            repository: "o/r",
            revision,
            family: "f",
            assets,
        }
    }

    fn write_install(p: &PinnedDir, dir: &Path) {
        std::fs::create_dir_all(dir.join("pkg")).unwrap();
        std::fs::write(dir.join("pkg/w.bin"), b"abc").unwrap();
        p.write_receipt(dir).unwrap();
    }

    #[test]
    fn existing_tells_ours_from_foreign() {
        let t = tmp("existing");
        let a = assets();
        let rev0 = "0".repeat(40);
        let rev1 = "1".repeat(40);
        let pin = view("m", &rev1, &a);
        let dest = t.0.join("m");

        assert_eq!(pin.existing(&dest), Existing::Absent);

        write_install(&pin, &dest);
        assert_eq!(pin.existing(&dest), Existing::Current);

        // Damaged at the current revision: ours, replaceable.
        std::fs::write(dest.join("pkg/w.bin"), b"abcd").unwrap();
        assert!(matches!(pin.existing(&dest), Existing::Replaceable(_)));

        // An older revision of the same alias and repository.
        std::fs::remove_dir_all(&dest).unwrap();
        write_install(&view("m", &rev0, &a), &dest);
        assert!(matches!(pin.existing(&dest), Existing::Replaceable(_)));

        // No receipt: not provably ours.
        std::fs::remove_file(dest.join("embedding-receipt.json")).unwrap();
        assert!(matches!(pin.existing(&dest), Existing::Foreign(_)));

        // Another alias's receipt.
        std::fs::remove_dir_all(&dest).unwrap();
        write_install(&view("other", &rev1, &a), &dest);
        assert!(matches!(pin.existing(&dest), Existing::Foreign(_)));

        // A symlink to a real install is not followed.
        std::fs::remove_dir_all(&dest).unwrap();
        let real = t.0.join("real");
        write_install(&pin, &real);
        std::os::unix::fs::symlink(&real, &dest).unwrap();
        assert!(matches!(pin.existing(&dest), Existing::Foreign(_)));
    }

    #[test]
    fn publish_replaces_and_rolls_back() {
        let t = tmp("publish");
        let dest = t.0.join("m");
        std::fs::create_dir_all(&dest).unwrap();
        std::fs::write(dest.join("old"), b"old").unwrap();

        // Staging is missing: the rename fails and the old install returns.
        let missing = t.0.join("staged-missing");
        let tomb = tombstone_path(&t.0, "m");
        assert!(publish(&missing, &dest, Some(&tomb)).is_err());
        assert!(dest.join("old").is_file(), "old install restored");
        assert!(!tomb.exists());

        let staged = t.0.join("staged");
        std::fs::create_dir_all(&staged).unwrap();
        std::fs::write(staged.join("new"), b"new").unwrap();
        let tomb = tombstone_path(&t.0, "m");
        publish(&staged, &dest, Some(&tomb)).unwrap();
        assert!(dest.join("new").is_file());
        assert!(!dest.join("old").exists());
        assert!(!tomb.exists(), "tombstone removed");
    }

    #[test]
    fn sweep_removes_only_this_alias() {
        let t = tmp("sweep");
        let mine = tombstone_path(&t.0, "a");
        let other = tombstone_path(&t.0, "a-b");
        std::fs::create_dir_all(&mine).unwrap();
        std::fs::create_dir_all(&other).unwrap();
        sweep_tombstones(&t.0, "a");
        assert!(!mine.exists());
        assert!(other.exists());
    }

    #[test]
    fn install_refuses_before_any_io() {
        let t = tmp("refuse");
        let a = assets();
        let rev = "1".repeat(40);
        let pin = view("m", &rev, &a);
        let store = Store::new(t.0.join("home"));
        let parent = t.0.join("home/models/embeddings");
        let dest = parent.join("m");
        let client = Client::new();
        let catalog = Catalog::embedded().unwrap();
        let cancel = CancelFlag::default();

        // A revision the catalog does not pin.
        let err = pin
            .install(
                &client,
                &catalog,
                &store,
                &dest,
                "main",
                &mut |_| {},
                &cancel,
            )
            .unwrap_err();
        assert!(err.contains("pinned at"), "{err}");
        assert!(
            !parent.exists(),
            "nothing created before the revision check"
        );

        // A current install.
        write_install(&pin, &dest);
        let err = pin
            .install(&client, &catalog, &store, &dest, &rev, &mut |_| {}, &cancel)
            .unwrap_err();
        assert!(err.contains("already installed"), "{err}");

        // A folder that is not ours.
        std::fs::remove_file(dest.join("embedding-receipt.json")).unwrap();
        let err = pin
            .install(&client, &catalog, &store, &dest, &rev, &mut |_| {}, &cancel)
            .unwrap_err();
        assert!(err.contains("not a TurboSpark install"), "{err}");
        assert!(dest.join("pkg/w.bin").is_file(), "foreign folder untouched");
    }

    #[test]
    fn validate_refuses_unsafe_or_unpinned() {
        let a = assets();
        let rev = "0".repeat(40);
        assert!(view("ok-1", &rev, &a).validate().is_ok());
        assert!(view("Bad", &rev, &a).validate().is_err());
        assert!(view("ok", "main", &a).validate().is_err());
        assert!(view("ok", &rev, &[]).validate().is_err());
        for bad in ["../a", "/a", "", "embedding-receipt.json"] {
            let a = vec![PinnedAsset {
                path: bad.into(),
                size: 1,
                sha256: "a".repeat(64),
            }];
            assert!(view("ok", &rev, &a).validate().is_err(), "{bad:?}");
        }
    }
}
