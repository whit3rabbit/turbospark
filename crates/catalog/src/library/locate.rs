//! Where a library sits on its drive.
//!
//! A library is recorded as a volume root plus a path relative to it, so a
//! drive that is renamed or remounted elsewhere is searched at the same
//! relative place. Everything here works from paths and `st_dev` alone, so
//! tests drive it with a temp folder of fake volumes.

use super::LocationEnv;
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};

/// The volume root of `path`, which must exist and be canonical.
///
/// A direct child of `env.volumes_root` is a volume root, which makes fake
/// volumes under a temp folder behave like `/Volumes/<name>`. Otherwise the
/// root is the highest ancestor still on the path's device. On APFS every
/// internal path, including the firmlinked Data volume, reports the device
/// of `/`, so an internal folder resolves to `/`.
pub fn volume_root_of(path: &Path, env: &LocationEnv) -> std::io::Result<PathBuf> {
    if let Ok(rest) = path.strip_prefix(&env.volumes_root) {
        if let Some(Component::Normal(volume)) = rest.components().next() {
            return Ok(env.volumes_root.join(volume));
        }
    }
    let device = std::fs::metadata(path)?.dev();
    let mut root = path;
    for ancestor in path.ancestors().skip(1) {
        if std::fs::metadata(ancestor)?.dev() != device {
            break;
        }
        root = ancestor;
    }
    Ok(root.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::library::MountSource;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ts-library-locate-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // The temp dir is behind the /var symlink, and the volume-root rules
        // compare canonical paths.
        dir.canonicalize().unwrap()
    }

    fn env_with_volumes_root(volumes_root: PathBuf) -> LocationEnv {
        LocationEnv {
            mount_source: MountSource::Fixed(Vec::new()),
            volumes_root,
            home_device: 0,
        }
    }

    #[test]
    fn a_direct_child_of_the_volumes_root_is_the_volume_root() {
        let base = scratch("direct-child");
        let volumes_root = base.join("Volumes");
        let library = volumes_root.join("Embedding Gemma Transcribe/TurboSpark/models");
        std::fs::create_dir_all(&library).unwrap();
        let env = env_with_volumes_root(volumes_root.clone());
        let volume = volumes_root.join("Embedding Gemma Transcribe");

        // Fake volumes share the temp dir's device, so only the direct-child
        // rule can stop the walk at the fake volume rather than at `/`.
        assert_eq!(volume_root_of(&library, &env).unwrap(), volume);
        assert_eq!(volume_root_of(&volume, &env).unwrap(), volume);

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn outside_the_volumes_root_the_highest_ancestor_on_the_same_device_is_the_root() {
        let base = scratch("same-device");
        let folder = base.join("Models");
        std::fs::create_dir_all(&folder).unwrap();
        // A volumes root that does not contain the folder, so the device walk
        // decides. On macOS every internal path, including the firmlinked
        // Data volume, reports the same device as `/`.
        let env = env_with_volumes_root(base.join("Volumes"));

        assert_eq!(volume_root_of(&folder, &env).unwrap(), PathBuf::from("/"));

        let _ = std::fs::remove_dir_all(&base);
    }
}
