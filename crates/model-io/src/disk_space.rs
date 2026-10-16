//! Free space on the volume a path will be created on.
//!
//! A multi-gigabyte install checks this before it downloads anything. The
//! catalog is `forbid(unsafe_code)` and the CLI should not own it either, so
//! the one `statvfs` call lives here, where the workspace's narrow `unsafe`
//! for model I/O already is.

use std::path::Path;

/// Bytes a non-root user can still write where `path` will be created: the
/// nearest existing ancestor's filesystem.
#[cfg(unix)]
pub fn available_bytes(path: &Path) -> Result<u64, String> {
    use std::os::unix::ffi::OsStrExt;
    let mut probe = path.to_path_buf();
    if probe.is_relative() {
        probe = std::env::current_dir()
            .map_err(|e| format!("reading the working directory: {e}"))?
            .join(probe);
    }
    while !probe.exists() {
        if !probe.pop() {
            return Err(format!(
                "no existing parent directory for {}",
                path.display()
            ));
        }
    }
    let c_path = std::ffi::CString::new(probe.as_os_str().as_bytes())
        .map_err(|_| "the path contains a NUL byte".to_string())?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `c_path` is a valid NUL-terminated string and `stats` is a
    // writable statvfs; the struct is read only after a zero return.
    let status = unsafe { libc::statvfs(c_path.as_ptr(), stats.as_mut_ptr()) };
    if status != 0 {
        return Err(format!(
            "reading free space on {}: {}",
            probe.display(),
            std::io::Error::last_os_error()
        ));
    }
    // SAFETY: statvfs returned zero, so it filled the struct.
    let stats = unsafe { stats.assume_init() };
    // The field widths differ by platform (u32 and u64 on macOS), so the
    // casts are not useless everywhere.
    #[allow(clippy::unnecessary_cast)]
    let free = (stats.f_bavail as u64).saturating_mul(stats.f_frsize as u64);
    Ok(free)
}

#[cfg(not(unix))]
pub fn available_bytes(_path: &Path) -> Result<u64, String> {
    Err("the free-disk check needs a unix platform".to_string())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn reads_the_nearest_existing_ancestor() {
        let root = std::env::temp_dir().join(format!("disk_space_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let deep = root.join("not").join("yet").join("made");
        let at_root = available_bytes(&root).unwrap();
        let at_deep = available_bytes(&deep).unwrap();
        assert!(at_root > 0);
        // Same filesystem, so within a whisker of each other.
        assert!(at_deep.abs_diff(at_root) < 1 << 30);
        let _ = std::fs::remove_dir_all(&root);
    }
}
