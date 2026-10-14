//! Compile-cache naming. Portable and pure apart from reading file
//! metadata, so the key is tested on every target.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Content key for a packaged model, as 16 hex characters.
///
/// Covers the package `Manifest.json` bytes, then every file under the
/// package (relative path, size, mtime) in sorted order, then `os_tag`.
/// Hashing all weight bytes would cost seconds per gigabyte on every open;
/// size plus mtime catches a re-download, and the manifest catches a
/// different conversion. `os_tag` is the OS build, because a compiled
/// `.mlmodelc` comes from the OS-bundled compiler and an OS update can
/// change its output.
pub fn cache_key(package: &Path, os_tag: &str) -> io::Result<String> {
    let mut h = Sha256::new();
    let meta = fs::metadata(package)?;
    if meta.is_dir() {
        let manifest = package.join("Manifest.json");
        if manifest.is_file() {
            h.update(b"manifest\0");
            h.update(fs::read(&manifest)?);
        }
        let mut files = Vec::new();
        walk(package, package, &mut files)?;
        files.sort();
        for rel in files {
            let m = fs::metadata(package.join(&rel))?;
            h.update(rel.to_string_lossy().as_bytes());
            h.update([0]);
            h.update(m.len().to_le_bytes());
            h.update(mtime_nanos(&m).to_le_bytes());
        }
    } else {
        // A single-file `.mlmodel` is small enough to hash outright.
        h.update(b"file\0");
        h.update(fs::read(package)?);
    }
    h.update(b"os\0");
    h.update(os_tag.as_bytes());
    let digest = h.finalize();
    Ok(digest[..8].iter().map(|b| format!("{b:02x}")).collect())
}

/// `<cache_root>/<stem>-<key>.mlmodelc`.
pub fn compiled_path_for(package: &Path, cache_root: &Path, key: &str) -> PathBuf {
    let stem = package
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".into());
    cache_root.join(format!("{stem}-{key}.mlmodelc"))
}

/// True when `name` is a cache entry this crate wrote for `stem`
/// (`<stem>-<16 hex>.mlmodelc`). Pruning only ever touches such names.
pub(crate) fn is_cache_entry_for(name: &str, stem: &str) -> bool {
    let Some(rest) = name.strip_prefix(stem) else {
        return false;
    };
    let Some(rest) = rest.strip_prefix('-') else {
        return false;
    };
    let Some(hex) = rest.strip_suffix(".mlmodelc") else {
        return false;
    };
    hex.len() == 16
        && hex
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let path = entry.path();
        if ty.is_dir() {
            walk(root, &path, out)?;
        } else if ty.is_file() {
            out.push(path.strip_prefix(root).unwrap_or(&path).to_path_buf());
        }
        // Symlinks are skipped: an install never contains them, and
        // following one could hash files outside the package.
    }
    Ok(())
}

fn mtime_nanos(m: &fs::Metadata) -> u128 {
    m.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

/// Copies a directory tree. Used when the compiler's temporary output sits
/// on a different volume than the cache and `rename` fails.
pub(crate) fn copy_dir(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let to = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir(&entry.path(), &to)?;
        } else if ty.is_file() {
            fs::copy(entry.path(), &to)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "ts-coreml-cache-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn key_changes_with_content_size_and_os() {
        let d = scratch("key");
        let pkg = d.join("M.mlpackage");
        fs::create_dir_all(pkg.join("Data/w")).unwrap();
        fs::write(pkg.join("Manifest.json"), b"{\"a\":1}").unwrap();
        fs::write(pkg.join("Data/w/weight.bin"), b"abcd").unwrap();
        let k1 = cache_key(&pkg, "25G83").unwrap();
        assert_eq!(k1.len(), 16);
        assert_eq!(k1, cache_key(&pkg, "25G83").unwrap());
        assert_ne!(k1, cache_key(&pkg, "25H1").unwrap());
        fs::write(pkg.join("Data/w/weight.bin"), b"abcde").unwrap();
        let k2 = cache_key(&pkg, "25G83").unwrap();
        assert_ne!(k1, k2);
        fs::write(pkg.join("Manifest.json"), b"{\"a\":2}").unwrap();
        assert_ne!(k2, cache_key(&pkg, "25G83").unwrap());
        fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn compiled_path_and_entry_match() {
        let p = compiled_path_for(
            Path::new("/x/Enc.mlpackage"),
            Path::new("/c"),
            "0123456789abcdef",
        );
        assert_eq!(p, Path::new("/c/Enc-0123456789abcdef.mlmodelc"));
        assert!(is_cache_entry_for("Enc-0123456789abcdef.mlmodelc", "Enc"));
        assert!(!is_cache_entry_for("Enc-0123456789ABCDEF.mlmodelc", "Enc"));
        assert!(!is_cache_entry_for(
            "Encoder-0123456789abcdef.mlmodelc",
            "Enc"
        ));
        assert!(!is_cache_entry_for("Enc-0123.mlmodelc", "Enc"));
        assert!(!is_cache_entry_for("Enc-0123456789abcdef.mlpackage", "Enc"));
    }

    #[test]
    fn copy_dir_copies_tree() {
        let d = scratch("copy");
        fs::create_dir_all(d.join("a/b")).unwrap();
        fs::write(d.join("a/b/f"), b"x").unwrap();
        copy_dir(&d.join("a"), &d.join("z")).unwrap();
        assert_eq!(fs::read(d.join("z/b/f")).unwrap(), b"x");
        fs::remove_dir_all(&d).unwrap();
    }
}
