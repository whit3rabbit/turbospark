//! Workspace-root validation and path containment.
//!
//! The root is declared by the CLIENT at `session/new`, so it is untrusted:
//! `/` or the home directory would make "confined to the workspace" empty.
//! Every tool path then resolves against the canonical root and must land
//! inside it after symlinks are resolved.
//!
//! Known limit: a check-then-use race (a symlink swapped between resolution
//! and the file operation) is not closed. The editor user and the adapter run
//! as the same user on the same machine; this guards the model, not a local
//! attacker racing the filesystem.

use std::path::{Component, Path, PathBuf};

#[derive(Debug, PartialEq, Eq)]
pub enum PathError {
    Invalid(String),
    Escapes,
}

impl std::fmt::Display for PathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PathError::Invalid(m) => write!(f, "{m}"),
            PathError::Escapes => write!(f, "path is outside the workspace root"),
        }
    }
}

/// Validates the client-declared working directory and returns its canonical
/// form. `allowed_roots` empty means any directory that passes the other
/// rules; non-empty means the root must lie inside one of them.
pub fn validate_root(cwd: &str, allowed_roots: &[PathBuf]) -> Result<PathBuf, PathError> {
    if cwd.is_empty() || cwd.contains('\0') {
        return Err(PathError::Invalid("cwd is empty or contains NUL".into()));
    }
    let p = Path::new(cwd);
    if !p.is_absolute() {
        return Err(PathError::Invalid("cwd must be an absolute path".into()));
    }
    let canon = std::fs::canonicalize(p)
        .map_err(|_| PathError::Invalid("cwd does not exist or is not readable".into()))?;
    if !canon.is_dir() {
        return Err(PathError::Invalid("cwd is not a directory".into()));
    }
    if canon.parent().is_none() {
        return Err(PathError::Invalid(
            "the filesystem root is not an acceptable workspace root".into(),
        ));
    }
    if let Some(home) = std::env::var_os("HOME") {
        if let Ok(h) = std::fs::canonicalize(home) {
            if h == canon {
                return Err(PathError::Invalid(
                    "the home directory itself is not an acceptable workspace root; \
                     open a project folder"
                        .into(),
                ));
            }
        }
    }
    if !allowed_roots.is_empty() {
        let ok = allowed_roots.iter().any(|a| {
            std::fs::canonicalize(a)
                .map(|a| canon.starts_with(a))
                .unwrap_or(false)
        });
        if !ok {
            return Err(PathError::Invalid(
                "cwd is not inside an allowed workspace root".into(),
            ));
        }
    }
    Ok(canon)
}

/// Resolves a tool path (absolute, or relative to `root`) to a real path that
/// is inside `root`. The target need not exist (a write may create it) but its
/// nearest existing ancestor must resolve inside the root.
pub fn resolve(root: &Path, input: &str) -> Result<PathBuf, PathError> {
    if input.is_empty() || input.contains('\0') {
        return Err(PathError::Invalid("path is empty or contains NUL".into()));
    }
    let p = Path::new(input);
    let joined = if p.is_absolute() {
        p.to_path_buf()
    } else {
        root.join(p)
    };

    // Lexical normalization first, so no `..` survives into the remainder.
    let mut norm = PathBuf::new();
    for c in joined.components() {
        match c {
            Component::Prefix(_) | Component::RootDir => norm.push(c.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if !norm.pop() {
                    return Err(PathError::Escapes);
                }
            }
            Component::Normal(s) => norm.push(s),
        }
    }

    // Canonicalize the longest existing prefix (follows symlinks, and fixes
    // forms like /var vs /private/var), then re-attach the missing tail.
    let mut existing = norm.clone();
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    loop {
        match std::fs::symlink_metadata(&existing) {
            Ok(_) => break,
            Err(_) => {
                let Some(name) = existing.file_name().map(|n| n.to_os_string()) else {
                    return Err(PathError::Escapes);
                };
                tail.push(name);
                if !existing.pop() {
                    return Err(PathError::Escapes);
                }
            }
        }
    }
    let mut real = std::fs::canonicalize(&existing)
        .map_err(|_| PathError::Invalid("path cannot be resolved".into()))?;
    for name in tail.into_iter().rev() {
        real.push(name);
    }
    if !real.starts_with(root) {
        return Err(PathError::Escapes);
    }
    Ok(real)
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) struct TempDir(pub PathBuf);
    impl TempDir {
        pub fn new(tag: &str) -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            let p = std::env::temp_dir().join(format!(
                "tsacp-{tag}-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&p).unwrap();
            TempDir(std::fs::canonicalize(p).unwrap())
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn relative_and_absolute_inside_root_resolve() {
        let t = TempDir::new("in");
        std::fs::write(t.0.join("a.txt"), "x").unwrap();
        assert_eq!(resolve(&t.0, "a.txt").unwrap(), t.0.join("a.txt"));
        assert_eq!(
            resolve(&t.0, t.0.join("a.txt").to_str().unwrap()).unwrap(),
            t.0.join("a.txt")
        );
        // A not-yet-existing file in an existing dir is fine (write target).
        assert_eq!(resolve(&t.0, "new.txt").unwrap(), t.0.join("new.txt"));
    }

    #[test]
    fn dotdot_escape_is_refused() {
        let t = TempDir::new("dd");
        assert_eq!(resolve(&t.0, "../outside").unwrap_err(), PathError::Escapes);
        assert_eq!(
            resolve(&t.0, "sub/../../x").unwrap_err(),
            PathError::Escapes
        );
        assert_eq!(
            resolve(&t.0, "/etc/passwd").unwrap_err(),
            PathError::Escapes
        );
    }

    #[test]
    fn dotdot_that_stays_inside_is_allowed() {
        let t = TempDir::new("dd2");
        std::fs::create_dir(t.0.join("sub")).unwrap();
        assert_eq!(resolve(&t.0, "sub/../a").unwrap(), t.0.join("a"));
    }

    #[cfg(unix)]
    #[test]
    fn symlink_out_of_root_is_refused() {
        let t = TempDir::new("sl");
        let outside = TempDir::new("sl-out");
        std::fs::write(outside.0.join("secret"), "s").unwrap();
        std::os::unix::fs::symlink(&outside.0, t.0.join("link")).unwrap();
        assert_eq!(
            resolve(&t.0, "link/secret").unwrap_err(),
            PathError::Escapes
        );
        // Creating a new file through the escaping link is refused too.
        assert_eq!(resolve(&t.0, "link/new").unwrap_err(), PathError::Escapes);
    }

    #[test]
    fn root_rules() {
        let t = TempDir::new("root");
        assert!(validate_root(t.0.to_str().unwrap(), &[]).is_ok());
        assert!(validate_root("relative/dir", &[]).is_err());
        assert!(validate_root("/", &[]).is_err());
        assert!(validate_root("", &[]).is_err());
        assert!(validate_root(t.0.join("missing").to_str().unwrap(), &[]).is_err());
        let f = t.0.join("file");
        std::fs::write(&f, "x").unwrap();
        assert!(validate_root(f.to_str().unwrap(), &[]).is_err());
    }

    #[test]
    fn allow_list_is_enforced() {
        let a = TempDir::new("allow-a");
        let b = TempDir::new("allow-b");
        let allow = vec![a.0.clone()];
        assert!(validate_root(a.0.to_str().unwrap(), &allow).is_ok());
        assert!(validate_root(b.0.to_str().unwrap(), &allow).is_err());
    }

    #[test]
    fn home_itself_is_refused() {
        // HOME is process-global; only assert when it is set and real.
        if let Some(h) = std::env::var_os("HOME") {
            if std::fs::canonicalize(&h).is_ok() {
                assert!(validate_root(h.to_str().unwrap(), &[]).is_err());
            }
        }
    }
}
