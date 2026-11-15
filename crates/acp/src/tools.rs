//! The v1 tool surface: read_file, list_dir, write_file, edit_file.
//!
//! Shell, search, and anything networked are out of scope for v1 (the draft
//! resolves the tool-surface question to file tools only; shell through an
//! editor channel needs its own threat review).
//!
//! Risk ladder (narrower than the Swift app's, stated in the draft): reads
//! are `Read` and run without a prompt, inside the workspace root only;
//! writes are `Write` and require the editor's approval. Anything not named
//! here does not exist.

use crate::paths::{self, PathError};
use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Risk {
    Read,
    Write,
}

impl Risk {
    /// The ACP `ToolKind` shown in the editor.
    pub fn acp_kind(self) -> &'static str {
        match self {
            Risk::Read => "read",
            Risk::Write => "edit",
        }
    }
    /// The permission name a grant and a denial message use.
    pub fn permission(self) -> &'static str {
        match self {
            Risk::Read => "read",
            Risk::Write => "edit",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// Most bytes `read_file` returns (the rest is reported as truncated).
    pub max_output_bytes: usize,
    /// Largest file `edit_file` will rewrite, and largest `write_file` body.
    pub max_file_bytes: usize,
    pub max_list_entries: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_output_bytes: 256 * 1024,
            max_file_bytes: 1024 * 1024,
            max_list_entries: 1000,
        }
    }
}

#[derive(Debug)]
pub enum Action {
    ReadFile {
        line: Option<usize>,
        limit: Option<usize>,
    },
    ListDir,
    WriteFile {
        content: String,
    },
    EditFile {
        old: String,
        new: String,
    },
}

#[derive(Debug)]
pub struct Plan {
    pub risk: Risk,
    pub path: PathBuf,
    pub action: Action,
    pub title: String,
}

fn str_arg<'a>(args: &'a Value, k: &str) -> Result<&'a str, String> {
    args.get(k)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing or non-string argument '{k}'"))
}

fn uint_arg(args: &Value, k: &str) -> Result<Option<usize>, String> {
    match args.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_u64()
            .map(|n| Some(n as usize))
            .ok_or_else(|| format!("argument '{k}' must be a non-negative integer")),
    }
}

/// Validates arguments and resolves the path inside `root`. Runs BEFORE any
/// permission prompt, so an escaping path is refused without asking the user
/// to approve something that could never run.
pub fn plan(root: &Path, name: &str, args: &Value, limits: &Limits) -> Result<Plan, String> {
    let raw = str_arg(args, "path")?;
    let path = paths::resolve(root, raw).map_err(|e| match e {
        PathError::Escapes => "refused: path is outside the workspace root".to_string(),
        PathError::Invalid(m) => format!("refused: {m}"),
    })?;
    let rel = path
        .strip_prefix(root)
        .unwrap_or(&path)
        .display()
        .to_string();
    let rel = if rel.is_empty() { ".".to_string() } else { rel };
    let (risk, action, title) = match name {
        "read_file" => (
            Risk::Read,
            Action::ReadFile {
                line: uint_arg(args, "line")?,
                limit: uint_arg(args, "limit")?,
            },
            format!("Read {rel}"),
        ),
        "list_dir" => (Risk::Read, Action::ListDir, format!("List {rel}")),
        "write_file" => {
            let content = str_arg(args, "content")?;
            if content.len() > limits.max_file_bytes {
                return Err(format!(
                    "refused: content is {} bytes, over the {} byte limit",
                    content.len(),
                    limits.max_file_bytes
                ));
            }
            (
                Risk::Write,
                Action::WriteFile {
                    content: content.to_string(),
                },
                format!("Write {rel}"),
            )
        }
        "edit_file" => {
            let old = str_arg(args, "old_text")?;
            let new = str_arg(args, "new_text")?;
            if old.is_empty() {
                return Err("old_text must not be empty".to_string());
            }
            if new.len() > limits.max_file_bytes {
                return Err("refused: new_text is over the size limit".to_string());
            }
            (
                Risk::Write,
                Action::EditFile {
                    old: old.to_string(),
                    new: new.to_string(),
                },
                format!("Edit {rel}"),
            )
        }
        other => {
            return Err(format!(
                "unknown tool '{}'",
                other.chars().take(40).collect::<String>()
            ))
        }
    };
    Ok(Plan {
        risk,
        path,
        action,
        title,
    })
}

/// File access, so the editor's buffers (client `fs/*`) and the disk can sit
/// behind one execution path.
pub trait FileBackend {
    /// Returns the text and whether it was cut at `max_bytes`.
    fn read_text(&self, path: &Path, max_bytes: usize) -> Result<(String, bool), String>;
    fn write_text(&self, path: &Path, content: &str) -> Result<(), String>;
}

pub struct DiskBackend;

impl FileBackend for DiskBackend {
    fn read_text(&self, path: &Path, max_bytes: usize) -> Result<(String, bool), String> {
        use std::io::Read;
        let meta = std::fs::metadata(path).map_err(|e| io_msg("cannot read", &e))?;
        if !meta.is_file() {
            return Err("not a regular file".to_string());
        }
        let f = std::fs::File::open(path).map_err(|e| io_msg("cannot open", &e))?;
        let mut buf = Vec::new();
        f.take(max_bytes as u64 + 1)
            .read_to_end(&mut buf)
            .map_err(|e| io_msg("cannot read", &e))?;
        let truncated = buf.len() > max_bytes;
        buf.truncate(max_bytes);
        utf8_prefix(buf, truncated)
    }

    fn write_text(&self, path: &Path, content: &str) -> Result<(), String> {
        // The parent must already exist: the tool never creates directories.
        std::fs::write(path, content).map_err(|e| io_msg("cannot write", &e))
    }
}

fn io_msg(what: &str, e: &std::io::Error) -> String {
    // Kind only: the OS message can embed the absolute path.
    format!("{what}: {}", e.kind())
}

pub fn utf8_prefix(buf: Vec<u8>, truncated: bool) -> Result<(String, bool), String> {
    match String::from_utf8(buf) {
        Ok(s) => Ok((s, truncated)),
        Err(e) => {
            let valid = e.utf8_error().valid_up_to();
            // A cut inside a multi-byte char is fine; a bad byte elsewhere is not.
            if truncated && e.utf8_error().error_len().is_none() {
                let mut b = e.into_bytes();
                b.truncate(valid);
                Ok((String::from_utf8(b).unwrap_or_default(), true))
            } else {
                Err("file is not valid UTF-8 text".to_string())
            }
        }
    }
}

pub fn execute(plan: &Plan, backend: &dyn FileBackend, limits: &Limits) -> Result<String, String> {
    match &plan.action {
        Action::ReadFile { line, limit } => {
            let (text, truncated) = backend.read_text(&plan.path, limits.max_output_bytes)?;
            let selected: String = if line.is_some() || limit.is_some() {
                let start = line.unwrap_or(1).max(1) - 1;
                let it = text.lines().skip(start);
                match limit {
                    Some(n) => it.take(*n).collect::<Vec<_>>().join("\n"),
                    None => it.collect::<Vec<_>>().join("\n"),
                }
            } else {
                text
            };
            Ok(if truncated {
                format!(
                    "{selected}\n[truncated at {} bytes]",
                    limits.max_output_bytes
                )
            } else {
                selected
            })
        }
        Action::ListDir => {
            let rd = std::fs::read_dir(&plan.path).map_err(|e| io_msg("cannot list", &e))?;
            let mut names: Vec<String> = Vec::new();
            let mut more = false;
            for entry in rd {
                let Ok(entry) = entry else { continue };
                if names.len() >= limits.max_list_entries {
                    more = true;
                    break;
                }
                let mut n = entry.file_name().to_string_lossy().into_owned();
                match entry.file_type() {
                    Ok(t) if t.is_dir() => n.push('/'),
                    Ok(t) if t.is_symlink() => n.push('@'),
                    _ => {}
                }
                names.push(n);
            }
            names.sort();
            let mut out = names.join("\n");
            if more {
                out.push_str("\n[more entries not shown]");
            }
            Ok(out)
        }
        Action::WriteFile { content } => {
            backend.write_text(&plan.path, content)?;
            Ok(format!("wrote {} bytes", content.len()))
        }
        Action::EditFile { old, new } => {
            let (text, truncated) = backend.read_text(&plan.path, limits.max_file_bytes)?;
            if truncated {
                return Err(format!(
                    "file is over the {} byte edit limit",
                    limits.max_file_bytes
                ));
            }
            let n = text.matches(old.as_str()).count();
            if n == 0 {
                return Err("old_text was not found in the file".to_string());
            }
            if n > 1 {
                return Err(format!(
                    "old_text matches {n} places; include more context so it matches once"
                ));
            }
            let updated = text.replacen(old.as_str(), new, 1);
            if updated.len() > limits.max_file_bytes {
                return Err("edited file would exceed the size limit".to_string());
            }
            backend.write_text(&plan.path, &updated)?;
            Ok("edit applied".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tmp(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("tsacp-tools-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        std::fs::canonicalize(p).unwrap()
    }

    #[test]
    fn classification() {
        let root = tmp("cls");
        let l = Limits::default();
        let r = plan(&root, "read_file", &json!({"path":"a"}), &l).unwrap();
        assert_eq!(r.risk, Risk::Read);
        let w = plan(&root, "write_file", &json!({"path":"a","content":"x"}), &l).unwrap();
        assert_eq!(w.risk, Risk::Write);
        let e = plan(
            &root,
            "edit_file",
            &json!({"path":"a","old_text":"x","new_text":"y"}),
            &l,
        )
        .unwrap();
        assert_eq!(e.risk, Risk::Write);
        assert!(plan(&root, "run_shell", &json!({"path":"a"}), &l).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn escape_is_refused_at_plan_time() {
        let root = tmp("esc");
        let l = Limits::default();
        let e = plan(
            &root,
            "write_file",
            &json!({"path":"../x","content":""}),
            &l,
        )
        .unwrap_err();
        assert!(e.contains("outside the workspace root"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_write_edit_roundtrip_on_disk() {
        let root = tmp("rw");
        let l = Limits::default();
        let w = plan(
            &root,
            "write_file",
            &json!({"path":"f.txt","content":"a\nb\nc\n"}),
            &l,
        )
        .unwrap();
        execute(&w, &DiskBackend, &l).unwrap();
        let r = plan(
            &root,
            "read_file",
            &json!({"path":"f.txt","line":2,"limit":1}),
            &l,
        )
        .unwrap();
        assert_eq!(execute(&r, &DiskBackend, &l).unwrap(), "b");
        let e = plan(
            &root,
            "edit_file",
            &json!({"path":"f.txt","old_text":"b","new_text":"B"}),
            &l,
        )
        .unwrap();
        execute(&e, &DiskBackend, &l).unwrap();
        assert_eq!(
            std::fs::read_to_string(root.join("f.txt")).unwrap(),
            "a\nB\nc\n"
        );
        let dup = plan(
            &root,
            "edit_file",
            &json!({"path":"f.txt","old_text":"\n","new_text":"x"}),
            &l,
        )
        .unwrap();
        assert!(execute(&dup, &DiskBackend, &l)
            .unwrap_err()
            .contains("matches 3"));
        let miss = plan(
            &root,
            "edit_file",
            &json!({"path":"f.txt","old_text":"zzz","new_text":"x"}),
            &l,
        )
        .unwrap();
        assert!(execute(&miss, &DiskBackend, &l).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_is_truncated_at_the_output_bound() {
        let root = tmp("trunc");
        std::fs::write(root.join("big"), "\u{e9}".repeat(100)).unwrap();
        let l = Limits {
            max_output_bytes: 11,
            ..Limits::default()
        };
        let r = plan(&root, "read_file", &json!({"path":"big"}), &l).unwrap();
        let out = execute(&r, &DiskBackend, &l).unwrap();
        assert!(out.ends_with("[truncated at 11 bytes]"));
        // 11 bytes cut a 2-byte char in half: only 5 whole chars remain.
        assert!(out.starts_with(&"\u{e9}".repeat(5)));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn write_over_size_limit_is_refused_and_non_utf8_read_fails() {
        let root = tmp("lim");
        let l = Limits {
            max_file_bytes: 4,
            ..Limits::default()
        };
        assert!(plan(
            &root,
            "write_file",
            &json!({"path":"a","content":"12345"}),
            &l
        )
        .is_err());
        std::fs::write(root.join("bin"), [0xff, 0xfe, 0x00]).unwrap();
        let r = plan(
            &root,
            "read_file",
            &json!({"path":"bin"}),
            &Limits::default(),
        )
        .unwrap();
        assert!(execute(&r, &DiskBackend, &Limits::default()).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn list_dir_marks_directories_and_sorts() {
        let root = tmp("ls");
        std::fs::create_dir(root.join("d")).unwrap();
        std::fs::write(root.join("b"), "").unwrap();
        std::fs::write(root.join("a"), "").unwrap();
        let l = Limits::default();
        let p = plan(&root, "list_dir", &json!({"path":"."}), &l).unwrap();
        assert_eq!(execute(&p, &DiskBackend, &l).unwrap(), "a\nb\nd/");
        let _ = std::fs::remove_dir_all(&root);
    }
}
