//! Local installs of Inkling 2 Mini from the two sources that need whole
//! files on disk rather than the range-streamed walk: a LookAtMyQuant packed
//! release directory (`docs/LAMQ.md`) and an Inkling 2 GGUF
//! (`docs/INKLING2_PHASE0.md`). The CLI (`turbospark model pull-local`), the
//! FFI and the Hugging Face route (which downloads the files first) all come
//! through [`install_inkling2_local`], so they share one order of steps:
//!
//! 1. refuse an existing destination; check free space;
//! 2. write the install into a sibling `.partial` staging directory;
//! 3. copy the tokenizer and chat files (from the LAMQ directory, or for a
//!    GGUF, which carries none, from a directory the caller names);
//! 4. read the result back as a run would (tokenizer renders a turn; manifest,
//!    resident index and expert layout load);
//! 5. rename into place and record the install.
//!
//! A failure at any step removes the staging directory and records nothing.

use std::path::{Path, PathBuf};

use crate::install::{
    record_into, today, verify_install, verify_tokenizer, InstallTarget, Installed,
};
use crate::library::LocationEnv;
use crate::store::{directory_bytes, InstalledModel, ModelModality, Store, StoreError};

/// The two local Inkling sources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Inkling2LocalSource {
    /// A LookAtMyQuant packed release directory.
    Lamq(PathBuf),
    /// An Inkling 2 GGUF file (`inkling2_loop` or `inkling2`).
    Gguf(PathBuf),
}

impl Inkling2LocalSource {
    pub fn path(&self) -> &Path {
        match self {
            Self::Lamq(p) | Self::Gguf(p) => p,
        }
    }

    /// `lamq` or `gguf`, for messages and the receipt.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Lamq(_) => "lamq",
            Self::Gguf(_) => "gguf",
        }
    }
}

/// Which Inkling intake `path` needs, or None when it is neither a
/// LookAtMyQuant Inkling release directory nor an Inkling 2 GGUF. Reads only
/// `manifest.json` or the GGUF header.
pub fn detect_inkling2_local(path: &Path) -> Result<Option<Inkling2LocalSource>, String> {
    let meta = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if meta.is_dir() {
        let manifest = path.join("manifest.json");
        let Ok(text) = std::fs::read_to_string(&manifest) else {
            return Ok(None);
        };
        // A `.gturbo` install also has a manifest.json; only the LAMQ
        // release's top-level `format` string selects this intake.
        let format = serde_json::from_str::<serde_json::Value>(&text)
            .ok()
            .and_then(|v| v.get("format").and_then(|f| f.as_str()).map(str::to_string));
        return Ok((format.as_deref() == Some(repack::LAMQ_INKLING2_FORMAT))
            .then(|| Inkling2LocalSource::Lamq(path.to_path_buf())));
    }
    if path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
    {
        let arch = repack::inkling2_gguf_architecture(path).map_err(|e| e.0)?;
        return Ok(arch.map(|_| Inkling2LocalSource::Gguf(path.to_path_buf())));
    }
    Ok(None)
}

/// Bytes a source occupies (a file, or every file in a directory).
fn source_bytes(source: &Inkling2LocalSource) -> u64 {
    match source {
        Inkling2LocalSource::Gguf(p) => std::fs::metadata(p).map_or(0, |m| m.len()),
        Inkling2LocalSource::Lamq(p) => directory_bytes(p),
    }
}

/// Free space an install needs beside its source: the written install (the
/// LAMQ expert padding makes it about 1.3x the release; a GGUF install is
/// about the file's size) plus a margin.
pub fn inkling2_local_required_bytes(source: &Inkling2LocalSource) -> u64 {
    let src = source_bytes(source);
    src.saturating_mul(3) / 2 + (1 << 30)
}

/// A sibling of `dest`, so the final rename stays on one filesystem.
fn staging_dir(dest: &Path) -> Result<PathBuf, String> {
    let name = dest
        .file_name()
        .ok_or_else(|| format!("{} has no directory name", dest.display()))?;
    let parent = dest.parent().filter(|p| !p.as_os_str().is_empty());
    let mut staging = std::ffi::OsString::from(".");
    staging.push(name);
    staging.push(".partial");
    Ok(parent.unwrap_or_else(|| Path::new(".")).join(staging))
}

fn refuse_existing(dest: &Path) -> Result<(), String> {
    if std::fs::symlink_metadata(dest).is_ok() {
        return Err(format!(
            "{} already exists; this never overwrites. Remove it or pick another alias",
            dest.display()
        ));
    }
    Ok(())
}

/// Writes, verifies and records an install of `source` at `target`.
/// `tokenizer_from` is required for a GGUF (a directory holding at least
/// `tokenizer.json`, such as an HF install or a download of the base repo)
/// and ignored for a LAMQ directory, which ships its own tokenizer files.
/// `alias` is the installed name; `repo` and `revision` are what the
/// receipt records as the origin.
#[allow(clippy::too_many_arguments)]
pub fn install_inkling2_local(
    source: &Inkling2LocalSource,
    alias: &str,
    target: &InstallTarget,
    tokenizer_from: Option<&Path>,
    repo: &str,
    revision: &str,
    available_bytes: u64,
    mut progress: impl FnMut(&str),
    store: &Store,
    env: &LocationEnv,
) -> Result<Installed, StoreError> {
    let failed = |e: String| StoreError::Io(e);
    let dest = &target.dir;
    refuse_existing(dest).map_err(failed)?;
    let need = inkling2_local_required_bytes(source);
    if available_bytes < need {
        return Err(failed(format!(
            "installing {} needs about {} free; {} is available where {} will be created",
            source.path().display(),
            crate::install::human_bytes(need),
            crate::install::human_bytes(available_bytes),
            dest.display()
        )));
    }
    let tokenizer_dir = match source {
        Inkling2LocalSource::Lamq(dir) => dir.clone(),
        Inkling2LocalSource::Gguf(_) => tokenizer_from
            .ok_or_else(|| {
                failed(
                    "a GGUF carries no tokenizer.json: name a directory holding the Inkling 2 \
                     tokenizer files (tokenizer.json, chat_template.jinja), such as an HF \
                     install or a download of thinkingmachines/Inkling-2-Mini"
                        .to_string(),
                )
            })?
            .to_path_buf(),
    };
    if !tokenizer_dir.join("tokenizer.json").is_file() {
        return Err(failed(format!(
            "{} has no tokenizer.json",
            tokenizer_dir.display()
        )));
    }
    let staging = staging_dir(dest).map_err(failed)?;
    if std::fs::symlink_metadata(&staging).is_ok() {
        return Err(failed(format!(
            "{} is left over from an interrupted install; remove it and run again",
            staging.display()
        )));
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| failed(format!("creating {}: {e}", parent.display())))?;
    }
    std::fs::create_dir_all(&staging)
        .map_err(|e| failed(format!("creating {}: {e}", staging.display())))?;

    let model_id = format!("{repo}@{revision}");
    let built = (|| -> Result<(), String> {
        match source {
            Inkling2LocalSource::Lamq(dir) => {
                let r =
                    repack::write_inkling2_lamq_install(dir, &staging, &model_id, |l| progress(l))
                        .map_err(|e| e.0)?;
                progress(&format!(
                    "wrote {} resident tensors, experts {:.2} GB",
                    r.resident_tensors,
                    r.expert_bytes_written as f64 / 1e9
                ));
            }
            Inkling2LocalSource::Gguf(file) => {
                let r =
                    repack::write_inkling2_gguf_install(file, &staging, &model_id, |l| progress(l))
                        .map_err(|e| e.0)?;
                progress(&format!(
                    "wrote {} resident tensors from a {} GGUF",
                    r.resident_tensors, r.arch
                ));
            }
        }
        repack::copy_inkling2_tokenizer_files(&tokenizer_dir, &staging).map_err(|e| e.0)?;
        verify_tokenizer(&staging, &mut progress)?;
        verify_install(&staging, &model_io::inkling2_mini(), &mut progress)?;
        refuse_existing(dest)?;
        std::fs::rename(&staging, dest).map_err(|e| format!("moving the install into place: {e}"))
    })();
    if let Err(e) = built {
        // The staging directory is this run's own; nothing else writes it.
        let _ = std::fs::remove_dir_all(&staging);
        return Err(store.install_failure(&target.root, &target.root, &target.library_name, e));
    }
    let model = InstalledModel {
        alias: alias.to_string(),
        repo: repo.to_string(),
        revision: revision.to_string(),
        path: dest.canonicalize().unwrap_or_else(|_| dest.clone()),
        family: model_io::ModelFamily::Inkling2Loop.as_str().to_string(),
        install_bytes: directory_bytes(dest),
        installed_on: today(),
        status: "unlisted".to_string(),
        kind: None,
        modality: ModelModality::Text,
        variant: Some(source.kind().to_string()),
        library: None,
        relative_path: None,
    };
    let model = record_into(store, &model, target, env)?;
    Ok(Installed {
        model,
        arch: model_io::inkling2_mini(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("turbospark-ink-local-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn detects_a_lamq_release_by_its_manifest_format_only() {
        let d = tmp("lamq");
        std::fs::write(
            d.join("manifest.json"),
            r#"{"format": "inkling2-packed-v1-sqyfr"}"#,
        )
        .unwrap();
        assert_eq!(
            detect_inkling2_local(&d).unwrap(),
            Some(Inkling2LocalSource::Lamq(d.clone()))
        );
        // A .gturbo install's manifest is not a LAMQ release.
        std::fs::write(
            d.join("manifest.json"),
            r#"{"magic": "GTURBO", "arch": {}}"#,
        )
        .unwrap();
        assert_eq!(detect_inkling2_local(&d).unwrap(), None);
        let empty = tmp("empty");
        assert_eq!(detect_inkling2_local(&empty).unwrap(), None);
    }

    #[test]
    fn a_non_inkling_gguf_is_not_detected() {
        let d = tmp("gguf");
        let p = d.join("x.gguf");
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        let key = b"general.architecture";
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key);
        b.extend_from_slice(&8u32.to_le_bytes());
        for (arch, want) in [
            ("llama", false),
            ("inkling2_loop", true),
            ("inkling2", true),
        ] {
            let mut f = b.clone();
            f.extend_from_slice(&(arch.len() as u64).to_le_bytes());
            f.extend_from_slice(arch.as_bytes());
            std::fs::write(&p, &f).unwrap();
            assert_eq!(detect_inkling2_local(&p).unwrap().is_some(), want, "{arch}");
        }
        assert!(detect_inkling2_local(&d.join("missing.gguf")).is_err());
    }

    #[test]
    fn staging_is_a_hidden_sibling() {
        let s = staging_dir(Path::new("/a/b/ink.gturbo")).unwrap();
        assert_eq!(s, Path::new("/a/b/.ink.gturbo.partial"));
    }
}
