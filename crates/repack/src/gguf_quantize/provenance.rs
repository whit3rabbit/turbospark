//! Provenance for a quantization, and a check that its inputs are still the
//! files their Hugging Face repos currently serve.
//!
//! A quantized GGUF depends on three inputs: the float source, the reference
//! whose type map was followed (`--recipe-from`), and the importance matrix.
//! Publishers revise these (Unsloth republishes GGUFs and imatrix files), so a
//! file made last month may no longer match what a fresh download gives. We
//! record each input's SHA-256 and, if the caller names it, its Hugging Face
//! location, then compare against the hash the Hub reports for that file now.
//!
//! The Hub is asked only for a public repo listing (`/api/models/<repo>`); no
//! token and nothing about the local machine is sent.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// The ggml commit the encoders were ported from and golden-tested against.
/// `scripts/check_ggml_drift.py` compares upstream against this.
pub const GGML_PARITY_COMMIT: &str = "b1b770d90";

const SCHEMA: u32 = 1;

/// Where an input lives on Hugging Face.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HfFile {
    pub repo: String,
    pub path: String,
}

impl HfFile {
    /// Parse `owner/repo:path/in/repo`.
    pub fn parse(s: &str) -> Result<HfFile, String> {
        let (repo, path) = s
            .split_once(':')
            .ok_or_else(|| format!("expected owner/repo:file, got {s:?}"))?;
        let ok = |p: &str| {
            !p.is_empty()
                && p.chars()
                    .all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
        };
        let valid_repo = repo
            .split_once('/')
            .is_some_and(|(a, b)| ok(a) && ok(b) && !b.contains('/'));
        if !valid_repo || path.is_empty() || path.contains("..") || path.starts_with('/') {
            return Err(format!("not a valid Hugging Face file reference: {s:?}"));
        }
        Ok(HfFile {
            repo: repo.to_string(),
            path: path.to_string(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputRecord {
    /// `source`, `reference` or `imatrix`.
    pub role: String,
    pub file_name: String,
    pub size: u64,
    pub sha256: String,
    pub hf: Option<HfFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub schema: u32,
    pub tool: String,
    pub ggml_parity_commit: String,
    pub created_unix: u64,
    pub inputs: Vec<InputRecord>,
    pub output_file_name: String,
    pub output_size: u64,
    pub output_sha256: String,
}

fn hash(path: &Path) -> Result<String, String> {
    model_io::hash_file(path, 4 << 20).map_err(|e| format!("{}: {e}", path.display()))
}

fn record_input(role: &str, path: &Path, hf: Option<HfFile>) -> Result<InputRecord, String> {
    let size = std::fs::metadata(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .len();
    Ok(InputRecord {
        role: role.to_string(),
        file_name: path
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned()),
        size,
        sha256: hash(path)?,
        hf,
    })
}

impl Provenance {
    /// Hash the inputs and the finished output. `inputs` is `(role, path, hf)`.
    pub fn record(
        inputs: &[(&str, &Path, Option<HfFile>)],
        output: &Path,
    ) -> Result<Provenance, String> {
        let mut recs = Vec::new();
        for (role, path, hf) in inputs {
            recs.push(record_input(role, path, hf.clone())?);
        }
        let out = record_input("output", output, None)?;
        Ok(Provenance {
            schema: SCHEMA,
            tool: format!(
                "turbospark-repack {} gguf_quantize",
                env!("CARGO_PKG_VERSION")
            ),
            ggml_parity_commit: GGML_PARITY_COMMIT.to_string(),
            created_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            inputs: recs,
            output_file_name: out.file_name,
            output_size: out.size,
            output_sha256: out.sha256,
        })
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("provenance serializes")
    }

    pub fn from_json(s: &str) -> Result<Provenance, String> {
        let p: Provenance = serde_json::from_str(s).map_err(|e| e.to_string())?;
        if p.schema != SCHEMA {
            return Err(format!(
                "provenance schema {} is not supported (expected {SCHEMA})",
                p.schema
            ));
        }
        Ok(p)
    }
}

/// One file in a Hub repo listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoFile {
    pub path: String,
    pub size: u64,
    /// LFS SHA-256, when the Hub reports one (large and binary files).
    pub sha256: Option<String>,
}

/// A repo listing plus the commit it describes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoListing {
    pub commit: String,
    pub files: Vec<RepoFile>,
}

/// Source of repo listings. Injectable so the check is testable offline.
pub trait Hub {
    fn listing(&self, repo: &str) -> Result<RepoListing, String>;
}

/// The public Hugging Face Hub.
pub struct HfHub;

impl Hub for HfHub {
    fn listing(&self, repo: &str) -> Result<RepoListing, String> {
        // Validated again here: this string becomes part of a URL.
        HfFile::parse(&format!("{repo}:x"))?;
        let url = format!("https://huggingface.co/api/models/{repo}?blobs=true");
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("turbospark-gguf-quantize")
            .build()
            .map_err(|e| e.to_string())?;
        let resp = client.get(&url).send().map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("{repo}: Hub returned {}", resp.status()));
        }
        let body = resp.text().map_err(|e| e.to_string())?;
        let v: serde_json::Value = serde_json::from_str(&body).map_err(|e| e.to_string())?;
        let commit = v["sha"].as_str().unwrap_or_default().to_string();
        let files = v["siblings"]
            .as_array()
            .ok_or_else(|| format!("{repo}: listing has no files"))?
            .iter()
            .filter_map(|s| {
                Some(RepoFile {
                    path: s["rfilename"].as_str()?.to_string(),
                    size: s["size"].as_u64().unwrap_or(0),
                    sha256: s["lfs"]["sha256"].as_str().map(str::to_string),
                })
            })
            .collect();
        Ok(RepoListing { commit, files })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    /// The Hub's file has the recorded SHA-256.
    UpToDate,
    /// The Hub now serves different bytes under the same name.
    Changed { recorded: String, current: String },
    /// Same size, but the Hub reports no SHA-256, so only the size could be compared.
    SizeOnly { same: bool },
    /// The repo no longer lists the file.
    Missing,
    /// No Hugging Face location was recorded for this input.
    Untracked,
    /// The Hub could not be asked.
    Unreachable(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub role: String,
    pub file_name: String,
    pub status: Status,
    /// The repo's current commit, when it was reached.
    pub commit: Option<String>,
}

impl Finding {
    /// Whether this finding means the recipe/inputs should be revisited.
    pub fn needs_attention(&self) -> bool {
        match &self.status {
            Status::UpToDate | Status::Untracked => false,
            Status::SizeOnly { same } => !same,
            Status::Changed { .. } | Status::Missing | Status::Unreachable(_) => true,
        }
    }
}

/// Compare each recorded input with the Hub's current file.
pub fn check_freshness(p: &Provenance, hub: &dyn Hub) -> Vec<Finding> {
    let mut cache: std::collections::HashMap<String, Result<RepoListing, String>> =
        Default::default();
    p.inputs
        .iter()
        .map(|inp| {
            let mk = |status, commit| Finding {
                role: inp.role.clone(),
                file_name: inp.file_name.clone(),
                status,
                commit,
            };
            let Some(hf) = &inp.hf else {
                return mk(Status::Untracked, None);
            };
            let listing = cache
                .entry(hf.repo.clone())
                .or_insert_with(|| hub.listing(&hf.repo))
                .clone();
            let listing = match listing {
                Ok(l) => l,
                Err(e) => return mk(Status::Unreachable(e), None),
            };
            let commit = Some(listing.commit.clone());
            let Some(file) = listing.files.iter().find(|f| f.path == hf.path) else {
                return mk(Status::Missing, commit);
            };
            match &file.sha256 {
                Some(cur) if *cur == inp.sha256 => mk(Status::UpToDate, commit),
                Some(cur) => mk(
                    Status::Changed {
                        recorded: inp.sha256.clone(),
                        current: cur.clone(),
                    },
                    commit,
                ),
                None => mk(
                    Status::SizeOnly {
                        same: file.size == inp.size,
                    },
                    commit,
                ),
            }
        })
        .collect()
}

pub fn write_provenance(path: &Path, p: &Provenance) -> Result<(), String> {
    std::fs::write(path, p.to_json()).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn read_provenance(path: &Path) -> Result<Provenance, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Provenance::from_json(&s)
}
