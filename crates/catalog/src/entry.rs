//! One catalog row and its constituent types.
//!
//! **Every field here exists because it cannot be derived from the repo
//! name.** That is the admission rule for the struct as well as for the
//! table: a field that could be computed from another one is a field that
//! will eventually disagree with it. Three of them are worth naming, because
//! each was learned the expensive way and each looks redundant until it is
//! not:
//!
//! 1. [`Sidecars::repo`] is a DIFFERENT repository from [`Source::repo`] on
//!    every GGUF row. A GGUF carries its tokenizer as llama.cpp metadata and
//!    this port loads an HF `tokenizer.json`, so the sidecars come from the
//!    checkpoint the GGUF was converted from. Nothing about the weights repo
//!    says which one that is.
//! 2. [`Sidecars::files`] is per REPOSITORY, never per family.
//!    `prism-ml/Ternary-Bonsai-27B-mlx-2bit` ships `merges.txt` and no
//!    `generation_config.json`; `mlx-community/Qwen3.8-27B-4bit` is the exact
//!    inverse, and the two are one architecture. Copying this list between
//!    checkpoints of one family is what cost a 20-minute re-stream
//!    (AGENTS.md Gotcha 47).
//! 3. [`Source::revision`] is a commit sha where the publisher offers one and
//!    `main` where it does not, and the difference MATTERS: a row pinned at
//!    `main` floats, so its frozen gate numbers stop meaning anything the
//!    next time that file is re-uploaded. Every mlx-community and prism-ml
//!    row here is pinned; every GGUF row floats, exactly as the corresponding
//!    `*_network.rs` test does today. What catches a float in practice is
//!    [`CatalogEntry::download_bytes`], which
//!    `tests/catalog_network.rs` re-reads with a HEAD request -- so that
//!    field is a fingerprint and not just a progress-bar input. Do not
//!    "tidy" a floating row by inventing a sha for it.

use serde::{Deserialize, Serialize};

/// How a checkpoint is packaged upstream, which decides which repack walk
/// installs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceKind {
    /// mlx-community-style safetensors carrying MLX `affine` quantization,
    /// installed through `repack::write_gemma4_install_streamed` and its two
    /// family wrappers.
    Mlx,
    /// A single `.gguf` file, installed through
    /// `repack::write_gguf_install_streamed`.
    Gguf,
}

impl SourceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SourceKind::Mlx => "mlx",
            SourceKind::Gguf => "gguf",
        }
    }
}

/// How far a row has actually been taken on real hardware.
///
/// **This is evidence, not intent.** The tiers are ordered by what was
/// measured rather than by how well the model is expected to do: a row is
/// [`Status::Verified`] only if a frozen row exists for it in
/// `docs/BENCHMARKS.md`, which means some number about it is asserted by a
/// test that can go red.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    /// Has a frozen quality-gate and/or memory-oracle row in
    /// `docs/BENCHMARKS.md`.
    Verified,
    /// Installed and generated coherent text here, with no frozen row.
    Runs,
    /// Installs and runs, and something in [`CatalogEntry::notes`]
    /// disqualifies it for ordinary use. Kept BECAUSE the disqualification is
    /// invisible otherwise -- see the `mixtral` row, which is correct and
    /// wants 54.5 GiB of expert-slot cache.
    Caveat,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Verified => "verified",
            Status::Runs => "runs",
            Status::Caveat => "caveat",
        }
    }
}

/// Where the weights live.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Source {
    pub kind: SourceKind,
    /// `owner/name` on Hugging Face.
    pub repo: String,
    /// A commit sha. Never `main`; see the module header.
    pub revision: String,
    /// The `.gguf` filename. Required for [`SourceKind::Gguf`] and forbidden
    /// for [`SourceKind::Mlx`], which is asserted rather than assumed
    /// (`CatalogEntry::validate`).
    #[serde(default)]
    pub file: Option<String>,
}

/// Where the tokenizer sidecars live, and which ones exist there.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sidecars {
    /// `None` means "the same repository as the weights". Always `Some` on a
    /// GGUF row; see the module header.
    #[serde(default)]
    pub repo: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    /// The files to fetch, exactly as they are named in that repository.
    pub files: Vec<String>,
}

/// One curated model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogEntry {
    /// The short name a user types. Unique across the table.
    pub alias: String,
    /// A human-readable name including the quantization, since two rows of
    /// one model differ only there.
    pub name: String,
    /// The `.gturbo` manifest family string (`ModelFamily::as_str`), used to
    /// display the row and to cross-check the probe.
    pub family: String,
    pub source: Source,
    pub sidecars: Sidecars,
    /// Bytes read off the network during a pull. For a GGUF row this is the
    /// published file size; for an MLX row the sum of the shards.
    pub download_bytes: u64,
    /// Approximate bytes the install occupies on disk. Used for the
    /// free-space check, which is why it is allowed to be approximate in the
    /// generous direction only.
    pub install_bytes: u64,
    pub status: Status,
    /// The gate targets that assert something about this exact artifact, by
    /// test name, so a reader can run them.
    #[serde(default)]
    pub gates: Vec<String>,
    #[serde(default)]
    pub notes: Option<String>,
}

impl CatalogEntry {
    /// The repository the sidecars come from, resolving the "same as the
    /// weights" default.
    pub fn sidecar_repo(&self) -> &str {
        self.sidecars.repo.as_deref().unwrap_or(&self.source.repo)
    }

    /// The revision the sidecars are pinned at, resolving the same default.
    ///
    /// Note it falls back to the WEIGHTS revision only when the repo also
    /// fell back. A sidecar repo named without a revision would otherwise
    /// inherit a commit sha from a different repository, which resolves to a
    /// 404 rather than to anything wrong -- but a 404 blamed on the wrong
    /// repo is a bad half-hour.
    pub fn sidecar_revision(&self) -> &str {
        match (&self.sidecars.repo, &self.sidecars.revision) {
            (_, Some(rev)) => rev,
            (None, None) => &self.source.revision,
            (Some(_), None) => "main",
        }
    }

    /// Structural checks that hold for every row, applied at load so a
    /// malformed user override fails at the point of reading rather than at
    /// the point of streaming.
    pub fn validate(&self) -> Result<(), String> {
        if self.alias.is_empty() {
            return Err("alias is empty".to_string());
        }
        if self.source.repo.split('/').count() != 2 {
            return Err(format!(
                "{}: repo {:?} is not owner/name",
                self.alias, self.source.repo
            ));
        }
        match (self.source.kind, &self.source.file) {
            (SourceKind::Gguf, None) => {
                return Err(format!("{}: a gguf row needs source.file", self.alias))
            }
            (SourceKind::Mlx, Some(f)) => {
                return Err(format!(
                    "{}: an mlx row must not name a file, got {f:?}",
                    self.alias
                ))
            }
            _ => {}
        }
        // A GGUF's tokenizer cannot come from the GGUF, so a row that leaves
        // this defaulted is claiming something impossible.
        if self.source.kind == SourceKind::Gguf && self.sidecars.repo.is_none() {
            return Err(format!(
                "{}: a gguf row needs sidecars.repo (a GGUF carries llama.cpp's \
                 tokenizer, not an HF tokenizer.json)",
                self.alias
            ));
        }
        if !self.sidecars.files.iter().any(|f| f == "tokenizer.json") {
            return Err(format!(
                "{}: sidecars.files must include tokenizer.json",
                self.alias
            ));
        }
        Ok(())
    }
}
