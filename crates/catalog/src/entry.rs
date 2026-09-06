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
    /// String identifier for this packaging format ("mlx" or "gguf").
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
    /// String representation of the verification status ("verified", "runs", or "caveat").
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

/// A separate repository supplying a multi-token-prediction head that
/// [`Source`]'s own conversion drops (`docs/MTP_SPECULATIVE.md` step 1;
/// `crates/repack/src/gemma4_checkpoint/mtp.rs`). The walk merges its
/// shard(s) into the same multi-shard registry the trunk uses, so the result
/// is one install with one resident index -- there is no second artifact and
/// no manifest flag, the same way the trunk's own quantization is a property
/// of the bytes rather than of a field.
///
/// `None` on every row whose conversion already carries a head, or whose
/// checkpoint has never published one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MtpSource {
    /// `owner/name` on Hugging Face. A DIFFERENT repository from
    /// [`Source::repo`] by construction: [`Source::repo`]'s own conversion is
    /// what dropped the head, so pointing this at it would find nothing.
    pub repo: String,
    /// A commit sha; see [`Source::revision`]'s note on floating pins.
    pub revision: String,
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

/// What this artifact actually did on one machine.
///
/// **These are OBSERVATIONS, and the ceilings and floors that guard them stay
/// in the oracle targets.** The two are different kinds of statement: a
/// `ChipBaseline` is an assertion with a per-row margin and a paragraph of
/// provenance explaining that margin, and JSON has nowhere to put the
/// paragraph. What lives here is the evidence those margins were chosen
/// against, so a recommendation can quote a number instead of estimating one.
/// `oracle_common::assert_agrees_with_catalog` ties the two together and
/// reddens in `cargo test --workspace` -- no install, no GPU -- if either
/// side moves without the other.
///
/// **THE CONVENTION IS WORST-OBSERVED, and it is not decoration.** A protocol
/// run produces three cases and usually several readings of each; what goes
/// in here is the SLOWEST reading of the slowest case, the FASTEST reading of
/// the fastest case, and the HIGHEST peak. So the pair brackets what a user
/// should expect rather than advertising a best case, and
/// [`Self::decode_tok_s_min`] is comparable with the oracle's floor by
/// construction. Pasting a favourable number in here would silently loosen
/// that cross-check.
///
/// Note the two tok/s fields are NOT "short case" and "long case".
/// `muse_glimmer` decodes its short case SLOWEST of the three (13.291 against
/// 15.341 and 14.483), so a schema keyed on case names would have to encode
/// which case is which; a min and a max do not care.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Measured {
    /// The chip's brand string, as `sysctl machdep.cpu.brand_string` reports
    /// it and as `bench::memory::chip_brand_string` reads it. The oracle
    /// matches its own `brand_substr` against this by substring, so
    /// "Apple M4 Max" here is found by a row spelled "Apple M4 Max" and by
    /// one spelled "Apple M4".
    pub chip: String,
    /// The context window the run opened. **Every footprint is a footprint at
    /// one window** and on a dense install the window is most of what is
    /// being reported (AGENTS.md Gotcha 40), so a row without this says
    /// nothing.
    pub context: u32,
    /// Expert-cache slots the run pinned. Inert on a dense install, and the
    /// dominant term on a streamed MoE one (Gotcha 36).
    pub expert_cache_slots: u32,
    /// Highest whole-session peak `phys_footprint` observed, in MiB.
    pub peak_footprint_mib: u64,
    /// Slowest reading of the slowest protocol case, in tokens per second.
    pub decode_tok_s_min: f64,
    /// Fastest reading of the fastest protocol case.
    pub decode_tok_s_max: f64,
    /// ISO date of the session these came from. Cross-session absolutes here
    /// have repeatedly failed to reproduce (AGENTS.md Gotcha 22), so a row
    /// that cannot be dated cannot be compared with another one.
    pub measured_on: String,
    /// Machine state and provenance, in the same spirit as the oracle's own
    /// `source` field: power source, and whether anything else was running.
    pub source: String,
}

/// What kind of row this is: an installable trunk model, or a vision-tower
/// sidecar with no text half of its own (vision memory sidecar, part A5).
///
/// **Defaults to [`EntryKind::Model`] on an absent `kind` key**, which is
/// AGENTS.md Gotcha 39's rule applied here: a default is a claim about what
/// silence means, and every row written before this field existed IS a
/// trunk model -- the thing a user types `pull <alias>` for and points
/// `--model` at. A tower row is never that: it has no tokenizer, cannot be
/// opened as a session on its own, and is not a fit candidate (a tower is
/// not a thing you "run" standalone).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum EntryKind {
    #[default]
    Model,
    VisionTower,
}

impl EntryKind {
    /// String identifier for this row kind ("model" or "vision-tower"),
    /// matching the JSON spelling exactly.
    pub fn as_str(self) -> &'static str {
        match self {
            EntryKind::Model => "model",
            EntryKind::VisionTower => "vision-tower",
        }
    }
}

/// One curated model.
///
/// Not `Eq`: [`Measured`] carries `f64` throughput readings. Nothing keys a
/// map on a whole entry, so the bound was never load-bearing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogEntry {
    /// The short name a user types. Unique across the table.
    pub alias: String,
    /// What kind of row this is. See [`EntryKind`].
    #[serde(default)]
    pub kind: EntryKind,
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
    /// What this artifact measured, one row per chip. Empty for a row nobody
    /// has taken through an oracle, which is a fact about the evidence rather
    /// than about the model -- see [`Measured`].
    #[serde(default)]
    pub measured: Vec<Measured>,
    #[serde(default)]
    pub notes: Option<String>,
    /// A separate repository to pull this checkpoint's multi-token-prediction
    /// head from, when `source`'s own conversion carries none. See
    /// [`MtpSource`].
    #[serde(default)]
    pub mtp: Option<MtpSource>,
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

    /// The measured row for a chip, matched the way the oracle matches its
    /// own baselines: by substring, most specific first, so "Apple M4 Max"
    /// finds a row recorded as "Apple M4 Max" and a hypothetical bare
    /// "Apple M4" row finds it too.
    ///
    /// Takes the LONGEST matching chip string rather than the first, because
    /// unlike the oracle's hand-ordered table this vector's order is whatever
    /// the JSON happened to list. Ordering that mattered but was invisible in
    /// the file is a trap this does not need to inherit.
    pub fn measured_for(&self, brand: &str) -> Option<&Measured> {
        self.measured
            .iter()
            .filter(|m| brand.contains(&m.chip))
            .max_by_key(|m| m.chip.len())
    }

    /// The best measurement available for this row REGARDLESS of chip, and
    /// the chip it was taken on.
    ///
    /// **FOR THROUGHPUT ONLY, AND ONLY TO REPORT.** A footprint transfers
    /// across silicon roughly at best (slot warming and memory layout vary)
    /// and a decode rate does not transfer at all, so this must never reach
    /// `Fit::counted` and must never be shown without naming its chip. It
    /// exists because every frozen row in this table was taken on one
    /// machine, so `measured_for` answers `None` for every other Mac and a
    /// user there is left with no throughput signal at all. Reporting a
    /// measurement at its own stated configuration is honest; presenting it
    /// as this machine's answer is not.
    ///
    /// Prefers a row for this chip when there is one, so the caller's
    /// `this_machine` flag is true whenever it can be.
    pub fn any_measured(&self, brand: &str) -> Option<(&Measured, bool)> {
        if let Some(m) = self.measured_for(brand) {
            return Some((m, true));
        }
        self.measured.first().map(|m| (m, false))
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
        // A tower has no tokenizer of its own, so it takes the OPPOSITE
        // requirement: `preprocessor_config.json` (what `vision_dir()` reads
        // for preprocessing settings, Part A3) rather than `tokenizer.json`.
        match self.kind {
            EntryKind::Model => {
                if !self.sidecars.files.iter().any(|f| f == "tokenizer.json") {
                    return Err(format!(
                        "{}: sidecars.files must include tokenizer.json",
                        self.alias
                    ));
                }
            }
            EntryKind::VisionTower => {
                if !self
                    .sidecars
                    .files
                    .iter()
                    .any(|f| f == "preprocessor_config.json")
                {
                    return Err(format!(
                        "{}: a vision-tower row needs sidecars.files to include \
                         preprocessor_config.json (a tower has no tokenizer, so it does \
                         not need tokenizer.json)",
                        self.alias
                    ));
                }
            }
        }
        if let Some(mtp) = &self.mtp {
            if mtp.repo.split('/').count() != 2 {
                return Err(format!(
                    "{}: mtp.repo {:?} is not owner/name",
                    self.alias, mtp.repo
                ));
            }
            if mtp.repo == self.source.repo {
                return Err(format!(
                    "{}: mtp.repo is the same repository as source.repo; a checkpoint \
                     whose own conversion carries a head needs no separate mtp source",
                    self.alias
                ));
            }
        }
        for m in &self.measured {
            m.validate(&self.alias)?;
        }
        Ok(())
    }
}

impl Measured {
    /// The same load-time discipline the rest of the row gets. Every check
    /// here is one that would otherwise surface as a wrong RECOMMENDATION
    /// rather than as an error: a zero context makes a footprint
    /// uninterpretable, a slot count outside the allowed set describes a run
    /// this engine cannot reproduce, and a min above a max means somebody
    /// filled the two fields in the order they appear in a bench footer
    /// rather than by the worst-observed convention.
    pub fn validate(&self, alias: &str) -> Result<(), String> {
        if self.chip.trim().is_empty() {
            return Err(format!("{alias}: a measured row needs a chip"));
        }
        if self.context == 0 {
            return Err(format!(
                "{alias}: measured row for {:?} has no context; every footprint \
                 is a footprint at one window",
                self.chip
            ));
        }
        if !foundation::runtime_config::ALLOWED_CACHE_SLOTS.contains(&self.expert_cache_slots) {
            return Err(format!(
                "{alias}: measured row for {:?} pins {} expert-cache slots, \
                 outside the allowed {:?}",
                self.chip,
                self.expert_cache_slots,
                foundation::runtime_config::ALLOWED_CACHE_SLOTS
            ));
        }
        if self.decode_tok_s_min <= 0.0
            || self.decode_tok_s_max <= 0.0
            || self.decode_tok_s_min.is_nan()
            || self.decode_tok_s_max.is_nan()
        {
            return Err(format!(
                "{alias}: measured row for {:?} has a non-positive decode rate",
                self.chip
            ));
        }
        if self.decode_tok_s_min > self.decode_tok_s_max {
            return Err(format!(
                "{alias}: measured row for {:?} has min {} above max {}",
                self.chip, self.decode_tok_s_min, self.decode_tok_s_max
            ));
        }
        if self.peak_footprint_mib == 0 {
            return Err(format!(
                "{alias}: measured row for {:?} has a zero peak footprint",
                self.chip
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod entry_kind_tests {
    use super::*;

    fn model_entry(kind: EntryKind, sidecar_files: &[&str]) -> CatalogEntry {
        CatalogEntry {
            alias: "x".to_string(),
            kind,
            name: "X".to_string(),
            family: "qwen35".to_string(),
            source: Source {
                kind: SourceKind::Mlx,
                repo: "owner/name".to_string(),
                revision: "main".to_string(),
                file: None,
            },
            sidecars: Sidecars {
                repo: None,
                revision: None,
                files: sidecar_files.iter().map(|s| s.to_string()).collect(),
            },
            download_bytes: 1,
            install_bytes: 1,
            status: Status::Runs,
            gates: Vec::new(),
            measured: Vec::new(),
            notes: None,
            mtp: None,
        }
    }

    /// An entry with no `kind` key deserializes as `Model`, and the field
    /// round-trips through the exact JSON spelling `models.json` uses.
    #[test]
    fn entry_kind_defaults_to_model_when_the_key_is_absent() {
        let json = r#"{"repo":"owner/name"}"#;
        #[derive(Deserialize)]
        struct Probe {
            #[serde(default)]
            kind: EntryKind,
        }
        let probe: Probe = serde_json::from_str(json).unwrap();
        assert_eq!(probe.kind, EntryKind::Model);
    }

    #[test]
    fn entry_kind_round_trips_through_its_kebab_case_spelling() {
        assert_eq!(
            serde_json::to_string(&EntryKind::Model).unwrap(),
            "\"model\""
        );
        assert_eq!(
            serde_json::to_string(&EntryKind::VisionTower).unwrap(),
            "\"vision-tower\""
        );
        assert_eq!(
            serde_json::from_str::<EntryKind>("\"vision-tower\"").unwrap(),
            EntryKind::VisionTower
        );
        assert_eq!(EntryKind::VisionTower.as_str(), "vision-tower");
        assert_eq!(EntryKind::Model.as_str(), "model");
    }

    #[test]
    fn a_model_row_requires_tokenizer_json_and_not_preprocessor_config() {
        let ok = model_entry(EntryKind::Model, &["tokenizer.json"]);
        assert!(ok.validate().is_ok(), "{:?}", ok.validate());

        let missing = model_entry(EntryKind::Model, &["preprocessor_config.json"]);
        let err = missing.validate().unwrap_err();
        assert!(err.contains("tokenizer.json"), "{err}");
    }

    #[test]
    fn a_vision_tower_row_requires_preprocessor_config_and_not_tokenizer_json() {
        let ok = model_entry(EntryKind::VisionTower, &["preprocessor_config.json"]);
        assert!(ok.validate().is_ok(), "{:?}", ok.validate());

        // Missing preprocessor_config.json is refused, even with a
        // tokenizer.json present -- a tower row is judged by its own
        // requirement, not by the model row's.
        let missing = model_entry(EntryKind::VisionTower, &["tokenizer.json"]);
        let err = missing.validate().unwrap_err();
        assert!(err.contains("preprocessor_config.json"), "{err}");
    }
}
