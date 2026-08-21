//! What should this machine run?
//!
//! The catalog says what has been run and [`crate::probe`] says what COULD
//! be, and until now neither asked how much memory the machine in front of
//! you has. This is that question: rank the curated table, and optionally
//! Hugging Face at large, by whether it fits here and by how much is known
//! about it.
//!
//! Three modules, in the order a recommendation is built:
//!
//! 1. [`fit`] -- the arithmetic. Does it fit, how much context does it
//!    afford, and which of `counted` and `mapped` is the binding term.
//! 2. [`rank`] -- the ordering. Fits, then evidence, then measured rate.
//! 3. [`discover`] -- the network arm, which turns popular Hugging Face
//!    repositories into candidates by putting each through the existing probe.
//!
//! **THE DEFAULT ARM TOUCHES NO NETWORK, and what it can answer is bounded by
//! that.** A curated row carries `install_bytes` and, for the eight that have
//! been through an oracle, a measured peak -- so those eight get an exact
//! answer instantly. The other six get their size and their evidence tier and
//! an honest `unknown` for the rest, because the slot cache and the KV are
//! functions of the checkpoint's SHAPE and nothing offline knows it.
//!
//! It would be easy to fill that hole with `known_architecture(family)`, and
//! that is a trap worth naming since the field is right there: one
//! architecture string covers several checkpoints of different shapes.
//! `llama` alone covers Mixtral 8x7B, Mistral 7B and TinyLlama 1.1B, whose
//! head dimensions and layer counts differ, and reading a baseline as though
//! it were a checkpoint's own shape is what shipped a `head_dim` of 128 to a
//! model with 64 (AGENTS.md Gotcha 39). `--probe` reads the real header.

mod discover;
mod fit;
mod rank;

pub use discover::{discover, DiscoverOptions};
pub use fit::{fit, CountedSource, Fit, FitVerdict, Shape};
pub use rank::{name_params_hint, rank, Evidence, Key};

use crate::entry::{CatalogEntry, Measured};
use crate::probe::ProbeReport;

/// The machine a recommendation is for.
///
/// Every field is a parameter and nothing here probes anything, which is what
/// keeps the whole module testable off macOS -- the same discipline
/// `model_io::ExpertCacheSlots::resolve` follows and for the same reason.
/// `crates/cli` fills it in from `runtime::physical_memory` and
/// `runtime::recommended_max_working_set`.
#[derive(Debug, Clone, Default)]
pub struct Machine {
    /// Installed physical memory. **The pool everything budgets from**,
    /// because `expert_cache_policy` and `context_policy` budget from it and
    /// a recommendation that disagreed with the `open()` it is recommending
    /// would be worse than none.
    pub physical_bytes: u64,
    /// What the Metal device says it will hold. Advisory: reported so the
    /// gap is visible, never budgeted from. `None` off macOS.
    pub working_set_bytes: Option<u64>,
    /// The chip's brand string, used to find the matching [`Measured`] row.
    /// Empty means no measured row can match, which is correct rather than
    /// unfortunate: a peak measured on an M4 Max says nothing about an M2.
    pub chip: String,
}

impl Machine {
    /// True when `counted` bytes would exceed what the Metal device says it
    /// will hold.
    ///
    /// **NOT `working_set < budget`**, which was the first shape of this and
    /// is useless: on a unified-memory Mac the recommended working set is
    /// about 75% of installed RAM, so it sits below `physical - 4 GiB` on
    /// every machine over 16 GB and the flag would fire always. What is worth
    /// saying is the rarer thing -- that this particular candidate's
    /// allocations are what cross the line, which is where the driver rather
    /// than the arithmetic decides the outcome.
    pub fn exceeds_working_set(&self, counted: u64) -> bool {
        self.working_set_bytes.is_some_and(|ws| counted > ws)
    }
}

/// Where a candidate came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Origin {
    /// A curated row, by alias.
    Catalog(String),
    /// A Hugging Face repository, with the weights file discovery picked.
    Discovered { repo: String, file: Option<String> },
}

impl Origin {
    /// What to type to install it.
    pub fn install_target(&self) -> String {
        match self {
            Self::Catalog(alias) => alias.clone(),
            Self::Discovered { repo, file } => match file {
                Some(f) => format!("--repo {repo} --file {f}"),
                None => format!("--repo {repo}"),
            },
        }
    }
}

/// One ranked candidate.
#[derive(Debug, Clone)]
pub struct Recommendation {
    pub origin: Origin,
    /// A human-readable name including the quantization, since two rows of
    /// one model differ only there.
    pub name: String,
    pub family: Option<String>,
    pub fit: Fit,
    pub evidence: Evidence,
    /// The measured row this machine's chip matched, if any.
    pub measured: Option<Measured>,
    /// True when the artifact is far smaller than its own name claims.
    pub suspicious: bool,
    /// Advisory lines: the probe's warnings, plus anything the fit turned up.
    pub notes: Vec<String>,
}

impl Recommendation {
    fn key(&self) -> Key {
        Key {
            runs: self.fit.verdict.runs(),
            evidence: self.evidence,
            tok_s: self.measured.as_ref().map(|m| m.decode_tok_s_min),
            install_bytes: self.fit.mapped,
            suspicious: self.suspicious,
        }
    }
}

/// Sort a mixed set of candidates in place.
///
/// The public form of [`rank`], so a caller that has concatenated curated and
/// discovered rows orders them ONCE rather than ranking each source and
/// interleaving the results, which is not the same thing.
pub fn rank_recommendations(rows: &mut [Recommendation]) {
    rank(rows, |r| r.key());
}

/// Rank the curated table for `machine` at `context`, without a network.
///
/// A row with a measured peak for this chip gets that peak; a row without one
/// reports [`FitVerdict::Unknown`] and says so. See the module header for why
/// the family's baseline is not used to fill the gap.
pub fn recommend_catalog(
    entries: &[&CatalogEntry],
    machine: &Machine,
    context: u32,
) -> Vec<Recommendation> {
    let mut out: Vec<Recommendation> = entries
        .iter()
        .map(|entry| from_entry(entry, machine, context, None))
        .collect();
    rank(&mut out, |r| r.key());
    out
}

/// Probe a curated row's own repository, the way `pull` would reach it.
///
/// Exists so a caller does not rebuild the `(repo, revision, file, sidecar
/// repo)` tuple by hand from four `CatalogEntry` fields -- which is the
/// arrangement `sidecar_repo()` and `sidecar_revision()` already exist to
/// stop being re-derived, since a GGUF row's sidecars come from a DIFFERENT
/// repository and inheriting the weights' revision for them 404s.
pub fn probe_entry(client: &crate::Client, entry: &CatalogEntry) -> Result<ProbeReport, String> {
    let repo = crate::RepoRef::new(&entry.source.repo, &entry.source.revision);
    let sidecars = crate::RepoRef::new(entry.sidecar_repo(), entry.sidecar_revision());
    crate::probe::probe(client, &repo, entry.source.file.as_deref(), Some(&sidecars))
}

/// Build a candidate from a curated row, optionally with a probe's shape.
///
/// `probed` is what `--probe` supplies: the checkpoint's real `ArchConfig`
/// and expert stride, which turn every `Unknown` above into arithmetic.
pub fn from_entry(
    entry: &CatalogEntry,
    machine: &Machine,
    context: u32,
    probed: Option<&ProbeReport>,
) -> Recommendation {
    let measured = entry.measured_for(&machine.chip).cloned();
    let shape = Shape {
        install_bytes: entry.install_bytes,
        expert_stride: probed.and_then(|p| p.expert_stride),
        arch: probed.and_then(|p| p.arch.clone()),
        measured_counted: None,
    };
    let mut notes = probed.map(|p| p.warnings.clone()).unwrap_or_default();

    // Fitted first, WITHOUT the measurement, because whether the measurement
    // applies depends on the slot count this fit resolves.
    let mut fitted = fit(
        &shape,
        machine.physical_bytes,
        context,
        model_io::ExpertCacheSlots::Auto,
    );

    // **A MEASURED PEAK APPLIES AT ONE CONTEXT AND ONE SLOT COUNT, AND
    // NOWHERE ELSE.** Both terms it is made of move with those: KV is a pure
    // function of the window (on a dense install it is most of the peak,
    // Gotcha 40), and the slot cache is `slots x layers x expert_stride` (on
    // a streamed MoE it is the dominant term -- Gemma 4 reads 2,175 MiB at 16
    // slots and 3,654 at 32). Quoting a 4,096/16 number against an 8,192/32
    // request is not an approximation, it is a different measurement.
    //
    // **AN UNPROBED ROW RESOLVES 16 SLOTS BY IGNORANCE, NOT BY ARITHMETIC**,
    // and that distinction is what the `shape_known` term below is for. With
    // no `ArchConfig` there is no expert stride, so `Auto` divides by nothing
    // and falls back to its floor -- which happens to equal the protocol's
    // pinned 16 and would make every measured row look applicable. So an
    // unprobed row takes the measurement AT THE CONFIGURATION IT WAS TAKEN AT
    // and says so, rather than claiming to describe the one `open()` would
    // choose here.
    let shape_known = shape.arch.is_some();
    match &measured {
        Some(m)
            if m.context == context
                && (!shape_known || m.expert_cache_slots as usize == fitted.slots) =>
        {
            fitted.counted = m.peak_footprint_mib * 1024 * 1024;
            fitted.counted_source = CountedSource::Measured;
            fitted.verdict = fit::verdict_for_counted(&fitted);
            if !shape_known {
                fitted.slots = m.expert_cache_slots as usize;
                notes.push(format!(
                    "that peak holds at {} expert-cache slots; nothing here has read this \
                     checkpoint's header, so what `--expert-cache-slots auto` would pick \
                     on this machine is unknown (probe it)",
                    m.expert_cache_slots
                ));
            }
        }
        Some(m) => {
            let mut why = Vec::new();
            if m.context != context {
                why.push(format!("at {} context, not {context}", m.context));
            }
            if m.expert_cache_slots as usize != fitted.slots {
                why.push(format!(
                    "at {} expert-cache slots, and this machine resolves {}",
                    m.expert_cache_slots, fitted.slots
                ));
            }
            notes.push(format!(
                "its measured peak of {} MiB was taken {}, so it is reported rather than \
                 applied",
                m.peak_footprint_mib,
                why.join(" and ")
            ));
        }
        None => {}
    }
    if machine.exceeds_working_set(fitted.counted) {
        notes.push(
            "its allocations exceed what the Metal device says it will hold, so the \
             driver rather than this arithmetic is what decides the outcome"
                .to_string(),
        );
    }

    Recommendation {
        origin: Origin::Catalog(entry.alias.clone()),
        name: entry.name.clone(),
        family: Some(entry.family.clone()),
        fit: fitted,
        evidence: Evidence::of(entry.status),
        measured,
        suspicious: false,
        notes,
    }
}

#[cfg(test)]
mod tests;
