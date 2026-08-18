//! What to suggest first, once [`super::fit`] has said what fits.
//!
//! ## Provenance
//!
//! Adapted from `src/discover.rs` of
//! <https://github.com/notactuallytreyanastasio/shoehorn> at commit
//! `17af0207e8b8`, MIT licensed (see `NOTICE`). What carried over is the
//! SHAPE: tier a candidate, sort by tier and then by size, and flag a
//! repository whose weights are far smaller than its name claims.
//! [`name_params_hint`] is the closest thing to a direct port here.
//!
//! What did not carry over is the tiering QUANTITY. Upstream tiers on the
//! bits-per-weight a budget affords, which is the right axis for a tool that
//! re-quantizes; this port installs published artifacts, so its tiers are the
//! fit verdict and the strength of the evidence behind the row.
//!
//! **EVIDENCE OUTRANKS EVERYTHING EXCEPT FITTING**, and that is this file's
//! one real opinion. A discovered repository can be larger, newer and more
//! downloaded than a curated row and it still sorts below one, because
//! nothing has run it here. The catalog's own `status` field already carries
//! that judgement -- it is evidence and not intent, and a row is `Verified`
//! only when some number about it is asserted by a test that can go red.

use crate::entry::Status;

/// How much is known about a candidate, worst to best.
///
/// Ordered so `derive(PartialOrd)` sorts them, which is why the variants are
/// written in that order rather than in the order a reader would list them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Evidence {
    /// Found on Hugging Face and probed. Nothing here has run it.
    Discovered,
    /// A curated row that installs and runs, with something in its notes that
    /// disqualifies it for ordinary use.
    Caveat,
    /// A curated row that installed and generated coherent text here, with no
    /// frozen row.
    Runs,
    /// A curated row with a frozen quality-gate or memory-oracle row.
    Verified,
}

impl Evidence {
    pub fn of(status: Status) -> Self {
        match status {
            Status::Verified => Self::Verified,
            Status::Runs => Self::Runs,
            Status::Caveat => Self::Caveat,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::Runs => "runs",
            Self::Caveat => "caveat",
            Self::Discovered => "unmeasured",
        }
    }
}

/// Sort candidates into the order a reader should consider them.
///
/// The key, in order: it FITS, then how much is KNOWN about it, then how fast
/// it was MEASURED to decode, then how big it is. The last is shoehorn's own
/// tie-break ("the biggest model that still fits well makes the best
/// suggestion") and is the only one of the four that is a heuristic rather
/// than a fact.
///
/// **Throughput is only ever quoted, never estimated.** Decode rate does not
/// track weight bytes on this engine -- one architecture reads 18.3 / 14.2 /
/// 19.0 tok/s at 1 / 2 / 4 bits (AGENTS.md Gotcha 47) -- so a candidate with
/// no measured row sorts as though it had no rate at all rather than being
/// given a guess derived from its size.
pub fn rank<T>(items: &mut [T], key: impl Fn(&T) -> Key) {
    items.sort_by(|a, b| key(b).cmp_to(&key(a)));
}

/// The sort key, extracted so `rank` can order anything that can produce one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Key {
    pub runs: bool,
    pub evidence: Evidence,
    /// Measured decode floor in tokens per second, when a row exists for this
    /// machine's chip.
    pub tok_s: Option<f64>,
    pub install_bytes: u64,
    /// Set when the artifact is far smaller than its own name claims. Sorts
    /// below everything that fits, however good its other columns look.
    pub suspicious: bool,
}

impl Key {
    /// Ascending "better", so `rank` reverses it.
    fn cmp_to(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        self.runs
            .cmp(&other.runs)
            .then(other.suspicious.cmp(&self.suspicious))
            .then(self.evidence.cmp(&other.evidence))
            .then(
                self.tok_s
                    .partial_cmp(&other.tok_s)
                    // `None` is "not measured", which must not beat a measured
                    // row; `partial_cmp` on two `None`s is `Equal` and on one
                    // `None` is `None`, so the fallback decides that case.
                    .unwrap_or(match (self.tok_s, other.tok_s) {
                        (Some(_), None) => Ordering::Greater,
                        (None, Some(_)) => Ordering::Less,
                        _ => Ordering::Equal,
                    }),
            )
            .then(self.install_bytes.cmp(&other.install_bytes))
    }
}

/// The largest `<number>B` parameter claim in a repository name
/// (`owner/Qwen3.8-27B-GGUF` -> 27.0).
///
/// Ported from shoehorn, where it spots a repository whose full-precision
/// source is a small draft companion rather than the named model. It does the
/// same job here against a QUANTIZED artifact, which is why
/// [`SUSPICIOUS_BYTES_PER_PARAM`] is what it is rather than upstream's 0.3 of
/// a 2-bytes-per-parameter source.
pub fn name_params_hint(repo: &str) -> Option<f64> {
    let s = repo.as_bytes();
    let mut best: Option<f64> = None;
    let mut i = 0;
    while i < s.len() {
        if s[i].is_ascii_digit() && (i == 0 || !s[i - 1].is_ascii_alphabetic()) {
            let start = i;
            while i < s.len() && (s[i].is_ascii_digit() || s[i] == b'.') {
                i += 1;
            }
            let num_end = i;
            if num_end < s.len()
                && (s[num_end] == b'b' || s[num_end] == b'B')
                && (num_end + 1 == s.len() || !s[num_end + 1].is_ascii_alphanumeric())
            {
                if let Ok(v) = repo[start..num_end].trim_matches('.').parse::<f64>() {
                    best = Some(best.map_or(v, |b: f64| b.max(v)));
                }
            }
        } else {
            i += 1;
        }
    }
    best
}

/// Bytes per parameter below which an artifact is not plausibly the model its
/// name claims.
///
/// The narrowest quantization this port executes is IQ3_XXS at about 0.26
/// bytes per weight, and a whole install carries an unquantized embedding
/// table and norms on top. A file under 0.15 bytes per claimed parameter is
/// therefore not a small quantization of that model; it is a different and
/// much smaller model wearing the name, which on Hugging Face is usually a
/// draft companion published beside the real one.
const SUSPICIOUS_BYTES_PER_PARAM: f64 = 0.15;

/// Whether `bytes` is too small to be the model `repo`'s name claims.
pub fn suspicious(repo: &str, bytes: u64) -> bool {
    name_params_hint(repo)
        .is_some_and(|billions| (bytes as f64) < billions * 1e9 * SUSPICIOUS_BYTES_PER_PARAM)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_hints_parse() {
        assert_eq!(name_params_hint("owner/Qwen3.8-27B-GGUF"), Some(27.0));
        assert_eq!(name_params_hint("owner/LFM2.5-2.6B-GGUF"), Some(2.6));
        assert_eq!(
            name_params_hint("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"),
            Some(8.0)
        );
        assert_eq!(name_params_hint("owner/maple-preview-GGUF"), None);
        // The largest claim wins, which is what makes the hint a floor on the
        // expected size rather than an average.
        assert_eq!(name_params_hint("owner/Llama-3.2-1B-and-3B"), Some(3.0));
        // **`8x7B` READS AS NO CLAIM AT ALL**, and that is upstream's
        // behaviour preserved rather than a bug to fix here: the `7` is
        // preceded by a letter, so it is skipped, and the `8` is not followed
        // by one. An MoE's total parameter count is not `experts x expert
        // size` anyway (Mixtral 8x7B is 47B, not 56B), so a hint derived from
        // that spelling would be wrong in the direction that matters -- it
        // would call a legitimate 26 GB install a draft companion.
        assert_eq!(name_params_hint("owner/Mixtral-8x7B-v0.1"), None);
    }

    /// The check has to pass a real quantization of a real model and fail a
    /// draft companion. Both figures below are real: `gemma4-iq3` is the
    /// narrowest install in the catalog at 12 GB for a 26B model, and a 27B
    /// repository whose only file is 900 MB is the case shoehorn's own
    /// comment describes.
    #[test]
    fn the_suspicious_check_passes_a_real_3_bit_install_and_fails_a_draft() {
        assert!(
            !suspicious("google/gemma-4-26B-A4B-it-UD-Q3_K_M", 12_000_000_000),
            "the narrowest quantization this port runs must not read as a draft"
        );
        assert!(suspicious("owner/Qwen3.8-27B-GGUF", 900_000_000));
        // No claim in the name means no opinion, never a refusal.
        assert!(!suspicious("owner/maple-preview-GGUF", 900_000_000));
    }

    fn key(runs: bool, evidence: Evidence, tok_s: Option<f64>, bytes: u64) -> Key {
        Key {
            runs,
            evidence,
            tok_s,
            install_bytes: bytes,
            suspicious: false,
        }
    }

    /// **A discovered row can never outrank a gated one that fits**, however
    /// much bigger or faster it looks. This is the ordering the whole module
    /// exists to impose.
    #[test]
    fn evidence_beats_size_and_fitting_beats_evidence() {
        let mut rows = vec![
            (
                "huge discovery",
                key(true, Evidence::Discovered, None, 90_000_000_000),
            ),
            (
                "verified",
                key(true, Evidence::Verified, Some(33.0), 13_000_000_000),
            ),
            (
                "refused verified",
                key(false, Evidence::Verified, Some(99.0), 1),
            ),
            ("runs", key(true, Evidence::Runs, None, 4_000_000_000)),
        ];
        rank(&mut rows, |r| r.1);
        assert_eq!(
            rows.iter().map(|r| r.0).collect::<Vec<_>>(),
            ["verified", "runs", "huge discovery", "refused verified"]
        );
    }

    /// Within one tier the measured rate decides, and an unmeasured row loses
    /// to a measured one rather than winning by default.
    #[test]
    fn a_measured_rate_outranks_an_unmeasured_row_of_the_same_tier() {
        let mut rows = vec![
            (
                "unmeasured",
                key(true, Evidence::Verified, None, 99_000_000_000),
            ),
            (
                "slow but measured",
                key(true, Evidence::Verified, Some(12.0), 1),
            ),
        ];
        rank(&mut rows, |r| r.1);
        assert_eq!(rows[0].0, "slow but measured");
    }

    /// A suspicious artifact sinks below everything that fits, because its
    /// other columns are describing a model it does not contain.
    #[test]
    fn a_suspicious_row_sinks_below_every_honest_fit() {
        let mut rows = vec![
            (
                "draft companion",
                Key {
                    suspicious: true,
                    ..key(true, Evidence::Verified, Some(99.0), 90_000_000_000)
                },
            ),
            ("ordinary", key(true, Evidence::Discovered, None, 1)),
        ];
        rank(&mut rows, |r| r.1);
        assert_eq!(rows[0].0, "ordinary");
    }
}
