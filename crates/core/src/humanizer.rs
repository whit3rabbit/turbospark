//! The humanizer rewrite flow: the byte-exact completion wrapper, the
//! model's sampling recipe, and the post-generation audit.
//!
//! Lives in `foundation` rather than beside its first caller (the CLI's
//! humanize mode) because the bench gates that freeze the model's quality
//! and memory rows need the same wrapper bytes, and because the FFI surface
//! a later Swift "humanizing" section will call is this crate's dependency
//! direction too. Pure: no IO, no Metal, no tokenizer.
//!
//! The model (`cowWhySo/humanizer-12b-mlx-8bit`, a dense Gemma 4 12B) is a
//! RAW COMPLETION rewriter, not a chat model: it takes
//! `instruction + "\n\n" + draft + separator` and ends at EOS. Its
//! publisher is explicit that a chat template breaks it and that `###`
//! must never be a stop string, so the wrapper below is byte-exact and
//! fingerprint-checked, and the stop set stays the tokenizer default
//! (EOS only).

/// The instruction, byte-for-byte from the pinned
/// `prompt_format.json` (`instr` field) at
/// `cowWhySo/humanizer-12b-mlx-8bit` @ baea046b. The embedded newlines are
/// part of the instruction; changing a single byte moves the model off the
/// distribution it was trained with, which is what the fingerprint test
/// below exists to catch.
pub const HUMANIZER_INSTRUCTION: &str = "Rewrite the text below so it reads like a person wrote it, not a language model.\n\nReorganize it as you see fit. Vary sentence length on purpose. Cut hedging,\nthroat-clearing, and any sentence that only announces what comes next.\nPrefer the concrete word over the abstract one. It is fine to sound uneven.\n\nEvery fact, number, unit, date, name and quotation must survive unchanged.";

/// The completion prompt's closing separator, from the same file's `sep`.
pub const HUMANIZER_SEPARATOR: &str = "\n\n### Rewritten:\n\n";

/// The publisher's own integrity check: SHA-256 of the wrapper built around
/// the one-character draft `X`, hex-prefixed. Mirrors
/// `humanizer_support.check_prompt` so a transcription slip cannot ship.
pub const PROMPT_FINGERPRINT: &str = "cc51d66b4c593fbe";

/// Builds the rewrite prompt: instruction, blank line, the draft
/// trimmed of surrounding whitespace, then the separator. Raw completion:
/// the caller encodes this verbatim WITH the tokenizer's BOS, and never
/// through a chat template.
pub fn build_humanize_prompt(draft: &str) -> String {
    format!(
        "{}\n\n{}{}",
        HUMANIZER_INSTRUCTION,
        draft.trim(),
        HUMANIZER_SEPARATOR
    )
}

/// The publisher's recipe, verified against their MLX runner
/// (`humanize_mlx.py`'s `make_sampler(temp=1.0, top_p=0.95, top_k=0,
/// min_p=0.0)`) and `generation_config.json`. The CLI defaults differ in
/// every field that matters (temperature 0.2, top_k 64), so the humanize
/// flow FORCES these rather than inheriting the CLI's.
pub const HUMANIZER_TEMPERATURE: f64 = 1.0;
pub const HUMANIZER_TOP_K: u32 = 0;
pub const HUMANIZER_TOP_P: f64 = 0.95;
pub const HUMANIZER_REPETITION_PENALTY: f64 = 1.0;

/// The model's context window; the publisher's runner refuses any request
/// where prompt + output budget exceeds it.
pub const HUMANIZER_CONTEXT: usize = 8192;

/// The default output budget: about 2.5 rewrites' worth of tokens per
/// draft token, floored and capped per the publisher's usage guide
/// (256..=2048), and always clampable by the caller via `--max-new`.
pub fn default_max_new(draft_tokens: usize) -> u32 {
    (draft_tokens * 5 / 2).clamp(256, 2048) as u32
}

// ---------------------------------------------------------------------------
// The audit: what the publisher's own tooling checks after a rewrite.
// ---------------------------------------------------------------------------

/// Which script dominates a text. Latin and CJK are the two the model
/// actually serves (the publisher documents English and Chinese drafts);
/// anything else, or a genuine tie, reports as `Mixed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Script {
    Latin,
    Cjk,
    Mixed,
}

/// The rewrite audit: fact (digit) drift, language drift, and verbatim
/// reuse. Every field is a WARNING for the reader, never a correction --
/// the publisher's own numbers say fact slips still happen (44/420 English
/// rewrites flagged, mostly single-word), so the audit's job is to point a
/// proofreader at the right sentences, not to pretend the rewrite is safe.
#[derive(Debug, Clone, PartialEq)]
pub struct RewriteAudit {
    /// Digit runs present in the draft whose count dropped in the rewrite
    /// (each missing OCCURRENCE, so `"37"` dropped twice reports twice).
    pub dropped_numbers: Vec<String>,
    /// Digit runs in the rewrite that the draft does not contain at all.
    pub invented_numbers: Vec<String>,
    /// `Some` when the rewrite's dominant script differs from the draft's
    /// (an English draft rewritten in Chinese, or the reverse).
    pub language_drift: bool,
    /// The share of the rewrite's word-level 5-grams that appear verbatim
    /// in the draft. The model was RL-penalized above 0.22 (median 0.165),
    /// so the threshold lives in [`REUSE_FLAG_THRESHOLD`], not here.
    pub reuse_ratio: f64,
}

/// The reuse level above which the audit flags verbatim copying.
pub const REUSE_FLAG_THRESHOLD: f64 = 0.22;

/// Extracts the maximal runs of ASCII digits ("Arabic numerals") from a
/// document. Runs, not individual digits, because `37` vs `3` `7` is the
/// fact slip that matters; the count of each run is what the audit diffs.
fn digit_runs(text: &str) -> Vec<String> {
    let mut runs = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        if c.is_ascii_digit() {
            current.push(c);
        } else if !current.is_empty() {
            runs.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        runs.push(current);
    }
    runs
}

/// Counts each distinct run.
fn count_runs(runs: &[String]) -> std::collections::HashMap<String, usize> {
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for run in runs {
        *counts.entry(run.clone()).or_insert(0) += 1;
    }
    counts
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x4E00..=0x9FFF      // CJK Unified Ideographs
        | 0x3400..=0x4DBF    // Extension A
        | 0x20000..=0x2A6DF  // Extension B
        | 0x3000..=0x303F    // CJK punctuation
        | 0xFF00..=0xFFEF    // fullwidth forms
    )
}

/// Dominant script by letter count. Only letters vote (digits and
/// punctuation are script-neutral); fewer than 4 voting characters is
/// undecidable and reports `Mixed`.
fn dominant_script(text: &str) -> Script {
    let mut latin = 0usize;
    let mut cjk = 0usize;
    for c in text.chars() {
        if c.is_ascii_alphabetic() {
            latin += 1;
        } else if is_cjk(c) {
            cjk += 1;
        }
    }
    if latin + cjk < 4 {
        return Script::Mixed;
    }
    match latin.cmp(&cjk) {
        std::cmp::Ordering::Greater => Script::Latin,
        std::cmp::Ordering::Less => Script::Cjk,
        std::cmp::Ordering::Equal => Script::Mixed,
    }
}

/// Lowercased word tokens: the units the 5-gram reuse check compares.
fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

/// The share of the rewrite's word-level 5-grams that appear verbatim in
/// the draft. A rewrite with fewer than 5 words scores 0.0 (nothing to
/// copy yet).
fn reuse_ratio(draft: &str, rewrite: &str) -> f64 {
    let draft_grams: std::collections::HashSet<[String; 5]> = {
        let w = words(draft);
        if w.len() < 5 {
            return 0.0;
        }
        (0..=w.len() - 5)
            .map(|i| {
                [
                    w[i].clone(),
                    w[i + 1].clone(),
                    w[i + 2].clone(),
                    w[i + 3].clone(),
                    w[i + 4].clone(),
                ]
            })
            .collect()
    };
    let rw = words(rewrite);
    if rw.len() < 5 {
        return 0.0;
    }
    let total = rw.len() - 4;
    let copied = (0..=rw.len() - 5)
        .filter(|&i| {
            draft_grams.contains(&[
                rw[i].clone(),
                rw[i + 1].clone(),
                rw[i + 2].clone(),
                rw[i + 3].clone(),
                rw[i + 4].clone(),
            ])
        })
        .count();
    copied as f64 / total as f64
}

/// Audits a rewrite against its draft. Pure: no IO, no allocation beyond
/// the report, so the CLI, a test, and later the FFI can all call it.
pub fn audit_rewrite(draft: &str, rewrite: &str) -> RewriteAudit {
    let (draft_counts, rewrite_counts) = (
        count_runs(&digit_runs(draft)),
        count_runs(&digit_runs(rewrite)),
    );
    let mut dropped_numbers = Vec::new();
    for (run, want) in &draft_counts {
        let got = rewrite_counts.get(run).copied().unwrap_or(0);
        for _ in got..*want {
            dropped_numbers.push(run.clone());
        }
    }
    let mut invented_numbers = Vec::new();
    for (run, got) in &rewrite_counts {
        let want = draft_counts.get(run).copied().unwrap_or(0);
        for _ in want..*got {
            invented_numbers.push(run.clone());
        }
    }
    dropped_numbers.sort();
    invented_numbers.sort();
    RewriteAudit {
        dropped_numbers,
        invented_numbers,
        language_drift: dominant_script(draft) != dominant_script(rewrite),
        reuse_ratio: reuse_ratio(draft, rewrite),
    }
}

impl RewriteAudit {
    /// The stderr report lines. Empty when the rewrite is clean; one line
    /// per finding otherwise, so stdout can stay pipeable rewritten text.
    pub fn warnings(&self) -> Vec<String> {
        let mut out = Vec::new();
        if !self.dropped_numbers.is_empty() {
            out.push(format!(
                "numbers dropped from the draft: {}",
                self.dropped_numbers.join(", ")
            ));
        }
        if !self.invented_numbers.is_empty() {
            out.push(format!(
                "numbers not in the draft: {}",
                self.invented_numbers.join(", ")
            ));
        }
        if self.language_drift {
            out.push(
                "language drift: the rewrite's dominant script differs from the draft's; \
                 consider resampling"
                    .to_string(),
            );
        }
        if self.reuse_ratio > REUSE_FLAG_THRESHOLD {
            out.push(format!(
                "verbatim reuse {:.3} above the {:.2} flag level; the rewrite leans on the \
                 draft's own phrasing",
                self.reuse_ratio, REUSE_FLAG_THRESHOLD
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapper_matches_the_publisher_fingerprint() {
        // sha256(build_prompt(pf, "X")) must start with the pinned prefix.
        // `sha2` is a dev-dependency; the fingerprint is the whole reason
        // the instruction is vendored as a const rather than read from a
        // file that could drift.
        use sha2::{Digest, Sha256};
        let prompt = build_humanize_prompt("X");
        let digest = Sha256::digest(prompt.as_bytes());
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert!(
            hex.starts_with(PROMPT_FINGERPRINT),
            "wrapper fingerprint {hex} does not start with {PROMPT_FINGERPRINT}"
        );
    }

    #[test]
    fn wrapper_trims_the_draft_and_keeps_inner_whitespace() {
        let prompt = build_humanize_prompt("  hello\nworld  \n");
        assert!(prompt.starts_with(HUMANIZER_INSTRUCTION));
        assert!(prompt.ends_with(HUMANIZER_SEPARATOR));
        assert!(
            prompt.contains("\n\nhello\nworld\n\n### Rewritten:"),
            "unexpected wrapper: {prompt:?}"
        );
    }

    #[test]
    fn default_budget_floors_caps_and_scales() {
        assert_eq!(default_max_new(1), 256);
        assert_eq!(default_max_new(100), 256);
        assert_eq!(default_max_new(1000), 2048);
        assert_eq!(default_max_new(4000), 2048);
        assert_eq!(default_max_new(400), 1000);
    }

    #[test]
    fn digit_audit_reports_dropped_and_invented() {
        let audit = audit_rewrite(
            "37 complaints and 12 emails in 2024, cost $5,000",
            "There were complaints and emails in 2024, about $5,000 overall, maybe 9 files",
        );
        assert_eq!(audit.dropped_numbers, vec!["12", "37"]);
        assert_eq!(audit.invented_numbers, vec!["9"]);
        assert!(!audit.language_drift);
    }

    #[test]
    fn digit_audit_is_quiet_when_numbers_survive() {
        let audit = audit_rewrite(
            "37 complaints and 12 emails",
            "12 emails arrived; 37 complaints followed",
        );
        assert!(audit.dropped_numbers.is_empty());
        assert!(audit.invented_numbers.is_empty());
        assert!(audit.warnings().is_empty());
    }

    #[test]
    fn language_drift_fires_on_script_flip() {
        assert!(audit_rewrite("Hello there, friend.", "你好，朋友。").language_drift);
        assert!(audit_rewrite("你好，朋友。", "Hello there, friend.").language_drift);
        assert!(!audit_rewrite("Bonjour l'ami.", "Hello there, friend.").language_drift);
    }

    #[test]
    fn reuse_ratio_flags_verbatim_spans() {
        let draft =
            "the committee reviewed the quarterly budget projections and then adjourned for lunch";
        // Identical text: everything is a copied 5-gram.
        let same = audit_rewrite(draft, draft);
        assert!(
            same.reuse_ratio > REUSE_FLAG_THRESHOLD,
            "{}",
            same.reuse_ratio
        );
        // A real rewrite shares almost no 5-grams.
        let fresh = audit_rewrite(
            draft,
            "lunch came after the committee finished going over budget numbers for the quarter",
        );
        assert!(
            fresh.reuse_ratio <= REUSE_FLAG_THRESHOLD,
            "{}",
            fresh.reuse_ratio
        );
        // Short texts have nothing to flag.
        assert_eq!(
            audit_rewrite("one two three", "one two three").reuse_ratio,
            0.0
        );
    }
}
