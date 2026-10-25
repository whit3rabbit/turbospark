//! Concatenated minimum-permutation WER (cpWER), matching the Frozen E scorer.
//!
//! The rules are those of the frozen qualification notebook (cell 8A), which
//! `crates/bench/scripts/frozen_e_score.py` copies verbatim:
//! - lowercase, then tokens are `[a-z0-9]+(?:'[a-z0-9]+)?` matches, so
//!   "white-board" is two tokens and "I'm" is one;
//! - reference words are sorted by start and hypothesis segments by
//!   `start_time` (both stable), then concatenated per speaker;
//! - the speaker lists are padded with empty streams to a square matrix and
//!   the minimum-cost one-to-one assignment is taken;
//! - the denominator is the reference token count.
//!
//! Changing any rule changes every frozen number, so the
//! `cpwer_reproduces_frozen_e_*` tests pin them to the frozen counts.

use std::collections::BTreeMap;

use super::seglst::{RefWord, Segment};

/// The frozen scorer refuses more than this many speakers on either side.
pub const MAX_SPEAKERS: usize = 8;

/// One clip's speaker-attributed score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpWer {
    /// Edit operations under the best speaker assignment.
    pub errors: usize,
    /// Reference tokens (the denominator).
    pub ref_tokens: usize,
    /// Distinct reference speakers.
    pub ref_speakers: usize,
    /// Distinct hypothesis speakers.
    pub hyp_speakers: usize,
    /// Chosen pairs `(reference speaker, hypothesis speaker, errors)`; `None`
    /// marks a padded empty stream.
    pub assignment: Vec<(Option<String>, Option<String>, usize)>,
}

/// Scoring refused because a side has more speakers than the frozen scorer allows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TooManySpeakers(pub usize);

impl std::fmt::Display for TooManySpeakers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} speakers exceeds the frozen scorer's bound of {MAX_SPEAKERS}",
            self.0
        )
    }
}

impl std::error::Error for TooManySpeakers {}

fn is_token_char(c: char) -> bool {
    c.is_ascii_lowercase() || c.is_ascii_digit()
}

/// Tokenizes like `re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", s.lower())`.
pub fn tokens(text: &str) -> Vec<String> {
    let lower: Vec<char> = text.to_lowercase().chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lower.len() {
        if !is_token_char(lower[i]) {
            i += 1;
            continue;
        }
        let start = i;
        while i < lower.len() && is_token_char(lower[i]) {
            i += 1;
        }
        // The optional apostrophe group only applies when a token character follows.
        if i + 1 < lower.len() && lower[i] == '\'' && is_token_char(lower[i + 1]) {
            i += 1;
            while i < lower.len() && is_token_char(lower[i]) {
                i += 1;
            }
        }
        out.push(lower[start..i].iter().collect());
    }
    out
}

/// Word-level Levenshtein distance.
pub fn edit_distance(a: &[String], b: &[String]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, x) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = (prev[j + 1] + 1)
                .min(cur[j] + 1)
                .min(prev[j] + usize::from(x != y));
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Scores one clip. `segments` must already be filtered to this clip's session.
pub fn cpwer(reference: &[RefWord], segments: &[Segment]) -> Result<CpWer, TooManySpeakers> {
    let mut ref_words: Vec<&RefWord> = reference.iter().collect();
    ref_words.sort_by(|a, b| a.start.total_cmp(&b.start));
    let mut refs: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for w in ref_words {
        refs.entry(w.speaker.as_str())
            .or_default()
            .extend(tokens(&w.text));
    }

    let mut segs: Vec<&Segment> = segments.iter().collect();
    segs.sort_by(|a, b| a.start_time.total_cmp(&b.start_time));
    let mut hyps: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for s in segs {
        hyps.entry(s.speaker.as_str())
            .or_default()
            .extend(tokens(&s.words));
    }

    let k = refs.len().max(hyps.len());
    if k > MAX_SPEAKERS {
        return Err(TooManySpeakers(k));
    }
    let mut a: Vec<(Option<&str>, &[String])> =
        refs.iter().map(|(s, w)| (Some(*s), w.as_slice())).collect();
    let mut b: Vec<(Option<&str>, &[String])> =
        hyps.iter().map(|(s, w)| (Some(*s), w.as_slice())).collect();
    a.resize(k, (None, &[]));
    b.resize(k, (None, &[]));

    let cost: Vec<Vec<usize>> = a
        .iter()
        .map(|(_, x)| b.iter().map(|(_, y)| edit_distance(x, y)).collect())
        .collect();
    let (errors, perm) = min_assignment(&cost);
    let assignment = perm
        .iter()
        .enumerate()
        .map(|(i, &j)| {
            (
                a[i].0.map(str::to_owned),
                b[j].0.map(str::to_owned),
                cost[i][j],
            )
        })
        .collect();
    Ok(CpWer {
        errors,
        ref_tokens: a.iter().map(|(_, w)| w.len()).sum(),
        ref_speakers: refs.len(),
        hyp_speakers: hyps.len(),
        assignment,
    })
}

/// Exact minimum-cost perfect matching on a small square matrix (k <= 8).
/// Brute force over permutations; the first minimum in lexicographic order wins,
/// which only affects the reported pairing, never the total.
fn min_assignment(cost: &[Vec<usize>]) -> (usize, Vec<usize>) {
    let k = cost.len();
    let mut perm: Vec<usize> = (0..k).collect();
    let mut best = (usize::MAX, perm.clone());
    loop {
        let total: usize = perm.iter().enumerate().map(|(i, &j)| cost[i][j]).sum();
        if total < best.0 {
            best = (total, perm.clone());
        }
        if !next_permutation(&mut perm) {
            break;
        }
    }
    if k == 0 {
        best.0 = 0;
    }
    best
}

fn next_permutation(v: &mut [usize]) -> bool {
    if v.len() < 2 {
        return false;
    }
    let mut i = v.len() - 1;
    while i > 0 && v[i - 1] >= v[i] {
        i -= 1;
    }
    if i == 0 {
        return false;
    }
    let mut j = v.len() - 1;
    while v[j] <= v[i - 1] {
        j -= 1;
    }
    v.swap(i - 1, j);
    v[i..].reverse();
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(speaker: &str, start: f64, text: &str) -> RefWord {
        RefWord {
            speaker: speaker.into(),
            start,
            end: start + 0.1,
            text: text.into(),
        }
    }

    fn seg(speaker: &str, start: f64, words: &str) -> Segment {
        Segment {
            session_id: "s".into(),
            speaker: speaker.into(),
            start_time: start,
            end_time: start + 1.0,
            words: words.into(),
        }
    }

    #[test]
    fn tokens_follow_the_frozen_regex() {
        assert_eq!(
            tokens("I'm on the White-board, 'Kay? rock'n'roll 2nd"),
            ["i'm", "on", "the", "white", "board", "kay", "rock'n", "roll", "2nd"]
        );
        assert_eq!(tokens("end'"), ["end"]);
        assert!(tokens(" ,.;- ").is_empty());
    }

    #[test]
    fn edit_distance_counts_word_operations() {
        let v = |s: &str| tokens(s);
        assert_eq!(edit_distance(&v("a b c"), &v("a b c")), 0);
        assert_eq!(edit_distance(&v("a b c"), &v("a x c")), 1);
        assert_eq!(edit_distance(&v("a b c"), &v("a c")), 1);
        assert_eq!(edit_distance(&v(""), &v("a b")), 2);
    }

    #[test]
    fn cpwer_picks_the_best_speaker_permutation() {
        let reference = [
            word("A", 0.0, "hello there"),
            word("B", 1.0, "general kenobi"),
        ];
        // Hypothesis labels are swapped relative to the reference.
        let hyp = [
            seg("speaker_1", 0.0, "hello there"),
            seg("speaker_0", 1.0, "general kenobi"),
        ];
        let s = cpwer(&reference, &hyp).unwrap();
        assert_eq!((s.errors, s.ref_tokens), (0, 4));
        assert!(s
            .assignment
            .contains(&(Some("A".into()), Some("speaker_1".into()), 0)));
    }

    #[test]
    fn cpwer_pads_unmatched_speakers() {
        // Three reference speakers, one hypothesis speaker: the two unmatched
        // reference streams count as deletions.
        let reference = [
            word("A", 0.0, "one two"),
            word("B", 1.0, "three"),
            word("C", 2.0, "four five six"),
        ];
        let hyp = [seg("speaker_0", 0.0, "four five six")];
        let s = cpwer(&reference, &hyp).unwrap();
        assert_eq!((s.errors, s.ref_tokens, s.hyp_speakers), (3, 6, 1));
        // An extra hypothesis speaker counts as insertions.
        let s = cpwer(
            &reference[..1],
            &[seg("x", 0.0, "one two"), seg("y", 1.0, "noise noise")],
        )
        .unwrap();
        assert_eq!((s.errors, s.ref_tokens), (2, 2));
    }

    #[test]
    fn cpwer_sorts_words_and_segments_by_start() {
        let reference = [word("A", 2.0, "second"), word("A", 1.0, "first")];
        let hyp = [seg("h", 5.0, "second"), seg("h", 0.5, "first")];
        assert_eq!(cpwer(&reference, &hyp).unwrap().errors, 0);
    }

    #[test]
    fn cpwer_refuses_more_than_eight_speakers() {
        let reference: Vec<RefWord> = (0..9)
            .map(|i| word(&format!("S{i}"), i as f64, "x"))
            .collect();
        assert_eq!(cpwer(&reference, &[]), Err(TooManySpeakers(9)));
    }
}
