//! Timestamped alignment data model shared by the NeMo transducer families.
//!
//! Reference: `mlx_audio/stt/models/nemo/alignment.py` at mlx-audio 0.5.7,
//! commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`. Timestamps are seconds
//! on the waveform timeline; `end` always equals `start + duration`.

use crate::{Result, SpeechError};

/// One decoded vocabulary piece pinned to its waveform interval.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignedToken {
    /// Vocabulary index of the decoded token.
    pub id: i32,
    /// Detokenized piece text with sentencepiece underscores widened.
    pub text: String,
    /// Waveform time the piece starts speaking, in seconds.
    pub start: f32,
    /// Waveform time the piece spans, in seconds.
    pub duration: f32,
    /// `start + duration`, in seconds.
    pub end: f32,
}

impl AlignedToken {
    pub fn new(id: i32, text: impl Into<String>, start: f32, duration: f32) -> Self {
        Self {
            id,
            text: text.into(),
            start,
            duration,
            end: start + duration,
        }
    }
}

/// A punctuation-delimited run of aligned tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignedSentence {
    pub text: String,
    pub tokens: Vec<AlignedToken>,
    pub start: f32,
    pub end: f32,
    pub duration: f32,
}

impl AlignedSentence {
    /// Sorts the tokens by start time and derives the interval from the
    /// first and last token, mirroring the upstream post-init. An empty
    /// token list yields a zero interval at the origin.
    pub fn new(text: impl Into<String>, mut tokens: Vec<AlignedToken>) -> Self {
        tokens.sort_by(|a, b| {
            a.start
                .partial_cmp(&b.start)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let (start, end) = match (tokens.first(), tokens.last()) {
            (Some(first), Some(last)) => (first.start, last.end),
            _ => (0.0, 0.0),
        };
        Self {
            text: text.into(),
            tokens,
            start,
            end,
            duration: end - start,
        }
    }
}

/// A transcript with its sentence-level alignment.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignedResult {
    pub text: String,
    pub sentences: Vec<AlignedSentence>,
}

impl AlignedResult {
    fn new(text: String, sentences: Vec<AlignedSentence>) -> Self {
        Self {
            text: text.trim().to_string(),
            sentences,
        }
    }
}

fn ends_sentence(token_text: &str, next_token_text: Option<&str>) -> bool {
    if token_text.contains('!')
        || token_text.contains('?')
        || token_text.contains('\u{3002}')
        || token_text.contains('\u{FF1F}')
        || token_text.contains('\u{FF01}')
    {
        return true;
    }
    // A period only closes a sentence when another piece follows with
    // whitespace of its own or the transcript ends.
    token_text.contains('.') && next_token_text.is_none_or(|next| next.contains(' '))
}

/// Groups aligned tokens into sentences at terminal punctuation.
pub fn tokens_to_sentences(tokens: Vec<AlignedToken>) -> Vec<AlignedSentence> {
    let mut sentences = Vec::new();
    let mut current: Vec<AlignedToken> = Vec::new();
    let mut tokens = tokens.into_iter().peekable();
    while let Some(token) = tokens.next() {
        let closes = ends_sentence(&token.text, tokens.peek().map(|t| t.text.as_str()));
        current.push(token);
        if closes {
            let text: String = current.iter().map(|t| t.text.as_str()).collect();
            sentences.push(AlignedSentence::new(text, std::mem::take(&mut current)));
        }
    }
    if !current.is_empty() {
        let text: String = current.iter().map(|t| t.text.as_str()).collect();
        sentences.push(AlignedSentence::new(text, current));
    }
    sentences
}

/// Joins sentence segments into one aligned transcript.
pub fn sentences_to_result(sentences: Vec<AlignedSentence>) -> AlignedResult {
    let text: String = sentences.iter().map(|s| s.text.as_str()).collect();
    AlignedResult::new(text, sentences)
}

fn time_ordered(a: &[AlignedToken], b: &[AlignedToken]) -> bool {
    match (a.last(), b.first()) {
        (Some(last), Some(first)) => last.end <= first.start,
        _ => true,
    }
}

fn cutoff_split(a: &[AlignedToken], b: &[AlignedToken]) -> Vec<AlignedToken> {
    let cutoff = (a[a.len() - 1].end + b[0].start) / 2.0;
    let mut merged: Vec<AlignedToken> = a.iter().filter(|t| t.end <= cutoff).cloned().collect();
    merged.extend(b.iter().filter(|t| t.start >= cutoff).cloned());
    merged
}

fn starts_match(a: &AlignedToken, b: &AlignedToken, half_overlap: f32) -> bool {
    a.id == b.id && (a.start - b.start).abs() < half_overlap
}

/// Merges two overlapping chunk alignments on their longest run of matching
/// tokens, preferring the side with the denser gaps.
///
/// Mirrors upstream `merge_longest_contiguous`. Fails when the overlap is
/// long but holds no matching run, which upstream surfaces as a runtime
/// error.
pub fn merge_longest_contiguous(
    a: Vec<AlignedToken>,
    b: Vec<AlignedToken>,
    overlap_duration: f32,
) -> Result<Vec<AlignedToken>> {
    if a.is_empty() {
        return Ok(b);
    }
    if b.is_empty() {
        return Ok(a);
    }
    if time_ordered(&a, &b) {
        let mut merged = a;
        merged.extend(b);
        return Ok(merged);
    }

    let a_end = a[a.len() - 1].end;
    let b_start = b[0].start;
    let overlap_a: Vec<&AlignedToken> = a
        .iter()
        .filter(|t| t.end > b_start - overlap_duration)
        .collect();
    let overlap_b: Vec<&AlignedToken> = b
        .iter()
        .filter(|t| t.start < a_end + overlap_duration)
        .collect();
    let enough_pairs = overlap_a.len() / 2;
    if overlap_a.len() < 2 || overlap_b.len() < 2 {
        return Ok(cutoff_split(&a, &b));
    }

    let half_overlap = overlap_duration / 2.0;
    let mut best: Vec<(usize, usize)> = Vec::new();
    for i in 0..overlap_a.len() {
        for j in 0..overlap_b.len() {
            if !starts_match(overlap_a[i], overlap_b[j], half_overlap) {
                continue;
            }
            let mut run = Vec::new();
            let (mut k, mut l) = (i, j);
            while k < overlap_a.len()
                && l < overlap_b.len()
                && starts_match(overlap_a[k], overlap_b[l], half_overlap)
            {
                run.push((k, l));
                k += 1;
                l += 1;
            }
            if run.len() > best.len() {
                best = run;
            }
        }
    }

    if best.len() < enough_pairs {
        return Err(SpeechError::Input {
            why: format!("chunk overlap holds no matching run of {enough_pairs} tokens"),
        });
    }

    let a_start_idx = a.len() - overlap_a.len();
    let first_a = a_start_idx + best[0].0;
    let mut merged: Vec<AlignedToken> = a[..first_a].to_vec();
    for pair in 0..best.len() {
        let (idx_a, idx_b) = (best[pair].0, best[pair].1);
        merged.push(overlap_a[idx_a].clone());
        if pair + 1 < best.len() {
            let (next_a, next_b) = (best[pair + 1].0, best[pair + 1].1);
            let gap_a = &overlap_a[idx_a + 1..next_a];
            let gap_b = &overlap_b[idx_b + 1..next_b];
            if gap_b.len() > gap_a.len() {
                merged.extend(gap_b.iter().map(|t| (*t).clone()));
            } else {
                merged.extend(gap_a.iter().map(|t| (*t).clone()));
            }
        }
    }
    let last_b = best[best.len() - 1].1;
    merged.extend(overlap_b[last_b + 1..].iter().map(|t| (*t).clone()));
    Ok(merged)
}

/// Merges two overlapping chunk alignments along their longest common
/// subsequence of matching tokens, falling back to a midpoint cutoff.
///
/// Mirrors upstream `merge_longest_common_subsequence`.
pub fn merge_longest_common_subsequence(
    a: Vec<AlignedToken>,
    b: Vec<AlignedToken>,
    overlap_duration: f32,
) -> Result<Vec<AlignedToken>> {
    if a.is_empty() {
        return Ok(b);
    }
    if b.is_empty() {
        return Ok(a);
    }
    if time_ordered(&a, &b) {
        let mut merged = a;
        merged.extend(b);
        return Ok(merged);
    }

    let a_end = a[a.len() - 1].end;
    let b_start = b[0].start;
    let overlap_a: Vec<&AlignedToken> = a
        .iter()
        .filter(|t| t.end > b_start - overlap_duration)
        .collect();
    let overlap_b: Vec<&AlignedToken> = b
        .iter()
        .filter(|t| t.start < a_end + overlap_duration)
        .collect();
    if overlap_a.len() < 2 || overlap_b.len() < 2 {
        return Ok(cutoff_split(&a, &b));
    }

    let half_overlap = overlap_duration / 2.0;
    let (rows, cols) = (overlap_a.len(), overlap_b.len());
    let stride = cols + 1;
    let mut dp = vec![0usize; (rows + 1) * stride];
    for i in 1..=rows {
        for j in 1..=cols {
            if starts_match(overlap_a[i - 1], overlap_b[j - 1], half_overlap) {
                dp[i * stride + j] = dp[(i - 1) * stride + (j - 1)] + 1;
            } else {
                dp[i * stride + j] = dp[(i - 1) * stride + j].max(dp[i * stride + (j - 1)]);
            }
        }
    }

    let mut pairs = Vec::new();
    let (mut i, mut j) = (rows, cols);
    while i > 0 && j > 0 {
        if starts_match(overlap_a[i - 1], overlap_b[j - 1], half_overlap) {
            pairs.push((i - 1, j - 1));
            i -= 1;
            j -= 1;
        } else if dp[(i - 1) * stride + j] > dp[i * stride + (j - 1)] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    pairs.reverse();

    if pairs.is_empty() {
        return Ok(cutoff_split(&a, &b));
    }

    let a_start_idx = a.len() - overlap_a.len();
    let first_a = a_start_idx + pairs[0].0;
    let mut merged: Vec<AlignedToken> = a[..first_a].to_vec();
    for pair in 0..pairs.len() {
        let (idx_a, idx_b) = (pairs[pair].0, pairs[pair].1);
        merged.push(overlap_a[idx_a].clone());
        if pair + 1 < pairs.len() {
            let (next_a, next_b) = (pairs[pair + 1].0, pairs[pair + 1].1);
            let gap_a = &overlap_a[idx_a + 1..next_a];
            let gap_b = &overlap_b[idx_b + 1..next_b];
            if gap_b.len() > gap_a.len() {
                merged.extend(gap_b.iter().map(|t| (*t).clone()));
            } else {
                merged.extend(gap_a.iter().map(|t| (*t).clone()));
            }
        }
    }
    let last_b = pairs[pairs.len() - 1].1;
    merged.extend(overlap_b[last_b + 1..].iter().map(|t| (*t).clone()));
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(id: i32, text: &str, start: f32, duration: f32) -> AlignedToken {
        AlignedToken::new(id, text, start, duration)
    }

    #[test]
    fn token_end_is_start_plus_duration() {
        let t = token(7, " hi", 0.4, 0.08);
        assert!((t.end - 0.48).abs() < 1e-6);
    }

    #[test]
    fn sentences_split_on_terminal_punctuation() {
        let tokens = vec![
            token(1, " hello", 0.0, 0.08),
            token(2, " world", 0.08, 0.08),
            // A period followed by a space in the next piece closes it.
            token(3, ".", 0.16, 0.08),
            token(4, " next", 0.24, 0.08),
            token(5, " one", 0.32, 0.08),
            // A bare period at the end closes the final sentence.
            token(6, ".", 0.4, 0.08),
        ];
        let result = sentences_to_result(tokens_to_sentences(tokens.clone()));
        assert_eq!(result.sentences.len(), 2);
        assert_eq!(result.sentences[0].text, " hello world.");
        assert_eq!(result.sentences[1].text, " next one.");
        // The result text is trimmed (upstream post-init) even though the
        // sentence segments keep their leading whitespace.
        assert_eq!(result.text, "hello world. next one.");
        assert_eq!(result.sentences[0].start, 0.0);
        assert_eq!(result.sentences[0].end, 0.24);
        assert_eq!(result.sentences[1].start, 0.24);
    }

    #[test]
    fn periods_without_following_whitespace_do_not_split() {
        let tokens = vec![token(1, " 3.5", 0.0, 0.08), token(2, "million", 0.08, 0.08)];
        let sentences = tokens_to_sentences(tokens);
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0].text, " 3.5million");
    }

    #[test]
    fn period_before_a_spaced_piece_splits_like_upstream() {
        // Upstream's documented "hacky" rule: any period followed by a
        // piece carrying its own whitespace closes the sentence, so a
        // decimal before a spaced word splits too.
        let tokens = vec![
            token(1, " 3.5", 0.0, 0.08),
            token(2, " million", 0.08, 0.08),
        ];
        let sentences = tokens_to_sentences(tokens);
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0].text, " 3.5");
    }

    #[test]
    fn cjk_terminators_split_without_whitespace() {
        let tokens = vec![
            token(1, "\u{4f60}\u{597d}", 0.0, 0.08),
            token(2, "\u{3002}", 0.08, 0.08),
            token(3, "\u{518d}\u{89c1}", 0.16, 0.08),
            token(4, "\u{FF01}", 0.24, 0.08),
        ];
        let sentences = tokens_to_sentences(tokens);
        assert_eq!(sentences.len(), 2);
    }

    #[test]
    fn sentence_tokens_are_sorted_by_start() {
        let tokens = vec![token(1, " b", 0.16, 0.08), token(2, " a", 0.0, 0.08)];
        let sentence = AlignedSentence::new(" a b", tokens);
        assert_eq!(sentence.tokens[0].text, " a");
        assert_eq!(sentence.start, 0.0);
        assert_eq!(sentence.end, 0.24);
    }

    #[test]
    fn disjoint_chunks_concatenate() {
        let a = vec![token(1, " hi", 0.0, 0.08)];
        let b = vec![token(2, " there", 1.0, 0.08)];
        let merged = merge_longest_contiguous(a.clone(), b.clone(), 0.5).unwrap();
        assert_eq!(merged.len(), 2);
        let merged = merge_longest_common_subsequence(a, b, 0.5).unwrap();
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn empty_side_passes_through() {
        let merged =
            merge_longest_contiguous(Vec::new(), vec![token(1, " hi", 0.0, 0.08)], 0.5).unwrap();
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn thin_overlap_splits_at_the_midpoint() {
        // With a single-token overlap on each side the merge falls back to
        // the midpoint cutoff: pieces not fully clear of the midpoint on
        // either side are dropped, mirroring upstream.
        let a = vec![token(1, " hi", 0.0, 0.08), token(2, " you", 0.08, 1.0)];
        let b = vec![token(3, " there", 1.02, 0.5)];
        let merged = merge_longest_contiguous(a, b, 0.5).unwrap();
        let texts: Vec<&str> = merged.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, vec![" hi"]);
    }

    #[test]
    fn matching_overlap_run_is_deduplicated() {
        let a = vec![
            token(1, " the", 0.0, 0.08),
            token(2, " quick", 0.08, 0.08),
            token(3, " brown", 0.16, 0.08),
            token(4, " fox", 0.24, 0.08),
        ];
        let b = vec![
            token(2, " quick", 0.10, 0.08),
            token(3, " brown", 0.18, 0.08),
            token(5, " jumps", 0.26, 0.08),
        ];
        let merged = merge_longest_contiguous(a, b, 0.5).unwrap();
        let texts: Vec<&str> = merged.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, vec![" the", " quick", " brown", " jumps"]);
    }

    #[test]
    fn lcs_merge_deduplicates_via_subsequence() {
        let a = vec![
            token(1, " the", 0.0, 0.08),
            token(2, " quick", 0.08, 0.08),
            token(3, " brown", 0.16, 0.08),
            token(4, " fox", 0.24, 0.08),
        ];
        let b = vec![
            token(2, " quick", 0.10, 0.08),
            token(3, " brown", 0.18, 0.08),
            token(5, " jumps", 0.26, 0.08),
        ];
        let merged = merge_longest_common_subsequence(a, b, 0.5).unwrap();
        let texts: Vec<&str> = merged.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, vec![" the", " quick", " brown", " jumps"]);
    }
}
