//! Prompt-copy drafting for greedy speculation (after mtplx, Apache-2.0,
//! github.com/youssofal/mtplx): when the last `n` committed tokens (`n` from
//! [`N_MAX`] down to [`N_MIN`]) occurred earlier in the history, propose the
//! tokens that followed that occurrence. Code edits, quoted passages and
//! structured output repeat long spans of their own context, which a learned
//! drafter proposes a few tokens at a time; a copy round proposes up to the
//! producer's copy cap in one verify. The verify decides every token, so a
//! wrong copy costs a verify, never output.
//!
//! Probation: a verify row costs about half a decode step (M4 Max, mlx4), so
//! a copy round starts at the learned drafter's own width (the verify a
//! drafter round pays, minus the draft) and doubles only after a round
//! accepts every proposal; any rejection drops it back. Backoff: a copy round that accepts
//! fewer than [`GOOD_ACCEPT`] proposals pauses copying for `2^misses` rounds
//! (capped at [`MAX_PAUSE`]), so text that matches by accident falls back to
//! the learned drafter.

use crate::TokenId;

/// Longest and shortest suffix tried, longest first.
pub(crate) const N_MAX: usize = 10;
pub(crate) const N_MIN: usize = 6;
/// A continuation shorter than this is left to the learned drafter.
pub(crate) const MIN_PROPOSALS: usize = 4;
/// Accepted proposals that count a copy round as a hit.
pub(crate) const GOOD_ACCEPT: usize = 2;
pub(crate) const MAX_PAUSE: u32 = 16;

/// The continuation of the most recent earlier occurrence of the longest
/// matching suffix of `history` (`N_MAX` down to `N_MIN` tokens), at most
/// `max` tokens, or `None`. The continuation is read from `history` itself,
/// so it may run into the suffix (a repeating pattern copies forward).
pub(crate) fn find_copy(history: &[TokenId], max: usize) -> Option<&[TokenId]> {
    let len = history.len();
    if max == 0 {
        return None;
    }
    for n in (N_MIN..=N_MAX).rev() {
        if len <= n {
            continue;
        }
        let suffix = &history[len - n..];
        // An occurrence starting at `j` ends at `j + n`, before the suffix's
        // own start, and has at least one token after it.
        for j in (0..len - n).rev() {
            if &history[j..j + n] == suffix {
                let from = j + n;
                let to = (from + max).min(len);
                return Some(&history[from..to]);
            }
        }
    }
    None
}

/// Per-run copy state: the pause left, the miss streak and the width.
#[derive(Debug, Default)]
pub(crate) struct PromptCopy {
    pause: u32,
    misses: u32,
    /// Proposals the next copy round may take; 0 means the start width.
    width: usize,
    pub(crate) rounds: usize,
    pub(crate) accepted: usize,
}

impl PromptCopy {
    /// Fills `out` with a copy round's proposals and returns true, or returns
    /// false (paused, no match, or too short a continuation). `start` is the
    /// width on probation, `max` the cap.
    pub(crate) fn propose(
        &mut self,
        history: &[TokenId],
        start: usize,
        max: usize,
        out: &mut Vec<TokenId>,
    ) -> bool {
        if self.pause > 0 {
            self.pause -= 1;
            return false;
        }
        let width = if self.width == 0 { start } else { self.width };
        match find_copy(history, max.min(width)) {
            Some(c) if c.len() >= MIN_PROPOSALS.min(start) => {
                out.clear();
                out.extend_from_slice(c);
                true
            }
            _ => false,
        }
    }

    /// Records a copy round's outcome: `accepted` of `proposed`.
    pub(crate) fn observe(&mut self, accepted: usize, proposed: usize) {
        self.rounds += 1;
        self.accepted += accepted;
        self.width = if accepted == proposed {
            proposed * 2
        } else {
            0
        };
        if accepted >= GOOD_ACCEPT {
            self.misses = 0;
        } else {
            self.misses = (self.misses + 1).min(5);
            self.pause = (1u32 << self.misses).min(MAX_PAUSE);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_the_continuation_of_the_longest_most_recent_match() {
        // 1..10, 77, 1..10, 88, then the suffix 1..10: the most recent
        // earlier occurrence (index 11) is followed by 88, and the copy runs
        // on into the suffix itself.
        let mut h: Vec<TokenId> = (1..=10).collect();
        h.push(77);
        h.extend(1..=10);
        h.push(88);
        h.extend(1..=10);
        assert_eq!(find_copy(&h, 4), Some(&[88, 1, 2, 3][..]));
        // The cap cuts the copy.
        assert_eq!(find_copy(&h, 1), Some(&[88][..]));
    }

    #[test]
    fn needs_at_least_n_min_matching_tokens() {
        let mut h: Vec<TokenId> = vec![1, 2, 3, 4, 5, 9, 9, 9];
        h.extend([1, 2, 3, 4, 5]);
        // Five matching tokens are below N_MIN.
        assert_eq!(find_copy(&h, 8), None);
        let mut h: Vec<TokenId> = vec![1, 2, 3, 4, 5, 6, 42, 43];
        h.extend([1, 2, 3, 4, 5, 6]);
        assert_eq!(find_copy(&h, 8), Some(&[42, 43, 1, 2, 3, 4, 5, 6][..]));
    }

    #[test]
    fn prefers_a_longer_suffix_over_a_more_recent_shorter_one() {
        // Ten-token match at 0 continues with 50; a later six-token match
        // (tokens 5..=10) continues with 60. Longest suffix wins.
        let mut h: Vec<TokenId> = (1..=10).collect();
        h.push(50);
        h.extend(5..=10);
        h.push(60);
        h.extend(1..=10);
        assert_eq!(find_copy(&h, 1), Some(&[50][..]));
    }

    #[test]
    fn backoff_pauses_after_a_miss_and_resets_on_a_hit() {
        let mut h: Vec<TokenId> = (1..=6).collect();
        h.extend([7, 8, 9, 10, 11]);
        h.extend(1..=6);
        let mut pc = PromptCopy::default();
        let mut out = Vec::new();
        assert!(pc.propose(&h, 7, 24, &mut out));
        assert_eq!(out.len(), 7);
        pc.observe(0, out.len());
        // One miss: paused for 2 rounds.
        assert!(!pc.propose(&h, 7, 24, &mut out));
        assert!(!pc.propose(&h, 7, 24, &mut out));
        assert!(pc.propose(&h, 7, 24, &mut out));
        pc.observe(1, out.len());
        // Two misses: 4 rounds.
        for _ in 0..4 {
            assert!(!pc.propose(&h, 7, 24, &mut out));
        }
        assert!(pc.propose(&h, 7, 24, &mut out));
        pc.observe(GOOD_ACCEPT, out.len());
        assert!(pc.propose(&h, 7, 24, &mut out));
        assert_eq!((pc.rounds, pc.accepted), (3, 1 + GOOD_ACCEPT));
    }

    #[test]
    fn width_doubles_on_full_acceptance_and_resets_on_a_rejection() {
        let h: Vec<TokenId> = (0..200).map(|i| i % 30).collect();
        let mut pc = PromptCopy::default();
        let mut out = Vec::new();
        let mut widths = Vec::new();
        for accept_all in [true, true, false, true] {
            assert!(pc.propose(&h, 7, 24, &mut out));
            widths.push(out.len());
            let n = out.len();
            pc.observe(if accept_all { n } else { n - 1 }, n);
        }
        // 7, then 14, then the cap 24; the rejection resets to 7.
        assert_eq!(widths, [7, 14, 24, 7]);
    }

    #[test]
    fn a_short_continuation_is_left_to_the_drafter() {
        let mut h: Vec<TokenId> = (1..=6).collect();
        h.extend([7, 8]);
        h.extend(1..=6);
        // Continuation 7, 8, 1, 2, ... is long enough here; cap it short.
        let mut pc = PromptCopy::default();
        let mut out = Vec::new();
        assert!(!pc.propose(&h, 7, MIN_PROPOSALS - 1, &mut out));
    }
}
