//! What the KV cache currently holds, described by the token ids that built
//! it, so the next turn can continue from it instead of re-prefilling.
//!
//! A chat client resends the whole transcript every turn, and
//! `run_raw_completion` resets and prefills from position 0 every time, so
//! the cost of a message grows with the length of the conversation. On a
//! streaming MoE install every replayed position also re-reads its experts,
//! which is the dominant cost rather than a rounding error.
//!
//! THE IDEA. After a turn, the producer's state covers some number of
//! positions. Wherever the next prompt AGREES with the ids that built them,
//! that state already IS the state at those positions: skip the reset, move
//! the cursor back to where the two diverge, and prefill only from there.
//!
//! It is the LONGEST COMMON PREFIX rather than an all-or-nothing match, and
//! that distinction is the difference between a feature that fires and one
//! that does not. The record covers the previous prompt AND its reply; the
//! next prompt carries that reply back as re-rendered text, and re-tokenizing
//! it does not reliably reproduce the ids that were generated. Measured in
//! the real chat REPL, an all-or-nothing match reused 0 of 33 and 0 of 49
//! tokens across three turns -- it never fired once.
//!
//! Moving the cursor BACK is what buys that, and it is not free: see
//! `RealForwardRunner::try_reuse_prefix` for the two cases that refuse it
//! (recurrent GDN state, and a sliding-window ring past its slack).
//!
//! WHY A RECORD AND NOT A COUNTER. It is tempting to derive the reusable
//! length from the caller's own bookkeeping ("prompt_tokens + new_tokens").
//! That invariant differs per caller -- whether the last sampled token was
//! fed back, whether a chunked prefill ran to completion, whether generation
//! stopped early -- and getting it wrong does not crash: it silently answers
//! from a state belonging to a different conversation. So the ids are
//! recorded WHERE THEY ARE FED, and this record is the only description of
//! the state anyone consults.
//!
//! INVARIANT: `fed[0..len]` are exactly the token ids the current state was
//! built from, in position order. Everything else follows from it.
//!
//! TAINT. Some inputs are not described by their token ids. An injected
//! image occupies a placeholder span whose ids are the same whatever picture
//! filled it (`RealForwardRunner::set_prompt_vision`), so an id comparison
//! would call two different images equal and answer the second from the
//! first one's state. A state that consumed such an input is tainted and is
//! never reused. The same applies to any path that advances the KV cursor
//! without recording what it fed.

use foundation::TokenId;

/// The token ids behind the current KV contents. See the module docs.
#[derive(Debug, Default)]
pub(crate) struct KvPrefix {
    fed: Vec<TokenId>,
    /// The state consumed something the ids do not describe.
    tainted: bool,
}

impl KvPrefix {
    /// Forget the state. Pair this with whatever clears the KV itself, so
    /// the two can never disagree.
    pub(crate) fn clear(&mut self) {
        self.fed.clear();
        self.tainted = false;
    }

    /// Mark the state as holding something token ids cannot describe.
    /// Sticky until [`Self::clear`]: a turn that saw one image is not
    /// described by its ids no matter what is fed afterwards.
    pub(crate) fn taint(&mut self) {
        self.tainted = true;
    }

    /// Record `tokens` fed at consecutive positions starting at `pos0`.
    ///
    /// A write that does not START exactly at `len` drops the record rather
    /// than truncating or padding it. Both alternatives are worse than
    /// losing the optimisation: a gap would claim coverage of positions
    /// nothing fed, and silently truncating would describe a shorter state
    /// than the KV actually holds, which is the same lie one position over.
    pub(crate) fn record(&mut self, tokens: &[TokenId], pos0: usize) {
        if pos0 != self.fed.len() {
            if pos0 < self.fed.len() {
                // A rewind (speculative rollback, or a re-prefill of the
                // same span) is describable: keep what still holds.
                self.fed.truncate(pos0);
            } else {
                self.fed.clear();
                self.tainted = true;
                return;
            }
        }
        self.fed.extend_from_slice(tokens);
    }

    /// Drop everything from `position` on, for a caller that rewound the KV
    /// there. Longer than the record is a no-op.
    pub(crate) fn rewind_to(&mut self, position: usize) {
        if position < self.fed.len() {
            self.fed.truncate(position);
        }
    }

    /// How many leading tokens of `prompt_ids` this state already holds.
    ///
    /// The LONGEST COMMON PREFIX, not an all-or-nothing match, and that is
    /// the difference between a feature that fires and one that does not.
    /// The record covers the previous prompt AND the tokens generated after
    /// it; the next prompt carries that reply back as re-rendered TEXT, and
    /// re-tokenizing it does not reliably reproduce the ids that were
    /// generated (BPE merges differently against the template's surrounding
    /// bytes). So the record and the new prompt agree on the prompt and
    /// then diverge at the reply. Measured in the real chat REPL: an
    /// all-or-nothing match reused 0 of 33 and 0 of 49 tokens over three
    /// turns, i.e. never fired at all, while the shared prefix was most of
    /// the prompt each time.
    ///
    /// Capped one BELOW the prompt length so at least one token is always
    /// left to feed: `decode` starts by sampling from `logits`, and
    /// prefilling nothing would hand it the previous turn's.
    ///
    /// The caller is responsible for moving the KV cursor back to this
    /// point, and for refusing when it cannot -- see
    /// `RealForwardRunner::try_reuse_prefix`.
    pub(crate) fn common_prefix(&self, prompt_ids: &[TokenId]) -> usize {
        if self.tainted || self.fed.is_empty() || prompt_ids.is_empty() {
            return 0;
        }
        let ceiling = self.fed.len().min(prompt_ids.len() - 1);
        (0..ceiling)
            .take_while(|&i| self.fed[i] == prompt_ids[i])
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn built(ids: &[TokenId]) -> KvPrefix {
        let mut p = KvPrefix::default();
        p.record(ids, 0);
        p
    }

    #[test]
    fn a_strict_prefix_reports_its_whole_length() {
        assert_eq!(built(&[1, 2, 3]).common_prefix(&[1, 2, 3, 4]), 3);
    }

    #[test]
    fn divergence_truncates_rather_than_refusing() {
        // The case the whole design turns on. The record holds the previous
        // prompt AND its reply; the next prompt carries that reply back
        // re-tokenized and diverges there. An all-or-nothing match returns 0
        // here and the feature never fires.
        assert_eq!(
            built(&[1, 2, 3, 9, 9]).common_prefix(&[1, 2, 3, 7, 7, 8]),
            3
        );
        // Divergence at the very first token leaves nothing.
        assert_eq!(built(&[1, 2, 3]).common_prefix(&[9, 2, 3, 4]), 0);
    }

    #[test]
    fn at_least_one_token_is_always_left_to_feed() {
        // `decode` samples from `logits` before generating, so a prefill that
        // fed nothing would emit a token from the PREVIOUS turn's state.
        assert_eq!(built(&[1, 2, 3]).common_prefix(&[1, 2, 3]), 2);
        assert_eq!(built(&[1, 2, 3]).common_prefix(&[1, 2]), 1);
        assert_eq!(built(&[1]).common_prefix(&[1]), 0);
    }

    #[test]
    fn taint_refuses_everything_and_only_clear_lifts_it() {
        let mut p = built(&[1, 2, 3]);
        p.taint();
        p.record(&[4], 3);
        assert_eq!(p.common_prefix(&[1, 2, 3, 4, 5]), 0);
        p.clear();
        p.record(&[1, 2, 3], 0);
        assert_eq!(p.common_prefix(&[1, 2, 3, 4]), 3);
    }

    #[test]
    fn a_gap_in_the_positions_taints_rather_than_claiming_coverage() {
        let mut p = built(&[1, 2, 3]);
        // Position 5 with nothing at 3 or 4: the state holds tokens this
        // record cannot name, so it must describe nothing.
        p.record(&[9], 5);
        assert_eq!(p.common_prefix(&[1, 2, 3, 9, 0, 0, 7]), 0);
    }

    #[test]
    fn re_recording_over_the_same_span_replaces_it() {
        let mut p = built(&[1, 2, 3]);
        p.record(&[7, 8], 1);
        assert_eq!(p.common_prefix(&[1, 7, 8, 4]), 3);
    }

    #[test]
    fn a_rewind_shortens_what_is_reusable() {
        let mut p = built(&[1, 2, 3, 4]);
        p.rewind_to(2);
        assert_eq!(p.common_prefix(&[1, 2, 3, 4]), 2);
        p.rewind_to(99);
        assert_eq!(p.common_prefix(&[1, 2, 3, 4]), 2);
    }

    #[test]
    fn an_empty_record_reuses_nothing() {
        assert_eq!(KvPrefix::default().common_prefix(&[1, 2, 3]), 0);
        assert_eq!(built(&[1, 2, 3]).common_prefix(&[]), 0);
    }
}
