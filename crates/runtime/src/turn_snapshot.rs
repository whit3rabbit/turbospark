//! A recurrent-state snapshot near the end of each prompt, so the NEXT turn
//! of a chat can rewind a gated-DeltaNet model to it instead of prefilling
//! the whole conversation again (`TURBOSPARK_GDN_TURN_SNAPSHOT=1`, default
//! off).
//!
//! WHY IT IS NEEDED. Prefix reuse (`crate::kv_prefix`) moves the KV cursor
//! back to where the new prompt stops agreeing with what was fed. A GDN
//! layer cannot move back: its state folds history through a
//! non-invertible update, so `try_reuse_prefix` refuses every rewind on the
//! qwen flows and reuse fires only on a pure continuation. A chat client
//! that resends the transcript without the model's reasoning (most
//! OpenAI-style clients) re-renders the previous reply differently from
//! what was generated, and on Qwen3.8's template the agreement ends one
//! token before the previous prompt's end. Every such turn then prefilled
//! the whole conversation.
//!
//! THE FIX. Split the prompt's prefill [`TURN_SNAPSHOT_MARGIN`] tokens
//! before its end and keep one full [`RollbackPoint`] there. The next
//! turn's prompt agrees with the fed ids at least that far whenever only
//! the reply and the generation-prompt tail were re-rendered, so
//! `try_reuse_prefix` restores the point, rewinds the KV (and an MTP
//! drafter) to it, and the loop prefills the re-rendered reply and the new
//! message only.
//!
//! COST. One GDN mirror held between turns (about 147 MiB on the 27B), not
//! counted by `context_policy`, which is one reason this is opt-in. The
//! split also moves one chunk boundary of every prompt, so the prefill's
//! last chunk runs as two passes.
//!
//! WHAT REFUSES (each costs a full prefill, nothing else): every family but
//! the dense qwen flow (`qwen4_exp` has no checkpoint support, and the MoE
//! flow has no real-install gate yet), a DFlash2 drafter (its context write reads
//! per-pass captures this does not restore), an MTP drafter that was not
//! primed up to the point (the plain loops do not prime it), an image
//! prompt, a session swap, and a match that ends at or before the point.

use crate::real_forward::{RealForwardRunner, RollbackPoint};
use foundation::TokenId;

/// How many tokens before the prompt's end the snapshot is taken. The
/// agreement measured on Qwen3.8's template ends 1 token early (the
/// generation prompt's `<think>\n` re-tokenizes with the next `\n`), and an
/// MTP drafter needs the match to run strictly PAST the point (its row
/// `S - 1` was primed with the token at `S`). 8 leaves slack for templates
/// that re-render a little more of the tail, at a cost of 8 re-prefilled
/// tokens a turn.
pub(crate) const TURN_SNAPSHOT_MARGIN: usize = 8;

/// `TURBOSPARK_GDN_TURN_SNAPSHOT`, default off, read once.
pub(crate) fn turn_snapshot_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TURBOSPARK_GDN_TURN_SNAPSHOT").as_deref() == Ok("1"))
}

/// Where a prompt of `prompt_len` tokens, of which `reused` are already in
/// the state, takes its snapshot: strictly inside the part still to be
/// prefilled, so the split always lands between two prefill passes.
pub(crate) fn cut_position(prompt_len: usize, reused: usize) -> Option<usize> {
    let cut = prompt_len.checked_sub(TURN_SNAPSHOT_MARGIN)?;
    (cut > reused).then_some(cut)
}

/// The kept point and what it needs to be restorable.
pub(crate) struct TurnSnapshot {
    pub(crate) point: RollbackPoint,
}

impl RealForwardRunner {
    /// Whether this runner takes turn snapshots at all. The DENSE qwen flow
    /// only: it is the one the real-install gate covers
    /// (`crates/bench/tests/turn_snapshot_gate.rs`), and a routed MoE flow
    /// needs its own before it shares the path.
    fn turn_snapshot_applies(&self) -> bool {
        turn_snapshot_enabled()
            && self.prefix_reuse_enabled
            && self.real_qwen.as_ref().is_some_and(|q| q.dense)
            && self.real_qwen4.is_none()
            && self.real_dflash.is_none()
            && self.prompt_vision.is_none()
    }

    /// See [`crate::LogitProducer::turn_snapshot_position`].
    pub(crate) fn turn_snapshot_cut(&self, prompt_ids: &[TokenId], reused: usize) -> Option<usize> {
        if !self.turn_snapshot_applies() {
            return None;
        }
        cut_position(prompt_ids.len(), reused)
    }

    /// See [`crate::LogitProducer::take_turn_snapshot`]. Skips, rather than
    /// takes a point it could not restore, when the state does not cover
    /// exactly `position` or an MTP drafter was not primed up to it.
    pub(crate) fn take_turn_snapshot_at(&mut self, position: usize) {
        self.turn_snapshot = None;
        if !self.turn_snapshot_applies()
            || self.kv.position() != position
            || self.kv_prefix.untainted_len() != Some(position)
            || self
                .real_mtp
                .as_ref()
                .is_some_and(|m| m.kv_position() != position)
        {
            return;
        }
        self.wait_pending_gdn_fold();
        self.turn_snapshot = Some(TurnSnapshot {
            point: self.checkpoint(),
        });
    }

    /// Rewind to the kept point when the prompt agrees with the fed ids for
    /// `keep` tokens. Returns the position now covered, or `None` when the
    /// point cannot serve this prompt, which `try_reuse_prefix` answers with
    /// its own GDN refusal and therefore a reset.
    pub(crate) fn try_restore_turn_snapshot(&mut self, keep: usize) -> Option<usize> {
        if !self.turn_snapshot_applies() {
            return None;
        }
        let at = self.turn_snapshot.as_ref()?.point.position();
        // The drafter's row `at - 1` holds the token at `at`, so with a
        // drafter the match must run past the point, not just reach it.
        let drafter = self.real_mtp.as_ref();
        let needed = if drafter.is_some() { at + 1 } else { at };
        let cursor = self.kv.position();
        if keep < needed || at > cursor || cursor - at > self.kv.max_safe_rewind() {
            return None;
        }
        if drafter.is_some_and(|m| m.kv_position() < at) {
            return None;
        }
        self.wait_pending_gdn_fold();
        let snapshot = self.turn_snapshot.take()?;
        self.rollback(&snapshot.point);
        if self.real_mtp.is_some() && self.mtp_rewind_to(at).is_err() {
            // Unreachable after the cursor check above. The target already
            // moved, so this relies on the caller's refusal and reset rather
            // than drafting over rows nobody wrote.
            return None;
        }
        // Still the state at `at`: later turns of the same conversation can
        // come back to it until a newer prompt replaces it.
        self.turn_snapshot = Some(snapshot);
        Some(at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cut_is_the_margin_before_the_end() {
        assert_eq!(cut_position(100, 0), Some(100 - TURN_SNAPSHOT_MARGIN));
    }

    #[test]
    fn no_cut_inside_the_reused_part_or_a_short_prompt() {
        assert_eq!(cut_position(TURN_SNAPSHOT_MARGIN, 0), None);
        assert_eq!(cut_position(TURN_SNAPSHOT_MARGIN - 1, 0), None);
        assert_eq!(cut_position(100, 100 - TURN_SNAPSHOT_MARGIN), None);
        assert_eq!(
            cut_position(100, 99 - TURN_SNAPSHOT_MARGIN),
            Some(100 - TURN_SNAPSHOT_MARGIN)
        );
    }
}
