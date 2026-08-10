//! Deadline arithmetic for the decode-loop rate cap (ROADMAP Phase P2).
//!
//! Deliberately does no sleeping and reads no clock of its own: every
//! method takes `now`, so the whole thing is testable at full speed and
//! the one `thread::sleep` in this crate stays visible at its call site in
//! `raw_completion::decode`.
//!
//! Deadlines are absolute (`anchor + n * interval`) rather than a
//! per-token "sleep the remainder", so the error of one slow token does
//! not accumulate across a thousand of them.

use std::time::{Duration, Instant};

/// How often the loop re-reads thermal pressure. At a typical 25-40 tok/s
/// this is under a second, and the probe is one message send against a
/// ~25 ms token, so the sampling cost is noise.
pub(crate) const THERMAL_POLL_TOKENS: usize = 16;

pub(crate) struct Pacer {
    anchor: Instant,
    tokens: u32,
    interval: Option<Duration>,
}

impl Pacer {
    pub(crate) fn new(cap: Option<f64>, now: Instant) -> Self {
        Self {
            anchor: now,
            tokens: 0,
            interval: interval_for(cap),
        }
    }

    /// Adopts a new cap. Re-anchors ONLY when the interval actually
    /// changes: a thermal poll that reports the same level as last time
    /// must not restart the schedule, or the pacer would drift by up to
    /// one interval every poll. When it does change, re-anchoring is what
    /// stops a schedule computed under the old rate from cashing out as
    /// one long stall or one burst of catch-up tokens.
    pub(crate) fn apply_cap(&mut self, cap: Option<f64>, now: Instant) {
        let interval = interval_for(cap);
        if interval == self.interval {
            return;
        }
        self.interval = interval;
        self.anchor = now;
        self.tokens = 0;
    }

    pub(crate) fn note_token(&mut self) {
        self.tokens = self.tokens.saturating_add(1);
    }

    /// How long until this token's slot opens, or `None` when uncapped,
    /// already past due, or due exactly now (a token that took longer than
    /// its slot is simply late; the pacer is a floor on spacing, not a
    /// metronome that can speed the model up). Zero folds into `None` so
    /// the caller never issues a sleep that cannot wait for anything.
    pub(crate) fn due_in(&self, now: Instant) -> Option<Duration> {
        let interval = self.interval?;
        let deadline = self
            .anchor
            .checked_add(interval.checked_mul(self.tokens)?)?;
        let wait = deadline.checked_duration_since(now)?;
        (!wait.is_zero()).then_some(wait)
    }
}

/// Seconds per token from tokens per second. A non-finite or non-positive
/// rate is no cap rather than an error: the parsers reject those before
/// they get here, and a division by zero deep in the decode loop is a
/// worse failure than ignoring a value that cannot mean anything.
fn interval_for(cap: Option<f64>) -> Option<Duration> {
    let cap = cap?;
    if !cap.is_finite() || cap <= 0.0 {
        return None;
    }
    Duration::try_from_secs_f64(1.0 / cap).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deadlines_are_absolute_multiples_of_the_interval() {
        let t0 = Instant::now();
        let mut pacer = Pacer::new(Some(10.0), t0);
        assert_eq!(pacer.due_in(t0), None, "no token noted yet");

        pacer.note_token();
        assert_eq!(pacer.due_in(t0), Some(Duration::from_millis(100)));
        assert_eq!(
            pacer.due_in(t0 + Duration::from_millis(60)),
            Some(Duration::from_millis(40))
        );

        pacer.note_token();
        pacer.note_token();
        // Third token's slot is at 300 ms, so at 250 ms there is 50 left.
        assert_eq!(
            pacer.due_in(t0 + Duration::from_millis(250)),
            Some(Duration::from_millis(50))
        );
        // A run that has fallen behind never sleeps.
        assert_eq!(pacer.due_in(t0 + Duration::from_millis(400)), None);
    }

    #[test]
    fn an_uncapped_pacer_is_never_due() {
        let t0 = Instant::now();
        let mut pacer = Pacer::new(None, t0);
        pacer.note_token();
        assert_eq!(pacer.due_in(t0), None);
        assert_eq!(pacer.due_in(t0 + Duration::from_secs(10)), None);
    }

    #[test]
    fn an_unchanged_cap_does_not_re_anchor_the_schedule() {
        let t0 = Instant::now();
        let mut pacer = Pacer::new(Some(10.0), t0);
        pacer.note_token();
        pacer.note_token();

        pacer.apply_cap(Some(10.0), t0 + Duration::from_millis(150));
        // Still the second token's 200 ms slot, measured from t0.
        assert_eq!(
            pacer.due_in(t0 + Duration::from_millis(150)),
            Some(Duration::from_millis(50))
        );
    }

    #[test]
    fn a_changed_cap_re_anchors_at_the_moment_it_changes() {
        let t0 = Instant::now();
        let mut pacer = Pacer::new(Some(10.0), t0);
        pacer.note_token();
        pacer.note_token();

        let step = t0 + Duration::from_millis(150);
        pacer.apply_cap(Some(5.0), step);
        // Counter restarted, so nothing is owed until the next token.
        assert_eq!(pacer.due_in(step), None);
        pacer.note_token();
        assert_eq!(pacer.due_in(step), Some(Duration::from_millis(200)));
    }

    #[test]
    fn dropping_a_cap_entirely_re_anchors_and_stops_pacing() {
        let t0 = Instant::now();
        let mut pacer = Pacer::new(Some(10.0), t0);
        pacer.note_token();
        pacer.apply_cap(None, t0);
        pacer.note_token();
        assert_eq!(pacer.due_in(t0), None);
    }

    #[test]
    fn unusable_rates_read_as_no_cap_rather_than_dividing_by_zero() {
        assert_eq!(interval_for(None), None);
        assert_eq!(interval_for(Some(0.0)), None);
        assert_eq!(interval_for(Some(-4.0)), None);
        assert_eq!(interval_for(Some(f64::NAN)), None);
        assert_eq!(interval_for(Some(f64::INFINITY)), None);
        assert_eq!(interval_for(Some(4.0)), Some(Duration::from_millis(250)));
    }
}
