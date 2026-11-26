//! What the lab reads out of the speculative loop (`crates/lab`): per round
//! timings and expert traffic (LAB-25, LAB-13), the margin behind every
//! committed token (LAB-03), and a SYNTHETIC acceptance mode (LAB-23).
//!
//! Installed per thread by a direct call, never by an environment variable,
//! so nothing a server or app process is started with can turn it on. With no
//! probe installed every hook is one thread-local check.
//!
//! **Synthetic acceptance produces wrong text by design.** It commits the
//! drafter's proposal whether or not the model agrees, for a random prefix of
//! each round, so the loop's rollback and commit costs can be measured at a
//! chosen acceptance rate. The state stays consistent (the model processed
//! exactly the committed ids); the ids are not the model's answer.

use std::cell::RefCell;

/// One speculative round.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Round {
    /// Seconds since the probe was installed, at the round's end.
    pub at_s: f64,
    /// Tokens proposed, and accepted.
    pub width: usize,
    pub accepted: usize,
    pub copy: bool,
    pub verify_ms: f64,
    /// Expert reads requested during the verify, and their bytes. Zero on a
    /// dense model.
    pub expert_requests: u64,
    pub expert_bytes: u64,
}

/// What a probed run recorded.
#[derive(Debug, Clone, Default)]
pub struct Record {
    pub rounds: Vec<Round>,
    /// For each committed token in order, how many logits the row that chose
    /// it led by (top-1 minus top-2). A token a synthetic round forced has
    /// none.
    pub margins: Vec<Option<f32>>,
    pub synthetic: bool,
}

struct State {
    record: Record,
    started: std::time::Instant,
    /// Chance each proposal is accepted, synthetic mode only.
    synthetic: Option<f64>,
    rng: u64,
}

thread_local! {
    static PROBE: RefCell<Option<State>> = const { RefCell::new(None) };
}

/// Start recording on this thread. `synthetic` is the chance each proposal is
/// accepted, in [0, 1].
pub fn install(synthetic: Option<f64>) {
    PROBE.with_borrow_mut(|p| {
        *p = Some(State {
            record: Record {
                synthetic: synthetic.is_some(),
                ..Record::default()
            },
            started: std::time::Instant::now(),
            synthetic: synthetic.map(|p| p.clamp(0.0, 1.0)),
            rng: 0x5eed_5bec_acce_9702,
        })
    });
}

/// Stop recording and hand back what was seen. `None` if nothing installed.
pub fn take() -> Option<Record> {
    PROBE.with_borrow_mut(|p| p.take().map(|s| s.record))
}

pub(crate) fn active() -> bool {
    PROBE.with_borrow(Option::is_some)
}

/// `None`: decide by the model as usual. `Some(true)`: treat this proposal as
/// accepted. `Some(false)`: reject it.
pub(crate) fn synthetic_accept() -> Option<bool> {
    PROBE.with_borrow_mut(|p| {
        let state = p.as_mut()?;
        let chance = state.synthetic?;
        state.rng ^= state.rng << 13;
        state.rng ^= state.rng >> 7;
        state.rng ^= state.rng << 17;
        Some(((state.rng >> 11) as f64 / (1u64 << 53) as f64) < chance)
    })
}

pub(crate) fn committed(margin: Option<f32>) {
    PROBE.with_borrow_mut(|p| {
        if let Some(state) = p {
            state.record.margins.push(margin);
        }
    });
}

pub(crate) fn round(mut round: Round) {
    PROBE.with_borrow_mut(|p| {
        if let Some(state) = p {
            round.at_s = state.started.elapsed().as_secs_f64();
            state.record.rounds.push(round);
        }
    });
}

/// Top-1 minus top-2 of a row. `None` for a row too short to have two.
pub fn margin(row: &[foundation::LogitValue]) -> Option<f32> {
    let mut best = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for v in row {
        let x = v.to_f32();
        if x > best {
            second = best;
            best = x;
        } else if x > second {
            second = x;
        }
    }
    (row.len() >= 2).then_some(best - second)
}

#[cfg(test)]
mod tests {
    use super::*;
    use foundation::LogitValue;

    #[test]
    fn nothing_is_recorded_or_overridden_without_a_probe() {
        assert!(!active());
        assert_eq!(synthetic_accept(), None);
        committed(Some(1.0));
        round(Round::default());
        assert!(take().is_none());
    }

    #[test]
    fn a_recording_probe_never_overrides_acceptance() {
        install(None);
        assert_eq!(synthetic_accept(), None);
        committed(Some(2.0));
        committed(None);
        round(Round {
            width: 3,
            accepted: 1,
            ..Round::default()
        });
        let r = take().unwrap();
        assert_eq!(r.margins, [Some(2.0), None]);
        assert_eq!(
            (r.rounds.len(), r.rounds[0].width, r.synthetic),
            (1, 3, false)
        );
        assert!(!active());
    }

    #[test]
    fn synthetic_acceptance_follows_its_rate_and_repeats() {
        let draw = |p: f64| {
            install(Some(p));
            let v: Vec<bool> = (0..4000).map(|_| synthetic_accept().unwrap()).collect();
            take();
            v
        };
        let a = draw(0.6);
        let rate = a.iter().filter(|&&x| x).count() as f64 / 4000.0;
        assert!((rate - 0.6).abs() < 0.03, "{rate}");
        assert_eq!(a, draw(0.6));
        assert!(draw(0.0).iter().all(|&x| !x));
        assert!(draw(1.0).iter().all(|&x| x));
    }

    #[test]
    fn the_margin_is_the_lead_of_the_best_logit() {
        let row: Vec<LogitValue> = [1.0f32, 5.0, 3.5]
            .iter()
            .map(|&x| LogitValue::from_f32(x))
            .collect();
        assert_eq!(margin(&row), Some(1.5));
        assert_eq!(margin(&row[..1]), None);
    }
}
