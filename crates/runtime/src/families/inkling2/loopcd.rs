//! LoopCD (arXiv 2610.02185) for the looped core: an EXPERIMENT, off unless
//! `TURBOSPARK_INKLING2_LOOPCD=hidden|adaptive` is set at open.
//!
//! The state of core pass `b` is its core block states and its output. Both
//! variants push the answer away from the one an early pass would have given:
//!
//! - `hidden`: before the final layers, `s' = s + w (s - s_b)`. No extra
//!   layers and no extra KV.
//! - `adaptive`: the final layers and head also run from `s_b`, on their own
//!   KV and conv slots, and the softcapped logits become
//!   `z + w (1 - (p1 - p2)) (z - z_b)`, `p1`/`p2` the top two probabilities.
//!
//! `TURBOSPARK_INKLING2_LOOPCD_OMEGA` is `w` and `..._PASS` is `b` (1-based,
//! default 1). `b` equal to the last pass makes every difference zero, so the
//! output must be bit-identical to LoopCD off: that is the plumbing gate.
//!
//! Single-token decode only. The multi-row forward (chunked prefill, verify,
//! DSpark) has no LoopCD form and is refused while it is on. The combines run
//! on the host, which costs a GPU sync; a Metal kernel is not worth writing
//! before the quality result is known.

use half::f16;

use crate::real_forward::RealForwardError;

const ENV_MODE: &str = "TURBOSPARK_INKLING2_LOOPCD";
const ENV_OMEGA: &str = "TURBOSPARK_INKLING2_LOOPCD_OMEGA";
const ENV_PASS: &str = "TURBOSPARK_INKLING2_LOOPCD_PASS";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    Hidden,
    Adaptive,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Policy {
    pub(crate) mode: Mode,
    pub(crate) omega: f32,
    /// 1-based core pass whose state is the reference.
    pub(crate) pass: u32,
}

impl Policy {
    pub(crate) fn from_env(passes: u32) -> Result<Option<Self>, RealForwardError> {
        let var = |name: &str| std::env::var(name).ok();
        Self::parse(
            var(ENV_MODE).as_deref(),
            var(ENV_OMEGA).as_deref(),
            var(ENV_PASS).as_deref(),
            passes,
        )
        .map_err(RealForwardError::Unsupported)
    }

    /// A value that cannot be honoured is refused, never ignored: an arm of an
    /// A/B that silently ran with LoopCD off would read as "no effect".
    fn parse(
        mode: Option<&str>,
        omega: Option<&str>,
        pass: Option<&str>,
        passes: u32,
    ) -> Result<Option<Self>, String> {
        let mode = match mode {
            None | Some("") | Some("off") => return Ok(None),
            Some("hidden") => Mode::Hidden,
            Some("adaptive") => Mode::Adaptive,
            Some(other) => return Err(format!("{ENV_MODE}={other}: use off, hidden or adaptive")),
        };
        let omega = match omega {
            None => match mode {
                Mode::Hidden => 0.3,
                Mode::Adaptive => 0.5,
            },
            Some(v) => v
                .parse::<f32>()
                .ok()
                .filter(|w| w.is_finite() && *w >= 0.0)
                .ok_or_else(|| format!("{ENV_OMEGA}={v}: use a number of 0 or more"))?,
        };
        let pass = match pass {
            None => 1,
            Some(v) => v
                .parse::<u32>()
                .ok()
                .filter(|b| (1..=passes).contains(b))
                .ok_or_else(|| format!("{ENV_PASS}={v}: use a core pass from 1 to {passes}"))?,
        };
        Ok(Some(Self { mode, omega, pass }))
    }
}

/// `s + w (s - r)` over `states`, in place. FP32 arithmetic, FP16 storage.
pub(crate) fn extrapolate(states: &mut [f16], reference: &[f16], omega: f32) {
    assert_eq!(states.len(), reference.len());
    for (s, r) in states.iter_mut().zip(reference) {
        let v = s.to_f32();
        *s = f16::from_f32(omega.mul_add(v - r.to_f32(), v));
    }
}

/// `z + w_t (z - z_ref)` in place over one softcapped logit row, with
/// `w_t = w (1 - (p1 - p2))`: a step the model is already sure of barely moves.
pub(crate) fn contrast_adaptive(logits: &mut [f16], reference: &[f16], omega: f32) {
    assert_eq!(logits.len(), reference.len());
    let (mut top, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for z in logits.iter().map(|z| z.to_f32()) {
        if z > top {
            (top, second) = (z, top);
        } else if z > second {
            second = z;
        }
    }
    // Shifted by the maximum so no exponential overflows.
    let sum: f32 = logits.iter().map(|z| (z.to_f32() - top).exp()).sum();
    let margin = (1.0 - (second - top).exp()) / sum;
    let w = omega * (1.0 - margin);
    for (z, r) in logits.iter_mut().zip(reference) {
        let v = z.to_f32();
        *z = f16::from_f32(w.mul_add(v - r.to_f32(), v));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn half(v: &[f32]) -> Vec<f16> {
        v.iter().map(|&x| f16::from_f32(x)).collect()
    }

    #[test]
    fn parse_defaults_and_refusals() {
        let p = |m, w, b| Policy::parse(m, w, b, 3);
        assert_eq!(p(None, Some("9"), None).unwrap(), None);
        assert_eq!(p(Some("off"), None, None).unwrap(), None);
        let hidden = p(Some("hidden"), None, None).unwrap().unwrap();
        assert_eq!((hidden.omega, hidden.pass), (0.3, 1));
        let adaptive = p(Some("adaptive"), None, Some("3")).unwrap().unwrap();
        assert_eq!((adaptive.omega, adaptive.pass), (0.5, 3));
        for bad in [
            p(Some("logits"), None, None),
            p(Some("hidden"), Some("-1"), None),
            p(Some("hidden"), Some("nan"), None),
            p(Some("hidden"), None, Some("0")),
            p(Some("hidden"), None, Some("4")),
        ] {
            assert!(bad.is_err(), "{bad:?}");
        }
    }

    #[test]
    fn extrapolate_pushes_away_from_the_reference() {
        let mut s = half(&[2.0, -1.0, 4.0]);
        extrapolate(&mut s, &half(&[1.0, 1.0, 4.0]), 0.5);
        assert_eq!(s, half(&[2.5, -2.0, 4.0]));
    }

    #[test]
    fn adaptive_strength_follows_the_top_two_margin() {
        // A tie has no margin, so the full strength applies.
        let mut z = half(&[1.0, 1.0]);
        contrast_adaptive(&mut z, &half(&[0.0, 2.0]), 0.5);
        assert_eq!(z, half(&[1.5, 0.5]));
        // p = [0.75, 0.25] for logits [ln 3, 0]: margin 0.5, strength halved.
        let mut z = half(&[3f32.ln(), 0.0]);
        let top = z[0].to_f32();
        contrast_adaptive(&mut z, &half(&[0.0, 0.0]), 1.0);
        assert!((z[0].to_f32() - 1.5 * top).abs() < 2e-3, "{z:?}");
        assert_eq!(z[1].to_f32(), 0.0);
        // A certain step barely moves.
        let mut z = half(&[30.0, -30.0]);
        contrast_adaptive(&mut z, &half(&[0.0, 0.0]), 0.5);
        assert_eq!(z, half(&[30.0, -30.0]));
    }

    #[test]
    fn a_reference_equal_to_the_state_changes_nothing() {
        let start = half(&[0.125, -7.5, 29.0, 0.0]);
        let mut z = start.clone();
        contrast_adaptive(&mut z, &start, 0.5);
        extrapolate(&mut z, &start, 0.3);
        assert_eq!(z, start);
    }
}
