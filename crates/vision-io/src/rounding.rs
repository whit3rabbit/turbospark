//! Python's `round()` convention.
//!
//! Adapted from the sconce vision crate (see `NOTICE`).

/// Round half to even ("banker's rounding"), which is what Python's built-in
/// `round()` does and what `f64::round` does NOT.
///
/// `smart_resize` spells its rounding `round(height / factor) * factor`, and
/// exact `.5` inputs are ordinary rather than exotic: any edge that is an odd
/// multiple of half the factor hits one. At factor 32, a height of 48 gives
/// `1.5` -- Python rounds to 2 (96 pixels, 3 merged rows), `f64::round` rounds
/// to 2 as well, but a height of 80 gives `2.5` where Python says 2 (64
/// pixels) and `f64::round` says 3 (96). That is a whole extra patch row per
/// axis, which changes the token count and every downstream position table.
pub fn round_half_to_even(x: f64) -> f64 {
    let rounded = x.round();
    if (x - x.trunc()).abs() == 0.5 && rounded % 2.0 != 0.0 {
        rounded - x.signum()
    } else {
        rounded
    }
}
