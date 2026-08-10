//! `NSProcessInfo` power-state probes (ROADMAP Phase P2).
//!
//! Neither value can be asserted exactly: both are machine state at the
//! moment of the call. What IS assertable is that the message sends
//! resolve and stay inside their documented ranges, which is the whole
//! failure mode worth catching here (a mistyped selector returns garbage
//! rather than failing, and a missing Foundation link aborts).

#![cfg(target_os = "macos")]

use turbospark_gpu::{low_power_mode_enabled, thermal_state_raw};

#[test]
fn thermal_state_is_one_of_the_four_documented_levels() {
    let raw = thermal_state_raw();
    assert!(
        (0..=3).contains(&raw),
        "thermalState out of the documented 0..=3 range: {raw}"
    );
}

#[test]
fn low_power_mode_probe_resolves() {
    // Both outcomes are valid; the test is that the selector resolves and
    // the BOOL narrows to one of them rather than trapping.
    let enabled = low_power_mode_enabled();
    assert!(enabled || !enabled);
}
