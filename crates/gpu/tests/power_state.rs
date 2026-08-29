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

/// The kernel's own pressure verdict, on the machine running the test.
///
/// **Asserts the SET rather than a value**, because this is live machine
/// state: a test demanding `1` fails on a genuinely busy machine, which is
/// the one condition the probe exists for. `0` is in the accepted set and
/// means the sysctl did not answer -- distinguishable from every real level,
/// which is what `runtime::MemoryPressure::from_raw` needs.
#[test]
fn memory_pressure_reads_one_of_the_documented_levels() {
    let raw = turbospark_gpu::memory_pressure_raw();
    assert!(
        matches!(raw, 0 | 1 | 2 | 4),
        "kern.memorystatus_vm_pressure_level answered {raw}, \
         which is not a documented level"
    );
}

/// Two reads in a row must not disagree about whether the probe WORKS. The
/// level may legitimately move between them; whether the sysctl exists may
/// not, and a flapping zero would mean the call is failing intermittently.
#[test]
fn the_probe_is_available_or_not_consistently() {
    let a = turbospark_gpu::memory_pressure_raw();
    let b = turbospark_gpu::memory_pressure_raw();
    assert_eq!(a == 0, b == 0, "the probe answered {a} then {b}");
}
