//! macOS power-state probes: `NSProcessInfo`'s `thermalState` and
//! `isLowPowerModeEnabled`. Both back ROADMAP Phase P2's rate control,
//! which lives in `crates/runtime` -- that crate is `#![forbid(unsafe_code)]`
//! and portable, so the two message sends live here, in the crate that
//! already reaches Objective-C, and cross the boundary as a plain integer
//! and a `bool`.
//!
//! Port-local: the Swift engine has no equivalent, so there is nothing to
//! mirror and the only contract is Apple's documented one.

use metal::objc::runtime::{Object, BOOL, NO};
use metal::objc::{class, msg_send, sel, sel_impl};

// `NSProcessInfo` is Foundation's. This process links Metal, which does not
// itself guarantee Foundation is on the link line, and `class!` PANICS on a
// class it cannot find. Naming the framework here turns a would-be runtime
// abort in the middle of a decode into a link-time guarantee.
#[link(name = "Foundation", kind = "framework")]
extern "C" {}

/// `NSProcessInfo.processInfo.thermalState`, raw: 0 nominal, 1 fair,
/// 2 serious, 3 critical. Returned unmapped so the meaning of each level
/// is decided in one place (`runtime::ThermalLevel::from_raw`) rather than
/// twice.
// The allow is for objc's `sel_impl!`, whose expansion carries a
// `cfg(feature = "cargo-clippy")` this crate does not declare.
#[allow(unexpected_cfgs)]
pub fn thermal_state_raw() -> i64 {
    // SAFETY: `+[NSProcessInfo processInfo]` returns the process-wide
    // singleton, which is neither autoreleased nor null once Foundation is
    // linked, and `thermalState` is a documented no-argument `NSInteger`
    // property on it.
    #[allow(unsafe_code)]
    unsafe {
        let info: *mut Object = msg_send![class!(NSProcessInfo), processInfo];
        msg_send![info, thermalState]
    }
}

/// `NSProcessInfo.processInfo.physicalMemory`, in bytes.
///
/// Backs the routed-expert cache's automatic slot sizing, which lives in
/// `crates/runtime` for the same reason the two probes above do: that crate
/// is `#![forbid(unsafe_code)]`, so the message send belongs here and crosses
/// the boundary as a plain integer.
///
/// This is INSTALLED memory, not free memory, and the caller is expected to
/// know that -- `runtime`'s resolver subtracts the install's own resident
/// bytes and a fixed reserve rather than treating this as headroom. The
/// honest alternative would be `host_statistics64`'s free page count, which
/// is a poor budget for a different reason: it moves second to second with
/// whatever else the machine is doing, so the same install would pick a
/// different slot count on each open and no two runs would be comparable.
/// A stable over-estimate that the caller discounts beats an accurate number
/// that is never the same twice.
// See `thermal_state_raw` for the allow.
#[allow(unexpected_cfgs)]
pub fn physical_memory() -> u64 {
    // SAFETY: same singleton as above; `physicalMemory` is a documented
    // no-argument `unsigned long long` property on it.
    #[allow(unsafe_code)]
    unsafe {
        let info: *mut Object = msg_send![class!(NSProcessInfo), processInfo];
        msg_send![info, physicalMemory]
    }
}

/// `NSProcessInfo.processInfo.isLowPowerModeEnabled`.
// See `thermal_state_raw` for the allow.
#[allow(unexpected_cfgs)]
pub fn low_power_mode_enabled() -> bool {
    // SAFETY: same singleton as above; `isLowPowerModeEnabled` is a
    // documented no-argument `BOOL` property on it.
    #[allow(unsafe_code)]
    unsafe {
        let info: *mut Object = msg_send![class!(NSProcessInfo), processInfo];
        let flag: BOOL = msg_send![info, isLowPowerModeEnabled];
        flag != NO
    }
}
