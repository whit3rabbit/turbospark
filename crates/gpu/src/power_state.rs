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

// The allow is for objc's `sel_impl!`, whose expansion carries a
// `cfg(feature = "cargo-clippy")` this crate does not declare.
#[allow(unexpected_cfgs)]
/// `NSProcessInfo.processInfo.thermalState`, raw: 0 nominal, 1 fair,
/// 2 serious, 3 critical. Returned unmapped so the meaning of each level
/// is decided in one place (`runtime::ThermalLevel::from_raw`) rather than
/// twice.
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

// See `thermal_state_raw` for the allow.
#[allow(unexpected_cfgs)]
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
pub fn physical_memory() -> u64 {
    // SAFETY: same singleton as above; `physicalMemory` is a documented
    // no-argument `unsigned long long` property on it.
    #[allow(unsafe_code)]
    unsafe {
        let info: *mut Object = msg_send![class!(NSProcessInfo), processInfo];
        msg_send![info, physicalMemory]
    }
}

/// `kern.memorystatus_vm_pressure_level`, raw: 1 normal, 2 warn, 4 critical.
/// Returned unmapped for [`thermal_state_raw`]'s reason -- the meaning of a
/// level is decided once, in `runtime::MemoryPressure::from_raw`. `0` when
/// the sysctl is unavailable, which that mapping reads as "no reading" and
/// never as "no pressure".
///
/// **THIS IS THE KERNEL'S OWN VERDICT AND NOT A FREE-PAGE COUNT, WHICH IS
/// WHAT MAKES IT USABLE HERE.** [`physical_memory`]'s doc declines
/// `host_statistics64`'s free pages on the grounds that a number moving
/// second to second makes two runs incomparable -- correct, and about a
/// BUDGET, which has to be stable across opens or no two footprints can be
/// compared. A watcher is the opposite job: it exists to see the thing that
/// moves. Reading a three-valued OS verdict rather than a page count keeps
/// the two from being the same instrument, and nothing here budgets from it.
///
/// There is no `NSProcessInfo` property for this the way there is for
/// thermal state, so this is a sysctl where its neighbours are message
/// sends. `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` is the push form and is not
/// used: the decode loop already polls on a token boundary, and a callback
/// would need a channel and a thread to reach it.
pub fn memory_pressure_raw() -> i64 {
    let mut level: libc::c_int = 0;
    let mut size = std::mem::size_of::<libc::c_int>();
    // SAFETY: `kern.memorystatus_vm_pressure_level` is a documented read-only
    // integer sysctl. The name is a NUL-terminated literal, the output
    // pointer is a live local of exactly `size` bytes, and no new value is
    // written (null pointer, zero length). A non-zero return leaves `level`
    // at its initialized 0, which the mapping reads as "no reading".
    #[allow(unsafe_code)]
    let rc = unsafe {
        libc::sysctlbyname(
            c"kern.memorystatus_vm_pressure_level".as_ptr(),
            (&mut level as *mut libc::c_int).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 {
        level as i64
    } else {
        0
    }
}

// See `thermal_state_raw` for the allow.
#[allow(unexpected_cfgs)]
/// `NSProcessInfo.processInfo.isLowPowerModeEnabled`.
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
