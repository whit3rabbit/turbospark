//! GPU performance-state residency from IOReport, for the idle-stall power
//! arm (KERN-05 Step 1a, arm P in `docs/designs/speedup-kern.md`).
//!
//! IOReport's "GPU Stats" group publishes, per channel, cumulative time in
//! each GPU performance state (off, idle, then the P-states). Two readings
//! around an interval give where the GPU spent it. Like `energy`, this needs
//! no sudo, opens `libIOReport.dylib` at run time, and measures the MACHINE:
//! other processes' GPU work counts. A host without the group gets an `Err`
//! reason and the probe records it instead of failing.
//!
//! Unsafe code is the CoreFoundation and IOReport calls below, under the same
//! invariants as `energy`: every `CFTypeRef` is created here or borrowed from
//! a live sample, and each created object is released once.

use std::collections::BTreeMap;
use std::ffi::{c_void, CString};

use crate::energy::{cf_string, read_string};

type CFTypeRef = *const c_void;
const IOREPORT: &str = "/usr/lib/libIOReport.dylib";
const GROUP: &str = "GPU Stats";
const SUBGROUP: &str = "GPU Performance States";

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFDictionaryGetValue(d: CFTypeRef, key: CFTypeRef) -> CFTypeRef;
    fn CFArrayGetCount(a: CFTypeRef) -> isize;
    fn CFArrayGetValueAtIndex(a: CFTypeRef, i: isize) -> CFTypeRef;
    fn CFRelease(v: CFTypeRef);
}

type CopyChannelsFn = unsafe extern "C" fn(CFTypeRef, CFTypeRef, u64, u64, u64) -> CFTypeRef;
type SubscribeFn =
    unsafe extern "C" fn(*const c_void, CFTypeRef, *mut CFTypeRef, u64, CFTypeRef) -> CFTypeRef;
type SamplesFn = unsafe extern "C" fn(CFTypeRef, CFTypeRef, CFTypeRef) -> CFTypeRef;
type StringOfFn = unsafe extern "C" fn(CFTypeRef) -> CFTypeRef;
type StateCountFn = unsafe extern "C" fn(CFTypeRef) -> i32;
type StateNameFn = unsafe extern "C" fn(CFTypeRef, i32) -> CFTypeRef;
type StateResidencyFn = unsafe extern "C" fn(CFTypeRef, i32) -> i64;

struct IoReport {
    copy_channels: CopyChannelsFn,
    subscribe: SubscribeFn,
    samples: SamplesFn,
    channel_name: StringOfFn,
    state_count: StateCountFn,
    state_name: StateNameFn,
    state_residency: StateResidencyFn,
}

impl IoReport {
    fn load() -> Result<Self, String> {
        let path = CString::new(IOREPORT).expect("static path");
        // SAFETY: dlopen with a valid C string; the handle is never closed,
        // so the resolved pointers stay valid for the process.
        let handle = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if handle.is_null() {
            return Err(format!("{IOREPORT}: dlopen failed"));
        }
        let sym = |name: &str| -> Result<*mut c_void, String> {
            let c = CString::new(name).expect("static symbol");
            // SAFETY: dlsym on a live handle with a valid C string.
            let p = unsafe { libc::dlsym(handle, c.as_ptr()) };
            if p.is_null() {
                Err(format!("{IOREPORT}: {name} missing"))
            } else {
                Ok(p)
            }
        };
        // SAFETY: each symbol is the IOReport function with the signature in
        // the struct; transmuting a non-null dlsym result is its contract.
        unsafe {
            Ok(Self {
                copy_channels: std::mem::transmute::<*mut c_void, CopyChannelsFn>(sym(
                    "IOReportCopyChannelsInGroup",
                )?),
                subscribe: std::mem::transmute::<*mut c_void, SubscribeFn>(sym(
                    "IOReportCreateSubscription",
                )?),
                samples: std::mem::transmute::<*mut c_void, SamplesFn>(sym(
                    "IOReportCreateSamples",
                )?),
                channel_name: std::mem::transmute::<*mut c_void, StringOfFn>(sym(
                    "IOReportChannelGetChannelName",
                )?),
                state_count: std::mem::transmute::<*mut c_void, StateCountFn>(sym(
                    "IOReportStateGetCount",
                )?),
                state_name: std::mem::transmute::<*mut c_void, StateNameFn>(sym(
                    "IOReportStateGetNameForIndex",
                )?),
                state_residency: std::mem::transmute::<*mut c_void, StateResidencyFn>(sym(
                    "IOReportStateGetResidency",
                )?),
            })
        }
    }
}

/// Cumulative residency per channel, then per state, in IOReport ticks.
pub type PstateReading = BTreeMap<String, BTreeMap<String, i64>>;

/// Share of an interval each channel spent in each state, states with no
/// time dropped. `None` when the channel or state sets changed or a counter
/// went backwards, which makes the interval unreadable.
pub fn shares(
    before: &PstateReading,
    after: &PstateReading,
) -> Option<BTreeMap<String, BTreeMap<String, f64>>> {
    if before.is_empty() || before.keys().ne(after.keys()) {
        return None;
    }
    let mut out = BTreeMap::new();
    for (channel, b) in before {
        let a = &after[channel];
        if b.keys().ne(a.keys()) {
            return None;
        }
        let mut deltas = BTreeMap::new();
        for (state, &bv) in b {
            let d = a[state].checked_sub(bv).filter(|d| *d >= 0)?;
            deltas.insert(state.clone(), d);
        }
        let total: i64 = deltas.values().sum();
        if total == 0 {
            continue;
        }
        let row = deltas
            .into_iter()
            .filter(|(_, d)| *d > 0)
            .map(|(s, d)| (s, d as f64 / total as f64))
            .collect();
        out.insert(channel.clone(), row);
    }
    Some(out)
}

/// A subscription to the GPU performance-state channels. Not `Send`: the raw
/// pointers keep it on the thread that opened it.
pub struct GpuPstateMeter {
    lib: IoReport,
    subscription: CFTypeRef,
    subscribed: CFTypeRef,
    channels_key: CFTypeRef,
}

impl GpuPstateMeter {
    /// Subscribes and takes one reading to prove at least one channel with
    /// states is there. `Err` names what is missing.
    pub fn open() -> Result<Self, String> {
        let lib = IoReport::load()?;
        let group = cf_string(GROUP);
        let subgroup = cf_string(SUBGROUP);
        // SAFETY: documented call with owned group and subgroup names; the
        // result is owned and released below.
        let channels = unsafe { (lib.copy_channels)(group, subgroup, 0, 0, 0) };
        // SAFETY: releasing the strings this function created.
        unsafe {
            CFRelease(group);
            CFRelease(subgroup);
        }
        if channels.is_null() {
            return Err(format!("IOReport {GROUP}/{SUBGROUP} is unavailable"));
        }
        let mut subscribed: CFTypeRef = std::ptr::null();
        // SAFETY: documented call; `subscribed` receives an owned dictionary.
        let subscription = unsafe {
            (lib.subscribe)(
                std::ptr::null(),
                channels,
                &mut subscribed,
                0,
                std::ptr::null(),
            )
        };
        // SAFETY: the subscription copied what it needs from `channels`.
        unsafe { CFRelease(channels) };
        if subscription.is_null() || subscribed.is_null() {
            return Err("IOReport GPU state subscription failed".into());
        }
        let meter = Self {
            lib,
            subscription,
            subscribed,
            channels_key: cf_string("IOReportChannels"),
        };
        if meter.read()?.is_empty() {
            return Err(format!("IOReport {GROUP}/{SUBGROUP} has no state channels"));
        }
        Ok(meter)
    }

    /// Cumulative residency per channel and state, now.
    pub fn read(&self) -> Result<PstateReading, String> {
        // SAFETY: a live subscription and its channel dictionary; the sample
        // is owned and released at the end.
        let data =
            unsafe { (self.lib.samples)(self.subscription, self.subscribed, std::ptr::null()) };
        if data.is_null() {
            return Err("IOReport GPU state sample failed".into());
        }
        let result = (|| {
            // SAFETY: dictionary lookup on the live sample with an owned key.
            let items = unsafe { CFDictionaryGetValue(data, self.channels_key) };
            if items.is_null() {
                return Err("IOReport GPU state sample has no channels".to_string());
            }
            let mut out = PstateReading::new();
            // SAFETY: array access within [0, count) on the live sample.
            let count = unsafe { CFArrayGetCount(items) };
            for i in 0..count {
                // SAFETY: as above; the item is borrowed from the sample.
                let item = unsafe { CFArrayGetValueAtIndex(items, i) };
                let name = read_string(unsafe { (self.lib.channel_name)(item) })?;
                // SAFETY: a state channel from the subscribed subgroup.
                let states = unsafe { (self.lib.state_count)(item) };
                if states <= 0 {
                    continue;
                }
                let mut row = BTreeMap::new();
                for s in 0..states {
                    // SAFETY: index within [0, states) on a live state channel.
                    let state = read_string(unsafe { (self.lib.state_name)(item, s) })?;
                    let ticks = unsafe { (self.lib.state_residency)(item, s) };
                    *row.entry(state).or_insert(0) += ticks;
                }
                // Several channels can share a name across dies: sum them.
                let entry: &mut BTreeMap<String, i64> = out.entry(name).or_default();
                for (state, ticks) in row {
                    *entry.entry(state).or_insert(0) += ticks;
                }
            }
            Ok(out)
        })();
        // SAFETY: releasing the sample this call created.
        unsafe { CFRelease(data) };
        result
    }
}

impl Drop for GpuPstateMeter {
    fn drop(&mut self) {
        // SAFETY: releasing the objects `open` created, once.
        unsafe {
            CFRelease(self.channels_key);
            CFRelease(self.subscribed);
            CFRelease(self.subscription);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reading(rows: &[(&str, &[(&str, i64)])]) -> PstateReading {
        rows.iter()
            .map(|(c, s)| {
                (
                    c.to_string(),
                    s.iter().map(|(n, v)| (n.to_string(), *v)).collect(),
                )
            })
            .collect()
    }

    #[test]
    fn shares_split_the_interval_and_drop_empty_states() {
        let before = reading(&[("GPUPH", &[("OFF", 10), ("IDLE", 0), ("P1", 5)])]);
        let after = reading(&[("GPUPH", &[("OFF", 40), ("IDLE", 0), ("P1", 15)])]);
        let s = shares(&before, &after).expect("readable");
        assert_eq!(s["GPUPH"].len(), 2);
        assert!((s["GPUPH"]["OFF"] - 0.75).abs() < 1e-12);
        assert!((s["GPUPH"]["P1"] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn shares_refuse_resets_and_changed_sets() {
        let before = reading(&[("GPUPH", &[("OFF", 10)])]);
        assert!(shares(&before, &reading(&[("GPUPH", &[("OFF", 9)])])).is_none());
        assert!(shares(&before, &reading(&[("GPUPH", &[("P1", 20)])])).is_none());
        assert!(shares(&before, &reading(&[("OTHER", &[("OFF", 20)])])).is_none());
        assert!(shares(&PstateReading::new(), &PstateReading::new()).is_none());
    }

    #[test]
    fn the_meter_opens_or_says_why() {
        // Host-dependent: either a reading with states, or a reason.
        match GpuPstateMeter::open() {
            Ok(meter) => {
                let r = meter.read().expect("second reading");
                assert!(r.values().any(|s| !s.is_empty()), "{r:?}");
                eprintln!("gpu pstate channels: {:?}", r.keys().collect::<Vec<_>>());
                eprintln!(
                    "states: {:?}",
                    r.values().next().map(|s| s.keys().collect::<Vec<_>>())
                );
            }
            Err(reason) => assert!(!reason.is_empty()),
        }
    }
}
