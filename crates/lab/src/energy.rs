//! Modeled whole-machine energy from IOReport's "Energy Model" counters.
//!
//! The mechanism behind `main-npu`'s `joules.py` (docs/MOE_PRIOR_ART.md,
//! "The Neural Engine work"): IOReport publishes cumulative joule counters
//! per SoC block (CPU, GPU, ANE, DRAM, ...) that any process reads without
//! sudo. They model the chip, not the wall: no supply losses, no fans, and
//! every process on the machine counts. A reading is comparable only with
//! another reading in the same experiment under the same host rules, never
//! with `docs/POWER_BASELINE.md` (powermetrics, wall side). Every record
//! that carries these joules says [`ENERGY_MODEL`].
//!
//! `libIOReport.dylib` lives in the dyld shared cache (no file on disk), so
//! it is opened with `dlopen` at runtime. A host without it, or without the
//! Energy Model group, gets an `Err` reason and the run proceeds without
//! energy columns. Nothing here is a cargo dependency.
//!
//! All unsafe code is the CoreFoundation and IOReport calls below. The
//! invariants: every `CFTypeRef` passed in is one this module created or
//! borrowed from a live sample dictionary, strings are read into a bounded
//! buffer, and each created object is released once.

use std::collections::BTreeMap;
use std::ffi::{c_char, c_void, CStr, CString};

use serde::{Deserialize, Serialize};

/// The `energy_model` value on every payload with IOReport joules.
pub const ENERGY_MODEL: &str = "ioreport_whole_machine";

type CFTypeRef = *const c_void;
const UTF8: u32 = 0x0800_0100;
const IOREPORT: &str = "/usr/lib/libIOReport.dylib";

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFStringCreateWithCString(alloc: *const c_void, s: *const c_char, enc: u32) -> CFTypeRef;
    fn CFStringGetCString(s: CFTypeRef, buf: *mut c_char, len: isize, enc: u32) -> u8;
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
type IntegerValueFn = unsafe extern "C" fn(CFTypeRef, i32) -> i64;

/// The IOReport entry points, resolved at runtime.
struct IoReport {
    copy_channels: CopyChannelsFn,
    subscribe: SubscribeFn,
    samples: SamplesFn,
    channel_name: StringOfFn,
    unit_label: StringOfFn,
    integer_value: IntegerValueFn,
}

impl IoReport {
    fn load() -> Result<Self, String> {
        let path = CString::new(IOREPORT).expect("static path");
        // SAFETY: dlopen with a valid C string; the handle is kept for the
        // life of the process (never closed), so the resolved function
        // pointers stay valid.
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
        // SAFETY: each symbol is the documented IOReport function with the
        // signature written in the struct; transmuting a non-null dlsym
        // result to that signature is the dlsym contract.
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
                unit_label: std::mem::transmute::<*mut c_void, StringOfFn>(sym(
                    "IOReportChannelGetUnitLabel",
                )?),
                integer_value: std::mem::transmute::<*mut c_void, IntegerValueFn>(sym(
                    "IOReportSimpleGetIntegerValue",
                )?),
            })
        }
    }
}

fn cf_string(s: &str) -> CFTypeRef {
    let c = CString::new(s).expect("no interior nul");
    // SAFETY: a valid C string and encoding; the result is owned (Create
    // rule) and released by the caller.
    unsafe { CFStringCreateWithCString(std::ptr::null(), c.as_ptr(), UTF8) }
}

fn read_string(s: CFTypeRef) -> Result<String, String> {
    if s.is_null() {
        return Err("IOReport returned a null string".into());
    }
    let mut buf = vec![0 as c_char; 1024];
    // SAFETY: a live CFString and a buffer of the stated length.
    let ok = unsafe { CFStringGetCString(s, buf.as_mut_ptr(), buf.len() as isize, UTF8) };
    if ok == 0 {
        return Err("IOReport string does not fit 1024 bytes".into());
    }
    // SAFETY: CFStringGetCString wrote a nul-terminated string into `buf`.
    Ok(unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned())
}

/// The SoC blocks the meter keeps, with an optional trailing index and an
/// optional `DIE_<n>_` prefix. Overlapping aliases IOReport also publishes
/// (per-cluster CPU rows, totals) are dropped so nothing is counted twice.
const BLOCKS: &[&str] = &[
    "GPU SRAM", "GPU", "ANE", "DRAM", "ISP", "AVE", "MSR", "AMCC", "DCS", "DISPEXT", "DISP",
    "VDEC", "SOC_AON", "SOC_REST",
];

fn strip_die(name: &str) -> &str {
    if let Some(rest) = name.strip_prefix("DIE_") {
        let digits = rest.chars().take_while(|c| c.is_ascii_digit()).count();
        if digits > 0 {
            if let Some(tail) = rest[digits..].strip_prefix('_') {
                return tail;
            }
        }
    }
    name
}

/// Whether a channel name is one the meter sums.
pub fn keep_channel(name: &str) -> bool {
    let base = strip_die(name);
    if base == "CPU Energy" {
        return true;
    }
    BLOCKS.iter().any(|b| {
        base.strip_prefix(b)
            .is_some_and(|rest| rest.chars().all(|c| c.is_ascii_digit()))
    })
}

/// The coarse block a kept channel belongs to: `cpu`, `gpu`, `ane`, `dram`
/// or `other`.
pub fn block_of(name: &str) -> &'static str {
    let base = strip_die(name);
    if base.starts_with("CPU") {
        "cpu"
    } else if base.starts_with("GPU") {
        "gpu"
    } else if base.starts_with("ANE") {
        "ane"
    } else if base.starts_with("DRAM") {
        "dram"
    } else {
        "other"
    }
}

fn unit_scale(unit: &str) -> Option<f64> {
    match unit {
        "J" => Some(1.0),
        "mJ" => Some(1e-3),
        "uJ" => Some(1e-6),
        "nJ" => Some(1e-9),
        _ => None,
    }
}

/// Cumulative joules per kept channel at one instant.
pub type EnergyReading = BTreeMap<String, f64>;

/// Joules spent between two readings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergyDelta {
    pub by_channel: BTreeMap<String, f64>,
    /// `cpu`, `gpu`, `ane`, `dram`, `other`, the kept channels summed.
    pub by_block: BTreeMap<String, f64>,
    pub total_j: f64,
    pub energy_model: String,
}

/// `after - before`. `None` when the channel set changed or any counter
/// went backwards (a reset), which makes the interval unreadable.
pub fn delta(before: &EnergyReading, after: &EnergyReading) -> Option<EnergyDelta> {
    if before.keys().ne(after.keys()) || before.is_empty() {
        return None;
    }
    let mut by_channel = BTreeMap::new();
    let mut by_block: BTreeMap<String, f64> = BTreeMap::new();
    let mut total_j = 0.0;
    for (name, &b) in before {
        let a = after[name];
        if a < b {
            return None;
        }
        let d = a - b;
        by_channel.insert(name.clone(), d);
        *by_block.entry(block_of(name).to_string()).or_default() += d;
        total_j += d;
    }
    Some(EnergyDelta {
        by_channel,
        by_block,
        total_j,
        energy_model: ENERGY_MODEL.into(),
    })
}

/// A subscription to the Energy Model counters. One per thread of use; the
/// raw pointers keep it off other threads.
pub struct EnergyMeter {
    lib: IoReport,
    subscription: CFTypeRef,
    subscribed: CFTypeRef,
    channels_key: CFTypeRef,
}

impl EnergyMeter {
    /// Subscribes, and takes one reading to prove the CPU and ANE channels
    /// are there. `Err` names what is missing; callers run without energy.
    pub fn open() -> Result<Self, String> {
        let lib = IoReport::load()?;
        let group = cf_string("Energy Model");
        // SAFETY: documented IOReport call with an owned group name and no
        // subgroup filter; the result is owned and released below.
        let channels = unsafe { (lib.copy_channels)(group, std::ptr::null(), 0, 0, 0) };
        // SAFETY: releasing the string this function created.
        unsafe { CFRelease(group) };
        if channels.is_null() {
            return Err("IOReport Energy Model counters are unavailable".into());
        }
        let mut subscribed: CFTypeRef = std::ptr::null();
        // SAFETY: documented call; `subscribed` receives an owned
        // dictionary the samples call needs.
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
            return Err("IOReport subscription failed".into());
        }
        let meter = Self {
            lib,
            subscription,
            subscribed,
            channels_key: cf_string("IOReportChannels"),
        };
        let first = meter.read()?;
        let has = |block: &str| first.keys().any(|k| block_of(k) == block);
        if !has("cpu") || !has("ane") {
            return Err(format!(
                "IOReport Energy Model lacks CPU or ANE channels (got {:?})",
                first.keys().collect::<Vec<_>>()
            ));
        }
        Ok(meter)
    }

    /// Cumulative joules per kept channel, now.
    pub fn read(&self) -> Result<EnergyReading, String> {
        // SAFETY: a live subscription and its channel dictionary; the
        // sample is owned and released at the end.
        let data =
            unsafe { (self.lib.samples)(self.subscription, self.subscribed, std::ptr::null()) };
        if data.is_null() {
            return Err("IOReport sample failed".into());
        }
        let result = (|| {
            // SAFETY: dictionary lookup on the live sample with an owned key.
            let items = unsafe { CFDictionaryGetValue(data, self.channels_key) };
            if items.is_null() {
                return Err("IOReport sample has no channels".to_string());
            }
            let mut out = EnergyReading::new();
            // SAFETY: array access within [0, count) on the live sample.
            let count = unsafe { CFArrayGetCount(items) };
            for i in 0..count {
                // SAFETY: as above; the item is borrowed from the sample.
                let item = unsafe { CFArrayGetValueAtIndex(items, i) };
                let name = read_string(unsafe { (self.lib.channel_name)(item) })?;
                if !keep_channel(&name) {
                    continue;
                }
                let unit = read_string(unsafe { (self.lib.unit_label)(item) })?;
                let scale = unit_scale(&unit)
                    .ok_or_else(|| format!("IOReport channel {name}: unit {unit:?}"))?;
                if out.contains_key(&name) {
                    return Err(format!("IOReport channel {name} appears twice"));
                }
                // SAFETY: the item is a simple integer channel.
                let value = unsafe { (self.lib.integer_value)(item, 0) };
                out.insert(name, value as f64 * scale);
            }
            Ok(out)
        })();
        // SAFETY: releasing the sample this call created.
        unsafe { CFRelease(data) };
        result
    }
}

impl Drop for EnergyMeter {
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

    #[test]
    fn kept_channels_and_blocks() {
        for name in [
            "CPU Energy",
            "GPU Energy0",
            "GPU SRAM0",
            "ANE0",
            "DRAM1",
            "DIE_1_ANE0",
        ] {
            // "GPU Energy0" is not a block name: only bare blocks with an index.
            let expect = name != "GPU Energy0";
            assert_eq!(keep_channel(name), expect, "{name}");
        }
        assert!(!keep_channel("ECPU"), "cluster rows overlap the aggregate");
        assert!(!keep_channel("PCPU"), "cluster rows overlap the aggregate");
        // A block name with or without an index is the block itself, as
        // joules.py's `GPU(?: SRAM)?\d*` reads it.
        assert!(keep_channel("GPU SRAM"));
        assert!(keep_channel("GPU"));
        assert!(!keep_channel("GPU Energy"), "not a bare block name");
        assert_eq!(block_of("DIE_0_GPU0"), "gpu");
        assert_eq!(block_of("CPU Energy"), "cpu");
        assert_eq!(block_of("ANE0"), "ane");
        assert_eq!(block_of("DRAM0"), "dram");
        assert_eq!(block_of("ISP0"), "other");
    }

    #[test]
    fn delta_sums_and_refuses_resets() {
        let before: EnergyReading = [("CPU Energy", 10.0), ("ANE0", 2.0), ("GPU0", 5.0)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        let mut after = before.clone();
        after.insert("CPU Energy".into(), 13.0);
        after.insert("ANE0".into(), 2.5);
        let d = delta(&before, &after).expect("monotone");
        assert_eq!(d.by_channel["CPU Energy"], 3.0);
        assert_eq!(d.by_block["ane"], 0.5);
        assert_eq!(d.by_block["gpu"], 0.0);
        assert!((d.total_j - 3.5).abs() < 1e-12);
        assert_eq!(d.energy_model, ENERGY_MODEL);
        // Swapping the arguments makes a counter go backwards: unreadable.
        assert!(delta(&after, &before).is_none());
        // A changed channel set is unreadable too.
        let mut other = after.clone();
        other.remove("GPU0");
        assert!(delta(&before, &other).is_none());
        assert!(delta(&EnergyReading::new(), &EnergyReading::new()).is_none());
    }
}
