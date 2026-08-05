//! Port of the Swift app's `AppMemorySampler`:
//! `task_info(mach_task_self_, TASK_VM_INFO).phys_footprint`, in bytes,
//! with peak tracking. `phys_footprint` (not `resident_size`) is what
//! every published Mference memory baseline reports, so the oracle must
//! measure the same counter. Direct mach FFI: the `libc` crate does not
//! bind `task_vm_info`, and nothing outside this crate needs the probe.

/// `task_vm_info` from `<mach/task_info.h>`, truncated after
/// `phys_footprint` (the REV0 prefix): the kernel copies out
/// `min(requested, available)` natural_t words, so requesting exactly this
/// prefix is valid on every macOS this crate supports. Field order and
/// widths must match the header byte for byte; a wrong count makes
/// `task_info` return `KERN_INVALID_ARGUMENT`, which surfaces as `None`
/// from [`AppMemorySampler::sample`] and fails the unit test below.
#[repr(C)]
#[derive(Default)]
struct TaskVmInfo {
    virtual_size: u64,
    region_count: i32,
    page_size: i32,
    resident_size: u64,
    resident_size_peak: u64,
    device: u64,
    device_peak: u64,
    internal: u64,
    internal_peak: u64,
    external: u64,
    external_peak: u64,
    reusable: u64,
    reusable_peak: u64,
    purgeable_volatile_pmap: u64,
    purgeable_volatile_resident: u64,
    purgeable_volatile_virtual: u64,
    compressed: u64,
    compressed_peak: u64,
    compressed_lifetime: u64,
    phys_footprint: u64,
}

const TASK_VM_INFO: u32 = 22;
const KERN_SUCCESS: i32 = 0;

extern "C" {
    static mach_task_self_: u32;
    fn task_info(task: u32, flavor: u32, info: *mut i32, count: *mut u32) -> i32;
    fn sysctlbyname(
        name: *const std::ffi::c_char,
        oldp: *mut std::ffi::c_void,
        oldlenp: *mut usize,
        newp: *mut std::ffi::c_void,
        newlen: usize,
    ) -> i32;
}

fn read_process_footprint() -> Option<u64> {
    let mut info = TaskVmInfo::default();
    let mut count = (std::mem::size_of::<TaskVmInfo>() / std::mem::size_of::<i32>()) as u32;
    let kr = unsafe {
        task_info(
            mach_task_self_,
            TASK_VM_INFO,
            (&mut info as *mut TaskVmInfo).cast(),
            &mut count,
        )
    };
    if kr == KERN_SUCCESS {
        Some(info.phys_footprint)
    } else {
        None
    }
}

/// Current-process physical footprint sampler with peak tracking, the
/// Swift `AppMemorySampler` contract: `sample()` returns the current
/// footprint and folds it into the peak; `peak_bytes()` is `None` until a
/// successful sample lands.
#[derive(Default)]
pub struct AppMemorySampler {
    peak_bytes: Option<u64>,
}

impl AppMemorySampler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset_peak(&mut self) {
        self.peak_bytes = None;
    }

    /// One `phys_footprint` sample; updates the peak. `None` when
    /// `task_info` fails.
    pub fn sample(&mut self) -> Option<u64> {
        let bytes = read_process_footprint()?;
        self.peak_bytes = Some(self.peak_bytes.map_or(bytes, |peak| peak.max(bytes)));
        Some(bytes)
    }

    pub fn peak_bytes(&self) -> Option<u64> {
        self.peak_bytes
    }
}

/// `sysctl machdep.cpu.brand_string`, e.g. `"Apple M5 Pro"`. Used by the
/// memory oracle to pick the matching Swift baseline row.
pub fn chip_brand_string() -> Option<String> {
    let name = c"machdep.cpu.brand_string";
    let mut len: usize = 0;
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            std::ptr::null_mut(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 || len == 0 {
        return None;
    }
    let mut buf = vec![0u8; len];
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            buf.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return None;
    }
    buf.truncate(len);
    while buf.last() == Some(&0) {
        buf.pop();
    }
    String::from_utf8(buf).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_reports_a_positive_footprint_and_tracks_the_peak() {
        let mut sampler = AppMemorySampler::new();
        assert_eq!(sampler.peak_bytes(), None);
        let first = sampler.sample().expect("task_info should succeed");
        assert!(first > 0, "a live process has a nonzero footprint");
        let _ = sampler.sample();
        let peak = sampler.peak_bytes().expect("peak set after sampling");
        assert!(peak >= first);
        sampler.reset_peak();
        assert_eq!(sampler.peak_bytes(), None);
    }

    #[test]
    fn chip_brand_string_is_nonempty() {
        let brand = chip_brand_string().expect("sysctl should succeed");
        assert!(!brand.is_empty());
    }
}
