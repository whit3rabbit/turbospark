//! What the Metal device says it will hold: `recommendedMaxWorkingSetSize`
//! and the device name.
//!
//! Here rather than in `crates/runtime` for the reason [`crate::power_state`]
//! is: that crate is `#![forbid(unsafe_code)]` and portable, so the call into
//! Metal lives in the crate that already reaches it and crosses the boundary
//! as a plain integer and a `String`.
//!
//! **This is an advisory ceiling and NOT the pool the sizing policies budget
//! from.** `runtime::expert_cache_policy` and `runtime::context_policy` both
//! budget from `physical_memory()`, and a recommendation that disagreed with
//! what `open()` is about to do would be worse than no recommendation at all.
//! What this number is good for is saying so out loud: on a unified-memory
//! Mac it lands around 75% of installed RAM, so a machine whose working set
//! sits below `physical - reserve` is one where the driver, not the
//! arithmetic, is the binding constraint.
//!
//! ## Provenance
//!
//! Adapted from `src/vram.rs` of
//! <https://github.com/notactuallytreyanastasio/shoehorn> at commit
//! `17af0207e8b8`, MIT licensed (see `NOTICE`). Two changes, both
//! deliberate:
//!
//! 1. **The NVML and `rocm-smi` arms are dropped rather than adapted.** This
//!    crate compiles to nothing off macOS (AGENTS.md Gotcha 8), so a CUDA or
//!    ROCm probe here would be permanently unreachable code that reads as
//!    multi-vendor support this port does not have.
//! 2. Upstream returns free VRAM on the discrete-GPU paths and the Metal
//!    working set on macOS, then treats the two as one number. Only the
//!    second survives, and its meaning is stated above rather than implied.

use metal::Device;

/// `MTLDevice.recommendedMaxWorkingSetSize` and `MTLDevice.name`, or `None`
/// when there is no default Metal device (a headless or unsupported host).
///
/// Not cached: this is called once when a recommendation is rendered, never
/// per token, and a stale answer would outlive an eGPU change for no gain.
pub fn recommended_max_working_set() -> Option<(u64, String)> {
    let device = Device::system_default()?;
    Some((
        device.recommended_max_working_set_size(),
        device.name().to_string(),
    ))
}

// Same TASK_VM_INFO prefix as the benchmark oracle, including CPU residency.
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
}
/// Current process physical footprint, including other resident model families.
pub fn process_footprint() -> Option<u64> {
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

#[cfg(test)]
mod tests {
    #[test]
    fn process_memory_admission_probe_reports_a_live_footprint() {
        assert!(super::process_footprint().is_some_and(|bytes| bytes > 0));
    }
}
