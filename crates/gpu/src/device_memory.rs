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
