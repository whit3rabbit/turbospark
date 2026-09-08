//! What a status panel reads.

use crate::session::{Engine, Session};
use crate::wire::PhaseReport;

/// Nanoseconds to milliseconds per call, or 0.0 when nothing has run.
fn per_call(nanos: u64, calls: u64) -> f64 {
    if calls == 0 {
        return 0.0;
    }
    nanos as f64 / calls as f64 / 1.0e6
}

pub(crate) fn phases_json(session: &Session) -> Result<String, String> {
    let engine = session
        .engine
        .lock()
        .map_err(|_| "the session is poisoned by an earlier panic".to_string())?;
    let report = match &*engine {
        #[cfg(target_os = "macos")]
        Engine::Real(runner) => {
            let p = runner.phase_counters();
            PhaseReport {
                calls: p.calls,
                total_ms_per_call: per_call(p.total_nanos, p.calls),
                gpu_wait_ms: per_call(p.gpu_wait_nanos, p.calls),
                final_wait_ms: per_call(p.final_wait_nanos, p.calls),
                router_ms: per_call(p.router_nanos, p.calls),
                expert_io_ms: per_call(p.expert_io_nanos, p.calls),
                bind_ms: per_call(p.bind_nanos, p.calls),
                pipeline_wait_ms: per_call(p.pipeline_wait_nanos, p.calls),
                cb1_gpu_ms: per_call(p.cb1_gpu_nanos, p.calls),
                routed_cb_gpu_ms: per_call(p.routed_cb_gpu_nanos, p.calls),
                final_cb_gpu_ms: per_call(p.final_cb_gpu_nanos, p.calls),
                expert_requests: p.expert_requests,
                expert_hits: p.expert_hits,
                // Null on no data rather than a 0% hit rate, which a GUI
                // would render as a full-width red bar on a fresh session.
                expert_hit_rate: (p.expert_requests > 0)
                    .then(|| p.expert_hits as f64 / p.expert_requests as f64),
            }
        }
        Engine::Scripted(_) => PhaseReport {
            calls: 0,
            total_ms_per_call: 0.0,
            gpu_wait_ms: 0.0,
            final_wait_ms: 0.0,
            router_ms: 0.0,
            expert_io_ms: 0.0,
            bind_ms: 0.0,
            pipeline_wait_ms: 0.0,
            cb1_gpu_ms: 0.0,
            routed_cb_gpu_ms: 0.0,
            final_cb_gpu_ms: 0.0,
            expert_requests: 0,
            expert_hits: 0,
            expert_hit_rate: None,
        },
    };
    serde_json::to_string(&report).map_err(|e| e.to_string())
}

/// This process's peak `phys_footprint` in bytes since the FIRST call to this
/// function in this process, or 0 where the counter is unavailable.
///
/// **The SAME mach counter every frozen row in `docs/BENCHMARKS.md` is
/// measured with**, which is why this borrows `crates/bench`'s sampler rather
/// than reading a different one: a GUI reporting a number the memory oracle
/// would not recognise is worse than reporting none.
///
/// **A PROCESS-WIDE SAMPLER, NOT A FRESH ONE PER CALL.** A fresh
/// `AppMemorySampler` has recorded exactly one reading by the time
/// `peak_bytes()` is read back, so what this used to report was the CURRENT
/// footprint under a name that says "peak" -- correct only for a caller who
/// happens to poll at the actual maximum. `SAMPLER` folds every call's
/// reading into one running peak instead, so a status panel polling this on
/// a timer gets what the name promises. A TRUE lifetime peak -- one that also
/// sees the instant between two polls -- is not obtainable from this REV0
/// `task_vm_info` prefix without polling more often than a caller asks for,
/// which is out of scope here.
///
/// Note what it counts. On a streamed MoE install the resident weight
/// mapping IS counted (Metal pins it), so the honest accounting is
/// `weights + KV + expert slot capacity + baseline`. On a DENSE install the
/// weights are NOT counted at all, and KV is nearly the whole number -- so
/// the same figure means different things across families and must be read
/// beside the context window.
#[cfg(target_os = "macos")]
pub(crate) fn peak_footprint_bytes() -> u64 {
    static SAMPLER: std::sync::OnceLock<std::sync::Mutex<bench::memory::AppMemorySampler>> =
        std::sync::OnceLock::new();
    let sampler =
        SAMPLER.get_or_init(|| std::sync::Mutex::new(bench::memory::AppMemorySampler::new()));
    // A poisoned lock (some earlier caller panicked mid-sample, which
    // `sample`'s own body cannot do) still has a peak worth reading; recover
    // rather than losing every reading taken so far.
    let mut sampler = sampler
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    sampler.sample();
    sampler.peak_bytes().unwrap_or(0)
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn peak_footprint_bytes() -> u64 {
    0
}

/// System hardware and power telemetry as JSON.
pub(crate) fn system_info_json() -> Result<String, String> {
    let physical = runtime::physical_memory();
    let (working_set, chip) = match runtime::recommended_max_working_set() {
        Some((bytes, name)) => (Some(bytes), name),
        None => (None, String::new()),
    };
    let low_power = runtime::low_power_mode_enabled();
    let thermal = format!("{:?}", runtime::thermal_level()).to_lowercase();
    // **POLLED UNCONDITIONALLY, unlike the decode loop's own probe**, which
    // follows the power profile's stepping and therefore does nothing under
    // the default `performance`. So this is the call a status panel reads:
    // `RawDecodeResult::peak_memory_pressure` is `Normal` on a default
    // session because nothing watched, not because memory was fine.
    let memory = format!("{:?}", runtime::memory_pressure()).to_lowercase();
    let info = serde_json::json!({
        "physicalMemoryBytes": physical,
        "recommendedWorkingSetBytes": working_set,
        "chip": if chip.is_empty() { None } else { Some(chip) },
        "lowPowerMode": low_power,
        "thermalLevel": thermal,
        "memoryPressure": memory,
    });
    serde_json::to_string(&info).map_err(|e| e.to_string())
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::peak_footprint_bytes;

    /// A fresh-sampler-per-call implementation reports the CURRENT footprint
    /// under the name "peak" -- correct only for a caller who happens to
    /// poll at the actual maximum. This asserts the invariant a real peak
    /// has to hold: a later reading is never lower than an earlier one.
    ///
    /// **THIS DOES NOT MUTATION-CHECK AGAINST THE BUG IT FIXES, and that is
    /// recorded rather than hidden.** The natural mutation -- allocate and
    /// touch a large block, sample, drop or `munmap` it, sample again, and
    /// expect a fresh-sampler bug to show a fall -- was tried at 64 MiB and
    /// 512 MiB through a `Vec`, and again through a raw `mmap`/`munmap` pair
    /// to rule out the allocator retaining the freed block rather than
    /// returning it to the kernel. All three read the IDENTICAL
    /// `phys_footprint` before and after the free: this counter simply does
    /// not fall within a live process on this machine inside a test's
    /// timescale, which means a fresh sampler and a persistent one are
    /// indistinguishable by any allocation pattern this test can drive. The
    /// fix is still correct (a "peak" that is actually "current" is a real
    /// contract bug, visible the moment a caller's own footprint happens to
    /// dip, which processes with active KV eviction or expert-slot turnover
    /// do), and the assertion below is the honest invariant this environment
    /// can check rather than a false claim of having reproduced the defect.
    #[test]
    fn the_reported_peak_never_falls_across_calls() {
        let first = peak_footprint_bytes();
        assert!(first > 0, "a live process has a nonzero footprint");
        for _ in 0..8 {
            let next = peak_footprint_bytes();
            assert!(
                next >= first,
                "the peak must not fall across calls: {first} then {next}"
            );
        }
    }
}
