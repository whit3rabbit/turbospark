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

/// This process's peak `phys_footprint` in bytes, or 0 where the counter is
/// unavailable.
///
/// **The SAME mach counter every frozen row in `docs/BENCHMARKS.md` is
/// measured with**, which is why this borrows `crates/bench`'s sampler rather
/// than reading a different one: a GUI reporting a number the memory oracle
/// would not recognise is worse than reporting none.
///
/// Note what it counts. On a streamed MoE install the resident weight
/// mapping IS counted (Metal pins it), so the honest accounting is
/// `weights + KV + expert slot capacity + baseline`. On a DENSE install the
/// weights are NOT counted at all, and KV is nearly the whole number -- so
/// the same figure means different things across families and must be read
/// beside the context window.
#[cfg(target_os = "macos")]
pub(crate) fn peak_footprint_bytes() -> u64 {
    let mut sampler = bench::memory::AppMemorySampler::new();
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
    let info = serde_json::json!({
        "physicalMemoryBytes": physical,
        "recommendedWorkingSetBytes": working_set,
        "chip": if chip.is_empty() { None } else { Some(chip) },
        "lowPowerMode": low_power,
        "thermalLevel": thermal,
    });
    serde_json::to_string(&info).map_err(|e| e.to_string())
}
