//! The phase breakdown, as data.
//!
//! `RealForwardRunner::phase_counters()` is cumulative over every forward
//! pass the runner served and has no reset, so a prefill number and a decode
//! number are DELTAS between snapshots taken at the boundaries. The bucket
//! definitions match `turbospark-check`'s `TURBOSPARK_PHASES=1` printout
//! (`crates/cli/src/generate/format.rs`), including "encode + logit readbk"
//! as total minus the named wall-clock buckets; `tests/phases_format.rs`
//! pins that.

use runtime::PhaseCounters;
use serde::{Deserialize, Serialize};

/// `after - before`, field by field.
pub fn delta(before: &PhaseCounters, after: &PhaseCounters) -> PhaseCounters {
    macro_rules! d {
        ($($f:ident),* $(,)?) => {
            PhaseCounters { $($f: after.$f.saturating_sub(before.$f),)* }
        };
    }
    d!(
        calls,
        total_nanos,
        gpu_wait_nanos,
        final_wait_nanos,
        router_nanos,
        expert_io_nanos,
        bind_nanos,
        pipeline_wait_nanos,
        cb1_gpu_nanos,
        routed_cb_gpu_nanos,
        final_cb_gpu_nanos,
        expert_requests,
        expert_hits,
        expert_io_bytes_requested,
        expert_io_bytes_physical,
        expert_io_samples,
        qwen_shared_submissions,
        mapped_prepare_calls,
        mapped_prepared_bytes,
        mapped_page_touches,
        mapped_advice_calls,
        mapped_advice_failures,
        mapped_prepare_nanos,
    )
}

/// One named wall-clock bucket.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Bucket {
    pub name: String,
    pub ms: f64,
    pub ms_per_token: f64,
    pub share: f64,
}

/// The breakdown for one stretch of forward passes (a prefill, or a decode).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseReport {
    /// Forward passes: one per token, the runtime's own divisor.
    pub calls: u64,
    pub total_ms: f64,
    pub buckets: Vec<Bucket>,
    pub gpu_busy_ms_per_token: GpuBusy,
    pub expert_requests: u64,
    pub expert_hits: u64,
    pub expert_bytes_requested: u64,
    /// `None` unless `TURBOSPARK_EXPERT_DISK_IO=1` sampled the reads: an
    /// unmeasured run and a fully cache-resident one both read 0 here.
    pub expert_bytes_physical: Option<u64>,
    pub raw: RawCounters,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuBusy {
    pub cb1: f64,
    pub routed: f64,
    pub final_cb: f64,
}

/// The counters themselves, so a later reader can recompute anything.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawCounters(pub serde_json::Value);

/// The bucket names and order, as `format.rs` prints them (trimmed).
pub const BUCKET_NAMES: [&str; 7] = [
    "gpu wait (layer cb1)",
    "final wait (end token)",
    "router readback+topk",
    "expert io (pread)",
    "routed bind+upload",
    "routed cb retire",
    "encode + logit readbk",
];

pub fn report(p: &PhaseCounters) -> PhaseReport {
    let calls = p.calls.max(1) as f64;
    let ms = |n: u64| n as f64 / 1e6;
    let accounted = p.gpu_wait_nanos
        + p.final_wait_nanos
        + p.router_nanos
        + p.expert_io_nanos
        + p.bind_nanos
        + p.pipeline_wait_nanos;
    let other = p.total_nanos.saturating_sub(accounted);
    let values = [
        p.gpu_wait_nanos,
        p.final_wait_nanos,
        p.router_nanos,
        p.expert_io_nanos,
        p.bind_nanos,
        p.pipeline_wait_nanos,
        other,
    ];
    let buckets = BUCKET_NAMES
        .iter()
        .zip(values)
        .map(|(name, n)| Bucket {
            name: name.to_string(),
            ms: ms(n),
            ms_per_token: ms(n) / calls,
            share: n as f64 / p.total_nanos.max(1) as f64,
        })
        .collect();
    PhaseReport {
        calls: p.calls,
        total_ms: ms(p.total_nanos),
        buckets,
        gpu_busy_ms_per_token: GpuBusy {
            cb1: ms(p.cb1_gpu_nanos) / calls,
            routed: ms(p.routed_cb_gpu_nanos) / calls,
            final_cb: ms(p.final_cb_gpu_nanos) / calls,
        },
        expert_requests: p.expert_requests,
        expert_hits: p.expert_hits,
        expert_bytes_requested: p.expert_io_bytes_requested,
        expert_bytes_physical: (p.expert_io_samples > 0).then_some(p.expert_io_bytes_physical),
        raw: RawCounters(raw_json(p)),
    }
}

fn raw_json(p: &PhaseCounters) -> serde_json::Value {
    serde_json::json!({
        "calls": p.calls, "total_nanos": p.total_nanos,
        "gpu_wait_nanos": p.gpu_wait_nanos, "final_wait_nanos": p.final_wait_nanos,
        "router_nanos": p.router_nanos, "expert_io_nanos": p.expert_io_nanos,
        "bind_nanos": p.bind_nanos, "pipeline_wait_nanos": p.pipeline_wait_nanos,
        "cb1_gpu_nanos": p.cb1_gpu_nanos, "routed_cb_gpu_nanos": p.routed_cb_gpu_nanos,
        "final_cb_gpu_nanos": p.final_cb_gpu_nanos,
        "expert_requests": p.expert_requests, "expert_hits": p.expert_hits,
        "expert_io_bytes_requested": p.expert_io_bytes_requested,
        "expert_io_bytes_physical": p.expert_io_bytes_physical,
        "expert_io_samples": p.expert_io_samples,
        "qwen_shared_submissions": p.qwen_shared_submissions,
        "mapped_prepare_calls": p.mapped_prepare_calls,
        "mapped_prepared_bytes": p.mapped_prepared_bytes,
        "mapped_page_touches": p.mapped_page_touches,
        "mapped_advice_calls": p.mapped_advice_calls,
        "mapped_advice_failures": p.mapped_advice_failures,
        "mapped_prepare_nanos": p.mapped_prepare_nanos,
    })
}

/// Page-cache class of a run's expert reads, from the measured physical to
/// requested ratio. Pairs whose arms differ in class are not comparable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheClass {
    /// No routed expert reads (dense, or mapped residency).
    NoReads,
    /// Reads happened but physical bytes were not sampled.
    Unmeasured,
    /// Physical under 5% of requested: a page-cache memcpy.
    Warm,
    /// Physical between 5% and 50%.
    Mixed,
    /// Physical over 50%: real device I/O.
    Cold,
}

pub fn cache_class(r: &PhaseReport) -> CacheClass {
    if r.expert_bytes_requested == 0 {
        return CacheClass::NoReads;
    }
    match r.expert_bytes_physical {
        None => CacheClass::Unmeasured,
        Some(p) => {
            let ratio = p as f64 / r.expert_bytes_requested as f64;
            if ratio < 0.05 {
                CacheClass::Warm
            } else if ratio <= 0.5 {
                CacheClass::Mixed
            } else {
                CacheClass::Cold
            }
        }
    }
}

/// Where a routed (MoE) stretch spends its wall time, as the reopen test
/// for Neural Engine expert work asks it (`docs/SPEEDUP_PLAYBOOK.md`, "Not
/// pursued": reopen "only once the GPU routed path makes MoE prefill
/// compute bound"). Shares are of `total_ms`:
///
/// - `routed_gpu_share`: GPU busy time of the routed command buffers, the
///   expert math itself.
/// - `io_share`: expert pread plus routed bind and upload, the streaming
///   cache's cost.
/// - `loop_share`: router readback and top-k plus "encode + logit readbk",
///   the per-token host loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoeBoundClass {
    /// No routed work in the stretch (a dense install or mapped residency
    /// with no routed buffers measured).
    Dense,
    IoBound,
    LoopBound,
    ComputeBound,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoeBound {
    pub routed_gpu_share: f64,
    pub io_share: f64,
    pub loop_share: f64,
    pub class: MoeBoundClass,
}

/// Compute bound when the routed GPU work is at least this share of the
/// wall time...
pub const COMPUTE_BOUND_MIN_ROUTED_GPU_SHARE: f64 = 0.5;
/// ...and expert I/O is under this share. Below both, the larger of the I/O
/// and host-loop shares names the bound. The thresholds are a reading aid
/// with the rule in one place; the shares themselves are in the record.
pub const COMPUTE_BOUND_MAX_IO_SHARE: f64 = 0.2;

pub fn moe_bound(r: &PhaseReport) -> MoeBound {
    let total = r.total_ms.max(f64::MIN_POSITIVE);
    let bucket = |name: &str| {
        r.buckets
            .iter()
            .find(|b| b.name == name)
            .map_or(0.0, |b| b.ms)
    };
    let routed_gpu_ms = r.gpu_busy_ms_per_token.routed * r.calls as f64;
    let io_ms = bucket("expert io (pread)") + bucket("routed bind+upload");
    let loop_ms = bucket("router readback+topk") + bucket("encode + logit readbk");
    let routed_gpu_share = routed_gpu_ms / total;
    let io_share = io_ms / total;
    let loop_share = loop_ms / total;
    let routed = r.expert_requests > 0 || routed_gpu_ms > 0.0 || io_ms > 0.0;
    MoeBound {
        routed_gpu_share,
        io_share,
        loop_share,
        class: classify_shares(routed_gpu_share, io_share, loop_share, routed),
    }
}

/// The class from the three shares alone, so a reader with medians over
/// several runs (`report::print_profile`) classifies the same way.
pub fn classify_shares(
    routed_gpu_share: f64,
    io_share: f64,
    loop_share: f64,
    routed: bool,
) -> MoeBoundClass {
    if !routed {
        MoeBoundClass::Dense
    } else if routed_gpu_share >= COMPUTE_BOUND_MIN_ROUTED_GPU_SHARE
        && io_share < COMPUTE_BOUND_MAX_IO_SHARE
    {
        MoeBoundClass::ComputeBound
    } else if io_share >= loop_share {
        MoeBoundClass::IoBound
    } else {
        MoeBoundClass::LoopBound
    }
}
