//! Mapped-residency eviction probe (ROADMAP P1 item 3): what happens to a
//! decode reading routed experts out of an `mmap` when the OS reclaims the
//! mapping's clean pages under memory pressure, and what refilling them
//! costs.
//!
//! # Why this probe exists
//!
//! `docs/EXPERT_RESIDENCY.md` measures mapped residency as a win -- 559
//! MiB against the streamed arm's 3,652, and 1.28x decode -- but every one
//! of those numbers was taken on a QUIET machine with the whole 12.3 GB
//! expert table comfortably resident. Pages nobody is charged for are pages
//! the OS may evict, and the doc's own "Not done" list names this
//! measurement as the thing that has to exist before `--expert-residency
//! auto` may ever resolve UP to mapped. This probe is that measurement:
//! it INDUCES pressure with a bounded, Drop-freed dirty allocation, and
//! reads decode throughput, `phys_footprint` and the process's fault
//! counters before, during and after.
//!
//! # Shapes
//!
//! - The runner opens EXPLICITLY mapped (`open_with_residency`), because
//!   every protocol opener in this crate pins `Streamed` on purpose
//!   (AGENTS.md Gotcha 35): a probe measuring mapped residency is exactly
//!   the caller that must say so rather than sense it.
//! - The pressure allocation is DIRTY (written, not just touched): clean
//!   anonymous pages would themselves be reclaimable and would pressure
//!   nothing. Dirty pages force the compressor/swapper, which is what
//!   evicts the CLEAN file-backed pages of the expert mapping -- the exact
//!   competition this probe reproduces.
//! - The budget is conservative and the probe stops as soon as the kernel
//!   reports Warn pressure, so a machine running this never approaches
//!   jetsam. Dropping the allocation (including on a panic) frees it.
//! - Assertions are ENGAGEMENT and degenerate-output only (finite, nonzero
//!   tokens): the numbers are printed as a verdict, nothing is frozen from
//!   one run (`mapped_expert_probe.rs`'s convention).
//!
//! # Run
//!
//! ```sh
//! TURBOSPARK_PROBE_INSTALL_DIR=~/models/gemma4.gturbo \
//!   cargo test -p turbospark-bench --test mapped_residency_eviction \
//!   --release -- --ignored --nocapture
//! ```
#![cfg(target_os = "macos")]

use std::path::PathBuf;

use runtime::{
    GenerationConfig, RateControl, RawDecodeProgress, RealForwardRunner, SteeringPolicy,
};
use selection::ShapingConfig;
use turbospark_bench::memory::{task_fault_counters, AppMemorySampler};

fn env_dir(key: &str) -> Option<PathBuf> {
    let raw = std::env::var(key).ok()?;
    if raw.trim().is_empty() {
        return None;
    }
    Some(PathBuf::from(match raw.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", std::env::var("HOME").unwrap_or_default()),
        None => raw,
    }))
}

/// The decode window every phase runs: long enough that a tok/s figure is
/// not one straggler token, short enough that three windows plus the
/// pressure ramp stay inside a probe run.
const MAX_NEW: u32 = 96;

const PROMPT: &str = "Explain how coastal wetlands reduce flood damage.";

/// What one decode window measured.
#[allow(dead_code)]
struct Window {
    tok_s: f64,
    new_tokens: u32,
    faults_delta: u64,
    pageins_delta: u64,
    peak_mib: f64,
    coherent: bool,
}

fn decode_window(
    runner: &mut RealForwardRunner,
    tokenizer: &tokenizer::MfTokenizer,
    sampler: &mut AppMemorySampler,
    label: &str,
) -> Window {
    let prompt_ids = tokenizer.encode(PROMPT, false);
    let vocab = runner.vocab_size();
    let max_context = 4096u32;
    let shaping = ShapingConfig::new(0.0, 1, None, 1.0, Some(1)).expect("greedy shaping");
    let config = GenerationConfig {
        shaping,
        max_new_tokens: MAX_NEW,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: RateControl::default(),
    };

    let (faults_before, pageins_before) =
        task_fault_counters().expect("TASK_EVENTS_INFO works on this platform");
    let mut text = String::new();
    let result = runtime::run_raw_completion(
        runner,
        tokenizer,
        &prompt_ids,
        &config,
        max_context,
        vocab,
        |event| match event {
            RawDecodeProgress::Token { delta, .. } => text.push_str(&delta),
            RawDecodeProgress::Tail(tail) => text.push_str(&tail),
            RawDecodeProgress::Prefill { .. } => {}
        },
    )
    .unwrap_or_else(|e| panic!("{label}: generation failed: {e}"));
    let (faults_after, pageins_after) =
        task_fault_counters().expect("TASK_EVENTS_INFO works on this platform");
    let peak = sampler.sample().expect("footprint sampling worked") as f64 / 1_048_576.0;

    let tok_s = if result.decode_seconds > 0.0 {
        result.new_tokens as f64 / result.decode_seconds
    } else {
        0.0
    };
    eprintln!(
        "mapped_residency_eviction: {label}: {:?}, {} new tokens, {tok_s:.3} tok/s, \
         faults +{}, pageins +{}, peak {peak:.1} MiB",
        result.reason,
        result.new_tokens,
        faults_after - faults_before,
        pageins_after - pageins_before,
    );
    eprintln!("mapped_residency_eviction: {label} output: {text}");

    Window {
        tok_s,
        new_tokens: result.new_tokens as u32,
        faults_delta: faults_after - faults_before,
        pageins_delta: pageins_after - pageins_before,
        peak_mib: peak,
        // Coherent-by-eye is the smoke contract; here it only has to be
        // finite and nonempty, and any stop reason but Cancelled is fine
        // for a window nobody cancels.
        coherent: result.new_tokens > 0 && text.chars().count() > 0,
    }
}

/// The dirty allocation that induces pressure. `Drop` frees it, so a panic
/// in the phases below cannot leak the pressure past the probe.
struct Pressure {
    _chunks: Vec<Vec<u8>>,
}

impl Pressure {
    /// Grows until the kernel reports Warn-or-worse pressure or `budget`
    /// bytes are held, whichever comes first. Every page is WRITTEN (see
    /// the module doc for why dirty specifically). Returns how much is
    /// held.
    fn apply(budget: u64) -> (Self, u64) {
        let mut chunks = Vec::new();
        let mut held = 0u64;
        let step = 64u64 << 20;
        while held < budget {
            if matches!(
                runtime::memory_pressure(),
                runtime::MemoryPressure::Warn | runtime::MemoryPressure::Critical
            ) {
                eprintln!(
                    "mapped_residency_eviction: pressure reached {:?} after {held} bytes held",
                    runtime::memory_pressure()
                );
                break;
            }
            let mut chunk = vec![0u8; step as usize];
            // Dirty every page: write the chunk index so the compiler
            // cannot elide the stores.
            for (page, byte) in chunk.chunks_exact_mut(4096).enumerate() {
                byte[0] = (page & 0xff) as u8;
            }
            held += step;
            chunks.push(chunk);
        }
        (Self { _chunks: chunks }, held)
    }
}

#[test]
#[ignore = "needs the real gemma4 install via TURBOSPARK_PROBE_INSTALL_DIR and \
            deliberately induces memory pressure; see the module doc"]
fn eviction_under_pressure_costs_throughput_and_faults_on_refill() {
    let Some(install) = env_dir("TURBOSPARK_PROBE_INSTALL_DIR") else {
        eprintln!("mapped_residency_eviction: TURBOSPARK_PROBE_INSTALL_DIR is not set; skipping.");
        return;
    };

    let arch = repack::peek_manifest_arch(&install).expect("manifest peeks");
    let tokenizer =
        tokenizer::MfTokenizer::load_from_dir(&install).expect("tokenizer loads from the install");
    // EXPLICIT Mapped: the protocol openers pin Streamed on purpose, and a
    // probe measuring the mapped mode is the caller that must say so.
    let mut runner = RealForwardRunner::open_with_residency(
        &install,
        arch,
        4096,
        runtime::ExpertCacheSlots::Fixed(16),
        runtime::DraftPolicies::off(),
        SteeringPolicy::off(),
        1,
        runtime::KvQuant::Off,
        runtime::ExpertResidency::Mapped,
    )
    .expect("the install opens under mapped residency");
    assert_eq!(
        runner.resolved_expert_residency(),
        runtime::ResolvedExpertResidency::Mapped,
        "engagement: this probe is meaningless if the open fell back to streamed"
    );

    let mut sampler = AppMemorySampler::new();

    // Phase 1: WARM decode. The first window also faults the pages in, so
    // a SECOND warm window is the honest warm baseline.
    let _cold = decode_window(&mut runner, &tokenizer, &mut sampler, "warm-up");
    let warm = decode_window(&mut runner, &tokenizer, &mut sampler, "warm");
    assert!(
        warm.coherent && warm.tok_s > 0.0,
        "the warm window must decode finitely and measurably"
    );

    // Phase 2: pressure, bounded well under what the machine cannot give
    // back: physical minus the install's resident region minus a reserve
    // that covers KV, scratch and everything else this process holds.
    let physical = runtime::physical_memory();
    let resident = std::fs::metadata(install.join("model_weights.bin"))
        .map(|m| m.len())
        .unwrap_or(0);
    let reserve = 6u64 << 30;
    let budget = physical.saturating_sub(resident).saturating_sub(reserve);
    eprintln!(
        "mapped_residency_eviction: pressure budget {budget} bytes (physical {physical}, \
         resident {resident}, reserve {reserve})"
    );
    let (pressure, held) = Pressure::apply(budget);
    eprintln!("mapped_residency_eviction: holding {held} dirty bytes");

    // Phase 3: decode UNDER pressure. Whatever the OS reclaimed, this
    // window pays back in faults; on a quiet-machine mapping that nothing
    // evicted, it reads like the warm window and the probe says so.
    let pressured = decode_window(&mut runner, &tokenizer, &mut sampler, "pressured");

    // Phase 4: release and decode again -- the recovery/refill view.
    drop(pressure);
    let recovered = decode_window(&mut runner, &tokenizer, &mut sampler, "recovered");

    assert!(
        pressured.coherent && recovered.coherent,
        "every window must decode finitely and nonemptily"
    );

    // THE VERDICT, printed and never frozen from one run. The interesting
    // ratios are pressured-against-warm (how much a reclaimed mapping costs
    // mid-decode) and recovered-against-warm (whether releasing pressure
    // restores the warm rate).
    eprintln!(
        "mapped_residency_eviction: verdict: warm {:.3} tok/s, pressured {:.3} tok/s \
         ({:.1}% of warm), recovered {:.3} tok/s ({:.1}% of warm); faults per window \
         warm +{} pressured +{} recovered +{}",
        warm.tok_s,
        pressured.tok_s,
        100.0 * pressured.tok_s / warm.tok_s,
        recovered.tok_s,
        100.0 * recovered.tok_s / warm.tok_s,
        warm.faults_delta,
        pressured.faults_delta,
        recovered.faults_delta,
    );
}
