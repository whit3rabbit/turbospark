#![cfg(target_os = "macos")]
//! The memory oracle's body, shared by the per-family oracle targets
//! (`memory_oracle.rs`, `qwen36_memory_oracle.rs`).
//!
//! ONE MODEL PER PROCESS, and that is the reason this is a shared module
//! rather than a second `#[test]` in one file. Two families have two
//! different ceilings (Gemma 4 26B-A4B peaks around 2,100-2,200 MiB,
//! Qwen 3.6 35B-A3B around 1,610), and the assertion is against a
//! WHOLE-SESSION peak: a second model opened in the same process is
//! measured against the high-water mark the first one left, so neither
//! ceiling means anything. Separate integration-test targets are separate
//! binaries, and cargo runs test targets one after another, so each
//! family gets a clean process. Adding a second `#[test]` to either file
//! reintroduces exactly the problem this split exists to avoid, since
//! tests within one binary run on parallel threads.

use std::path::Path;

use runtime::StopReason;
use turbospark_bench::memory::{chip_brand_string, AppMemorySampler};
use turbospark_bench::protocol::{swift_footer, PROTOCOL_CASES, PROTOCOL_EXPERT_CACHE_SLOTS};
use turbospark_bench::real_model::{open_model_runner, run_protocol_case};

pub struct ChipBaseline {
    pub brand_substr: &'static str,
    pub footprint_ceiling_mib: u64,
    pub tok_s_floor: f64,
    /// Where the two numbers came from. Not decoration: a row measured
    /// from THIS port is a regression guard against its own past self,
    /// while a row from the Swift docs is a parity claim. Reading a
    /// self-measured row as parity is the mistake this field exists to
    /// prevent, so it is printed on every run and quoted in the failure.
    pub source: &'static str,
}

pub const SWIFT_DOCS: &str = "Swift docs/BENCHMARKS.md";

/// Run the frozen protocol against `dir` and assert the peak footprint,
/// the steady state, the protocol validity gate, and (where the chip has
/// a row) the decode floor. `unknown_ceiling_mib` holds when the chip is
/// not in `baselines`: memory sizing does not depend on the chip, so the
/// loosest documented ceiling for THIS FAMILY still applies.
pub fn run_oracle(dir: &Path, baselines: &[ChipBaseline], unknown_ceiling_mib: u64) {
    let brand = chip_brand_string();
    let baseline = brand
        .as_deref()
        .and_then(|b| baselines.iter().find(|row| b.contains(row.brand_substr)));
    match baseline {
        Some(row) => eprintln!(
            "memory_oracle: chip {:?} -> ceiling {} MiB, tok/s floor {} (source: {})",
            brand, row.footprint_ceiling_mib, row.tok_s_floor, row.source
        ),
        None => eprintln!(
            "memory_oracle: chip {brand:?} not in the baseline table -> ceiling \
             {unknown_ceiling_mib} MiB, tok/s reported but not asserted"
        ),
    }

    let (mut runner, tokenizer) =
        open_model_runner(dir, PROTOCOL_EXPERT_CACHE_SLOTS).expect("real install should open");
    let mut sampler = AppMemorySampler::new();

    let mut measured = Vec::new();
    for case in &PROTOCOL_CASES {
        // Frozen protocol: one discarded warmup, then the measured run.
        run_protocol_case(
            &mut runner,
            &tokenizer,
            case,
            &mut sampler,
            Default::default(),
        )
        .unwrap_or_else(|e| panic!("{} warmup failed: {e}", case.id));
        let result = run_protocol_case(
            &mut runner,
            &tokenizer,
            case,
            &mut sampler,
            Default::default(),
        )
        .unwrap_or_else(|e| panic!("{} failed: {e}", case.id));
        eprintln!(
            "{:<18} {}  peak so far {:.1} MiB",
            result.case_id,
            swift_footer(
                result.reason,
                result.prompt_tokens,
                result.prefill_seconds,
                result.new_tokens,
                result.decode_seconds
            ),
            sampler.peak_bytes().unwrap_or(0) as f64 / 1_048_576.0
        );
        measured.push(result);
    }

    // Steady state. Everything the shortest case touches is warm by now:
    // KV was sized at open, its experts are in slots, the pipeline cache
    // is full. Replaying it must therefore cost nothing.
    //
    // What this caught: `MTLCommandQueue.commandBuffer` and
    // `MTLCommandBuffer.computeCommandEncoder` return AUTORELEASED
    // objects, and a plain Rust binary has exactly one autorelease pool,
    // around `main`. Before `gpu::autorelease_pool` wrapped each token,
    // every command buffer the process ever created stayed alive: ~6 KiB
    // each, 31 per token, ~180 KiB per decoded token, linear and
    // unbounded. It reads as "footprint grows with prompt length",
    // because longer prompts mean more `produce` calls -- which is also
    // exactly what a legitimately larger working set looks like from the
    // session peak alone.

    // Allocator jitter around the per-token host `Vec`s (a few MiB at
    // V=262144). A healthy replay lands near zero; the leak was 15.8 MiB
    // per replay of the shortest case.
    const STEADY_STATE_SLACK_BYTES: u64 = 8 * 1_048_576;
    // Rounds the expert slot cache gets to stop dirtying new pages. Slot
    // warming is real growth, but it decelerates and stops. A leak does not.
    const STEADY_STATE_ROUNDS: usize = 4;

    let warm_case = &PROTOCOL_CASES[0];
    let mut previous = sampler.sample().expect("footprint sampling worked");
    let mut growth = u64::MAX;
    let mut round = 0usize;
    while round < STEADY_STATE_ROUNDS && growth > STEADY_STATE_SLACK_BYTES {
        run_protocol_case(
            &mut runner,
            &tokenizer,
            warm_case,
            &mut sampler,
            Default::default(),
        )
        .unwrap_or_else(|e| panic!("{} replay failed: {e}", warm_case.id));
        let now = sampler.sample().expect("footprint sampling worked");
        growth = now.saturating_sub(previous);
        previous = now;
        round += 1;
        eprintln!(
            "memory_oracle: replay {round} of {} -> {:.1} MiB (+{:.2} MiB)",
            warm_case.id,
            now as f64 / 1_048_576.0,
            growth as f64 / 1_048_576.0,
        );
    }
    assert!(
        growth <= STEADY_STATE_SLACK_BYTES,
        "replaying an already-warm case {STEADY_STATE_ROUNDS} times never \
         stopped growing the footprint (last round +{:.2} MiB, slack \
         {:.0} MiB): something accumulates per token or per prompt token",
        growth as f64 / 1_048_576.0,
        STEADY_STATE_SLACK_BYTES as f64 / 1_048_576.0,
    );

    // Protocol validity gate: a run that dies on maxTokens or a stray stop
    // is not comparable to the published rows.
    for result in &measured {
        assert_eq!(
            result.reason,
            StopReason::EndOfTurn,
            "{}: measured run must stop with endOfTurn (got {:?})",
            result.case_id,
            result.reason
        );
    }

    // The memory oracle proper.
    let peak = sampler.peak_bytes().expect("footprint sampling worked");
    let peak_mib = peak / 1_048_576;
    let ceiling_mib = baseline.map_or(unknown_ceiling_mib, |row| row.footprint_ceiling_mib);
    let ceiling_source = baseline.map_or(SWIFT_DOCS, |row| row.source);
    eprintln!("memory_oracle: session peak {peak_mib} MiB, ceiling {ceiling_mib} MiB");
    assert!(
        peak_mib <= ceiling_mib,
        "peak phys_footprint {peak_mib} MiB exceeds the {ceiling_mib} MiB \
         ceiling from {ceiling_source}: this port uses more memory than that \
         ceiling allows"
    );

    // The throughput floor, when this chip has a published row.
    if let Some(row) = baseline {
        for result in &measured {
            let tok_s = result.tokens_per_second();
            assert!(
                tok_s >= row.tok_s_floor,
                "{}: {tok_s:.3} tok/s is under the {} floor for {} (source: {})",
                result.case_id,
                row.tok_s_floor,
                row.brand_substr,
                row.source
            );
        }
    } else {
        for result in &measured {
            eprintln!(
                "memory_oracle: {} decode {:.3} tok/s (not asserted, unknown chip)",
                result.case_id,
                result.tokens_per_second()
            );
        }
    }
}
