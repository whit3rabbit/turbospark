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
use turbospark_bench::protocol::{
    swift_footer, ProtocolCase, PROTOCOL_CASES, PROTOCOL_EXPERT_CACHE_SLOTS, PROTOCOL_MAX_CONTEXT,
    PROTOCOL_MAX_NEW,
};
use turbospark_bench::real_model::{open_model_runner_with_context, run_protocol_case_with_budget};

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
///
/// `allow(dead_code)` because each oracle target compiles its own copy of
/// this module and `mistral_memory_oracle.rs` calls the windowed form only.
#[allow(dead_code)]
pub fn run_oracle(dir: &Path, baselines: &[ChipBaseline], unknown_ceiling_mib: u64) {
    run_oracle_at_context(dir, baselines, unknown_ceiling_mib, PROTOCOL_MAX_CONTEXT)
}

/// [`run_oracle_at_context`] with the GENERATION BUDGET named too.
///
/// The second per-family parameter. `gpt-oss` needs it: Harmony puts the
/// model's reasoning in an `analysis` channel BEFORE the answer, so its three
/// cases need 818 / 1,780 / 1,211 tokens to reach `<|return|>` and two of
/// three stop on `maxTokens` at the shared 1,024 -- which the validity gate
/// below refuses, correctly (a truncated run is not comparable) and for a
/// reason that is not a defect. See `run_protocol_case_with_budget`.
#[allow(dead_code)]
pub fn run_oracle_at_context(
    dir: &Path,
    baselines: &[ChipBaseline],
    unknown_ceiling_mib: u64,
    max_context: u32,
) {
    run_oracle_with_budget(
        dir,
        baselines,
        unknown_ceiling_mib,
        max_context,
        PROTOCOL_MAX_NEW,
    )
}

/// [`run_oracle`] at a family-specific KV window.
///
/// EVERY CEILING IS A CEILING AT ONE WINDOW, and on a dense install the
/// window is most of what is being asserted: Mistral 7B's KV at 4,096 is
/// 537 MiB of a 684 MiB peak (AGENTS.md Gotcha 40), so doubling the window
/// nearly doubles the number. The window is therefore printed on every run
/// beside the ceiling, and a row's comment has to state it -- comparing two
/// rows measured at different windows is meaningless in a way that the two
/// numbers alone do not reveal.
///
/// Why any family needs this at all: the protocol freezes the PROSE, and its
/// token count is the checkpoint's tokenizer's answer. `long-synthesis` is
/// 3,444 tokens under Mistral's 32k vocab against 2,842 under
/// Qwen3-30B-A3B's 152k, and `3444 + PROTOCOL_MAX_NEW > 4096`, so the case
/// does not run and the `endOfTurn` gate below cannot be satisfied.
pub fn run_oracle_with_budget(
    dir: &Path,
    baselines: &[ChipBaseline],
    unknown_ceiling_mib: u64,
    max_context: u32,
    max_new: u32,
) {
    run_oracle_over_cases(
        dir,
        baselines,
        unknown_ceiling_mib,
        max_context,
        max_new,
        &PROTOCOL_CASES,
    )
}

/// [`run_oracle_with_budget`] over a CHOSEN subset of the protocol cases,
/// rather than always all three.
///
/// `qwen4_exp` is why this exists: its window is capped at 2,048
/// (`real_model_params::QWEN4_EXP_MAX_CONTEXT`, the checkpoint's own
/// `compressed_attention.index_budget`, not a chosen number), and
/// `long-synthesis` alone tokenizes to 2,940 under this family's vocab --
/// over the window before a single generated token is added. There is no
/// context at which that case can run on this family today, so an oracle
/// that iterates `PROTOCOL_CASES` unconditionally cannot be written for it;
/// this is the same body with the CASE LIST also a parameter. Every other
/// family's oracle is unaffected: `run_oracle_with_budget` still runs all
/// three by construction, not by a caller remembering to pass them.
///
/// `cases` must be non-empty and its first entry becomes the steady-state
/// replay case, exactly as `PROTOCOL_CASES[0]` (`short-explanation`) always
/// has been.
///
/// `allow(dead_code)` for the reason `run_oracle` has it: not every oracle
/// target's own compiled copy of this module calls every entry point.
#[allow(dead_code)]
pub fn run_oracle_over_cases(
    dir: &Path,
    baselines: &[ChipBaseline],
    unknown_ceiling_mib: u64,
    max_context: u32,
    max_new: u32,
    cases: &[ProtocolCase],
) {
    run_oracle_over_cases_with_slots(
        dir,
        baselines,
        unknown_ceiling_mib,
        max_context,
        max_new,
        cases,
        PROTOCOL_EXPERT_CACHE_SLOTS,
    )
}

/// Same protocol with an explicit cache size for large expert families.
#[allow(dead_code, clippy::too_many_arguments)]
pub fn run_oracle_over_cases_with_slots(
    dir: &Path,
    baselines: &[ChipBaseline],
    unknown_ceiling_mib: u64,
    max_context: u32,
    max_new: u32,
    cases: &[ProtocolCase],
    slots: usize,
) {
    assert!(
        !cases.is_empty(),
        "an oracle over zero cases has nothing to measure"
    );
    let brand = chip_brand_string();
    let baseline = brand
        .as_deref()
        .and_then(|b| baselines.iter().find(|row| b.contains(row.brand_substr)));
    match baseline {
        Some(row) => eprintln!(
            "memory_oracle: chip {:?} -> ceiling {} MiB at {max_context} context, \
             {max_new} max_new, tok/s floor {} (source: {})",
            brand, row.footprint_ceiling_mib, row.tok_s_floor, row.source
        ),
        None => eprintln!(
            "memory_oracle: chip {brand:?} not in the baseline table -> ceiling \
             {unknown_ceiling_mib} MiB at {max_context} context, tok/s reported \
             but not asserted"
        ),
    }

    eprintln!("memory_oracle: {max_new} max_new, {slots} expert slots");
    let (mut runner, tokenizer) =
        open_model_runner_with_context(dir, slots, max_context).expect("real install should open");
    let mut sampler = AppMemorySampler::new();

    let mut measured = Vec::new();
    for case in cases {
        // Frozen protocol: one discarded warmup, then the measured run.
        run_protocol_case_with_budget(
            &mut runner,
            &tokenizer,
            case,
            &mut sampler,
            Default::default(),
            max_context,
            max_new,
        )
        .unwrap_or_else(|e| panic!("{} warmup failed: {e}", case.id));
        let result = run_protocol_case_with_budget(
            &mut runner,
            &tokenizer,
            case,
            &mut sampler,
            Default::default(),
            max_context,
            max_new,
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

    let warm_case = &cases[0];
    let mut previous = sampler.sample().expect("footprint sampling worked");
    let mut growth = u64::MAX;
    let mut round = 0usize;
    while round < STEADY_STATE_ROUNDS && growth > STEADY_STATE_SLACK_BYTES {
        run_protocol_case_with_budget(
            &mut runner,
            &tokenizer,
            warm_case,
            &mut sampler,
            Default::default(),
            max_context,
            max_new,
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
    eprintln!(
        "memory_oracle: session peak {peak_mib} MiB, ceiling {ceiling_mib} MiB \
         (at {max_context} context)"
    );
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

/// **THE CATALOG AND THE ORACLES HAVE TO AGREE, AND NOTHING ELSE MAKES THEM.**
///
/// `models.json` carries a `measured` block per row: the OBSERVATIONS a
/// recommendation quotes. A `ChipBaseline` carries a ceiling and a floor:
/// ASSERTIONS with a per-row margin, plus the paragraph of provenance that
/// justifies the margin, which is why the two are not one table (JSON has
/// nowhere to put the paragraph, and each margin is a judgement --
/// `memory_oracle.rs` takes peak +5% and `qwen38_memory_oracle.rs` +13%,
/// and both say why).
///
/// What that leaves is two numbers describing one run in two files, which is
/// the shape of every count this repo has watched rot. This is the tie: it
/// runs in `cargo test --workspace` with no install and no GPU, so an edit to
/// either side that contradicts the other fails on the edit rather than the
/// next time somebody happens to have a 13 GB install on disk.
///
/// **ONLY SELF-MEASURED ROWS ARE CHECKED.** `memory_oracle.rs` carries rows
/// sourced from the Swift engine's own published numbers for chips nothing
/// here has ever run on (M5 Pro, M2). Requiring a catalog row for those would
/// mean inventing measurements this machine never took, which is the exact
/// failure the `source` field exists to prevent.
///
/// `allow(dead_code)` for the reason `run_oracle` has it: every oracle target
/// compiles its own copy of this module.
#[allow(dead_code)]
pub fn assert_agrees_with_catalog(
    alias: &str,
    baselines: &[ChipBaseline],
    context: u32,
    slots: u32,
) {
    let catalog = turbospark_catalog::Catalog::embedded().expect("the embedded catalog parses");
    let entry = catalog
        .get(alias)
        .unwrap_or_else(|| panic!("{alias}: this oracle asserts an install with no catalog row"));

    for row in baselines {
        if !row.source.contains("this port") {
            continue;
        }
        let m = entry.measured_for(row.brand_substr).unwrap_or_else(|| {
            panic!(
                "{alias}: the {} baseline is this port's own measurement and models.json \
                 records nothing for that chip. The ceiling and floor beside it were \
                 calibrated from numbers that now live nowhere.",
                row.brand_substr
            )
        });

        assert_eq!(
            m.context, context,
            "{alias}: models.json records a peak at {} context and this oracle runs at \
             {context}. Every footprint is a footprint at one window.",
            m.context
        );
        assert_eq!(
            m.expert_cache_slots, slots,
            "{alias}: models.json records {} expert-cache slots and this oracle pins \
             {slots}. On a streamed MoE that term is the dominant one.",
            m.expert_cache_slots
        );
        assert!(
            row.footprint_ceiling_mib >= m.peak_footprint_mib,
            "{alias}: the {} ceiling is {} MiB, UNDER the {} MiB peak models.json records. \
             One of the two moved without the other.",
            row.brand_substr,
            row.footprint_ceiling_mib,
            m.peak_footprint_mib
        );
        // A ceiling far above its own evidence has stopped being a guard. The
        // widest margin any row takes today is qwen38's +13%; 50% is loose
        // enough never to flake on a re-freeze and tight enough that a
        // doubling cannot hide.
        assert!(
            row.footprint_ceiling_mib <= m.peak_footprint_mib * 3 / 2,
            "{alias}: the {} ceiling is {} MiB against a measured {} MiB peak, over 1.5x \
             its own evidence. A ceiling that loose cannot catch a regression.",
            row.brand_substr,
            row.footprint_ceiling_mib,
            m.peak_footprint_mib
        );
        assert!(
            row.tok_s_floor <= m.decode_tok_s_min,
            "{alias}: the {} floor is {} tok/s, ABOVE the {} tok/s slowest reading \
             models.json records -- this oracle cannot pass on the machine it was \
             calibrated on.",
            row.brand_substr,
            row.tok_s_floor,
            m.decode_tok_s_min
        );
    }
}
