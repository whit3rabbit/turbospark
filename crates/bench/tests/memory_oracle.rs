#![cfg(target_os = "macos")]
//! The memory oracle: runs the frozen community-protocol benchmark
//! against a REAL Gemma 4 `.gturbo` install and asserts this port stays
//! at or under the published Swift baselines.
//!
//! Not run by default (needs a ~14.6 GB install and takes minutes; use
//! --release or the tok/s numbers are meaningless):
//!
//!   MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
//!     cargo test -p mrefrust-bench --test memory_oracle --release -- --ignored --nocapture
//!
//! What is asserted:
//! - Peak `phys_footprint` (the exact counter the Swift baselines report,
//!   sampled at the same cadence) <= the documented Swift ceiling + ~5%
//!   headroom (the Swift docs' own repeat-run variance). Asserted always;
//!   memory sizing does not depend on the chip.
//! - Decode tok/s >= the documented floor, per case, ONLY when the chip
//!   brand matches a baseline row. A tok/s failure here means "the port
//!   decodes slower than its baseline on this hardware", which is a true
//!   finding rather than a broken test. Note the floors are not all the
//!   same kind of number: each row's `source` says whether it came from
//!   the Swift docs or from this port measuring itself.
//! - Every measured case must stop with `endOfTurn`, the frozen
//!   protocol's validity gate.
//! - Replaying an already-warm case stops growing the footprint. The
//!   ceiling above cannot see a slow leak: it is a whole-session peak
//!   against a published number, so anything that accumulates hides under
//!   unused expert slot capacity until it is hundreds of MiB. This one is
//!   scale-free. It lives in the SAME test, on the SAME runner, on
//!   purpose -- a second `#[test]` opens a second model in the same
//!   process and cargo runs the two in parallel, which doubles the
//!   footprint and fails the ceiling for no real reason.

use mrefrust_bench::memory::{chip_brand_string, AppMemorySampler};
use mrefrust_bench::protocol::{swift_footer, PROTOCOL_CASES, PROTOCOL_EXPERT_CACHE_SLOTS};
use mrefrust_bench::real_model::{open_model_runner, run_protocol_case};
use runtime::StopReason;

struct ChipBaseline {
    brand_substr: &'static str,
    footprint_ceiling_mib: u64,
    tok_s_floor: f64,
    /// Where the two numbers came from. Not decoration: a row measured
    /// from THIS port is a regression guard against its own past self,
    /// while a row from the Swift docs is a parity claim. Reading a
    /// self-measured row as parity is the mistake this field exists to
    /// prevent, so it is printed on every run and quoted in the failure.
    source: &'static str,
}

const SWIFT_DOCS: &str = "Swift docs/BENCHMARKS.md";

/// Per-chip rows, MOST SPECIFIC SUBSTRING FIRST -- the lookup takes the
/// first `contains` hit, so "Apple M4 Max" must precede any future bare
/// "Apple M4" row, and "Apple M2" deliberately also matches M2 Pro/Max
/// (whose real floors are strictly higher, which is safe for a floor).
///
/// Ceilings are the measured peak range's high end + ~5%, the cross-run
/// variance band; more headroom would mask a regression on the order of
/// one KV layer. Floors are below the SLOWEST case, since one floor is
/// asserted against every case.
const BASELINES: &[ChipBaseline] = &[
    // M5 Pro 24GB: peak footprint 2,126-2,142 MiB, decode 31.01-35.17 tok/s.
    ChipBaseline {
        brand_substr: "Apple M5 Pro",
        footprint_ceiling_mib: 2250,
        tok_s_floor: 31.0,
        source: SWIFT_DOCS,
    },
    // M4 Max 36GB (the development machine; see CLAUDE.local.md).
    //
    // NOT a Swift baseline: the Swift engine has no published row for this
    // chip and has not been run here, so these are THIS PORT's own numbers
    // and the row only catches regressions against them. Replace both with
    // Swift's if that run ever happens -- a real parity number would very
    // likely be tighter.
    //
    // Three clean oracle runs, merge a772b67, release build:
    //   peak footprint     2,120 / 2,197 / 2,197 MiB
    //   short-explanation  20.348 / 20.272 / 20.402 tok/s
    //   medium-review      15.972 / 15.773 / 15.783 tok/s
    //   long-synthesis     11.710 / 11.601 / 11.623 tok/s
    //
    // Two more after split-KV was wired (2026-08-06, on battery):
    //   peak footprint     2,126 / 2,125 MiB
    //   short-explanation  22.726 / 23.524 tok/s
    //   medium-review      21.126 / 19.978 tok/s
    //   long-synthesis     20.712 / 21.477 tok/s
    //
    // The 77 MiB peak spread is expert-slot warming, which depends on
    // which experts the sampled route actually touches, so the ceiling is
    // the high end + ~5% and lands ABOVE the generic 2,250 default rather
    // than below it. That is the honest number for this machine; a 2,250
    // ceiling here would flake on the spread alone. Split-KV did not move
    // the peak, as expected: it adds 512 KiB of attention scratch.
    //
    // One more after the expert-read chunking and read pool, same day:
    //   peak footprint     2,100 MiB
    //   short-explanation  25.060 tok/s
    //   medium-review      23.694 tok/s
    //   long-synthesis     23.068 tok/s
    //
    // The floor was 10.0 when the slowest case ran at 11.6. Split-KV took
    // that case to ~21, which left the old floor unable to catch losing
    // the entire change -- a regression to 11.6 would still have passed.
    // 15.0 sits ~25% under the slowest case at the time, which is wider
    // than any run-to-run spread observed here (the worst pair differs by
    // 1.1 tok/s) and still fails loudly if the split is lost.
    //
    // Then the sampler fix (2026-08-07): `selection::select` had been
    // full-sorting all 262144 candidates per token, ~18.9 ms against a
    // ~25 ms forward pass. Two oracle runs after it, on AC:
    //   peak footprint     2,107 / 2,183 MiB
    //   short-explanation  40.687 / 41.268 tok/s
    //   medium-review      37.745 / 37.708 tok/s
    //   long-synthesis     34.674 / 34.720 tok/s
    //
    // Floor raised 15.0 -> 25.0. Unlike the read-pool raise this is NOT
    // one sample: `scripts/parity.sh` measured the same three cases twice
    // more in the same session (40.763/40.668, 38.417/38.468,
    // 34.556/34.637), so the slowest case has three independent readings
    // inside 0.12 tok/s. 25.0 sits ~28% under it, wider than any spread
    // seen here, and would fail loudly if the whole sampler fix were lost
    // (which would land the slowest case back near 23).
    //
    // A REAL Swift comparison exists for this chip (2026-08-07,
    // `docs/BENCHMARKS.md`, reproduce with `scripts/parity.sh`): the same
    // install through `../Mference`'s MferenceCLI decodes at 41.1 / 38.6
    // / 34.3 tok/s where this port now does 40.7 / 38.4 / 34.6 -- parity
    // within 1%. Those numbers are still deliberately NOT the floor here.
    // This row's job is to catch regressions against this port's own
    // past; a floor set at a competitor's measured rate would flake on
    // the difference between two machine states rather than on a real
    // change. `source` below stays honest about that.
    ChipBaseline {
        brand_substr: "Apple M4 Max",
        footprint_ceiling_mib: 2300,
        tok_s_floor: 25.0,
        source: "this port, measured locally -- NOT a Swift baseline",
    },
    // M2 8GB: peak footprint 1,776-1,971 MiB, decode 5.10-6.30 tok/s.
    ChipBaseline {
        brand_substr: "Apple M2",
        footprint_ceiling_mib: 2070,
        tok_s_floor: 5.1,
        source: SWIFT_DOCS,
    },
];

/// Unknown chip: memory parity is chip-independent, so hold the loosest
/// documented ceiling; throughput is only reported.
const UNKNOWN_CHIP_FOOTPRINT_CEILING_MIB: u64 = 2250;

fn install_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("MREFRUST_GEMMA4_INSTALL_DIR").map(std::path::PathBuf::from)
}

#[test]
#[ignore = "needs a real ~14.6 GB Gemma 4 .gturbo install (MREFRUST_GEMMA4_INSTALL_DIR)"]
fn real_install_peak_footprint_and_throughput_meet_swift_baselines() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "memory_oracle: MREFRUST_GEMMA4_INSTALL_DIR is not set; skipping. \
             Point it at a repacked Gemma 4 .gturbo install to run the oracle."
        );
        return;
    };

    let brand = chip_brand_string();
    let baseline = brand
        .as_deref()
        .and_then(|b| BASELINES.iter().find(|row| b.contains(row.brand_substr)));
    match baseline {
        Some(row) => eprintln!(
            "memory_oracle: chip {:?} -> ceiling {} MiB, tok/s floor {} (source: {})",
            brand, row.footprint_ceiling_mib, row.tok_s_floor, row.source
        ),
        None => eprintln!(
            "memory_oracle: chip {brand:?} not in the baseline table -> ceiling \
             {UNKNOWN_CHIP_FOOTPRINT_CEILING_MIB} MiB, tok/s reported but not asserted"
        ),
    }

    let (mut runner, tokenizer) =
        open_model_runner(&dir, PROTOCOL_EXPERT_CACHE_SLOTS).expect("real install should open");
    let mut sampler = AppMemorySampler::new();

    let mut measured = Vec::new();
    for case in &PROTOCOL_CASES {
        // Frozen protocol: one discarded warmup, then the measured run.
        run_protocol_case(&mut runner, &tokenizer, case, &mut sampler)
            .unwrap_or_else(|e| panic!("{} warmup failed: {e}", case.id));
        let result = run_protocol_case(&mut runner, &tokenizer, case, &mut sampler)
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
        run_protocol_case(&mut runner, &tokenizer, warm_case, &mut sampler)
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
    let ceiling_mib = baseline.map_or(UNKNOWN_CHIP_FOOTPRINT_CEILING_MIB, |row| {
        row.footprint_ceiling_mib
    });
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
