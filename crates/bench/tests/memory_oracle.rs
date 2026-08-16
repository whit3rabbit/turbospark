#![cfg(target_os = "macos")]
//! The memory oracle: runs the frozen community-protocol benchmark
//! against a REAL Gemma 4 `.gturbo` install and asserts this port stays
//! at or under the published Swift baselines.
//!
//! Not run by default (needs a ~14.6 GB install and takes minutes; use
//! --release or the tok/s numbers are meaningless):
//!
//!   TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
//!     cargo test -p turbospark-bench --test memory_oracle --release -- --ignored --nocapture
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
//!
//! The body lives in `oracle_common` because Qwen 3.6 has its own target
//! (`qwen36_memory_oracle.rs`) with its own ceiling. Same reason as
//! above, one level up: two families cannot share one session peak.

mod oracle_common;

/// Per-chip rows, MOST SPECIFIC SUBSTRING FIRST -- the lookup takes the
/// first `contains` hit, so "Apple M4 Max" must precede any future bare
/// "Apple M4" row, and "Apple M2" deliberately also matches M2 Pro/Max
/// (whose real floors are strictly higher, which is safe for a floor).
///
/// Ceilings are the measured peak range's high end + ~5%, the cross-run
/// variance band; more headroom would mask a regression on the order of
/// one KV layer. Floors are below the SLOWEST case, since one floor is
/// asserted against every case.
const BASELINES: &[oracle_common::ChipBaseline] = &[
    // M5 Pro 24GB: peak footprint 2,126-2,142 MiB, decode 31.01-35.17 tok/s.
    oracle_common::ChipBaseline {
        brand_substr: "Apple M5 Pro",
        footprint_ceiling_mib: 2250,
        tok_s_floor: 31.0,
        source: oracle_common::SWIFT_DOCS,
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
    // THE FLOOR WAS RE-EXAMINED 2026-08-16 AFTER THE HOST-SAMPLER FIX AND
    // DELIBERATELY LEFT AT 25.0. It had been flagged as "very loose, 25.0
    // against a slowest case reading 38.2". It is not loose; 38.2 was the
    // most favourable reading of the noisiest case. Four readings of the
    // same three cases, same binary, same install, same day:
    //
    //   short-explanation  43.960 / 45.648 / 45.254 / 44.6    spread  3.8%
    //   medium-review      40.798 / 41.841 / 41.496 / 40.8    spread  2.6%
    //   long-synthesis     35.234 / 33.039 / 37.559 / 35.8    spread 14.0%
    //
    // (first three this session -- one battery, two AC -- the fourth from
    // a concurrent session on AC. Peaks 2,104 / 2,175 / 2,155 / 2,174 MiB,
    // inside the documented 77 MiB expert-slot-warming band.)
    //
    // **THE CASE THAT GOVERNS THE FLOOR HAS FOUR TO FIVE TIMES THE
    // VARIANCE OF ITS SIBLINGS**, and that is structural rather than bad
    // luck: `long-synthesis` prefills 3,015 tokens for ~67 s and then
    // decodes only 599, so its short decode window sits downstream of a
    // long hot prefill and is the arm most exposed to thermal state and to
    // whatever else is on the machine. One floor is asserted against every
    // case, so it is this one's worst reading that sets it.
    //
    // 25.0 is therefore 0.757 x the WORST observed (33.039), which is a
    // tighter margin than the ~0.72 this row's earlier raises used, not a
    // looser one. Raising to 27.5 would be 0.83 x that reading -- inside
    // the case's own 14% spread, i.e. a test that fails on machine state.
    //
    // Nor would raising it buy anything. NO floor at a margin this case's
    // variance permits can catch losing a recent optimization: the
    // 2026-08-15 sampler fix is worth 9.4% at the protocol's sampled
    // settings, and losing it entirely lands `long-synthesis` near 32,
    // above any defensible floor. That is a statement about what a floor
    // is for rather than a defect -- this one exists to catch the
    // catastrophic classes (2026-08-06 split-KV, 2026-08-07 sampler, both
    // of which roughly halved throughput), and it still does.
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
    oracle_common::ChipBaseline {
        brand_substr: "Apple M4 Max",
        footprint_ceiling_mib: 2300,
        tok_s_floor: 25.0,
        source: "this port, measured locally -- NOT a Swift baseline",
    },
    // M2 8GB: peak footprint 1,776-1,971 MiB, decode 5.10-6.30 tok/s.
    oracle_common::ChipBaseline {
        brand_substr: "Apple M2",
        footprint_ceiling_mib: 2070,
        tok_s_floor: 5.1,
        source: oracle_common::SWIFT_DOCS,
    },
];

/// Unknown chip: memory parity is chip-independent, so hold the loosest
/// documented ceiling; throughput is only reported.
const UNKNOWN_CHIP_FOOTPRINT_CEILING_MIB: u64 = 2250;

fn install_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("TURBOSPARK_GEMMA4_INSTALL_DIR").map(std::path::PathBuf::from)
}

#[test]
#[ignore = "needs a real ~14.6 GB Gemma 4 .gturbo install (TURBOSPARK_GEMMA4_INSTALL_DIR)"]
fn real_install_peak_footprint_and_throughput_meet_swift_baselines() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "memory_oracle: TURBOSPARK_GEMMA4_INSTALL_DIR is not set; skipping. \
             Point it at a repacked Gemma 4 .gturbo install to run the oracle."
        );
        return;
    };
    oracle_common::run_oracle(&dir, BASELINES, UNKNOWN_CHIP_FOOTPRINT_CEILING_MIB);
}
