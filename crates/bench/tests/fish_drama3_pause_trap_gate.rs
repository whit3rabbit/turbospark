#![cfg(target_os = "macos")]
//! The pause-trap gate for Fish Audio Drama 3, against a REAL install.
//!
//! Some takes speak the first sentence, turn or the words before a `[pause]`
//! and then only silence until the chunk's frame cap
//! (`docs/FISH_DRAMA3_PHASE0.md`, sampling recipe). `FishDrama3Runner`
//! reports such a take as stopped early. This gate speaks the three texts
//! that trapped the most takes, at fixed seeds and the default recipe, and
//! holds the count of takes reported stopped early to the frozen
//! measurement. A seed fixes the codes, so on one install the count is
//! exact: a change that raises it is a regression in the recipe, the loop or
//! the numerics. mlx-audio's recipe (`SamplingParams::REFERENCE`) traps more
//! than the frozen count at these seeds, so reverting the default fails here.
//!
//!   TURBOSPARK_FISH_DRAMA3_INSTALL_DIR=~/models/fish-drama3-4bit \
//!     cargo test -p turbospark-bench --test fish_drama3_pause_trap_gate --release -- --ignored --nocapture

use runtime::fish_drama3::{Drama3Request, FishDrama3Runner, DEFAULT_EXPERT_SLOTS, MAX_CONTEXT};
use runtime::{ExpertCacheSlots, ExpertResidency};

/// A sentence gap, two turn changes, and a `[pause]` cue, with their seeds.
const CASES: [(&str, std::ops::Range<u64>); 3] = [
    (
        "Welcome back. This is a short voiceover test with no reference voice.",
        0..12,
    ),
    (
        "<|speaker:0|> Are you coming tonight? <|speaker:1|> I don't know yet. \
         <|speaker:0|> Let me know by six.",
        0..6,
    ),
    (
        "[laughs] Oh, that is the best thing I have heard all week. [pause] Seriously.",
        0..6,
    ),
];

/// Takes of [`CASES`] reported stopped early under the default recipe, on
/// the pinned 4-bit install at 16 streamed slots (M4 Max, 2026-10-16).
const FROZEN_STOPPED: usize = 4;

#[test]
#[ignore = "needs a real Drama 3 install via TURBOSPARK_FISH_DRAMA3_INSTALL_DIR"]
fn fish_drama3_pause_traps_stay_at_the_frozen_count() {
    let Ok(dir) = std::env::var("TURBOSPARK_FISH_DRAMA3_INSTALL_DIR") else {
        eprintln!("skipping: TURBOSPARK_FISH_DRAMA3_INSTALL_DIR is unset");
        return;
    };
    let mut runner = FishDrama3Runner::open(
        std::path::Path::new(&dir),
        ExpertCacheSlots::Fixed(DEFAULT_EXPERT_SLOTS),
        ExpertResidency::Streamed,
        MAX_CONTEXT,
    )
    .expect("the real install should open");
    let mut stopped = Vec::new();
    let mut takes = 0usize;
    for (text, seeds) in CASES {
        for seed in seeds {
            let request = Drama3Request {
                text,
                reference: None,
                seed,
                sampling: Default::default(),
            };
            let take = runner
                .synthesize(&request, &|| false, |_| true)
                .expect("a Drama 3 request should synthesize");
            takes += 1;
            let chunks: Vec<String> = take
                .chunks
                .iter()
                .map(|c| format!("{}/{} {:?}", c.frames, c.limit, c.stop))
                .collect();
            eprintln!(
                "pause gate: seed {seed:>2} {chunks:?}, longest silence {} frames, stopped early \
                 {}: {text}",
                take.longest_silence_frames,
                take.stopped_early()
            );
            if take.stopped_early() {
                stopped.push((seed, text));
            }
        }
    }
    eprintln!(
        "pause gate: {} of {takes} takes stopped early (frozen {FROZEN_STOPPED})",
        stopped.len()
    );
    assert!(
        stopped.len() <= FROZEN_STOPPED,
        "{} of {takes} takes stopped early, above the frozen {FROZEN_STOPPED}: {stopped:?}",
        stopped.len()
    );
}
