#![cfg(target_os = "macos")]
//! The pause-trap gate for Fish Audio Drama 3, against a REAL install.
//!
//! Some takes speak the first sentence, turn or the words before a `[pause]`
//! and then only silence until the chunk's frame cap
//! (`docs/FISH_DRAMA3_PHASE0.md`, sampling recipe). `FishDrama3Runner`
//! reports such a take as stopped early (`Drama3Take::stopped_early`: a chunk
//! that ends without EOS, or a silent run no completed take has). This gate
//! speaks the three texts that trapped the most takes at fixed seeds and
//! holds the count of takes reported stopped early, per configuration, to
//! its frozen measurement. A seed fixes the codes, so on one install each
//! count is exact: a change that raises one is a regression in the recipe,
//! the loop, the guard or the numerics. A request that fails is a failure
//! too, reported after every count.
//!
//! The default run is the product configuration: Fish Audio's recipe with the
//! pause guard on. Two variables select the others, for a before and after:
//! `TURBOSPARK_DRAMA3_GATE_RECIPE=reference` (mlx-audio's 0.7/0.7/30, the
//! default before Fish Audio's) and `TURBOSPARK_DRAMA3_GATE_GUARD=off`
//! (`FishDrama3Runner::set_pause_guard(None)`). Reference and guard off is
//! the old loop.
//!
//!   TURBOSPARK_FISH_DRAMA3_INSTALL_DIR=~/models/fish-drama3-4bit \
//!     cargo test -p turbospark-bench --test fish_drama3_pause_trap_gate --release -- --ignored --nocapture

use runtime::fish_drama3::{
    Drama3Request, Drama3Sampling, FishDrama3Runner, DEFAULT_EXPERT_SLOTS, MAX_CONTEXT,
};
use runtime::{ExpertCacheSlots, ExpertResidency};

/// A sentence gap, two turn changes, and a `[pause]` cue: a short name, the
/// text and its seeds.
const CASES: [(&str, &str, std::ops::Range<u64>); 3] = [
    (
        "voiceover",
        "Welcome back. This is a short voiceover test with no reference voice.",
        0..12,
    ),
    (
        "dialogue",
        "<|speaker:0|> Are you coming tonight? <|speaker:1|> I don't know yet. \
         <|speaker:0|> Let me know by six.",
        0..6,
    ),
    (
        "pause cue",
        "[laughs] Oh, that is the best thing I have heard all week. [pause] Seriously.",
        0..6,
    ),
];

/// Takes of [`CASES`] reported stopped early, per configuration, on the
/// pinned 4-bit install at 16 streamed slots (M4 Max, 2026-10-18). `(recipe,
/// guard on, frozen count)`. Per line (voiceover, dialogue, pause cue) they
/// measured 0/0/0, 0/1/3, 1/0/1 and 4/4/4.
const FROZEN: [(&str, bool, usize); 4] = [
    ("fish", true, 0),
    ("fish", false, 4),
    ("reference", true, 2),
    ("reference", false, 12),
];

#[test]
#[ignore = "needs a real Drama 3 install via TURBOSPARK_FISH_DRAMA3_INSTALL_DIR"]
fn fish_drama3_pause_traps_stay_at_the_frozen_count() {
    let Ok(dir) = std::env::var("TURBOSPARK_FISH_DRAMA3_INSTALL_DIR") else {
        eprintln!("skipping: TURBOSPARK_FISH_DRAMA3_INSTALL_DIR is unset");
        return;
    };
    let recipe = std::env::var("TURBOSPARK_DRAMA3_GATE_RECIPE").unwrap_or_else(|_| "fish".into());
    let sampling = match recipe.as_str() {
        "fish" => Drama3Sampling::default(),
        // `SamplingParams::REFERENCE`, which the bench crate cannot name.
        "reference" => Drama3Sampling {
            temperature: Some(0.7),
            top_p: Some(0.7),
            top_k: Some(30),
            max_frames: None,
        },
        other => panic!("TURBOSPARK_DRAMA3_GATE_RECIPE={other:?}: fish or reference"),
    };
    let guard = match std::env::var("TURBOSPARK_DRAMA3_GATE_GUARD").as_deref() {
        Err(_) | Ok("on") => true,
        Ok("off") => false,
        Ok(other) => panic!("TURBOSPARK_DRAMA3_GATE_GUARD={other:?}: on or off"),
    };
    let frozen = FROZEN
        .iter()
        .find(|(r, g, _)| *r == recipe && *g == guard)
        .map(|(_, _, n)| *n)
        .expect("every recipe and guard pair has a frozen count");
    let config = format!(
        "recipe {recipe}, guard {}",
        if guard { "on" } else { "off" }
    );

    let mut runner = FishDrama3Runner::open(
        std::path::Path::new(&dir),
        ExpertCacheSlots::Fixed(DEFAULT_EXPERT_SLOTS),
        ExpertResidency::Streamed,
        MAX_CONTEXT,
    )
    .expect("the real install should open");
    if !guard {
        runner.set_pause_guard(None);
    }
    let mut stopped = Vec::new();
    let mut failed = Vec::new();
    let mut takes = 0usize;
    let mut per_line = Vec::new();
    for (name, text, seeds) in CASES {
        let (mut line_stopped, mut line_takes) = (0usize, 0usize);
        for seed in seeds {
            let request = Drama3Request {
                text,
                reference: None,
                seed,
                sampling,
            };
            takes += 1;
            line_takes += 1;
            let take = match runner.synthesize(&request, &|| false, |_| true) {
                Ok(take) => take,
                Err(e) => {
                    eprintln!("pause gate [{config}]: {name} seed {seed:>2} FAILED: {e}");
                    failed.push((name, seed, e));
                    continue;
                }
            };
            let chunks: Vec<String> = take
                .chunks
                .iter()
                .map(|c| {
                    format!(
                        "{}/{} {:?} {} retries",
                        c.frames, c.limit, c.stop, c.retries
                    )
                })
                .collect();
            eprintln!(
                "pause gate [{config}]: {name} seed {seed:>2} {chunks:?}, longest silence {} \
                 frames, stopped early {}",
                take.longest_silence_frames,
                take.stopped_early()
            );
            if take.stopped_early() {
                stopped.push((name, seed));
                line_stopped += 1;
            }
        }
        per_line.push(format!("{name} {line_stopped} of {line_takes}"));
    }
    eprintln!(
        "pause gate [{config}]: {} of {takes} takes stopped early (frozen {frozen}); {}; {} \
         failed",
        stopped.len(),
        per_line.join(", "),
        failed.len()
    );
    assert!(failed.is_empty(), "requests failed: {failed:?}");
    // At a frozen count of 0 this reads `len() <= 0`, which clippy flags. The
    // form stays so a later freeze above 0 needs only the constant changed.
    #[allow(clippy::absurd_extreme_comparisons)]
    let within = stopped.len() <= frozen;
    assert!(
        within,
        "[{config}] {} of {takes} takes stopped early, above the frozen {frozen}: {stopped:?}",
        stopped.len()
    );
}
