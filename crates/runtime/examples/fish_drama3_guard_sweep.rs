//! Fish Audio Drama 3 pause guard sweep: how often the guard brings trapped
//! takes back under different restore points, trigger lengths and retry
//! budgets.
//!
//! ```sh
//! cargo run --release -p turbospark-runtime --example fish_drama3_guard_sweep -- \
//!     <install-dir> <trapped|heldout> <off|hold:retry:retries> [...]
//! ```
//!
//! Speaks a set of takes through `FishDrama3Runner` once per guard setting,
//! with the shipped voiced table and `resume_frames`. `off` runs without the
//! guard. A take recovers when every chunk ends by EOS and the runner does not
//! report it stopped early. Prints each take's chunks, frames and retries,
//! then a line per setting.
//!
//! - `trapped`: [`TRAPS`], takes that trap without the guard: the pause-trap
//!   gate's four and four from the voiced-table training run. The guard's
//!   settings were first compared on these.
//! - `heldout`: [`HELDOUT`], seeds and texts no setting was chosen on. Some
//!   trap without the guard, some do not, so `off` says which, and a setting
//!   is judged on the takes `off` leaves stopped early.

#![cfg(target_os = "macos")]

use audio::tts::fish_drama3::generate::PauseGuard;
use turbospark_runtime::fish_drama3::{
    Drama3Request, FishDrama3Runner, DEFAULT_EXPERT_SLOTS, MAX_CONTEXT,
};
use turbospark_runtime::{ExpertCacheSlots, ExpertResidency};

/// Takes that trap without the guard, with their seeds.
const TRAPS: [(&str, u64); 8] = [
    (
        "<|speaker:0|> Are you coming tonight? <|speaker:1|> I don't know yet. \
         <|speaker:0|> Let me know by six.",
        3,
    ),
    (
        "[laughs] Oh, that is the best thing I have heard all week. [pause] Seriously.",
        0,
    ),
    (
        "[laughs] Oh, that is the best thing I have heard all week. [pause] Seriously.",
        2,
    ),
    (
        "[laughs] Oh, that is the best thing I have heard all week. [pause] Seriously.",
        3,
    ),
    (
        "<|speaker:0|> Have you seen my glasses? <|speaker:1|> They are on your head.",
        1,
    ),
    (
        "<|speaker:0|> Is the soup ready? <|speaker:1|> Almost. Give it ten more minutes.",
        2,
    ),
    ("Wait for it. [pause] There it is.", 0),
    ("Wait for it. [pause] There it is.", 2),
];

/// New seeds of the gate's trapping texts and two texts no run has seen.
fn heldout() -> Vec<(&'static str, u64)> {
    const PAUSE: &str =
        "[laughs] Oh, that is the best thing I have heard all week. [pause] Seriously.";
    const TURNS: &str = "<|speaker:0|> Are you coming tonight? <|speaker:1|> I don't know yet. \
                         <|speaker:0|> Let me know by six.";
    const SIGH: &str = "[sighs] I did not expect that. [pause] But here we are.";
    const READY: &str = "<|speaker:0|> Ready? <|speaker:1|> Not yet. <|speaker:0|> Take your time.";
    let mut takes = Vec::new();
    takes.extend((6..18).map(|seed| (PAUSE, seed)));
    takes.extend((6..12).map(|seed| (TURNS, seed)));
    takes.extend((0..6).map(|seed| (SIGH, seed)));
    takes.extend((0..6).map(|seed| (READY, seed)));
    takes
}

fn parse(setting: &str) -> Result<Option<PauseGuard>, String> {
    if setting == "off" {
        return Ok(None);
    }
    let parts: Vec<usize> = setting
        .split(':')
        .map(str::parse)
        .collect::<Result<_, _>>()
        .map_err(|e| format!("{setting:?}: {e}"))?;
    let [hold_after, retry_after, max_retries] = parts[..] else {
        return Err(format!("{setting:?}: want hold:retry:retries"));
    };
    Ok(Some(PauseGuard {
        hold_after,
        retry_after,
        max_retries,
        ..PauseGuard::MEASURED
    }))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let install = std::path::PathBuf::from(args.next().ok_or("install dir")?);
    let takes: Vec<(&str, u64)> = match args.next().as_deref() {
        Some("trapped") => TRAPS.to_vec(),
        Some("heldout") => heldout(),
        other => return Err(format!("take set {other:?}: trapped or heldout").into()),
    };
    let settings = args.collect::<Vec<_>>();
    let guards = settings
        .iter()
        .map(|a| parse(a))
        .collect::<Result<Vec<_>, _>>()?;
    if guards.is_empty() {
        return Err("give at least one off or hold:retry:retries setting".into());
    }
    let mut runner = FishDrama3Runner::open(
        &install,
        ExpertCacheSlots::Fixed(DEFAULT_EXPERT_SLOTS),
        ExpertResidency::Streamed,
        MAX_CONTEXT,
    )?;
    for (name, guard) in settings.iter().zip(guards) {
        runner.set_pause_guard(guard);
        let (mut recovered, mut retries, mut frames) = (0, 0, 0);
        for &(text, seed) in &takes {
            let request = Drama3Request {
                text,
                reference: None,
                seed,
                sampling: Default::default(),
            };
            let take = runner.synthesize(&request, &|| false, |_| true)?;
            let ok = !take.stopped_early();
            recovered += usize::from(ok);
            let chunks: Vec<String> = take
                .chunks
                .iter()
                .map(|c| {
                    retries += c.retries;
                    frames += c.frames;
                    format!(
                        "{}/{} {:?} {} retries",
                        c.frames, c.limit, c.stop, c.retries
                    )
                })
                .collect();
            println!(
                "{name} seed {seed}: {chunks:?}, longest silence {}, recovered {ok}: {text}",
                take.longest_silence_frames
            );
        }
        println!(
            "setting {name}: {recovered} of {} recovered, {retries} retries, {frames} kept frames",
            takes.len()
        );
    }
    Ok(())
}
