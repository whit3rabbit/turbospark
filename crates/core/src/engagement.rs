//! Engagement lines: proof that an opt-in lane actually ran.
//!
//! An A/B arm sets a variable and trusts that the runtime took the lane.
//! Twice that trust was wrong (a shell split an arm's variables, and a
//! forced kernel also forced a path the arm did not mean to change), and the
//! measurement read a real effect as noise. With `TURBOSPARK_ENGAGEMENT=1`
//! each lane prints `[engaged] NAME` to stderr the first time it runs, and
//! the lab refuses an arm whose log lacks the line its knob promises
//! (`crates/lab/src/knobs.rs`, evaluator item L3 in
//! `docs/MLX_SERVE_KERNELS.md`).
//!
//! Off by default, and cheap either way: one relaxed atomic load per call
//! site after the first, no lock, no allocation, nothing on the GPU.

use std::sync::OnceLock;

/// `TURBOSPARK_ENGAGEMENT=1`, read once.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("TURBOSPARK_ENGAGEMENT").as_deref() == Ok("1"))
}

/// The line a lane prints; one place so the lab and the runtime agree.
pub fn line(name: &str) -> String {
    format!("[engaged] {name}")
}

/// Mark that the lane `$name` (a string literal) ran. Prints once per call
/// site per process, and only under `TURBOSPARK_ENGAGEMENT=1`.
#[macro_export]
macro_rules! engaged {
    ($name:literal) => {{
        static SEEN: ::std::sync::atomic::AtomicBool = ::std::sync::atomic::AtomicBool::new(false);
        if !SEEN.load(::std::sync::atomic::Ordering::Relaxed)
            && $crate::engagement::enabled()
            && !SEEN.swap(true, ::std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!("{}", $crate::engagement::line($name));
        }
    }};
}
