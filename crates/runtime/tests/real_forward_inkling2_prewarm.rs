//! The open-time GPU touch of an Inkling 2 resident install
//! (`RealForwardRunner::inkling2_prewarm`), on a real install.
//!
//! ```text
//! INKLING2_INSTALL=~/models/inkling2-mini-gturbo cargo test --release \
//!     -p turbospark-runtime --test real_forward_inkling2_prewarm -- --ignored --nocapture
//! ```
//!
//! The first command buffer that references the resident experts waits 1 to
//! 3 s before its GPU work starts while the driver maps them. The prewarm
//! submits a touch at open without waiting, so an idle gap after open (the app
//! waiting for a user's first message) absorbs that cost. This opens, idles
//! for a few seconds, and times the first token, with the prewarm on and then
//! off (each in a fresh runner; the switch is read at open). The bound is
//! loose on purpose: with the touch on, the first token after the gap must
//! cost under a second, where without it this machine measured 1.5 to 2.8 s.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use half::f16;
use turbospark_runtime::{LogitProducer, RealForwardRunner};

fn first_token_after_idle(install: &std::path::Path, prewarm: bool) -> Duration {
    std::env::set_var(
        "TURBOSPARK_INKLING2_PREWARM",
        if prewarm { "1" } else { "0" },
    );
    let arch = model_io::inkling2_mini();
    let vocab = arch.vocab_size as usize;
    let mut runner = RealForwardRunner::open_with_residency(
        install,
        arch,
        2048,
        turbospark_runtime::ExpertCacheSlots::Auto,
        turbospark_runtime::DraftPolicies::off(),
        turbospark_runtime::SteeringPolicy::off(),
        1,
        turbospark_runtime::KvQuant::Off,
        turbospark_runtime::ExpertResidency::Resident,
    )
    .expect("open resident");
    // The user typing their first message.
    std::thread::sleep(Duration::from_secs(5));
    let mut row = vec![f16::ZERO; vocab];
    let t = Instant::now();
    runner.produce(1000, 0, &mut row).expect("produce");
    t.elapsed()
}

#[test]
#[ignore = "needs INKLING2_INSTALL (a real Inkling 2 install, resident experts)"]
fn prewarm_hides_the_first_pass_stall_behind_an_idle_gap() {
    let install = PathBuf::from(std::env::var("INKLING2_INSTALL").expect("INKLING2_INSTALL"));
    let warm = first_token_after_idle(&install, true);
    let cold = first_token_after_idle(&install, false);
    eprintln!(
        "first token after a 5 s idle gap: prewarm {:.0} ms, no prewarm {:.0} ms",
        warm.as_secs_f64() * 1e3,
        cold.as_secs_f64() * 1e3
    );
    assert!(
        warm < Duration::from_secs(1),
        "the first token still paid the mapping stall: {warm:?}"
    );
}
