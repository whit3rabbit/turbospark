//! The bounded keep-warm thread (KERN-05 Step 3, `src/keep_warm.rs`) on a
//! real device: it names its buffers while the owner is idle, stays silent
//! while the owner's context is busy, stops after the window, and joins on
//! drop. Timing bounds are loose on purpose; the decision rule itself is unit
//! tested in the module.
#![cfg(target_os = "macos")]

use std::time::{Duration, Instant};

use turbospark_gpu::{memory_pressure_raw, KeepWarm, KeepWarmConfig, MetalContext};

fn cfg(every_ms: u64, window_ms: u64) -> KeepWarmConfig {
    KeepWarmConfig {
        every_ms,
        window_ms,
    }
}

fn buffer(context: &MetalContext, len: u64) -> metal::Buffer {
    context
        .device()
        .new_buffer(len, metal::MTLResourceOptions::StorageModeShared)
}

/// The fire assertions need the kernel's "normal" verdict; under pressure
/// the thread must pause instead, which is asserted in its place.
fn pressure_normal() -> bool {
    memory_pressure_raw() == 1
}

#[test]
fn fires_while_the_owner_is_idle_and_stops_after_the_window() {
    let context = MetalContext::new().expect("Metal device available on this machine");
    // A pass now starts the window from here.
    context.begin_pass().commit_and_wait();
    let kw = KeepWarm::start(
        &context,
        vec![buffer(&context, 64), buffer(&context, 4096)],
        cfg(50, 1000),
    )
    .expect("start");
    std::thread::sleep(Duration::from_millis(1600));
    let s = kw.stats();
    assert_eq!(s.failed, 0, "{s:?}");
    if !pressure_normal() {
        assert!(s.paused > 0 && s.fired == 0, "pressure must pause: {s:?}");
        return;
    }
    // About 1000 / 50 = 20 ticks fit in the window; scheduling can lose some.
    assert!((5..=22).contains(&s.fired), "{s:?}");
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(kw.stats().fired, s.fired, "fired after the window closed");
}

#[test]
fn a_busy_context_keeps_it_silent_and_idle_resumes_it() {
    let context = MetalContext::new().expect("Metal device available on this machine");
    let kw =
        KeepWarm::start(&context, vec![buffer(&context, 64)], cfg(100, 10_000)).expect("start");
    let busy_until = Instant::now() + Duration::from_millis(600);
    while Instant::now() < busy_until {
        context.begin_pass().commit_and_wait();
        std::thread::sleep(Duration::from_millis(10));
    }
    let s = kw.stats();
    assert_eq!(s.fired, 0, "fired while the owner was busy: {s:?}");
    assert_eq!(
        s.paused, 0,
        "sampled pressure while the owner was busy: {s:?}"
    );
    std::thread::sleep(Duration::from_millis(600));
    let s = kw.stats();
    if pressure_normal() {
        assert!(s.fired >= 2, "did not resume once idle: {s:?}");
    } else {
        assert!(s.paused > 0, "{s:?}");
    }
}

#[test]
fn drop_joins_promptly() {
    let context = MetalContext::new().expect("Metal device available on this machine");
    let kw =
        KeepWarm::start(&context, vec![buffer(&context, 64)], cfg(5000, 3_600_000)).expect("start");
    let t = Instant::now();
    drop(kw);
    assert!(
        t.elapsed() < Duration::from_millis(500),
        "{:?}",
        t.elapsed()
    );
}

#[test]
fn start_refuses_no_buffers_and_a_buffer_without_a_word() {
    let context = MetalContext::new().expect("Metal device available on this machine");
    assert!(KeepWarm::start(&context, Vec::new(), cfg(750, 120_000)).is_err());
    assert!(KeepWarm::start(&context, vec![buffer(&context, 2)], cfg(750, 120_000)).is_err());
}
