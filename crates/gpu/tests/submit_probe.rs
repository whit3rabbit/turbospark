//! The submit probe on a real device: a delay per waited buffer while
//! installed, nothing otherwise.

#![cfg(target_os = "macos")]

use turbospark_gpu::{submit_probe, MetalContext};

#[test]
fn waited_buffers_are_timed_only_while_the_probe_is_installed() {
    let context = MetalContext::new().expect("metal");
    context.begin_pass().commit_and_wait();
    assert!(submit_probe::take().is_empty(), "nothing installed");

    submit_probe::install();
    context.begin_pass().commit_and_wait();
    context.begin_pass().commit().wait();
    let delays = submit_probe::take();
    assert_eq!(delays.len(), 2, "{delays:?}");
    assert!(
        delays.iter().all(|d| *d >= 0.0 && *d < 10_000.0),
        "{delays:?}"
    );
    assert!(submit_probe::take().is_empty(), "taken once");

    submit_probe::remove();
    context.begin_pass().commit_and_wait();
    assert!(submit_probe::take().is_empty());
}
