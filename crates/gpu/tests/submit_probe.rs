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
    // Every other way a decode waits: the first version hooked only the two
    // above and read nothing on a real model.
    context.begin_pass().commit().wait_ref();
    context.begin_pass().commit().wait_with_gpu_time();
    context.begin_pass().commit_and_wait_with_gpu_time();
    context
        .begin_pass()
        .commit_and_wait_checked()
        .expect("empty pass completes");
    let delays = submit_probe::take();
    assert_eq!(delays.len(), 6, "{delays:?}");
    assert!(
        delays.iter().all(|d| *d >= 0.0 && *d < 10_000.0),
        "{delays:?}"
    );
    assert!(submit_probe::take().is_empty(), "taken once");

    submit_probe::remove();
    context.begin_pass().commit_and_wait();
    assert!(submit_probe::take().is_empty());
}
