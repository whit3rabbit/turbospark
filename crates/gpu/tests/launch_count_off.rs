//! Without `TURBOSPARK_LAUNCH_COUNT=1` nothing is counted.
#![cfg(target_os = "macos")]

use turbospark_gpu::{autorelease_pool, launch_count, MetalContext};

#[test]
fn counting_is_off_by_default() {
    std::env::remove_var("TURBOSPARK_LAUNCH_COUNT");
    autorelease_pool(|| {
        let context = MetalContext::new().expect("Metal device");
        context.begin_pass().commit_and_wait();
        assert!(!launch_count::enabled());
        assert_eq!(launch_count::take(), launch_count::LaunchCounts::default());
    });
}
