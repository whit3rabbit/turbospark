//! `TURBOSPARK_LAUNCH_COUNT=1` on a real device: what is encoded is what is
//! counted. One test per binary, because the switch is read once per process
//! and the counters are process-wide.
#![cfg(target_os = "macos")]

use metal::FunctionConstantValues;
use turbospark_gpu::{autorelease_pool, dispatch_threads_3d, launch_count, MetalContext};

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void bump(device uint *x [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    x[i] += 1;
}
"#;

#[test]
fn counts_dispatches_command_buffers_and_routes() {
    std::env::set_var("TURBOSPARK_LAUNCH_COUNT", "1");
    assert!(launch_count::enabled());
    autorelease_pool(|| {
        let mut context = MetalContext::new().expect("Metal device");
        let constants = FunctionConstantValues::new();
        let plain = context
            .pipeline(SOURCE, "bump", &constants, &[])
            .expect("pipeline");
        // The same function under another constants key is another route.
        let keyed = context
            .pipeline(SOURCE, "bump", &constants, &[1])
            .expect("pipeline");
        let buffer = context.new_buffer_with_data(&[0u32; 4]);
        let _ = launch_count::take();

        // Two passes: 3 + 2 dispatches through each of the encode paths.
        let pass = context.begin_pass();
        pass.encode_threadgroups(&plain, &[(&buffer, 0, 0)], &[], 4, 1);
        pass.encode_threadgroups_3d(&plain, &[(&buffer, 0, 0)], &[], (4, 1, 1), (1, 1, 1));
        pass.encode_threads_3d(&plain, &[(&buffer, 0, 0)], &[], (4, 1, 1), (1, 1, 1));
        pass.commit_and_wait();
        let pass = context.begin_pass_concurrent_labeled("two");
        pass.encode_threads_3d(&keyed, &[(&buffer, 0, 0)], &[], (4, 1, 1), (1, 1, 1));
        pass.memory_barrier_with_buffers(&[&buffer]);
        pass.encode_threads_3d(&keyed, &[(&buffer, 0, 0)], &[], (4, 1, 1), (1, 1, 1));
        pass.commit_and_wait();
        // A one-shot helper is one more of each.
        dispatch_threads_3d(&context, &plain, &[(&buffer, 0)], &[], (4, 1, 1), (1, 1, 1));
        // The idle fence is not a launch.
        context.wait_until_idle();

        let counts = launch_count::take();
        assert_eq!(counts.dispatches, 6, "{counts:?}");
        assert_eq!(counts.command_buffers, 3, "{counts:?}");
        assert_eq!(counts.routes.len(), 2, "{counts:?}");
        assert!(
            counts.routes.iter().all(|r| r.starts_with("bump#")),
            "{counts:?}"
        );
        // Every dispatch above really ran.
        assert_eq!(
            turbospark_gpu::read_buffer_bytes(&buffer, 0, 4),
            6u32.to_le_bytes()
        );

        assert_eq!(launch_count::take(), launch_count::LaunchCounts::default());
    });
}
