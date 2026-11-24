#![cfg(target_os = "macos")]
//! Not a gate: the real GPU time of `inkling2_router_select_k8` at decode,
//! 48 dispatches a token (one per layer application) at the Mini's 128
//! experts, in ONE pass (as decode encodes them), so a dispatch-profile
//! reading (one encoder per dispatch, which inflates small kernels) can be
//! priced against the 2% rule. The empty pass is subtracted.
//!
//! cargo test --release -p turbospark-gpu --test inkling2_router_select_cost -- --ignored --nocapture

use turbospark_gpu::{encode_inkling2_router_select, MetalContext};

#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn router_select_gpu_time_per_token() {
    let mut context = MetalContext::new().expect("Metal device");
    const E: usize = 128;
    const PER_TOKEN: usize = 48;
    const TOKENS: usize = 64;
    let logits: Vec<u8> = (0..E)
        .flat_map(|i| (((i * 37) % 101) as f32 * 0.01 - 0.5).to_le_bytes())
        .collect();
    let bias: Vec<u8> = (0..E)
        .flat_map(|i| ((i % 7) as f32 * 0.001).to_le_bytes())
        .collect();
    let lb = context.new_buffer_with_data(&logits);
    let bb = context.new_buffer_with_data(&bias);
    let sel = context.new_output_buffer(8 * 4);
    let w = context.new_output_buffer(8 * 2);
    let run = |context: &mut MetalContext, n: usize| -> f64 {
        let pass = context.begin_pass();
        for _ in 0..n {
            encode_inkling2_router_select(
                context,
                &pass,
                (&lb, 0),
                (&bb, 0),
                (&sel, 0),
                (&w, 0),
                E as u32,
                8.0,
            )
            .unwrap();
        }
        pass.commit_and_wait_with_gpu_time()
    };
    run(&mut context, PER_TOKEN * TOKENS);
    let best = |context: &mut MetalContext, n: usize| {
        (0..5).map(|_| run(context, n)).fold(f64::MAX, f64::min)
    };
    let full = best(&mut context, PER_TOKEN * TOKENS);
    let one = best(&mut context, 1);
    let per_token_ms = (full - one) / TOKENS as f64 * 1e3;
    println!(
        "router_select: {:.3} ms per token over {PER_TOKEN} dispatches ({:.2} us each)",
        per_token_ms,
        per_token_ms * 1e3 / PER_TOKEN as f64
    );
}
