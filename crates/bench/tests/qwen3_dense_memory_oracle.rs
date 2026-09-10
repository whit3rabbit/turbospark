#![cfg(target_os = "macos")]
//! Completion and warm-session memory regression for the small dense checkpoint.
mod oracle_common;
mod qwen3_dense_common;
#[test]
#[ignore = "needs pinned Qwen3-0.6B Q8_0 install and release build"]
fn qwen3_dense_memory_and_complete_answers() {
    let dir = qwen3_dense_common::install_dir();
    let ceiling = std::env::var("TURBOSPARK_QWEN3_DENSE_CEILING_MIB")
        .expect("set explicit provisional memory ceiling in MiB")
        .parse()
        .unwrap();
    // Reasoning completion lengths are measured before a budget is frozen.
    oracle_common::run_oracle_with_budget(&dir, &[], ceiling, 8192, 4096);
}
