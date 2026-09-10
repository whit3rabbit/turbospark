#![cfg(target_os = "macos")]
//! First-run MiniMax protocol. No throughput baseline exists yet.
mod oracle_common;
#[test]
#[ignore = "needs the pinned MiniMax Q4_K_M install and release build"]
fn minimax_memory_and_complete_answers() {
    let dir =
        std::env::var_os("TURBOSPARK_MINIMAX_INSTALL_DIR").expect("set MiniMax install directory");
    assert_eq!(
        repack::peek_manifest_arch(&std::path::PathBuf::from(&dir)).unwrap(),
        model_io::minimax_m2(),
        "gate requires the witnessed original MiniMax-M2 shape"
    );
    // An explicit run limit until two completed processes establish a baseline.
    let ceiling = std::env::var("TURBOSPARK_MINIMAX_CEILING_MIB")
        .expect("set an explicit provisional memory ceiling in MiB")
        .parse()
        .unwrap();
    oracle_common::run_oracle_over_cases_with_slots(
        &std::path::PathBuf::from(dir),
        &[],
        ceiling,
        8192,
        4096,
        &turbospark_bench::protocol::PROTOCOL_CASES,
        8,
    );
}
