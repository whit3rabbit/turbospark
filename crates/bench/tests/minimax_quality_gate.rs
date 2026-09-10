#![cfg(target_os = "macos")]
//! Measure twice before adding a frozen MiniMax baseline row.
mod quality_common;
#[test]
#[ignore = "needs the pinned MiniMax Q4_K_M install and release build"]
fn minimax_quality_and_determinism() {
    let dir =
        std::env::var_os("TURBOSPARK_MINIMAX_INSTALL_DIR").expect("set MiniMax install directory");
    assert_eq!(
        repack::peek_manifest_arch(&std::path::PathBuf::from(&dir)).unwrap(),
        model_io::minimax_m2(),
        "gate requires the witnessed original MiniMax-M2 shape"
    );
    // Close the template's forced-open thought before scoring plain prose.
    // Eight slots already equals top-k; a smaller pressure arm cannot run.
    quality_common::run_quality_gate_with_slots(
        &std::path::PathBuf::from(dir),
        &[],
        "</think>\n\n",
        8192,
        None,
        8,
    );
}
