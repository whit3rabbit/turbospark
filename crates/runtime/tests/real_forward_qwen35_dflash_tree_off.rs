#![cfg(target_os = "macos")]
//! `TURBOSPARK_DFLASH_TREE=0` keeps the chain. Its own binary because the
//! switch is read once per process and `real_forward_qwen35_dflash_tree.rs`
//! runs the default (trees).

use turbospark_repack::{
    build_synthetic_qwen_gdn_dense_install_with_dflash_wide_heads, peek_manifest_arch,
};
use turbospark_runtime::{DflashDraftPolicy, DraftPolicies, MtpDraftPolicy, RealForwardRunner};

#[test]
fn the_opt_out_refuses_tree_rounds() {
    let dir =
        std::env::temp_dir().join(format!("turbospark-dflash-tree-off-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    build_synthetic_qwen_gdn_dense_install_with_dflash_wide_heads(
        &dir,
        248_100,
        64,
        "dflash-tree-off",
        4,
    )
    .expect("the dense install with a DFlash2 drafter writes");
    // SAFETY (edition 2021: not unsafe): the only test in this binary, set
    // before the runner reads the switch.
    std::env::set_var("TURBOSPARK_DFLASH_TREE", "0");
    let arch = peek_manifest_arch(&dir).expect("manifest peeks");
    let runner = RealForwardRunner::open_with_options_and_speculation(
        &dir,
        arch,
        4096,
        16,
        DraftPolicies {
            mtp: MtpDraftPolicy::Off,
            dflash: DflashDraftPolicy::Fixed(7),
        },
    )
    .expect("the dflash install opens");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(runner.dflash_speculation_blocker().is_none());
    assert_eq!(runner.dflash_tree_nodes(), None);
}
