#![cfg(all(target_os = "macos", feature = "ane-prefill"))]
//! Without `TURBOSPARK_ANE_PREFILL_DFLASH=1` an open DFlash2 drafter keeps
//! the ANE prefill split off, whatever the split's mode: the app defaults the
//! split to `.auto`, and the pairing has not passed its gates. Its own test
//! binary because the switch is read once per process, and
//! `real_forward_qwen35_dflash_ane_split.rs` sets it.

use turbospark_repack::{build_synthetic_qwen_gdn_dense_install_with_dflash, peek_manifest_arch};
use turbospark_runtime::{
    AnePrefillMode, DflashDraftPolicy, DraftPolicies, MtpDraftPolicy, RealForwardRunner,
    SpeculativeProducer,
};

#[test]
fn an_open_drafter_keeps_the_split_off_by_default() {
    if std::env::var("TURBOSPARK_ANE_PREFILL_DFLASH").as_deref() == Ok("1") {
        eprintln!("TURBOSPARK_ANE_PREFILL_DFLASH=1 is set for this run; nothing to check");
        return;
    }
    let dir = std::env::temp_dir().join(format!(
        "turbospark-dflash-ane-default-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    build_synthetic_qwen_gdn_dense_install_with_dflash(&dir, 248_100, 64, "dflash-ane-default", 4)
        .expect("the dense install with a DFlash2 drafter writes");
    let arch = peek_manifest_arch(&dir).expect("manifest peeks");
    let mut r = RealForwardRunner::open_with_options_and_speculation(
        &dir,
        arch,
        4096,
        16,
        DraftPolicies {
            mtp: MtpDraftPolicy::Off,
            dflash: DflashDraftPolicy::Fixed(2),
        },
    )
    .expect("the install opens");
    let _ = std::fs::remove_dir_all(&dir);
    for mode in [AnePrefillMode::Split, AnePrefillMode::Break] {
        r.set_ane_prefill(mode, None);
        assert_eq!(
            r.drafter_prefill_chunk(),
            foundation::DEFAULT_CHUNK_SIZE as usize,
            "{mode:?}: the speculative loop must keep the default span"
        );
    }
}
