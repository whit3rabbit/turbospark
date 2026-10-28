//! The `lfm3` family at the runtime boundary, on a synthetic install.
//!
//! Phase 1 of `docs/NEW_MODEL.md`: the install the real GGUF walk wrote must
//! get through `RealForwardRunner::open`'s manifest, arch and resident-index
//! checks. There is no decode flow yet (`families/lfm3/` is Phase 3), so the
//! only acceptable outcome is the deliberate by-name refusal at family
//! dispatch -- any EARLIER error means the install itself is wrong, and a
//! success would mean some neighbour's flow had claimed the family.

#![cfg(target_os = "macos")]

use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_repack::build_synthetic_lfm3_install;
use turbospark_runtime::RealForwardRunner;

fn tempdir() -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("turbospark-lfm3-{}-{unique}", std::process::id()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn the_synthetic_install_reaches_the_family_refusal() {
    let dir = tempdir();
    let arch = build_synthetic_lfm3_install(&dir, 64, 6, "lfm3-toy").expect("the install writes");
    let err = match RealForwardRunner::open(&dir, arch) {
        Ok(_) => panic!("lfm3 opened with no decode flow: a neighbour's flow claimed it"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("the lfm3 family has no decode flow yet"),
        "open failed before family dispatch, so the install is wrong: {err}"
    );
}
