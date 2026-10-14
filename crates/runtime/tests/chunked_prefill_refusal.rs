#![cfg(target_os = "macos")]
//! The refuse-by-name contract for a chunked driver's INSTALL-level
//! refusals: `supports_chunked_prefill` and the driver's own refusal must
//! agree, so `--prefill-chunk`'s default routing falls back to sequential
//! silently while `prefill_chunk` itself, reached directly or via
//! `TURBOSPARK_PREFILL_CHUNK`, still names what it does not serve
//! (`crates/runtime/CLAUDE.md` Gotcha 14 / 7's A/B-seam contract).
//!
//! Until 2026-10-14 this file pinned that the whole qwen MoE flow
//! (`qwenGdnMoe`) was refused. It has a driver now
//! (`families/qwen/moe_prefill.rs`), and what is still refused at the install
//! level is `TURBOSPARK_QWEN_SHARED_READ_OVERLAP` and the driver's own kill
//! switch, `TURBOSPARK_QWEN_MOE_CHUNKED=0`, both read once at open. Each test
//! sets its variable under `ENV_LOCK`, so the two cannot leak into each other.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use half::f16;
use turbospark_repack::build_synthetic_qwen_gdn_moe_install;
use turbospark_runtime::{ChunkedPrefillRunner, RealForwardRunner};

/// Serializes the env mutations below: tests in one binary run in parallel.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn tempdir() -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "turbospark-chunked-refusal-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn the_shared_read_overlap_seam_is_refused_by_name_and_by_the_predicate() {
    let dir = tempdir();
    let arch = build_synthetic_qwen_gdn_moe_install(&dir, 128, 4, 8, "tiny-qwen-refusal")
        .expect("qwen install builds");
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("TURBOSPARK_QWEN_SHARED_READ_OVERLAP", "1");
    let opened = RealForwardRunner::open(&dir, arch);
    std::env::remove_var("TURBOSPARK_QWEN_SHARED_READ_OVERLAP");
    let mut runner = opened.expect("qwen install opens");
    assert!(
        !runner.supports_chunked_prefill(),
        "an install the MoE driver refuses must not report chunked-prefill support"
    );
    let mut logits = vec![f16::from_f32(0.0); 128];
    let err = runner
        .prefill_chunk(&[5, 9], 0, &mut logits)
        .expect_err("the driver must refuse by name, not fall back");
    assert!(
        err.contains("TURBOSPARK_QWEN_SHARED_READ_OVERLAP"),
        "the refusal must name the seam; got {err}"
    );
}

#[test]
fn the_kill_switch_falls_back_to_sequential_and_is_refused_by_name() {
    let dir = tempdir();
    let arch = build_synthetic_qwen_gdn_moe_install(&dir, 128, 4, 8, "tiny-qwen-killswitch")
        .expect("qwen install builds");
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("TURBOSPARK_QWEN_MOE_CHUNKED", "0");
    let opened = RealForwardRunner::open(&dir, arch);
    std::env::remove_var("TURBOSPARK_QWEN_MOE_CHUNKED");
    let mut runner = opened.expect("qwen install opens");
    // `false` is what makes the CLI and server take the sequential path
    // without an error, which is the switch's whole purpose.
    assert!(
        !runner.supports_chunked_prefill(),
        "TURBOSPARK_QWEN_MOE_CHUNKED=0 must turn the MoE driver off"
    );
    let mut logits = vec![f16::from_f32(0.0); 128];
    let err = runner
        .prefill_chunk(&[5, 9], 0, &mut logits)
        .expect_err("a direct call must still be refused by name");
    assert!(err.contains("TURBOSPARK_QWEN_MOE_CHUNKED"), "got {err}");
}
