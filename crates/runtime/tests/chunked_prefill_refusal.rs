#![cfg(target_os = "macos")]
//! One narrow regression check the widening in this session's other test
//! files does not cover: a family this session did NOT touch (the qwen
//! hybrid linear/full-attention flow, `QwenGdnMoe`) must still hard-refuse
//! `prefill_chunk` by name, exactly as it did before Gemma 4, `llama`,
//! `muse_glimmer` and `gpt-oss` were widened one by one. `--prefill-chunk`'s
//! own default routing falls back to sequential silently
//! (`supports_chunked_prefill`), but `prefill_chunk` itself, reached
//! directly or via `TURBOSPARK_PREFILL_CHUNK`, must still name the flow it
//! does not serve (`crates/runtime/CLAUDE.md` Gotcha 14 / 7's A/B-seam
//! contract).

use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use turbospark_repack::build_synthetic_qwen_gdn_moe_install;
use turbospark_runtime::{ChunkedPrefillRunner, RealForwardRunner};

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
fn the_qwen_flow_still_refuses_chunked_prefill_by_name() {
    let dir = tempdir();
    let arch = build_synthetic_qwen_gdn_moe_install(&dir, 128, 4, 8, "tiny-qwen-refusal")
        .expect("qwen install builds");
    let mut runner = RealForwardRunner::open(&dir, arch).expect("qwen install opens");
    assert!(
        !runner.supports_chunked_prefill(),
        "the qwen flow must not report chunked-prefill support; this session widened \
         gemma4, llama and muse_glimmer/gpt-oss only"
    );
    let mut logits = vec![f16::from_f32(0.0); 128];
    let err = runner
        .prefill_chunk(&[5, 9], 0, &mut logits)
        .expect_err("the qwen flow must refuse chunked prefill by name, not fall back");
    assert!(
        err.contains("Gemma 4") || err.to_lowercase().contains("gemma"),
        "the refusal must name the flows it DOES serve rather than being silent, got: {err}"
    );
}
