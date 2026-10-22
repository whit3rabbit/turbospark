#![cfg(target_os = "macos")]
//! One process-local capture test: no other case races environment changes.
use half::f16;
use std::path::PathBuf;
use tokenizer::MfTokenizer;
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};

#[test]
fn real_driver_records_batch_boundaries_tail_decode_and_reset() {
    let dir = std::env::temp_dir().join(format!("ts-route-batches-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("capture.json");
    std::env::set_var("TURBOSPARK_ROUTER_HIST", &path);
    std::env::set_var("TURBOSPARK_ROUTER_BATCHES", "1");
    std::env::set_var("TURBOSPARK_ROUTER_TRACE", "1");
    let arch =
        turbospark_repack::build_synthetic_qwen_gdn_moe_install(&dir, 128, 4, 32, "router-capture")
            .unwrap();
    let tokenizer = MfTokenizer::load_from_dir(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer"),
    )
    .unwrap();
    let tokens = tokenizer.encode("repeat a test token sequence", false);
    assert!(!tokens.is_empty());
    let ids: Vec<_> = tokens.iter().copied().cycle().take(65).collect();
    // Compare observation-on against the same real driver with observation
    // disabled. Cache state may change, but output bits must not.
    std::env::remove_var("TURBOSPARK_ROUTER_HIST");
    let mut baseline = RealForwardRunner::open_with_options(&dir, arch.clone(), 256, 16).unwrap();
    let mut expected = vec![f16::ZERO; 128];
    baseline.reset();
    baseline.prefill_chunk(&ids, 0, &mut expected).unwrap();
    baseline.produce(ids[0], 65, &mut expected).unwrap();
    drop(baseline);
    std::env::set_var("TURBOSPARK_ROUTER_HIST", &path);
    {
        let mut runner = RealForwardRunner::open_with_options(&dir, arch, 256, 16).unwrap();
        let mut logits = vec![f16::ZERO; 128];
        runner.reset();
        runner.prefill_chunk(&ids, 0, &mut logits).unwrap();
        runner.produce(ids[0], 65, &mut logits).unwrap();
        assert_eq!(logits, expected, "capture changed inference bits");
        runner.reset();
        runner.prefill_chunk(&ids[..1], 0, &mut logits).unwrap();
    }
    let data: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let rows = data["batches"]["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 16);
    for layer in 0..4 {
        let layer_rows: Vec<_> = rows.iter().filter(|r| r["layer"] == layer).collect();
        assert_eq!(layer_rows[0]["token_count"], 64);
        assert_eq!(layer_rows[1]["token_start"], 64);
        assert_eq!(layer_rows[1]["token_count"], 1);
        assert_eq!(layer_rows[2]["phase"], "decode");
        assert_eq!(layer_rows[3]["request"], 2);
        for r in layer_rows {
            assert_eq!(r["completed"], true);
            let count = r["token_count"].as_u64().unwrap();
            assert_eq!(r["routes"].as_array().unwrap().len() as u64, count);
            assert_eq!(
                r["cache_hits"].as_u64().unwrap() + r["cache_misses"].as_u64().unwrap(),
                count * 8
            );
        }
    }
    for key in [
        "TURBOSPARK_ROUTER_HIST",
        "TURBOSPARK_ROUTER_BATCHES",
        "TURBOSPARK_ROUTER_TRACE",
    ] {
        std::env::remove_var(key);
    }
    // Keep a copy outside the install for deterministic replay inspection.
    let output = std::env::temp_dir().join("ane-router-synthetic-capture.json");
    std::fs::copy(path, output).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
}
