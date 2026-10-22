#![cfg(target_os = "macos")]
//! Route evidence, not a throughput benchmark. Requires explicit artifact paths.
use foundation::LogitValue;
use runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};
use std::path::PathBuf;

#[test]
#[ignore = "requires the real streamed Qwen MoE install and an unused capture path"]
fn captures_real_prefill_tails_decode_and_reuse() {
    let model =
        PathBuf::from(std::env::var_os("TURBOSPARK_QWEN36_INSTALL_DIR").expect("model path"));
    let output = PathBuf::from(std::env::var_os("TURBOSPARK_ROUTER_HIST").expect("capture path"));
    assert!(!output.exists(), "refusing to replace existing evidence");
    for key in [
        "TURBOSPARK_BATCHED_GEMV",
        "TURBOSPARK_QWEN_MOE_CHUNKED",
        "TURBOSPARK_PREFILL_STEEL",
    ] {
        assert!(
            std::env::var_os(key).is_none(),
            "{key} overrides this probe"
        );
    }
    std::env::set_var("TURBOSPARK_ROUTER_BATCHES", "1");
    std::env::set_var("TURBOSPARK_ROUTER_TRACE", "1");
    std::env::set_var("TURBOSPARK_ROUTER_BATCH_LIMIT", "2000000");
    let arch = repack::peek_manifest_arch(&model).expect("manifest");
    assert_eq!(arch.family, model_io::ModelFamily::QwenGdnMoe);
    let layers = usize::try_from(arch.num_layers).expect("positive layer count");
    let tokenizer = tokenizer::MfTokenizer::load_from_dir(&model).expect("bundled tokenizer");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut corpus = String::new();
    for name in [
        "docs/BATCHED_PREFILL.md",
        "docs/GTURBO.md",
        "docs/TESTING.md",
    ] {
        corpus.push_str(&std::fs::read_to_string(root.join(name)).expect("corpus"));
        corpus.push('\n');
    }
    let tokens = tokenizer.encode(&corpus, false);
    assert!(tokens.len() > 4096, "corpus too short");
    let lengths: [usize; 7] = [64, 65, 129, 256, 512, 1024, 64];
    let open = || {
        RealForwardRunner::open_with_residency(
            &model,
            arch.clone(),
            2048,
            model_io::ExpertCacheSlots::Fixed(16),
            runtime::DraftPolicies::off(),
            runtime::SteeringPolicy::off(),
            1,
            model_io::KvQuant::Off,
            model_io::ExpertResidency::Streamed,
        )
        .expect("open real model")
    };
    // A fresh runner isolates observation from cache-slot history.
    std::env::remove_var("TURBOSPARK_ROUTER_HIST");
    let mut baseline = open();
    baseline.set_batched_gemv_prefill(true);
    let mut expected = vec![LogitValue::from_f32(0.0); baseline.vocab_size()];
    baseline.reset();
    baseline
        .prefill_chunk(&tokens[..64], 0, &mut expected)
        .expect("unobserved prefill");
    baseline
        .produce(tokens[63], 64, &mut expected)
        .expect("unobserved decode");
    drop(baseline);
    std::env::set_var("TURBOSPARK_ROUTER_HIST", &output);
    let mut runner = open();
    assert_eq!(runner.resolved_expert_residency().as_str(), "streamed");
    runner.set_batched_gemv_prefill(true);
    let mut logits = vec![LogitValue::from_f32(0.0); runner.vocab_size()];
    let mut reused = None;
    for (request, &length) in lengths.iter().enumerate() {
        // Repeat the first prompt after other requests to observe reset and cache reuse.
        let start = if request == lengths.len() - 1 {
            0
        } else {
            request * 311
        };
        let ids = &tokens[start..start + length];
        runner.reset();
        for offset in (0..length).step_by(128) {
            runner
                .prefill_chunk(&ids[offset..length.min(offset + 128)], offset, &mut logits)
                .expect("real prefill");
        }
        runner
            .produce(ids[length - 1], length, &mut logits)
            .expect("decode position");
        assert!(
            logits.iter().all(|x| x.to_f32().is_finite()),
            "non-finite logits"
        );
        if request == 0 {
            assert_eq!(logits, expected, "observation changed real-model bits");
            reused = Some(logits.clone());
        } else if request == lengths.len() - 1 {
            assert_eq!(
                reused.as_ref().unwrap(),
                &logits,
                "reset/reuse changed bits"
            );
        }
        eprintln!(
            "executed request {request}: {length} prompt rows, one decode row, streamed slots=16"
        );
    }
    drop(runner);
    let capture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&output).expect("capture written"))
            .expect("capture JSON");
    assert_eq!(capture["batches"]["truncated"], false);
    let rows = capture["batches"]["rows"]
        .as_array()
        .expect("detailed batches");
    let expected_batches: usize = lengths.iter().map(|n| n.div_ceil(64) + 1).sum();
    assert_eq!(rows.len(), expected_batches * layers);
    for (request, &length) in lengths.iter().enumerate() {
        let request_rows: Vec<_> = rows
            .iter()
            .filter(|r| r["request"] == request + 1)
            .collect();
        for layer in 0..layers {
            let layer_rows: Vec<_> = request_rows
                .iter()
                .filter(|r| r["layer"] == layer)
                .collect();
            assert_eq!(layer_rows.len(), length.div_ceil(64) + 1);
            for (batch, row) in layer_rows.iter().enumerate() {
                let decode = batch == layer_rows.len() - 1;
                let count = if decode {
                    1
                } else {
                    (length - batch * 64).min(64)
                };
                assert_eq!(row["phase"], if decode { "decode" } else { "prefill" });
                assert_eq!(row["token_start"], if decode { length } else { batch * 64 });
                assert_eq!(row["token_count"], count);
                assert_eq!(row["routes"].as_array().unwrap().len(), count);
                assert_eq!(row["completed"], true);
                assert_eq!(row["io_samples"], count);
            }
        }
    }
    eprintln!(
        "validated {} complete captured layer batches at {}",
        rows.len(),
        output.display()
    );
}
