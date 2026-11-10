#![cfg(target_os = "macos")]

use half::f16;
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};

#[test]
fn wide_and_shared_bank_prefill_preserve_logits_and_decode() {
    // This target owns its process so the open-time switch cannot race the
    // other Qwen4 integration tests. A tiny QSA budget exercises sparse
    // attention inside the first wide chunk, and the tail crosses 128 rows.
    let dir = std::env::temp_dir().join(format!("qwen4-prefill-bank-{}", std::process::id()));
    let vocab = 2048;
    let arch = turbospark_repack::build_synthetic_qwen4_exp_decode_install_with_indexer_budget(
        &dir,
        vocab,
        "qwen4-prefill-bank",
        16,
    )
    .unwrap();
    let tokens: Vec<i32> = (0..146).map(|i| ((i * 37 + 11) % vocab) as i32).collect();
    let mut reference = None;
    for (env, enabled, cap, engages) in [
        ("off", None, None, false),
        ("wide", None, None, true),
        ("off", Some(true), None, true),
        ("bank", Some(false), None, false),
        ("bank", Some(true), Some(0), false),
    ] {
        std::env::set_var("TURBOSPARK_QWEN4_PREFILL", env);
        let mut runner = RealForwardRunner::open_with_max_context(&dir, arch.clone(), 256).unwrap();
        runner.configure_moe_prefill_bank(enabled, Default::default(), cap);
        assert!(runner.supports_moe_prefill_bank());
        assert_eq!(runner.moe_prefill_bank_enabled(), enabled == Some(true));
        let mut logits = vec![f16::ZERO; vocab as usize];
        runner
            .prefill_chunk(&tokens[..145], 0, &mut logits)
            .unwrap();
        assert_eq!(
            runner.take_prefill_experiment_footprint().is_some(),
            engages,
            "unexpected admission for {env}, {enabled:?}, {cap:?}"
        );
        let prefill = logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let prefill_hits = runner.phase_counters().expert_hits;
        runner.produce(tokens[145], 145, &mut logits).unwrap();
        if enabled == Some(true) && engages {
            assert_eq!(
                runner.phase_counters().expert_hits,
                prefill_hits,
                "the first decode has a cold cache"
            );
        }
        let decode = logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let outputs = (prefill, decode);
        if let Some(expected) = &reference {
            assert_eq!(&outputs, expected, "{env}, {enabled:?}, {cap:?}");
        } else {
            reference = Some(outputs);
        }
        if enabled == Some(true) && engages {
            runner.reset();
            runner.set_routed_batch_prefill(true);
            assert!(!runner.supports_moe_prefill_bank());
            runner
                .prefill_chunk(&tokens[..145], 0, &mut logits)
                .unwrap();
            assert!(
                runner.take_prefill_experiment_footprint().is_none(),
                "routed batching uses the existing path"
            );
        }
    }
    std::env::remove_var("TURBOSPARK_QWEN4_PREFILL");
    std::fs::remove_dir_all(dir).unwrap();
}
