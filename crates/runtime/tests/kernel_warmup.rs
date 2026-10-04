#![cfg(target_os = "macos")]
//! Compile coverage and unchanged state through every decode-attention bucket.

use half::f16;
use turbospark_repack::build_synthetic_qwen_gdn_moe_install;
use turbospark_runtime::{LogitProducer, RealForwardRunner};

#[test]
fn compile_only_warmup_covers_dispatch_and_preserves_logits_and_reset() {
    let dir = std::env::temp_dir().join(format!("turbospark-kernel-warmup-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let arch = build_synthetic_qwen_gdn_moe_install(&dir, 128, 4, 8, "warmup").unwrap();
    let open = || RealForwardRunner::open_with_max_context(&dir, arch.clone(), 300).unwrap();
    let mut baseline = open();
    let mut prepared = open();
    let allocations = prepared.gpu_buffer_allocations();
    let phases = prepared.phase_counters();
    let before = prepared.metal_compilation_stats();
    assert!(prepared.prepare_kernel_warmup(0).is_err());
    assert_eq!(
        prepared.metal_compilation_stats().pipeline_creations,
        before.pipeline_creations
    );
    let stats = prepared.prepare_kernel_warmup(300).unwrap();
    assert!(
        stats.registrations > stats.unique_keys,
        "equivalent layers must deduplicate"
    );
    assert!(stats.pipeline_creations > 0);
    assert_eq!(prepared.gpu_buffer_allocations(), allocations);
    assert_eq!(
        prepared.phase_counters().calls,
        phases.calls,
        "warmup ran a forward pass"
    );
    let compiled = prepared.metal_compilation_stats();
    let again = prepared.prepare_kernel_warmup(300).unwrap();
    assert_eq!(again.pipeline_creations, 0, "warmup keys were not reusable");
    let tokenizer = tokenizer::MfTokenizer::load_from_dir(
        &std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/ChatMLTokenizer"),
    )
    .unwrap();
    let tokens = tokenizer.encode("hello", false);
    assert!(!tokens.is_empty());
    assert!(tokens.iter().all(|t| (0..128).contains(t)));
    // Context 257 reaches chunk counts 1, 2, 4, 8, 16. A short decode cannot
    // establish that warmup covers later specialization transitions.
    for steps in [257, 3] {
        baseline.reset();
        prepared.reset();
        for position in 0..steps {
            let token = tokens[position % tokens.len()];
            let mut expected = vec![f16::ZERO; 128];
            let mut actual = vec![f16::ZERO; 128];
            baseline.produce(token, position, &mut expected).unwrap();
            prepared.produce(token, position, &mut actual).unwrap();
            assert_eq!(
                actual, expected,
                "logits changed at {position}, {steps}-token pass"
            );
        }
    }
    let after = prepared.metal_compilation_stats();
    assert_eq!(
        after.pipeline_creations, compiled.pipeline_creations,
        "runtime reached an unprepared specialization"
    );
    assert_eq!(after.library_compiles, compiled.library_compiles);
    assert_eq!(
        after.function_specializations,
        compiled.function_specializations
    );
    assert_eq!(prepared.gpu_buffer_allocations(), allocations);
    drop(baseline);
    drop(prepared);
    std::fs::remove_dir_all(dir).unwrap();
}
