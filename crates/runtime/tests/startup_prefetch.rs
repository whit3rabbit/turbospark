#![cfg(target_os = "macos")]
//! Synthetic Metal gate for startup accounting and explicit expert preparation.

use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use turbospark_repack::build_synthetic_llama_real_install;
use turbospark_runtime::{
    DraftPolicies, ExpertCacheSlots, ExpertResidency, KvQuant, LogitProducer, RealForwardRunner,
    SteeringPolicy,
};

#[test]
fn prefetch_is_bounded_and_preserves_logits_in_both_residency_modes() {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-startup-prefetch-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let arch = build_synthetic_llama_real_install(&dir, 128, 2, 4, "startup-prefetch").unwrap();
    for residency in [ExpertResidency::Streamed, ExpertResidency::Mapped] {
        let open = || {
            RealForwardRunner::open_with_residency(
                &dir,
                arch.clone(),
                32,
                ExpertCacheSlots::Fixed(16),
                DraftPolicies::off(),
                SteeringPolicy::off(),
                1,
                KvQuant::Off,
                residency,
            )
            .unwrap()
        };
        let mut baseline = open();
        let mut prepared = open();
        let startup = prepared.startup_stats();
        let phases = [
            startup.manifest_index_ms,
            startup.resident_mapping_ms,
            startup.kv_scratch_ms,
            startup.expert_setup_ms,
            startup.family_state_ms,
            startup.session_state_ms,
        ];
        assert!(phases.iter().all(|ms| ms.is_finite() && *ms >= 0.0));
        assert!(startup.total_open_ms >= phases.iter().sum::<f64>());
        assert!(startup.expert_setup_ms > 0.0);
        assert!(startup.family_state_ms > 0.0);
        let allocations = prepared.gpu_buffer_allocations();
        let probe = prepared.prefetch_experts(&[(0, 0)], u64::MAX).unwrap();
        assert_eq!(probe.prepared_experts, 1);
        assert!(probe.bytes_prepared > 0);
        let bounded = prepared
            .prefetch_experts(&[(0, 0), (0, 0), (1, 1), (0, 1)], 2 * probe.bytes_prepared)
            .unwrap();
        assert_eq!(bounded.requested_experts, 4);
        assert_eq!(bounded.prepared_experts, 2);
        assert_eq!(bounded.duplicate_experts, 1);
        assert_eq!(bounded.skipped_budget_experts, 1);
        assert_eq!(bounded.bytes_prepared, 2 * probe.bytes_prepared);
        assert_eq!(
            bounded.mapped_page_touches > 0,
            residency == ExpertResidency::Mapped
        );
        assert_eq!(allocations, prepared.gpu_buffer_allocations());
        let empty = prepared.prefetch_experts(&[(0, 0)], 0).unwrap();
        assert_eq!(empty.prepared_experts, 0);
        assert_eq!(empty.bytes_prepared, 0);
        assert!(prepared
            .prefetch_experts(&[(0, 0), (99, 0)], u64::MAX)
            .is_err());
        assert!(prepared.prefetch_experts(&[(0, 4)], u64::MAX).is_err());
        for (position, token) in [5, 9, 2].into_iter().enumerate() {
            let mut expected = vec![f16::ZERO; 128];
            let mut actual = vec![f16::ZERO; 128];
            baseline.produce(token, position, &mut expected).unwrap();
            prepared.produce(token, position, &mut actual).unwrap();
            assert_eq!(
                expected, actual,
                "prefetch moved logits at position {position}"
            );
        }
        prepared.flush_pipeline_cache();
        assert!(prepared.metal_compilation_stats().pipeline_creations > 0);
    }
    std::fs::remove_dir_all(dir).unwrap();
}
