#![cfg(target_os = "macos")]
//! Scheduling parity with real Metal, pressured expert slots, and recurrent state.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use model_io::ArchConfig;
use tokenizer::MfTokenizer;
use turbospark_repack::build_synthetic_qwen_gdn_moe_install;
use turbospark_runtime::{
    DraftPolicies, ExpertCacheSlots, ExpertResidency, KvQuant, LogitProducer, PhaseCounters,
    RealForwardRunner, SteeringPolicy,
};

const VOCAB: usize = 128;
const LAYERS: usize = 4;
const EXPERTS: usize = 12;
const TOP_K: usize = 8;
const OVERLAP_ENV: &str = "TURBOSPARK_QWEN_SHARED_READ_OVERLAP";
const PREP_ENV: &str = "TURBOSPARK_MAPPED_DEMAND_PREP";

struct EnvSnapshot(Vec<(&'static str, Option<OsString>)>);

impl EnvSnapshot {
    fn capture() -> Self {
        Self(
            [OVERLAP_ENV, PREP_ENV]
                .into_iter()
                .map(|key| (key, std::env::var_os(key)))
                .collect(),
        )
    }
}

impl Drop for EnvSnapshot {
    fn drop(&mut self) {
        for (key, value) in &self.0 {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }
}

struct Install(PathBuf);

impl Install {
    fn new(tag: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "turbospark-qwen-schedule-{tag}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn build(&self) -> ArchConfig {
        build_synthetic_qwen_gdn_moe_install(
            &self.0,
            VOCAB as i64,
            LAYERS as i64,
            EXPERTS as i64,
            "expert-schedule",
        )
        .unwrap()
    }
}

impl Drop for Install {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[derive(Debug, Clone, Copy)]
struct Arm {
    residency: ExpertResidency,
    overlap: bool,
    prep: &'static str,
}

impl Arm {
    fn open(self, dir: &Path, arch: &ArchConfig) -> RealForwardRunner {
        std::env::set_var(OVERLAP_ENV, if self.overlap { "1" } else { "0" });
        std::env::set_var(PREP_ENV, self.prep);
        RealForwardRunner::open_with_residency(
            dir,
            arch.clone(),
            32,
            // Eight slots for twelve experts forces misses and hits after
            // the first token, rather than testing only a fully warm bank.
            ExpertCacheSlots::Fixed(TOP_K),
            DraftPolicies::off(),
            SteeringPolicy::off(),
            1,
            KvQuant::Off,
            self.residency,
        )
        .unwrap()
    }

    fn check_counters(self, phases: PhaseCounters) {
        let layer_calls = phases.calls * LAYERS as u64;
        assert_eq!(phases.expert_requests, layer_calls * TOP_K as u64);
        assert_eq!(
            phases.qwen_shared_submissions,
            if self.overlap && self.residency == ExpertResidency::Streamed {
                layer_calls
            } else {
                0
            },
            "shared overlap was not exercised: {self:?}"
        );
        let preparing = self.residency == ExpertResidency::Mapped && self.prep != "off";
        assert_eq!(
            phases.mapped_prepare_calls,
            if preparing { layer_calls } else { 0 },
            "mapped preparation was not exercised: {self:?}"
        );
        assert_eq!(phases.mapped_prepared_bytes > 0, preparing);
        assert_eq!(
            phases.mapped_page_touches > 0,
            preparing && self.prep == "touch"
        );
        assert_eq!(
            phases.mapped_advice_calls > 0,
            preparing && self.prep == "advice"
        );
    }
}

fn walk(runner: &mut RealForwardRunner, tokens: &[i32]) -> Vec<Vec<u16>> {
    runner.reset();
    let mut trace = Vec::with_capacity(tokens.len());
    for (position, &token) in tokens.iter().enumerate() {
        let mut logits = vec![f16::ZERO; VOCAB];
        runner.produce(token, position, &mut logits).unwrap();
        assert!(logits.iter().all(|value| value.is_finite()));
        trace.push(logits.into_iter().map(f16::to_bits).collect());
    }
    trace
}

fn mixed_prefill(runner: &mut RealForwardRunner, tokens: &[i32]) -> Vec<u16> {
    runner.reset();
    let mut logits = vec![f16::ZERO; VOCAB];
    for (position, &token) in tokens.iter().enumerate() {
        if position + 1 == tokens.len() {
            runner.produce(token, position, &mut logits).unwrap();
        } else {
            runner
                .produce_prefill(token, position, &mut logits)
                .unwrap();
        }
    }
    logits.into_iter().map(f16::to_bits).collect()
}

/// Zero all affine planes, since packed zeros alone retain the affine bias.
/// Assert changed bytes so a missing tensor cannot masquerade as sensitivity.
fn zero_projection(dir: &Path, suffix: &str) {
    let path = dir.join("model_weights.bin");
    let index = model_io::load_resident_index(&path).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    let mut tensors = 0;
    let mut changed = 0;
    for entry in index.entries.values() {
        if !entry.name.ends_with(suffix) {
            continue;
        }
        assert_eq!(entry.dtype, 5, "fixture shared projections are INT8");
        for (offset, size) in [
            (entry.file_offset, entry.size_bytes),
            (entry.scale_offset, entry.scale_size),
            (entry.bias_offset, entry.bias_size),
        ] {
            assert!(size > 0);
            let region = &mut bytes[offset as usize..(offset + size) as usize];
            changed += region.iter().filter(|&&value| value != 0).count();
            region.fill(0);
        }
        tensors += 1;
    }
    assert_eq!(tensors, LAYERS, "mutation missed a shared projection");
    assert!(changed > 0, "mutation did not change any bytes");
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn scheduling_preserves_bits_across_misses_hits_prefill_reset_and_reopen() {
    // This binary has one case: all env changes occur sequentially, with
    // options captured when each runner opens and restored even on panic.
    let _env = EnvSnapshot::capture();
    let install = Install::new("parity");
    let arch = install.build();
    assert!(arch.shared_expert_gated);
    assert_eq!(arch.top_k_experts, TOP_K as i64);

    // An invalid knob value is refused at open, before any mapping work.
    std::env::set_var(PREP_ENV, "sometimes");
    let refusal = match RealForwardRunner::open_with_residency(
        &install.0,
        arch.clone(),
        32,
        ExpertCacheSlots::Fixed(TOP_K),
        DraftPolicies::off(),
        SteeringPolicy::off(),
        1,
        KvQuant::Off,
        ExpertResidency::Mapped,
    ) {
        Ok(_) => panic!("an invalid TURBOSPARK_MAPPED_DEMAND_PREP must refuse the open"),
        Err(error) => error.to_string(),
    };
    assert!(
        refusal
            .to_string()
            .contains("expected off, advice, or touch"),
        "unexpected refusal: {refusal}"
    );

    let tokenizer = MfTokenizer::load_from_dir(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer"),
    )
    .unwrap();
    let prompt: Vec<i32> = tokenizer
        .encode("hello world", false)
        .into_iter()
        .take(8)
        .collect();
    let history: Vec<i32> = tokenizer
        .encode("quick fox jumps", false)
        .into_iter()
        .take(12)
        .collect();
    assert_eq!(prompt.len(), 8);
    assert_eq!(history.len(), 12);
    assert_ne!(&prompt, &history[..prompt.len()]);
    assert!(prompt
        .iter()
        .chain(&history)
        .all(|&token| (0..VOCAB as i32).contains(&token)));

    let baseline = Arm {
        residency: ExpertResidency::Streamed,
        overlap: false,
        prep: "off",
    };
    let mut reference = baseline.open(&install.0, &arch);
    let expected_prompt = walk(&mut reference, &prompt);
    let expected_history = walk(&mut reference, &history);
    assert_eq!(walk(&mut reference, &prompt), expected_prompt);
    assert_eq!(
        mixed_prefill(&mut reference, &prompt),
        *expected_prompt.last().unwrap()
    );
    baseline.check_counters(reference.phase_counters());
    drop(reference);

    for arm in [
        baseline,
        Arm {
            overlap: true,
            ..baseline
        },
        Arm {
            residency: ExpertResidency::Mapped,
            ..baseline
        },
        Arm {
            residency: ExpertResidency::Mapped,
            prep: "advice",
            ..baseline
        },
        Arm {
            residency: ExpertResidency::Mapped,
            prep: "touch",
            ..baseline
        },
        Arm {
            residency: ExpertResidency::Mapped,
            overlap: true,
            prep: "touch",
        },
    ] {
        let mut runner = arm.open(&install.0, &arch);
        let mut first_logits = vec![f16::ZERO; VOCAB];
        runner.produce(prompt[0], 0, &mut first_logits).unwrap();
        assert_eq!(
            first_logits
                .into_iter()
                .map(f16::to_bits)
                .collect::<Vec<_>>(),
            expected_prompt[0],
            "first cold token parity: {arm:?}"
        );
        let cold = runner.phase_counters();
        assert_eq!(cold.expert_requests, (LAYERS * TOP_K) as u64);
        assert_eq!(
            cold.expert_hits,
            if arm.residency == ExpertResidency::Mapped {
                cold.expert_requests
            } else {
                0
            }
        );
        arm.check_counters(cold);
        let allocations = runner.gpu_buffer_allocations();

        assert_eq!(
            walk(&mut runner, &prompt),
            expected_prompt,
            "cold/reset parity: {arm:?}"
        );
        assert_eq!(
            walk(&mut runner, &history),
            expected_history,
            "changed-history parity: {arm:?}"
        );
        assert_eq!(
            walk(&mut runner, &prompt),
            expected_prompt,
            "warm/reset parity: {arm:?}"
        );
        assert_eq!(
            mixed_prefill(&mut runner, &prompt),
            *expected_prompt.last().unwrap(),
            "headless prefill parity: {arm:?}"
        );
        let phases = runner.phase_counters();
        assert!(
            phases.expert_hits > cold.expert_hits,
            "no warm expert hits: {arm:?}"
        );
        arm.check_counters(phases);
        assert_eq!(
            runner.gpu_buffer_allocations(),
            allocations,
            "per-token allocation: {arm:?}"
        );
        drop(runner);

        let mut reopened = arm.open(&install.0, &arch);
        assert_eq!(
            walk(&mut reopened, &prompt),
            expected_prompt,
            "reopen parity: {arm:?}"
        );
        arm.check_counters(reopened.phase_counters());
    }

    // A missing or stale shared branch could leave parity green if the
    // fixture's shared contribution or sigmoid gate were unobservable.
    for suffix in [
        "mlp.shared_expert.down_proj.weight",
        "mlp.shared_expert_gate.weight",
    ] {
        let damaged = Install::new("sensitivity");
        let damaged_arch = damaged.build();
        zero_projection(&damaged.0, suffix);
        let mut runner = baseline.open(&damaged.0, &damaged_arch);
        assert_ne!(
            walk(&mut runner, &prompt),
            expected_prompt,
            "shared mutation was unobservable: {suffix}"
        );
    }
}
