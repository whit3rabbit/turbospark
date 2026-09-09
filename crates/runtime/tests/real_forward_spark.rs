//! The `spark2_5` decode flow, on a synthetic install.
//!
//! Shaped like `real_forward_muse.rs` and for its reason
//! (`crates/runtime/CLAUDE.md` Gotcha 11): a synthetic fixture's weights are
//! UNTRAINED, so "it decoded" is nearly worthless as evidence. The
//! assertions are **DOES THIS INPUT REACH THE MATH**: for each tensor that
//! makes this a ninth flow, perturb it in the install and require the logits
//! to move.
//!
//! Two shape notes this family ADDS to muse's list, both fixture-level:
//! `attn_gate` is `[hidden, 4]` -- four SCALARS, not a `[q_dim, hidden]`
//! matrix -- so a flow that read it at the wrong width gets garbage scale,
//! which the perturbation below still catches but the frozen digest pins;
//! and `attn_qkv` is FUSED, so perturbing any of its three row ranges (q,
//! k, v) must move the logits, which is what the split test checks.

#![cfg(target_os = "macos")]

use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_repack::build_synthetic_spark_install;
use turbospark_runtime::{LogitProducer, RealForwardRunner};

const VOCAB: i64 = 64;
/// A multiple of 4 so the `[0, 0, 0, 1]` window pattern is whole: layers 0-2
/// slide at theta 1e3 over the whole head and layer 3 is FULL at theta 5e3
/// over its leading quarter. Eight gives two periods, so the alternation is
/// exercised twice.
const LAYERS: i64 = 8;
/// Deliberately larger than the fixture's 8-token sliding window, so the
/// sliding layers wrap their ring while the full ones do not.
const STEPS: usize = 12;

fn tempdir() -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("turbospark-spark-{}-{unique}", std::process::id()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn install() -> (std::path::PathBuf, model_io::ArchConfig) {
    let dir = tempdir();
    let arch = build_synthetic_spark_install(&dir, VOCAB, LAYERS, "spark-toy")
        .expect("the install writes");
    (dir, arch)
}

/// Decodes `steps` greedy tokens and returns the last step's logits.
fn decode(dir: &std::path::Path, arch: &model_io::ArchConfig, steps: usize) -> Vec<f32> {
    let mut runner =
        RealForwardRunner::open(dir, arch.clone()).expect("the spark2_5 install opens");
    let vocab = runner.vocab_size();
    let mut logits = vec![half::f16::ZERO; vocab];
    for step in 0..steps {
        runner
            .produce((step % VOCAB as usize) as i32, step, &mut logits)
            .expect("produce");
    }
    logits.iter().map(|v| v.to_f32()).collect()
}

fn baseline() -> (std::path::PathBuf, model_io::ArchConfig, Vec<f32>) {
    let (dir, arch) = install();
    let logits = decode(&dir, &arch, STEPS);
    (dir, arch, logits)
}

/// Flips EVERY byte of one resident tensor, in place. Returns false when
/// the tensor is absent, so a caller naming a tensor this family does not
/// have fails loudly rather than silently "passing".
///
/// The WHOLE tensor, not the muse file's first 512 bytes: the fused
/// `q_k_v_proj` here is 16 KB of q8_0 blocks laid out along the IN dim, so
/// a 512-byte flip touches four of its 512 output rows -- under one percent
/// -- and on this DC-dominated fixture that is below an FP16 ulp of the
/// logits, which reads exactly like "the flow is not reading it". A full
/// flip cannot miss, and the hash comparison makes any hit count.
fn perturb(dir: &std::path::Path, tensor: &str) -> bool {
    let path = dir.join("model_weights.bin");
    let index = model_io::load_resident_index(&path).expect("index");
    let Some(entry) = index.entries.get(tensor) else {
        return false;
    };
    let mut bytes = std::fs::read(&path).expect("read");
    let start = entry.file_offset as usize;
    let end = start + entry.size_bytes as usize;
    for b in &mut bytes[start..end] {
        *b ^= 0x3C;
    }
    std::fs::write(&path, bytes).expect("write");
    true
}

/// **THE FROZEN DIGEST'S ONE SURVIVOR, measured rather than assumed.**
/// Swapping the two classes' (rotary width, theta) assignments leaves this
/// file's every test green, the frozen digest included: rope still runs, and
/// what the swap changes is absorbed by the residual stream's common-mode
/// term below an FP16 ulp of these logits. Skipping rope ENTIRELY does
/// redden the digest, so the digest proves rope runs and nothing more. The
/// per-class SELECTION is tested directly by
/// `families/spark/attn.rs`'s `the_rope_selection_is_per_layer_class` on the
/// named function the flow calls; whether the assignment is RIGHT for the
/// checkpoint is the real-model quality gate's question.
///
/// Whether a perturbed install's logits differ AT ALL from the baseline's.
///
/// **A HASH, NOT A MAGNITUDE, and that is a property of this fixture rather
/// than a style choice.** The shared q8_0 generator's weights carry a small
/// negative mean, so every GEMV adds a common-mode term and the residual
/// stream is nearly ONE CONSTANT VECTOR by the top of the stack
/// (`docs/SPARK_PHASE0.md`'s open items). Magnitude-based reachability
/// thresholds then read FP16 quantization noise: a full-tensor XOR at the
/// last layer moves the logits by ~3 ulps, while the same perturbation at
/// layer 0 moves them 15x more through downstream amplification -- both far
/// below what the same code costs on the muse fixture. The decode path is
/// bit-deterministic on one device (`a_replayed_generation_is_bit_identical`),
/// so ANY one-bit change in ANY of the 64 logits is reachable evidence.
fn logits_differ(base: &[f32], moved: &[f32]) -> bool {
    assert_eq!(base.len(), moved.len());
    fnv1a(base) != fnv1a(moved)
}

#[test]
fn a_spark_install_decodes() {
    let (_dir, _arch, logits) = baseline();
    assert_eq!(logits.len(), VOCAB as usize);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "every logit must be finite"
    );
}

/// EVERY tensor this family carries must reach the logits.
///
/// `self_attn.g_proj` is the headwise gate (four scalars, a shape unique to
/// this family); `self_attn.q_k_v_proj` is the FUSED projection a
/// neighbour's flow would look for under three other names. The two plain
/// norms are here because this family has exactly two per layer, unlike
/// muse's four centered ones.
#[test]
fn every_layer_tensor_reaches_the_logits() {
    let (_base_dir, _base_arch, base) = baseline();
    for suffix in [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_k_v_proj.weight",
        "self_attn.g_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ] {
        let (dir, arch) = install();
        let name = format!("language_model.model.layers.0.{suffix}");
        assert!(perturb(&dir, &name), "{name} must exist in the install");
        let moved = decode(&dir, &arch, STEPS);
        assert!(
            logits_differ(&base, &moved),
            "perturbing {suffix} did not move the logits: the flow is not reading it"
        );
    }
}

/// The split's three RANGES are separately live: the fused tensor's q, k and
/// v quarters feed different consumers (queries, the K cache, the V cache),
/// and a split that dropped or doubled a range would still read the tensor.
/// This is the stride-confusion guard in `(crates/repack)` Gotcha 48's
/// shape, lifted to the flow: perturb byte ranges the split must keep
/// disjoint, and require each to move the logits through its OWN consumer.
#[test]
fn the_fused_qkv_ranges_each_reach_their_consumer() {
    let (_base_dir, _base_arch, base) = baseline();
    let path_probe = install();
    let index =
        model_io::load_resident_index(&path_probe.0.join("model_weights.bin")).expect("index");
    let entry = index
        .entries
        .get("language_model.model.layers.0.self_attn.q_k_v_proj.weight")
        .expect("the fused tensor exists");
    let start = entry.file_offset as usize;
    let end = start + entry.size_bytes as usize;
    let span = end - start;
    // Row layout along the output dim is q (256 of 512 rows) then k (128)
    // then v (128), so in BYTES q is the first HALF of the tensor and k and
    // v one quarter each. Each probe flips its WHOLE range.
    //
    // **WHY WHOLE RANGES AND NOT A FEW BYTES**: the fixture's untrained
    // gate weights saturate (heads 0 and 1 carry g < -2.4, sigmoid 0.06 and
    // 0.08), and the residual stream is DC-dominated, so a single-row query
    // perturbation vanishes below an FP16 ulp of the final logits even in a
    // live head. Flipping the complete range cannot miss, and the disjoint
    // ranges are what the stride-confusion guard needs: each flip must move
    // the logits through its OWN consumer and nothing else changes.
    let probes = [
        ("q", start..start + span / 2),
        ("k", start + span / 2..start + 3 * span / 4),
        ("v", start + 3 * span / 4..end),
    ];
    for (name, range) in probes {
        let (dir, arch) = install();
        let path = dir.join("model_weights.bin");
        let mut bytes = std::fs::read(&path).expect("read");
        for b in &mut bytes[range.clone()] {
            *b ^= 0x3C;
        }
        std::fs::write(&path, bytes).expect("write");
        let moved = decode(&dir, &arch, STEPS);
        assert!(
            logits_differ(&base, &moved),
            "perturbing the fused tensor's {name} range did not move the logits: the \
             split is not delivering it"
        );
    }
}

/// The MODEL-level tensors, including the tied head: perturbing the
/// embedding must move the logits TWICE (once as the input embedding, once
/// as the head), which the single-perturbation form cannot distinguish and
/// does not need to -- the tied-head reachability is what matters.
#[test]
fn the_model_level_tensors_reach_the_logits() {
    let (_base_dir, _base_arch, base) = baseline();
    for name in [
        "language_model.model.norm.weight",
        "language_model.model.embed_tokens.weight",
    ] {
        let (dir, arch) = install();
        assert!(perturb(&dir, name), "{name} must exist");
        let moved = decode(&dir, &arch, STEPS);
        assert!(
            logits_differ(&base, &moved),
            "perturbing {name} did not move the logits"
        );
    }
}

/// THE HEAD IS NOT SOFTCAPPED: this family applies no output transform at
/// all, so a logit large enough to prove the point must come through
/// unchanged. On untrained weights some logits are already large; the guard
/// is that NO saturating transform was stacked on top.
/// AGENTS.md Gotcha 16: `produce` writes LOGITS, never probabilities.
#[test]
fn the_head_does_not_normalize() {
    let (_dir, _arch, logits) = baseline();
    let sum: f32 = logits.iter().sum();
    assert!(
        logits.iter().any(|v| *v < 0.0),
        "a softmaxed head would be all non-negative"
    );
    assert!(
        (sum - 1.0).abs() > 1e-2,
        "logits summing to 1.0 means the head normalized: {sum}"
    );
}

/// Prefill (which skips the head) must advance every other side effect
/// exactly as `produce` does.
#[test]
fn prefill_then_decode_matches_all_produce() {
    let (dir, arch) = install();
    let vocab = VOCAB as usize;
    let tokens: Vec<i32> = (0..STEPS).map(|i| (i % vocab) as i32).collect();

    let mut all = RealForwardRunner::open(&dir, arch.clone()).expect("open");
    let mut logits_all = vec![half::f16::ZERO; vocab];
    for (i, &t) in tokens.iter().enumerate() {
        all.produce(t, i, &mut logits_all).expect("produce");
    }

    let mut split = RealForwardRunner::open(&dir, arch.clone()).expect("open");
    let mut logits_split = vec![half::f16::ZERO; vocab];
    for (i, &t) in tokens.iter().enumerate() {
        if i + 1 < tokens.len() {
            split
                .produce_prefill(t, i, &mut logits_split)
                .expect("prefill");
        } else {
            split.produce(t, i, &mut logits_split).expect("produce");
        }
    }
    assert_eq!(logits_all, logits_split);
}

/// The decode hot path allocates no Metal buffers.
#[test]
fn decode_hot_path_allocates_no_gpu_buffers() {
    let (dir, arch) = install();
    let mut runner = RealForwardRunner::open(&dir, arch).expect("open");
    let mut logits = vec![half::f16::ZERO; VOCAB as usize];
    runner.produce(1, 0, &mut logits).expect("warm");
    let before = runner.gpu_buffer_allocations();
    for step in 1..6 {
        runner.produce(2, step, &mut logits).expect("produce");
    }
    assert_eq!(before, runner.gpu_buffer_allocations());
}

/// Generation is DETERMINISTIC on one warm runner.
#[test]
fn a_replayed_generation_is_bit_identical() {
    let (dir, arch) = install();
    let mut runner = RealForwardRunner::open(&dir, arch).expect("open");
    let mut a = vec![half::f16::ZERO; VOCAB as usize];
    let mut b = vec![half::f16::ZERO; VOCAB as usize];
    for step in 0..STEPS {
        runner.produce((step % 7) as i32, step, &mut a).expect("a");
    }
    runner.reset();
    for step in 0..STEPS {
        runner.produce((step % 7) as i32, step, &mut b).expect("b");
    }
    assert_eq!(a, b);
}

/// Rewrites one `manifest.json -> arch` field and returns the install dir.
/// Every refusal below is a BACKSTOP behind `arch_validation`, which fires
/// first -- so each case patches the manifest AND passes the matching arch,
/// or it tests the wrong gate (the muse file's recorded shape).
fn install_with_arch_field(key: &str, value: serde_json::Value) -> std::path::PathBuf {
    let (dir, _) = install();
    let path = dir.join("manifest.json");
    let mut m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    m["arch"][key] = value;
    std::fs::write(&path, m.to_string()).unwrap();
    dir
}

/// **THE PER-CLASS ROPE GUARD, muse's NoPE guard inverted.** The full layers
/// rotate at theta 5e6 over their leading quarter and the sliding ones at
/// theta 1e4 over the whole head; a flow that applied ONE theta everywhere
/// is finite, fluent and wrong. The state refuses a zero on either side --
/// muse's zero-means-NoPE convention is that family's, not this one's.
#[test]
fn a_zero_full_rope_theta_is_refused() {
    let dir = install_with_arch_field("fullRopeTheta", serde_json::json!(0.0));
    let mut arch = tiny_arch();
    arch.full_rope_theta = 0.0;
    let err = match RealForwardRunner::open(&dir, arch) {
        Ok(_) => panic!("a NoPE reading of this family is refused"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(msg.contains("per class") || msg.contains("rotate"), "{msg}");
}

/// The activation is EXACT-ERF GELU, and the string naming it is checked
/// rather than sniffed: `"gelu_pytorch_tanh"` is a different function the
/// flow must not silently run, and anything else is an install built from
/// something else.
#[test]
fn a_tanh_gelu_activation_is_refused() {
    let dir = install_with_arch_field("hiddenActivation", serde_json::json!("gelu_pytorch_tanh"));
    let mut arch = tiny_arch();
    arch.hidden_activation = "gelu_pytorch_tanh".to_string();
    let err = match RealForwardRunner::open(&dir, arch) {
        Ok(_) => panic!("the tanh approximation is refused"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("gelu"), "{err}");
}

/// TIED, the inverse of muse's axis: an untied install claims a head this
/// family does not have.
#[test]
fn an_untied_head_is_refused() {
    let dir = install_with_arch_field("tieWordEmbeddings", serde_json::json!(false));
    let mut arch = tiny_arch();
    arch.tie_word_embeddings = false;
    let err = match RealForwardRunner::open(&dir, arch) {
        Ok(_) => panic!("an untied head is refused"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("tie") || err.to_string().contains("embedding"),
        "{err}"
    );
}

/// An all-full mask is refused: it would size every KV buffer at
/// `max_context` and decode a different model past the window.
#[test]
fn a_mask_with_no_sliding_layer_is_refused() {
    let ones: Vec<u8> = vec![1; LAYERS as usize];
    let dir = install_with_arch_field("fullAttentionLayerMask", serde_json::json!(ones));
    let mut arch = tiny_arch();
    arch.full_attention_layer_mask = vec![1; LAYERS as usize];
    let err = match RealForwardRunner::open(&dir, arch) {
        Ok(_) => panic!("an all-full mask is refused"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("sliding"), "{err}");
}

/// This family is DENSE. An install declaring experts is refused rather than
/// half-served.
#[test]
fn a_nonzero_expert_count_is_refused() {
    let dir = install_with_arch_field("numExperts", serde_json::json!(8));
    let mut arch = tiny_arch();
    arch.num_experts = 8;
    let err = match RealForwardRunner::open(&dir, arch) {
        Ok(_) => panic!("experts are refused"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("dense"), "{err}");
}

/// The flow refuses another family outright, so flow selection can never
/// fall through to tensor naming (`crates/runtime` Gotcha 3).
#[test]
fn another_familys_arch_is_refused() {
    let (dir, _) = install();
    let mut arch = tiny_arch();
    arch.family = model_io::ModelFamily::Llama;
    // The manifest still says spark2_5, so `arch_validation` fires first --
    // which is itself the point: BOTH gates refuse, and neither is
    // load-bearing alone.
    assert!(RealForwardRunner::open(&dir, arch).is_err());
}

fn tiny_arch() -> model_io::ArchConfig {
    turbospark_repack::tiny_spark_arch(VOCAB, LAYERS)
}

/// A FROZEN DIGEST of the synthetic install's logits.
///
/// Same reason as muse's: every other test in this file is SELF-RELATIVE, so
/// a mutation that changes the MATH for every arm equally leaves them green.
/// The per-class rope selection, the headwise gate and the split ordering
/// are exactly the mutations a reachability test cannot see.
///
/// It is a CHANGE DETECTOR, not a correctness claim. If the fixture's
/// constants change, re-freeze deliberately and say why. Device-branched on
/// the muse file's evidence: a virtualized Metal device moves individual
/// FP16 roundings, so it takes the tolerant path against the same frozen
/// values.
#[test]
fn the_synthetic_flows_arithmetic_is_frozen() {
    let (_dir, _arch, logits) = baseline();
    let context = gpu::MetalContext::new().expect("Metal device");
    let device_name = context.device().name().to_string();
    drop(context);
    if device_name.contains("Paravirtual") {
        println!(
            "device {device_name:?} is virtualized, not the real Apple Silicon this hash was \
             taken on; comparing against the frozen reference with a tolerance instead"
        );
        assert_eq!(logits.len(), FROZEN_LOGITS.len());
        for (i, (&got, &want)) in logits.iter().zip(FROZEN_LOGITS.iter()).enumerate() {
            let diff = (got - want).abs();
            let tol = 0.02f32.max(want.abs() * 0.02);
            assert!(
                diff <= tol,
                "logit {i}: the spark2_5 flow's arithmetic moved: got {got}, want {want} \
                 (diff {diff}, tolerance {tol})"
            );
        }
        return;
    }
    assert_eq!(
        fnv1a(&logits),
        FROZEN_LOGIT_HASH,
        "the spark2_5 flow's arithmetic moved"
    );
}

/// FNV-1a over the logits' FP16 bit patterns.
fn fnv1a(logits: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in logits {
        for b in half::f16::from_f32(*v).to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    }
    h
}

/// Frozen 2026-09-08 against the fixture at VOCAB 64 / LAYERS 8 / STEPS 12,
/// measured on real Apple Silicon (Apple M4 Max).
const FROZEN_LOGIT_HASH: u64 = 12_128_447_552_392_821_541;

/// The exact values behind [`FROZEN_LOGIT_HASH`], for the virtualized-device
/// branch (see muse's file for why that branch exists).
#[rustfmt::skip]
const FROZEN_LOGITS: [f32; VOCAB as usize] = [
    -1.8037109, 0.12597656, 0.6113281, 1.2861328, -1.8037109, 0.12597656, 0.6113281, 1.2861328,
    -1.8037109, 0.12597656, 0.6113281, 1.2861328, -1.8037109, 0.12597656, 0.6113281, 1.2861328,
    -1.8037109, 0.12597656, 0.6113281, 1.2861328, -1.8037109, 0.12597656, 0.6113281, 1.2861328,
    -1.8037109, 0.12597656, 0.6113281, 1.2861328, -1.8037109, 0.12597656, 0.6113281, 1.2861328,
    -1.8037109, 0.12597656, 0.6113281, 1.2861328, -1.8037109, 0.12597656, 0.6113281, 1.2861328,
    -1.8037109, 0.12597656, 0.6113281, 1.2861328, -1.8037109, 0.12597656, 0.6113281, 1.2861328,
    -1.8037109, 0.12597656, 0.6113281, 1.2861328, -1.8037109, 0.12597656, 0.6113281, 1.2861328,
    -1.8037109, 0.12597656, 0.6113281, 1.2861328, -1.8037109, 0.12597656, 0.6113281, 1.2861328
];
