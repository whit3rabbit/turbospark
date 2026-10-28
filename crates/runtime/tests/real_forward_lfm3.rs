//! The `lfm3` decode flow, on a synthetic install built through the real
//! GGUF walk.
//!
//! Untrained weights make "it decoded" nearly worthless (`crates/runtime`
//! Gotcha 11), so the assertions are reachability and state: perturb each
//! tensor this family adds and require the logits to move, and read the
//! short-conv tails directly to prove the conv ran and `reset()` rewound it.

#![cfg(target_os = "macos")]

use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_repack::build_synthetic_lfm3_install;
use turbospark_runtime::{LogitProducer, RealForwardRunner};

const VOCAB: i64 = 64;
/// `[conv, conv, full, conv, sliding, conv, full, conv]`: every kind, and
/// both attention classes twice over.
const LAYERS: i64 = 8;
/// Past the fixture's 8-token window, so the sliding layers wrap their ring.
const STEPS: usize = 12;

fn tempdir() -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("turbospark-lfm3-{}-{unique}", std::process::id()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn install() -> (std::path::PathBuf, model_io::ArchConfig) {
    let dir = tempdir();
    let arch = build_synthetic_lfm3_install(&dir, VOCAB, LAYERS, "lfm3-toy").expect("writes");
    (dir, arch)
}

fn decode(dir: &std::path::Path, arch: &model_io::ArchConfig, steps: usize) -> Vec<f32> {
    let mut runner = RealForwardRunner::open(dir, arch.clone()).expect("the lfm3 install opens");
    let mut logits = vec![half::f16::ZERO; runner.vocab_size()];
    for step in 0..steps {
        runner
            .produce((step % VOCAB as usize) as i32, step, &mut logits)
            .expect("produce");
    }
    logits.iter().map(|v| v.to_f32()).collect()
}

/// Flips every byte of one resident tensor in place (the spark file's
/// reason for the whole tensor: a partial flip can sit below an FP16 ulp of
/// these DC-dominated logits). False when the tensor is absent.
fn perturb(dir: &std::path::Path, tensor: &str) -> bool {
    let path = dir.join("model_weights.bin");
    let index = model_io::load_resident_index(&path).expect("index");
    let Some(entry) = index.entries.get(tensor) else {
        return false;
    };
    let mut bytes = std::fs::read(&path).expect("read");
    let start = entry.file_offset as usize;
    for b in &mut bytes[start..start + entry.size_bytes as usize] {
        *b ^= 0x3C;
    }
    std::fs::write(&path, bytes).expect("write");
    true
}

/// FNV-1a over the logits' FP16 bit patterns. The decode path is
/// bit-deterministic on one device, so any one-bit change is evidence.
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

#[test]
fn an_lfm3_install_decodes_finite_raw_logits() {
    let (dir, arch) = install();
    let logits = decode(&dir, &arch, STEPS);
    assert_eq!(logits.len(), VOCAB as usize);
    assert!(logits.iter().all(|v| v.is_finite()));
    // AGENTS.md Gotcha 16: raw logits, never probabilities.
    assert!(
        logits.iter().any(|v| *v < 0.0),
        "a softmaxed head is non-negative"
    );
    let sum: f32 = logits.iter().sum();
    assert!(
        (sum - 1.0).abs() > 1e-2,
        "logits summing to 1 means a normalizing head"
    );
}

/// Every per-layer tensor reaches the logits: the conv layer's three (layer
/// 1), the attention layer's v / o / gate on a SLIDING layer (4) and a FULL
/// one (6), the four norms plus the FFN, and the model-level tensors. q, k
/// and their norms are the next test's job.
#[test]
fn every_layer_tensor_reaches_the_logits() {
    let (base_dir, base_arch) = install();
    let base = decode(&base_dir, &base_arch, STEPS);
    let mut cases: Vec<String> = Vec::new();
    for suffix in [
        "conv.in_proj.weight",
        "conv.conv.weight",
        "conv.out_proj.weight",
        "operator_norm.weight",
        "post_operator_norm.weight",
        "ffn_norm.weight",
        "post_ffn_norm.weight",
        "feed_forward.w1.weight",
        "feed_forward.w2.weight",
        "feed_forward.w3.weight",
    ] {
        cases.push(format!("language_model.model.layers.1.{suffix}"));
    }
    for layer in [4, 6] {
        for suffix in ["v_proj.weight", "o_proj.weight"] {
            cases.push(format!(
                "language_model.model.layers.{layer}.self_attn.{suffix}"
            ));
        }
    }
    // Layer 6's gate saturates on these untrained weights at the real scale
    // (spark's recorded fixture issue); the sharpened test covers it.
    cases.push("language_model.model.layers.4.self_attn.gate_proj.weight".to_string());
    cases.push("language_model.model.embedding_norm.weight".to_string());
    cases.push("language_model.model.embed_tokens.weight".to_string());
    let mut unreached = Vec::new();
    for name in cases {
        let (dir, arch) = install();
        assert!(perturb(&dir, &name), "{name} must exist in the install");
        let moved = decode(&dir, &arch, STEPS);
        if fnv1a(&base) == fnv1a(&moved) {
            unreached.push(name);
        }
    }
    assert!(
        unreached.is_empty(),
        "perturbing these did not move the logits, so the flow is not reading them: {unreached:#?}"
    );
}

/// q, k and their per-head norms reach the logits, on a SLIDING layer (4)
/// and a FULL one (6), WITH THE ATTENTION SCALE SHARPENED TO 8.
///
/// **WHY SHARPENED, measured 2026-10-27.** On this fixture, RMS-normed q and
/// k at the real `128^-0.5` give scores of roughly +-0.25, so the softmax is
/// nearly uniform and a q or k perturbation moves the output below an FP16
/// ulp of the logits: all four read as unreached at the real scale on both
/// layers, while v and o reach. At scale 8 all four reach on both layers.
/// Layer 2 (full, two conv layers into the stack) stays unreached even at
/// scale 8, its gate included, while the deeper full layer 6 reaches on
/// every input: layer 2's inputs barely vary by position on untrained
/// weights, so its softmax weights cannot matter. That is the fixture, not
/// the full-layer path, which layer 6 exercises.
#[test]
fn q_k_and_their_norms_reach_the_logits_under_a_sharpened_scale() {
    let sharpened = |perturbed: Option<&str>| {
        let (dir, mut arch) = install();
        let path = dir.join("manifest.json");
        let mut m: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        m["arch"]["attentionScale"] = serde_json::json!(8.0);
        std::fs::write(&path, m.to_string()).unwrap();
        arch.attention_scale = 8.0;
        if let Some(name) = perturbed {
            assert!(perturb(&dir, name), "{name} must exist");
        }
        fnv1a(&decode(&dir, &arch, STEPS))
    };
    let base = sharpened(None);
    let mut unreached = Vec::new();
    for (layer, suffixes) in [
        (4, &["q_proj", "k_proj", "q_norm", "k_norm"][..]),
        (
            6,
            &["q_proj", "k_proj", "q_norm", "k_norm", "gate_proj"][..],
        ),
    ] {
        for suffix in suffixes {
            let name = format!("language_model.model.layers.{layer}.self_attn.{suffix}.weight");
            if sharpened(Some(&name)) == base {
                unreached.push(name);
            }
        }
    }
    assert!(
        unreached.is_empty(),
        "unreached under scale 8: {unreached:#?}"
    );
}

/// The conv tails are the evidence the conv ran (an untrained block that
/// produced zeros would still decode). Zero at open, nonzero after decode on
/// EVERY conv layer, zero again after `reset()`, and absent on attention
/// layers.
#[test]
fn the_conv_tails_carry_state_and_reset_rewinds_them() {
    let (dir, arch) = install();
    let mut runner = RealForwardRunner::open(&dir, arch.clone()).expect("open");
    let conv_layers: Vec<usize> = (0..LAYERS as usize)
        .filter(|&l| arch.layer_is_short_conv(l))
        .collect();
    assert_eq!(conv_layers, vec![0, 1, 3, 5, 7]);
    for &l in &conv_layers {
        assert_eq!(
            runner.short_conv_tail_abs_max(l),
            Some(0.0),
            "layer {l} at open"
        );
    }
    assert_eq!(runner.short_conv_tail_abs_max(2), None);
    let mut logits = vec![half::f16::ZERO; VOCAB as usize];
    for step in 0..STEPS {
        runner
            .produce((step % 7) as i32, step, &mut logits)
            .expect("produce");
    }
    for &l in &conv_layers {
        let m = runner.short_conv_tail_abs_max(l).unwrap();
        assert!(m > 0.0 && m.is_finite(), "layer {l} tail after decode: {m}");
    }
    runner.reset();
    for &l in &conv_layers {
        assert_eq!(
            runner.short_conv_tail_abs_max(l),
            Some(0.0),
            "layer {l} after reset"
        );
    }
}

/// Generation after `reset()` replays bit-identically. Rewinding only the
/// KV would leave the previous generation in every conv tail.
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

/// Prefill skips the head and must advance every other side effect, the
/// conv tails included.
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
    let mut split = RealForwardRunner::open(&dir, arch).expect("open");
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

/// No chunked driver yet: the sequential path serves prefill.
#[test]
fn chunked_prefill_is_not_claimed() {
    let (dir, arch) = install();
    let runner = RealForwardRunner::open(&dir, arch).expect("open");
    assert!(!runner.supports_chunked_prefill());
}

/// A FROZEN DIGEST: every test above is self-relative, so a mutation that
/// changes the math for every arm equally leaves them green (AGENTS.md
/// Gotcha 51). A change detector, not a correctness claim: re-freeze
/// deliberately and say why if the fixture changes.
#[test]
fn the_synthetic_flows_arithmetic_is_frozen() {
    let (dir, arch) = install();
    let logits = decode(&dir, &arch, STEPS);
    let context = gpu::MetalContext::new().expect("Metal device");
    let device_name = context.device().name().to_string();
    drop(context);
    let hash = fnv1a(&logits);
    if device_name.contains("Paravirtual") {
        println!("virtualized device {device_name:?}: hash {hash} not compared");
        return;
    }
    assert_eq!(hash, FROZEN_LOGIT_HASH, "the lfm3 flow's arithmetic moved");
}

/// Frozen 2026-10-27 at VOCAB 64 / LAYERS 8 / STEPS 12 on Apple M4 Max.
const FROZEN_LOGIT_HASH: u64 = 4_648_377_871_517_979_813;
