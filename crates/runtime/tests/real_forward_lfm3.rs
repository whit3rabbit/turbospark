//! The `lfm3` decode flow, on a synthetic install built through the real
//! GGUF walk.
//!
//! Untrained weights make "it decoded" nearly worthless (`crates/runtime`
//! Gotcha 11), so the assertions are reachability and state: perturb each
//! tensor this family adds and require the logits to move, and read the
//! short-conv tails directly to prove the conv ran and `reset()` rewound it.

#![cfg(target_os = "macos")]

use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_repack::{build_synthetic_lfm3_install, build_synthetic_lfm3_mlx_install};
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};

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

/// The GGUF fixture is all Q8_0, which has no batched GEMM: it keeps the
/// sequential path, and a chunk call is refused by name rather than looped.
#[test]
fn chunked_prefill_is_not_claimed_on_q8_0() {
    let (dir, arch) = install();
    let mut runner = RealForwardRunner::open(&dir, arch).expect("open");
    assert!(!runner.supports_chunked_prefill());
    let mut logits = vec![half::f16::ZERO; VOCAB as usize];
    let err = runner
        .prefill_chunk(&[1, 2, 3], 0, &mut logits)
        .expect_err("a Q8_0 install refuses the chunk driver");
    assert!(err.contains("INT4-affine"), "{err}");
}

fn mlx_install(head_dim: usize) -> (std::path::PathBuf, model_io::ArchConfig) {
    let dir = tempdir();
    let arch = build_synthetic_lfm3_mlx_install(&dir, VOCAB, LAYERS, "lfm3-toy-mlx", head_dim)
        .expect("writes");
    (dir, arch)
}

/// Long enough that a sliding layer's ring (window 8 plus the 128-token
/// chunk cap) wraps inside a 64-row micro-batch.
const PROMPT: usize = 200;

fn prompt() -> Vec<i32> {
    (0..PROMPT)
        .map(|i| ((i * 7 + 3) % VOCAB as usize) as i32)
        .collect()
}

/// `produce` over the prompt, then `extra` greedy-free steps on fixed ids:
/// every logits vector, the prompt's last first.
fn sequential(dir: &std::path::Path, arch: &model_io::ArchConfig, extra: usize) -> Vec<Vec<u16>> {
    let mut runner = RealForwardRunner::open(dir, arch.clone()).expect("open");
    let mut logits = vec![half::f16::ZERO; VOCAB as usize];
    for (i, &t) in prompt().iter().enumerate() {
        runner.produce(t, i, &mut logits).expect("produce");
    }
    let mut out = vec![logits.iter().map(|v| v.to_bits()).collect()];
    for step in 0..extra {
        runner
            .produce((step % 5) as i32, PROMPT + step, &mut logits)
            .expect("produce");
        out.push(logits.iter().map(|v| v.to_bits()).collect());
    }
    out
}

/// The prompt through `prefill_chunk` calls of `split` tokens, then the same
/// decode steps as [`sequential`].
fn chunked(
    dir: &std::path::Path,
    arch: &model_io::ArchConfig,
    split: &[usize],
    exact: bool,
    extra: usize,
) -> Vec<Vec<u16>> {
    assert_eq!(split.iter().sum::<usize>(), PROMPT);
    let mut runner = RealForwardRunner::open(dir, arch.clone()).expect("open");
    assert!(runner.supports_chunked_prefill());
    if exact {
        runner.set_prefill_steel(false);
        runner.set_prefill_rows_attention(false);
    }
    let tokens = prompt();
    let mut logits = vec![half::f16::ZERO; VOCAB as usize];
    let mut at = 0usize;
    for &n in split {
        runner
            .prefill_chunk(&tokens[at..at + n], at, &mut logits)
            .expect("prefill_chunk");
        at += n;
    }
    let mut out = vec![logits.iter().map(|v| v.to_bits()).collect()];
    for step in 0..extra {
        runner
            .produce((step % 5) as i32, PROMPT + step, &mut logits)
            .expect("produce");
        out.push(logits.iter().map(|v| v.to_bits()).collect());
    }
    out
}

/// The exact arm (scalar GEMM, one-row attention) is bit-identical to
/// sequential prefill, and so is every decode step after it: K/V slots, the
/// ring and the conv tails were all left as `produce` leaves them. Splits
/// cover one call, odd boundaries, and micro-batches shorter than the conv's
/// six-row tail.
#[test]
fn the_exact_chunk_arm_matches_sequential_bit_for_bit() {
    let (dir, arch) = mlx_install(64);
    let want = sequential(&dir, &arch, 6);
    for split in [&[PROMPT][..], &[5, 60, 3, 132], &[1, 2, 3, 64, 65, 65]] {
        let got = chunked(&dir, &arch, split, true, 6);
        assert_eq!(got, want, "split {split:?}");
    }
}

fn as_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect()
}

/// The default arm (Steel GEMMs, multi-row attention on full layers at
/// head_dim 128) is not bit-exact, so it is held close instead: same argmax
/// and a small max difference relative to the logits' spread, at the
/// prompt's last row and after decode.
#[test]
fn the_default_chunk_arm_stays_close_to_sequential() {
    for head_dim in [64usize, 128] {
        let (dir, arch) = mlx_install(head_dim);
        let want = sequential(&dir, &arch, 3);
        let got = chunked(&dir, &arch, &[PROMPT], false, 3);
        for (step, (g, w)) in got.iter().zip(&want).enumerate() {
            let (g, w) = (as_f32(g), as_f32(w));
            let spread = w.iter().cloned().fold(f32::MIN, f32::max)
                - w.iter().cloned().fold(f32::MAX, f32::min);
            let max_diff = g
                .iter()
                .zip(&w)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff <= 0.02 * spread.max(1e-3),
                "head_dim {head_dim} step {step}: max diff {max_diff} over spread {spread}"
            );
        }
    }
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

/// `produce` for `steps` tokens with the fused gate+up+silu GEMV on or off:
/// every logits vector.
fn decode_ffn_arm(
    dir: &std::path::Path,
    arch: &model_io::ArchConfig,
    fused: bool,
    steps: usize,
) -> Vec<Vec<u16>> {
    let mut runner = RealForwardRunner::open(dir, arch.clone()).expect("open");
    runner.set_lfm3_fused_ffn(fused);
    let mut logits = vec![half::f16::ZERO; VOCAB as usize];
    (0..steps)
        .map(|i| {
            runner
                .produce(((i * 7 + 3) % VOCAB as usize) as i32, i, &mut logits)
                .expect("produce");
            logits.iter().map(|v| v.to_bits()).collect()
        })
        .collect()
}

/// The fused gate+up+silu GEMV is bit-identical to the gemv, gemv, silu_mul
/// chain at every step, past the sliding ring's wrap. The INT4 fixture is
/// the one that reaches it; the Q8_0 fixture keeps the chain.
#[test]
fn the_fused_ffn_matches_the_three_dispatch_chain_bit_for_bit() {
    let (dir, arch) = mlx_install(64);
    let chain = decode_ffn_arm(&dir, &arch, false, 2 * STEPS);
    let fused = decode_ffn_arm(&dir, &arch, true, 2 * STEPS);
    assert_ne!(chain[0], chain[chain.len() - 1], "the logits never moved");
    assert_eq!(fused, chain);
}
