#![cfg(target_os = "macos")]
//! Chunked prefill for the MoE half of the qwen flow (`qwenGdnMoe`), the
//! driver in `crates/runtime/src/families/qwen/moe_prefill.rs`.
//!
//! **The bar is byte-identity against the SEQUENTIAL path**, as in every
//! other chunked test in this crate: each case compares against
//! `produce_prefill` / `produce`, never against another chunked arm. What
//! this file proves is that grouping tokens into fewer command buffers does
//! not change the bytes, on both routed layouts this family ships: INT4
//! affine experts (Qwen 3.6, Ornith 35B) and GGUF MXFP4 experts re-blocked
//! from MLX mxfp4 (MiniCPM-V 4.7). The fixtures are deterministic but not
//! trained.
//!
//! The fixtures carry 32 experts at top-8 and open at 16 slots unless a case
//! says otherwise, so the cache MISSES and EVICTS during prefill. At 16
//! experts every expert would fit and `RoutedSlot::protect` would never
//! decide anything.
//!
//! **Each case names its ARM** (`Arm`), because unset is the automatic pick,
//! which takes the batched arm on the INT4 fixture. The batched arm runs here
//! with Steel off: its scalar GEMMs are bit-exact against the per-token GEMVs
//! and this fixture cannot see the real-install floor of the M-row GDN and
//! attention blocks, so `==` pins the WIRING (rows, offsets, the shared-expert
//! seed), not the arithmetic. Steel has its own tolerance case. The MXFP4
//! fixture's resident matrices are BF16, which the batched arm refuses, so
//! it runs the per-token arm only.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;
const EXPERTS: i64 = 32;
/// The fixtures' own `top_k_experts` (`num_experts.min(8)`).
const TOP_K: usize = 8;
const SLOTS: usize = 16;
const MAX_CONTEXT: usize = 256;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-qwen-moe-chunked-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[derive(Clone, Copy, Debug)]
enum Layout {
    Affine,
    Mxfp4,
}

fn open_runner(tag: &str, layout: Layout, slots: usize) -> RealForwardRunner {
    let dir = temp_dir(tag);
    let arch = match layout {
        Layout::Affine => turbospark_repack::build_synthetic_qwen_gdn_moe_install(
            &dir,
            VOCAB,
            LAYERS,
            EXPERTS,
            "tiny-qwen36-chunked",
        ),
        Layout::Mxfp4 => turbospark_repack::build_synthetic_qwen_gdn_moe_mxfp4_install(
            &dir,
            VOCAB,
            LAYERS,
            EXPERTS,
            "tiny-qwen36-mxfp4-chunked",
        ),
    }
    .expect("MoE qwen install builds");
    assert_eq!(arch.top_k_experts as usize, TOP_K, "fixture top_k changed");
    let runner = RealForwardRunner::open_with_options(&dir, arch, MAX_CONTEXT, slots)
        .expect("MoE qwen install opens");
    let _ = std::fs::remove_dir_all(&dir);
    runner
}

/// Which arm of the driver a case runs (`TURBOSPARK_BATCHED_GEMV=0` or `=1`).
#[derive(Clone, Copy, Debug)]
enum Arm {
    PerToken,
    /// The M-row arm with Steel off, so its GEMMs stay bit-exact.
    Batched,
}

/// Every (layout, arm) pair the byte-identity cases run.
const CASES: [(Layout, Arm); 3] = [
    (Layout::Affine, Arm::PerToken),
    (Layout::Affine, Arm::Batched),
    (Layout::Mxfp4, Arm::PerToken),
];

fn open_arm(tag: &str, layout: Layout, slots: usize, arm: Arm) -> RealForwardRunner {
    let mut runner = open_runner(tag, layout, slots);
    // Before the first chunk: the Steel choice is read when the batched
    // scratch is allocated.
    runner.set_prefill_steel(false);
    runner.set_batched_gemv_prefill(matches!(arm, Arm::Batched));
    runner
}

fn make_prompt(len: usize) -> Vec<i32> {
    (0..len)
        .map(|p| ((p * 37 + 11) % VOCAB as usize) as i32)
        .collect()
}

/// The reference: every prompt token through `produce_prefill` but the last,
/// which goes through `produce`, exactly as `run_raw_completion` does.
fn sequential_prefill(runner: &mut RealForwardRunner, tokens: &[i32]) -> Vec<f32> {
    runner.reset();
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let last = tokens.len() - 1;
    for (position, &token) in tokens.iter().enumerate() {
        if position == last {
            runner.produce(token, position, &mut logits)
        } else {
            runner.produce_prefill(token, position, &mut logits)
        }
        .expect("sequential prefill succeeds");
    }
    logits.iter().map(|v| v.to_f32()).collect()
}

/// The same prompt through `prefill_chunk`, split into spans of `chunk`.
fn chunked_prefill(runner: &mut RealForwardRunner, tokens: &[i32], chunk: usize) -> Vec<f32> {
    runner.reset();
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let mut offset = 0usize;
    while offset < tokens.len() {
        let take = (tokens.len() - offset).min(chunk);
        runner
            .prefill_chunk(&tokens[offset..offset + take], offset, &mut logits)
            .expect("chunked prefill succeeds");
        offset += take;
    }
    logits.iter().map(|v| v.to_f32()).collect()
}

/// Greedy decode `steps` tokens after the prompt, returning the tokens and
/// the final logits. The runner's state is whatever prefill left.
fn greedy_continue(
    runner: &mut RealForwardRunner,
    first: i32,
    start: usize,
    steps: usize,
) -> (Vec<i32>, Vec<f32>) {
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let mut token = first;
    let mut out = Vec::new();
    for i in 0..steps {
        runner
            .produce(token, start + i, &mut logits)
            .expect("decode succeeds");
        token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
            .map(|(i, _)| i as i32)
            .unwrap();
        out.push(token);
    }
    (out, logits.iter().map(|v| v.to_f32()).collect())
}

fn argmax(logits: &[f32]) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as i32)
        .unwrap()
}

#[test]
fn a_moe_qwen_install_reports_chunked_prefill_support() {
    for layout in [Layout::Affine, Layout::Mxfp4] {
        let runner = open_runner("supports", layout, SLOTS);
        assert!(
            runner.supports_chunked_prefill(),
            "a {layout:?} MoE qwen install must be servable by the chunked driver"
        );
    }
}

#[test]
fn the_chunk_boundary_does_not_move_the_logits() {
    // Spans of 2 and up put several tokens into one micro-batch, which is
    // what would catch a single-row `moe_x` or `router_logits_f32`: every
    // token but the last would route on, or read, the last token's row.
    for (layout, arm) in CASES {
        let mut runner = open_arm("boundary", layout, SLOTS, arm);
        let prompt = make_prompt(11);
        let expected = sequential_prefill(&mut runner, &prompt);
        assert!(
            expected.iter().all(|v| v.is_finite()),
            "{layout:?}/{arm:?}: the reference must be finite"
        );
        assert!(
            expected.iter().any(|&v| v != expected[0]),
            "{layout:?}/{arm:?}: degenerate reference logits, this fixture cannot discriminate"
        );
        for chunk in [1usize, 2, 3, 4, 7, 11] {
            let actual = chunked_prefill(&mut runner, &prompt, chunk);
            assert_eq!(
                actual, expected,
                "{layout:?}/{arm:?}: chunk span {chunk} changed the logits"
            );
        }
    }
}

#[test]
fn a_prompt_longer_than_one_micro_batch_is_byte_identical() {
    // 37 tokens as ONE chunk: on the per-token arm, and on the batched arm
    // with Steel off, two full 16-token micro-batches and a 5-token tail
    // inside a single `prefill_chunk` call, so the micro-batch boundary (not
    // only the chunk boundary) is crossed with the expert cache warm.
    for (layout, arm) in CASES {
        let mut runner = open_arm("long", layout, SLOTS, arm);
        let prompt = make_prompt(37);
        let expected = sequential_prefill(&mut runner, &prompt);
        let actual = chunked_prefill(&mut runner, &prompt, prompt.len());
        assert_eq!(
            actual, expected,
            "{layout:?}/{arm:?}: a 37-token chunk changed the logits"
        );
    }
}

#[test]
fn decode_after_a_chunked_prefill_continues_identically() {
    // Prefill leaves the KV cache, the GDN state and the expert cache behind.
    // A driver that left any of them different would show here even when the
    // prefill's own last logits matched.
    for (layout, arm) in CASES {
        let mut runner = open_arm("continue", layout, SLOTS, arm);
        let prompt = make_prompt(13);
        let seq_logits = sequential_prefill(&mut runner, &prompt);
        let (seq_tokens, seq_tail) =
            greedy_continue(&mut runner, argmax(&seq_logits), prompt.len(), 8);

        let chunk_logits = chunked_prefill(&mut runner, &prompt, 5);
        assert_eq!(
            chunk_logits, seq_logits,
            "{layout:?}/{arm:?}: prefill logits"
        );
        let (chunk_tokens, chunk_tail) =
            greedy_continue(&mut runner, argmax(&chunk_logits), prompt.len(), 8);
        assert_eq!(
            chunk_tokens, seq_tokens,
            "{layout:?}/{arm:?}: decoded tokens diverged"
        );
        assert_eq!(
            chunk_tail, seq_tail,
            "{layout:?}/{arm:?}: final decode logits"
        );
    }
}

/// The `banks == 1` fallback, below `2 * top_k` slots. The loop must hand
/// the next token an EMPTY `protect` set there (AGENTS.md Gotcha 64), or
/// `ExpertCache::plan` asserts with "expert cache cannot place requested
/// misses": at `top_k` slots the previous token's reservation would leave no
/// room at all.
#[test]
fn the_one_bank_fallback_reproduces_the_sequential_logits() {
    let slots = TOP_K;
    assert!(
        slots < 2 * TOP_K,
        "only meaningful below the pipelining threshold"
    );
    for (layout, arm) in CASES {
        let mut runner = open_arm("one-bank", layout, slots, arm);
        let prompt = make_prompt(11);
        let expected = sequential_prefill(&mut runner, &prompt);
        let actual = chunked_prefill(&mut runner, &prompt, prompt.len());
        assert_eq!(
            actual, expected,
            "{layout:?}/{arm:?}: the banks == 1 fallback changed the logits"
        );
    }
}

/// `TURBOSPARK_ROUTED_BATCH` names an arm this driver does not have, so it is
/// refused by name rather than served by the per-token loop under the
/// batched label (`crates/runtime/CLAUDE.md` Gotcha 22).
#[test]
fn the_batched_routed_seam_is_refused_by_name_on_this_family() {
    let mut runner = open_runner("routed-batch-refused", Layout::Affine, SLOTS);
    runner.set_routed_batch_prefill(true);
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let err = runner
        .prefill_chunk(&make_prompt(4), 0, &mut logits)
        .expect_err("the batched routed seam must be refused on this family");
    assert!(err.contains("TURBOSPARK_ROUTED_BATCH"), "got {err}");
}

/// The batched arm is served on an INT4 install now (the cases above), so the
/// by-name refusal moves to an install it cannot serve: the MXFP4 fixture's
/// resident matrices are BF16, which the M-row GEMM has no kernel for.
/// Forced, it is refused by name; unset, auto takes the per-token arm.
#[test]
fn the_batched_gemv_seam_is_refused_by_name_on_a_non_int4_install() {
    let mut runner = open_runner("batched-gemv-refused", Layout::Mxfp4, SLOTS);
    runner.set_batched_gemv_prefill(true);
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let err = runner
        .prefill_chunk(&make_prompt(4), 0, &mut logits)
        .expect_err("the batched resident-GEMV seam must be refused on a BF16 install");
    assert!(
        err.contains("TURBOSPARK_BATCHED_GEMV") && err.contains("INT4-affine"),
        "got {err}"
    );
}

/// With `TURBOSPARK_BATCHED_GEMV` unset and no setter call, the driver picks
/// the batched arm by itself on the INT4 fixture and still reproduces the
/// sequential logits. The allocation count is how the test knows which arm
/// ran: only the batched arm builds its M-row scratch. On the MXFP4 fixture
/// the same pick falls back to the per-token arm, silently, and allocates
/// nothing more than a runner held to that arm.
#[test]
fn an_unset_seam_takes_the_batched_arm_where_it_qualifies() {
    if std::env::var_os("TURBOSPARK_BATCHED_GEMV").is_some() {
        eprintln!("TURBOSPARK_BATCHED_GEMV is set, which replaces the automatic pick; skipping");
        return;
    }
    let prompt = make_prompt(37);
    for (layout, takes_batched) in [(Layout::Affine, true), (Layout::Mxfp4, false)] {
        let mut per_token = open_arm("auto-per-token", layout, SLOTS, Arm::PerToken);
        let expected = sequential_prefill(&mut per_token, &prompt);
        let _ = chunked_prefill(&mut per_token, &prompt, prompt.len());
        let per_token_buffers = per_token.gpu_buffer_allocations();

        let mut auto = open_runner("auto", layout, SLOTS);
        // Steel off so the comparison below can be bit for bit; the Steel
        // arm has its own case.
        auto.set_prefill_steel(false);
        let _ = sequential_prefill(&mut auto, &prompt);
        let actual = chunked_prefill(&mut auto, &prompt, prompt.len());
        assert_eq!(
            auto.gpu_buffer_allocations() > per_token_buffers,
            takes_batched,
            "{layout:?}: the automatic pick took the wrong arm"
        );
        assert_eq!(
            actual, expected,
            "{layout:?}: the automatically chosen arm must reproduce the sequential logits"
        );
    }
}

/// The default batched arm runs its INT4 GEMMs on the vendored Steel kernel
/// at a 64-row micro-batch, which rounds each weight to an FP16 operand and
/// reduces in the matrix units' order. So it must NOT reproduce the
/// sequential logits bit for bit (the proof that Steel ran: 37 rows in one
/// micro-batch clears its 32-row floor), it must pick the same next token,
/// and it must stay within 1% of the largest logit, the dense arm's bound.
#[test]
fn the_steel_prefill_arm_stays_close_to_the_sequential_logits() {
    for var in ["TURBOSPARK_BATCHED_GEMV", "TURBOSPARK_PREFILL_STEEL"] {
        if std::env::var_os(var).is_some() {
            eprintln!("{var} is set, which replaces the default arm; skipping");
            return;
        }
    }
    let prompt = make_prompt(37);
    let mut per_token = open_arm("steel-ref", Layout::Affine, SLOTS, Arm::PerToken);
    let expected = sequential_prefill(&mut per_token, &prompt);

    let mut steel = open_runner("steel-arm", Layout::Affine, SLOTS);
    let actual = chunked_prefill(&mut steel, &prompt, prompt.len());
    assert_ne!(
        actual, expected,
        "bit-identical logits: the Steel GEMM did not run"
    );
    assert_eq!(
        argmax(&actual),
        argmax(&expected),
        "the Steel arm changed the next token"
    );
    let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let worst = actual
        .iter()
        .zip(&expected)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("steel prefill: worst |logit diff| {worst} against max |logit| {scale}");
    assert!(
        worst <= 0.01 * scale,
        "worst logit diff {worst} against max {scale}"
    );
}

/// The batched K/V projections write M ADJACENT cache rows, so a micro-batch
/// that straddles the KV capacity is refused by name when the arm is forced.
/// Unreachable at a real context (admission refuses a prompt past it), which
/// is why the condition is manufactured here with a window narrower than the
/// prompt: a guard nobody has run is only an argument.
#[test]
fn the_batched_arm_refuses_a_micro_batch_that_wraps_the_kv_ring() {
    let dir = temp_dir("batched-wrap");
    let arch = turbospark_repack::build_synthetic_qwen_gdn_moe_install(
        &dir,
        VOCAB,
        LAYERS,
        EXPERTS,
        "tiny-qwen36-wrap",
    )
    .expect("MoE qwen install builds");
    let mut runner = RealForwardRunner::open_with_options(&dir, arch, 8, SLOTS)
        .expect("MoE qwen install opens at a narrow context");
    let _ = std::fs::remove_dir_all(&dir);
    runner.set_prefill_steel(false);
    runner.set_batched_gemv_prefill(true);
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let err = runner
        .prefill_chunk(&make_prompt(11), 0, &mut logits)
        .expect_err("a micro-batch wider than the KV ring must be refused");
    assert!(
        err.contains("wraps") && err.contains("KV capacity"),
        "the refusal must name what it cannot do; got {err}"
    );
}

fn open_kv_bits_runner(tag: &str) -> RealForwardRunner {
    let dir = temp_dir(tag);
    let arch = turbospark_repack::build_synthetic_qwen_gdn_moe_install(
        &dir,
        VOCAB,
        LAYERS,
        EXPERTS,
        "tiny-qwen36-kv-bits",
    )
    .expect("MoE qwen install builds");
    let runner = RealForwardRunner::open_with_kv_quant(
        &dir,
        arch,
        MAX_CONTEXT,
        model_io::ExpertCacheSlots::Fixed(SLOTS),
        turbospark_runtime::DraftPolicies::off(),
        turbospark_runtime::SteeringPolicy::off(),
        1,
        model_io::KvQuant::TurboQuant {
            k_bits: 3,
            v_bits: 4,
        },
    )
    .expect("MoE qwen install opens with kv-bits");
    let _ = std::fs::remove_dir_all(&dir);
    runner
}

/// `--kv-bits` is not wired to the batched K/V projections, which write the
/// cache directly, so forcing the batched arm under it is refused by name.
#[test]
fn the_batched_arm_refuses_kv_bits_by_name() {
    let mut runner = open_kv_bits_runner("batched-kv-bits");
    runner.set_batched_gemv_prefill(true);
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let err = runner
        .prefill_chunk(&make_prompt(4), 0, &mut logits)
        .expect_err("the batched arm under --kv-bits must be refused");
    assert!(
        err.contains("--kv-bits") && err.contains("BATCHED_GEMV"),
        "got {err}"
    );
}

/// The automatic pick asks the same question BEFORE the first layer, so under
/// `--kv-bits` it takes the per-token arm and prefills, where reaching the
/// batched block's own deeper refusal would fail the prompt mid-layer.
#[test]
fn an_unset_seam_falls_back_to_the_per_token_arm_under_kv_bits() {
    if std::env::var_os("TURBOSPARK_BATCHED_GEMV").is_some() {
        eprintln!("TURBOSPARK_BATCHED_GEMV is set, which replaces the automatic pick; skipping");
        return;
    }
    let mut runner = open_kv_bits_runner("auto-kv-bits");
    let prompt = make_prompt(11);
    let expected = sequential_prefill(&mut runner, &prompt);
    let actual = chunked_prefill(&mut runner, &prompt, prompt.len());
    assert_eq!(
        actual, expected,
        "auto under --kv-bits must take the per-token arm and match sequential"
    );
}
