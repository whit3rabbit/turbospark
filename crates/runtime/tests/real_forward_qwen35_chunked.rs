#![cfg(target_os = "macos")]
//! Chunked prefill on the DENSE half of the qwen linear-attention flow
//! (`qwenGdnDense`, ROADMAP's 1-bit entry): the sixth
//! [`ChunkedPrefillRunner`] implementation, and the same "step 1" shape as
//! `real_forward_llama_dense_chunked.rs` -- loop the EXISTING per-token
//! kernels inside a micro-batch, batching command buffers rather than GEMVs
//! (`crates/runtime/src/families/qwen/prefill.rs`'s header has the full
//! design argument, including why the GDN recurrent state's ordering is
//! unaffected by chunking).
//!
//! **The bar is byte-identity against the SEQUENTIAL path, not coherence**,
//! exactly as the llama and Gemma 4 chunked tests establish: every case here
//! compares against `produce_prefill` / `produce`, never against another
//! chunked arm. That reference is not a fresh baseline invented for this
//! file -- it is the same sequential flow `real_forward_qwen35.rs` already
//! pins with its own frozen digest, so what THIS file is proving is narrower
//! and deliberately so: that grouping tokens into fewer command buffers does
//! not change the bytes, not that the underlying math is right (that
//! question belongs to `real_forward_qwen35.rs` and `qwen38_quality_gate`).
//! The fixture's weights are deterministic but not trained.

use half::f16;
use turbospark_repack::{
    build_synthetic_qwen_gdn_dense_install, build_synthetic_qwen_gdn_dense_install_at_bits,
    build_synthetic_qwen_gdn_moe_install,
};
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};
use turbospark_vision_io::MropePositions;

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;
const MOE_EXPERTS: i64 = 8;
/// The width the batched arm needs. `encode_gemm_any` is INT4-affine only,
/// so the 1-bit default this file's other cases use cannot reach the M-row
/// GEMM at all -- which is what
/// [`the_batched_gemv_seam_is_refused_at_a_width_with_no_kernel`] pins.
const INT4: u32 = 4;
/// Same length as the llama/Gemma 4 chunked fixtures: enough to cross
/// several micro-batch boundaries at the smaller spans in the sweep below
/// and to leave the last one partial. This architecture's fixture mask is
/// `qwen_hybrid_layer_mask(4)` (matching `real_forward_qwen35.rs`'s own
/// comment): layers 0-2 are gated DeltaNet (mask-2, linear) and layer 3 is
/// full attention (mask-1), so every chunk span below exercises the GDN
/// recurrent state crossing a `prefill_chunk` boundary at least once.
const PROMPT: [i32; 11] = [5, 9, 2, 7, 1, 3, 8, 4, 6, 0, 11];

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-qwen35-dense-chunked-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn open_runner(tag: &str) -> RealForwardRunner {
    let dir = temp_dir(tag);
    build_synthetic_qwen_gdn_dense_install(&dir, VOCAB, LAYERS, "tiny-bonsai")
        .expect("dense qwen3_5 install builds");
    let peeked = turbospark_repack::peek_manifest_arch(&dir)
        .expect("a dense manifest peeks against the qwen3_5 baseline");
    RealForwardRunner::open(&dir, peeked).expect("a dense qwen3_5 install opens")
}

/// The same install at INT4, which is the only width the batched arm's
/// `encode_gemm_any` has a kernel for.
fn open_int4_runner(tag: &str) -> RealForwardRunner {
    let dir = temp_dir(tag);
    build_synthetic_qwen_gdn_dense_install_at_bits(&dir, VOCAB, LAYERS, "tiny-int4", INT4)
        .expect("an INT4 dense qwen3_5 install builds");
    let peeked = turbospark_repack::peek_manifest_arch(&dir)
        .expect("a dense manifest peeks against the qwen3_5 baseline");
    RealForwardRunner::open(&dir, peeked).expect("an INT4 dense qwen3_5 install opens")
}

/// The reference: every prompt token through `produce_prefill` but the
/// last, which goes through `produce`, exactly as `run_raw_completion` does.
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

#[test]
fn a_dense_qwen_install_reports_chunked_prefill_support() {
    let runner = open_runner("supports");
    assert!(
        runner.supports_chunked_prefill(),
        "a dense qwen install must be servable by the chunked driver"
    );
}

#[test]
fn a_chunked_prefill_is_byte_identical_to_the_sequential_one() {
    let mut runner = open_runner("whole-chunk");

    let expected = sequential_prefill(&mut runner, &PROMPT);
    assert!(
        expected.iter().all(|v| v.is_finite()),
        "the reference itself must be finite before anything is compared to it"
    );
    // The fixture has to be able to SEE a difference: a prompt whose logits
    // never move cannot distinguish a working driver from a broken one.
    assert!(
        expected.iter().any(|&v| v != expected[0]),
        "degenerate reference logits: this fixture cannot discriminate"
    );

    let actual = chunked_prefill(&mut runner, &PROMPT, PROMPT.len());
    assert_eq!(
        actual, expected,
        "chunked prefill must reproduce the sequential logits exactly"
    );
}

#[test]
fn the_chunk_boundary_does_not_move_the_logits() {
    // The same question one level out, and the one that would catch a
    // driver whose per-token row leaked across a micro-batch, across
    // layers, or across the GDN recurrent state's own crossing of a
    // `prefill_chunk` boundary. Spans of 1 also cover the degenerate
    // micro-batch of a single token.
    let mut runner = open_runner("boundary");

    let expected = sequential_prefill(&mut runner, &PROMPT);
    for chunk in [1usize, 2, 3, 4, 7, 11] {
        let actual = chunked_prefill(&mut runner, &PROMPT, chunk);
        assert_eq!(
            actual, expected,
            "chunk span {chunk} changed the logits; a per-token scratch row or the GDN \
             state's ordering is a function of the chunk boundary"
        );
    }
}

/// `MFERENCE_BATCHED_GEMV` is WIRED on this family since 2026-08-29, so what
/// stays refused is the WIDTH rather than the family: `encode_gemm_any` has
/// an INT4-affine kernel and nothing else, and the 1-bit and 2-bit
/// checkpoints of this same architecture reach no M-row GEMM at all.
///
/// The refusal has to name BOTH -- the seam the caller set and the width
/// that defeats it -- because either half alone sends the reader somewhere
/// useless: the seam alone reads as "this family cannot", which stopped
/// being true, and the width alone does not say which request produced it.
#[test]
fn the_batched_gemv_seam_is_refused_at_a_width_with_no_kernel() {
    let mut runner = open_runner("batched-gemv-refused");
    runner.set_batched_gemv_prefill(true);

    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let err = runner
        .prefill_chunk(&PROMPT, 0, &mut logits)
        .expect_err("the batched resident-GEMV seam must be refused at a width with no kernel");
    let text = err.to_string();
    assert!(
        text.contains("MFERENCE_BATCHED_GEMV"),
        "the refusal must name the seam the caller set; got {text}"
    );
    assert!(
        text.contains("INT4"),
        "the refusal must name the width that has a kernel; got {text}"
    );
}

/// The batched arm against the SEQUENTIAL path, at INT4.
///
/// **WHAT THIS PINS IS THE WIRING, NOT THE ARITHMETIC**, and the difference
/// is measured rather than hedged. On the REAL install the M-row and
/// per-token paths differ by 6.2e-8 to 1.5e-5 nats with the argmax agreeing
/// on every row -- a batched-vs-cached shape floor every engine has
/// (`crates/bench/tests/batched_forward_probe.rs`, commit `e8deb6c`). This
/// fixture is BLIND to that floor: `real_forward_qwen35_batched_onset.rs`
/// sweeps the same two paths on the same builder at the same width and reads
/// 0 differing logits of 128 at every span, because untrained weights at
/// this scale do not land near a rounding boundary.
///
/// So byte-identity here is achievable and worth asserting -- it catches a
/// wrong row offset, a norm reading the wrong slot, a residual landing in
/// the wrong row, a GDN state advanced out of order -- and it must NOT be
/// read as evidence that the two arms agree numerically on a real install.
/// That question belongs to `qwen38_quality_gate`.
#[test]
fn the_batched_gemv_arm_reproduces_the_sequential_logits() {
    let mut runner = open_int4_runner("batched-gemv-identity");

    let expected = sequential_prefill(&mut runner, &PROMPT);
    assert!(
        expected.iter().all(|v| v.is_finite()),
        "the reference itself must be finite before anything is compared to it"
    );
    assert!(
        expected.iter().any(|&v| v != expected[0]),
        "degenerate reference logits: this fixture cannot discriminate"
    );

    runner.set_batched_gemv_prefill(true);
    let actual = chunked_prefill(&mut runner, &PROMPT, PROMPT.len());
    assert_eq!(
        actual, expected,
        "the batched-GEMV arm must reproduce the sequential logits on a fixture blind to the \
         shape floor; a difference here is a wiring bug, not the floor"
    );
}

/// The same question across micro-batch boundaries. Spans of 1 exercise the
/// degenerate M-row GEMM of a single row, which is where a batched kernel
/// that indexed its rows wrongly would still look right.
#[test]
fn the_batched_gemv_arm_is_span_invariant() {
    let mut runner = open_int4_runner("batched-gemv-spans");

    let expected = sequential_prefill(&mut runner, &PROMPT);
    runner.set_batched_gemv_prefill(true);
    for chunk in [1usize, 2, 3, 4, 7, 11] {
        let actual = chunked_prefill(&mut runner, &PROMPT, chunk);
        assert_eq!(
            actual, expected,
            "chunk span {chunk} moved the batched arm's logits; a row offset or the GDN \
             state's ordering is a function of the micro-batch boundary"
        );
    }
}

/// The batched arm's KV-wrap backstop, reached by shrinking the context
/// until a micro-batch cannot fit below it.
///
/// `encode_full_attention_block_batched`'s k/v projections write M ADJACENT
/// cache slots in one dispatch, so a micro-batch straddling the ring's wrap
/// would scatter into row 0 -- finite, fluent, and attending to the wrong
/// keys. `produce_batched` carries the same check; this is the chunked
/// driver's copy of it.
///
/// It is UNREACHABLE at any real context (a prompt longer than the window is
/// refused upstream, so the modulo is the identity), which is exactly why it
/// needs a test that manufactures the condition: a guard whose only evidence
/// is an argument that it cannot fire is a guard nobody has run.
#[test]
fn the_batched_arm_refuses_a_micro_batch_that_wraps_the_kv_ring() {
    let dir = temp_dir("batched-gemv-wrap");
    build_synthetic_qwen_gdn_dense_install_at_bits(&dir, VOCAB, LAYERS, "tiny-int4", INT4)
        .expect("an INT4 dense qwen3_5 install builds");
    let peeked = turbospark_repack::peek_manifest_arch(&dir).expect("a dense manifest peeks");
    // A window narrower than the prompt, so the first micro-batch alone
    // overruns it.
    let mut runner = RealForwardRunner::open_with_options(&dir, peeked, 8, 16)
        .expect("an INT4 dense install opens at a narrow context");
    let _ = std::fs::remove_dir_all(&dir);
    runner.set_batched_gemv_prefill(true);

    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let err = runner
        .prefill_chunk(&PROMPT, 0, &mut logits)
        .expect_err("a micro-batch wider than the KV ring must be refused");
    let text = err.to_string();
    assert!(
        text.contains("wraps") && text.contains("KV capacity"),
        "the refusal must name what it cannot do; got {text}"
    );
}

/// The DEFAULT arm must allocate no M-row scratch, and the batched arm must
/// allocate it exactly once.
///
/// This is the memory claim in `families/qwen/prefill.rs`'s header stated as
/// a test rather than as an argument. `BatchedScratch` is ~10 MiB on the real
/// install (mostly its `batch * vocab` logits plane, which this driver never
/// reads -- the head stays a single-row GEMV), so allocating it at open would
/// move `qwen38_memory_oracle`'s frozen row for a seam almost nobody sets.
/// A counter is what catches that; a footprint ceiling with 87 MiB of
/// headroom would not.
#[test]
fn the_m_row_scratch_is_allocated_only_when_the_seam_is_on_and_only_once() {
    let mut runner = open_int4_runner("batched-gemv-alloc");
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];

    // Warm every lazy allocation the DEFAULT path has, so the baseline below
    // is a steady state rather than a first run.
    runner.reset();
    runner
        .prefill_chunk(&PROMPT, 0, &mut logits)
        .expect("the default arm prefills");
    let default_arm = runner.gpu_buffer_allocations();

    runner.reset();
    runner
        .prefill_chunk(&PROMPT, 0, &mut logits)
        .expect("the default arm prefills again");
    assert_eq!(
        runner.gpu_buffer_allocations(),
        default_arm,
        "the default arm allocated a buffer on a second identical chunk"
    );

    runner.set_batched_gemv_prefill(true);
    runner.reset();
    runner
        .prefill_chunk(&PROMPT, 0, &mut logits)
        .expect("the batched arm prefills");
    let after_first = runner.gpu_buffer_allocations();
    assert!(
        after_first > default_arm,
        "the batched arm allocated nothing, so either the scratch was already built on the \
         default path (which would move the frozen oracle row) or the arm never ran"
    );

    runner.reset();
    runner
        .prefill_chunk(&PROMPT, 0, &mut logits)
        .expect("the batched arm prefills again");
    assert_eq!(
        runner.gpu_buffer_allocations(),
        after_first,
        "ensure_batched_prefill is not idempotent: a second chunk reallocated the M-row \
         scratch, which leaks one copy per chunk on a long prompt"
    );
}

/// The vision refusal, by name rather than a silent fallback
/// (`families/qwen/prefill.rs`'s header, `crates/runtime/CLAUDE.md` Gotcha
/// 27): the image injection is this family's only embedding call site
/// today, and the chunked driver does not carry a second one.
///
/// The position table is DEGENERATE (`(p, p, p)` triples, no spans) and no
/// image embedding is attached -- following
/// `vision_inject_synthetic.rs`'s own "image-free map" construction -- so
/// this proves the refusal fires on the mere PRESENCE of a `prompt_vision`
/// map, which is what `supports_chunked_prefill` and the driver's own guard
/// both check, rather than on any particular span content.
#[test]
fn a_vision_prompt_is_refused_by_name() {
    let mut runner = open_runner("vision-refused");
    let positions = MropePositions {
        triples: (0..PROMPT.len() as i32).map(|p| (p, p, p)).collect(),
        rope_delta: 0,
        spans: Vec::new(),
    };
    runner
        .set_prompt_vision(&[], &positions, PROMPT.len())
        .expect("an image-free map validates");
    assert!(
        !runner.supports_chunked_prefill(),
        "an install with a live prompt_vision map must not report chunked-prefill support"
    );

    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let err = runner
        .prefill_chunk(&PROMPT, 0, &mut logits)
        .expect_err("a vision prompt must be refused by the chunked driver");
    assert!(
        err.to_string().contains("text-only"),
        "the refusal must name the reason; got {err}"
    );
}

/// An open drafter is refused by name too (`families/qwen/prefill.rs`'s
/// header): the driver does not encode the DFlash2 aux-capture hook the
/// sequential dense branch fires on every pass, so silently prefilling
/// through it would leave the drafter reading a stale/empty capture.
#[test]
fn an_open_drafter_is_refused_by_name() {
    let dir = temp_dir("drafter-refused");
    turbospark_repack::build_synthetic_qwen_gdn_dense_install_with_mtp(
        &dir, VOCAB, LAYERS, "mtp-toy", 1,
    )
    .expect("a dense qwen3_5 install WITH an MTP head builds");
    let peeked = turbospark_repack::peek_manifest_arch(&dir).expect("a manifest with a head peeks");
    let mut runner = RealForwardRunner::open_with_options_and_speculation(
        &dir,
        peeked,
        4096,
        16,
        turbospark_runtime::DraftPolicies::mtp(turbospark_runtime::MtpDraftPolicy::Fixed(2)),
    )
    .expect("install with a head opens with the head asked for");
    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        !runner.supports_chunked_prefill(),
        "an install with an open drafter must not report chunked-prefill support"
    );

    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let err = runner
        .prefill_chunk(&PROMPT, 0, &mut logits)
        .expect_err("an open drafter must be refused by the chunked driver");
    assert!(
        err.to_string().contains("aux capture"),
        "the refusal must name the reason; got {err}"
    );
}

/// The MoE half (`qwenGdnMoe`) stays refused by name, proving the dense-only
/// scope holds even though `RealQwenState` serves both families from one
/// flow. Mirrors `real_forward_llama_dense_chunked.rs`'s equivalent case for
/// its own MoE sibling.
#[test]
fn a_moe_qwen_install_is_still_refused_by_name() {
    let dir = temp_dir("moe-refused");
    let arch =
        build_synthetic_qwen_gdn_moe_install(&dir, VOCAB, LAYERS, MOE_EXPERTS, "tiny-qwen36")
            .expect("MoE qwen install builds");
    let mut runner = RealForwardRunner::open(&dir, arch).expect("MoE qwen install opens");
    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        !runner.supports_chunked_prefill(),
        "a MoE qwen install must not report chunked-prefill support"
    );

    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let err = runner
        .prefill_chunk(&PROMPT, 0, &mut logits)
        .expect_err("a MoE qwen install must be refused by the chunked driver");
    assert!(
        err.to_string().contains("qwenGdnDense") || err.to_string().contains("dense qwen"),
        "the refusal must name what IS served; got {err}"
    );
}
