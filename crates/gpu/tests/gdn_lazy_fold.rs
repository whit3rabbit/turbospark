//! Parity tests, on real Metal hardware, for the LAZY verify tape
//! (`TURBOSPARK_GDN_LAZY_TAPE`, docs/KAPITSA_KERNELS.md item 1). The round
//! end folds recorded rank-1 updates instead of restore-plus-replay, so the
//! property everything else stands on is: folding the recorded `(k, g,
//! delta)` of N rows from the block-start state is BIT-identical to the
//! eager `gdn_delta_step_prefill` write-back after the same N rows, and the
//! lazy build's y is bit-identical to the eager build's. The partial-row
//! case is the retaining rollback's contract, tested at its own row counts.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_compute::f32_to_bf16;
use turbospark_gpu::{
    encode_copy_rows_fp16_strided, encode_gdn_delta_prefill, encode_gdn_delta_prefill_lazy,
    encode_gdn_state_fold, read_buffer_f16, read_f32_buffer, write_buffer_bytes, GdnShape,
    MetalBuffer, MetalContext,
};

const HK: usize = 2;
const HV: usize = 4;
const DK: usize = 32;
const DV: usize = 32;
const ROWS: usize = 5;

fn shape() -> GdnShape {
    GdnShape {
        num_k_heads: HK as u32,
        num_v_heads: HV as u32,
        key_head_dim: DK as u32,
        value_head_dim: DV as u32,
        conv_kernel_size: 4,
    }
}

fn deterministic(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (((state % 2000) as f32 / 1000.0) - 1.0) * scale
        })
        .collect()
}

fn bf16_bits(seed: u64, n: usize, scale: f32, center: f32) -> Vec<u16> {
    deterministic(seed, n, scale)
        .iter()
        .map(|&v| f32_to_bf16(center + v))
        .collect()
}

fn f16_values(seed: u64, n: usize, scale: f32) -> Vec<f16> {
    deterministic(seed, n, scale)
        .iter()
        .map(|&v| f16::from_f32(v))
        .collect()
}

fn half_bytes(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn zeroed(context: &MetalContext, bytes: usize) -> MetalBuffer {
    let buffer = context.new_output_buffer(bytes as u64);
    write_buffer_bytes(&buffer, 0, &vec![0u8; bytes]);
    buffer
}

/// Post-qk-norm conv_out rows ([q|k|v] per row), a/b gate rows, and the two
/// bf16 scalar weights the g recompute reads. Everything the delta kernel
/// and the fold consume, already in the halves/bf16 the GPU sees.
struct Inputs {
    conv_out: MetalBuffer,
    a: MetalBuffer,
    b: MetalBuffer,
    a_log: MetalBuffer,
    dt_bias: MetalBuffer,
    k_width: usize,
    delta_width: usize,
}

impl Inputs {
    fn new(context: &MetalContext, rows: usize) -> Self {
        let c = shape().qkv_dim() as usize;
        let conv = f16_values(0xD1CE, rows * c, 1.0);
        let a = f16_values(0xA000, rows * HV, 1.5);
        let b = f16_values(0xB000, rows * HV, 1.0);
        let a_log = bf16_bits(0xA100, HV, 1.25, 0.25);
        let dt_bias = bf16_bits(0xA200, HV, 0.5, 0.0);
        Self {
            conv_out: context.new_buffer_with_data(&half_bytes(&conv)),
            a: context.new_buffer_with_data(&half_bytes(&a)),
            b: context.new_buffer_with_data(&half_bytes(&b)),
            a_log: context.new_buffer_with_data(&u16_bytes(&a_log)),
            dt_bias: context.new_buffer_with_data(&u16_bytes(&dt_bias)),
            k_width: HK * DK,
            delta_width: HV * DV,
        }
    }
}

fn start_state(context: &MetalContext) -> MetalBuffer {
    zeroed(context, HV * DV * DK * 4)
}

/// One combined run: the eager kernel over `rows` rows and the lazy kernel
/// plus fold over the same rows, each from the SAME start state. Returns
/// both end states, both y planes, and the lazy state SNAPSHOT taken between
/// the record pass and the fold (the window the runtime's invariant lives
/// in).
#[allow(clippy::type_complexity)]
fn run_eager_and_lazy(
    context: &mut MetalContext,
    inputs: &Inputs,
    rows: usize,
) -> (
    (MetalBuffer, MetalBuffer),
    (MetalBuffer, MetalBuffer, MetalBuffer),
    Vec<f32>,
) {
    let n = rows as u32;
    let vd = shape().value_dim() as usize;

    // Eager: state advanced in place by the kernel's own write-back.
    let state_eager = start_state(context);
    let y_eager = zeroed(context, rows * vd * 2);
    let pass = context.begin_pass();
    encode_gdn_delta_prefill(
        context,
        &pass,
        shape(),
        (&inputs.conv_out, 0),
        (&inputs.a, 0),
        (&inputs.b, 0),
        (&inputs.a_log, 0),
        (&inputs.dt_bias, 0),
        &state_eager,
        (&y_eager, 0),
        n,
    )
    .expect("eager delta prefill");
    pass.commit_and_wait();

    // Lazy: the state must still hold the start values after the record
    // pass (that is the whole point), with k/delta recorded for the fold.
    // Record and fold are SEPARATE passes so the snapshot between them is
    // observable.
    let state_lazy = start_state(context);
    let y_lazy = zeroed(context, rows * vd * 2);
    let k_tape = zeroed(context, rows * inputs.k_width * 2);
    let delta_tape = zeroed(context, rows * inputs.delta_width * 4);
    let pass = context.begin_pass();
    encode_copy_rows_fp16_strided(
        context,
        &pass,
        (&inputs.conv_out, (inputs.k_width * 2) as u64),
        (&k_tape, 0),
        n,
        inputs.k_width as u32,
        shape().qkv_dim(),
        inputs.k_width as u32,
    )
    .expect("k capture");
    encode_gdn_delta_prefill_lazy(
        context,
        &pass,
        shape(),
        (&inputs.conv_out, 0),
        (&inputs.a, 0),
        (&inputs.b, 0),
        (&inputs.a_log, 0),
        (&inputs.dt_bias, 0),
        &state_lazy,
        (&y_lazy, 0),
        (&delta_tape, 0),
        n,
    )
    .expect("lazy delta prefill");
    pass.commit_and_wait();
    let untouched = read_f32_buffer(&state_lazy, HV * DV * DK);

    let pass = context.begin_pass();
    encode_gdn_state_fold(
        context,
        &pass,
        shape(),
        (&k_tape, 0),
        (&delta_tape, 0),
        (&inputs.a, 0),
        (&inputs.a_log, 0),
        (&inputs.dt_bias, 0),
        &state_lazy,
        n,
    )
    .expect("fold");
    pass.commit_and_wait();

    (
        (state_eager, y_eager),
        (state_lazy, y_lazy, delta_tape),
        untouched,
    )
}

fn assert_state_identical(got: &[f32], want: &[f32], what: &str) {
    let mismatch = got
        .iter()
        .zip(want)
        .position(|(&g, &w)| g.to_bits() != w.to_bits());
    if let Some(i) = mismatch {
        panic!(
            "{what}: state diverges at element {i} (got {:e}, want {:e})",
            got[i], want[i]
        );
    }
}

#[test]
fn lazy_prefill_leaves_state_untouched_and_y_bit_identical() {
    let mut context = MetalContext::new().expect("Metal device");
    let inputs = Inputs::new(&context, ROWS);
    let ((_state_eager, y_eager), (_state_lazy, y_lazy, _delta), untouched) =
        run_eager_and_lazy(&mut context, &inputs, ROWS);

    let want_y = read_buffer_f16(&y_eager, 0, ROWS * inputs.delta_width);
    let got_y = read_buffer_f16(&y_lazy, 0, ROWS * inputs.delta_width);
    assert!(
        want_y
            .iter()
            .zip(&got_y)
            .all(|(&w, &g)| w.to_bits() == g.to_bits()),
        "lazy y must be bit-identical to the eager build's"
    );
    assert!(
        want_y.iter().any(|&v| v.to_f32() != 0.0),
        "y must be non-trivial for the identity check to mean anything"
    );

    // The lazy build's ENTIRE contract is that the live state never moved:
    // the snapshot taken between the record pass and the fold still holds
    // the start values.
    assert!(
        untouched.iter().all(|&v| v == 0.0),
        "lazy prefill must leave the state at its start values"
    );
}

#[test]
fn fold_matches_the_eager_write_back_after_the_same_rows() {
    let mut context = MetalContext::new().expect("Metal device");
    let inputs = Inputs::new(&context, ROWS);
    let ((state_eager, _), (state_lazy, _, _), _untouched) =
        run_eager_and_lazy(&mut context, &inputs, ROWS);

    let want = read_f32_buffer(&state_eager, HV * DV * DK);
    let got = read_f32_buffer(&state_lazy, HV * DV * DK);
    assert_state_identical(&got, &want, "fold over all rows");
}

#[test]
fn partial_fold_matches_an_eager_run_over_that_prefix() {
    // The retaining-rollback case: keep FEWER rows than the verify fed. The
    // fold over `keep` rows from the block start must equal an eager run
    // over exactly those `keep` rows.
    for keep in [1, 3, ROWS - 1] {
        let mut context = MetalContext::new().expect("Metal device");
        let inputs = Inputs::new(&context, ROWS);
        let vd = shape().value_dim() as usize;

        // Reference: eager over the prefix only (the kernel reads the first
        // `keep` rows and stops).
        let state_eager = start_state(&context);
        let y = zeroed(&context, keep * vd * 2);
        let pass = context.begin_pass();
        encode_gdn_delta_prefill(
            &mut context,
            &pass,
            shape(),
            (&inputs.conv_out, 0),
            (&inputs.a, 0),
            (&inputs.b, 0),
            (&inputs.a_log, 0),
            (&inputs.dt_bias, 0),
            &state_eager,
            (&y, 0),
            keep as u32,
        )
        .expect("eager prefix");
        pass.commit_and_wait();

        // Lazy over ALL rows (recording everything), then fold only `keep`.
        let state_lazy = start_state(&context);
        let y_lazy = zeroed(&context, ROWS * vd * 2);
        let k_tape = zeroed(&context, ROWS * inputs.k_width * 2);
        let delta_tape = zeroed(&context, ROWS * inputs.delta_width * 4);
        let pass = context.begin_pass();
        encode_copy_rows_fp16_strided(
            &mut context,
            &pass,
            (&inputs.conv_out, (inputs.k_width * 2) as u64),
            (&k_tape, 0),
            ROWS as u32,
            inputs.k_width as u32,
            shape().qkv_dim(),
            inputs.k_width as u32,
        )
        .expect("k capture");
        encode_gdn_delta_prefill_lazy(
            &mut context,
            &pass,
            shape(),
            (&inputs.conv_out, 0),
            (&inputs.a, 0),
            (&inputs.b, 0),
            (&inputs.a_log, 0),
            (&inputs.dt_bias, 0),
            &state_lazy,
            (&y_lazy, 0),
            (&delta_tape, 0),
            ROWS as u32,
        )
        .expect("lazy prefill");
        encode_gdn_state_fold(
            &mut context,
            &pass,
            shape(),
            (&k_tape, 0),
            (&delta_tape, 0),
            (&inputs.a, 0),
            (&inputs.a_log, 0),
            (&inputs.dt_bias, 0),
            &state_lazy,
            keep as u32,
        )
        .expect("partial fold");
        pass.commit_and_wait();

        let want = read_f32_buffer(&state_eager, HV * DV * DK);
        let got = read_f32_buffer(&state_lazy, HV * DV * DK);
        assert_state_identical(&got, &want, &format!("fold over {keep} of {ROWS} rows"));
    }
}
