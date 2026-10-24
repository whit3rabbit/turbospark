//! `FC_GDN_DK` parity, on real Metal hardware. The baked `Dk = 128` build of
//! every delta kernel (decode, eager prefill, lazy prefill, fold) must be
//! BIT-identical to the runtime-`Dk` build: same y, same state, same delta
//! tape. A unit-test module (not `tests/`) because forcing the runtime build
//! needs the thread-local override [`baked_key_dim`] reads, and a
//! thread-local is the one switch parallel tests cannot race on.

use std::cell::Cell;

use half::f16;
use turbospark_compute::f32_to_bf16;

use super::{
    baked_key_dim, delta_pipeline, encode_gdn_delta_decode, encode_gdn_delta_prefill,
    encode_gdn_delta_prefill_lazy, encode_gdn_state_fold, GdnShape, BAKED_KEY_DIM,
};
use crate::bytes::read_f32_buffer;
use crate::context::{read_buffer_f16, write_buffer_bytes, MetalContext};

thread_local! {
    /// When set, [`baked_key_dim`] returns 0 on this thread: the
    /// runtime-`Dk` build every dispatch would take at another key dim.
    pub(super) static FORCE_RUNTIME_KEY_DIM: Cell<bool> = const { Cell::new(false) };
}

const ROWS: usize = 8;

fn shape(key_head_dim: u32) -> GdnShape {
    GdnShape {
        num_k_heads: 2,
        num_v_heads: 4,
        key_head_dim,
        value_head_dim: 32,
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

fn half_bytes(seed: u64, n: usize, scale: f32) -> Vec<u8> {
    deterministic(seed, n, scale)
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
        .collect()
}

fn bf16_bytes(seed: u64, n: usize, scale: f32, center: f32) -> Vec<u8> {
    deterministic(seed, n, scale)
        .iter()
        .flat_map(|&v| f32_to_bf16(center + v).to_le_bytes())
        .collect()
}

struct Inputs {
    conv_out: metal::Buffer,
    a: metal::Buffer,
    b: metal::Buffer,
    a_log: metal::Buffer,
    dt_bias: metal::Buffer,
    state_bytes: Vec<u8>,
}

impl Inputs {
    fn new(context: &MetalContext, shape: GdnShape) -> Self {
        let hv = shape.num_v_heads as usize;
        let c = shape.qkv_dim() as usize;
        let state_len = hv * shape.value_head_dim as usize * shape.key_head_dim as usize;
        // A nonzero start state, so the register load and every `s *= g`
        // carry real values through both builds.
        let state_bytes = deterministic(0x5EED, state_len, 0.5)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        Self {
            conv_out: context.new_buffer_with_data(&half_bytes(0xD1CE, ROWS * c, 1.0)),
            a: context.new_buffer_with_data(&half_bytes(0xA000, ROWS * hv, 1.5)),
            b: context.new_buffer_with_data(&half_bytes(0xB000, ROWS * hv, 1.0)),
            a_log: context.new_buffer_with_data(&bf16_bytes(0xA100, hv, 1.25, 0.25)),
            dt_bias: context.new_buffer_with_data(&bf16_bytes(0xA200, hv, 0.5, 0.0)),
            state_bytes,
        }
    }

    fn state(&self, context: &MetalContext) -> metal::Buffer {
        context.new_buffer_with_data(&self.state_bytes)
    }
}

fn zeroed(context: &MetalContext, bytes: usize) -> metal::Buffer {
    let buffer = context.new_output_buffer(bytes as u64);
    write_buffer_bytes(&buffer, 0, &vec![0u8; bytes]);
    buffer
}

/// Everything one build produces from the same inputs, as raw bits.
#[derive(Debug, PartialEq, Eq)]
struct Outputs {
    decode_y: Vec<u16>,
    decode_state: Vec<u32>,
    eager_y: Vec<u16>,
    eager_state: Vec<u32>,
    lazy_y: Vec<u16>,
    lazy_delta: Vec<u32>,
    lazy_state: Vec<u32>,
    folded_state: Vec<u32>,
}

fn bits16(v: Vec<f16>) -> Vec<u16> {
    v.into_iter().map(f16::to_bits).collect()
}

fn bits32(v: Vec<f32>) -> Vec<u32> {
    v.into_iter().map(f32::to_bits).collect()
}

fn run_all(context: &mut MetalContext, shape: GdnShape, inputs: &Inputs) -> Outputs {
    let vd = shape.value_dim() as usize;
    let k_width = (shape.num_k_heads * shape.key_head_dim) as usize;
    let state_len = inputs.state_bytes.len() / 4;
    let n = ROWS as u32;

    let decode_state = inputs.state(context);
    let decode_y = zeroed(context, vd * 2);
    let eager_state = inputs.state(context);
    let eager_y = zeroed(context, ROWS * vd * 2);
    let lazy_state = inputs.state(context);
    let lazy_y = zeroed(context, ROWS * vd * 2);
    let k_tape = zeroed(context, ROWS * k_width * 2);
    let delta_tape = zeroed(context, ROWS * vd * 4);

    let pass = context.begin_pass();
    encode_gdn_delta_decode(
        context,
        &pass,
        shape,
        (&inputs.conv_out, 0),
        (&inputs.a, 0),
        (&inputs.b, 0),
        (&inputs.a_log, 0),
        (&inputs.dt_bias, 0),
        &decode_state,
        (&decode_y, 0),
    )
    .expect("delta decode");
    encode_gdn_delta_prefill(
        context,
        &pass,
        shape,
        (&inputs.conv_out, 0),
        (&inputs.a, 0),
        (&inputs.b, 0),
        (&inputs.a_log, 0),
        (&inputs.dt_bias, 0),
        &eager_state,
        (&eager_y, 0),
        n,
    )
    .expect("eager delta prefill");
    crate::encode_copy_rows_fp16_strided(
        context,
        &pass,
        (&inputs.conv_out, (k_width * 2) as u64),
        (&k_tape, 0),
        n,
        k_width as u32,
        shape.qkv_dim(),
        k_width as u32,
    )
    .expect("k capture");
    encode_gdn_delta_prefill_lazy(
        context,
        &pass,
        shape,
        (&inputs.conv_out, 0),
        (&inputs.a, 0),
        (&inputs.b, 0),
        (&inputs.a_log, 0),
        (&inputs.dt_bias, 0),
        &lazy_state,
        (&lazy_y, 0),
        (&delta_tape, 0),
        n,
    )
    .expect("lazy delta prefill");
    pass.commit_and_wait();
    let lazy_state_snapshot = bits32(read_f32_buffer(&lazy_state, state_len));

    let pass = context.begin_pass();
    encode_gdn_state_fold(
        context,
        &pass,
        shape,
        (&k_tape, 0),
        (&delta_tape, 0),
        (&inputs.a, 0),
        (&inputs.a_log, 0),
        (&inputs.dt_bias, 0),
        &lazy_state,
        n,
    )
    .expect("fold");
    pass.commit_and_wait();

    Outputs {
        decode_y: bits16(read_buffer_f16(&decode_y, 0, vd)),
        decode_state: bits32(read_f32_buffer(&decode_state, state_len)),
        eager_y: bits16(read_buffer_f16(&eager_y, 0, ROWS * vd)),
        eager_state: bits32(read_f32_buffer(&eager_state, state_len)),
        lazy_y: bits16(read_buffer_f16(&lazy_y, 0, ROWS * vd)),
        lazy_delta: bits32(read_f32_buffer(&delta_tape, ROWS * vd)),
        lazy_state: lazy_state_snapshot,
        folded_state: bits32(read_f32_buffer(&lazy_state, state_len)),
    }
}

fn runtime_build(context: &mut MetalContext, shape: GdnShape, inputs: &Inputs) -> Outputs {
    FORCE_RUNTIME_KEY_DIM.with(|f| f.set(true));
    let out = run_all(context, shape, inputs);
    FORCE_RUNTIME_KEY_DIM.with(|f| f.set(false));
    out
}

fn assert_bit_identical(got: &Outputs, want: &Outputs) {
    // Field by field so a failure names the kernel that drifted.
    assert_eq!(got.decode_y, want.decode_y, "decode y");
    assert_eq!(got.decode_state, want.decode_state, "decode state");
    assert_eq!(got.eager_y, want.eager_y, "eager prefill y");
    assert_eq!(got.eager_state, want.eager_state, "eager prefill state");
    assert_eq!(got.lazy_y, want.lazy_y, "lazy prefill y");
    assert_eq!(got.lazy_delta, want.lazy_delta, "lazy delta tape");
    assert_eq!(got.lazy_state, want.lazy_state, "lazy record pass state");
    assert_eq!(got.folded_state, want.folded_state, "folded state");
}

#[test]
fn baked_key_dim_selects_only_128() {
    assert_eq!(baked_key_dim(BAKED_KEY_DIM), BAKED_KEY_DIM);
    for dk in [32, 64, 96, 160, 256] {
        assert_eq!(baked_key_dim(dk), 0, "Dk {dk}");
    }
    FORCE_RUNTIME_KEY_DIM.with(|f| f.set(true));
    assert_eq!(baked_key_dim(BAKED_KEY_DIM), 0);
    FORCE_RUNTIME_KEY_DIM.with(|f| f.set(false));
}

#[test]
fn baked_dk128_is_bit_identical_to_the_runtime_build() {
    let mut context = MetalContext::new().expect("metal");
    let shape = shape(BAKED_KEY_DIM);
    assert_eq!(
        baked_key_dim(shape.key_head_dim),
        BAKED_KEY_DIM,
        "the baked arm must bake"
    );
    let inputs = Inputs::new(&context, shape);
    let want = runtime_build(&mut context, shape, &inputs);
    let got = run_all(&mut context, shape, &inputs);
    // Guards against a vacuous pass: the recurrence moved the state, and
    // the lazy record pass left it at the start.
    assert_ne!(want.eager_state, want.lazy_state, "eager state never moved");
    assert_eq!(
        want.lazy_state,
        bits32(
            inputs
                .state_bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        ),
        "lazy record pass wrote the state"
    );
    assert_eq!(want.folded_state, want.eager_state, "fold != eager");
    assert_bit_identical(&got, &want);
}

/// The parity test above is vacuous if both arms share one cache bucket:
/// the second build would silently reuse the first (crate Gotcha 1).
#[test]
fn baked_and_runtime_builds_are_distinct_pipelines() {
    let mut context = MetalContext::new().expect("metal");
    for (name, record, write_state) in [
        ("gdn_delta_step_decode", false, true),
        ("gdn_delta_step_prefill", false, true),
        ("gdn_delta_step_prefill", true, false),
        ("gdn_state_fold", false, true),
    ] {
        let runtime = delta_pipeline(&mut context, name, record, write_state, 0).expect("runtime");
        let baked =
            delta_pipeline(&mut context, name, record, write_state, BAKED_KEY_DIM).expect("baked");
        let again = delta_pipeline(&mut context, name, record, write_state, 0).expect("again");
        assert!(
            !std::ptr::eq(&*runtime, &*baked),
            "{name} ({record}, {write_state}): baked build reused the runtime pipeline"
        );
        assert!(
            std::ptr::eq(&*runtime, &*again),
            "{name}: cache miss on rebuild"
        );
    }
}

/// The fallback: at other key dims nothing is baked, so the default
/// dispatch and the forced runtime build are the same pipeline, and the
/// lazy-fold contract still holds there.
#[test]
fn other_key_dims_keep_the_runtime_build() {
    let mut context = MetalContext::new().expect("metal");
    for dk in [64, 256] {
        let shape = shape(dk);
        assert_eq!(baked_key_dim(shape.key_head_dim), 0, "Dk {dk}");
        let inputs = Inputs::new(&context, shape);
        let want = runtime_build(&mut context, shape, &inputs);
        let got = run_all(&mut context, shape, &inputs);
        assert_eq!(
            want.folded_state, want.eager_state,
            "Dk {dk}: fold != eager"
        );
        assert_bit_identical(&got, &want);
    }
}

/// Baked against runtime `Dk` at `Qwen/Qwen3.8-27B`'s GDN geometry
/// (`gdn_prefill_share_bench.rs`), every delta kernel, interleaved arm by
/// arm inside one process so thermal drift hits both. Reports rather than
/// asserts; per-round ms is the per-call time across 48 GDN layers.
#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn baked_dk128_speed_at_qwen38_shapes() {
    const REPS: usize = 64;
    const ROUNDS: usize = 5;
    const GDN_LAYERS: f64 = 48.0;
    let mut context = MetalContext::new().expect("metal");
    let shape = GdnShape {
        num_k_heads: 16,
        num_v_heads: 48,
        key_head_dim: 128,
        value_head_dim: 128,
        conv_kernel_size: 4,
    };
    let hv = shape.num_v_heads as usize;
    let vd = shape.value_dim() as usize;
    let k_width = (shape.num_k_heads * shape.key_head_dim) as usize;
    let state_len = vd * shape.key_head_dim as usize;
    let max_rows = 16;
    let c = shape.qkv_dim() as usize;
    let conv = context.new_buffer_with_data(&half_bytes(0xD1CE, max_rows * c, 1.0));
    let a = context.new_buffer_with_data(&half_bytes(0xA000, max_rows * hv, 1.5));
    let b = context.new_buffer_with_data(&half_bytes(0xB000, max_rows * hv, 1.0));
    let a_log = context.new_buffer_with_data(&bf16_bytes(0xA100, hv, 1.25, 0.25));
    let dt_bias = context.new_buffer_with_data(&bf16_bytes(0xA200, hv, 0.5, 0.0));
    let state = zeroed(&context, state_len * 4);
    let y = zeroed(&context, max_rows * vd * 2);
    let k_tape = zeroed(&context, max_rows * k_width * 2);
    let delta_tape = zeroed(&context, max_rows * vd * 4);

    #[derive(Clone, Copy, PartialEq)]
    enum Kernel {
        Decode,
        Eager,
        Lazy,
        Fold,
    }
    let time = |context: &mut MetalContext, kernel: Kernel, rows: u32, runtime: bool| -> f64 {
        FORCE_RUNTIME_KEY_DIM.with(|f| f.set(runtime));
        let pass = context.begin_pass();
        for _ in 0..REPS {
            match kernel {
                Kernel::Decode => encode_gdn_delta_decode(
                    context,
                    &pass,
                    shape,
                    (&conv, 0),
                    (&a, 0),
                    (&b, 0),
                    (&a_log, 0),
                    (&dt_bias, 0),
                    &state,
                    (&y, 0),
                ),
                Kernel::Eager => encode_gdn_delta_prefill(
                    context,
                    &pass,
                    shape,
                    (&conv, 0),
                    (&a, 0),
                    (&b, 0),
                    (&a_log, 0),
                    (&dt_bias, 0),
                    &state,
                    (&y, 0),
                    rows,
                ),
                Kernel::Lazy => encode_gdn_delta_prefill_lazy(
                    context,
                    &pass,
                    shape,
                    (&conv, 0),
                    (&a, 0),
                    (&b, 0),
                    (&a_log, 0),
                    (&dt_bias, 0),
                    &state,
                    (&y, 0),
                    (&delta_tape, 0),
                    rows,
                ),
                Kernel::Fold => encode_gdn_state_fold(
                    context,
                    &pass,
                    shape,
                    (&k_tape, 0),
                    (&delta_tape, 0),
                    (&a, 0),
                    (&a_log, 0),
                    (&dt_bias, 0),
                    &state,
                    rows,
                ),
            }
            .expect("encode");
        }
        let secs = pass.commit_and_wait_with_gpu_time();
        FORCE_RUNTIME_KEY_DIM.with(|f| f.set(false));
        secs / REPS as f64
    };

    println!("\nqwen38-27B GDN, {REPS} reps per pass, best of {ROUNDS}, interleaved");
    println!("kernel         rows   runtime us   baked us   speedup   saved ms/round (x48)");
    for (kernel, label) in [
        (Kernel::Decode, "decode"),
        (Kernel::Eager, "eager prefill"),
        (Kernel::Lazy, "lazy prefill"),
        (Kernel::Fold, "fold"),
    ] {
        let row_counts: &[u32] = if kernel == Kernel::Decode {
            &[1]
        } else {
            &[8, 16]
        };
        for &rows in row_counts {
            // Untimed warm pass per arm (cold-vs-warm GPU, Gotcha 20).
            time(&mut context, kernel, rows, true);
            time(&mut context, kernel, rows, false);
            let (mut runtime, mut baked) = (f64::INFINITY, f64::INFINITY);
            for _ in 0..ROUNDS {
                runtime = runtime.min(time(&mut context, kernel, rows, true));
                baked = baked.min(time(&mut context, kernel, rows, false));
            }
            println!(
                "{label:<13}  {rows:>4}   {:>10.2}   {:>8.2}   {:>6.2}x   {:>8.3}",
                runtime * 1e6,
                baked * 1e6,
                runtime / baked,
                (runtime - baked) * GDN_LAYERS * 1e3
            );
        }
    }
}
