//! Bench prototype for kapitsa's shift-free nibble masking
//! (docs/KAPITSA_KERNELS.md item 6), following the `dequant_int4_mma.metal`
//! dead-end protocol: a prototype shader kept OUT of the production library,
//! an `#[ignore]`d release bench whose RATIO is the deliverable, and a
//! bit-identity test that must pass before any ratio means anything.
//!
//! The idea: `dequant_int4_gemv_simd_body` extracts the high nibble of each
//! weight byte with a shift (`b >> 4`). Masking (`b & 0xF0`) returns 16x the
//! nibble, which is exact in fp32, and the factor cancels against x
//! prescaled by 1/16 -- also exact, a power of two -- so
//! `fma(hi_masked, x/16, dot)` is bit-identical to `fma(hi, x, dot)`. The
//! prescaled copy is one elementwise pass over x, which in production would
//! fold into the norm that produced x (x is shared by every row); the timed
//! arm here pays a separate prescale dispatch per GEMV to stay conservative.
//!
//! ```sh
//! cargo test -p turbospark-gpu --test gemv_mask_dequant_bench --release -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use metal::{FunctionConstantValues, MTLDataType};
use turbospark_gpu::{
    autorelease_pool, read_buffer_f16, write_buffer_bytes, Int4ResidentMatrix, MetalBuffer,
    MetalContext,
};

static MASK_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint kGroupSize = 64;

// xs[i] = (i odd) ? x[i] / 16 : x[i]. Division by 16 is exact in fp32.
kernel void prescale_x_f16(
    device const half* x  [[buffer(0)]],
    device half*       xs [[buffer(1)]],
    constant uint&     n  [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    float v = float(x[tid]);
    xs[tid] = half((tid & 1u) ? v * 0.0625f : v);
}

// `dequant_int4_gemv_simd_body` with the high nibbles masked instead of
// shifted: `float(b & 0xF0) * (x/16)` == `float(b >> 4) * x` exactly, and
// the low nibbles, the affine factoring (acc = fma(s, dot, acc); acc =
// fma(b, sum, acc) per group), the lane split, and the accumulation order
// are byte-for-byte the scalar kernel's. sum is computed from the ORIGINAL
// x, so the bias term is untouched.
kernel void dequant_int4_gemv_mask_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    x      [[buffer(3)]],
    device const half*    xs     [[buffer(4)]],
    device half*          y      [[buffer(5)]],
    constant uint&        M      [[buffer(6)]],
    constant uint&        N      [[buffer(7)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= M) return;
    const uint n_groups  = N / kGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const bfloat*  s_row = scales + uint(row) * n_groups;
    device const bfloat*  b_row = biases + uint(row) * n_groups;

    float acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g  = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa  = *((device const half4*)(x  + elem));
        const half4 xb  = *((device const half4*)(x  + elem + 4u));
        const half4 xsa = *((device const half4*)(xs + elem));
        const half4 xsb = *((device const half4*)(xs + elem + 4u));
        const uint b0 =  w4        & 0xFFu;
        const uint b1 = (w4 >> 8)  & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
        float dot = 0.0f;
        dot = fma(float(b0 & 0x0Fu), e0, dot);
        dot = fma(float(b0 & 0xF0u), float(xsa.y), dot);
        dot = fma(float(b1 & 0x0Fu), e2, dot);
        dot = fma(float(b1 & 0xF0u), float(xsa.w), dot);
        dot = fma(float(b2 & 0x0Fu), e4, dot);
        dot = fma(float(b2 & 0xF0u), float(xsb.y), dot);
        dot = fma(float(b3 & 0x0Fu), e6, dot);
        dot = fma(float(b3 & 0xF0u), float(xsb.w), dot);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint8_t byte = W_row[g * (kGroupSize / 2) + lane];
        const uint ebase = g * kGroupSize + lane * 2u;
        const float x0 = float(x[ebase]);
        const float x1 = float(x[ebase + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte & 0xF0u)), float(xs[ebase + 1u]), dot);
        const float sum = x0 + x1;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = half(acc);
    }
}
"#;

fn mask_constants() -> FunctionConstantValues {
    let values = FunctionConstantValues::new();
    let zero: u32 = 0;
    let use_fc = false;
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 20);
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 21);
    values.set_constant_value_at_index((&use_fc as *const bool).cast(), MTLDataType::Bool, 22);
    values
}

fn half_bytes(v: &[half::f16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
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

/// One quantized INT4 matrix with real packed nibbles (2-byte-aligned pad),
/// laid out weights|scales|biases in one buffer, plus the x vectors.
struct Fixture {
    weights: MetalBuffer,
    weights_offset: u64,
    scales_offset: u64,
    biases_offset: u64,
    x: MetalBuffer,
    rows: usize,
    cols: usize,
}

impl Fixture {
    fn new(context: &MetalContext, rows: usize, cols: usize, seed: u64) -> Self {
        let mut weights = Vec::new();
        let mut scales = Vec::new();
        let mut biases = Vec::new();
        for r in 0..rows {
            let q = turbospark_compute::quantize_int4_affine(&deterministic(
                seed + r as u64 * 97 + 1,
                cols,
                0.5,
            ));
            weights.extend_from_slice(&q.packed);
            scales.extend_from_slice(&q.scales);
            biases.extend_from_slice(&q.biases);
        }
        let weights_offset = 4u64; // 2-byte aligned, not 4: the resident shape
        let scales_offset = weights_offset + weights.len() as u64;
        let biases_offset = scales_offset + (scales.len() * 2) as u64;
        let mut bytes = vec![0u8; biases_offset as usize + biases.len() * 2];
        bytes[weights_offset as usize..weights_offset as usize + weights.len()]
            .copy_from_slice(&weights);
        bytes[scales_offset as usize..scales_offset as usize + scales.len() * 2].copy_from_slice(
            &scales
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        bytes[biases_offset as usize..biases_offset as usize + biases.len() * 2].copy_from_slice(
            &biases
                .iter()
                .flat_map(|b| b.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        let x_halfs: Vec<half::f16> = deterministic(seed + 99, cols, 1.0)
            .iter()
            .map(|&v| half::f16::from_f32(v))
            .collect();
        Self {
            weights: context.new_buffer_with_data(&bytes),
            weights_offset,
            scales_offset,
            biases_offset,
            x: context.new_buffer_with_data(&half_bytes(&x_halfs)),
            rows,
            cols,
        }
    }

    fn matrix<'a>(&'a self) -> Int4ResidentMatrix<'a> {
        Int4ResidentMatrix {
            buffer: &self.weights,
            weights_offset: self.weights_offset,
            scales_offset: self.scales_offset,
            biases_offset: self.biases_offset,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

fn prescale(
    context: &mut MetalContext,
    pass: &turbospark_gpu::PassEncoder,
    x: &MetalBuffer,
    xs: &MetalBuffer,
    n: usize,
) {
    let pipeline = context
        .pipeline(
            MASK_SOURCE,
            "prescale_x_f16",
            &FunctionConstantValues::new(),
            b"",
        )
        .unwrap();
    let n32 = n as u32;
    let bytes = n32.to_le_bytes();
    let grid = (n as u64).div_ceil(256);
    pass.encode_threadgroups(
        &pipeline,
        &[(x, 0, 0), (xs, 1, 0)],
        &[(&bytes[..], 2)],
        grid,
        256,
    );
}

fn masked_gemv(
    context: &mut MetalContext,
    pass: &turbospark_gpu::PassEncoder,
    fixture: &Fixture,
    xs: &MetalBuffer,
    y: &MetalBuffer,
) {
    prescale(context, pass, &fixture.x, xs, fixture.cols);
    let pipeline = context
        .pipeline(
            MASK_SOURCE,
            "dequant_int4_gemv_mask_simd",
            &mask_constants(),
            b"",
        )
        .unwrap();
    let (m32, n32) = (fixture.rows as u32, fixture.cols as u32);
    let mbytes = m32.to_le_bytes();
    let nbytes = n32.to_le_bytes();
    let grid = (fixture.rows as u64).div_ceil(8);
    pass.encode_threadgroups(
        &pipeline,
        &[
            (&fixture.weights, 0, fixture.weights_offset),
            (&fixture.weights, 1, fixture.scales_offset),
            (&fixture.weights, 2, fixture.biases_offset),
            (&fixture.x, 3, 0),
            (xs, 4, 0),
            (y, 5, 0),
        ],
        &[(&mbytes[..], 6), (&nbytes[..], 7)],
        grid,
        256,
    );
}

/// The masking rewrite must be invisible: identical output bits at every
/// fixture shape, remainder groups included.
#[test]
fn masked_gemv_is_bit_identical_to_the_scalar_kernel() {
    let mut context = MetalContext::new().expect("Metal device");
    for &(rows, cols) in &[(256usize, 2048usize), (37, 1792), (8, 4096)] {
        let fixture = Fixture::new(&context, rows, cols, 0xE0_000 + rows as u64);
        let xs = context.new_output_buffer((cols * 2) as u64);
        write_buffer_bytes(&xs, 0, &vec![0u8; cols * 2]);

        let reference = context.new_output_buffer((rows * 2) as u64);
        let pass = context.begin_pass();
        turbospark_gpu::encode_dequant_int4_gemv_resident(
            &mut context,
            &pass,
            &fixture.matrix(),
            (&fixture.x, 0),
            (&reference, 0),
        )
        .unwrap();
        pass.commit_and_wait();

        let masked = context.new_output_buffer((rows * 2) as u64);
        let pass = context.begin_pass();
        masked_gemv(&mut context, &pass, &fixture, &xs, &masked);
        pass.commit_and_wait();

        let want = read_buffer_f16(&reference, 0, rows);
        let got = read_buffer_f16(&masked, 0, rows);
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert!(
                g.to_bits() == w.to_bits(),
                "{rows}x{cols}: masked output diverges at {i} (got {g:?}, want {w:?})"
            );
        }
    }
}

/// The ratio is the deliverable. Interleaved arms, one process, discarded
/// warmup, one timed command buffer per arm per round -- the
/// `gemv_bandwidth_bench` discipline.
#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn masked_gemv_ratio_against_the_scalar_kernel() {
    use turbospark_gpu::encode_dequant_int4_gemv_resident;

    const ROUNDS: usize = 5;
    const TARGET_BYTES: u64 = 512 * 1024 * 1024;
    let shapes: [(&str, usize, usize); 4] = [
        ("expert  512x2048", 512, 2048),
        ("o_proj 2048x2048", 2048, 2048),
        ("q_proj 4096x2048", 4096, 2048),
        ("ffn    512x17408", 512, 17408),
    ];

    let mut context = MetalContext::new().expect("Metal device");
    let fixtures: Vec<Fixture> = shapes
        .iter()
        .map(|(_, r, c)| Fixture::new(&context, *r, *c, 0xE1_000 + *r as u64))
        .collect();
    let xses: Vec<MetalBuffer> = shapes
        .iter()
        .map(|(_, _, c)| context.new_output_buffer((*c * 2) as u64))
        .collect();
    let ys: Vec<MetalBuffer> = shapes
        .iter()
        .map(|(_, r, _)| context.new_output_buffer((*r * 2) as u64))
        .collect();

    let weight_bytes = |r: usize, c: usize| (r * c / 2 + r * c / 64 * 4) as u64;

    let scalar_arm = |context: &mut MetalContext, i: usize, repeats: usize| -> f64 {
        autorelease_pool(|| {
            let pass = context.begin_pass();
            for _ in 0..repeats {
                encode_dequant_int4_gemv_resident(
                    context,
                    &pass,
                    &fixtures[i].matrix(),
                    (&fixtures[i].x, 0),
                    (&ys[i], 0),
                )
                .unwrap();
            }
            pass.commit_and_wait_with_gpu_time()
        })
    };
    let masked_arm = |context: &mut MetalContext, i: usize, repeats: usize| -> f64 {
        autorelease_pool(|| {
            let pass = context.begin_pass();
            for _ in 0..repeats {
                masked_gemv(context, &pass, &fixtures[i], &xses[i], &ys[i]);
            }
            pass.commit_and_wait_with_gpu_time()
        })
    };

    // Warmup + fault-in.
    for (i, (_, r, c)) in shapes.iter().enumerate() {
        let repeats = (TARGET_BYTES / weight_bytes(*r, *c)).max(1) as usize;
        scalar_arm(&mut context, i, repeats);
        masked_arm(&mut context, i, repeats);
    }

    println!("\nshape               scalar_s  masked_s  masked/scalar (5 interleaved rounds)");
    let mut ratios = Vec::new();
    for _ in 0..ROUNDS {
        for (i, (label, r, c)) in shapes.iter().enumerate() {
            let repeats = (TARGET_BYTES / weight_bytes(*r, *c)).max(1) as usize;
            let a = scalar_arm(&mut context, i, repeats);
            let b = masked_arm(&mut context, i, repeats);
            ratios.push(b / a);
            println!("{label}   {a:8.5}  {b:8.5}   {:.4}", b / a);
        }
    }
    let mean: f64 = ratios.iter().sum::<f64>() / ratios.len() as f64;
    println!(
        "\nmean masked/scalar = {mean:.4} over {} rounds. The arm pays one\n\
         prescale dispatch per GEMV; production would fold the prescale into\n\
         the norm that produced x, so treat this as the LOWER bound and the\n\
         fold as free upside. Ratios only; absolutes move with clocks.\n",
        ratios.len()
    );
}
