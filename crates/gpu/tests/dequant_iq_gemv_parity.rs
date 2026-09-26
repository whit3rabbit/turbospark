//! Runs the three IQ GEMV kernels on real Metal hardware against the CPU
//! references in `turbospark_compute::quant_gguf_iq` (ROADMAP Phase S). The
//! kernel rule: no quant kernel is trusted before this file exists and passes.
//!
//! NO QUANTIZER FEEDS THIS ONE, unlike every quant parity test before it. The
//! IQ types have no `quantize_*` sibling in `crates/compute` (an encoder means
//! a codebook nearest-neighbour search nothing in this port calls), so the
//! blocks here are built by picking VALID CODE POINTS directly: random grid
//! indices, sign indices, scale nibbles and quant nibbles, assembled into the
//! byte layout. That is not a workaround. It covers the code space uniformly,
//! where an encoder only ever emits the subset it happens to choose, and it
//! means a kernel and a fixture cannot share an encoder's mistake.
//!
//! Both sides are then handed the SAME bytes, so there is no quantization
//! error to cancel and what is left is FP16 rounding and reduction order.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{dequant_iq_gemv, encode_embed_lookup_iq4_xs, IqBlockType, MetalContext};

/// One LCG, so every fixture below is reproducible and independent of the
/// random-number policy anywhere else.
struct Lcg(u32);

impl Lcg {
    fn new(seed: u32) -> Self {
        Self(seed.wrapping_mul(2_654_435_761).wrapping_add(1))
    }
    fn byte(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 16) as u8
    }
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((self.0 >> 8) as f32 / (1u32 << 23) as f32) - 1.0
    }
}

/// FP16 0.0625, small enough that a 256-element dot product of full-range
/// codebook values stays well inside FP16 (the dynamic-range trap recorded in
/// CLAUDE.local: a fixture spanning the whole byte range once overflowed FP16
/// before the head and read as a block-layout bug).
const F16_SMALL: u16 = 0x2C00;

fn iq4_nl_row(n: usize, seed: u32) -> Vec<u8> {
    let mut lcg = Lcg::new(seed);
    let mut out = Vec::with_capacity(n / 32 * 18);
    for _ in 0..n / 32 {
        out.extend_from_slice(&F16_SMALL.to_le_bytes());
        for _ in 0..16 {
            out.push(lcg.byte());
        }
    }
    out
}

fn iq4_xs_row(n: usize, seed: u32) -> Vec<u8> {
    let mut lcg = Lcg::new(seed);
    let mut out = Vec::with_capacity(n / 256 * 136);
    for _ in 0..n / 256 {
        out.extend_from_slice(&F16_SMALL.to_le_bytes());
        // scales_h: two bits per sub-block. Left fully random so the +/- 32
        // bias is exercised on both sides of zero.
        out.extend_from_slice(&u16::from(lcg.byte()).to_le_bytes());
        for _ in 0..4 {
            out.push(lcg.byte());
        }
        for _ in 0..128 {
            out.push(lcg.byte());
        }
    }
    out
}

fn iq3_xxs_row(n: usize, seed: u32) -> Vec<u8> {
    let mut lcg = Lcg::new(seed);
    let mut out = Vec::with_capacity(n / 256 * 98);
    for _ in 0..n / 256 {
        out.extend_from_slice(&F16_SMALL.to_le_bytes());
        // 64 grid indices: every byte is a valid index into a 256-entry table,
        // so this needs no masking and reaches the whole codebook.
        for _ in 0..64 {
            out.push(lcg.byte());
        }
        // Eight sign-and-scale words, fully random: the sign indices are
        // 7-bit fields the decoder masks, and the top nibble is the scale.
        for _ in 0..32 {
            out.push(lcg.byte());
        }
    }
    out
}

/// Valid arbitrary blocks for the six new 256-element layouts. Every field
/// is an index, sign bit, or scale bit, so a random byte is valid. IQ1_M is
/// the exception: its scale words also carry `d`, which is pinned to a small
/// finite f16 rather than accepting a random NaN.
fn lowbit_iq_row(n: usize, bytes: usize, seed: u32, iq1_m: bool) -> Vec<u8> {
    let mut lcg = Lcg::new(seed);
    let mut out = Vec::with_capacity(n / 256 * bytes);
    for _ in 0..n / 256 {
        let mut block: Vec<u8> = (0..bytes).map(|_| lcg.byte()).collect();
        if iq1_m {
            // Reassembles to f16 0x2c00 through IQ1_M's four scale words.
            block[48..56].copy_from_slice(&[0, 0, 0, 0, 0, 0xc0, 0, 0x20]);
        } else {
            block[..2].copy_from_slice(&F16_SMALL.to_le_bytes());
        }
        out.extend_from_slice(&block);
    }
    out
}

fn x_vector(n: usize, seed: u32) -> (Vec<f32>, Vec<f16>) {
    let mut lcg = Lcg::new(seed);
    let f32s: Vec<f32> = (0..n).map(|_| lcg.unit()).collect();
    let f16s = f32s.iter().map(|&v| f16::from_f32(v)).collect();
    (f32s, f16s)
}

fn assert_matches(label: &str, gpu: &[f16], cpu: &[f32], n: usize) {
    assert_eq!(gpu.len(), cpu.len());
    let gpu_f32: Vec<f32> = gpu.iter().map(|v| v.to_f32()).collect();
    let err = turbospark_compute::max_abs_diff(&gpu_f32, cpu);
    let scale = cpu.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1.0);
    let bound = scale * 1e-2 + (n as f32) * 1e-4;
    assert!(
        err < bound,
        "{label}: err = {err}, bound = {bound}, gpu[0] = {}, cpu[0] = {}",
        gpu_f32[0],
        cpu[0]
    );
}

/// `m = 5` on purpose in all three: not a multiple of the eight rows per
/// threadgroup, so the kernel's early-return guard is exercised rather than
/// assumed.
const M: usize = 5;

#[test]
fn iq4_nl_matches_the_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let n = 4 * 32;
    let rows: Vec<Vec<u8>> = (0..M).map(|r| iq4_nl_row(n, 11 + r as u32)).collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
    let (x_f32, x_f16) = x_vector(n, 71);

    let cpu = turbospark_compute::dequant_iq4_nl_gemv(&refs, &x_f32, n);
    let gpu = dequant_iq_gemv(&mut context, IqBlockType::Iq4Nl, &refs, &x_f16, n).unwrap();
    assert_matches("iq4_nl", &gpu, &cpu, n);
}

#[test]
fn iq4_xs_matches_the_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let n = 2 * 256;
    let rows: Vec<Vec<u8>> = (0..M).map(|r| iq4_xs_row(n, 23 + r as u32)).collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
    let (x_f32, x_f16) = x_vector(n, 73);

    let cpu = turbospark_compute::dequant_iq4_xs_gemv(&refs, &x_f32, n);
    let gpu = dequant_iq_gemv(&mut context, IqBlockType::Iq4Xs, &refs, &x_f16, n).unwrap();
    assert_matches("iq4_xs", &gpu, &cpu, n);
}

#[test]
fn iq4_xs_embedding_lookup_matches_the_cpu_reference_for_selected_row() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let (vocab, d, token) = (7usize, 2 * 256, 5u32);
    let rows: Vec<Vec<u8>> = (0..vocab)
        .map(|row| iq4_xs_row(d, 127 + row as u32))
        .collect();
    let table: Vec<u8> = rows.iter().flatten().copied().collect();
    let table_buffer = context.new_buffer_with_data(&table);
    let out = context.new_output_buffer((d * std::mem::size_of::<u16>()) as u64);
    let out_scale = 2.5f32;
    let pass = context.begin_pass();
    encode_embed_lookup_iq4_xs(
        &mut context,
        &pass,
        (&table_buffer, 0),
        (&out, 0),
        token,
        d as u32,
        out_scale,
    )
    .expect("GPU dispatch succeeds");
    pass.commit_and_wait();

    let row_bytes = d / 256 * 136;
    let row = &table[token as usize * row_bytes..(token as usize + 1) * row_bytes];
    let want: Vec<f32> = turbospark_compute::quant_gguf_iq::dequantize_iq4_xs(row, d)
        .into_iter()
        .map(|value| value * out_scale)
        .collect();
    let got: Vec<f16> = {
        let ptr = out.contents() as *const u16;
        let bits = unsafe { std::slice::from_raw_parts(ptr, d) };
        bits.iter().map(|&value| f16::from_bits(value)).collect()
    };
    assert_matches("IQ4_XS embedding row 5, scale 2.5", &got, &want, d);
}

#[test]
fn iq3_xxs_matches_the_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let n = 2 * 256;
    let rows: Vec<Vec<u8>> = (0..M).map(|r| iq3_xxs_row(n, 37 + r as u32)).collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
    let (x_f32, x_f16) = x_vector(n, 79);

    let cpu = turbospark_compute::dequant_iq3_xxs_gemv(&refs, &x_f32, n);
    let gpu = dequant_iq_gemv(&mut context, IqBlockType::Iq3Xxs, &refs, &x_f16, n).unwrap();
    assert_matches("iq3_xxs", &gpu, &cpu, n);
}

/// The scale-nibble sweep, which random blocks reach only by luck at the ends.
///
/// IQ3_XXS's `db = d * (0.5 + nibble) * 0.5` runs from 0.25d to 7.75d, a 31x
/// span, and the two ends are where a kernel that dropped the `0.5 +` or the
/// trailing `* 0.5` still looks right in the middle. One row per nibble, same
/// grid indices throughout, so the only thing varying is the scale.
#[test]
fn iq3_xxs_agrees_at_every_scale_nibble() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let n = 256;
    let rows: Vec<Vec<u8>> = (0..16u32)
        .map(|nibble| {
            let mut row = iq3_xxs_row(n, 5);
            for ib in 0..8 {
                let at = 2 + 64 + 4 * ib + 3;
                row[at] = (row[at] & 0x0F) | ((nibble as u8) << 4);
            }
            row
        })
        .collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
    let (x_f32, x_f16) = x_vector(n, 83);

    let cpu = turbospark_compute::dequant_iq3_xxs_gemv(&refs, &x_f32, n);
    let gpu = dequant_iq_gemv(&mut context, IqBlockType::Iq3Xxs, &refs, &x_f16, n).unwrap();
    assert_matches("iq3_xxs scale sweep", &gpu, &cpu, n);
    // The sweep has to actually sweep, or it proves nothing: nibble 15's row
    // must be far from nibble 0's.
    assert!(
        cpu[15].abs() > 4.0 * cpu[0].abs(),
        "the scale sweep is flat: {} vs {}",
        cpu[0],
        cpu[15]
    );
}

#[test]
fn dense_gsq_rco_iq_layouts_match_the_cpu_references() {
    type CpuGemv = fn(&[&[u8]], &[f32], usize) -> Vec<f32>;
    type Case = (IqBlockType, usize, bool, CpuGemv);

    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let n = 2 * 256;
    let cases: [Case; 6] = [
        (
            IqBlockType::Iq2Xxs,
            66,
            false,
            turbospark_compute::dequant_iq2_xxs_gemv,
        ),
        (
            IqBlockType::Iq2Xs,
            74,
            false,
            turbospark_compute::dequant_iq2_xs_gemv,
        ),
        (
            IqBlockType::Iq1S,
            50,
            false,
            turbospark_compute::dequant_iq1_s_gemv,
        ),
        (
            IqBlockType::Iq3S,
            110,
            false,
            turbospark_compute::dequant_iq3_s_gemv,
        ),
        (
            IqBlockType::Iq2S,
            82,
            false,
            turbospark_compute::dequant_iq2_s_gemv,
        ),
        (
            IqBlockType::Iq1M,
            56,
            true,
            turbospark_compute::dequant_iq1_m_gemv,
        ),
    ];
    let (x_f32, x_f16) = x_vector(n, 101);
    for (case, bytes, iq1_m, cpu_fn) in cases {
        let rows: Vec<Vec<u8>> = (0..M)
            .map(|r| lowbit_iq_row(n, bytes, 109 + r as u32, iq1_m))
            .collect();
        let refs: Vec<&[u8]> = rows.iter().map(Vec::as_slice).collect();
        let cpu = cpu_fn(&refs, &x_f32, n);
        let gpu = dequant_iq_gemv(&mut context, case, &refs, &x_f16, n).unwrap();
        assert_matches(&format!("{case:?}"), &gpu, &cpu, n);
    }
}
