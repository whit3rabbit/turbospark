#![cfg(target_os = "macos")]
//! Parity test for the port-local `moe_gguf.metal` decode pairs against
//! `turbospark_compute`'s block-quant GEMVs and gated activation, on real Metal
//! hardware, reading expert blobs through a real argument buffer exactly as
//! the runtime does (ROADMAP Phase G Stage 2).
//!
//! What differs from the vendored INT4 sibling's test, and why each case
//! below exists: a GGUF blob has NO scale or bias planes, so six of the nine
//! `MoeExpertOffsets` fields are zero and the three weight offsets are all
//! that locate anything. A kernel that kept the vendored addressing would
//! read weights as scales and produce finite garbage, and one that kept the
//! affine group of 64 would stride a 34-byte block wrongly.
//!
//! Both block types run the SAME cases through the same body. That is the
//! point rather than a convenience: Q8_0 and Q4_K have separate kernels with
//! identical contracts, and a difference between them that only one block
//! type's fixture exercises is exactly what a shared body catches.

use half::f16;
use turbospark_gpu::{MetalContext, MoeExpertOffsets, RoutedBlobsBuffer, MAX_STREAMED_EXPERTS};

/// The two GGUF block types with a routed-expert pair. Q6_K is deliberately
/// absent: no real checkpoint puts it in an expert (Qwen 3.6's Q4_K_M carries
/// exactly one Q6_K tensor and it is `output.weight`), so there is no such
/// kernel to test.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Block {
    Q8_0,
    Q4K,
}

impl Block {
    /// `(hidden, ffn)`. Q8_0 rows need a whole number of 32 elements and F is
    /// deliberately not a multiple of 64, so no affine group size can hide in
    /// it; Q4_K rows need a whole number of 256, which forces both up.
    fn dims(self) -> (usize, usize) {
        match self {
            Block::Q8_0 => (64, 96),
            Block::Q4K => (256, 512),
        }
    }

    fn quantize(self, row: &[f32]) -> Vec<u8> {
        match self {
            Block::Q8_0 => turbospark_compute::quantize_q8_0(row),
            Block::Q4K => turbospark_compute::quantize_q4_k(row),
        }
    }

    fn gemv(self, rows: &[&[u8]], x: &[f32], n: usize) -> Vec<f32> {
        match self {
            Block::Q8_0 => turbospark_compute::dequant_q8_0_gemv(rows, x, n),
            Block::Q4K => turbospark_compute::dequant_q4_k_gemv(rows, x, n),
        }
    }
}

fn deterministic_row(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(0x9E37_79B9);
    (0..n)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state = state.wrapping_add(i as u64);
            ((state % 2000) as f32 / 1000.0) - 1.0
        })
        .collect()
}

/// `rows` byte runs of `cols` elements each, in the given block type.
fn quant_rows(block: Block, rows: usize, cols: usize, seed: u64) -> Vec<Vec<u8>> {
    (0..rows)
        .map(|r| {
            let row = deterministic_row(seed.wrapping_add(r as u64 * 97 + 1), cols);
            block.quantize(&row)
        })
        .collect()
}

/// Packs gate/up/down into one blob the way the GGUF repack walk does: three
/// contiguous byte runs, nothing else. The six scale/bias offsets stay zero,
/// which is the whole point of the separate kernels.
fn build_blob(gate: &[Vec<u8>], up: &[Vec<u8>], down: &[Vec<u8>]) -> (Vec<u8>, MoeExpertOffsets) {
    let mut blob = Vec::new();
    let mut at = [0u32; 3];
    for (i, rows) in [gate, up, down].into_iter().enumerate() {
        at[i] = blob.len() as u32;
        for r in rows {
            blob.extend_from_slice(r);
        }
    }
    (
        blob,
        MoeExpertOffsets {
            gate_w: at[0],
            gate_s: 0,
            gate_b: 0,
            up_w: at[1],
            up_s: 0,
            up_b: 0,
            down_w: at[2],
            down_s: 0,
            down_b: 0,
        },
    )
}

fn to_le(v: &[f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}

fn read_f16(buffer: &metal::Buffer, n: usize) -> Vec<f32> {
    let ptr = buffer.contents() as *const u16;
    let bits = unsafe { std::slice::from_raw_parts(ptr, n) };
    bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
}

/// One expert's contribution, in FP32, rounding `acts` through FP16 exactly
/// where the kernels do.
fn expert_reference(
    block: Block,
    gate: &[Vec<u8>],
    up: &[Vec<u8>],
    down: &[Vec<u8>],
    x: &[f32],
    use_silu: bool,
) -> Vec<f32> {
    let (d_dim, f_dim) = block.dims();
    let g: Vec<&[u8]> = gate.iter().map(|r| r.as_slice()).collect();
    let u: Vec<&[u8]> = up.iter().map(|r| r.as_slice()).collect();
    let gate_out = block.gemv(&g, x, d_dim);
    let up_out = block.gemv(&u, x, d_dim);

    let activated = if use_silu {
        gate_out.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
    } else {
        turbospark_compute::gelu_tanh(&gate_out)
    };
    let acts: Vec<f32> = activated
        .iter()
        .zip(up_out.iter())
        .map(|(&a, &uv)| f16::from_f32(a * uv).to_f32())
        .collect();

    let d: Vec<&[u8]> = down.iter().map(|r| r.as_slice()).collect();
    block.gemv(&d, &acts, f_dim)
}

/// One expert's gate, up and down row sets, each a byte run per row.
type Expert = (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<Vec<u8>>);

/// The shared body: `top_k` experts of one block type through both kernels,
/// compared against the CPU reference, under the caller's activation.
fn run_case(block: Block, use_silu: bool, top_k: usize) {
    let mut context = MetalContext::new().expect("Metal device");
    let (d_dim, f_dim) = block.dims();

    let experts: Vec<Expert> = (0..top_k)
        .map(|e| {
            let seed = 5000 + 100 * e as u64;
            (
                quant_rows(block, f_dim, d_dim, seed + 1),
                quant_rows(block, f_dim, d_dim, seed + 2),
                quant_rows(block, d_dim, f_dim, seed + 3),
            )
        })
        .collect();

    let x32: Vec<f32> = (0..d_dim).map(|i| ((i as f32) * 0.21).sin()).collect();
    let x16: Vec<f16> = x32.iter().map(|&v| f16::from_f32(v)).collect();
    let x32_rounded: Vec<f32> = x16.iter().map(|v| v.to_f32()).collect();
    let residual32: Vec<f32> = (0..d_dim)
        .map(|i| ((i as f32) * 0.11).cos() * 0.5)
        .collect();
    let residual16: Vec<f16> = residual32.iter().map(|&v| f16::from_f32(v)).collect();
    let weights: Vec<f32> = (0..top_k).map(|e| 0.6 - 0.15 * e as f32).collect();

    let mut expected: Vec<f32> = residual16.iter().map(|v| v.to_f32()).collect();
    for (e, (gate, up, down)) in experts.iter().enumerate() {
        let out = expert_reference(block, gate, up, down, &x32_rounded, use_silu);
        for (dst, o) in expected.iter_mut().zip(out.iter()) {
            *dst += weights[e] * o;
        }
    }

    let mut blobs = Vec::new();
    let mut offsets = None;
    for (gate, up, down) in &experts {
        let (blob, off) = build_blob(gate, up, down);
        offsets.get_or_insert(off);
        blobs.push(context.new_buffer_with_data(&blob));
    }
    let offsets = offsets.unwrap();

    let x_buf = context.new_buffer_with_data(&to_le(&x16));
    let residual_buf = context.new_buffer_with_data(&to_le(&residual16));
    // Sized and zeroed for ALL EIGHT slots, not `top_k`: phase 2 reduces
    // every slot unconditionally, and a garbage row times a zero weight is
    // still NaN (AGENTS.md Gotcha 8).
    let acts_buf = context.new_buffer_with_data(&vec![0u8; MAX_STREAMED_EXPERTS * f_dim * 2]);
    let y_buf = context.new_output_buffer((d_dim * 2) as u64);
    let mut routing = vec![f16::from_f32(0.0); MAX_STREAMED_EXPERTS];
    for (slot, &w) in weights.iter().enumerate() {
        routing[slot] = f16::from_f32(w);
    }
    let routing_buf = context.new_buffer_with_data(&to_le(&routing));

    let routed = RoutedBlobsBuffer::new(&mut context, use_silu).expect("arg buffer");
    let blob_refs: Vec<(&metal::Buffer, u64)> = blobs.iter().map(|b| (b, 0u64)).collect();
    routed
        .bind(&mut context, use_silu, &blob_refs)
        .expect("bind blobs");

    let pass = context.begin_pass();
    for blob in &blobs {
        pass.use_read_buffer(blob);
    }
    // Branching around both calls rather than through a pair of function
    // pointers: the two signatures differ only in kernel name, and spelling
    // them out keeps the argument order visible at each call site.
    match block {
        Block::Q8_0 => {
            turbospark_gpu::encode_moe_phase1_q8_0(
                &mut context,
                &pass,
                &routed,
                &offsets,
                (&x_buf, 0),
                (&acts_buf, 0),
                d_dim as u32,
                f_dim as u32,
                top_k as u32,
                use_silu,
            )
            .expect("phase1");
            turbospark_gpu::encode_moe_phase2_q8_0(
                &mut context,
                &pass,
                &routed,
                &offsets,
                (&acts_buf, 0),
                (&routing_buf, 0),
                (&residual_buf, 0),
                (&y_buf, 0),
                d_dim as u32,
                f_dim as u32,
                use_silu,
            )
            .expect("phase2");
        }
        Block::Q4K => {
            turbospark_gpu::encode_moe_phase1_q4_k(
                &mut context,
                &pass,
                &routed,
                &offsets,
                (&x_buf, 0),
                (&acts_buf, 0),
                d_dim as u32,
                f_dim as u32,
                top_k as u32,
                use_silu,
            )
            .expect("phase1");
            turbospark_gpu::encode_moe_phase2_q4_k(
                &mut context,
                &pass,
                &routed,
                &offsets,
                (&acts_buf, 0),
                (&routing_buf, 0),
                (&residual_buf, 0),
                (&y_buf, 0),
                d_dim as u32,
                f_dim as u32,
                use_silu,
            )
            .expect("phase2");
        }
    }
    pass.commit_and_wait();

    let got = read_f16(&y_buf, d_dim);
    for d in 0..d_dim {
        let want = expected[d];
        let diff = (got[d] - want).abs();
        let tol = 5e-2_f32.max(want.abs() * 3e-2);
        assert!(
            diff <= tol,
            "{block:?} silu={use_silu} top_k={top_k} d={d}: got {} want {want} (diff {diff})",
            got[d]
        );
    }
}

#[test]
fn the_decode_pair_matches_the_cpu_reference() {
    run_case(Block::Q8_0, false, 2);
}

/// The activation is a function constant shared with the vendored kernels,
/// so it is exercised rather than assumed: SiLU and GELU differ enough that
/// a kernel wired to the wrong one fails this while passing the case above.
#[test]
fn the_silu_activation_constant_reaches_the_gguf_kernels() {
    run_case(Block::Q8_0, true, 2);
}

/// Eight slots is the reduce's fixed shape. Filling every one of them checks
/// that phase 1 writes the whole `acts` range the reduce reads, which a
/// two-slot case cannot: there, six slots are covered by the zero-fill.
#[test]
fn all_eight_slots_participate() {
    run_case(Block::Q8_0, false, MAX_STREAMED_EXPERTS);
}

#[test]
fn the_q4_k_decode_pair_matches_the_cpu_reference() {
    run_case(Block::Q4K, false, 2);
}

/// Qwen 3.6, the one family whose experts are Q4_K, uses SiLU. So this arm is
/// the production combination rather than a symmetry with the Q8_0 case.
#[test]
fn the_silu_activation_constant_reaches_the_q4_k_kernels() {
    run_case(Block::Q4K, true, 2);
}

#[test]
fn all_eight_q4_k_slots_participate() {
    run_case(Block::Q4K, false, MAX_STREAMED_EXPERTS);
}
