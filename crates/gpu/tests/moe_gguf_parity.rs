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

/// One block layout. Q6_K is deliberately absent: no real checkpoint puts it
/// in an expert (Qwen 3.6's Q4_K_M carries exactly one Q6_K tensor and it is
/// `output.weight`), so there is no such kernel to test.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Q8_0,
    Q4K,
    Iq3Xxs,
    Iq4Xs,
    Iq4Nl,
}

impl Kind {
    /// Elements per block, which is the constraint on a row's length.
    fn block_elems(self) -> usize {
        match self {
            Kind::Q8_0 | Kind::Iq4Nl => 32,
            Kind::Q4K | Kind::Iq3Xxs | Kind::Iq4Xs => 256,
        }
    }

    /// One row's bytes.
    ///
    /// The two families get here differently, and the difference is not a
    /// shortcut. Q8_0 and Q4_K go through a real quantizer, so the row means
    /// something. The IQ types have no quantizer in this port at all (an
    /// encoder would be a codebook nearest-neighbour search nothing calls), so
    /// their rows are assembled from random VALID CODE POINTS. Both sides of
    /// the comparison are then handed the same bytes, which is what the test
    /// needs; whether those bytes came from fitting a weight vector is
    /// irrelevant to whether a kernel decodes them like the reference does.
    fn row(self, cols: usize, seed: u64) -> Vec<u8> {
        assert_eq!(cols % self.block_elems(), 0);
        match self {
            Kind::Q8_0 => turbospark_compute::quantize_q8_0(&deterministic_row(seed, cols)),
            Kind::Q4K => turbospark_compute::quantize_q4_k(&deterministic_row(seed, cols)),
            _ => self.synthetic_row(cols, seed),
        }
    }

    /// Random valid IQ bytes.
    ///
    /// `d` IS PER TYPE AND IT IS NOT COSMETIC. This is the dynamic-range trap
    /// recorded in CLAUDE.local, arriving through the MoE chain rather than
    /// through a single GEMV: with a shared scale the three types reach very
    /// different maxima (IQ3_XXS's grid tops out at 62 under a scale nibble
    /// worth 7.75, IQ4_XS's table at 127 under a sub-scale of 32, IQ4_NL's at
    /// 127 flat), and the chain squares them -- gate times up, then a second
    /// GEMV. The first version of this fixture used one scale for all three
    /// and produced `inf` on BOTH sides, which reads as a kernel bug and is
    /// a fixture bug. Each `d` below is a power of two chosen to put the
    /// largest representable weight just under 1.0, matching what the Q8_0 and
    /// Q4_K arms get from quantizing values in [-1, 1].
    fn synthetic_row(self, cols: usize, seed: u64) -> Vec<u8> {
        let d: u16 = match self {
            Kind::Iq3Xxs => 0x1800, // 2^-9;  max |w| = 7.75 * 62  * d = 0.94
            Kind::Iq4Xs => 0x0C00,  // 2^-12; max |w| = 32   * 127 * d = 0.99
            Kind::Iq4Nl => 0x2000,  // 2^-7;  max |w| =        127 * d = 0.99
            other => unreachable!("{other:?} has a quantizer"),
        };
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let mut byte = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 33) as u8
        };
        let (payload, per_block) = match self {
            // f16 d, then 16 nibble bytes.
            Kind::Iq4Nl => (16usize, 32usize),
            // f16 d, u16 scales_h, 4 scales_l, 128 nibble bytes.
            Kind::Iq4Xs => (134, 256),
            // f16 d, 64 grid indices, 32 bytes of sign-and-scale words.
            Kind::Iq3Xxs => (96, 256),
            other => unreachable!("{other:?} has a quantizer"),
        };
        let mut out = Vec::new();
        for _ in 0..cols / per_block {
            out.extend_from_slice(&d.to_le_bytes());
            for _ in 0..payload {
                out.push(byte());
            }
        }
        out
    }

    fn gemv(self, rows: &[&[u8]], x: &[f32], n: usize) -> Vec<f32> {
        match self {
            Kind::Q8_0 => turbospark_compute::dequant_q8_0_gemv(rows, x, n),
            Kind::Q4K => turbospark_compute::dequant_q4_k_gemv(rows, x, n),
            Kind::Iq3Xxs => turbospark_compute::dequant_iq3_xxs_gemv(rows, x, n),
            Kind::Iq4Xs => turbospark_compute::dequant_iq4_xs_gemv(rows, x, n),
            Kind::Iq4Nl => turbospark_compute::dequant_iq4_nl_gemv(rows, x, n),
        }
    }
}

/// One routed-expert case: which layout each PHASE reads.
///
/// The two phases are separate because a real mixed checkpoint makes them
/// separate. Q8_0 and Q4_K installs are uniform, and every install before
/// ROADMAP Phase S was, but the Phase S candidate puts IQ3_XXS in
/// `ffn_gate_up_exps` and IQ4_NL in `ffn_down_exps` of the SAME expert, and a
/// different pair again on layer 29. A harness that could only express one
/// layout per expert could not test the thing that actually ships.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Block {
    /// What phase 1 reads (gate and up).
    gate_up: Kind,
    /// What phase 2 reads (down).
    down: Kind,
    /// `(hidden, ffn)`.
    dims: (usize, usize),
}

/// Q8_0 throughout, as Gemma 4's published GGUF is. F is deliberately not a
/// multiple of 64, so no affine group size can hide in it.
const Q8_0: Block = Block {
    gate_up: Kind::Q8_0,
    down: Kind::Q8_0,
    dims: (64, 96),
};

/// Q4_K throughout, as Qwen 3.6's Q4_K_M experts are. Both dims are forced up
/// because a K-quant row cannot be a partial superblock.
const Q4K: Block = Block {
    gate_up: Kind::Q4K,
    down: Kind::Q4K,
    dims: (256, 512),
};

/// The Phase S candidate's normal layer: IQ3_XXS gate/up over IQ4_NL down.
/// Mixed in exactly the way the real file is, and `f_dim` is 96 rather than a
/// multiple of 256 precisely because IQ4_NL's block is 32 -- a harness that
/// rounded both dims up to the larger block would never notice a kernel that
/// used the wrong one.
const IQ3_MIX: Block = Block {
    gate_up: Kind::Iq3Xxs,
    down: Kind::Iq4Nl,
    dims: (256, 96),
};

/// The candidate's layer 29: IQ4_XS gate/up over Q8_0 down.
const IQ4XS_MIX: Block = Block {
    gate_up: Kind::Iq4Xs,
    down: Kind::Q8_0,
    dims: (256, 96),
};

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
fn quant_rows(kind: Kind, rows: usize, cols: usize, seed: u64) -> Vec<Vec<u8>> {
    (0..rows)
        .map(|r| kind.row(cols, seed.wrapping_add(r as u64 * 97 + 1)))
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
    let (d_dim, f_dim) = block.dims;
    let g: Vec<&[u8]> = gate.iter().map(|r| r.as_slice()).collect();
    let u: Vec<&[u8]> = up.iter().map(|r| r.as_slice()).collect();
    let gate_out = block.gate_up.gemv(&g, x, d_dim);
    let up_out = block.gate_up.gemv(&u, x, d_dim);

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
    block.down.gemv(&d, &acts, f_dim)
}

/// One expert's gate, up and down row sets, each a byte run per row.
type Expert = (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<Vec<u8>>);

/// The shared body: `top_k` experts of one block type through both kernels,
/// compared against the CPU reference, under the caller's activation.
fn run_case(block: Block, use_silu: bool, top_k: usize) {
    let mut context = MetalContext::new().expect("Metal device");
    let (d_dim, f_dim) = block.dims;

    let experts: Vec<Expert> = (0..top_k)
        .map(|e| {
            let seed = 5000 + 100 * e as u64;
            (
                quant_rows(block.gate_up, f_dim, d_dim, seed + 1),
                quant_rows(block.gate_up, f_dim, d_dim, seed + 2),
                quant_rows(block.down, d_dim, f_dim, seed + 3),
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
    // The two phases are selected INDEPENDENTLY, which is the whole reason
    // this harness exists in its current shape: a mixed expert reads one
    // layout in phase 1 and another in phase 2.
    let phase1 = match block.gate_up {
        Kind::Q8_0 => turbospark_gpu::encode_moe_phase1_q8_0,
        Kind::Q4K => turbospark_gpu::encode_moe_phase1_q4_k,
        Kind::Iq3Xxs => turbospark_gpu::encode_moe_phase1_iq3_xxs,
        Kind::Iq4Xs => turbospark_gpu::encode_moe_phase1_iq4_xs,
        Kind::Iq4Nl => panic!("no IQ4_NL phase 1: no real file puts it in gate/up"),
    };
    let phase2 = match block.down {
        Kind::Q8_0 => turbospark_gpu::encode_moe_phase2_q8_0,
        Kind::Q4K => turbospark_gpu::encode_moe_phase2_q4_k,
        Kind::Iq4Nl => turbospark_gpu::encode_moe_phase2_iq4_nl,
        other => panic!("no {other:?} phase 2: no real file puts it in down"),
    };
    phase1(
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
    phase2(
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
    run_case(Q8_0, false, 2);
}

/// The activation is a function constant shared with the vendored kernels,
/// so it is exercised rather than assumed: SiLU and GELU differ enough that
/// a kernel wired to the wrong one fails this while passing the case above.
#[test]
fn the_silu_activation_constant_reaches_the_gguf_kernels() {
    run_case(Q8_0, true, 2);
}

/// Eight slots is the reduce's fixed shape. Filling every one of them checks
/// that phase 1 writes the whole `acts` range the reduce reads, which a
/// two-slot case cannot: there, six slots are covered by the zero-fill.
#[test]
fn all_eight_slots_participate() {
    run_case(Q8_0, false, MAX_STREAMED_EXPERTS);
}

#[test]
fn the_q4_k_decode_pair_matches_the_cpu_reference() {
    run_case(Q4K, false, 2);
}

/// Qwen 3.6, the one family whose experts are Q4_K, uses SiLU. So this arm is
/// the production combination rather than a symmetry with the Q8_0 case.
#[test]
fn the_silu_activation_constant_reaches_the_q4_k_kernels() {
    run_case(Q4K, true, 2);
}

#[test]
fn all_eight_q4_k_slots_participate() {
    run_case(Q4K, false, MAX_STREAMED_EXPERTS);
}

/// The Phase S candidate's normal layer, and the first MIXED expert any test
/// here has run: IQ3_XXS gate/up feeding an IQ4_NL down. Gemma 4 uses GELU.
#[test]
fn the_iq3_xxs_over_iq4_nl_expert_matches_the_cpu_reference() {
    run_case(IQ3_MIX, false, 2);
}

#[test]
fn all_eight_iq3_xxs_slots_participate() {
    run_case(IQ3_MIX, false, MAX_STREAMED_EXPERTS);
}

/// The candidate's layer 29: IQ4_XS gate/up over a Q8_0 down. Also the case
/// that proves the two phases really are independent, since it pairs a new
/// kernel with an old one.
#[test]
fn the_iq4_xs_over_q8_0_expert_matches_the_cpu_reference() {
    run_case(IQ4XS_MIX, false, 2);
}

/// SiLU on an IQ pair. Gemma 4 is a GELU model so this is not the production
/// combination, but the activation is a function constant threaded through
/// the same `constants_key`, and a new kernel that failed to declare it would
/// silently take the default.
#[test]
fn the_silu_activation_constant_reaches_the_iq_kernels() {
    run_case(IQ3_MIX, true, 2);
}
