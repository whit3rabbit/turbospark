//! One whole `qwen3_5` vision block, GPU against CPU, at the tower's real
//! shapes. ROADMAP M-V2's integration gate.
//!
//! The per-kernel cases in `vision_parity.rs` prove each kernel computes what
//! `turbospark_compute::vision` says it does. They cannot prove the kernels
//! COMPOSE: every one of them passes while the block wires them together
//! wrongly -- a q/k/v slice taken in the wrong order, rope applied to `v`,
//! the residual added before the norm instead of after. Those are the errors
//! that produce fluent, wrong images, and this file is what sees them.
//!
//! # No split kernel
//!
//! The fused `qkv` projection writes `[seq, 3, heads, head_dim]`, so `q`, `k`
//! and `v` are interleaved per token and none of them is contiguous. Rather
//! than add a scatter kernel, this runs the projection as THREE matmuls at
//! three byte offsets into the same `[3456, 1152]` weight -- the weight is
//! row-major by output, so its first 1152 rows ARE q's projection. Each
//! writes a contiguous `[seq, hidden]` buffer that the rope and attention
//! kernels can read directly, and the arithmetic is identical.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_compute::vision::{
    attention_scale, bidirectional_attention, gelu_tanh_vision, layer_norm, matmul_bias,
    rope_vision_2d,
};
use turbospark_gpu::{
    encode_vision_attention, encode_vision_gelu, encode_vision_layer_norm, encode_vision_matmul,
    encode_vision_residual_add, encode_vision_rope_2d, GeluKind, MetalContext,
};

/// The real tower, from `docs/VISION_PHASE0.md` item 1. `SEQ` is the one
/// number that is not the checkpoint's: a real page is thousands of patches
/// and the CPU reference is a triple loop, so this runs the real WIDTHS at a
/// short sequence. Every shape that could be transposed or mis-strided is a
/// width.
const SEQ: usize = 8;
const HIDDEN: usize = 1152;
const HEADS: usize = 16;
const HEAD_DIM: usize = HIDDEN / HEADS; // 72
const INTERMEDIATE: usize = 4304;
const EPS: f32 = 1e-6;

fn to_le(values: &[f16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect()
}

fn read_f16(buffer: &metal::Buffer, len: usize) -> Vec<f32> {
    let ptr = buffer.contents() as *const u16;
    let bits = unsafe { std::slice::from_raw_parts(ptr, len) };
    bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
}

/// Round through FP16 and back. Every tensor the block sees is FP16 storage,
/// so the CPU reference has to run on the SAME values the kernels do --
/// comparing against the f32 originals would fold the storage rounding into
/// the parity bound and hide a real error of the same size.
fn quantize(values: &[f32]) -> Vec<f32> {
    values.iter().map(|&v| f16::from_f32(v).to_f32()).collect()
}

/// Deterministic, non-flat, and scaled so activations stay in the range a
/// real block sees rather than saturating FP16.
fn weights(n: usize, phase: f32, amp: f32) -> Vec<f32> {
    quantize(
        &(0..n)
            .map(|i| ((i as f32) * 0.0137 + phase).sin() * amp)
            .collect::<Vec<f32>>(),
    )
}

struct BlockWeights {
    norm1_w: Vec<f32>,
    norm1_b: Vec<f32>,
    qkv_w: Vec<f32>,
    qkv_b: Vec<f32>,
    proj_w: Vec<f32>,
    proj_b: Vec<f32>,
    norm2_w: Vec<f32>,
    norm2_b: Vec<f32>,
    fc1_w: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2_w: Vec<f32>,
    fc2_b: Vec<f32>,
}

impl BlockWeights {
    fn new() -> Self {
        Self {
            norm1_w: weights(HIDDEN, 0.1, 1.0),
            norm1_b: weights(HIDDEN, 0.2, 0.1),
            qkv_w: weights(3 * HIDDEN * HIDDEN, 0.3, 0.03),
            qkv_b: weights(3 * HIDDEN, 0.4, 0.05),
            proj_w: weights(HIDDEN * HIDDEN, 0.5, 0.03),
            proj_b: weights(HIDDEN, 0.6, 0.05),
            norm2_w: weights(HIDDEN, 0.7, 1.0),
            norm2_b: weights(HIDDEN, 0.8, 0.1),
            fc1_w: weights(INTERMEDIATE * HIDDEN, 0.9, 0.03),
            fc1_b: weights(INTERMEDIATE, 1.0, 0.05),
            fc2_w: weights(HIDDEN * INTERMEDIATE, 1.1, 0.03),
            fc2_b: weights(HIDDEN, 1.2, 0.05),
        }
    }
}

/// The reference block, straight off `Qwen3VLMoEVisionBlock.__call__`:
///
/// ```text
/// h = h + attn(norm1(h))
/// h = h + mlp(norm2(h))
/// ```
fn cpu_block(x: &[f32], w: &BlockWeights, freqs: &[f32]) -> Vec<f32> {
    let half = HEAD_DIM / 2;
    let mut h = x.to_vec();

    // --- attention half ---
    let mut n1 = Vec::with_capacity(SEQ * HIDDEN);
    for t in 0..SEQ {
        n1.extend(layer_norm(
            &h[t * HIDDEN..(t + 1) * HIDDEN],
            &w.norm1_w,
            &w.norm1_b,
            EPS,
        ));
    }

    // Three projections out of the one fused weight, exactly as the GPU side
    // slices it: rows 0..HIDDEN are q, HIDDEN..2*HIDDEN are k, the rest v.
    let mut qkv = Vec::with_capacity(3);
    for part in 0..3 {
        let wslice = &w.qkv_w[part * HIDDEN * HIDDEN..(part + 1) * HIDDEN * HIDDEN];
        let bslice = &w.qkv_b[part * HIDDEN..(part + 1) * HIDDEN];
        qkv.push(quantize(&matmul_bias(
            &n1,
            wslice,
            Some(bslice),
            SEQ,
            HIDDEN,
            HIDDEN,
        )));
    }

    // Rope on q and k, NEVER on v -- the reference applies it to the first
    // two only, and applying it to v produces a finite, wrong image.
    for target in qkv.iter_mut().take(2) {
        for t in 0..SEQ {
            let row = &freqs[t * half..(t + 1) * half];
            for head in 0..HEADS {
                let base = t * HIDDEN + head * HEAD_DIM;
                let rotated = rope_vision_2d(&target[base..base + HEAD_DIM], row);
                target[base..base + HEAD_DIM].copy_from_slice(&quantize(&rotated));
            }
        }
    }

    let attn = quantize(&bidirectional_attention(
        &qkv[0],
        &qkv[1],
        &qkv[2],
        SEQ,
        HEADS,
        HEAD_DIM,
        attention_scale(HEAD_DIM),
    ));
    let proj = quantize(&matmul_bias(
        &attn,
        &w.proj_w,
        Some(&w.proj_b),
        SEQ,
        HIDDEN,
        HIDDEN,
    ));
    for (slot, value) in h.iter_mut().zip(&proj) {
        *slot = f16::from_f32(*slot + value).to_f32();
    }

    // --- MLP half ---
    let mut n2 = Vec::with_capacity(SEQ * HIDDEN);
    for t in 0..SEQ {
        n2.extend(layer_norm(
            &h[t * HIDDEN..(t + 1) * HIDDEN],
            &w.norm2_w,
            &w.norm2_b,
            EPS,
        ));
    }
    let f1 = quantize(&matmul_bias(
        &n2,
        &w.fc1_w,
        Some(&w.fc1_b),
        SEQ,
        HIDDEN,
        INTERMEDIATE,
    ));
    // The per-block MLP uses the TANH approximation. The merger uses erf.
    let act = quantize(&gelu_tanh_vision(&f1));
    let f2 = quantize(&matmul_bias(
        &act,
        &w.fc2_w,
        Some(&w.fc2_b),
        SEQ,
        INTERMEDIATE,
        HIDDEN,
    ));
    for (slot, value) in h.iter_mut().zip(&f2) {
        *slot = f16::from_f32(*slot + value).to_f32();
    }
    h
}

#[allow(clippy::too_many_lines)]
fn gpu_block(context: &mut MetalContext, x: &[f32], w: &BlockWeights, freqs: &[f32]) -> Vec<f32> {
    gpu_block_with(context, x, w, freqs, GeluKind::Tanh)
}

fn gpu_block_with(
    context: &mut MetalContext,
    x: &[f32],
    w: &BlockWeights,
    freqs: &[f32],
    gelu: GeluKind,
) -> Vec<f32> {
    let half_bytes = |n: usize| (n * 2) as u64;
    let upload = |context: &MetalContext, v: &[f32]| {
        context.new_buffer_with_data(&to_le(
            &v.iter().map(|&f| f16::from_f32(f)).collect::<Vec<f16>>(),
        ))
    };
    // B7: `freqs` is F32 on the wire, unlike every other tensor this block
    // reads -- no FP16 narrowing here.
    let upload_f32 = |context: &MetalContext, v: &[f32]| context.new_buffer_with_data(v);

    let h_buf = upload(context, x);
    let freq_buf = upload_f32(context, freqs);
    let n1_w = upload(context, &w.norm1_w);
    let n1_b = upload(context, &w.norm1_b);
    let qkv_w = upload(context, &w.qkv_w);
    let qkv_b = upload(context, &w.qkv_b);
    let proj_w = upload(context, &w.proj_w);
    let proj_b = upload(context, &w.proj_b);
    let n2_w = upload(context, &w.norm2_w);
    let n2_b = upload(context, &w.norm2_b);
    let fc1_w = upload(context, &w.fc1_w);
    let fc1_b = upload(context, &w.fc1_b);
    let fc2_w = upload(context, &w.fc2_w);
    let fc2_b = upload(context, &w.fc2_b);

    let normed = context.new_output_buffer(half_bytes(SEQ * HIDDEN));
    let q = context.new_output_buffer(half_bytes(SEQ * HIDDEN));
    let k = context.new_output_buffer(half_bytes(SEQ * HIDDEN));
    let v = context.new_output_buffer(half_bytes(SEQ * HIDDEN));
    let attn = context.new_output_buffer(half_bytes(SEQ * HIDDEN));
    let proj = context.new_output_buffer(half_bytes(SEQ * HIDDEN));
    let hidden_wide = context.new_output_buffer(half_bytes(SEQ * INTERMEDIATE));

    let pass = context.begin_pass();

    // h -> norm1
    encode_vision_layer_norm(
        context,
        &pass,
        (&h_buf, 0),
        (&n1_w, 0),
        (&n1_b, 0),
        (&normed, 0),
        SEQ as u32,
        HIDDEN as u32,
        EPS,
    )
    .expect("norm1");

    // Three projections at three offsets into one fused weight.
    let part_w = half_bytes(HIDDEN * HIDDEN);
    let part_b = half_bytes(HIDDEN);
    for (part, out) in [&q, &k, &v].into_iter().enumerate() {
        encode_vision_matmul(
            context,
            &pass,
            (&normed, 0),
            (&qkv_w, part as u64 * part_w),
            Some((&qkv_b, part as u64 * part_b)),
            (out, 0),
            SEQ as u32,
            HIDDEN as u32,
            HIDDEN as u32,
        )
        .expect("qkv");
    }

    // Rope on q and k only.
    for target in [&q, &k] {
        encode_vision_rope_2d(
            context,
            &pass,
            (target, 0),
            (&freq_buf, 0),
            SEQ as u32,
            HEADS as u32,
            HEAD_DIM as u32,
        )
        .expect("rope");
    }

    encode_vision_attention(
        context,
        &pass,
        (&q, 0),
        (&k, 0),
        (&v, 0),
        (&attn, 0),
        SEQ as u32,
        HEADS as u32,
        HEAD_DIM as u32,
        attention_scale(HEAD_DIM),
    )
    .expect("attention");

    encode_vision_matmul(
        context,
        &pass,
        (&attn, 0),
        (&proj_w, 0),
        Some((&proj_b, 0)),
        (&proj, 0),
        SEQ as u32,
        HIDDEN as u32,
        HIDDEN as u32,
    )
    .expect("proj");
    encode_vision_residual_add(
        context,
        &pass,
        (&h_buf, 0),
        (&proj, 0),
        (SEQ * HIDDEN) as u32,
    )
    .expect("residual 1");

    // h -> norm2 -> fc1 -> gelu -> fc2 -> residual
    encode_vision_layer_norm(
        context,
        &pass,
        (&h_buf, 0),
        (&n2_w, 0),
        (&n2_b, 0),
        (&normed, 0),
        SEQ as u32,
        HIDDEN as u32,
        EPS,
    )
    .expect("norm2");
    encode_vision_matmul(
        context,
        &pass,
        (&normed, 0),
        (&fc1_w, 0),
        Some((&fc1_b, 0)),
        (&hidden_wide, 0),
        SEQ as u32,
        HIDDEN as u32,
        INTERMEDIATE as u32,
    )
    .expect("fc1");
    encode_vision_gelu(
        context,
        &pass,
        (&hidden_wide, 0),
        (SEQ * INTERMEDIATE) as u32,
        gelu,
    )
    .expect("gelu");
    encode_vision_matmul(
        context,
        &pass,
        (&hidden_wide, 0),
        (&fc2_w, 0),
        Some((&fc2_b, 0)),
        (&proj, 0),
        SEQ as u32,
        INTERMEDIATE as u32,
        HIDDEN as u32,
    )
    .expect("fc2");
    encode_vision_residual_add(
        context,
        &pass,
        (&h_buf, 0),
        (&proj, 0),
        (SEQ * HIDDEN) as u32,
    )
    .expect("residual 2");

    pass.commit_and_wait();
    read_f16(&h_buf, SEQ * HIDDEN)
}

fn block_inputs() -> (Vec<f32>, Vec<f32>, BlockWeights) {
    let x = weights(SEQ * HIDDEN, 2.0, 1.5);
    // Frequency rows as `turbospark_vision_io::vision_rope_freq_rows` emits
    // them: one row of `head_dim / 2` per token, shared by every head.
    // Not run through `quantize` (B7): `freqs` stays F32 all the way to the
    // kernel now, unlike every other tensor `block_inputs` builds.
    let freqs: Vec<f32> = (0..SEQ * HEAD_DIM / 2)
        .map(|i| (i as f32) * 0.011)
        .collect();
    (x, freqs, BlockWeights::new())
}

#[test]
fn one_whole_block_matches_the_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let (x, freqs, w) = block_inputs();

    let cpu = cpu_block(&x, &w, &freqs);
    let gpu = gpu_block(&mut context, &x, &w, &freqs);
    assert_eq!(gpu.len(), cpu.len());
    // Finiteness at the point the measurement is taken: NaN reads as a
    // perfect score on the `worst`-abs-diff fold below (`f32::max`-style
    // comparisons silently skip a NaN entry rather than surfacing it),
    // which is AGENTS.md Gotcha 59 one level down from the real-model
    // instruments that already guard this way.
    assert!(
        gpu.iter().all(|v| v.is_finite()),
        "this port's block produced a non-finite value"
    );
    assert!(
        cpu.iter().all(|v| v.is_finite()),
        "the cpu reference produced a non-finite value"
    );

    // A block is two norms, five GEMMs of up to 4,304 terms, an attention and
    // two residual adds, every intermediate rounded to FP16. The error
    // accumulates across all of that, so the bound is stated relative to the
    // output's own magnitude rather than as the per-kernel 2e-3.
    let scale = cpu.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    let mut worst = (0.0f32, 0usize);
    for (i, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
        let d = (g - c).abs();
        if d > worst.0 {
            worst = (d, i);
        }
    }
    assert!(
        worst.0 <= 2e-2 * scale,
        "worst {} at {} (scale {scale}): gpu {} cpu {}",
        worst.0,
        worst.1,
        gpu[worst.1],
        cpu[worst.1]
    );

    // And the block DID something: an output equal to its input would pass
    // any agreement bound, and is what a chain of no-ops produces.
    let mut moved = 0.0f32;
    for (g, x0) in gpu.iter().zip(&x) {
        moved = moved.max((g - x0).abs());
    }
    assert!(
        moved > 0.05 * scale,
        "the block barely moved its input: {moved}"
    );
}

#[test]
fn the_block_is_sensitive_to_every_weight_it_reads() {
    // GUARDS THE CASE ABOVE, and is the thing a composition test exists for:
    // a wiring error usually shows up as a tensor that reaches nothing. Each
    // weight is perturbed in turn and the output must move. Note both norms
    // and both residuals are exercised implicitly -- a norm whose weight is
    // ignored fails here.
    let mut context = MetalContext::new().expect("Metal device");
    let (x, freqs, base) = block_inputs();
    let reference = gpu_block(&mut context, &x, &base, &freqs);

    let names: [&str; 12] = [
        "norm1_w", "norm1_b", "qkv_w", "qkv_b", "proj_w", "proj_b", "norm2_w", "norm2_b", "fc1_w",
        "fc1_b", "fc2_w", "fc2_b",
    ];
    for (index, name) in names.iter().enumerate() {
        let mut w = BlockWeights::new();
        let target: &mut Vec<f32> = match index {
            0 => &mut w.norm1_w,
            1 => &mut w.norm1_b,
            2 => &mut w.qkv_w,
            3 => &mut w.qkv_b,
            4 => &mut w.proj_w,
            5 => &mut w.proj_b,
            6 => &mut w.norm2_w,
            7 => &mut w.norm2_b,
            8 => &mut w.fc1_w,
            9 => &mut w.fc1_b,
            10 => &mut w.fc2_w,
            _ => &mut w.fc2_b,
        };
        // A LOCAL perturbation rather than a global scale: scaling a whole
        // weight moves the output through sheer magnitude even when the
        // tensor is barely read, which would make this test pass on a wiring
        // it should catch.
        for slot in target.iter_mut().take(64) {
            *slot = f16::from_f32(*slot + 0.5).to_f32();
        }

        let perturbed = gpu_block(&mut context, &x, &w, &freqs);
        let moved = reference
            .iter()
            .zip(&perturbed)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            moved > 1e-3,
            "{name} does not reach the block output ({moved})"
        );
    }
}

#[test]
fn rope_reaches_q_and_k_but_not_v() {
    // The one wiring error in this block that every per-kernel case passes
    // through: rope applied to `v` as well. It changes the output, stays
    // finite, and reads as an ordinary image. Detected by holding the block
    // fixed and moving ONLY the frequency rows -- if `v` were rotated, the
    // CPU reference (which rotates only q and k) would stop agreeing.
    let mut context = MetalContext::new().expect("Metal device");
    let (x, _, w) = block_inputs();

    // Frequencies large enough that a rotation is unmistakable. Not run
    // through `quantize` (B7): see `block_inputs`.
    let freqs: Vec<f32> = (0..SEQ * HEAD_DIM / 2)
        .map(|i| 0.3 + (i as f32) * 0.05)
        .collect();
    let cpu = cpu_block(&x, &w, &freqs);
    let gpu = gpu_block(&mut context, &x, &w, &freqs);
    assert!(
        gpu.iter().all(|v| v.is_finite()),
        "this port's block produced a non-finite value"
    );

    let scale = cpu.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    let bound = 2e-2 * scale;
    let worst = gpu
        .iter()
        .zip(&cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst <= bound, "worst {worst} (bound {bound})");

    // The fixture discriminates, and the bar is THE PARITY BOUND rather than
    // a fraction of the output's magnitude. That is the question actually
    // being asked: the assertion above can only catch a v-rotating block if
    // rotating changes the output by more than the agreement it allows. A
    // fraction-of-scale bar is arbitrary and was set too high on the first
    // pass -- rope is a ROTATION, so it preserves every norm it touches and
    // its effect downstream of a softmax and two projections is real but
    // modest, measured here at about 2.5x the tolerance.
    let zero = vec![0.0f32; SEQ * HEAD_DIM / 2];
    let unrotated = cpu_block(&x, &w, &zero);
    let gap = cpu
        .iter()
        .zip(&unrotated)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        gap > 2.0 * bound,
        "rotating moves the output by {gap}, only {:.1}x the parity bound of \
         {bound}; this fixture cannot see a block that rotates v too",
        gap / bound
    );
}

#[test]
fn the_blocks_gelu_choice_is_invisible_at_the_parity_bound() {
    // A RECORDED LIMITATION, not a passing check of the wiring.
    //
    // Swapping the block's `GeluKind::Tanh` for `Erf` survives every other
    // case in this file, and this states why rather than leaving a future
    // reader to assume the block test covers it. The two GELUs agree to about
    // 3e-4 while a block's output carries the accumulated FP16 error of two
    // norms, five GEMMs of up to 4,304 terms and an attention -- so the
    // difference is two orders of magnitude under the bound the parity
    // assertion has to allow. No tightening fixes that; the tolerance is real.
    //
    // What DOES pin the choice is `vision_parity.rs`: both kernels are
    // checked against their own CPU references, and
    // `the_two_gelu_kernels_are_different_pipelines` proves the selection
    // reaches the GPU. That the block passes `Tanh` is then a one-line fact
    // readable at the call site, matching `MLP.act_fn = nn.GELU(approx="tanh")`
    // in the reference.
    let mut context = MetalContext::new().expect("Metal device");
    let (x, freqs, w) = block_inputs();

    let tanh = gpu_block_with(&mut context, &x, &w, &freqs, GeluKind::Tanh);
    let erf = gpu_block_with(&mut context, &x, &w, &freqs, GeluKind::Erf);

    let scale = tanh.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    let bound = 2e-2 * scale;
    let gap = tanh
        .iter()
        .zip(&erf)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // The selection reaches the GPU at all.
    assert!(
        gap > 0.0,
        "the two GELU kinds produced an identical block output"
    );
    // And it is below what parity can adjudicate. If this ever fails, the
    // block test HAS gained the power to see the choice and this comment is
    // stale -- which is a better problem than the silent version.
    assert!(
        gap < bound,
        "the GELU choice now moves the block by {gap}, above the parity bound \
         of {bound}; the blindness this test records no longer holds"
    );
}
