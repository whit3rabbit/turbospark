//! Is transcoding GGUF's F32 norms back to BF16 lossless? (ROADMAP Phase G
//! Stage 2, the non-kernel remainder.)
//!
//! GGUF ships Gemma 4's norms and its router as F32 where this port's kernels
//! want BF16 (`rms_norm_bf16w`) and INT8 (`router_gemv_gemma4_r4`). The
//! roadmap left that as an open tradeoff: an F32 path costs two more kernels
//! and bandwidth on the resident core, while a repack-time transcode costs
//! the lossless-repack property for those tensors.
//!
//! That tradeoff is only real if the F32 values carry information a BF16
//! cannot hold. Gemma's norms are BF16 in the original checkpoint and
//! llama.cpp's converter UPCASTS them, so the question is empirical and this
//! is the measurement: an upcast BF16 has sixteen zero low bits, and
//! narrowing it back is exact. Do not reason about this from the format docs,
//! which only say the stored type is F32.
//!
//! The router is a separate question with a separate answer, and it gets its
//! own test below rather than being folded in: it is genuinely lossy either
//! way, and the point is that its loss is the loss this port already accepts.
//!
//! Costs a few KB of ranged reads, not a download.
//!
//! ```sh
//! cargo test -p mrefrust-repack --test gguf_f32_transcode_network --release -- --ignored --nocapture
//! ```

use compute::{bf16_to_f32, f32_to_bf16};
use mrefrust_repack::{fetch_gguf_header, GgufHeader, HttpRangeSource, RangeSource};

const GEMMA4_Q8_0: &str = "https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-Q8_0.gguf";

/// Read a whole F32 tensor. Only called on norms and the router, which are
/// vectors or small matrices, never on an expert.
fn read_f32(source: &dyn RangeSource, header: &GgufHeader, name: &str) -> Vec<f32> {
    let info = header
        .tensors
        .get(name)
        .unwrap_or_else(|| panic!("no tensor {name}"));
    assert_eq!(info.ggml_type, 0, "{name} is not F32");
    let (start, end) = header
        .absolute_range(name)
        .expect("in the table")
        .expect("has a size");
    let bytes = source.read_range(start, end).expect("ranged read");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Every F32 norm tensor on layer 0, plus the two model-level ones. Named
/// individually rather than pattern-matched: a norm this port forgets to
/// transcode would otherwise pass by not being looked at.
const NORMS: &[&str] = &[
    "blk.0.attn_norm.weight",
    "blk.0.attn_q_norm.weight",
    "blk.0.attn_k_norm.weight",
    "blk.0.post_attention_norm.weight",
    "blk.0.ffn_norm.weight",
    "blk.0.post_ffw_norm.weight",
    "output_norm.weight",
];

#[test]
#[ignore = "network: reads a few KB off a 27 GB remote checkpoint"]
fn the_f32_norms_are_upcast_bf16_and_narrow_back_exactly() {
    let source = HttpRangeSource::new(GEMMA4_Q8_0);
    let header = fetch_gguf_header(&source).expect("GGUF header");

    let mut checked = 0usize;
    for name in NORMS {
        if !header.tensors.contains_key(*name) {
            println!("(absent, skipped: {name})");
            continue;
        }
        let values = read_f32(&source, &header, name);
        assert!(!values.is_empty());

        // The whole claim in one line: narrowing to BF16 and widening back
        // must be the identity. If llama.cpp had produced genuinely F32 norms
        // this fails on the first element with a non-zero low half.
        let lossy = values
            .iter()
            .filter(|&&v| bf16_to_f32(f32_to_bf16(v)) != v)
            .count();
        println!(
            "{name}: {} values, {lossy} would lose bits in BF16",
            values.len()
        );
        assert_eq!(
            lossy, 0,
            "{name} carries F32 precision a BF16 transcode would drop"
        );
        checked += 1;
    }
    assert!(
        checked >= 5,
        "only {checked} norms found; the name list is stale"
    );
    println!("\nAll {checked} norm tensors are exactly representable in BF16.");
    println!("A repack-time transcode is therefore LOSSLESS for norms, and the");
    println!("roadmap's F32-path-versus-transcode tradeoff does not apply to them.");
}

/// The router is the other half of the remainder and does NOT get the same
/// answer, which is the point of measuring it separately: it is F32 in the
/// file, the runtime wants INT8, and that transcode genuinely quantizes.
///
/// What is measured is the thing downstream actually consumes. A router's
/// output is not read as numbers, it is argsorted and the top `top_k`
/// experts are dispatched, so the question is whether quantization changes
/// WHICH experts get picked, not by how much a logit moved.
///
/// Per-element relative error is the wrong metric here and was tried first:
/// it reads 166 on this tensor, entirely from weights near zero (a 3.6e-6
/// weight against a 0.0006 quantization step), and says nothing about
/// routing. Do not reintroduce it.
#[test]
#[ignore = "network: reads a few KB off a 27 GB remote checkpoint"]
fn int8_transcoding_the_router_does_not_change_which_experts_route() {
    let source = HttpRangeSource::new(GEMMA4_Q8_0);
    let header = fetch_gguf_header(&source).expect("GGUF header");

    let name = "blk.0.ffn_gate_inp.weight";
    let values = read_f32(&source, &header, name);
    let hidden = header
        .metadata
        .iter()
        .find(|(k, _)| k.ends_with(".embedding_length"))
        .and_then(|(_, v)| v.as_u64())
        .expect("embedding_length") as usize;
    let experts = values.len() / hidden;
    assert_eq!(values.len() % hidden, 0, "router is not a whole matrix");
    let amax = values.iter().fold(0f32, |a, &v| a.max(v.abs()));
    println!("{name}: {experts} experts x {hidden}, max |w| {amax:.6}");

    // Per-row INT8 affine, exactly what `repack.rs` applies to the MLX
    // checkpoint's router.
    let rows: Vec<Vec<f32>> = values
        .chunks_exact(hidden)
        .map(|row| {
            let q = compute::quantize_int8_affine(row);
            compute::dequantize_int8_affine(&q, hidden)
        })
        .collect();

    let exact_rows: Vec<&[f32]> = values.chunks_exact(hidden).collect();
    let quant_rows: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();

    const TOP_K: usize = 8;
    const TRIALS: usize = 32;
    let mut agree = 0usize;
    let mut top1 = 0usize;
    // Worst ratio, over every expert that entered or left a top-k set, of how
    // far it sat from the cut in EXACT logits against how much quantization
    // moved any logit in that trial. Under 1.0 means every flip happened
    // inside the noise: quantization never reordered a decided pair, it only
    // resolved ties it could not see. Self-calibrating, so there is no
    // invented tolerance here.
    let mut worst_ratio = 0f32;
    for t in 0..TRIALS {
        // Deterministic pseudo-random activations, unit-ish scale.
        let mut s = (t as u32).wrapping_mul(2_654_435_761).wrapping_add(17);
        let x: Vec<f32> = (0..hidden)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32 / (1u32 << 23) as f32) - 1.0
            })
            .collect();

        let score = |w: &[&[f32]]| -> Vec<f32> {
            w.iter()
                .map(|row| row.iter().zip(x.iter()).map(|(a, b)| a * b).sum())
                .collect()
        };
        let exact_logits = score(&exact_rows);
        let quant_logits = score(&quant_rows);
        let top_k = |l: &[f32]| -> Vec<usize> {
            let mut order: Vec<usize> = (0..experts).collect();
            order.sort_by(|&a, &b| l[b].total_cmp(&l[a]));
            order.truncate(TOP_K);
            order
        };
        let exact = top_k(&exact_logits);
        let quantized = top_k(&quant_logits);

        if exact == quantized {
            agree += 1;
        }
        if exact[0] == quantized[0] {
            top1 += 1;
        }

        // How much quantization perturbed any logit at all, this trial.
        let noise = exact_logits
            .iter()
            .zip(quant_logits.iter())
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        // The exact-logit value at the cut: the last expert the exact router
        // admits. Anything flipping in or out is compared against this.
        let cut = exact_logits[exact[TOP_K - 1]];
        let in_exact: std::collections::HashSet<usize> = exact.iter().copied().collect();
        let in_quant: std::collections::HashSet<usize> = quantized.iter().copied().collect();
        for e in in_exact.symmetric_difference(&in_quant) {
            if noise > 0.0 {
                worst_ratio = worst_ratio.max((exact_logits[*e] - cut).abs() / noise);
            }
        }
    }
    println!("over {TRIALS} random activations: top-{TOP_K} set identical on {agree}, top-1 identical on {top1}");
    println!("worst flip distance from the cut, in units of that trial's quantization noise: {worst_ratio:.3}");

    // Top-1 is the one position with no tie to resolve: the best expert is
    // separated from the field, so quantization must not move it.
    assert_eq!(
        top1,
        TRIALS,
        "INT8 transcoding changed the top-1 expert on {} of {TRIALS} trials",
        TRIALS - top1
    );
    // The set does change, and that is expected rather than alarming: with
    // 128 experts and a cut at 8, the experts either side of the cut are
    // routinely within quantization noise of each other. What must hold is
    // that EVERY flip is such a case. A flip further from the cut than the
    // noise could reach would mean quantization reordered a decided pair.
    assert!(
        worst_ratio <= 1.0,
        "an expert {worst_ratio:.3} noise-widths from the cut changed sides; \
         quantization is reordering decided pairs, not just breaking ties"
    );
    println!("\nThe router is lossy either way, but the transcode applies the SAME");
    println!("INT8 affine the MLX-derived install already carries, and the routing");
    println!("decision survives it. No quality is lost that this port has not");
    println!("already accepted and measured.");
}
