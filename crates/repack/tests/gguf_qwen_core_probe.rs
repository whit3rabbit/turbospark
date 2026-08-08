//! THROWAWAY probe, the Qwen sibling of `gguf_norm_convention_probe.rs`:
//! does the Q4_K_M GGUF install's resident core agree with the MLX one?
//!
//! Written because the real Qwen GGUF install decodes without erroring and
//! produces incoherent text, which is the signature of a resident tensor
//! that arrived with the right shape and the wrong values. The Gemma probe
//! answered the same question for that family (bit-identical, which killed
//! the norm "+1" hypothesis); the repack ALSO reported 131 Qwen tensors
//! losing bits on the way to BF16, so this file cannot assume the same.
//!
//!   MREFRUST_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
//!   MREFRUST_QWEN36_GGUF_INSTALL_DIR=~/models/qwen36-gguf.gturbo \
//!     cargo test -p mrefrust-repack --test gguf_qwen_core_probe --release -- --ignored --nocapture

use std::path::Path;

fn bf16_tensor(dir: &Path, name: &str) -> Option<Vec<f32>> {
    let index =
        model_io::load_resident_index(&dir.join("model_weights.bin")).expect("resident index");
    let e = index.entries.get(name)?;
    if e.dtype != 1 {
        println!("  (skipped {name} in {}: dtype {})", dir.display(), e.dtype);
        return None;
    }
    let bytes = std::fs::read(dir.join("model_weights.bin")).expect("weights");
    let start = e.file_offset as usize;
    Some(
        bytes[start..start + e.size_bytes as usize]
            .chunks_exact(2)
            .map(|c| compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
    )
}

#[test]
#[ignore = "needs both Qwen installs"]
fn the_resident_core_agrees_between_the_two_qwen_installs() {
    let mlx = std::path::PathBuf::from(std::env::var_os("MREFRUST_QWEN36_INSTALL_DIR").unwrap());
    let gguf =
        std::path::PathBuf::from(std::env::var_os("MREFRUST_QWEN36_GGUF_INSTALL_DIR").unwrap());

    for name in [
        "language_model.model.layers.0.input_layernorm.weight",
        "language_model.model.layers.0.post_attention_layernorm.weight",
        "language_model.model.norm.weight",
        // The gated-DeltaNet parameters, which are where a mismapped or
        // mis-transcoded tensor would hurt most: they drive a recurrence.
        "language_model.model.layers.0.linear_attn.A_log",
        "language_model.model.layers.0.linear_attn.dt_bias",
        "language_model.model.layers.0.linear_attn.norm.weight",
        "language_model.model.layers.0.linear_attn.conv1d.weight",
    ] {
        let (Some(a), Some(b)) = (bf16_tensor(&mlx, name), bf16_tensor(&gguf, name)) else {
            println!("{name}: absent from one install, skipped");
            continue;
        };
        assert_eq!(a.len(), b.len(), "{name} length");
        let n = a.len() as f32;
        let mean_a = a.iter().sum::<f32>() / n;
        let mean_b = b.iter().sum::<f32>() / n;
        let diff: Vec<f32> = a.iter().zip(&b).map(|(x, y)| y - x).collect();
        let mean_d = diff.iter().sum::<f32>() / n;
        let max_dev = diff.iter().fold(0.0f32, |m, &d| m.max((d - mean_d).abs()));
        let corr = compute::pearson(&a, &b);
        println!(
            "{name}\n  mlx mean {mean_a:+.5}  gguf mean {mean_b:+.5}  \
             mean(gguf-mlx) {mean_d:+.5}  max deviation from that constant {max_dev:.6}  \
             pearson {corr:+.5}"
        );
    }
}

/// Reports the worst relative error between two vectors, which is the metric
/// a CANDIDATE TRANSFORM is judged by: correlation says "related", this says
/// "the same numbers".
fn worst_rel(label: &str, want: &[f32], got: &[f32]) {
    assert_eq!(want.len(), got.len(), "{label}: length");
    let worst = want
        .iter()
        .zip(got)
        .map(|(w, g)| (w - g).abs() / w.abs().max(1e-6))
        .fold(0.0f32, f32::max);
    println!(
        "  {label}: worst relative error {worst:.6}  pearson {:+.5}",
        compute::pearson(want, got)
    );
}

/// The candidate transforms, measured rather than argued.
///
/// llama.cpp's converter is documented to apply `A = -exp(A_log)` for the
/// qwen3next family and to squeeze conv1d's HF `[channels, 1, kernel]` down
/// to two dims, whose GGUF storage order is then the reverse of what this
/// port's flat read expects. Both are hypotheses until they land on the
/// bytes, which is the discipline `FUSED_GATE_FIRST` established: correlate
/// the transform, do not reason about the converter.
#[test]
#[ignore = "needs both Qwen installs"]
fn candidate_transforms_against_the_mlx_install() {
    let mlx = std::path::PathBuf::from(std::env::var_os("MREFRUST_QWEN36_INSTALL_DIR").unwrap());
    let gguf =
        std::path::PathBuf::from(std::env::var_os("MREFRUST_QWEN36_GGUF_INSTALL_DIR").unwrap());
    let pair = |name: &str| {
        (
            bf16_tensor(&mlx, name).expect("mlx tensor"),
            bf16_tensor(&gguf, name).expect("gguf tensor"),
        )
    };

    println!("A_log: is the GGUF side -exp(A_log)?");
    let (a_log_mlx, a_gguf) = pair("language_model.model.layers.0.linear_attn.A_log");
    let neg_exp: Vec<f32> = a_log_mlx.iter().map(|v| -v.exp()).collect();
    worst_rel("-exp(mlx A_log) vs gguf", &neg_exp, &a_gguf);
    // The inverse direction, in case the file stores the log and the install
    // stores the value rather than the other way round.
    let log_neg: Vec<f32> = a_gguf.iter().map(|v| (-v).max(1e-30).ln()).collect();
    worst_rel("log(-gguf) vs mlx A_log", &a_log_mlx, &log_neg);

    println!("dt_bias: same multiset, different order?");
    let (dt_mlx, dt_gguf) = pair("language_model.model.layers.0.linear_attn.dt_bias");
    let mut sorted_mlx = dt_mlx.clone();
    let mut sorted_gguf = dt_gguf.clone();
    sorted_mlx.sort_by(f32::total_cmp);
    sorted_gguf.sort_by(f32::total_cmp);
    worst_rel("sorted(mlx) vs sorted(gguf)", &sorted_mlx, &sorted_gguf);

    // dt_bias being a pure permutation changes the question for every other
    // per-head vector: an ELEMENT-WISE test of a candidate transform fails on
    // the ordering alone, whatever the transform. So each candidate is also
    // tried against sorted sides, which is invariant to the permutation and
    // still rejects a wrong transform.
    println!("A_log again, invariant to the ordering:");
    let mut s_gguf = a_gguf.clone();
    s_gguf.sort_by(f32::total_cmp);
    for (label, values) in [
        ("sorted(mlx A_log)", a_log_mlx.clone()),
        ("sorted(-exp(mlx A_log))", neg_exp.clone()),
    ] {
        let mut sorted = values;
        sorted.sort_by(f32::total_cmp);
        worst_rel(&format!("{label} vs sorted(gguf)"), &sorted, &s_gguf);
    }

    // The LEAD hypothesis, and it needs no matcher: Gotcha 29 says GGUF
    // stores dims fastest-varying first, so a logical `[channels, kernel]`
    // lands as `[kernel, channels]` and the walk copies it verbatim. A flat
    // read is then a strided permutation of the same multiset, which is
    // exactly what a sorted match reports. Exact, and immune to the
    // duplicate-value problem the index map has.
    println!("conv1d: is the GGUF side simply the transpose?");
    let (conv_mlx, conv_gguf) = pair("language_model.model.layers.0.linear_attn.conv1d.weight");
    const CONV_K: usize = 4;
    let channels = conv_mlx.len() / CONV_K;
    let transposed: Vec<f32> = (0..CONV_K)
        .flat_map(|k| (0..channels).map(move |c| (c, k)))
        .map(|(c, k)| conv_mlx[c * CONV_K + k])
        .collect();
    worst_rel("transpose(mlx) vs gguf", &transposed, &conv_gguf);

    // Transpose is refuted (see the run above), and element-wise the two
    // flat vectors correlate +0.82, which a full transpose could not do. So
    // most channels agree and a REGION moved. Located by per-channel
    // equality, which duplicate values cannot confuse the way the index map
    // can: a channel is 4 consecutive values and either matches or does not.
    println!("conv1d: which channels differ?");
    let differing: Vec<usize> = (0..channels)
        .filter(|&c| {
            let at = c * CONV_K;
            conv_mlx[at..at + CONV_K] != conv_gguf[at..at + CONV_K]
        })
        .collect();
    match (differing.first(), differing.last()) {
        (Some(&first), Some(&last)) => println!(
            "  {} of {channels} channels differ, from {first} to {last}; first few {:?}",
            differing.len(),
            &differing[..differing.len().min(8)]
        ),
        _ => println!("  every channel matches"),
    }

    // 3840 differing channels from 4224 to 8063 is 30 x 128, and with the
    // channel run laid out `[q 2048 | k 2048 | v 4096]` at a 128-wide head,
    // that is exactly the 30 V heads the dt_bias de-interleave MOVES: under
    // `2h` / `2(h - 16) + 1`, heads 0 and 31 are fixed points and the other
    // 30 are not. So the candidate is the same one convention, applied to
    // the V region of the channel run.
    println!("conv1d: the V-head de-interleave, applied to the v region only?");
    const V_AT: usize = 4096;
    const HEAD: usize = 128;
    let v_heads = (channels - V_AT) / HEAD;
    let mut candidate = conv_mlx.clone();
    for h in 0..v_heads {
        let from = if h < v_heads / 2 {
            2 * h
        } else {
            2 * (h - v_heads / 2) + 1
        };
        let (dst, src) = ((V_AT + h * HEAD) * CONV_K, (V_AT + from * HEAD) * CONV_K);
        candidate[dst..dst + HEAD * CONV_K].copy_from_slice(&conv_mlx[src..src + HEAD * CONV_K]);
    }
    worst_rel("v-head de-interleave(mlx) vs gguf", &candidate, &conv_gguf);

    println!("conv1d: is it a permutation too?");
    let mut s_conv_mlx = conv_mlx.clone();
    let mut s_conv_gguf = conv_gguf.clone();
    s_conv_mlx.sort_by(f32::total_cmp);
    s_conv_gguf.sort_by(f32::total_cmp);
    worst_rel("sorted(mlx) vs sorted(gguf)", &s_conv_mlx, &s_conv_gguf);

    // Both are permutations of the same values, so the remaining question is
    // WHICH permutation. Printed as an index map rather than guessed at: with
    // 32 distinct values the pattern is readable by eye, and whatever
    // structure it has (a head-group reorder, an interleave) shows up as
    // arithmetic in the indices.
    println!("dt_bias: the permutation, gguf index -> mlx index");
    println!("  {:?}", permutation(&dt_gguf, &dt_mlx));

    // Does the SAME map explain A_log, once its transform is undone?
    println!("A_log: does the dt_bias permutation carry over?");
    let dt_map = permutation(&dt_gguf, &dt_mlx);
    let deinterleaved: Vec<f32> = dt_map.iter().map(|&i| neg_exp[i.min(31)]).collect();
    worst_rel(
        "dt_bias's map applied to -exp(mlx) vs gguf",
        &deinterleaved,
        &a_gguf,
    );

    // conv1d is identity at the start, so the reorder is confined to a
    // region. Report where it begins and how wide the strides are: the
    // channel run covers q, k and v, and only one of those need move.
    println!("conv1d: where the identity stops");
    let conv_map = permutation(&conv_gguf, &conv_mlx);
    let first_moved = conv_map.iter().enumerate().find(|(i, &m)| m != *i);
    match first_moved {
        Some((i, &m)) => println!(
            "  first non-identity at channel {i} -> {m}, of {} (map around it: {:?})",
            conv_map.len(),
            &conv_map[i.saturating_sub(2)..(i + 6).min(conv_map.len())]
        ),
        None => println!("  identity throughout"),
    }
}

/// For each position in `from`, the position of the same value in `to`.
/// `usize::MAX` where the value is not found, which is what a near-miss
/// (a transform, not a permutation) looks like.
fn permutation(from: &[f32], to: &[f32]) -> Vec<usize> {
    from.iter()
        .map(|v| to.iter().position(|w| w == v).unwrap_or(usize::MAX))
        .collect()
}
