use compute::{quantize_int1_affine_symmetric, quantize_int2_affine_ternary};

use super::dense::{
    companion_dtype, group_for, DFLASH_LAYERS, DFLASH_RANK, GROUP, HEAD_DIM, HIDDEN, INTER,
    NUM_HEADS, NUM_KV_HEADS,
};
use crate::synthetic_real::{bf16_vector, deterministic_row, u16_le, Tensor};

/// An UNQUANTIZED vector, F16 like every unquantized tensor in the real
/// checkpoint -- and unlike the Qwen 3.6 fixture's BF16, which this file was
/// forked from.
pub(crate) fn f16_vector(name: &str, n: usize, center: f32, seed: u64) -> Tensor {
    let bits: Vec<u16> = deterministic_row(seed, n)
        .iter()
        .map(|&j| compute::f32_to_f16(center + j * 0.05))
        .collect();
    Tensor {
        name: name.to_string(),
        dtype: "F16",
        shape: vec![n as u64],
        bytes: u16_le(&bits),
    }
}

/// One sub-4-bit-affine weight plus its two FP16 companions, at `bits` = 1 or
/// 2.
pub(crate) fn packed_triple(
    name: &str,
    rows: usize,
    cols: usize,
    seed: u64,
    bits: u32,
) -> Vec<Tensor> {
    let group = group_for(bits);
    assert_eq!(
        cols % group,
        0,
        "{name}: {cols} columns is not a whole number of {group}-element groups"
    );
    let base = name.strip_suffix(".weight").unwrap();
    let mut packed = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for r in 0..rows {
        let row = deterministic_row(seed.wrapping_add(r as u64 * 97 + 1), cols);
        // Each width takes its OWN quantizer, not a shared one with a level
        // count: at one bit the rule is a sign and at two it is a ternary
        // threshold, and both are stated where the layout they produce is.
        let (p, s, b) = match bits {
            1 => {
                let q = quantize_int1_affine_symmetric(&row, GROUP);
                (q.packed, q.scales, q.biases)
            }
            2 => {
                let q = quantize_int2_affine_ternary(&row, GROUP);
                (q.packed, q.scales, q.biases)
            }
            // The width the REAL `Qwen/Qwen3.8-27B` install carries, and the
            // only one with a batched GEMM (`docs/MTP_SPECULATIVE.md` step
            // 4). Its quantizer takes no group argument because `compute`
            // fixes INT4 at `GROUP_SIZE`, which is exactly why `group_for`
            // has to agree with it rather than with the constant above.
            4 => {
                let q = compute::quantize_int4_affine(&row);
                (q.packed, q.scales, q.biases)
            }
            other => panic!("this fixture builds 1-, 2- or 4-bit installs, not {other}"),
        };
        packed.extend_from_slice(&p);
        scales.extend_from_slice(&s);
        biases.extend_from_slice(&b);
    }
    let groups = cols / group;
    let companion = companion_dtype(bits);
    vec![
        Tensor {
            name: name.to_string(),
            dtype: "U32",
            shape: vec![rows as u64, (cols * bits as usize / 32) as u64],
            bytes: packed,
        },
        Tensor {
            name: format!("{base}.scales"),
            dtype: companion,
            shape: vec![rows as u64, groups as u64],
            bytes: u16_le(&scales),
        },
        Tensor {
            name: format!("{base}.biases"),
            dtype: companion,
            shape: vec![rows as u64, groups as u64],
            bytes: u16_le(&biases),
        },
    ]
}

/// A BF16 rank-2 matrix, the dtype and rank the MTP head's projections
/// carry in the official checkpoint.
///
/// Rank-2 is the whole discriminator the walk uses to decide "quantize this"
/// against "narrow this", so a helper that produced rank-1 would silently
/// route a projection down the norm path.
pub(crate) fn bf16_matrix(name: &str, rows: usize, cols: usize, seed: u64) -> Tensor {
    let mut bits: Vec<u16> = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for &j in deterministic_row(seed.wrapping_add(r as u64 * 131 + 1), cols).iter() {
            bits.push(compute::f32_to_bf16(j * 0.05));
        }
    }
    Tensor {
        name: name.to_string(),
        dtype: "BF16",
        shape: vec![rows as u64, cols as u64],
        bytes: u16_le(&bits),
    }
}

/// The multi-token-prediction head, as `Qwen/Qwen3.8-27B` publishes it:
/// fifteen tensors, BF16 throughout, name for name against
/// `crates/repack/tests/mtp_head_network.rs`'s `EXPECTED` table.
///
/// **BF16 where the trunk around it is F16-and-packed, and that asymmetry is
/// the real situation rather than fixture sloppiness.** The trunk install is
/// streamed from an mlx-community conversion that DROPS `mtp.*` entirely, so
/// the head can only come from the official checkpoint -- two publishers, two
/// dtypes, one install. It is also what makes the narrowing assertion mean
/// something: a BF16 source narrows to BF16 losslessly, so this fixture's
/// head must report ZERO lossy values while Bonsai's F16 trunk reports many.
///
/// Three shapes here are load-bearing and each defeats a mutation that a
/// tidier fixture would pass:
///
/// - `fc` is `[hidden, 2 * hidden]`, the only tensor in the model with that
///   width. It takes the CONCATENATION of a normalized next-token embedding
///   and a normalized trunk hidden state, so a fixture with `2 * hidden`
///   anywhere else could not catch a transposed or half-width read.
/// - `q_proj` has `2 * num_heads * head_dim` rows while `o_proj`'s INPUT is
///   the unhalved `num_heads * head_dim`. That asymmetry is `attn_output_gate`
///   being true; a non-gated head would have the same number on both, so
///   equal widths would make the gate unobservable.
/// - the two `pre_fc_norm_*` vectors take DIFFERENT seeds, so swapping them
///   changes the output. Seeded alike they are interchangeable and the
///   embedding/hidden ordering is untestable.
pub(crate) fn mtp_head_tensors() -> Vec<Tensor> {
    let q_out = NUM_HEADS * HEAD_DIM;
    let kv_out = NUM_KV_HEADS * HEAD_DIM;
    let mut ts = vec![
        // The head's own structure, above the block.
        bf16_matrix("mtp.fc.weight", HIDDEN, 2 * HIDDEN, 9_100),
        bf16_vector("mtp.pre_fc_norm_embedding.weight", HIDDEN, 1.0, 9_101),
        bf16_vector("mtp.pre_fc_norm_hidden.weight", HIDDEN, 1.0, 9_102),
        bf16_vector("mtp.norm.weight", HIDDEN, 1.0, 9_103),
    ];
    // The block: every shape a TRUNK full-attention layer's, which is what
    // makes the draft step reuse `families/qwen/attn.rs` with no new kernel.
    ts.push(bf16_vector(
        "mtp.layers.0.input_layernorm.weight",
        HIDDEN,
        1.0,
        9_110,
    ));
    ts.push(bf16_vector(
        "mtp.layers.0.post_attention_layernorm.weight",
        HIDDEN,
        1.0,
        9_111,
    ));
    ts.push(bf16_matrix(
        "mtp.layers.0.self_attn.q_proj.weight",
        2 * q_out,
        HIDDEN,
        9_120,
    ));
    ts.push(bf16_matrix(
        "mtp.layers.0.self_attn.k_proj.weight",
        kv_out,
        HIDDEN,
        9_121,
    ));
    ts.push(bf16_matrix(
        "mtp.layers.0.self_attn.v_proj.weight",
        kv_out,
        HIDDEN,
        9_122,
    ));
    ts.push(bf16_matrix(
        "mtp.layers.0.self_attn.o_proj.weight",
        HIDDEN,
        q_out,
        9_123,
    ));
    ts.push(bf16_vector(
        "mtp.layers.0.self_attn.q_norm.weight",
        HEAD_DIM,
        1.0,
        9_124,
    ));
    ts.push(bf16_vector(
        "mtp.layers.0.self_attn.k_norm.weight",
        HEAD_DIM,
        1.0,
        9_125,
    ));
    ts.push(bf16_matrix(
        "mtp.layers.0.mlp.gate_proj.weight",
        INTER,
        HIDDEN,
        9_130,
    ));
    ts.push(bf16_matrix(
        "mtp.layers.0.mlp.up_proj.weight",
        INTER,
        HIDDEN,
        9_131,
    ));
    ts.push(bf16_matrix(
        "mtp.layers.0.mlp.down_proj.weight",
        HIDDEN,
        INTER,
        9_132,
    ));
    ts
}

/// The DFlash2 drafter's conv taps: rank 3, `[2 sides, 2 taps, hidden]`,
/// which no other tensor in any checkpoint here is. The rank is the whole
/// point of the helper -- a rank-2 stand-in would route down the projection
/// arm and the fixture could not see the rank-3 ingest at all.
pub(crate) fn bf16_base_kernel(name: &str, seed: u64) -> Tensor {
    let flat = bf16_vector(name, 2 * 2 * HIDDEN, 1.0, seed);
    Tensor {
        name: flat.name,
        dtype: flat.dtype,
        shape: vec![2, 2, HIDDEN as u64],
        bytes: flat.bytes,
    }
}

pub(crate) fn dflash_drafter_tensors(vocab: usize) -> Vec<Tensor> {
    let q_out = NUM_HEADS * HEAD_DIM;
    let kv_out = NUM_KV_HEADS * HEAD_DIM;
    let groups = HIDDEN / 16;
    let conv_rows = 2 * 2 * groups;
    let mut ts: Vec<Tensor> = Vec::with_capacity(6 + 15 * DFLASH_LAYERS);

    ts.push(bf16_matrix(
        "dflash.candidate_selector.hidden_projection.weight",
        DFLASH_RANK,
        HIDDEN,
        20_000,
    ));
    ts.push(bf16_matrix(
        "dflash.candidate_selector.predecessor_codebook",
        vocab,
        DFLASH_RANK,
        20_001,
    ));
    ts.push(bf16_matrix(
        "dflash.candidate_selector.successor_codebook",
        vocab,
        DFLASH_RANK,
        20_002,
    ));
    ts.push(bf16_matrix(
        "dflash.fc.weight",
        HIDDEN,
        DFLASH_LAYERS * HIDDEN,
        20_003,
    ));
    ts.push(bf16_vector(
        "dflash.hidden_norm.weight",
        HIDDEN,
        1.0,
        20_004,
    ));
    ts.push(bf16_vector("dflash.norm.weight", HIDDEN, 1.0, 20_005));

    for l in 0..DFLASH_LAYERS {
        let p = format!("dflash.layers.{l}");
        let seed = 21_000 + 100 * l as u64;
        ts.push(bf16_base_kernel(
            &format!("{p}.attention_conv.base_kernel"),
            seed,
        ));
        ts.push(bf16_matrix(
            &format!("{p}.attention_conv.kernel_projection.weight"),
            conv_rows,
            HIDDEN,
            seed + 1,
        ));
        ts.push(bf16_vector(
            &format!("{p}.input_layernorm.weight"),
            HIDDEN,
            1.0,
            seed + 2,
        ));
        ts.push(bf16_matrix(
            &format!("{p}.mlp.down_proj.weight"),
            HIDDEN,
            INTER,
            seed + 3,
        ));
        ts.push(bf16_matrix(
            &format!("{p}.mlp.gate_proj.weight"),
            INTER,
            HIDDEN,
            seed + 4,
        ));
        ts.push(bf16_matrix(
            &format!("{p}.mlp.up_proj.weight"),
            INTER,
            HIDDEN,
            seed + 5,
        ));
        ts.push(bf16_base_kernel(
            &format!("{p}.mlp_conv.base_kernel"),
            seed + 6,
        ));
        ts.push(bf16_matrix(
            &format!("{p}.mlp_conv.kernel_projection.weight"),
            conv_rows,
            HIDDEN,
            seed + 7,
        ));
        ts.push(bf16_vector(
            &format!("{p}.post_attention_layernorm.weight"),
            HIDDEN,
            1.0,
            seed + 8,
        ));
        ts.push(bf16_vector(
            &format!("{p}.self_attn.k_norm.weight"),
            HEAD_DIM,
            1.0,
            seed + 9,
        ));
        ts.push(bf16_matrix(
            &format!("{p}.self_attn.k_proj.weight"),
            kv_out,
            HIDDEN,
            seed + 10,
        ));
        ts.push(bf16_matrix(
            &format!("{p}.self_attn.o_proj.weight"),
            HIDDEN,
            q_out,
            seed + 11,
        ));
        ts.push(bf16_vector(
            &format!("{p}.self_attn.q_norm.weight"),
            HEAD_DIM,
            1.0,
            seed + 12,
        ));
        ts.push(bf16_matrix(
            &format!("{p}.self_attn.q_proj.weight"),
            q_out,
            HIDDEN,
            seed + 13,
        ));
        ts.push(bf16_matrix(
            &format!("{p}.self_attn.v_proj.weight"),
            kv_out,
            HIDDEN,
            seed + 14,
        ));
    }
    ts
}
