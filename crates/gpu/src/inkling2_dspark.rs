//! Host-side dispatch for `shaders/inkling2_dspark.metal`, the Inkling 2
//! DSpark block drafter's own ops. PORT-LOCAL; contract in the shader header.
//! The drafter proposes and the target's verify decides, so these are held
//! to the CPU reference by proposal agreement (`inkling2_dspark_accept`), not
//! bit for bit.

use crate::bytes::u32_bytes;
use crate::{GpuError, MetalContext, PassEncoder};
use metal::FunctionConstantValues;

const SOURCE: &str = include_str!("shaders/inkling2_dspark.metal");

type Buf<'a> = (&'a metal::Buffer, u64);

/// The widest Markov latent the logits kernel stages in threadgroup memory.
pub const DSPARK_MAX_RANK: u32 = 256;
/// Vocabulary entries per partial of [`encode_dspark_markov_logits`].
pub const DSPARK_LOGITS_PER_GROUP: u32 = 256;

fn pipeline(
    context: &mut MetalContext,
    kernel: &'static str,
) -> Result<metal::ComputePipelineState, GpuError> {
    context.pipeline(SOURCE, kernel, &FunctionConstantValues::new(), b"")
}

fn bad(msg: &str) -> GpuError {
    GpuError::InvalidInput(format!("dspark: {msg}"))
}

fn binds<'a>(bufs: &[Buf<'a>]) -> Vec<(&'a metal::Buffer, u64, u64)> {
    bufs.iter()
        .enumerate()
        .map(|(i, &(b, o))| (b, i as u64, o))
        .collect()
}

/// `out[r] = rms(sum_k alpha[k] * taps[r][k], w)` over 4 taps of `h`.
#[allow(clippy::too_many_arguments)]
pub fn encode_dspark_tap_mix_norm(
    context: &mut MetalContext,
    pass: &PassEncoder,
    taps: Buf<'_>,
    w: Buf<'_>,
    out: Buf<'_>,
    alpha: [f32; 4],
    h: u32,
    tap_stride: u32,
    eps: f32,
    rows: u32,
) -> Result<(), GpuError> {
    if h == 0 || rows == 0 || tap_stride < 4 * h {
        return Err(bad("tap mix needs h, rows > 0 and a stride of 4 taps"));
    }
    let a: Vec<u8> = alpha.iter().flat_map(|v| v.to_le_bytes()).collect();
    let n = binds(&[taps, w, out]);
    let p = pipeline(context, "dspark_tap_mix_norm")?;
    pass.encode_threadgroups(
        &p,
        &n,
        &[
            (&a, 3),
            (u32_bytes(&h), 4),
            (u32_bytes(&tap_stride), 5),
            (&eps.to_le_bytes(), 6),
        ],
        u64::from(rows),
        256,
    );
    Ok(())
}

/// RMS norm of F32 rows `[rows][h]` with a BF16 weight, FP16 out.
#[allow(clippy::too_many_arguments)]
pub fn encode_dspark_rms_f32(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: Buf<'_>,
    w: Buf<'_>,
    out: Buf<'_>,
    h: u32,
    eps: f32,
    rows: u32,
) -> Result<(), GpuError> {
    let p = pipeline(context, "dspark_rms_f32")?;
    pass.encode_threadgroups(
        &p,
        &binds(&[x, w, out]),
        &[(u32_bytes(&h), 3), (&eps.to_le_bytes(), 4)],
        u64::from(rows),
        256,
    );
    Ok(())
}

/// F32 block input `[rows][h]`: row 0 the BF16-rounded `emb`, the rest `mask`.
pub fn encode_dspark_init_x(
    context: &mut MetalContext,
    pass: &PassEncoder,
    emb: Buf<'_>,
    mask: Buf<'_>,
    x: Buf<'_>,
    h: u32,
    rows: u32,
) -> Result<(), GpuError> {
    let p = pipeline(context, "dspark_init_x")?;
    pass.encode_threads_3d(
        &p,
        &binds(&[emb, mask, x]),
        &[(u32_bytes(&h), 3)],
        (u64::from(h), u64::from(rows), 1),
        (256, 1, 1),
    );
    Ok(())
}

/// `x = bf16(x) + d` over `n` F32 / FP16 elements.
pub fn encode_dspark_residual(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: Buf<'_>,
    d: Buf<'_>,
    n: u32,
) -> Result<(), GpuError> {
    let p = pipeline(context, "dspark_residual")?;
    pass.encode_threads_3d(
        &p,
        &binds(&[x, d]),
        &[(u32_bytes(&n), 2)],
        (u64::from(n), 1, 1),
        (256, 1, 1),
    );
    Ok(())
}

/// `act = silu(g) * u` over `n` FP16 elements.
pub fn encode_dspark_silu_mul(
    context: &mut MetalContext,
    pass: &PassEncoder,
    g: Buf<'_>,
    u: Buf<'_>,
    act: Buf<'_>,
    n: u32,
) -> Result<(), GpuError> {
    let p = pipeline(context, "dspark_silu_mul")?;
    pass.encode_threads_3d(
        &p,
        &binds(&[g, u, act]),
        &[(u32_bytes(&n), 3)],
        (u64::from(n), 1, 1),
        (256, 1, 1),
    );
    Ok(())
}

/// NeoX RoPE in place on `[rows][n_heads][hd]`, row `r` at `pos0 + r`.
#[allow(clippy::too_many_arguments)]
pub fn encode_dspark_rope_rows(
    context: &mut MetalContext,
    pass: &PassEncoder,
    x: Buf<'_>,
    n_heads: u32,
    hd: u32,
    pos0: u32,
    theta: f32,
    rows: u32,
) -> Result<(), GpuError> {
    if hd % 2 != 0 {
        return Err(bad("rope needs an even head_dim"));
    }
    let p = pipeline(context, "dspark_rope_rows")?;
    pass.encode_threads_3d(
        &p,
        &binds(&[x]),
        &[
            (u32_bytes(&n_heads), 1),
            (u32_bytes(&hd), 2),
            (u32_bytes(&pos0), 3),
            (&theta.to_le_bytes(), 4),
        ],
        (u64::from(hd / 2), u64::from(n_heads), u64::from(rows)),
        (u64::from(hd / 2).min(64), 1, 1),
    );
    Ok(())
}

/// Context K (roped) and V rows for positions `p0 ..` into ring caches of
/// `cap` rows.
#[allow(clippy::too_many_arguments)]
pub fn encode_dspark_ctx_store(
    context: &mut MetalContext,
    pass: &PassEncoder,
    k: Buf<'_>,
    v: Buf<'_>,
    kc: Buf<'_>,
    vc: Buf<'_>,
    kv_heads: u32,
    hd: u32,
    p0: u32,
    cap: u32,
    theta: f32,
    rows: u32,
) -> Result<(), GpuError> {
    if cap == 0 || rows > cap {
        return Err(bad("ctx store needs rows <= cap"));
    }
    let p = pipeline(context, "dspark_ctx_store")?;
    pass.encode_threads_3d(
        &p,
        &binds(&[k, v, kc, vc]),
        &[
            (u32_bytes(&kv_heads), 4),
            (u32_bytes(&hd), 5),
            (u32_bytes(&p0), 6),
            (u32_bytes(&cap), 7),
            (&theta.to_le_bytes(), 8),
        ],
        (u64::from(hd / 2), u64::from(kv_heads), u64::from(rows)),
        (u64::from(hd / 2).min(64), 1, 1),
    );
    Ok(())
}

/// Shape of one drafter attention dispatch.
#[derive(Debug, Clone, Copy)]
pub struct DsparkAttnShape {
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// Ring rows of the context cache.
    pub cap: u32,
    pub n_block: u32,
}

/// The block's attention: context `[first, t)` from the ring plus every block
/// row, bidirectional.
#[allow(clippy::too_many_arguments)]
pub fn encode_dspark_attention(
    context: &mut MetalContext,
    pass: &PassEncoder,
    q: Buf<'_>,
    kc: Buf<'_>,
    vc: Buf<'_>,
    kb: Buf<'_>,
    vb: Buf<'_>,
    out: Buf<'_>,
    first: u32,
    t: u32,
    s: DsparkAttnShape,
) -> Result<(), GpuError> {
    if s.kv_heads == 0
        || s.heads % s.kv_heads != 0
        || s.head_dim % 32 != 0
        || s.head_dim > 128
        || first > t
        || t - first > s.cap
        || s.n_block == 0
    {
        return Err(bad(&format!("attention shape {s:?}, range {first}..{t}")));
    }
    let p = pipeline(context, "dspark_attention")?;
    pass.encode_threadgroups_3d(
        &p,
        &binds(&[q, kc, vc, kb, vb, out]),
        &[
            (u32_bytes(&first), 6),
            (u32_bytes(&t), 7),
            (u32_bytes(&s.cap), 8),
            (u32_bytes(&s.n_block), 9),
            (u32_bytes(&s.heads), 10),
            (u32_bytes(&s.kv_heads), 11),
            (u32_bytes(&s.head_dim), 12),
        ],
        (u64::from(s.heads), u64::from(s.n_block), 1),
        (256, 1, 1),
    );
    Ok(())
}

/// One Markov step's RNN input `[state | w1[last] | hidden | feature]`.
#[allow(clippy::too_many_arguments)]
pub fn encode_dspark_markov_input(
    context: &mut MetalContext,
    pass: &PassEncoder,
    state: Buf<'_>,
    w1: Buf<'_>,
    last: Buf<'_>,
    hidden: Buf<'_>,
    feature: Buf<'_>,
    input: Buf<'_>,
    rank: u32,
    h: u32,
) -> Result<(), GpuError> {
    let p = pipeline(context, "dspark_markov_input")?;
    pass.encode_threads_3d(
        &p,
        &binds(&[state, w1, last, hidden, feature, input]),
        &[(u32_bytes(&rank), 6), (u32_bytes(&h), 7)],
        (u64::from(2 * rank + 2 * h), 1, 1),
        (256, 1, 1),
    );
    Ok(())
}

/// The Markov gates: updates `state` in place, writes `latent`.
pub fn encode_dspark_markov_gate(
    context: &mut MetalContext,
    pass: &PassEncoder,
    g: Buf<'_>,
    bias: Buf<'_>,
    state: Buf<'_>,
    latent: Buf<'_>,
    rank: u32,
) -> Result<(), GpuError> {
    let p = pipeline(context, "dspark_markov_gate")?;
    pass.encode_threads_3d(
        &p,
        &binds(&[g, bias, state, latent]),
        &[(u32_bytes(&rank), 4)],
        (u64::from(rank), 1, 1),
        (256, 1, 1),
    );
    Ok(())
}

/// Corrected logits and per-group argmax partials over `n_vocab` entries;
/// `fold_t` is `[rank][stride]`. Returns the number of partials.
#[allow(clippy::too_many_arguments)]
pub fn encode_dspark_markov_logits(
    context: &mut MetalContext,
    pass: &PassEncoder,
    base: Buf<'_>,
    fold_b: Buf<'_>,
    fold_t: Buf<'_>,
    latent: Buf<'_>,
    part_v: Buf<'_>,
    part_i: Buf<'_>,
    n_vocab: u32,
    stride: u32,
    rank: u32,
) -> Result<u32, GpuError> {
    if rank == 0 || rank > DSPARK_MAX_RANK || stride < n_vocab {
        return Err(bad("markov logits: rank or stride"));
    }
    let groups = n_vocab.div_ceil(DSPARK_LOGITS_PER_GROUP);
    let p = pipeline(context, "dspark_markov_logits")?;
    pass.encode_threadgroups(
        &p,
        &binds(&[base, fold_b, fold_t, latent, part_v, part_i]),
        &[
            (u32_bytes(&n_vocab), 6),
            (u32_bytes(&stride), 7),
            (u32_bytes(&rank), 8),
        ],
        u64::from(groups),
        256,
    );
    Ok(groups)
}

/// Reduces the partials to the step's token: `last[0]` and `proposals[step]`.
#[allow(clippy::too_many_arguments)]
pub fn encode_dspark_markov_argmax(
    context: &mut MetalContext,
    pass: &PassEncoder,
    part_v: Buf<'_>,
    part_i: Buf<'_>,
    last: Buf<'_>,
    proposals: Buf<'_>,
    n_part: u32,
    step: u32,
) -> Result<(), GpuError> {
    let p = pipeline(context, "dspark_markov_argmax")?;
    pass.encode_threadgroups(
        &p,
        &binds(&[part_v, part_i, last, proposals]),
        &[(u32_bytes(&n_part), 4), (u32_bytes(&step), 5)],
        1,
        256,
    );
    Ok(())
}

/// FP16 `dst[cols][rows]` from BF16 `src[rows][cols]`.
pub fn encode_dspark_transpose_bf16(
    context: &mut MetalContext,
    pass: &PassEncoder,
    src: Buf<'_>,
    dst: Buf<'_>,
    rows: u32,
    cols: u32,
) -> Result<(), GpuError> {
    let p = pipeline(context, "dspark_transpose_bf16")?;
    pass.encode_threads_3d(
        &p,
        &binds(&[src, dst]),
        &[(u32_bytes(&rows), 2), (u32_bytes(&cols), 3)],
        (u64::from(cols), u64::from(rows), 1),
        (64, 4, 1),
    );
    Ok(())
}

/// FP16 copy of a BF16 vector.
pub fn encode_dspark_bf16_to_half(
    context: &mut MetalContext,
    pass: &PassEncoder,
    src: Buf<'_>,
    dst: Buf<'_>,
    n: u32,
) -> Result<(), GpuError> {
    let p = pipeline(context, "dspark_bf16_to_half")?;
    pass.encode_threads_3d(
        &p,
        &binds(&[src, dst]),
        &[(u32_bytes(&n), 2)],
        (u64::from(n), 1, 1),
        (256, 1, 1),
    );
    Ok(())
}
