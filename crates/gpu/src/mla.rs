//! Host-side dispatch for the `deepseek2` MLA kernels in
//! `shaders/mla.metal`. Port-local (no Swift original exists -- Swift has
//! no MLA); every kernel's contract is the same-named function in
//! `turbospark_compute::mla`, which `tests/mla_parity.rs` holds each
//! dispatch to on real hardware.
//!
//! The compressed-cache design these serve is in `docs/DEEPSEEK2_PHASE0.md`:
//! one `[c ; k_pe]` row per token per layer, absorbed queries, V read from
//! the row itself.

use half::f16;
use metal::FunctionConstantValues;

use crate::bytes::{f32_bytes, half_slice_to_le_bytes, read_half_buffer, u32_bytes};
use crate::context::{dispatch_threads_3d, GpuError, MetalContext, PassEncoder};

/// One library for the whole file: the kernels here share no helper with
/// another shader (the Q8_0 block read is inlined), so no concat is needed
/// and the address-keyed pipeline cache gets one stable source.
const SOURCE: &str = include_str!("shaders/mla.metal");

const THREADS_PER_GROUP: u64 = 256;
const ROWS_PER_THREADGROUP: u64 = 8;

fn no_function_constants() -> FunctionConstantValues {
    FunctionConstantValues::new()
}

/// In-place `mla_kv_norm` over `rows` fused `[rank ; tail]` rows of
/// `row_stride` halves each. `rows == 1` is the decode shape (the token's
/// kv_a projection); a prefill micro-batch passes one row per token.
#[allow(clippy::too_many_arguments)]
pub fn encode_mla_kv_norm(
    context: &mut MetalContext,
    pass: &PassEncoder,
    data: (&metal::Buffer, u64),
    weight: (&metal::Buffer, u64),
    rows: u32,
    rank: u32,
    row_stride: u32,
    eps: f32,
) -> Result<(), GpuError> {
    if row_stride < rank {
        return Err(GpuError::PipelineCreate(format!(
            "mla_kv_norm row_stride {row_stride} cannot be below rank {rank}"
        )));
    }
    let pipeline = context.pipeline(SOURCE, "mla_kv_norm", &no_function_constants(), b"")?;
    pass.encode_threadgroups_3d(
        &pipeline,
        &[(data.0, 0, data.1), (weight.0, 1, weight.1)],
        &[
            (u32_bytes(&rank), 2),
            (f32_bytes(&eps), 3),
            (u32_bytes(&row_stride), 4),
        ],
        (rows.max(1) as u64, 1, 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}

/// Whole-buffer [`encode_mla_kv_norm`], for parity fixtures. The weight is
/// the RESIDENT byte form: BF16, little-endian u16 per element, exactly
/// what the transcode writes.
pub fn mla_kv_norm(
    context: &mut MetalContext,
    data: &[f16],
    weight_bf16: &[u16],
    rows: usize,
    rank: usize,
    row_stride: usize,
    eps: f32,
) -> Result<Vec<f16>, GpuError> {
    assert_eq!(data.len(), rows * row_stride);
    let buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(data));
    let w_bytes: Vec<u8> = weight_bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
    let w_buffer = context.new_buffer_with_data(&w_bytes);
    let pipeline = context.pipeline(SOURCE, "mla_kv_norm", &no_function_constants(), b"")?;
    dispatch_threads_3d(
        context,
        &pipeline,
        &[(&buffer, 0), (&w_buffer, 1)],
        &[
            (u32_bytes(&(rank as u32)), 2),
            (f32_bytes(&eps), 3),
            (u32_bytes(&(row_stride as u32)), 4),
        ],
        // dispatch_threads takes TOTAL threads, not threadgroup counts.
        ((rows.max(1) as u64) * THREADS_PER_GROUP, 1, 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(read_half_buffer(&buffer, data.len()))
}

/// In-place `mla_rope_q_pe`: rotates the trailing `rotary_dim` of each
/// `head_dim`-wide head, window starting at `window_offset`. `rows` is the
/// token count (one q row per token, `heads * head_dim` halves each).
#[allow(clippy::too_many_arguments)]
pub fn encode_mla_rope_q_pe(
    context: &mut MetalContext,
    pass: &PassEncoder,
    data: (&metal::Buffer, u64),
    frequencies: (&metal::Buffer, u64),
    rows: u32,
    num_heads: u32,
    head_dim: u32,
    window_offset: u32,
    rotary_dim: u32,
    position: u32,
    mscale: f32,
) -> Result<(), GpuError> {
    if rotary_dim % 2 != 0 || window_offset + rotary_dim > head_dim {
        return Err(GpuError::PipelineCreate(format!(
            "mla_rope_q_pe: rotary {rotary_dim} / window {window_offset} / head {head_dim} \
             is not a valid even window"
        )));
    }
    let pipeline = context.pipeline(SOURCE, "mla_rope_q_pe", &no_function_constants(), b"")?;
    pass.encode_threadgroups_3d(
        &pipeline,
        &[(data.0, 0, data.1), (frequencies.0, 1, frequencies.1)],
        &[
            (u32_bytes(&position), 2),
            (u32_bytes(&head_dim), 3),
            (u32_bytes(&window_offset), 4),
            (f32_bytes(&mscale), 5),
            (u32_bytes(&rotary_dim), 6),
        ],
        (
            (rotary_dim / 2).max(1) as u64,
            num_heads.max(1) as u64,
            rows.max(1) as u64,
        ),
        (1, 1, 1),
    );
    Ok(())
}

/// Whole-buffer [`encode_mla_rope_q_pe`], for parity fixtures. One q row,
/// `heads * head_dim` halves.
#[allow(clippy::too_many_arguments)]
pub fn mla_rope_q_pe(
    context: &mut MetalContext,
    data: &[f16],
    frequencies: &[f32],
    num_heads: usize,
    head_dim: usize,
    window_offset: usize,
    rotary_dim: usize,
    position: u32,
    mscale: f32,
) -> Result<Vec<f16>, GpuError> {
    let buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(data));
    let f_bytes: Vec<u8> = frequencies.iter().flat_map(|f| f.to_le_bytes()).collect();
    let f_buffer = context.new_buffer_with_data(&f_bytes);
    let pipeline = context.pipeline(SOURCE, "mla_rope_q_pe", &no_function_constants(), b"")?;
    dispatch_threads_3d(
        context,
        &pipeline,
        &[(&buffer, 0), (&f_buffer, 1)],
        &[
            (u32_bytes(&position), 2),
            (u32_bytes(&(head_dim as u32)), 3),
            (u32_bytes(&(window_offset as u32)), 4),
            (f32_bytes(&mscale), 5),
            (u32_bytes(&(rotary_dim as u32)), 6),
        ],
        ((rotary_dim as u64 / 2).max(1), num_heads.max(1) as u64, 1),
        (1, 1, 1),
    );
    Ok(read_half_buffer(&buffer, data.len()))
}

/// Encoder-level `mla_absorb_q_q8_0`: `W` is the whole kv_b resident tensor
/// (rows of `nope` and `v_dim` halves interleave per head), `q` the nope
/// halves, `out` receives `heads * kv_lora` halves. Grid
/// `(kv_lora/8, heads)` threadgroups of 256.
#[allow(clippy::too_many_arguments)]
pub fn encode_mla_absorb_q(
    context: &mut MetalContext,
    pass: &PassEncoder,
    weights: (&metal::Buffer, u64),
    q: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    heads: u32,
    nope: u32,
    kv_lora: u32,
    v_dim: u32,
    rope_dim: u32,
) -> Result<(), GpuError> {
    if nope % 32 != 0 || kv_lora % 8 != 0 {
        return Err(GpuError::PipelineCreate(format!(
            "mla_absorb_q needs nope {nope} divisible by 32 and kv_lora {kv_lora} by 8"
        )));
    }
    let pipeline = context.pipeline(SOURCE, "mla_absorb_q_q8_0", &no_function_constants(), b"")?;
    pass.encode_threadgroups_3d(
        &pipeline,
        &[(weights.0, 0, weights.1), (q.0, 1, q.1), (out.0, 2, out.1)],
        &[
            (u32_bytes(&heads), 3),
            (u32_bytes(&nope), 4),
            (u32_bytes(&kv_lora), 5),
            (u32_bytes(&v_dim), 6),
            (u32_bytes(&rope_dim), 7),
        ],
        // The grid covers the FUSED row: kv_lora dot rows plus the pe tail.
        (
            ((kv_lora + rope_dim) as u64 / ROWS_PER_THREADGROUP).max(1),
            heads.max(1) as u64,
            1,
        ),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}

/// Whole-buffer [`encode_mla_absorb_q`], for parity fixtures. `w_rows` are
/// the dequant source byte runs for ALL heads back to back
/// (`heads * (nope + v_dim)` rows of `q8_0_row_bytes(nope)`), exactly how
/// the resident tensor stores them.
#[allow(clippy::too_many_arguments)]
pub fn mla_absorb_q(
    context: &mut MetalContext,
    w_rows: &[&[u8]],
    q: &[f16],
    heads: usize,
    nope: usize,
    kv_lora: usize,
    v_dim: usize,
    rope_dim: usize,
) -> Result<Vec<f16>, GpuError> {
    // kv_b rows are KV_LORA wide: the latent is the row axis of the stored
    // matrix, and the absorb reduces across ROWS.
    let row_bytes = kv_lora / 32 * 34;
    let mut w_bytes = Vec::with_capacity(w_rows.len() * row_bytes);
    for row in w_rows {
        assert_eq!(row.len(), row_bytes);
        w_bytes.extend_from_slice(row);
    }
    assert_eq!(q.len(), heads * (nope + rope_dim));
    let w_buffer = context.new_buffer_with_data(&w_bytes);
    let q_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(q));
    let out_rows = kv_lora + rope_dim;
    let out_buffer = context.new_output_buffer((heads * out_rows * 2) as u64);
    let pipeline = context.pipeline(SOURCE, "mla_absorb_q_q8_0", &no_function_constants(), b"")?;
    dispatch_threads_3d(
        context,
        &pipeline,
        &[(&w_buffer, 0), (&q_buffer, 1), (&out_buffer, 2)],
        &[
            (u32_bytes(&(heads as u32)), 3),
            (u32_bytes(&(nope as u32)), 4),
            (u32_bytes(&(kv_lora as u32)), 5),
            (u32_bytes(&(v_dim as u32)), 6),
            (u32_bytes(&(rope_dim as u32)), 7),
        ],
        // dispatch_threads takes TOTAL threads; the grid covers the FUSED row.
        (
            (out_rows as u64 / ROWS_PER_THREADGROUP).max(1) * THREADS_PER_GROUP,
            heads.max(1) as u64,
            1,
        ),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(read_half_buffer(&out_buffer, heads * out_rows))
}

/// Encoder-level `mla_attention_decode`: MQA over the compressed rows.
/// `cache` is the K buffer; row `t` lives at `cache_offset + t *
/// cache_row * 2` bytes. `out` receives `heads * kv_lora` halves.
#[allow(clippy::too_many_arguments)]
pub fn encode_mla_attention_decode(
    context: &mut MetalContext,
    pass: &PassEncoder,
    q: (&metal::Buffer, u64),
    cache: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    heads: u32,
    cache_row: u32,
    kv_lora: u32,
    seq_len: u32,
    scale: f32,
) -> Result<(), GpuError> {
    if kv_lora != 512 {
        return Err(GpuError::PipelineCreate(format!(
            "mla_attention_decode fixes its accumulator at kv_lora 512 (2 halves x 256 \
             threads); got {kv_lora}"
        )));
    }
    let pipeline = context.pipeline(
        SOURCE,
        "mla_attention_decode",
        &no_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups_3d(
        &pipeline,
        &[(q.0, 0, q.1), (cache.0, 1, cache.1), (out.0, 2, out.1)],
        &[
            (u32_bytes(&heads), 3),
            (u32_bytes(&cache_row), 4),
            (u32_bytes(&kv_lora), 5),
            (u32_bytes(&seq_len), 6),
            (f32_bytes(&scale), 7),
        ],
        (heads.max(1) as u64, 1, 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}

/// Whole-buffer [`encode_mla_attention_decode`], for parity fixtures.
#[allow(clippy::too_many_arguments)]
pub fn mla_attention_decode(
    context: &mut MetalContext,
    q: &[f16],
    cache: &[f16],
    heads: usize,
    cache_row: usize,
    kv_lora: usize,
    seq_len: usize,
    scale: f32,
) -> Result<Vec<f16>, GpuError> {
    assert_eq!(q.len(), heads * cache_row);
    assert_eq!(cache.len(), seq_len * cache_row);
    let q_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(q));
    let cache_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(cache));
    let out_buffer = context.new_output_buffer((heads * kv_lora * 2) as u64);
    let pipeline = context.pipeline(
        SOURCE,
        "mla_attention_decode",
        &no_function_constants(),
        b"",
    )?;
    dispatch_threads_3d(
        context,
        &pipeline,
        &[(&q_buffer, 0), (&cache_buffer, 1), (&out_buffer, 2)],
        &[
            (u32_bytes(&(heads as u32)), 3),
            (u32_bytes(&(cache_row as u32)), 4),
            (u32_bytes(&(kv_lora as u32)), 5),
            (u32_bytes(&(seq_len as u32)), 6),
            (f32_bytes(&scale), 7),
        ],
        // dispatch_threads takes TOTAL threads.
        ((heads.max(1) as u64) * THREADS_PER_GROUP, 1, 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(read_half_buffer(&out_buffer, heads * kv_lora))
}

/// Encoder-level `mla_v_combine_q8_0`: the mirror of the absorb kernel.
#[allow(clippy::too_many_arguments)]
pub fn encode_mla_v_combine(
    context: &mut MetalContext,
    pass: &PassEncoder,
    weights: (&metal::Buffer, u64),
    attn: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    heads: u32,
    nope: u32,
    kv_lora: u32,
    v_dim: u32,
) -> Result<(), GpuError> {
    if kv_lora % 32 != 0 || v_dim % 8 != 0 {
        return Err(GpuError::PipelineCreate(format!(
            "mla_v_combine needs kv_lora {kv_lora} divisible by 32 and v_dim {v_dim} by 8"
        )));
    }
    let pipeline = context.pipeline(SOURCE, "mla_v_combine_q8_0", &no_function_constants(), b"")?;
    pass.encode_threadgroups_3d(
        &pipeline,
        &[
            (weights.0, 0, weights.1),
            (attn.0, 1, attn.1),
            (out.0, 2, out.1),
        ],
        &[
            (u32_bytes(&heads), 3),
            (u32_bytes(&nope), 4),
            (u32_bytes(&kv_lora), 5),
            (u32_bytes(&v_dim), 6),
        ],
        (
            (v_dim as u64 / ROWS_PER_THREADGROUP).max(1),
            heads.max(1) as u64,
            1,
        ),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}

/// Whole-buffer [`encode_mla_v_combine`], for parity fixtures.
pub fn mla_v_combine(
    context: &mut MetalContext,
    w_rows: &[&[u8]],
    attn: &[f16],
    heads: usize,
    nope: usize,
    kv_lora: usize,
    v_dim: usize,
) -> Result<Vec<f16>, GpuError> {
    let row_bytes = kv_lora / 32 * 34;
    let mut w_bytes = Vec::with_capacity(w_rows.len() * row_bytes);
    for row in w_rows {
        assert_eq!(row.len(), row_bytes);
        w_bytes.extend_from_slice(row);
    }
    let w_buffer = context.new_buffer_with_data(&w_bytes);
    let a_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(attn));
    let out_buffer = context.new_output_buffer((heads * v_dim * 2) as u64);
    let pipeline = context.pipeline(SOURCE, "mla_v_combine_q8_0", &no_function_constants(), b"")?;
    dispatch_threads_3d(
        context,
        &pipeline,
        &[(&w_buffer, 0), (&a_buffer, 1), (&out_buffer, 2)],
        &[
            (u32_bytes(&(heads as u32)), 3),
            (u32_bytes(&(nope as u32)), 4),
            (u32_bytes(&(kv_lora as u32)), 5),
            (u32_bytes(&(v_dim as u32)), 6),
        ],
        // dispatch_threads takes TOTAL threads.
        (
            (v_dim as u64 / ROWS_PER_THREADGROUP).max(1) * THREADS_PER_GROUP,
            heads.max(1) as u64,
            1,
        ),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(read_half_buffer(&out_buffer, heads * v_dim))
}

/// Encoder-level `mla_cache_write`: copy `count` halves from a scratch row
/// into the cache slot at `dst_offset`.
pub fn encode_mla_cache_write(
    context: &mut MetalContext,
    pass: &PassEncoder,
    src: (&metal::Buffer, u64),
    dst: (&metal::Buffer, u64),
    count: u32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(SOURCE, "mla_cache_write", &no_function_constants(), b"")?;
    pass.encode_threadgroups_3d(
        &pipeline,
        &[(src.0, 0, src.1), (dst.0, 1, dst.1)],
        &[(u32_bytes(&count), 2)],
        (count.div_ceil(THREADS_PER_GROUP as u32) as u64, 1, 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}
