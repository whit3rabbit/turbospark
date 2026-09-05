//! Host-side dispatch for `shaders/attention_indexed.metal`: single-token
//! decode attention over an EXPLICIT LIST of key/value positions. PORT-LOCAL
//! (no Swift counterpart), built for `qwen4_exp`'s QSA attention application
//! (`docs/QWEN4_PHASE0.md` section 5, `docs/QWEN4_EXP.md`'s QSA sections).
//!
//! Pass 1 is `attention_decode_indexed_partial`, which is
//! `attention_decode_partial` with its position range replaced by a walk
//! over `positions[i]` and nothing else changed (the shader header says why
//! that restraint is the whole design). Pass 2 is attention.metal's OWN
//! `attention_decode_combine`, unchanged, reading the same `(m, d, o)`
//! partial layout out of the same [`AttentionScratch`] the dense path uses.
//! Chunking follows [`chunks_for`] over the LIST LENGTH, so with the identity
//! list the dispatch is the dense dispatch and `tests/attention_indexed_parity.rs`
//! asserts the outputs are bit-identical.
//!
//! Matches `turbospark_compute::indexed_attention` (a gather followed by the
//! dense CPU reference) within FP16 tolerance.

use half::f16;

use crate::attention_decode::{
    attention_constants_key, attention_function_constants, chunks_for, AttentionScratch,
    SOURCE as ATTENTION_SOURCE,
};
use crate::bytes::{f32_bytes, half_slice_to_le_bytes, read_half_buffer, u32_bytes};
use crate::context::{GpuError, MetalContext, PassEncoder};

const SOURCE: &str = include_str!("shaders/attention_indexed.metal");
const THREADS_PER_GROUP: u64 = 256; // kIdxAttnThreads, == attention.metal's kAttnThreads.

/// `Q: [num_q_heads, head_dim]` at a `(buffer, byte offset)` view; `K`/`V`
/// read in place from LINEAR `[stored_tokens, num_kv_heads, head_dim]`
/// buffers (a `KvCacheManager` full layer's, offset 0); `positions` is a
/// `[n_sel]` `u32` list of the rows to attend over, each `< stored_tokens`;
/// output `[num_q_heads, head_dim]` FP16 to `out`.
///
/// No ring addressing: the one family this serves has no sliding-window
/// layers, and a ring would need the list's entries mapped through
/// `p % cap` on the host, which nothing asks for yet.
///
/// The host cannot cheaply check every list entry against `stored_tokens`
/// on the GPU side; it checks what it can (the K/V buffers hold at least
/// `n_sel` rows, since a list longer than the storage cannot be valid) and
/// the caller that builds the list owns the per-entry bound.
#[allow(clippy::too_many_arguments)]
pub fn encode_attention_decode_indexed(
    context: &mut MetalContext,
    pass: &PassEncoder,
    q: (&metal::Buffer, u64),
    k_buffer: &metal::Buffer,
    v_buffer: &metal::Buffer,
    positions: (&metal::Buffer, u64),
    n_sel: u32,
    scratch: &AttentionScratch,
    out: (&metal::Buffer, u64),
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    scale: f32,
) -> Result<(), GpuError> {
    assert_eq!(num_q_heads % num_kv_heads, 0);
    assert!(
        n_sel > 0,
        "indexed attention needs at least one selected position"
    );
    assert!(
        positions.0.length() >= positions.1 + n_sel as u64 * 4,
        "positions buffer too small for n_sel"
    );
    let min_kv_bytes = (n_sel * num_kv_heads * head_dim) as u64 * 2;
    assert!(k_buffer.length() >= min_kv_bytes, "K buffer too small");
    assert!(v_buffer.length() >= min_kv_bytes, "V buffer too small");

    let num_chunks = chunks_for(n_sel);
    // Ceiling division, as the dense host does: the LAST chunk is the short
    // one, and no list entry falls past the final chunk's end.
    let chunk_len = n_sel.div_ceil(num_chunks);

    let partial_pipeline = context.pipeline(
        SOURCE,
        "attention_decode_indexed_partial",
        &metal::FunctionConstantValues::new(),
        b"",
    )?;
    pass.encode_threadgroups(
        &partial_pipeline,
        &[
            (q.0, 0, q.1),
            (k_buffer, 1, 0),
            (v_buffer, 2, 0),
            (&scratch.m, 3, 0),
            (&scratch.d, 4, 0),
            (&scratch.o, 5, 0),
            (positions.0, 9, positions.1),
        ],
        &[
            (u32_bytes(&head_dim), 6),
            (u32_bytes(&num_q_heads), 7),
            (u32_bytes(&num_kv_heads), 8),
            (u32_bytes(&n_sel), 10),
            (u32_bytes(&chunk_len), 11),
            (u32_bytes(&num_chunks), 12),
            (f32_bytes(&scale), 13),
        ],
        (num_q_heads * num_chunks) as u64,
        THREADS_PER_GROUP,
    );

    // attention.metal's combine, specialized exactly as the dense path
    // specializes it for this chunk count (its `FC_ATTN_NUM_CHUNKS` is read
    // unconditionally, and the key carries it -- `attention_decode.rs`'s
    // own doc on why a shared key would be a silent wrong pipeline). No
    // ring, no sinks: neither exists on the family this serves.
    let constants = attention_function_constants(scale, 0, num_chunks, false);
    let constants_key = attention_constants_key(scale, 0, num_chunks, false);
    let combine_pipeline = context.pipeline(
        ATTENTION_SOURCE,
        "attention_decode_combine",
        &constants,
        &constants_key,
    )?;
    pass.encode_threadgroups(
        &combine_pipeline,
        &[
            (&scratch.m, 0, 0),
            (&scratch.d, 1, 0),
            (&scratch.o, 2, 0),
            (out.0, 3, out.1),
        ],
        &[(u32_bytes(&head_dim), 4), (u32_bytes(&num_chunks), 5)],
        num_q_heads as u64,
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// Slice-taking entry for parity tests, mirroring `attention_decode`:
/// uploads Q/K/V and the position list, runs both passes, reads the result
/// back. `k`/`v` are `[seq_len, num_kv_heads, head_dim]`; every entry of
/// `positions` must be `< seq_len`.
#[allow(clippy::too_many_arguments)]
pub fn attention_decode_indexed(
    context: &mut MetalContext,
    q: &[f16],
    k: &[f16],
    v: &[f16],
    positions: &[u32],
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    scale: f32,
) -> Result<Vec<f16>, GpuError> {
    assert_eq!(q.len(), (num_q_heads * head_dim) as usize);
    assert_eq!(k.len(), v.len());
    let row = (num_kv_heads * head_dim) as usize;
    assert_eq!(k.len() % row, 0);
    let seq_len = k.len() / row;
    assert!(
        positions.iter().all(|&p| (p as usize) < seq_len),
        "every selected position must be inside the sequence"
    );

    let q_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(q));
    let k_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(k));
    let v_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(v));
    let positions_buffer = context.new_buffer_with_data(positions);
    let scratch = AttentionScratch::new(context, num_q_heads, head_dim);
    let out_buffer = context.new_output_buffer((num_q_heads * head_dim) as u64 * 2);

    let pass = context.begin_pass();
    encode_attention_decode_indexed(
        context,
        &pass,
        (&q_buffer, 0),
        &k_buffer,
        &v_buffer,
        (&positions_buffer, 0),
        positions.len() as u32,
        &scratch,
        (&out_buffer, 0),
        head_dim,
        num_q_heads,
        num_kv_heads,
        scale,
    )?;
    pass.commit_and_wait();
    Ok(read_half_buffer(
        &out_buffer,
        (num_q_heads * head_dim) as usize,
    ))
}
