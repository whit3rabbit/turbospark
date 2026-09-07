//! Runs the TurboQuant decode attention kernels
//! (`attention_decode_partial_tq` + `attention_decode_combine_tq`, and the
//! indexed sparse pass-1) on real Metal hardware and checks them against
//! `turbospark_compute::kv_quant_attention::causal_attention_tq`.
#![cfg(target_os = "macos")]

use half::f16;
use model_io::KvQuant;
use turbospark_compute::kv_quant::{quantize_row, QuantizedRow};
use turbospark_compute::kv_quant::{KEY_SEED, VALUE_SEED};
use turbospark_compute::kv_quant_attention::{causal_attention_tq, TqTables};
use turbospark_gpu::{
    encode_attention_decode_indexed_tq, encode_attention_decode_tq, KvQuantTables, MetalContext,
    TqAttentionScratch,
};

/// Deterministic, INDEPENDENT pseudo-random values in `[-1, 1)` (splitmix64
/// on the flat index) -- `attention_indexed_parity.rs`'s `unit()`, copied
/// rather than shared because that file's own header explains why a
/// sinusoid over the flat index (this file's `build_fixture` pattern,
/// `sin(seed * 7 + d * 1.31 + 0.5)`) makes each row a small rotation of its
/// neighbour: fine for a full-causal parity case with nothing to tell
/// apart, and NOT fine for an indexed test that has to distinguish
/// `positions[i]` from row `i`.
fn unit(i: usize, salt: u64) -> f32 {
    let mut z = (i as u64)
        .wrapping_add(salt)
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

struct IndexedFixture {
    q: Vec<f32>,
    k_rows: Vec<QuantizedRow>,
    v_rows: Vec<QuantizedRow>,
    k_tables_cpu: TqTables,
    v_tables_cpu: TqTables,
}

/// Independent-row TQ fixture, keys at magnitude 3 against unit queries
/// (`attention_indexed_parity.rs`'s fixture doc: this is what gives a
/// wrong row set an O(0.1) gap rather than one buried under TQ's own
/// quantization error).
fn build_indexed_fixture(
    dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
    k_bits: u8,
    v_bits: u8,
) -> IndexedFixture {
    let k_tables_cpu = TqTables::new(dim, k_bits, KEY_SEED);
    let v_tables_cpu = TqTables::new(dim, v_bits, VALUE_SEED);

    let q: Vec<f32> = (0..num_q_heads * dim).map(|i| unit(i, 1)).collect();

    let mut k_rows = Vec::with_capacity(seq_len * num_kv_heads);
    let mut v_rows = Vec::with_capacity(seq_len * num_kv_heads);
    for p in 0..seq_len {
        for h in 0..num_kv_heads {
            let base = (p * num_kv_heads + h) * dim;
            let k_row: Vec<f32> = (0..dim).map(|d| unit(base + d, 2) * 3.0).collect();
            let v_row: Vec<f32> = (0..dim).map(|d| unit(base + d, 3)).collect();
            k_rows.push(quantize_row(
                &k_row,
                &k_tables_cpu.signs,
                &k_tables_cpu.midpoints,
                k_bits,
            ));
            v_rows.push(quantize_row(
                &v_row,
                &v_tables_cpu.signs,
                &v_tables_cpu.midpoints,
                v_bits,
            ));
        }
    }

    IndexedFixture {
        q,
        k_rows,
        v_rows,
        k_tables_cpu,
        v_tables_cpu,
    }
}

/// Gathers `fixture`'s rows at `positions` (in list order) and runs plain
/// (non-causal, over exactly what it is handed) TQ attention over the
/// gathered subset -- the TQ analogue of `attention_indexed_parity.rs`'s
/// `run_indexed_cpu`, since `causal_attention_tq` already has no internal
/// masking: it attends over every row it is passed, whatever positions
/// those rows came from.
fn run_indexed_cpu_tq(
    dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    fixture: &IndexedFixture,
    positions: &[u32],
) -> Vec<f32> {
    let gather = |rows: &[QuantizedRow]| -> Vec<QuantizedRow> {
        positions
            .iter()
            .flat_map(|&p| {
                (0..num_kv_heads).map(move |h| rows[p as usize * num_kv_heads + h].clone())
            })
            .collect()
    };
    let scale = 1.0 / (dim as f32).sqrt();
    causal_attention_tq(
        &fixture.q,
        &gather(&fixture.k_rows),
        &gather(&fixture.v_rows),
        dim,
        num_q_heads,
        num_kv_heads,
        positions.len(),
        Some(scale),
        None,
        &fixture.k_tables_cpu,
        &fixture.v_tables_cpu,
    )
}

/// The fixture must be able to see the two mutations these tests exist to
/// catch: reading row `i` instead of `positions[i]`, and dropping the
/// chunk offset (a repeat of the first `prefix` selected rows). Mirrors
/// `attention_indexed_parity.rs`'s guard of the same name.
fn assert_fixture_discriminates_tq(
    dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    fixture: &IndexedFixture,
    positions: &[u32],
    prefix: usize,
    tol: f32,
) {
    let right = run_indexed_cpu_tq(dim, num_q_heads, num_kv_heads, fixture, positions);
    let wrong_rows: Vec<u32> = (0..positions.len() as u32).collect();
    let by_index = run_indexed_cpu_tq(dim, num_q_heads, num_kv_heads, fixture, &wrong_rows);
    let by_prefix = run_indexed_cpu_tq(
        dim,
        num_q_heads,
        num_kv_heads,
        fixture,
        &positions[..prefix],
    );
    let gap = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let g1 = gap(&right, &by_index);
    let g2 = gap(&right, &by_prefix);
    // 3x rather than the dense-kernel fixture's 10x: `tol` here is TQ's own
    // ~0.06 quantization-error budget, an order of magnitude looser than
    // the FP16 dense case's 2e-3, and the observed gaps (0.30-0.59) already
    // clear it several times over -- 3x is still a wide, deliberate margin,
    // just scaled to what this fixture actually produces at this tolerance
    // rather than borrowing a multiplier calibrated for a tighter one.
    assert!(
        g1 > 3.0 * tol,
        "fixture cannot tell positions[i] from row i (gap {g1} vs tol {tol})"
    );
    assert!(
        g2 > 3.0 * tol,
        "fixture cannot tell the full list from its first {prefix} entries (gap {g2} vs tol {tol})"
    );
}

fn to_f16(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

fn half_bytes(v: &[f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}

fn to_f32(v: &[f16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

/// BF16 is the top 16 bits of the FP32 word, the same hand-rolled
/// convention `attention_sinks.rs` uses for the dense kernel's sinks
/// buffer (AGENTS.md Gotcha 3 forbids hand-rolling FP16, not BF16).
fn to_le_bf16(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&((x.to_bits() >> 16) as u16).to_le_bytes());
    }
    out
}

fn bf16_round(x: f32) -> f32 {
    f32::from_bits((x.to_bits() >> 16) << 16)
}

fn read_half(buffer: &metal::Buffer, len: usize) -> Vec<f16> {
    let ptr = buffer.contents() as *const u16;
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    slice.iter().map(|&b| f16::from_bits(b)).collect()
}

/// Packs a list of [`QuantizedRow`]s (row-major `[seq_len, num_kv_heads]`)
/// into the `[1 + packed_words]` u32 GPU layout `attention_tq.metal` reads.
fn pack_rows(rows: &[QuantizedRow], packed_words: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows.len() * (1 + packed_words) * 4);
    for r in rows {
        out.extend_from_slice(&r.norm.to_bits().to_le_bytes());
        assert_eq!(r.words.len(), packed_words);
        for w in &r.words {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
    out
}

struct Fixture {
    q: Vec<f32>,
    k_rows: Vec<QuantizedRow>,
    v_rows: Vec<QuantizedRow>,
    k_tables_cpu: TqTables,
    v_tables_cpu: TqTables,
}

fn build_fixture(
    dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
    k_bits: u8,
    v_bits: u8,
) -> Fixture {
    let k_tables_cpu = TqTables::new(dim, k_bits, KEY_SEED);
    let v_tables_cpu = TqTables::new(dim, v_bits, VALUE_SEED);

    let q: Vec<f32> = (0..num_q_heads * dim)
        .map(|i| ((i as f32) * 0.617 + 0.11).sin() * 2.5)
        .collect();

    let mut k_rows = Vec::with_capacity(seq_len * num_kv_heads);
    let mut v_rows = Vec::with_capacity(seq_len * num_kv_heads);
    for p in 0..seq_len {
        for h in 0..num_kv_heads {
            let seed = (p * num_kv_heads + h) as f32;
            let k_row: Vec<f32> = (0..dim)
                .map(|d| ((seed * 7.0 + d as f32 * 1.31 + 0.5).sin()) * 3.0)
                .collect();
            let v_row: Vec<f32> = (0..dim)
                .map(|d| ((seed * 3.0 + d as f32 * 0.77 + 1.7).cos()) * 3.0)
                .collect();
            k_rows.push(quantize_row(
                &k_row,
                &k_tables_cpu.signs,
                &k_tables_cpu.midpoints,
                k_bits,
            ));
            v_rows.push(quantize_row(
                &v_row,
                &v_tables_cpu.signs,
                &v_tables_cpu.midpoints,
                v_bits,
            ));
        }
    }

    Fixture {
        q,
        k_rows,
        v_rows,
        k_tables_cpu,
        v_tables_cpu,
    }
}

fn run_dense_case(
    dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
    k_bits: u8,
    v_bits: u8,
) {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let quant = KvQuant::TurboQuant { k_bits, v_bits };
    let tables = KvQuantTables::new(context.device(), dim, quant).expect("tq tables build");

    let fixture = build_fixture(dim, num_q_heads, num_kv_heads, seq_len, k_bits, v_bits);
    let scale = 1.0 / (dim as f32).sqrt();

    let cpu = causal_attention_tq(
        &fixture.q,
        &fixture.k_rows,
        &fixture.v_rows,
        dim,
        num_q_heads,
        num_kv_heads,
        seq_len,
        Some(scale),
        None,
        &fixture.k_tables_cpu,
        &fixture.v_tables_cpu,
    );

    let k_packed_words = model_io::tq_packed_words(dim as i64, k_bits) as usize;
    let v_packed_words = model_io::tq_packed_words(dim as i64, v_bits) as usize;
    let k_bytes = pack_rows(&fixture.k_rows, k_packed_words);
    let v_bytes = pack_rows(&fixture.v_rows, v_packed_words);

    let q_buffer = context.new_buffer_with_data(&half_bytes(&to_f16(&fixture.q)));
    let k_buffer = context.new_buffer_with_data(&k_bytes);
    let v_buffer = context.new_buffer_with_data(&v_bytes);
    let scratch = TqAttentionScratch::new(&context, num_q_heads as u32, dim as u32);
    let out_buffer = context.new_output_buffer((num_q_heads * dim * 2) as u64);

    let pass = context.begin_pass();
    encode_attention_decode_tq(
        &mut context,
        &pass,
        (&q_buffer, 0),
        &k_buffer,
        &v_buffer,
        &scratch,
        (&out_buffer, 0),
        dim as u32,
        num_q_heads as u32,
        num_kv_heads as u32,
        seq_len as u32,
        0,
        scale,
        &tables,
        None,
    )
    .expect("dispatch succeeds");
    pass.commit_and_wait();

    let gpu = to_f32(&read_half(&out_buffer, num_q_heads * dim));
    assert_eq!(gpu.len(), cpu.len());
    let err = turbospark_compute::max_abs_diff(&gpu, &cpu);
    assert!(
        err < 0.05,
        "dim={dim} k_bits={k_bits} v_bits={v_bits} seq_len={seq_len}: err={err}"
    );
}

#[test]
fn matches_cpu_reference_gqa_k3_v4_d64() {
    run_dense_case(64, 4, 2, 6, 3, 4);
}

#[test]
fn matches_cpu_reference_single_head_k4_v4_d128() {
    run_dense_case(128, 1, 1, 5, 4, 4);
}

#[test]
fn matches_cpu_reference_over_multiple_split_kv_chunks() {
    // seq_len large enough that `chunks_for` picks more than one chunk,
    // exercising the combine kernel's cross-chunk rescale on ROTATED
    // partials (the thing dense attention.metal only has to do in real
    // space).
    run_dense_case(64, 2, 1, 300, 2, 3);
}

#[test]
fn identity_positions_match_dense_attention_tq() {
    let dim = 64usize;
    let num_q_heads = 2usize;
    let num_kv_heads = 1usize;
    let seq_len = 20usize;
    let k_bits = 3u8;
    let v_bits = 4u8;

    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let quant = KvQuant::TurboQuant { k_bits, v_bits };
    let tables = KvQuantTables::new(context.device(), dim, quant).expect("tq tables build");
    let fixture = build_fixture(dim, num_q_heads, num_kv_heads, seq_len, k_bits, v_bits);
    let scale = 1.0 / (dim as f32).sqrt();

    let k_packed_words = model_io::tq_packed_words(dim as i64, k_bits) as usize;
    let v_packed_words = model_io::tq_packed_words(dim as i64, v_bits) as usize;
    let k_bytes = pack_rows(&fixture.k_rows, k_packed_words);
    let v_bytes = pack_rows(&fixture.v_rows, v_packed_words);

    let q_buffer = context.new_buffer_with_data(&half_bytes(&to_f16(&fixture.q)));
    let k_buffer = context.new_buffer_with_data(&k_bytes);
    let v_buffer = context.new_buffer_with_data(&v_bytes);

    // Dense reference dispatch.
    let dense_scratch = TqAttentionScratch::new(&context, num_q_heads as u32, dim as u32);
    let dense_out = context.new_output_buffer((num_q_heads * dim * 2) as u64);
    let pass = context.begin_pass();
    encode_attention_decode_tq(
        &mut context,
        &pass,
        (&q_buffer, 0),
        &k_buffer,
        &v_buffer,
        &dense_scratch,
        (&dense_out, 0),
        dim as u32,
        num_q_heads as u32,
        num_kv_heads as u32,
        seq_len as u32,
        0,
        scale,
        &tables,
        None,
    )
    .unwrap();
    pass.commit_and_wait();
    let dense = to_f32(&read_half(&dense_out, num_q_heads * dim));

    // Indexed dispatch over the identity list.
    let positions: Vec<u32> = (0..seq_len as u32).collect();
    let positions_bytes: Vec<u8> = positions.iter().flat_map(|p| p.to_le_bytes()).collect();
    let positions_buffer = context.new_buffer_with_data(&positions_bytes);
    let idx_scratch = TqAttentionScratch::new(&context, num_q_heads as u32, dim as u32);
    let idx_out = context.new_output_buffer((num_q_heads * dim * 2) as u64);
    let pass2 = context.begin_pass();
    encode_attention_decode_indexed_tq(
        &mut context,
        &pass2,
        (&q_buffer, 0),
        &k_buffer,
        &v_buffer,
        (&positions_buffer, 0),
        seq_len as u32,
        &idx_scratch,
        (&idx_out, 0),
        dim as u32,
        num_q_heads as u32,
        num_kv_heads as u32,
        scale,
        &tables,
    )
    .unwrap();
    pass2.commit_and_wait();
    let indexed = to_f32(&read_half(&idx_out, num_q_heads * dim));

    let err = turbospark_compute::max_abs_diff(&dense, &indexed);
    assert!(
        err < 1e-3,
        "identity indexed dispatch should match dense: err={err}"
    );
}

/// AGENTS.md/CLAUDE.md B3: `attention_tq.metal`'s `kTqAttnMaxHeadDim` array
/// ceiling, mirroring `attention_decode_parity.rs`'s guard for the dense
/// kernel. `dim` must still be `rht_supported` (a power of two in 32..=512)
/// for `KvQuantTables::new` to build at all, so this exercises the largest
/// legal width rather than something past it -- the point is that the new
/// `head_dim <= MAX_DECODE_ATTENTION_HEAD_DIM` assert does not regress a
/// legal dispatch at the boundary. `crates/streaming`/`real_forward_init`'s
/// `rht_supported` gate is what actually stops a checkpoint from reaching
/// this dispatch above 512 in production.
#[test]
fn dim_512_the_widest_rht_supported_width_still_dispatches() {
    run_dense_case(512, 1, 1, 2, 4, 4);
}

/// AGENTS.md/CLAUDE.md B4: `encode_attention_decode_indexed_tq` used to
/// accept `n_sel == 0` by masking it to 1 for the CHUNK COUNT only
/// (`chunks_for(n_sel.max(1))`), then passing the raw `n_sel` (still 0) to
/// the shader as the loop bound -- every lane's partial stayed at its
/// initial `(m=-inf, d=0, o=0)` and the combine pass divided 0/0 into NaN.
/// This calls the encoder directly (not through a slice-taking wrapper) so
/// the positions buffer is a real, non-empty allocation and the panic this
/// test expects is the new `n_sel > 0` guard, not an unrelated zero-length
/// buffer failure.
#[test]
#[should_panic]
fn zero_selected_positions_is_refused() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let dim = 32usize;
    let tables = KvQuantTables::new(
        context.device(),
        dim,
        KvQuant::TurboQuant {
            k_bits: 4,
            v_bits: 4,
        },
    )
    .expect("tq tables build");

    let q_buffer = context.new_buffer_with_data(&half_bytes(&to_f16(&vec![0.1f32; dim])));
    let k_packed_words = model_io::tq_packed_words(dim as i64, 4) as usize;
    let v_packed_words = model_io::tq_packed_words(dim as i64, 4) as usize;
    let k_buffer = context.new_buffer_with_data(&vec![0u8; (1 + k_packed_words) * 4]);
    let v_buffer = context.new_buffer_with_data(&vec![0u8; (1 + v_packed_words) * 4]);
    // A real, non-empty positions buffer: this test is about `n_sel`, the
    // separate argument the shader actually loops over, never about buffer
    // length.
    let positions_buffer = context.new_buffer_with_data(&[0u32]);
    let scratch = TqAttentionScratch::new(&context, 1, dim as u32);
    let out_buffer = context.new_output_buffer((dim * 2) as u64);

    let pass = context.begin_pass();
    let _ = encode_attention_decode_indexed_tq(
        &mut context,
        &pass,
        (&q_buffer, 0),
        &k_buffer,
        &v_buffer,
        (&positions_buffer, 0),
        0,
        &scratch,
        (&out_buffer, 0),
        dim as u32,
        1,
        1,
        1.0 / (dim as f32).sqrt(),
        &tables,
    );
}

/// T1: `identity_positions_match_dense_attention_tq` above is the only
/// positions-aware TQ case, and it only ever passes the identity list, so
/// `positions[i]` indexing and the chunk-offset arithmetic
/// (`attention_indexed_tq`'s own copy of `attention_indexed.metal`'s) are
/// invisible to it. These three port `attention_indexed_parity.rs`'s
/// fixture shape (independent rows, `assert_fixture_discriminates` first,
/// a strided subset, a multi-chunk subset, poisoned unselected rows) onto
/// the TQ kernel.
#[allow(clippy::too_many_arguments)]
fn run_indexed_gpu_tq(
    context: &mut MetalContext,
    dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
    k_bits: u8,
    v_bits: u8,
    fixture: &IndexedFixture,
    positions: &[u32],
) -> Vec<f32> {
    let quant = KvQuant::TurboQuant { k_bits, v_bits };
    let tables = KvQuantTables::new(context.device(), dim, quant).expect("tq tables build");
    let scale = 1.0 / (dim as f32).sqrt();

    let k_packed_words = model_io::tq_packed_words(dim as i64, k_bits) as usize;
    let v_packed_words = model_io::tq_packed_words(dim as i64, v_bits) as usize;
    let k_bytes = pack_rows(&fixture.k_rows, k_packed_words);
    let v_bytes = pack_rows(&fixture.v_rows, v_packed_words);

    let q_buffer = context.new_buffer_with_data(&half_bytes(&to_f16(&fixture.q)));
    let k_buffer = context.new_buffer_with_data(&k_bytes);
    let v_buffer = context.new_buffer_with_data(&v_bytes);
    let positions_bytes: Vec<u8> = positions.iter().flat_map(|p| p.to_le_bytes()).collect();
    let positions_buffer = context.new_buffer_with_data(&positions_bytes);
    let scratch = TqAttentionScratch::new(context, num_q_heads as u32, dim as u32);
    let out_buffer = context.new_output_buffer((num_q_heads * dim * 2) as u64);

    let pass = context.begin_pass();
    encode_attention_decode_indexed_tq(
        context,
        &pass,
        (&q_buffer, 0),
        &k_buffer,
        &v_buffer,
        (&positions_buffer, 0),
        positions.len() as u32,
        &scratch,
        (&out_buffer, 0),
        dim as u32,
        num_q_heads as u32,
        num_kv_heads as u32,
        scale,
        &tables,
    )
    .expect("indexed tq dispatch");
    pass.commit_and_wait();
    let _ = seq_len; // documents the fixture's stored row count; not read directly here.
    to_f32(&read_half(&out_buffer, num_q_heads * dim))
}

/// A NON-CONTIGUOUS subset (every third row plus the last), single chunk,
/// against the TQ CPU reference.
#[test]
fn a_strided_subset_matches_the_cpu_reference_tq() {
    let dim = 32usize;
    let num_q_heads = 4usize;
    let num_kv_heads = 2usize;
    let seq_len = 40usize;
    let k_bits = 3u8;
    let v_bits = 4u8;

    let fixture = build_indexed_fixture(dim, num_q_heads, num_kv_heads, seq_len, k_bits, v_bits);
    let mut positions: Vec<u32> = (0..seq_len as u32).step_by(3).collect();
    positions.push(seq_len as u32 - 1);
    positions.sort_unstable();
    positions.dedup();
    assert_eq!(positions.len(), 14);

    let tol = 0.06;
    assert_fixture_discriminates_tq(dim, num_q_heads, num_kv_heads, &fixture, &positions, 7, tol);
    let cpu = run_indexed_cpu_tq(dim, num_q_heads, num_kv_heads, &fixture, &positions);
    let mut context = MetalContext::new().expect("Metal device");
    let gpu = run_indexed_gpu_tq(
        &mut context,
        dim,
        num_q_heads,
        num_kv_heads,
        seq_len,
        k_bits,
        v_bits,
        &fixture,
        &positions,
    );
    let err = turbospark_compute::max_abs_diff(&gpu, &cpu);
    assert!(err < tol, "strided subset: err={err}");
}

/// Enough selected positions to split into more than one chunk
/// (`chunks_for(48) == 2`), exercising the chunk-slice arithmetic and the
/// cross-chunk combine, against the TQ CPU reference.
#[test]
fn a_multi_chunk_subset_matches_the_cpu_reference_tq() {
    let dim = 32usize;
    let num_q_heads = 4usize;
    let num_kv_heads = 2usize;
    let seq_len = 100usize;
    let k_bits = 4u8;
    let v_bits = 3u8;

    let fixture = build_indexed_fixture(dim, num_q_heads, num_kv_heads, seq_len, k_bits, v_bits);
    // 48 of 100 rows: 4-row runs with gaps, plus a ragged remainder.
    let mut positions: Vec<u32> = Vec::new();
    for block in (0..24).step_by(2) {
        positions.extend((block * 4..block * 4 + 4).map(|p| p as u32));
    }
    assert_eq!(positions.len(), 48);
    assert!(*positions.last().unwrap() < seq_len as u32);

    let tol = 0.06;
    assert_fixture_discriminates_tq(
        dim,
        num_q_heads,
        num_kv_heads,
        &fixture,
        &positions,
        24,
        tol,
    );
    let cpu = run_indexed_cpu_tq(dim, num_q_heads, num_kv_heads, &fixture, &positions);
    let mut context = MetalContext::new().expect("Metal device");
    let gpu = run_indexed_gpu_tq(
        &mut context,
        dim,
        num_q_heads,
        num_kv_heads,
        seq_len,
        k_bits,
        v_bits,
        &fixture,
        &positions,
    );
    let err = turbospark_compute::max_abs_diff(&gpu, &cpu);
    assert!(err < tol, "multi-chunk subset: err={err}");
}

/// Rows off the list must never be READ. TQ has no bit pattern that
/// decodes to NaN (every packed row is a bounded codebook index), so this
/// poisons unselected rows with a DIFFERENT, extreme-but-valid quantized
/// row instead: max codebook index in every word and a norm two orders of
/// magnitude past the fixture's own -- a stray read shows up as a gross,
/// bit-visible change rather than staying silently within tolerance.
#[test]
fn unselected_rows_are_never_read_tq() {
    let dim = 32usize;
    let num_q_heads = 4usize;
    let num_kv_heads = 2usize;
    let seq_len = 70usize;
    let k_bits = 4u8;
    let v_bits = 4u8;

    let fixture = build_indexed_fixture(dim, num_q_heads, num_kv_heads, seq_len, k_bits, v_bits);
    let positions: Vec<u32> = (0..seq_len as u32).filter(|p| p % 7 < 4).collect();
    assert_eq!(positions.len(), 40);

    let tol = 0.06;
    assert_fixture_discriminates_tq(
        dim,
        num_q_heads,
        num_kv_heads,
        &fixture,
        &positions,
        20,
        tol,
    );

    let mut context = MetalContext::new().expect("Metal device");
    let clean = run_indexed_gpu_tq(
        &mut context,
        dim,
        num_q_heads,
        num_kv_heads,
        seq_len,
        k_bits,
        v_bits,
        &fixture,
        &positions,
    );
    assert!(clean.iter().all(|x| x.is_finite()));
    let cpu = run_indexed_cpu_tq(dim, num_q_heads, num_kv_heads, &fixture, &positions);
    assert!(
        turbospark_compute::max_abs_diff(&clean, &cpu) < tol,
        "clean run must also be the right answer, or a kernel reading only a \
         prefix of the list would pass the poisoned comparison below trivially"
    );

    let k_packed_words = model_io::tq_packed_words(dim as i64, k_bits) as usize;
    let v_packed_words = model_io::tq_packed_words(dim as i64, v_bits) as usize;
    let poison_k = QuantizedRow {
        norm: fixture
            .k_tables_cpu
            .codebook
            .iter()
            .cloned()
            .fold(1.0f32, f32::max)
            * 100.0,
        words: vec![u32::MAX; k_packed_words],
    };
    let poison_v = QuantizedRow {
        norm: fixture
            .v_tables_cpu
            .codebook
            .iter()
            .cloned()
            .fold(1.0f32, f32::max)
            * 100.0,
        words: vec![u32::MAX; v_packed_words],
    };
    let mut poisoned = IndexedFixture {
        q: fixture.q.clone(),
        k_rows: fixture.k_rows.clone(),
        v_rows: fixture.v_rows.clone(),
        k_tables_cpu: fixture.k_tables_cpu.clone(),
        v_tables_cpu: fixture.v_tables_cpu.clone(),
    };
    let mut poisoned_rows = 0;
    for p in 0..seq_len {
        if positions.binary_search(&(p as u32)).is_err() {
            for h in 0..num_kv_heads {
                poisoned.k_rows[p * num_kv_heads + h] = poison_k.clone();
                poisoned.v_rows[p * num_kv_heads + h] = poison_v.clone();
            }
            poisoned_rows += 1;
        }
    }
    assert_eq!(poisoned_rows, 30);

    let poisoned_out = run_indexed_gpu_tq(
        &mut context,
        dim,
        num_q_heads,
        num_kv_heads,
        seq_len,
        k_bits,
        v_bits,
        &poisoned,
        &positions,
    );
    let clean_bits: Vec<u16> = to_f16(&clean).iter().map(|x| x.to_bits()).collect();
    let poisoned_bits: Vec<u16> = to_f16(&poisoned_out).iter().map(|x| x.to_bits()).collect();
    assert_eq!(
        clean_bits, poisoned_bits,
        "a corrupted unselected row reached the output: the kernel read a row off the list"
    );
}

/// T2: `attention_tq.rs`'s sinks path (`FC_TQATTN_HAS_SINKS`, constant 80,
/// separate from `attention_decode.rs`'s constant-70 dense sinks path
/// `attention_sinks.rs` already covers) is never dispatched with
/// `sinks: Some(...)` anywhere in this file -- every existing TQ case
/// passes `None`, so this file's own suite already covers the sink-free
/// dispatch. Ports `attention_sinks.rs`'s shape (sinks comparable to the
/// real scores, a DOMINANT sink, and a NEGLIGIBLE one) onto the TQ kernel,
/// against `causal_attention_tq`'s own `sinks` argument.
fn assert_tq_sinks_match_cpu(
    dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
    k_bits: u8,
    v_bits: u8,
    sink_scale: f32,
) {
    let mut context = MetalContext::new().expect("Metal device");
    let quant = KvQuant::TurboQuant { k_bits, v_bits };
    let tables = KvQuantTables::new(context.device(), dim, quant).expect("tq tables build");

    let fixture = build_fixture(dim, num_q_heads, num_kv_heads, seq_len, k_bits, v_bits);
    let scale = 1.0 / (dim as f32).sqrt();
    // Rounded through BF16 on the reference side too, matching
    // `attention_sinks.rs`'s reasoning: that is the width the kernel reads,
    // so it must not be a second source of disagreement.
    let sinks: Vec<f32> = (0..num_q_heads)
        .map(|i| bf16_round(((i as f32) * 0.37).sin() * sink_scale))
        .collect();

    let cpu = causal_attention_tq(
        &fixture.q,
        &fixture.k_rows,
        &fixture.v_rows,
        dim,
        num_q_heads,
        num_kv_heads,
        seq_len,
        Some(scale),
        Some(&sinks),
        &fixture.k_tables_cpu,
        &fixture.v_tables_cpu,
    );

    let k_packed_words = model_io::tq_packed_words(dim as i64, k_bits) as usize;
    let v_packed_words = model_io::tq_packed_words(dim as i64, v_bits) as usize;
    let k_bytes = pack_rows(&fixture.k_rows, k_packed_words);
    let v_bytes = pack_rows(&fixture.v_rows, v_packed_words);

    let q_buffer = context.new_buffer_with_data(&half_bytes(&to_f16(&fixture.q)));
    let k_buffer = context.new_buffer_with_data(&k_bytes);
    let v_buffer = context.new_buffer_with_data(&v_bytes);
    let sink_buffer = context.new_buffer_with_data(&to_le_bf16(&sinks));
    let scratch = TqAttentionScratch::new(&context, num_q_heads as u32, dim as u32);
    let out_buffer = context.new_output_buffer((num_q_heads * dim * 2) as u64);

    let pass = context.begin_pass();
    encode_attention_decode_tq(
        &mut context,
        &pass,
        (&q_buffer, 0),
        &k_buffer,
        &v_buffer,
        &scratch,
        (&out_buffer, 0),
        dim as u32,
        num_q_heads as u32,
        num_kv_heads as u32,
        seq_len as u32,
        0,
        scale,
        &tables,
        Some((&sink_buffer, 0)),
    )
    .expect("dispatch with sinks succeeds");
    pass.commit_and_wait();

    let gpu = to_f32(&read_half(&out_buffer, num_q_heads * dim));
    assert_eq!(gpu.len(), cpu.len());
    let err = turbospark_compute::max_abs_diff(&gpu, &cpu);
    assert!(
        err < 0.06,
        "dim={dim} k_bits={k_bits} v_bits={v_bits} sink_scale={sink_scale}: err={err}"
    );
}

/// Sinks comparable in size to the real scores, where the term genuinely
/// changes the answer.
#[test]
fn tq_sinks_match_the_cpu_reference() {
    assert_tq_sinks_match_cpu(32, 4, 2, 20, 3, 4, 1.0);
}

/// Sinks well ABOVE the score maximum: they take most of the probability
/// mass and the output shrinks toward zero. The case that fails if the
/// sink is left out of the combine pass's running maximum.
#[test]
fn a_dominant_tq_sink_takes_most_of_the_mass() {
    assert_tq_sinks_match_cpu(32, 4, 2, 20, 4, 4, 8.0);
}

/// Sinks far BELOW every score, where the term is nearly a no-op -- the
/// case a broken kernel is most likely to pass, kept as a control on the
/// two above.
#[test]
fn a_negligible_tq_sink_barely_moves_the_output() {
    assert_tq_sinks_match_cpu(32, 4, 2, 20, 4, 3, -8.0);
}
