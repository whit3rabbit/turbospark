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
