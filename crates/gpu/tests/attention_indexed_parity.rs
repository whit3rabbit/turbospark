#![cfg(target_os = "macos")]
//! Parity tests for `attention_decode_indexed_partial` (decode attention
//! over an explicit position list, `qwen4_exp`'s QSA attention application)
//! against `turbospark_compute::indexed_attention` and against the DENSE
//! kernel it is copied from.
//!
//! The load-bearing case is the bitwise one: with the identity list at equal
//! chunk count the indexed dispatch must produce the dense kernel's exact
//! bytes, because that equality is what lets the dense kernel's real-model
//! verification stand in for this kernel's. The NaN case is the other half:
//! rows NOT on the list must be unreadable, and NaN in every unselected row
//! is the loudest possible detector of a stray read (AGENTS.md Gotcha 59,
//! turned from a trap into an instrument).

use half::f16;
use turbospark_gpu::{
    attention_decode_indexed, encode_attention_decode, encode_attention_decode_indexed,
    AttentionScratch, MetalContext,
};

fn to_f16(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

fn to_f32(v: &[f16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

fn to_le(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

fn read_f32_bits(buffer: &metal::Buffer) -> Vec<u32> {
    let len = buffer.length() as usize / 4;
    let ptr = buffer.contents() as *const u32;
    unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
}

fn read_halfs(buffer: &metal::Buffer, len: usize) -> Vec<f16> {
    let ptr = buffer.contents() as *const u16;
    let bits = unsafe { std::slice::from_raw_parts(ptr, len) };
    bits.iter().map(|&b| f16::from_bits(b)).collect()
}

struct Shape {
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
}

/// Deterministic, INDEPENDENT pseudo-random values in `[-1, 1)` (splitmix64
/// on the flat index). Not a sinusoid over the flat index: `sin(i * 1.3)`
/// over `[seq, heads, head_dim]` makes each key row a small rotation of the
/// previous one, so the scores are near-periodic in position and two large
/// row sets sample that pattern identically -- measured as a 0.0008 gap
/// between attending over `positions[i]` and over row `i` on the real shape,
/// where independent rows give O(0.1).
fn unit(i: usize, salt: u64) -> f32 {
    let mut z = (i as u64)
        .wrapping_add(salt)
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn fixture(s: &Shape) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q_len = s.num_q_heads * s.head_dim;
    let kv_len = s.seq_len * s.num_kv_heads * s.head_dim;
    // Every row distinct and independent, and the softmax PEAKED: keys at
    // magnitude 3 against unit queries give the scaled scores a spread of
    // about 1.0 whatever the head_dim (`3 * sqrt(d / 9) / sqrt(d)`), so
    // attending over a wrong row set moves the output by O(0.1). A first
    // draft used keys at 0.3, which made the softmax near-uniform and the
    // output the mean of V over whatever rows were read -- close to zero
    // for ANY row set, so two mutations (wrong row, dropped chunk offset)
    // survived within tolerance. Every CPU-compared test below asserts its
    // fixture can see the wrong row set before trusting the comparison.
    let q: Vec<f32> = (0..q_len).map(|i| unit(i, 1)).collect();
    let k: Vec<f32> = (0..kv_len).map(|i| unit(i, 2) * 3.0).collect();
    let v: Vec<f32> = (0..kv_len).map(|i| unit(i, 3)).collect();
    (q, k, v)
}

fn assert_close(gpu: &[f32], cpu: &[f32], tol: f32, what: &str) {
    assert_eq!(gpu.len(), cpu.len());
    let mut worst = 0.0f32;
    for (i, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        assert!(g.is_finite(), "{what}: non-finite GPU output at {i}: {g}");
        let diff = (g - c).abs();
        worst = worst.max(diff);
        assert!(
            diff <= tol,
            "{what}: element {i} differs: gpu {g} vs cpu {c} (|diff| {diff} > {tol})"
        );
    }
    eprintln!(
        "{what}: worst |diff| {worst:.3e} over {} elements",
        gpu.len()
    );
}

/// The fixture must be able to SEE the two mutations the parity tests are
/// meant to catch: reading row `i` instead of `positions[i]` (the CPU answer
/// over `0..n_sel` must differ from the right one) and dropping the chunk
/// offset (the CPU answer over the first `prefix` selected rows must
/// differ). Both gaps must clear the parity tolerance by a wide margin, or a
/// green parity line proves nothing about either.
fn assert_fixture_discriminates(
    s: &Shape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    positions: &[u32],
    prefix: usize,
    tol: f32,
) {
    let right = run_indexed_cpu(s, q, k, v, positions);
    let wrong_rows: Vec<u32> = (0..positions.len() as u32).collect();
    let by_index = run_indexed_cpu(s, q, k, v, &wrong_rows);
    let by_prefix = run_indexed_cpu(s, q, k, v, &positions[..prefix]);
    let gap = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let g1 = gap(&right, &by_index);
    let g2 = gap(&right, &by_prefix);
    assert!(
        g1 > 10.0 * tol,
        "fixture cannot tell positions[i] from row i (gap {g1} vs tol {tol})"
    );
    assert!(
        g2 > 10.0 * tol,
        "fixture cannot tell the full list from its first {prefix} entries (gap {g2} vs tol {tol})"
    );
}

fn run_indexed_cpu(s: &Shape, q: &[f32], k: &[f32], v: &[f32], positions: &[u32]) -> Vec<f32> {
    let pos: Vec<usize> = positions.iter().map(|&p| p as usize).collect();
    let scale = 1.0 / (s.head_dim as f32).sqrt();
    turbospark_compute::indexed_attention(
        q,
        k,
        v,
        &pos,
        s.head_dim,
        s.num_q_heads,
        s.num_kv_heads,
        Some(scale),
    )
}

fn run_indexed_gpu(
    context: &mut MetalContext,
    s: &Shape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    positions: &[u32],
) -> Vec<f16> {
    let scale = 1.0 / (s.head_dim as f32).sqrt();
    attention_decode_indexed(
        context,
        &to_f16(q),
        &to_f16(k),
        &to_f16(v),
        positions,
        s.head_dim as u32,
        s.num_q_heads as u32,
        s.num_kv_heads as u32,
        scale,
    )
    .expect("indexed attention dispatch")
}

/// Rounds the FP32 fixture through FP16 before the CPU reference sees it,
/// so the comparison is against what the GPU actually read.
fn f16_round(v: &[f32]) -> Vec<f32> {
    to_f32(&to_f16(v))
}

/// A NON-CONTIGUOUS subset (every third row plus the last one) at a GQA
/// shape, single chunk (13 selected < 16), against the CPU reference.
#[test]
fn a_strided_subset_matches_the_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let s = Shape {
        head_dim: 8,
        num_q_heads: 4,
        num_kv_heads: 2,
        seq_len: 40,
    };
    let (q, k, v) = fixture(&s);
    let (q, k, v) = (f16_round(&q), f16_round(&k), f16_round(&v));
    let mut positions: Vec<u32> = (0..s.seq_len as u32).step_by(3).collect();
    positions.push(s.seq_len as u32 - 1);
    positions.sort_unstable();
    positions.dedup();
    assert_eq!(positions.len(), 14);

    // Single chunk here, so the prefix check uses half the list.
    assert_fixture_discriminates(&s, &q, &k, &v, &positions, 7, 2e-3);
    let cpu = run_indexed_cpu(&s, &q, &k, &v, &positions);
    let gpu = to_f32(&run_indexed_gpu(&mut context, &s, &q, &k, &v, &positions));
    assert_close(&gpu, &cpu, 2e-3, "strided subset");
}

/// Enough selected positions (48) to split into more than one chunk
/// (`chunks_for(48) == 2`), so the chunk-slice arithmetic and the combine
/// over two partials are exercised, against the CPU reference.
#[test]
fn a_multi_chunk_subset_matches_the_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let s = Shape {
        head_dim: 16,
        num_q_heads: 4,
        num_kv_heads: 2,
        seq_len: 100,
    };
    let (q, k, v) = fixture(&s);
    let (q, k, v) = (f16_round(&q), f16_round(&k), f16_round(&v));
    // 48 of 100 rows: a block-ish pattern of 4-row runs with gaps, plus the
    // ragged tail, which is the shape QSA's mask actually has.
    let mut positions: Vec<u32> = Vec::new();
    for block in (0..24).step_by(2) {
        positions.extend((block * 4..block * 4 + 4).map(|p| p as u32));
    }
    assert_eq!(positions.len(), 48);
    assert!(*positions.last().unwrap() < s.seq_len as u32);

    // `chunks_for(48) == 2`, chunk_len 24: a dropped chunk offset reads the
    // first 24 selected rows twice.
    assert_fixture_discriminates(&s, &q, &k, &v, &positions, 24, 2e-3);
    let cpu = run_indexed_cpu(&s, &q, &k, &v, &positions);
    let gpu = to_f32(&run_indexed_gpu(&mut context, &s, &q, &k, &v, &positions));
    assert_close(&gpu, &cpu, 2e-3, "multi-chunk subset");
}

/// THE BITWISE CASE. With the identity list the indexed dispatch must equal
/// the dense `encode_attention_decode` dispatch byte for byte: same chunk
/// count (both derive it from a range of `seq_len`), same reduction, same
/// recurrence, same combine. `seq_len = 96` makes `chunks_for` pick 4, so
/// what is compared is the split path, not the trivial one-chunk one.
#[test]
fn the_identity_list_is_bit_identical_to_the_dense_kernel() {
    let mut context = MetalContext::new().expect("Metal device");
    let s = Shape {
        head_dim: 32,
        num_q_heads: 6,
        num_kv_heads: 2,
        seq_len: 96,
    };
    let (q, k, v) = fixture(&s);
    let (q16, k16, v16) = (to_f16(&q), to_f16(&k), to_f16(&v));
    let scale = 1.0 / (s.head_dim as f32).sqrt();
    let out_len = s.num_q_heads * s.head_dim;

    let q_buf = context.new_buffer_with_data(&to_le(&q16));
    let k_buf = context.new_buffer_with_data(&to_le(&k16));
    let v_buf = context.new_buffer_with_data(&to_le(&v16));
    let scratch = AttentionScratch::new(&context, s.num_q_heads as u32, s.head_dim as u32);

    let dense_out = context.new_output_buffer(out_len as u64 * 2);
    let pass = context.begin_pass();
    encode_attention_decode(
        &mut context,
        &pass,
        (&q_buf, 0),
        &k_buf,
        &v_buf,
        &scratch,
        (&dense_out, 0),
        s.head_dim as u32,
        s.num_q_heads as u32,
        s.num_kv_heads as u32,
        s.seq_len as u32,
        0,
        0,
        scale,
        None,
    )
    .expect("dense dispatch");
    pass.commit_and_wait();
    let dense = read_halfs(&dense_out, out_len);
    let dense_partials = (
        read_f32_bits(&scratch.m),
        read_f32_bits(&scratch.d),
        read_f32_bits(&scratch.o),
    );

    let identity: Vec<u32> = (0..s.seq_len as u32).collect();
    let positions_buf = context.new_buffer_with_data(&identity);
    let indexed_out = context.new_output_buffer(out_len as u64 * 2);
    let pass = context.begin_pass();
    encode_attention_decode_indexed(
        &mut context,
        &pass,
        (&q_buf, 0),
        &k_buf,
        &v_buf,
        (&positions_buf, 0),
        identity.len() as u32,
        &scratch,
        (&indexed_out, 0),
        s.head_dim as u32,
        s.num_q_heads as u32,
        s.num_kv_heads as u32,
        scale,
    )
    .expect("indexed dispatch");
    pass.commit_and_wait();
    let indexed = read_halfs(&indexed_out, out_len);
    let indexed_partials = (
        read_f32_bits(&scratch.m),
        read_f32_bits(&scratch.d),
        read_f32_bits(&scratch.o),
    );

    // The FP32 PARTIALS, not just the FP16 output. FP16 rounding of the
    // output absorbs reassociation-level differences, so an indexed
    // dispatch at ONE chunk against the dense kernel's four still rounded
    // to the same halfs (measured: that mutation survived the output-only
    // form of this test). The scratch is shared between the two runs, so
    // its slots are either rewritten identically or left as the dense run
    // wrote them; a different chunk count writes a different slot set and
    // a different `m`/`d` per slot, and this comparison sees both.
    assert_eq!(
        dense_partials, indexed_partials,
        "identity-list indexed attention must write the dense kernel's exact FP32 partials"
    );
    assert!(dense.iter().all(|x| x.is_finite()));
    let dense_bits: Vec<u16> = dense.iter().map(|x| x.to_bits()).collect();
    let indexed_bits: Vec<u16> = indexed.iter().map(|x| x.to_bits()).collect();
    assert_eq!(
        dense_bits, indexed_bits,
        "identity-list indexed attention must be bit-identical to the dense kernel"
    );
}

/// Rows off the list must never be READ, not merely weighted to zero. Every
/// unselected K and V row is set to NaN; one stray read of any of them
/// poisons the online softmax for that head, so the output must be
/// bit-identical to the run over clean data.
#[test]
fn unselected_rows_are_never_read() {
    let mut context = MetalContext::new().expect("Metal device");
    let s = Shape {
        head_dim: 8,
        num_q_heads: 4,
        num_kv_heads: 2,
        seq_len: 70,
    };
    let (q, k, v) = fixture(&s);
    // 40 selected (multi-chunk), 30 unselected, interleaved.
    let positions: Vec<u32> = (0..s.seq_len as u32).filter(|p| p % 7 < 4).collect();
    assert_eq!(positions.len(), 40);

    let (q, k, v) = (f16_round(&q), f16_round(&k), f16_round(&v));
    let clean = run_indexed_gpu(&mut context, &s, &q, &k, &v, &positions);
    assert!(
        clean.iter().all(|x| x.is_finite()),
        "clean run must be finite"
    );
    // The clean run must ALSO be the right answer: a kernel that read only
    // a prefix of the list would never touch a poisoned row and would pass
    // the clean-vs-poisoned comparison below on its own. (`chunks_for(40)
    // == 2`, chunk_len 20.)
    assert_fixture_discriminates(&s, &q, &k, &v, &positions, 20, 2e-3);
    assert_close(
        &to_f32(&clean),
        &run_indexed_cpu(&s, &q, &k, &v, &positions),
        2e-3,
        "NaN case, clean run",
    );

    let row = s.num_kv_heads * s.head_dim;
    let (mut k_poison, mut v_poison) = (k.clone(), v.clone());
    let mut poisoned_rows = 0;
    for p in 0..s.seq_len {
        if positions.binary_search(&(p as u32)).is_err() {
            k_poison[p * row..(p + 1) * row].fill(f32::NAN);
            v_poison[p * row..(p + 1) * row].fill(f32::NAN);
            poisoned_rows += 1;
        }
    }
    assert_eq!(poisoned_rows, 30);
    let poisoned = run_indexed_gpu(&mut context, &s, &q, &k_poison, &v_poison, &positions);

    let clean_bits: Vec<u16> = clean.iter().map(|x| x.to_bits()).collect();
    let poisoned_bits: Vec<u16> = poisoned.iter().map(|x| x.to_bits()).collect();
    assert_eq!(
        clean_bits, poisoned_bits,
        "NaN in an unselected row reached the output: the kernel read a row off the list"
    );
}

/// The real `qwen4_exp` shape: `head_dim` 256, 24 query heads over 2 KV
/// heads, 3,000 stored rows of which 2,051 are selected (512 blocks of 4
/// plus a 3-row tail), `chunks_for(2051) == 16`. Against the CPU reference.
#[test]
fn the_real_qwen4_exp_shape_matches_the_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let s = Shape {
        head_dim: 256,
        num_q_heads: 24,
        num_kv_heads: 2,
        seq_len: 3000,
    };
    let (q, k, v) = fixture(&s);
    // Sharper than the shared fixture: over 2,051 rows a spread-2 softmax
    // still averages hundreds of them, and two row sets sharing most of
    // their rows then agree within tolerance (measured 0.0019 against a
    // 0.004 tolerance on the first draft). Scaling the query 2.5x puts the
    // mass on a handful of rows, which differ between the sets.
    let q: Vec<f32> = q.iter().map(|x| x * 2.5).collect();
    let (q, k, v) = (f16_round(&q), f16_round(&k), f16_round(&v));
    // Keep three blocks in four of the 750 complete ones (562), keep the
    // LAST 512 of those (the real `block_topk`, and a set that overlaps
    // rows `0..2051` as little as a block pattern allows), then add three
    // trailing rows standing in for the ragged tail -- 2,051 rows, the most
    // QSA ever selects.
    let complete_blocks = s.seq_len / 4; // 750
    let mut chosen: Vec<usize> = (0..complete_blocks).filter(|b| b % 4 != 1).collect();
    chosen.drain(..chosen.len() - 512);
    assert_eq!(chosen.len(), 512);
    let mut positions: Vec<u32> = chosen
        .iter()
        .flat_map(|&b| (b * 4..b * 4 + 4).map(|p| p as u32))
        .collect();
    positions.extend([2997u32, 2998, 2999]);
    positions.sort_unstable();
    positions.dedup();
    assert_eq!(positions.len(), 2051);

    // `chunks_for(2051) == 16`, chunk_len 129.
    assert_fixture_discriminates(&s, &q, &k, &v, &positions, 129, 4e-3);
    let cpu = run_indexed_cpu(&s, &q, &k, &v, &positions);
    let gpu = to_f32(&run_indexed_gpu(&mut context, &s, &q, &k, &v, &positions));
    assert_close(&gpu, &cpu, 4e-3, "real shape");
}
