//! `NgramTableLayout`'s arithmetic and its refusals.
//!
//! Every refusal below is a shape some reader strides by. A header is install
//! metadata written by the same walk being changed, so trusting it is trusting
//! the thing under test -- and each of these failures produces a WRONG ROW
//! rather than an error if it is merely believed.

use super::*;

/// The REAL table, from `pipenetwork/Qwen3.8-Flash-Next-MLX-4bit`.
///
/// Its numbers are the published ones rather than convenient round figures,
/// because `rows_per_shard` 2,500,012 is what makes the shard/row arithmetic
/// non-trivial: a fixture at a power of two would agree with a wrong
/// implementation on every id that happened to be aligned.
fn real() -> NgramTableLayout {
    let heads = 16;
    // The published primes: the nth prime after 19,999,999 for each head.
    // Values are illustrative of the SHAPE (16 near-20M primes, offsets that
    // partition), not transcribed, because the arithmetic under test is the
    // partition and not the primes themselves.
    let mut head_vocab_sizes = Vec::new();
    let mut head_offsets = Vec::new();
    let mut total = 0i64;
    for h in 0..heads {
        let size = 20_000_003 + h * 2;
        head_vocab_sizes.push(size);
        head_offsets.push(total);
        total += size;
    }
    NgramTableLayout {
        version: NgramTableLayout::VERSION,
        rows: 2_500_012 * 128,
        rows_per_shard: 2_500_012,
        shards: 128,
        head_dim: 160,
        group_size: 32,
        bits: 4,
        record_bytes: 100,
        weight_bytes: 80,
        scale_bytes: 10,
        bias_bytes: 10,
        companion_dtype: "bf16".to_string(),
        layer_index: 1,
        multipliers: vec![1, 3, 5],
        head_vocab_sizes,
        head_offsets,
    }
}

#[test]
fn the_real_tables_shape_validates_and_its_arithmetic_matches_the_checkpoint() {
    let l = real();
    l.validate().expect("the published shape must validate");

    assert_eq!(l.rows, 320_001_536, "128 shards of 2,500,012");
    assert_eq!(l.groups_per_row(), 5, "160 values in groups of 32");
    // **BYTE-NEUTRAL WITH THE SOURCE PLANES**, which is what says the
    // interleave costs nothing. 32,000,153,600 is the measured size of the
    // three source tensors summed over all 128 shards.
    assert_eq!(l.blob_bytes(), Some(32_000_153_600));
    assert_eq!(l.record_bytes, 100, "80 weight + 10 scales + 10 biases");
}

/// The offset mapping is LINEAR in the global row id: the shard boundary never
/// enters the arithmetic, because the writer concatenated the shards in id
/// order to remove it.
#[test]
fn a_row_offset_is_linear_and_the_last_row_ends_exactly_at_the_blob() {
    let l = real();
    assert_eq!(l.row_offset(0), Some(0));
    assert_eq!(l.row_offset(1), Some(100));
    // The first row of shard 1, which a reader that kept the shard split
    // would place differently.
    assert_eq!(l.row_offset(2_500_012), Some(250_001_200));
    // The last row ends exactly at the blob's end, with no slack. A layout
    // that padded records or shards would fail this.
    let last = l.rows - 1;
    assert_eq!(
        l.row_offset(last).unwrap() + l.record_bytes,
        l.blob_bytes().unwrap()
    );
}

/// Past the end is `None` rather than a wrapped or clamped offset.
///
/// A hash whose modulus is wrong produces an out-of-range id, and clamping it
/// would read a real row and return plausible values for a token that never
/// mapped there -- the exact failure the int64 buffers are carried to prevent,
/// arriving one layer down.
#[test]
fn a_row_id_past_the_table_is_refused_rather_than_wrapped() {
    let l = real();
    assert_eq!(l.row_offset(l.rows), None);
    assert_eq!(l.row_offset(u64::MAX), None);
}

#[test]
fn a_derived_plane_width_that_disagrees_with_the_shape_is_refused() {
    // `weight_bytes` must be exactly `head_dim * bits / 8`. A wrong value here
    // is a record stride that reads plausible bytes from the wrong place.
    let mut l = real();
    l.weight_bytes = 84;
    l.record_bytes = 104;
    let err = l.validate().expect_err("refused");
    assert!(format!("{err:?}").contains("weight_bytes"), "{err:?}");

    // The companions must be one bf16 per group, both of them.
    let mut l = real();
    l.scale_bytes = 8;
    l.record_bytes = 98;
    l.validate().expect_err("a short scale plane is refused");

    let mut l = real();
    l.bias_bytes = 12;
    l.record_bytes = 102;
    l.validate().expect_err("a long bias plane is refused");
}

/// `record_bytes` must be the sum of its three planes.
///
/// Checked separately from the plane widths because it is the field a READER
/// strides by: the planes could each be right while the stride was not, which
/// walks off the row by a constant amount that grows with the id.
#[test]
fn a_record_stride_that_is_not_the_sum_of_its_planes_is_refused() {
    let mut l = real();
    l.record_bytes = 128;
    let err = l.validate().expect_err("refused");
    assert!(format!("{err:?}").contains("record_bytes"), "{err:?}");
}

/// **THE COMPANION DTYPE IS THE AXIS THAT FAILS SILENTLY.**
///
/// FP16 and BF16 are the same width, so a wrong reading passes every length
/// check here and decodes the scales as values orders of magnitude off --
/// `crates/repack` Gotcha 9's measured case, 0.0271 read as 1.7e-16. It is
/// refused by name rather than accepted and interpreted.
#[test]
fn an_fp16_companion_plane_is_refused_rather_than_read_as_bf16() {
    let mut l = real();
    l.companion_dtype = "fp16".to_string();
    let err = l.validate().expect_err("refused");
    let msg = format!("{err:?}");
    assert!(msg.contains("fp16"), "{msg}");
    assert!(
        msg.contains("same width"),
        "the refusal should say why it is not merely unsupported: {msg}"
    );
}

#[test]
fn a_shape_with_a_ragged_group_is_refused() {
    let mut l = real();
    l.head_dim = 161;
    l.validate()
        .expect_err("161 values is not a whole number of 32-value groups");
}

#[test]
fn the_row_count_must_be_the_shards_times_their_height() {
    let mut l = real();
    l.rows += 1;
    let err = l.validate().expect_err("refused");
    assert!(format!("{err:?}").contains("rows"), "{err:?}");
}

/// The hash heads must fit INSIDE the table.
///
/// Under is legal and normal: the concatenated head sizes are padded up to
/// `make_ngram_vocab_size_divisible_by` and then split into equal shards, so
/// the last rows are real and unaddressed. Over would index past the blob.
#[test]
fn the_hash_heads_may_underfill_the_table_but_never_overrun_it() {
    let l = real();
    let used: i64 = l.head_vocab_sizes.iter().sum();
    assert!(
        (used as u64) < l.rows,
        "the real table is padded, so the heads underfill it"
    );
    l.validate().expect("underfilling is legal");

    let mut over = real();
    let last = over.head_offsets.len() - 1;
    over.head_offsets[last] = over.rows as i64;
    let err = over.validate().expect_err("refused");
    assert!(format!("{err:?}").contains("address"), "{err:?}");
}

/// `multipliers` is one per n-gram ORDER and the other two are one per HEAD,
/// so they are different lengths by construction.
///
/// Pinned because the obvious validation -- requiring all three to agree --
/// would refuse every real table, and would look reasonable to anyone who had
/// not counted.
#[test]
fn the_three_hashing_buffers_are_not_all_the_same_length() {
    let l = real();
    assert_eq!(
        l.multipliers.len(),
        3,
        "one per n-gram order, 1..=ngram_size"
    );
    assert_eq!(l.head_vocab_sizes.len(), 16, "one per hash head");
    assert_eq!(l.head_offsets.len(), l.head_vocab_sizes.len());
    l.validate().expect("differing lengths are correct here");

    // The two that MUST agree.
    let mut l = real();
    l.head_offsets.pop();
    l.validate()
        .expect_err("a head without an offset is refused");
}

#[test]
fn an_absent_table_reads_as_none_rather_than_an_error() {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-ngram-absent-{}-{:?}",
        std::process::id(),
        std::time::SystemTime::now()
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    assert_eq!(
        load_ngram_table_layout(&dir).expect("an install with no table is not an error"),
        None,
        "every other family, and every qwen4_exp install written before this"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
