use super::{affine_group_size, ring_spans};
use model_io::ResidentIndexEntry;

/// The real checkpoints' shape, in miniature: `bits` per element, one FP16
/// scale and one FP16 bias per group.
fn entry(rows: usize, cols: usize, group: usize, bits: usize) -> ResidentIndexEntry {
    let groups = rows * (cols / group);
    ResidentIndexEntry {
        name: "w".to_string(),
        dtype: if bits == 1 { 15 } else { 16 },
        file_offset: 4096,
        size_bytes: (rows * cols * bits / 8) as u64,
        shape: (rows as u32, cols as u32, 0, 0),
        scale_offset: 8192,
        scale_size: (groups * 2) as u64,
        bias_offset: 16384,
        bias_size: (groups * 2) as u64,
    }
}

/// THE POINT OF THE DERIVATION: the group size comes off the tensor, so
/// two tensors in one install may disagree and neither has to match a
/// constant somebody wrote down. 128 is the published checkpoints'; 64 is
/// what a per-checkpoint constant would have forced on this row.
#[test]
fn the_group_size_is_read_off_the_companion_planes() {
    assert_eq!(
        affine_group_size(&entry(8, 256, 128, 1), "w", 8, 256, 1).unwrap(),
        128
    );
    assert_eq!(
        affine_group_size(&entry(8, 256, 64, 1), "w", 8, 256, 1).unwrap(),
        64
    );
    assert_eq!(
        affine_group_size(&entry(3, 384, 128, 1), "w", 3, 384, 1).unwrap(),
        128
    );
}

/// The same derivation at TWO bits, where only the packed run moves.
///
/// The pair of assertions is the point: identical companion planes and a
/// packed run of exactly twice the size yield the same group size, which
/// is what says `bits` reaches the one conjunct it should and no other.
#[test]
fn the_derivation_takes_the_width_as_a_parameter() {
    assert_eq!(
        affine_group_size(&entry(8, 256, 128, 2), "w", 8, 256, 2).unwrap(),
        128
    );
    // ...and each width REFUSES the other's packed run, which is the only
    // thing that tells the two entry shapes apart.
    assert!(affine_group_size(&entry(8, 256, 128, 1), "w", 8, 256, 2).is_err());
    assert!(affine_group_size(&entry(8, 256, 128, 2), "w", 8, 256, 1).is_err());
}

/// Each conjunct is also a check on the install, and this is the one that
/// matters most: the two planes are the same width, so a companion region
/// that is half the size it should be passes every other length check.
#[test]
fn a_malformed_sub_four_bit_entry_is_refused_rather_than_dispatched() {
    let mut half_scales = entry(8, 256, 128, 1);
    half_scales.scale_size /= 2;
    assert!(affine_group_size(&half_scales, "w", 8, 256, 1).is_err());

    let mut no_companions = entry(8, 256, 128, 1);
    no_companions.scale_size = 0;
    no_companions.bias_size = 0;
    assert!(affine_group_size(&no_companions, "w", 8, 256, 1).is_err());

    // A shape the caller and the file disagree about.
    assert!(affine_group_size(&entry(8, 256, 128, 1), "w", 4, 256, 1).is_err());

    // A group that is not a whole number of bytes: the kernel ASSERTS
    // this, so reaching it would abort the process rather than error.
    let ragged = entry(1, 12, 4, 1);
    assert!(affine_group_size(&ragged, "w", 1, 12, 1).is_err());
}

/// The spans have to cover the write exactly and land on the physical
/// slots `position % capacity` names, or a batched projection writes
/// somebody else's rows.
#[test]
fn a_ring_write_splits_at_the_wrap_and_nowhere_else() {
    // Clear of the wrap: one span, and it is the whole write.
    assert_eq!(ring_spans(1152, 0, 16), [(0, 16), (16, 0)]);
    assert_eq!(ring_spans(1152, 1000, 16), [(0, 16), (16, 0)]);
    // Ending exactly ON the wrap still does not straddle.
    assert_eq!(ring_spans(1152, 1136, 16), [(0, 16), (16, 0)]);
    // Straddling: 4 rows before the wrap, 12 after.
    assert_eq!(ring_spans(1152, 1148, 16), [(0, 4), (4, 12)]);
    // A base past the first lap addresses by modulus, not by lap.
    assert_eq!(ring_spans(1152, 1152 + 1148, 16), [(0, 4), (4, 12)]);
    // Single-row writes never split, which is what makes the split a
    // no-op for every per-token caller.
    assert_eq!(ring_spans(1152, 1151, 1), [(0, 1), (1, 0)]);

    // The spans partition the write, and each row lands where
    // `physical_slot` would put it. Swept across a whole lap so no
    // single lucky base carries the claim.
    for base in 0..(2 * 1152) {
        for rows in 1..=16usize {
            let spans = ring_spans(1152, base, rows);
            assert_eq!(spans[0].1 + spans[1].1, rows, "base {base} rows {rows}");
            for (offset, count) in spans {
                if count == 0 {
                    continue;
                }
                // Contiguous from the span's own first physical slot,
                // which is what one GEMM writing `count` adjacent rows
                // assumes, and inside the buffer.
                let start = (base + offset) % 1152;
                assert!(start + count <= 1152, "base {base} rows {rows}");
                for row in 0..count {
                    assert_eq!(start + row, (base + offset + row) % 1152);
                }
            }
        }
    }
}

#[test]
#[should_panic(expected = "capacity must be positive")]
fn ring_spans_zero_capacity_panics() {
    let _ = ring_spans(0, 0, 1);
}

#[test]
#[should_panic(expected = "rows <= capacity")]
fn ring_spans_rows_exceeding_capacity_panics() {
    let _ = ring_spans(10, 0, 11);
}

#[test]
fn owned_rows_derives_group_size_and_slices_companion_planes() {
    let rows = 2;
    let cols = 128;
    let group_size = 64;
    let groups_per_row = cols / group_size;
    let total_groups = rows * groups_per_row;
    let packed_bytes = rows * cols / 2;
    let companion_bytes = total_groups * 2;

    let mut data = vec![0u8; 1024];
    let packed_off = 0;
    let scale_off = 256;
    let bias_off = 512;
    for i in 0..packed_bytes {
        data[packed_off + i] = (i + 1) as u8;
    }
    for i in 0..total_groups {
        data[scale_off + i * 2] = (10 + i) as u8;
        data[bias_off + i * 2] = (20 + i) as u8;
    }

    let mut entries = std::collections::HashMap::new();
    entries.insert(
        "test_tensor".to_string(),
        ResidentIndexEntry {
            name: "test_tensor".to_string(),
            dtype: 1,
            file_offset: 0,
            size_bytes: packed_bytes as u64,
            shape: (rows as u32, cols as u32, 0, 0),
            scale_offset: scale_off as u64,
            scale_size: companion_bytes as u64,
            bias_offset: bias_off as u64,
            bias_size: companion_bytes as u64,
        },
    );
    let index = model_io::ResidentIndex {
        header: model_io::ResidentIndexHeader {
            index_size: 0,
            resident_size: 1024,
            entry_count: 1,
        },
        entries,
    };

    let result = super::owned_rows(&index, &data, "test_tensor", rows, cols).unwrap();
    assert_eq!(result.len(), rows);
    assert_eq!(result[0].packed.len(), cols / 2);
    assert_eq!(result[0].scales.len(), groups_per_row);
    assert_eq!(result[0].biases.len(), groups_per_row);
    assert_eq!(result[1].scales.len(), groups_per_row);
    assert_eq!(result[0].scales[0], 10);
    assert_eq!(result[0].scales[1], 11);
    assert_eq!(result[1].scales[0], 12);
    assert_eq!(result[1].scales[1], 13);
}
