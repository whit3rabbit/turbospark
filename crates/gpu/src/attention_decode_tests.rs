use super::{chunks_for, MAX_CHUNKS, MIN_POSITIONS_PER_CHUNK};

/// The parity tests exercise the chunked path but cannot pin this
/// mapping: a `chunks_for` that always returned 1 would still match
/// the CPU reference, just slowly. These are the boundaries that
/// matter -- never 0 (the kernel takes `% num_chunks`), never above
/// the scratch's capacity, and 1 while the range is short enough that
/// splitting would cost more than it buys.
#[test]
fn chunk_count_stays_within_its_bounds() {
    for range in [1u32, 2, 15, 16, 31, 32, 255, 256, 1024, 4096, u32::MAX] {
        let chunks = chunks_for(range);
        assert!(chunks >= 1, "range {range}: {chunks}");
        assert!(chunks <= MAX_CHUNKS, "range {range}: {chunks}");
        assert!(chunks.is_power_of_two(), "range {range}: {chunks}");
        assert!(
            chunks == 1 || chunks * MIN_POSITIONS_PER_CHUNK <= range,
            "range {range} split {chunks} ways leaves chunks under {MIN_POSITIONS_PER_CHUNK}"
        );
    }
}

#[test]
fn short_ranges_stay_unsplit_and_long_ones_saturate() {
    assert_eq!(chunks_for(0), 1);
    assert_eq!(chunks_for(MIN_POSITIONS_PER_CHUNK * 2 - 1), 1);
    assert_eq!(chunks_for(MIN_POSITIONS_PER_CHUNK * 2), 2);
    assert_eq!(chunks_for(MIN_POSITIONS_PER_CHUNK * MAX_CHUNKS), MAX_CHUNKS);
    assert_eq!(chunks_for(u32::MAX), MAX_CHUNKS);
}
