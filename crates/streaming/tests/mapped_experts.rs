//! `MappedExpertLayer` against a real (synthetic) layer file on disk.
//!
//! The load-bearing case is `the_mapped_bytes_are_the_bytes_the_streamer_copies`:
//! mapped residency is only a throughput and footprint change if the bytes it
//! hands the kernels are the bytes the `pread` path would have copied. Every
//! byte-identity claim downstream of this rests on it.

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_streaming::{
    ExpertCachePolicy, MappedExpertLayer, PreadExpertStreamer, StreamLayout,
};

const EXPERT_STRIDE: u64 = 64;
const EXPERTS_PER_LAYER: usize = 4;

/// Expert `e`'s blob is `EXPERT_STRIDE` bytes, every byte equal to `e`, so a
/// read can be verified by content alone -- and, more to the point here, a
/// read of the WRONG expert is visible rather than plausible.
fn write_layer_file(header_bytes: usize) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir =
        std::env::temp_dir().join(format!("turbospark-mapped-{}-{unique}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("layer_00.bin");
    let mut file = std::fs::File::create(&path).unwrap();
    // 0xFF, which is no expert's fill, so a reader that ignores the header
    // offset reads something identifiable rather than expert 0's bytes.
    file.write_all(&vec![0xFFu8; header_bytes]).unwrap();
    for e in 0..EXPERTS_PER_LAYER {
        file.write_all(&vec![e as u8; EXPERT_STRIDE as usize])
            .unwrap();
    }
    path
}

fn layout(path: &std::path::Path, stream_offset: u64) -> StreamLayout {
    StreamLayout {
        path: path.display().to_string(),
        stream_offset,
        stream_size: EXPERT_STRIDE * EXPERTS_PER_LAYER as u64,
        experts_per_layer: EXPERTS_PER_LAYER,
        expert_stride: EXPERT_STRIDE,
        expert_offsets: None,
    }
}

#[test]
fn the_mapped_bytes_are_the_bytes_the_streamer_copies() {
    let path = write_layer_file(0);
    let mapped = MappedExpertLayer::open(layout(&path, 0)).unwrap();
    let mut streamer =
        PreadExpertStreamer::open(layout(&path, 0), EXPERTS_PER_LAYER, ExpertCachePolicy::Lfu)
            .unwrap();

    for expert in 0..EXPERTS_PER_LAYER {
        let slot = streamer.load_expert(0, expert).unwrap();
        let copied = streamer.slot_data(slot);
        let in_place = mapped.expert_bytes(expert).expect("expert is in range");
        assert_eq!(
            copied, in_place,
            "expert {expert}: mapped bytes differ from the streamer's copy"
        );
    }
}

/// The fixture has to be able to SEE a wrong expert before the case above
/// means anything: if every expert held the same bytes, reading the wrong one
/// would pass. Gotchas 48 and 50's discipline, on a third axis.
#[test]
fn the_fixture_distinguishes_one_expert_from_another() {
    let path = write_layer_file(0);
    let mapped = MappedExpertLayer::open(layout(&path, 0)).unwrap();
    let first = mapped.expert_bytes(0).unwrap().to_vec();
    for expert in 1..EXPERTS_PER_LAYER {
        assert_ne!(
            first,
            mapped.expert_bytes(expert).unwrap(),
            "expert {expert} is byte-identical to expert 0; this fixture cannot \
             tell a correct read from a wrong one"
        );
    }
}

/// The writer need not emit dense `expert * stride` offsets, which is why
/// `StreamLayout` carries an explicit table. A mapped read must honour it.
#[test]
fn a_permuted_offset_table_is_honoured() {
    let path = write_layer_file(0);
    let mut permuted = layout(&path, 0);
    // Expert 0 lives where expert 3's blob is written, and vice versa.
    permuted.expert_offsets = Some(vec![3 * EXPERT_STRIDE, EXPERT_STRIDE, 2 * EXPERT_STRIDE, 0]);
    let mapped = MappedExpertLayer::open(permuted).unwrap();
    assert!(mapped.expert_bytes(0).unwrap().iter().all(|&b| b == 3));
    assert!(mapped.expert_bytes(3).unwrap().iter().all(|&b| b == 0));
}

/// `ResidentBuffer` rounds the file offset DOWN to a page boundary so the
/// mapping base is page-aligned, and reports the difference. Every expert
/// offset is relative to that aligned base, so the shift has to be added.
///
/// This is the case that makes the `shift` term load-bearing: drop it and
/// expert 0 reads the file's header instead of its own blob, which is a wrong
/// read rather than an error.
#[test]
fn a_non_zero_stream_offset_is_shifted_onto_the_aligned_base() {
    const HEADER: usize = 16;
    let path = write_layer_file(HEADER);
    let mapped = MappedExpertLayer::open(layout(&path, HEADER as u64)).unwrap();

    // The mapping starts at the page boundary below the header, so the shift
    // must be exactly the header on any page size over 16 bytes.
    assert_eq!(mapped.expert_offset(0), HEADER as u64);
    for expert in 0..EXPERTS_PER_LAYER {
        assert!(
            mapped
                .expert_bytes(expert)
                .unwrap()
                .iter()
                .all(|&b| b == expert as u8),
            "expert {expert} read the header or a neighbour rather than its blob"
        );
    }
}

/// Mirrors `PreadExpertStreamer::open`'s check and for its reason: a short
/// file is where a bad layout shows up, and the alternative to failing here is
/// an out-of-range expert offset discovered far downstream.
#[test]
fn a_short_file_is_refused_at_open() {
    let path = write_layer_file(0);
    let mut oversized = layout(&path, 0);
    oversized.stream_size *= 4;
    assert!(MappedExpertLayer::open(oversized).is_err());
}

#[test]
fn an_out_of_range_expert_is_none_rather_than_a_panic() {
    let path = write_layer_file(0);
    let mapped = MappedExpertLayer::open(layout(&path, 0)).unwrap();
    assert!(mapped.expert_bytes(EXPERTS_PER_LAYER + 10).is_none());
}
