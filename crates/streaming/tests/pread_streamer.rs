//! End-to-end tests for `PreadExpertStreamer` against a real (synthetic)
//! layer file on disk.

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

use mrefrust_streaming::{ExpertCachePolicy, PreadExpertStreamer, StreamLayout};

const EXPERT_STRIDE: u64 = 64;
const EXPERTS_PER_LAYER: usize = 4;

fn write_layer_file() -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "mrefrust-streaming-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("layer_00.bin");

    // Expert `e`'s blob is EXPERT_STRIDE bytes, every byte equal to `e`, so
    // reads can be verified by content alone.
    let mut file = std::fs::File::create(&path).unwrap();
    for e in 0..EXPERTS_PER_LAYER {
        file.write_all(&vec![e as u8; EXPERT_STRIDE as usize])
            .unwrap();
    }
    path
}

fn layout(path: &std::path::Path) -> StreamLayout {
    StreamLayout {
        path: path.display().to_string(),
        stream_offset: 0,
        stream_size: EXPERT_STRIDE * EXPERTS_PER_LAYER as u64,
        experts_per_layer: EXPERTS_PER_LAYER,
        expert_stride: EXPERT_STRIDE,
        expert_offsets: None,
    }
}

#[test]
fn load_expert_reads_the_correct_blob() {
    let path = write_layer_file();
    let mut streamer = PreadExpertStreamer::open(layout(&path), 2, ExpertCachePolicy::Lfu).unwrap();
    let slot = streamer.load_expert(0, 2).unwrap();
    assert!(streamer.slot_data(slot).iter().all(|&b| b == 2));
}

#[test]
fn load_experts_cached_reuses_slots_on_repeat_request() {
    let path = write_layer_file();
    let mut streamer = PreadExpertStreamer::open(layout(&path), 4, ExpertCachePolicy::Lfu).unwrap();

    let slots1 = streamer.load_experts_cached(&[0, 1]).unwrap();
    let slots2 = streamer.load_experts_cached(&[0]).unwrap();
    assert_eq!(slots1[0], slots2[0]);
    for (i, &slot) in slots1.iter().enumerate() {
        assert!(streamer.slot_data(slot).iter().all(|&b| b == i as u8));
    }
}

#[test]
fn offset_beyond_stream_size_is_rejected() {
    let path = write_layer_file();
    let mut streamer = PreadExpertStreamer::open(layout(&path), 2, ExpertCachePolicy::Lfu).unwrap();
    let err = streamer.load_expert(0, EXPERTS_PER_LAYER + 10).unwrap_err();
    assert!(matches!(
        err,
        mrefrust_streaming::StreamerError::OffsetOutOfRange { .. }
    ));
}

#[test]
fn speculative_reservation_reads_and_publishes() {
    let path = write_layer_file();
    let mut streamer = PreadExpertStreamer::open(layout(&path), 4, ExpertCachePolicy::Lfu).unwrap();

    let reservation = streamer.reserve_speculative_slots(&[3], 0);
    assert_eq!(reservation.len(), 1);
    let bytes = streamer.execute_speculative_reservation(&reservation);
    assert_eq!(bytes, EXPERT_STRIDE);

    let resident = streamer.resident_experts_snapshot();
    assert!(resident.contains(&Some(3)));
}

#[test]
fn advise_experts_reports_one_coalesced_call_for_adjacent_experts() {
    let path = write_layer_file();
    let streamer = PreadExpertStreamer::open(layout(&path), 4, ExpertCachePolicy::Lfu).unwrap();
    let result = streamer.advise_experts(&[0, 1]);
    assert_eq!(result.requested, 2);
    assert_eq!(result.calls, 1);
}

/// A stride large enough that one expert's blob splits into several
/// parallel read chunks, which the 64-byte fixtures above never do.
///
/// The blob is a byte pattern with period 251 (coprime with any power of
/// two, so it cannot align with a chunk boundary): a chunk read at the
/// wrong offset, dropped, or written to the wrong place shows up as a
/// mismatched byte rather than an accidentally-identical one. Two misses
/// are requested at once so both the multi-chunk and multi-slot
/// disjointness paths run together.
#[test]
fn multi_chunk_reads_reassemble_each_blob_exactly() {
    const BIG_STRIDE: u64 = 3 * 1024 * 1024;
    const EXPERTS: usize = 3;

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "mrefrust-streaming-big-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("layer_00.bin");

    let blob = |e: usize| -> Vec<u8> {
        (0..BIG_STRIDE as usize)
            .map(|i| ((i % 251) as u8).wrapping_add(e as u8 * 7))
            .collect()
    };
    let mut file = std::fs::File::create(&path).unwrap();
    for e in 0..EXPERTS {
        file.write_all(&blob(e)).unwrap();
    }
    drop(file);

    let layout = StreamLayout {
        path: path.display().to_string(),
        stream_offset: 0,
        stream_size: BIG_STRIDE * EXPERTS as u64,
        experts_per_layer: EXPERTS,
        expert_stride: BIG_STRIDE,
        expert_offsets: None,
    };
    let mut streamer = PreadExpertStreamer::open(layout, EXPERTS, ExpertCachePolicy::Lfu).unwrap();

    let slots = streamer.load_experts_cached(&[2, 0]).unwrap();
    assert_eq!(streamer.slot_data(slots[0]), blob(2).as_slice());
    assert_eq!(streamer.slot_data(slots[1]), blob(0).as_slice());

    // Single miss: the case that used to run fully single-threaded, and
    // the one the chunk split exists for.
    let solo = streamer.load_experts_cached(&[1]).unwrap();
    assert_eq!(streamer.slot_data(solo[0]), blob(1).as_slice());

    std::fs::remove_dir_all(&dir).ok();
}
