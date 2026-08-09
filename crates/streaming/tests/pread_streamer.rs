//! End-to-end tests for `PreadExpertStreamer` against a real (synthetic)
//! layer file on disk.

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_streaming::{ExpertCachePolicy, PreadExpertStreamer, StreamLayout};

const EXPERT_STRIDE: u64 = 64;
const EXPERTS_PER_LAYER: usize = 4;

fn write_layer_file() -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-streaming-{}-{unique}",
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
        turbospark_streaming::StreamerError::OffsetOutOfRange { .. }
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
        "turbospark-streaming-big-{}-{unique}",
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

/// The stream layout takes its stride from the LAYER, not from a caller's
/// model-wide number (ROADMAP Phase S).
///
/// It used to be a parameter, and every install to date is uniform across
/// layers, so a caller passing the model-wide maximum was indistinguishable
/// from a correct one. On a mixed sub-4-bit install it is not: the candidate
/// there has one layer 1.6x the other twenty-nine, and a uniform reader
/// over-reads all twenty-nine by that factor on every cache miss.
///
/// Paired with `each_layer_is_padded_to_its_own_stride_not_the_model_wide_maximum`
/// in `crates/repack/tests/gturbo_writer.rs`: that one proves the bytes are
/// written narrow, this one proves they are addressed narrow.
#[test]
fn the_stream_layout_takes_its_stride_from_the_layer() {
    const STRIDE: u64 = 2048;
    let layer = model_io::LayerLayout {
        layer: 0,
        file: "layer_00.bin".to_string(),
        expert_stride: STRIDE,
        experts: (0..3)
            .map(|e| model_io::ExpertEntry {
                expert: e,
                offset: e as u64 * STRIDE,
                size: STRIDE,
                sub_tensors: Default::default(),
            })
            .collect(),
    };
    let layout = StreamLayout::from_packed_experts_layer(&layer, std::path::Path::new("/tmp"));
    assert_eq!(layout.expert_stride, STRIDE);
    assert_eq!(layout.stream_size, 3 * STRIDE);
    assert_eq!(layout.expert_offset(0, 2), 2 * STRIDE);
}
