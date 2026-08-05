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
