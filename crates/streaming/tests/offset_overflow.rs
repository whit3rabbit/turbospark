//! Offset arithmetic in the streamer chain must refuse a wrap, not take it.
//!
//! The release profile has no overflow checks, so a wrapped `offset + stride`
//! comes out small, passes the `<= stream_size` bound, and the pread lands on
//! valid-but-wrong bytes. A decoded `layout.json` cannot produce these values
//! (model-io bounds every expert range with checked math), but `StreamLayout`
//! and `LayerLayout` have public fields, so the streamer cannot lean on that.
//! Under the debug test profile an unchecked add panics instead, so each test
//! here fails on the old code by panicking and passes only on a refusal.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_streaming::{
    coalesced_adjacent_advice_ranges, ExpertCachePolicy, PreadExpertStreamer, StreamLayout,
    StreamerError,
};

const STRIDE: u64 = 8;
const HEADER: u64 = 16;

/// A 16-byte header of 0xEE followed by two experts of 0x00 and 0x01. A read
/// that wraps back into the header would return 0xEE bytes as an expert.
fn write_file() -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-streaming-overflow-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("layer_00.bin");
    let mut bytes = vec![0xEE; HEADER as usize];
    bytes.extend(std::iter::repeat_n(0u8, STRIDE as usize));
    bytes.extend(std::iter::repeat_n(1u8, STRIDE as usize));
    std::fs::write(&path, bytes).unwrap();
    path
}

/// Expert 0's explicit offset is -8 modulo 2^64: `offset + stride` wraps to 0
/// and `stream_offset + offset` wraps to 8, inside the header.
fn wrapping_layout(path: &std::path::Path) -> StreamLayout {
    StreamLayout {
        path: path.display().to_string(),
        stream_offset: HEADER,
        stream_size: 2 * STRIDE,
        experts_per_layer: 2,
        expert_stride: STRIDE,
        expert_offsets: Some(vec![0u64.wrapping_sub(STRIDE), STRIDE]),
    }
}

fn is_out_of_range<T>(result: Result<T, StreamerError>) -> bool {
    matches!(result, Err(StreamerError::OffsetOutOfRange { .. }))
}

#[test]
fn direct_load_refuses_a_wrapping_expert_offset() {
    let path = write_file();
    let mut streamer =
        PreadExpertStreamer::open(wrapping_layout(&path), 2, ExpertCachePolicy::Lfu).unwrap();
    assert!(is_out_of_range(streamer.load_expert_into_slot(0, 0, 0)));
    // The neighbor is untouched by the refusal.
    let slot = streamer.load_expert(0, 1).unwrap();
    assert!(streamer.slot_data(slot).iter().all(|&b| b == 1));
}

#[test]
fn cached_load_refuses_a_wrapping_expert_offset() {
    let path = write_file();
    let mut streamer =
        PreadExpertStreamer::open(wrapping_layout(&path), 2, ExpertCachePolicy::Lfu).unwrap();
    let plan = streamer.plan_experts_cached(&[0], &HashSet::new());
    assert!(is_out_of_range(streamer.execute_expert_cache_plan(&plan)));
}

#[test]
fn advice_drops_a_wrapping_expert_offset() {
    let path = write_file();
    let streamer =
        PreadExpertStreamer::open(wrapping_layout(&path), 2, ExpertCachePolicy::Lfu).unwrap();
    // Expert 0 is dropped as out of range rather than advised at offset 8.
    let result = streamer.advise_experts(&[0]);
    assert_eq!(result.requested, 1);
    assert_eq!(result.skipped, 1, "{result:?}");
    assert_eq!(result.bytes, 0, "{result:?}");
}

#[test]
fn uniform_formula_refuses_a_layer_index_that_wraps() {
    let path = write_file();
    let mut layout = wrapping_layout(&path);
    layout.expert_offsets = None;
    let mut streamer = PreadExpertStreamer::open(layout, 2, ExpertCachePolicy::Lfu).unwrap();
    // `layer * (2 * 8)` with layer = 2^60 is 2^64, which wraps to 0 and would
    // read expert 0 for a layer that does not exist.
    assert!(is_out_of_range(streamer.load_expert(1usize << 60, 0)));
}

#[test]
fn open_refuses_a_window_that_wraps() {
    let path = write_file();
    let mut layout = wrapping_layout(&path);
    layout.stream_offset = u64::MAX;
    assert!(is_out_of_range(PreadExpertStreamer::open(
        layout,
        1,
        ExpertCachePolicy::Lfu
    )));
}

#[test]
fn packed_layer_window_saturates_instead_of_wrapping() {
    let layer = model_io::LayerLayout {
        layer: 0,
        file: "layer_00.bin".to_string(),
        expert_stride: STRIDE,
        experts: vec![model_io::ExpertEntry {
            expert: 0,
            offset: u64::MAX,
            size: STRIDE,
            sub_tensors: Default::default(),
        }],
    };
    let stream = StreamLayout::from_packed_experts_layer(&layer, std::path::Path::new("/x"));
    // A wrapped window would be 7 bytes and a short file would pass `open`.
    assert_eq!(stream.stream_size, u64::MAX);
}

#[test]
fn advice_coalescing_does_not_wrap() {
    let merged = coalesced_adjacent_advice_ranges(&[(u64::MAX - 1, 4), (u64::MAX, 4)]);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].0, u64::MAX - 1);
}
