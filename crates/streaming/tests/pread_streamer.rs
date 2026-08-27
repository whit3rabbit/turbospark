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

/// A direct load writes a slot's bytes outside any cache plan, so the
/// cache must stop believing that slot holds what it held.
///
/// Without the invalidation the second request scores a HIT and hands back
/// the slot the direct load overwrote, so the caller computes with a
/// different expert's weights and nothing errors -- Gotcha 27's failure
/// mode reached through the API instead of through dispatch order. The two
/// paths are never mixed in production today, but both are `pub` and
/// nothing but this test says they may not be.
#[test]
fn a_direct_load_leaves_no_stale_residency_for_the_cache_to_hit_on() {
    let path = write_layer_file();
    let mut streamer = PreadExpertStreamer::open(layout(&path), 2, ExpertCachePolicy::Lfu).unwrap();

    let cached = streamer.load_experts_cached(&[0]).unwrap();
    let slot = cached[0];
    assert!(streamer.slot_data(slot).iter().all(|&b| b == 0));

    // Overwrite that slot with a different expert, behind the cache's back.
    streamer.load_expert_into_slot(0, 3, slot).unwrap();
    assert!(streamer.slot_data(slot).iter().all(|&b| b == 3));

    // Whichever slot the plan picks now, it has to contain expert 0's bytes.
    let again = streamer.load_experts_cached(&[0]).unwrap();
    assert!(streamer.slot_data(again[0]).iter().all(|&b| b == 0));
}

/// A read that fails inside the parallel pool must return an error rather
/// than hang, and must leave the cache claiming nothing.
///
/// This is the only test that drives `read_pool`'s error path at all. The
/// whole safety argument rests on `run_batch` blocking until every claim
/// drops and on `Claim`'s `Drop` (not the happy path) signalling
/// completion, so a failing read that returned early from the worker loop
/// would deadlock the submitter here. A hang IS this test's failure mode;
/// there is nothing to assert about it beyond reaching the next line.
///
/// The file is truncated AFTER open, because `open` refuses a short file up
/// front, and the stride is chosen to split into several chunks so the
/// pool runs rather than `run_batch`'s single-chunk inline shortcut.
#[test]
fn a_failed_pooled_read_returns_an_error_and_commits_nothing() {
    const BIG_STRIDE: u64 = 3 * 1024 * 1024;
    const EXPERTS: usize = 2;

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-streaming-trunc-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("layer_00.bin");

    let mut file = std::fs::File::create(&path).unwrap();
    for e in 0..EXPERTS {
        file.write_all(&vec![e as u8; BIG_STRIDE as usize]).unwrap();
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
    let mut streamer = PreadExpertStreamer::open(layout, 2, ExpertCachePolicy::Lfu).unwrap();

    // Expert 1's blob is now almost entirely past EOF; expert 0's is intact.
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(BIG_STRIDE + 8)
        .unwrap();

    let err = streamer.load_experts_cached(&[1]).unwrap_err();
    assert!(
        matches!(
            err,
            turbospark_streaming::StreamerError::SizeMismatch { .. }
                | turbospark_streaming::StreamerError::PreadFailed { .. }
        ),
        "unexpected error: {err}"
    );
    assert!(
        !streamer.resident_experts_snapshot().contains(&Some(1)),
        "a failed read must not commit its plan"
    );

    // The streamer is still usable, and expert 1 still misses.
    let intact = streamer.load_experts_cached(&[0]).unwrap();
    assert!(streamer.slot_data(intact[0]).iter().all(|&b| b == 0));

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

/// The subdirectory is a PARAMETER, and it reaches the path and nothing else
/// (ROADMAP M-V3).
///
/// The vision tower streams out of `packed_vision/blobs.bin` through this
/// same type, which is the whole reuse claim -- `StreamLayout` interprets
/// nothing, so a "layer" is a file and an "expert" is a fixed-stride blob
/// inside it, and a block loop wants exactly that.
///
/// Asserted by DIFFERENCE rather than by checking one path, because the thing
/// that could break is the subdir being ignored: hardcode `packed_experts`
/// back into `from_packed_layer_in` and the two paths below become equal
/// while every other field still agrees.
#[test]
fn the_packed_subdirectory_reaches_the_path_and_nothing_else() {
    const STRIDE: u64 = 4096;
    let layer = model_io::LayerLayout {
        layer: 0,
        file: "blobs.bin".to_string(),
        expert_stride: STRIDE,
        experts: (0..2)
            .map(|e| model_io::ExpertEntry {
                expert: e,
                offset: e as u64 * STRIDE,
                size: STRIDE,
                sub_tensors: Default::default(),
            })
            .collect(),
    };
    let dir = std::path::Path::new("/tmp");
    let experts = StreamLayout::from_packed_layer_in(&layer, dir, model_io::PACKED_EXPERTS_DIR);
    let vision = StreamLayout::from_packed_layer_in(&layer, dir, model_io::PACKED_VISION_DIR);

    assert!(
        vision.path.ends_with("packed_vision/blobs.bin"),
        "{}",
        vision.path
    );
    assert_ne!(
        experts.path, vision.path,
        "the subdirectory did not reach the path"
    );
    // Everything else is identical, which is what says the parameter is a
    // path and not a second format.
    assert_eq!(experts.expert_stride, vision.expert_stride);
    assert_eq!(experts.stream_size, vision.stream_size);
    assert_eq!(experts.experts_per_layer, vision.experts_per_layer);
    assert_eq!(experts.expert_offsets, vision.expert_offsets);
    // And the default-subdir constructor still agrees with the explicit one.
    assert_eq!(
        StreamLayout::from_packed_experts_layer(&layer, dir).path,
        experts.path
    );
}

/// The window spans the highest OFFSET, not `expert count * stride`.
///
/// Every writer to date emits dense `e * stride` offsets, on which the two
/// formulas are equal -- which is exactly what makes this untestable
/// against a real install and worth a fixture. The offsets here are
/// permuted and sparse, as the doc on `from_packed_experts_layer` says a
/// packed layout may be; under the count-based size the last expert's
/// bounds check would reject it and blame the offset.
#[test]
fn the_stream_window_spans_the_highest_offset_not_the_expert_count() {
    const STRIDE: u64 = 2048;
    let offsets = [4 * STRIDE, 0, 2 * STRIDE];
    let layer = model_io::LayerLayout {
        layer: 0,
        file: "layer_00.bin".to_string(),
        expert_stride: STRIDE,
        experts: offsets
            .iter()
            .enumerate()
            .map(|(e, &offset)| model_io::ExpertEntry {
                expert: e,
                offset,
                size: STRIDE,
                sub_tensors: Default::default(),
            })
            .collect(),
    };
    let layout = StreamLayout::from_packed_experts_layer(&layer, std::path::Path::new("/tmp"));

    assert_eq!(layout.stream_size, 5 * STRIDE);
    assert_eq!(layout.expert_offset(0, 0), 4 * STRIDE);
    // The count-based window would have been 3 * STRIDE, i.e. too small for
    // expert 0 by two whole strides.
    assert!(layout.expert_offset(0, 0) + STRIDE <= layout.stream_size);
}
