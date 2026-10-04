//! Bounded prefetch reads real ranges without modifying slot-cache ownership.

use std::collections::HashSet;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_streaming::{
    ExpertCachePolicy, MappedExpertLayer, PreadExpertStreamer, StreamLayout,
};

struct Fixture {
    dir: std::path::PathBuf,
    layout: StreamLayout,
}

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "turbospark-expert-prefetch-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("layer.bin");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&[0xff; 16]).unwrap();
        for value in [3u8, 7, 11] {
            file.write_all(&vec![value; 64 * 1024]).unwrap();
        }
        Self {
            dir,
            layout: StreamLayout {
                path: path.display().to_string(),
                stream_offset: 16,
                stream_size: 3 * 64 * 1024,
                experts_per_layer: 3,
                expert_stride: 64 * 1024,
                expert_offsets: Some(vec![2 * 64 * 1024, 0, 64 * 1024]),
            },
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.dir).unwrap();
    }
}

#[test]
fn streamed_prefetch_honors_offsets_and_leaves_cache_and_slots_untouched() {
    let fixture = Fixture::new();
    let mut streamer =
        PreadExpertStreamer::open(fixture.layout.clone(), 2, ExpertCachePolicy::Lfu).unwrap();
    let slot = streamer.load_experts_cached(&[1]).unwrap()[0];
    let before = streamer.resident_experts_snapshot();
    let slot_before = streamer.slot_data(slot).to_vec();
    let mut scratch = vec![0u8; 997];
    if streamer.nocache_active() {
        assert!(streamer.prefetch_expert(0, &mut scratch).is_err());
        return;
    }
    assert_eq!(
        streamer.prefetch_expert(0, &mut scratch).unwrap(),
        64 * 1024
    );
    // The selected expert is stored last, and each chunk must start there.
    assert!(scratch.iter().all(|&byte| byte == 11));
    assert_eq!(before, streamer.resident_experts_snapshot());
    assert_eq!(slot_before, streamer.slot_data(slot));
    assert_eq!(
        streamer
            .plan_experts_cached(&[0], &HashSet::new())
            .misses
            .len(),
        1
    );
}

#[test]
fn a_truncated_prefetch_is_an_error_without_publishing_cache_residency() {
    let fixture = Fixture::new();
    let streamer =
        PreadExpertStreamer::open(fixture.layout.clone(), 2, ExpertCachePolicy::Lfu).unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&fixture.layout.path)
        .unwrap()
        .set_len(16 + 2 * 64 * 1024 + 40)
        .unwrap();
    assert!(streamer.prefetch_expert(0, &mut [0u8; 128]).is_err());
    assert!(streamer
        .resident_experts_snapshot()
        .iter()
        .all(Option::is_none));
}

#[test]
fn both_prefetch_paths_refuse_a_missing_explicit_offset() {
    let mut fixture = Fixture::new();
    fixture.layout.expert_offsets = Some(vec![0]);
    let streamer =
        PreadExpertStreamer::open(fixture.layout.clone(), 2, ExpertCachePolicy::Lfu).unwrap();
    let mapped = MappedExpertLayer::open(fixture.layout.clone()).unwrap();
    assert!(streamer.prefetch_expert(1, &mut [0u8; 64]).is_err());
    assert!(mapped.prefetch_expert(1).is_err());
    assert!(streamer.prefetch_expert(3, &mut [0u8; 64]).is_err());
    assert!(mapped.prefetch_expert(3).is_err());
}

#[test]
fn both_prefetch_paths_refuse_overflowing_explicit_ranges() {
    let mut fixture = Fixture::new();
    fixture.layout.expert_offsets = Some(vec![u64::MAX, 0, 64 * 1024]);
    let streamer =
        PreadExpertStreamer::open(fixture.layout.clone(), 2, ExpertCachePolicy::Lfu).unwrap();
    let mapped = MappedExpertLayer::open(fixture.layout.clone()).unwrap();
    assert!(streamer.prefetch_expert(0, &mut [0u8; 64]).is_err());
    assert!(mapped.prefetch_expert(0).is_err());
}

#[test]
fn mapped_prefetch_touches_pages_without_changing_expert_bytes() {
    let fixture = Fixture::new();
    let mapped = MappedExpertLayer::open(fixture.layout.clone()).unwrap();
    let before = mapped.expert_bytes(0).unwrap().to_vec();
    let touched = mapped.prefetch_expert(0).unwrap();
    assert!(touched > 1, "the fixture crosses VM pages");
    assert_eq!(before, mapped.expert_bytes(0).unwrap());
    assert!(before.iter().all(|&byte| byte == 11));
}
