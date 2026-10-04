//! Demand-selected page preparation is bounded, read-only, and validated
//! before touching any selected range. These are CPU fixture checks, not
//! claims about physical I/O, residency, or model throughput.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_streaming::{
    MappedExpertLayer, MappedPagePreparationMode, StreamLayout, StreamerError,
};

fn page_size() -> u64 {
    // SAFETY: reads a process constant; the fixture must match the host VM.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    assert!(page > 32);
    page as u64
}

struct Fixture {
    dir: PathBuf,
    path: PathBuf,
    header: u64,
    stream_size: u64,
}

impl Fixture {
    fn new(header: u64, stream_size: u64) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "turbospark-page-preparation-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("experts.bin");
        let mut bytes: Vec<u8> = (0..header + stream_size)
            .map(|index| (index.wrapping_mul(17) % 251) as u8)
            .collect();
        bytes[..header as usize].fill(255);
        std::fs::write(&path, bytes).unwrap();
        Self {
            dir,
            path,
            header,
            stream_size,
        }
    }

    fn layout(&self, experts: usize, stride: u64, offsets: Option<Vec<u64>>) -> StreamLayout {
        StreamLayout {
            path: self.path.display().to_string(),
            stream_offset: self.header,
            stream_size: self.stream_size,
            experts_per_layer: experts,
            expert_stride: stride,
            expert_offsets: offsets,
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.dir).unwrap();
    }
}

fn overlapping_layer(fixture: &Fixture) -> MappedExpertLayer {
    let page = page_size();
    MappedExpertLayer::open(fixture.layout(
        3,
        32,
        Some(vec![
            page - fixture.header - 8,
            page - fixture.header + 16,
            4 * page - fixture.header + 16,
        ]),
    ))
    .unwrap()
}

#[test]
fn touch_deduplicates_ids_and_shared_pages_without_changing_bytes() {
    let page = page_size();
    let fixture = Fixture::new(16, 8 * page + 37);
    let mapped = overlapping_layer(&fixture);
    let before = mapped.page_aligned_bytes().to_vec();
    let expert_before: Vec<_> = (0..3)
        .map(|expert| mapped.expert_bytes(expert).unwrap().to_vec())
        .collect();
    assert_ne!(expert_before[0], expert_before[1]);
    let selected = [0, 1, 0, 2, 1];
    let stats = mapped
        .prepare_selected_experts(&selected, MappedPagePreparationMode::Touch, 3 * page)
        .unwrap();
    assert_eq!(selected, [0, 1, 0, 2, 1]);
    assert_eq!(stats.requested_experts, 5);
    assert_eq!(stats.unique_experts, 3);
    assert_eq!(stats.duplicate_experts, 2);
    assert_eq!(stats.prepared_experts, 3);
    assert_eq!(stats.skipped_budget_experts, 0);
    assert_eq!(stats.selected_bytes, 96);
    assert_eq!(stats.bytes_prepared, 3 * page);
    assert_eq!(stats.pages_prepared, 3);
    assert_eq!(stats.page_touches, 3);
    assert_eq!(stats.advisory_ranges, 0);
    assert_eq!(stats.page_size, page);
    assert!(stats.elapsed_ms.is_finite() && stats.elapsed_ms >= 0.0);
    assert_eq!(mapped.page_aligned_bytes(), before);
    for (expert, before) in expert_before.iter().enumerate() {
        assert_eq!(mapped.expert_bytes(expert).unwrap(), before);
    }
    assert_eq!(std::fs::read(&fixture.path).unwrap(), before);
}

#[test]
fn advice_merges_adjacent_pages_but_preserves_gaps() {
    let page = page_size();
    let fixture = Fixture::new(16, 8 * page + 37);
    let mapped = overlapping_layer(&fixture);
    let before = mapped.page_aligned_bytes().to_vec();
    let stats = mapped
        .prepare_selected_experts(&[2, 1, 0], MappedPagePreparationMode::Advice, 3 * page)
        .unwrap();
    assert_eq!(stats.prepared_experts, 3);
    assert_eq!(stats.bytes_prepared, 3 * page);
    assert_eq!(stats.pages_prepared, 3);
    assert_eq!(stats.page_touches, 0);
    assert_eq!(stats.advisory_ranges, 2);
    assert!(stats.advisory_errors <= stats.advisory_ranges);
    assert_eq!(mapped.page_aligned_bytes(), before);
    assert_eq!(std::fs::read(&fixture.path).unwrap(), before);
}

#[test]
fn budget_admits_whole_experts_in_order_and_charges_shared_pages_once() {
    let page = page_size();
    let fixture = Fixture::new(16, 8 * page + 37);
    let mapped = overlapping_layer(&fixture);
    let small = mapped
        .prepare_selected_experts(&[0, 2, 1], MappedPagePreparationMode::Touch, page)
        .unwrap();
    assert_eq!(small.prepared_experts, 1);
    assert_eq!(small.skipped_budget_experts, 2);
    assert_eq!(small.bytes_prepared, page);
    assert_eq!(small.page_touches, 1);
    let shared = mapped
        .prepare_selected_experts(&[1, 0, 2], MappedPagePreparationMode::Touch, 2 * page)
        .unwrap();
    assert_eq!(shared.prepared_experts, 2);
    assert_eq!(shared.skipped_budget_experts, 1);
    assert_eq!(shared.bytes_prepared, 2 * page);
    assert_eq!(shared.page_touches, 2);
}

#[test]
fn mapping_tail_is_clipped_and_zero_length_experts_are_not_read() {
    let page = page_size();
    let fixture = Fixture::new(0, 2 * page + 37);
    let mapped = MappedExpertLayer::open(fixture.layout(1, 37, Some(vec![2 * page]))).unwrap();
    let stats = mapped
        .prepare_selected_experts(&[0], MappedPagePreparationMode::Touch, 37)
        .unwrap();
    assert_eq!(stats.selected_bytes, 37);
    assert_eq!(stats.bytes_prepared, 37);
    assert_eq!(stats.pages_prepared, 1);
    assert_eq!(stats.page_touches, 1);
    let empty =
        MappedExpertLayer::open(fixture.layout(1, 0, Some(vec![fixture.stream_size]))).unwrap();
    let stats = empty
        .prepare_selected_experts(&[0], MappedPagePreparationMode::Touch, u64::MAX)
        .unwrap();
    assert_eq!(stats.unique_experts, 1);
    assert_eq!(stats.prepared_experts, 0);
    assert_eq!(stats.bytes_prepared, 0);
    assert_eq!(stats.page_touches, 0);
}

#[test]
fn zero_budget_and_empty_selection_do_no_preparation() {
    let page = page_size();
    let fixture = Fixture::new(0, 3 * page);
    let mapped = MappedExpertLayer::open(fixture.layout(3, page, None)).unwrap();
    for mode in [
        MappedPagePreparationMode::Advice,
        MappedPagePreparationMode::Touch,
    ] {
        let stats = mapped
            .prepare_selected_experts(&[0, 1, 0], mode, 0)
            .unwrap();
        assert_eq!(stats.unique_experts, 2);
        assert_eq!(stats.duplicate_experts, 1);
        assert_eq!(stats.skipped_budget_experts, 2);
        assert_eq!(stats.prepared_experts, 0);
        assert_eq!(stats.bytes_prepared, 0);
        assert_eq!(stats.page_touches, 0);
        assert_eq!(stats.advisory_ranges, 0);
        assert!(mapped.prepare_selected_experts(&[0, 3], mode, 0).is_err());
        let stats = mapped.prepare_selected_experts(&[], mode, page).unwrap();
        assert_eq!(stats.requested_experts, 0);
        assert_eq!(stats.bytes_prepared, 0);
        assert_eq!(stats.page_touches, 0);
        assert_eq!(stats.advisory_ranges, 0);
    }
    let empty_bank = MappedExpertLayer::open(fixture.layout(0, page, None)).unwrap();
    assert_eq!(
        empty_bank
            .prepare_selected_experts(&[], MappedPagePreparationMode::Touch, page)
            .unwrap()
            .prepared_experts,
        0
    );
    assert!(empty_bank
        .prepare_selected_experts(&[0], MappedPagePreparationMode::Touch, page)
        .is_err());
}

#[test]
fn malformed_tables_ranges_and_uniform_arithmetic_are_rejected() {
    let page = page_size();
    let fixture = Fixture::new(16, 3 * page);
    for offsets in [
        Some(vec![0]),
        Some(vec![0, u64::MAX]),
        Some(vec![0, 3 * page]),
    ] {
        let mapped = MappedExpertLayer::open(fixture.layout(2, 32, offsets)).unwrap();
        for mode in [
            MappedPagePreparationMode::Advice,
            MappedPagePreparationMode::Touch,
        ] {
            assert!(matches!(
                mapped.prepare_selected_experts(&[0, 1], mode, u64::MAX),
                Err(StreamerError::OffsetOutOfRange { .. })
            ));
        }
    }
    let mapped = MappedExpertLayer::open(fixture.layout(3, u64::MAX, None)).unwrap();
    assert!(matches!(
        mapped.prepare_selected_experts(&[2], MappedPagePreparationMode::Touch, u64::MAX),
        Err(StreamerError::OffsetOutOfRange { .. })
    ));
}

#[test]
fn selection_cap_is_explicit_even_when_all_ids_are_duplicates() {
    let page = page_size();
    let fixture = Fixture::new(0, page);
    let mapped = MappedExpertLayer::open(fixture.layout(1, page, None)).unwrap();
    let accepted = mapped
        .prepare_selected_experts(&[0; 64], MappedPagePreparationMode::Touch, page)
        .unwrap();
    assert_eq!(accepted.unique_experts, 1);
    assert_eq!(accepted.duplicate_experts, 63);
    assert_eq!(accepted.page_touches, 1);
    assert!(matches!(
        mapped.prepare_selected_experts(&[0; 65], MappedPagePreparationMode::Touch, page),
        Err(StreamerError::OpenFailed { detail, .. }) if detail.contains("at most 64")
    ));
}

#[test]
fn a_later_invalid_selection_is_rejected_before_touching_a_valid_page() {
    const CHILD_ENV: &str = "TURBOSPARK_TEST_MAPPED_VALIDATION_CHILD";
    if std::env::var_os(CHILD_ENV).is_none() {
        // Isolate the protected mapping: an eager-touch mutation must fail
        // this child, without crashing the rest of the test executable.
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "a_later_invalid_selection_is_rejected_before_touching_a_valid_page",
                "--nocapture",
            ])
            .env(CHILD_ENV, "1")
            .status()
            .unwrap();
        assert!(
            status.success(),
            "validation touched a protected page: {status}"
        );
        return;
    }
    let page = page_size();
    let fixture = Fixture::new(0, 2 * page);
    let mapped = MappedExpertLayer::open(fixture.layout(2, page, None)).unwrap();
    let pointer = mapped.page_aligned_bytes().as_ptr().cast_mut().cast();
    // SAFETY: this child exclusively owns the page-aligned mapping. No page
    // reader runs while its first page is protected, and it is restored
    // before mapping destruction.
    assert_eq!(
        unsafe { libc::mprotect(pointer, page as usize, libc::PROT_NONE) },
        0
    );
    let result = mapped.prepare_selected_experts(&[0, 2], MappedPagePreparationMode::Touch, page);
    assert_eq!(
        unsafe { libc::mprotect(pointer, page as usize, libc::PROT_READ) },
        0
    );
    assert!(matches!(
        result,
        Err(StreamerError::OffsetOutOfRange { offset: 2 })
    ));
}
