//! Zero-copy prefetch cache over any [`RangeSource`].
//!
//! Callers that know the upcoming tensor byte ranges (such as a resident
//! tensor window or a routed MoE layer) call [`PrefetchedRangeSource::prefetch`]
//! to fetch them in one coalesced, concurrent [`RangeSource::read_ranges`]
//! call. Subsequent [`RangeSource::read_range`] calls pop their `Vec<u8>`
//! directly out of the map with no extra copy.

use std::collections::HashMap;
use std::sync::Mutex;

use super::{DownloadError, RangeSource};

/// Maps `(start, end_exclusive)` to `(bytes, remaining_expected_reads)`.
/// When the remaining count reaches zero, the entry is removed and returned
/// without cloning.
type PrefetchMap = HashMap<(u64, u64), (Vec<u8>, usize)>;

/// Wraps a [`RangeSource`] with an explicit, bounded prefetch buffer.
pub(crate) struct PrefetchedRangeSource<'a> {
    inner: &'a dyn RangeSource,
    entries: Mutex<PrefetchMap>,
}

impl<'a> PrefetchedRangeSource<'a> {
    pub(crate) fn new(inner: &'a dyn RangeSource) -> Self {
        Self {
            inner,
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Fetches `ranges` through `inner.read_ranges` and stores them for
    /// upcoming `read_range` calls. Duplicate ranges are fetched once.
    pub(crate) fn prefetch(&self, ranges: &[(u64, u64)]) -> Result<(), DownloadError> {
        if ranges.is_empty() {
            return Ok(());
        }
        let mut unique: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
        let mut counts: HashMap<(u64, u64), usize> = HashMap::with_capacity(ranges.len());
        for &range in ranges {
            let entry = counts.entry(range).or_insert(0);
            if *entry == 0 {
                unique.push(range);
            }
            *entry += 1;
        }
        let fetched = self.inner.read_ranges(&unique)?;
        let mut map = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        for (range, bytes) in unique.into_iter().zip(fetched) {
            let add = counts[&range];
            match map.get_mut(&range) {
                Some((existing, count)) => {
                    *existing = bytes;
                    *count += add;
                }
                None => {
                    map.insert(range, (bytes, add));
                }
            }
        }
        Ok(())
    }

    /// Drops any prefetched entries that were not consumed.
    pub(crate) fn clear(&self) {
        self.entries
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clear();
    }
}

impl RangeSource for PrefetchedRangeSource<'_> {
    fn read_range(&self, start: u64, end_exclusive: u64) -> Result<Vec<u8>, DownloadError> {
        let hit = {
            let mut map = self.entries.lock().unwrap_or_else(|p| p.into_inner());
            if let Some((_, count)) = map.get_mut(&(start, end_exclusive)) {
                if *count <= 1 {
                    map.remove(&(start, end_exclusive)).map(|(bytes, _)| bytes)
                } else {
                    *count -= 1;
                    map.get(&(start, end_exclusive))
                        .map(|(bytes, _)| bytes.clone())
                }
            } else {
                None
            }
        };
        match hit {
            Some(bytes) => Ok(bytes),
            None => self.inner.read_range(start, end_exclusive),
        }
    }

    fn read_ranges(&self, ranges: &[(u64, u64)]) -> Result<Vec<Vec<u8>>, DownloadError> {
        self.inner.read_ranges(ranges)
    }
}
