//! Multi-shard reading, header resolving, and shape utilities.

use std::collections::HashMap;
use std::sync::Mutex;

use super::config::Gemma4Error;
use crate::ranged_download::RangeSource;
use crate::safetensors_header::{SafetensorsHeader, TensorInfo};

/// On-disk page alignment unit for `.gturbo` files (the Swift repacker's
/// `Layout.pageBytes`): fixed at 16 KiB regardless of host page size.
pub const GTURBO_PAGE_BYTES: u64 = 16_384;

/// Target window size for coalesced tensor prefetch batches (8 x 16 MiB chunks,
/// filling all HTTP range workers while bounding transient prefetch memory).
pub(crate) const PREFETCH_WINDOW_BYTES: u64 = 128 * 1024 * 1024;

/// A multi-shard checkpoint view: real HF checkpoints split their tensors
/// across several `model-NNNNN-of-NNNNN.safetensors` files (the
/// `model.safetensors.index.json` weight map), and companion tensors may
/// live in a different shard than their weight -- so lookups go through one
/// merged name registry, exactly like the Swift planner's `registry`.
pub struct Gemma4Shards<'a> {
    shards: Vec<(&'a SafetensorsHeader, &'a dyn RangeSource)>,
    by_name: HashMap<&'a str, usize>,
    /// Bounded on-demand prefetch cache populated by [`Self::prefetch_guard`]
    /// and popped zero-copy by [`Self::read`].
    prefetch: Mutex<HashMap<String, (Vec<u8>, usize)>>,
}

/// RAII guard that clears any unconsumed prefetched tensors on drop.
pub(crate) struct PrefetchGuard<'s, 'a> {
    shards: &'s Gemma4Shards<'a>,
}

impl Drop for PrefetchGuard<'_, '_> {
    fn drop(&mut self) {
        self.shards
            .prefetch
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clear();
    }
}

impl<'a> Gemma4Shards<'a> {
    /// Creates multi-shard tensor registry mapping names to shard index.
    ///
    /// A tensor present in two shards is a corrupt checkpoint (the shard
    /// split is supposed to partition the tensor set), and the old
    /// behaviour -- last shard silently wins -- would read one shard's
    /// bytes under a name the manifest also lets a caller resolve from a
    /// different shard, with no error anywhere.
    pub fn new(
        shards: Vec<(&'a SafetensorsHeader, &'a dyn RangeSource)>,
    ) -> Result<Self, Gemma4Error> {
        let mut by_name = HashMap::new();
        for (i, (header, _)) in shards.iter().enumerate() {
            for name in header.tensors.keys() {
                if by_name.insert(name.as_str(), i).is_some() {
                    return Err(Gemma4Error::ShapeMismatch {
                        tensor: name.clone(),
                        detail: "present in two shards".to_string(),
                    });
                }
            }
        }
        Ok(Self {
            shards,
            by_name,
            prefetch: Mutex::new(HashMap::new()),
        })
    }

    /// Creates single-shard tensor registry wrapper.
    pub fn single(header: &'a SafetensorsHeader, source: &'a dyn RangeSource) -> Self {
        Self::new(vec![(header, source)]).expect("a single shard cannot collide with itself")
    }

    /// Resolves the shard header and byte source for a given tensor name.
    pub fn shard_of(
        &self,
        name: &str,
    ) -> Result<&(&'a SafetensorsHeader, &'a dyn RangeSource), Gemma4Error> {
        let i = *self
            .by_name
            .get(name)
            .ok_or_else(|| Gemma4Error::MissingTensor(name.to_string()))?;
        Ok(&self.shards[i])
    }

    /// Returns `(shard_index, start, end_exclusive)` for a tensor without
    /// reading its bytes.
    pub(crate) fn location_of(&self, name: &str) -> Result<(usize, u64, u64), Gemma4Error> {
        let shard_idx = *self
            .by_name
            .get(name)
            .ok_or_else(|| Gemma4Error::MissingTensor(name.to_string()))?;
        let (header, _) = self.shards[shard_idx];
        let (start, end) = header
            .absolute_range(name)
            .ok_or_else(|| Gemma4Error::MissingTensor(name.to_string()))?;
        Ok((shard_idx, start, end))
    }

    /// Looks up metadata information for a given tensor name across shards.
    pub fn info(&self, name: &str) -> Result<&'a TensorInfo, Gemma4Error> {
        let (header, _) = self.shard_of(name)?;
        header
            .tensors
            .get(name)
            .ok_or_else(|| Gemma4Error::MissingTensor(name.to_string()))
    }

    /// Returns true if a tensor with the given name exists in any shard.
    pub fn contains(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    /// Prefetches `names` across shards using each shard's
    /// [`RangeSource::read_ranges`], so small adjacent tensors coalesce into
    /// large concurrent HTTP range transfers. Subsequent [`Self::read`] calls
    /// pop the prefetched buffers without copying.
    pub(crate) fn prefetch_guard<'s>(
        &'s self,
        names: &[&str],
    ) -> Result<PrefetchGuard<'s, 'a>, Gemma4Error> {
        if names.is_empty() {
            return Ok(PrefetchGuard { shards: self });
        }
        let mut counts: HashMap<&str, usize> = HashMap::with_capacity(names.len());
        let mut by_shard: Vec<Vec<(&str, u64, u64)>> = vec![Vec::new(); self.shards.len()];
        for &name in names {
            let entry = counts.entry(name).or_insert(0);
            *entry += 1;
            if *entry == 1 {
                let (shard_idx, start, end) = self.location_of(name)?;
                by_shard[shard_idx].push((name, start, end));
            }
        }
        let mut fetched_all: Vec<(String, Vec<u8>, usize)> = Vec::with_capacity(counts.len());
        for (shard_idx, items) in by_shard.into_iter().enumerate() {
            if items.is_empty() {
                continue;
            }
            let ranges: Vec<(u64, u64)> = items.iter().map(|&(_, s, e)| (s, e)).collect();
            let (_, source) = self.shards[shard_idx];
            let payloads = source.read_ranges(&ranges)?;
            for ((name, _, _), bytes) in items.into_iter().zip(payloads) {
                let count = counts[name];
                fetched_all.push((name.to_string(), bytes, count));
            }
        }
        let mut map = self.prefetch.lock().unwrap_or_else(|p| p.into_inner());
        for (name, bytes, count) in fetched_all {
            map.insert(name, (bytes, count));
        }
        Ok(PrefetchGuard { shards: self })
    }

    /// Reads raw tensor byte payload from its hosting shard range source.
    pub fn read(&self, name: &str) -> Result<Vec<u8>, Gemma4Error> {
        let hit = {
            let mut map = self.prefetch.lock().unwrap_or_else(|p| p.into_inner());
            if let Some((_, count)) = map.get_mut(name) {
                if *count <= 1 {
                    map.remove(name).map(|(bytes, _)| bytes)
                } else {
                    *count -= 1;
                    map.get(name).map(|(bytes, _)| bytes.clone())
                }
            } else {
                None
            }
        };
        if let Some(bytes) = hit {
            return Ok(bytes);
        }
        let (header, source) = self.shard_of(name)?;
        let (start, end) = header
            .absolute_range(name)
            .ok_or_else(|| Gemma4Error::MissingTensor(name.to_string()))?;
        Ok(source.read_range(start, end)?)
    }

    /// Returns an iterator over all tensor names present across all shards.
    pub fn names(&self) -> impl Iterator<Item = &'a String> + '_ {
        self.shards.iter().flat_map(|(h, _)| h.tensors.keys())
    }
}

/// Normalizes tensor shape slice to 4-tuple of u32 dimensions.
pub fn shape4(shape: &[u64]) -> (u32, u32, u32, u32) {
    let get = |i: usize| shape.get(i).copied().unwrap_or(0) as u32;
    (get(0), get(1), get(2), get(3))
}

/// Converts little-endian byte slice into u16 vector.
pub fn le_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}
