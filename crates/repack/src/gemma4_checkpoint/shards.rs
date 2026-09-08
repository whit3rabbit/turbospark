//! Multi-shard reading, header resolving, and shape utilities.

use super::config::Gemma4Error;
use crate::ranged_download::RangeSource;
use crate::safetensors_header::{SafetensorsHeader, TensorInfo};

/// On-disk page alignment unit for `.gturbo` files (the Swift repacker's
/// `Layout.pageBytes`): fixed at 16 KiB regardless of host page size.
pub const GTURBO_PAGE_BYTES: u64 = 16_384;

/// A multi-shard checkpoint view: real HF checkpoints split their tensors
/// across several `model-NNNNN-of-NNNNN.safetensors` files (the
/// `model.safetensors.index.json` weight map), and companion tensors may
/// live in a different shard than their weight -- so lookups go through one
/// merged name registry, exactly like the Swift planner's `registry`.
pub struct Gemma4Shards<'a> {
    shards: Vec<(&'a SafetensorsHeader, &'a dyn RangeSource)>,
    by_name: std::collections::HashMap<&'a str, usize>,
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
        let mut by_name = std::collections::HashMap::new();
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
        Ok(Self { shards, by_name })
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

    /// Reads raw tensor byte payload from its hosting shard range source.
    pub fn read(&self, name: &str) -> Result<Vec<u8>, Gemma4Error> {
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
