//! A validated logical address space over single or split GGUF files.
//! Tensor offsets retain their source shard without staging weight payloads.

use crate::{DownloadError, GgufHeader, RangeSource};

pub struct GgufSet<S> {
    pub header: GgufHeader,
    pub bytes: u64,
    sources: Vec<(u64, u64, S)>,
}

fn split_declared(header: &GgufHeader) -> Result<bool, String> {
    let split = header.metadata.contains_key("split.count");
    if !split && header.metadata.keys().any(|key| key.starts_with("split.")) {
        return Err("split metadata requires split.count".into());
    }
    Ok(split)
}

/// Resolve the standard split filename, requiring the caller to name shard 1.
/// Counts are bounded before allocating or issuing requests.
pub fn gguf_shard_names(file: &str, header: &GgufHeader) -> Result<Vec<String>, String> {
    if !split_declared(header)? {
        return Ok(vec![file.to_owned()]);
    }
    let count = header.metadata["split.count"]
        .as_u64()
        .ok_or("split.count must be an integer")?;
    if !(1..=1024).contains(&count) {
        return Err("split.count must be between 1 and 1024".into());
    }
    if header.metadata.get("split.no").and_then(|v| v.as_u64()) != Some(0) {
        return Err("point at GGUF shard 1 (split.no = 0)".into());
    }
    if count == 1 {
        return Ok(vec![file.to_owned()]);
    }
    let suffix = format!("-00001-of-{count:05}.gguf");
    let prefix = file
        .strip_suffix(&suffix)
        .ok_or("split GGUF filename disagrees with split.count or does not name shard 1")?;
    Ok((1..=count)
        .map(|i| format!("{prefix}-{i:05}-of-{count:05}.gguf"))
        .collect())
}

impl<S: RangeSource> GgufSet<S> {
    pub fn new(shards: Vec<(GgufHeader, S, u64)>) -> Result<Self, String> {
        let first = &shards.first().ok_or("no GGUF shards")?.0;
        let split = split_declared(first)?;
        let integer = |h: &GgufHeader, key: &str| {
            h.metadata
                .get(key)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| format!("missing or invalid {key}"))
        };
        let count = if split {
            integer(first, "split.count")?
        } else {
            1
        };
        if count == 0 || count > 1024 || count != shards.len() as u64 {
            return Err("incomplete GGUF shard set".into());
        }
        let expected = if split {
            integer(first, "split.tensors.count")?
        } else {
            first.tensors.len() as u64
        };
        let mut header = first.clone();
        header.tensors.clear();
        // Offsets below are absolute in the logical concatenation of files.
        header.data_region_start = 0;
        let mut sources = Vec::new();
        let mut base = 0u64;
        for (i, (h, source, length)) in shards.into_iter().enumerate() {
            // Empty shards have no tensor range to catch a truncated header.
            if h.data_region_start > length {
                return Err(format!("GGUF header exceeds shard {} length", i + 1));
            }
            if h.version != header.version || h.alignment != header.alignment {
                return Err("inconsistent GGUF version or alignment".into());
            }
            if split
                && (integer(&h, "split.no")? != i as u64
                    || integer(&h, "split.count")? != count
                    || integer(&h, "split.tensors.count")? != expected)
            {
                return Err("inconsistent GGUF split metadata".into());
            }
            for (key, value) in &h.metadata {
                if key.starts_with("split.") {
                    continue;
                }
                if header.metadata.get(key).is_some_and(|first| first != value) {
                    return Err(format!("inconsistent GGUF metadata: {key}"));
                }
                // Later shards may introduce keys absent from the first.
                // Retain their first value so subsequent shards must agree.
                header
                    .metadata
                    .entry(key.clone())
                    .or_insert_with(|| value.clone());
            }
            for (name, info) in &h.tensors {
                let (start, end) = h.absolute_range(name).unwrap().map_err(|e| e.to_string())?;
                if start < h.data_region_start || end > length {
                    return Err(format!("tensor {name} exceeds shard {} length", i + 1));
                }
                let mut mapped = info.clone();
                mapped.offset = base.checked_add(start).ok_or("GGUF offset overflow")?;
                if header.tensors.insert(name.clone(), mapped).is_some() {
                    return Err(format!("duplicate GGUF tensor: {name}"));
                }
            }
            let end = base.checked_add(length).ok_or("GGUF length overflow")?;
            sources.push((base, end, source));
            base = end;
        }
        if header.tensors.len() as u64 != expected {
            return Err("GGUF tensor count disagrees with split.tensors.count".into());
        }
        Ok(Self {
            header,
            bytes: base,
            sources,
        })
    }
}

impl<S: RangeSource> RangeSource for GgufSet<S> {
    fn read_range(&self, start: u64, end: u64) -> Result<Vec<u8>, DownloadError> {
        if start > end || end > self.bytes {
            return Err(DownloadError::Request("invalid split GGUF range".into()));
        }
        if start == end {
            return Ok(Vec::new());
        }
        // Every tensor belongs to one shard. Crossing a shard boundary is a
        // caller bug, rather than a reason to allocate a combined payload.
        let (base, _, source) = self
            .sources
            .iter()
            .find(|(base, limit, _)| start >= *base && end <= *limit)
            .ok_or_else(|| DownloadError::Request("GGUF tensor crosses shard boundary".into()))?;
        source.read_range(start - base, end - base)
    }
}
