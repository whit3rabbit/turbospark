//! `mmap`'d view of `model_weights.bin`'s resident tensor data region.
//! Ported from `Infrastructure/ModelIO/ResidentBuffer.swift`.
//!
//! The Swift original wraps the mapping directly in an `MTLBuffer` so
//! kernels can address it with zero copies. That GPU-specific wrapping
//! belongs to the `gpu` crate (Phase 6); this crate exposes the mapped
//! bytes as a plain slice, which a GPU backend can then upload or wrap
//! without this crate depending on any GPU API.

use std::fs::File;
use std::path::Path;

use memmap2::{Mmap, MmapOptions};

use crate::error::ModelError;

/// `mmap`'d window covering `[file_offset, file_offset + resident_size)`
/// inside a file, exposed as a byte slice starting at the resident bytes
/// (the mapping itself starts at the page-aligned offset below that).
pub struct ResidentBuffer {
    mapping: Mmap,
    slice_shift: usize,
    resident_size: usize,
}

impl ResidentBuffer {
    pub fn map(file_path: &Path, file_offset: u64, resident_size: u64) -> Result<Self, ModelError> {
        let file = File::open(file_path).map_err(|e| ModelError::IoFailed {
            call: "open".to_string(),
            detail: e.to_string(),
        })?;
        let page_size = page_size_bytes();
        let aligned_offset = (file_offset / page_size) * page_size;
        let slice_shift = (file_offset - aligned_offset) as usize;
        let mapped_len = slice_shift + resident_size as usize;

        let mapping = unsafe_map(&file, aligned_offset, mapped_len)?;
        Ok(Self {
            mapping,
            slice_shift,
            resident_size: resident_size as usize,
        })
    }

    /// The resident bytes, starting at logical offset 0 (== `file_offset`
    /// passed to [`ResidentBuffer::map`]).
    pub fn data(&self) -> &[u8] {
        &self.mapping[self.slice_shift..self.slice_shift + self.resident_size]
    }
}

fn page_size_bytes() -> u64 {
    4096
}

fn unsafe_map(file: &File, offset: u64, len: usize) -> Result<Mmap, ModelError> {
    // SAFETY (per the `memmap2` crate contract): the mapped file must not be
    // truncated concurrently with the mapping's lifetime. Model install
    // directories are treated as read-only, immutable inputs once verified;
    // nothing in this workspace writes to a `.gturbo` install after load.
    #[allow(unsafe_code)]
    unsafe {
        MmapOptions::new()
            .offset(offset)
            .len(len)
            .map(file)
            .map_err(|e| ModelError::IoFailed {
                call: "mmap".to_string(),
                detail: e.to_string(),
            })
    }
}
