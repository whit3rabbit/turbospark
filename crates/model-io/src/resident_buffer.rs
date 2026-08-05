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
        // The decode path touches scattered tensors, not a sequential
        // sweep; the Swift original advises POSIX_MADV_RANDOM for the
        // same reason. Advisory only: failure changes nothing observable.
        let _ = mapping.advise(memmap2::Advice::Random);
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

    /// The whole page-aligned mapping, starting at the page boundary at or
    /// below `file_offset`. A GPU backend wrapping this memory zero-copy
    /// (Metal's `newBufferWithBytesNoCopy` requires a page-aligned base)
    /// wraps this slice and adds [`ResidentBuffer::slice_shift`] to every
    /// tensor offset.
    pub fn mapped_bytes(&self) -> &[u8] {
        &self.mapping
    }

    /// Byte distance from the mapping base to logical offset 0 of
    /// [`ResidentBuffer::data`]. Zero whenever `file_offset` was already
    /// page-aligned (the `.gturbo` writer page-aligns the resident region).
    pub fn slice_shift(&self) -> usize {
        self.slice_shift
    }
}

fn page_size_bytes() -> u64 {
    // Real page size, not a constant: Apple Silicon macOS uses 16 KiB
    // pages, and a hardcoded 4096 would produce a non-page-aligned mmap
    // offset (mmap would fail) for file offsets between 4 KiB multiples
    // and 16 KiB multiples.
    // SAFETY: sysconf(_SC_PAGESIZE) reads a process constant; no memory
    // is touched.
    #[allow(unsafe_code)]
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page > 0 {
        page as u64
    } else {
        4096
    }
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
