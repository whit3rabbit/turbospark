//! Zero-copy Metal view of the resident weights mapping. Ported from
//! `Infrastructure/ModelIO/ResidentBuffer.swift`'s
//! `makeBuffer(bytesNoCopy:)` wrap: the whole mmap'd resident region
//! becomes ONE `MTLBuffer` (shared storage, unified memory), and every
//! tensor is addressed as an offset into it. Nothing is copied; the GPU
//! reads the same physical pages the page cache faulted in from
//! `model_weights.bin`.
//!
//! Owning the [`model_io::ResidentBuffer`] here is what makes the wrap
//! sound: the `MTLBuffer` is created with no deallocator and aliases the
//! mapping, so the mapping must outlive it — both live and die with this
//! struct (fields drop in declaration order: the Metal buffer first, the
//! mapping after).

use metal::{Device, MTLResourceOptions};
use model_io::ResidentBuffer;

use crate::context::GpuError;

/// Wraps caller-owned page-aligned memory (e.g. one expert-streamer slot,
/// `posix_memalign`'d and page-rounded) in a shared-storage `MTLBuffer`
/// without copying — the per-slot half of the Swift original's streamer
/// design. The caller must keep the allocation alive for the buffer's
/// whole lifetime and must not free it before dropping the buffer.
pub fn wrap_page_aligned_no_copy(
    device: &Device,
    ptr: *const u8,
    len: usize,
) -> Result<metal::Buffer, GpuError> {
    let buffer = device.new_buffer_with_bytes_no_copy(
        ptr.cast(),
        len as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    if buffer.length() != len as u64 {
        return Err(GpuError::BufferCreate(format!(
            "newBufferWithBytesNoCopy rejected allocation (base {ptr:p}, len {len})"
        )));
    }
    Ok(buffer)
}

pub struct ResidentGpuWeights {
    // Declaration order is load-bearing; see module docs.
    buffer: metal::Buffer,
    resident: ResidentBuffer,
}

impl ResidentGpuWeights {
    /// Wraps `resident`'s whole mapping in one shared-storage `MTLBuffer`
    /// without copying. The mapping base is page-aligned (an mmap
    /// guarantee), which is what `newBufferWithBytesNoCopy` requires.
    pub fn wrap(device: &Device, resident: ResidentBuffer) -> Result<Self, GpuError> {
        let mapped = resident.mapped_bytes();
        let buffer = device.new_buffer_with_bytes_no_copy(
            mapped.as_ptr().cast(),
            mapped.len() as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        // Metal returns nil (a null object; every message to it answers
        // zero) when the pointer or length is unacceptable.
        if buffer.length() != mapped.len() as u64 {
            return Err(GpuError::BufferCreate(format!(
                "newBufferWithBytesNoCopy rejected the resident mapping \
                 (base {:p}, len {})",
                mapped.as_ptr(),
                mapped.len()
            )));
        }
        Ok(Self { buffer, resident })
    }

    /// The one shared `MTLBuffer` covering the whole resident region.
    pub fn buffer(&self) -> &metal::Buffer {
        &self.buffer
    }

    /// Converts a logical resident-region offset (0 == the first resident
    /// byte, i.e. `ResidentIndexEntry.file_offset - header.index_size`)
    /// into an offset usable with [`ResidentGpuWeights::buffer`].
    pub fn gpu_offset(&self, logical_offset: u64) -> u64 {
        self.resident.slice_shift() as u64 + logical_offset
    }

    /// CPU view of the resident bytes (logical offset 0 first), for the
    /// host-side readers that still need it (embedding lookup bridge,
    /// tests).
    pub fn data(&self) -> &[u8] {
        self.resident.data()
    }
}
