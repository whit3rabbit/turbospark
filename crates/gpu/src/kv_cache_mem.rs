//! Low-level memory operations and page advice for KV cache buffers.

pub(crate) fn page_size_bytes() -> usize {
    // SAFETY: sysconf(_SC_PAGESIZE) reads a process constant; no memory is
    // touched. 16 KiB on Apple Silicon; the reset() advise rounds buffer
    // lengths DOWN to whole pages, so the real page size advises more of
    // each buffer than a hardcoded 4096 would.
    #[allow(unsafe_code)]
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page > 0 {
        page as usize
    } else {
        4096
    }
}

pub(crate) fn write_into(buffer: &metal::Buffer, offset: usize, bytes: &[u8]) {
    assert!(offset + bytes.len() <= buffer.length() as usize);
    // SAFETY: `buffer` is a live shared-storage `MTLBuffer`; the range
    // [offset, offset + bytes.len()) is inside its allocation (asserted
    // above). The caller sequences this against GPU reads the same way the
    // Swift original does: the write happens before the command buffer
    // that reads the slot is committed.
    #[allow(unsafe_code)]
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            (buffer.contents() as *mut u8).add(offset),
            bytes.len(),
        );
    }
}

pub(crate) fn advise_dontneed(
    buffer: &metal::Buffer,
    page_size: usize,
    seen: &mut Vec<*const std::ffi::c_void>,
) {
    let ptr = buffer.contents() as *const std::ffi::c_void;
    if seen.contains(&ptr) {
        return;
    }
    seen.push(ptr);
    let len = (buffer.length() as usize / page_size) * page_size;
    if len > 0 {
        // SAFETY: `ptr` is the base address of a live `MTLBuffer` allocated
        // with shared storage mode (CPU-and-GPU-visible, page-aligned by
        // Metal's allocator), and `len` is rounded down to a whole number
        // of pages so this never advises past the buffer's own allocation.
        // `MADV_DONTNEED` only affects physical residency, never the
        // buffer's validity as an object.
        #[allow(unsafe_code)]
        unsafe {
            libc::posix_madvise(ptr as *mut std::ffi::c_void, len, libc::POSIX_MADV_DONTNEED);
        }
    }
}
