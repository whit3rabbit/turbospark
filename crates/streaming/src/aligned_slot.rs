use crate::error::StreamerError;

/// The Swift original's `scratchAlignment`: slot bases are 2 MiB-aligned
/// (comfortably page-aligned on any page size), sized up to whole pages.
const SLOT_ALIGNMENT: usize = 2 * 1024 * 1024;

/// One expert slot's backing memory: page-aligned, page-rounded, allocated
/// once. Exposes its base pointer so a GPU backend can wrap it no-copy.
pub struct AlignedSlot {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the allocation is plain heap memory; the streamer alone decides
// which threads write which slot (disjointly, during plan execution).
#[allow(unsafe_code)]
unsafe impl Send for AlignedSlot {}
#[allow(unsafe_code)]
unsafe impl Sync for AlignedSlot {}

impl AlignedSlot {
    pub(crate) fn allocate(len: usize) -> Result<Self, StreamerError> {
        let page = page_size();
        let rounded = len.div_ceil(page) * page;
        let mut raw: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: standard posix_memalign call; alignment is a power of
        // two and a multiple of pointer size; failure is checked below.
        #[allow(unsafe_code)]
        let rc = unsafe { libc::posix_memalign(&mut raw, SLOT_ALIGNMENT, rounded.max(page)) };
        if rc != 0 || raw.is_null() {
            return Err(StreamerError::AllocationFailed {
                detail: format!(
                    "posix_memalign({SLOT_ALIGNMENT}, {}) failed with {rc}",
                    rounded.max(page)
                ),
            });
        }
        // SAFETY: freshly allocated, at least `rounded` bytes.
        #[allow(unsafe_code)]
        unsafe {
            std::ptr::write_bytes(raw as *mut u8, 0, rounded.max(page));
        }
        Ok(Self {
            ptr: raw as *mut u8,
            len: rounded.max(page),
        })
    }

    /// Returns a raw const pointer to the slot memory allocation base.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Returns the length in bytes of the slot memory allocation.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the slot allocation is 0 bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` is a live allocation of `len` bytes owned by self.
        #[allow(unsafe_code)]
        unsafe {
            std::slice::from_raw_parts(self.ptr, self.len)
        }
    }

    #[allow(dead_code)]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, with exclusive access through &mut self.
        #[allow(unsafe_code)]
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr, self.len)
        }
    }
}

impl Drop for AlignedSlot {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from posix_memalign and is freed exactly once.
        #[allow(unsafe_code)]
        unsafe {
            libc::free(self.ptr as *mut std::ffi::c_void);
        }
    }
}

fn page_size() -> usize {
    // SAFETY: reads a process constant.
    #[allow(unsafe_code)]
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page > 0 {
        page as usize
    } else {
        4096
    }
}
