use crate::error::StreamerError;

/// The Swift original's `scratchAlignment`: slot bases are 2 MiB-aligned
/// (comfortably page-aligned on any page size), sized up to whole pages.
const SLOT_ALIGNMENT: usize = 2 * 1024 * 1024;

/// One expert slot's backing memory: page-aligned, page-rounded, allocated
/// once. Exposes its base pointer so a GPU backend can wrap it no-copy.
pub struct AlignedSlot {
    ptr: *mut u8,
    len: usize,
    /// Set by [`AlignedSlot::wire`]; `Drop` unwires before freeing so the
    /// allocator never gets back pages that are still wired.
    wired: bool,
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
            wired: false,
        })
    }

    /// Wires the slot's pages (`mlock`) so the kernel cannot compress or
    /// swap them. A GPU pass reading a compressed no-copy slot waits for the
    /// kernel to restore it, which is what made larger qwen4 caches lose
    /// (docs/QWEN4_EXP.md, "Why larger expert caches lose").
    pub(crate) fn wire(&mut self) -> std::io::Result<()> {
        if self.wired {
            return Ok(());
        }
        // SAFETY: `ptr` is a live allocation of `len` bytes owned by self.
        #[allow(unsafe_code)]
        let rc = unsafe { libc::mlock(self.ptr as *const std::ffi::c_void, self.len) };
        if rc != 0 {
            return Err(std::io::Error::last_os_error());
        }
        self.wired = true;
        Ok(())
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
        if self.wired {
            // SAFETY: the same live range `wire` locked. A failure leaves
            // the pages wired until exit, which is all it can cost.
            #[allow(unsafe_code)]
            unsafe {
                libc::munlock(self.ptr as *const std::ffi::c_void, self.len);
            }
        }
        // SAFETY: `ptr` came from posix_memalign and is freed exactly once.
        #[allow(unsafe_code)]
        unsafe {
            libc::free(self.ptr as *mut std::ffi::c_void);
        }
    }
}

/// The most this process may wire, or `None` when neither limit can be read:
/// the smaller of `RLIMIT_MEMLOCK` (when finite) and macOS's per-user
/// `vm.user_wire_limit`. A ceiling, not a promise: other processes' wired
/// pages count against the same user limit, so a request under it can
/// still fail at `mlock`, and the caller must handle that too.
pub fn wire_limit_bytes() -> Option<u64> {
    let mut limit: Option<u64> = None;
    let mut rl = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: getrlimit writes one rlimit through a valid pointer.
    #[allow(unsafe_code)]
    let rc = unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut rl) };
    if rc == 0 && rl.rlim_cur != libc::RLIM_INFINITY {
        limit = Some(rl.rlim_cur);
    }
    #[cfg(target_os = "macos")]
    {
        let mut value: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let name = c"vm.user_wire_limit";
        // SAFETY: sysctlbyname reads a u64 into a buffer of its size; the
        // name is NUL-terminated and no new value is written.
        #[allow(unsafe_code)]
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                (&mut value as *mut u64).cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc == 0 && len == std::mem::size_of::<u64>() && value > 0 {
            limit = Some(limit.map_or(value, |l| l.min(value)));
        }
    }
    limit
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

#[cfg(test)]
mod tests {
    use super::AlignedSlot;

    #[test]
    fn wire_limit_is_readable_on_macos() {
        // The limit the runtime checks before it asks for a wired cache. On
        // macOS the user wire limit always exists; elsewhere it may not.
        if cfg!(target_os = "macos") {
            assert!(super::wire_limit_bytes().is_some_and(|b| b > 0));
        }
    }

    #[test]
    fn wire_locks_once_and_drop_unwires() {
        let mut slot = AlignedSlot::allocate(3 * 1024 * 1024).unwrap();
        slot.wire().unwrap();
        assert!(slot.wired);
        // A second call is a no-op, not a second lock to undo.
        slot.wire().unwrap();
        assert!(slot.wired);
        drop(slot);
    }
}
