//! The three things every entry point in this crate needs: a status code, a
//! place to leave a message, and a panic guard.
//!
//! **UNWINDING ACROSS THE FFI BOUNDARY IS UNDEFINED BEHAVIOUR, AND THIS
//! WORKSPACE CANNOT OPT OUT OF UNWINDING.** The root `Cargo.toml` records
//! why `panic = "abort"` must stay off: two `Drop` impls are load-bearing on
//! the unwind path (`PassEncoder`'s sends `endEncoding`, without which Metal
//! aborts the process from `-[_MTLCommandEncoder dealloc]` while the real
//! error is still travelling up the stack, and `read_pool`'s `Claim` is what
//! `run_batch` blocks on). So panics here are real, they must be caught, and
//! [`guard`] is the only correct way to write a function in this crate.

use std::cell::RefCell;
use std::os::raw::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Success.
pub const TS_OK: c_int = 0;
/// A null pointer, a non-UTF-8 string, or a value outside its allowed set.
pub const TS_ERR_INVALID_ARGUMENT: c_int = 1;
/// The install could not be opened: missing directory, unreadable manifest,
/// unsupported architecture, or a context window that does not fit.
pub const TS_ERR_OPEN: c_int = 2;
/// Generation failed. The session is still usable.
pub const TS_ERR_GENERATE: c_int = 3;
/// A JSON argument did not parse, or a JSON result could not be built.
pub const TS_ERR_JSON: c_int = 4;
/// The engine is not available on this platform.
pub const TS_ERR_UNSUPPORTED: c_int = 5;
/// A panic was caught at the boundary. The process is intact but the
/// operation did not happen, and this is a BUG in this crate rather than
/// anything a caller did.
pub const TS_ERR_PANIC: c_int = 6;

thread_local! {
    /// The last error message, per thread.
    ///
    /// Per THREAD rather than per session, because the calls that fail
    /// hardest are the ones with no session to hang a message on
    /// (`ts_session_open`), and because a Swift wrapper reads it on the same
    /// thread it made the call from. The consequence a caller must respect:
    /// read it immediately after a non-zero return, before making another
    /// call on that thread.
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Records `message` as this thread's last error and returns `code`, so a
/// failing arm reads `return fail(TS_ERR_OPEN, e)`.
pub fn fail(code: c_int, message: impl Into<String>) -> c_int {
    let message = message.into();
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message);
    code
}

/// Clears this thread's error slot. Called at the top of every entry point,
/// so a stale message from an earlier call cannot be read as this one's.
pub fn clear_error() {
    LAST_ERROR.with(|slot| slot.borrow_mut().clear());
}

/// Copies this thread's last error into `buf` as a NUL-terminated string and
/// returns the message's length in bytes, EXCLUDING the NUL.
///
/// The return value is the length the message actually has, not the number
/// of bytes written, so a caller that passed too small a buffer can size one
/// and call again. `buf` may be null, which is how a caller asks for the
/// length alone.
///
/// # Safety
/// `buf` must be null or point to at least `cap` writable bytes.
pub unsafe fn read_last_error(buf: *mut c_char, cap: usize) -> usize {
    LAST_ERROR.with(|slot| {
        let message = slot.borrow();
        let bytes = message.as_bytes();
        if !buf.is_null() && cap > 0 {
            // One byte reserved for the NUL, and the copy is truncated at a
            // BYTE boundary rather than a char boundary: the result is a C
            // string either way, and a message is diagnostic text rather
            // than anything a caller parses.
            let n = bytes.len().min(cap - 1);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, n);
            *buf.add(n) = 0;
        }
        bytes.len()
    })
}

/// Runs `body` with panics caught, mapping one to [`TS_ERR_PANIC`].
///
/// Every `extern "C"` function in this crate is a call to this and nothing
/// else. `AssertUnwindSafe` is honest here rather than a shrug: the values
/// that cross into `body` are raw pointers and `&mut` borrows the caller
/// already owns, and a panic leaves the session's `Mutex` poisoned, which
/// every subsequent call reports as an error rather than reading through.
pub fn guard(body: impl FnOnce() -> c_int) -> c_int {
    clear_error();
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(code) => code,
        Err(payload) => {
            // Recover whatever the panic carried, so the message names the
            // failure rather than only its existence.
            let detail = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic payload".to_string());
            fail(
                TS_ERR_PANIC,
                format!("panic caught at the FFI boundary: {detail}"),
            )
        }
    }
}
