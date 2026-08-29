//! Core C ABI entry points: errors, string memory management, and system metrics.

use std::os::raw::{c_char, c_int};

use crate::abi::{self, guard_result};
use crate::strings;
use crate::telemetry;

/// Copies this thread's last error message into `buf`, returning the
/// message's own length in bytes excluding the NUL.
///
/// Pass a null `buf` to ask for the length alone. The return value is the
/// message's length rather than the number of bytes written, so a truncated
/// read tells the caller what buffer to allocate.
#[no_mangle]
pub unsafe extern "C" fn ts_last_error(buf: *mut c_char, cap: usize) -> usize {
    // NOT wrapped in `guard`: it returns a length rather than a code, and it
    // is what a caller reaches for when something has already gone wrong, so
    // it must not itself be able to clear the slot it is reading.
    abi::read_last_error(buf, cap)
}

/// Frees a string this library handed out through a `char **`.
#[no_mangle]
pub unsafe extern "C" fn ts_string_free(ptr: *mut c_char) {
    strings::free(ptr);
}

/// This process's peak physical footprint in bytes, or 0 where unavailable.
#[no_mangle]
pub unsafe extern "C" fn ts_peak_footprint_bytes() -> u64 {
    // No `guard`: nothing here can fail or panic, and there is no code to
    // return through.
    telemetry::peak_footprint_bytes()
}

/// Hardware and power telemetry for this machine, as JSON.
#[no_mangle]
pub unsafe extern "C" fn ts_system_info_json(out: *mut *mut c_char) -> c_int {
    guard_result(|| {
        let json = telemetry::system_info_json().map_err(|e| (abi::TS_ERR_JSON, e))?;
        strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}
