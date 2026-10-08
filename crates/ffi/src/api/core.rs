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
    // Guarded like every teardown: a panic in a Drop must come back as a
    // recorded error, not unwind across `extern "C"`.
    abi::guard_value((), || strings::free(ptr));
}

/// This process's peak physical footprint in bytes, or 0 where unavailable.
#[no_mangle]
pub unsafe extern "C" fn ts_peak_footprint_bytes() -> u64 {
    // No `guard`: nothing here can fail or panic, and there is no code to
    // return through.
    telemetry::peak_footprint_bytes()
}

/// The ABI revision of this library. See `TS_ABI_VERSION` in the header.
#[no_mangle]
pub unsafe extern "C" fn ts_abi_version() -> u32 {
    // Cannot fail, but guarded like every other entry point: the invariant is
    // "nothing unwinds across `extern \"C\"`", not "nothing here can panic".
    abi::guard_value(0, || abi::ABI_VERSION)
}

/// Build information as JSON: ABI revision, crate version, and whether debug
/// assertions are on.
#[no_mangle]
pub unsafe extern "C" fn ts_build_info_json(out: *mut *mut c_char) -> c_int {
    guard_result(|| {
        let json = serde_json::json!({
            "abiVersion": abi::ABI_VERSION,
            "version": env!("CARGO_PKG_VERSION"),
            "debugAssertions": cfg!(debug_assertions),
        })
        .to_string();
        strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// Hardware and power telemetry for this machine, as JSON.
#[no_mangle]
pub unsafe extern "C" fn ts_system_info_json(out: *mut *mut c_char) -> c_int {
    guard_result(|| {
        let json = telemetry::system_info_json().map_err(|e| (abi::TS_ERR_JSON, e))?;
        strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The header is the copy Swift compiles against, and nothing else ties
    /// it to the library, so the two numbers are compared here.
    #[test]
    fn abi_version_matches_the_header() {
        let header = include_str!("../../include/turbospark.h");
        let declared: u32 = header
            .lines()
            .find_map(|l| l.strip_prefix("#define TS_ABI_VERSION "))
            .expect("header must define TS_ABI_VERSION")
            .trim()
            .parse()
            .expect("TS_ABI_VERSION must be an integer literal");
        assert_eq!(declared, abi::ABI_VERSION);
        assert_eq!(unsafe { ts_abi_version() }, abi::ABI_VERSION);
    }

    #[test]
    fn build_info_reports_the_abi_version_and_crate_version() {
        let mut out: *mut c_char = std::ptr::null_mut();
        assert_eq!(unsafe { ts_build_info_json(&mut out) }, abi::TS_OK);
        let text = unsafe { std::ffi::CStr::from_ptr(out) }
            .to_str()
            .unwrap()
            .to_string();
        unsafe { ts_string_free(out) };
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["abiVersion"], abi::ABI_VERSION);
        assert_eq!(v["version"], env!("CARGO_PKG_VERSION"));
        assert!(v["debugAssertions"].is_boolean());
        assert_eq!(
            unsafe { ts_build_info_json(std::ptr::null_mut()) },
            abi::TS_ERR_INVALID_ARGUMENT
        );
    }
}
