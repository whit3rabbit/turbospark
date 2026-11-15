//! Preserve the quantizer ABI on platforms where this crate does not link the
//! repack crate (it is a macOS-only dependency here).

use std::os::raw::{c_char, c_int, c_void};

use crate::abi::{self, guard_result};
use crate::TsInstallCallback;

fn no() -> (c_int, String) {
    (
        abi::TS_ERR_UNSUPPORTED,
        "GGUF quantization is supported on macOS only".into(),
    )
}

/// This platform cannot quantize.
///
/// # Safety
/// Non-null output pointers must be writable.
#[no_mangle]
pub unsafe extern "C" fn ts_quantize_gguf(
    _options_json: *const c_char,
    _cb: TsInstallCallback,
    _userdata: *mut c_void,
    result_json: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        if !result_json.is_null() {
            *result_json = std::ptr::null_mut();
        }
        Err(no())
    })
}

/// No quantization can be running here.
#[no_mangle]
pub unsafe extern "C" fn ts_quantize_cancel() -> c_int {
    abi::guard_value(0, || 0)
}

/// This platform cannot quantize.
///
/// # Safety
/// Non-null output pointers must be writable.
#[no_mangle]
pub unsafe extern "C" fn ts_quantize_recipe_from_gguf_json(
    _path: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        if !out.is_null() {
            *out = std::ptr::null_mut();
        }
        Err(no())
    })
}

/// This platform cannot quantize.
///
/// # Safety
/// Non-null output pointers must be writable.
#[no_mangle]
pub unsafe extern "C" fn ts_quantize_types_json(out: *mut *mut c_char) -> c_int {
    guard_result(|| {
        if !out.is_null() {
            *out = std::ptr::null_mut();
        }
        Err(no())
    })
}

/// This platform cannot quantize.
///
/// # Safety
/// Non-null output pointers must be writable.
#[no_mangle]
pub unsafe extern "C" fn ts_quantize_check_updates_json(
    _provenance_path: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        if !out.is_null() {
            *out = std::ptr::null_mut();
        }
        Err(no())
    })
}
