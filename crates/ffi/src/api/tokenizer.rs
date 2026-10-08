//! A tokenizer opened on its own, without the engine behind it.
//!
//! `ts_session_open` maps gigabytes and compiles pipelines, which is the wrong
//! price for "how many tokens is this paragraph" in a context meter or a
//! prompt-budget preview that runs before any model is loaded. The tokenizer
//! files are small, so this handle reads only those.
//!
//! Text-level calls only. Rendering a conversation and fitting a window need
//! the family, dialect and reasoning table the engine resolves at open, so
//! they stay on `TsSession` rather than growing a second, partial copy here.

use std::os::raw::{c_char, c_int};

use tokenizer::MfTokenizer;

use crate::abi::{self, guard_result};
use crate::strings;

/// The opaque handle `TsTokenizer *` points at.
pub struct TokenizerHandle {
    tokenizer: MfTokenizer,
}

/// `TsTokenizer *` in C.
pub type TsTokenizer = TokenizerHandle;

unsafe fn borrow<'a>(ptr: *const TsTokenizer) -> Result<&'a TsTokenizer, (c_int, String)> {
    ptr.as_ref().ok_or((
        abi::TS_ERR_INVALID_ARGUMENT,
        "tokenizer must not be null".to_string(),
    ))
}

/// Opens the tokenizer of a `.gturbo` directory or installed alias. An
/// existing directory wins over an alias, exactly as in `ts_session_open`.
#[no_mangle]
pub unsafe extern "C" fn ts_tokenizer_open(
    model_dir: *const c_char,
    out: *mut *mut TsTokenizer,
) -> c_int {
    guard_result(|| {
        if out.is_null() {
            return Err((abi::TS_ERR_INVALID_ARGUMENT, "out must not be null".into()));
        }
        *out = std::ptr::null_mut();
        let model = strings::required(model_dir, "modelDir")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let resolved = catalog::resolve_model_arg(model);
        if !resolved.is_dir() {
            return Err((
                abi::TS_ERR_OPEN,
                format!(
                    "{} is not a directory (and matched no installed alias)",
                    resolved.display()
                ),
            ));
        }
        let tokenizer = MfTokenizer::load_from_dir(&resolved).map_err(|e| {
            (
                abi::TS_ERR_OPEN,
                format!(
                    "failed to load a tokenizer from {}: {e}",
                    resolved.display()
                ),
            )
        })?;
        *out = Box::into_raw(Box::new(TokenizerHandle { tokenizer }));
        Ok(())
    })
}

/// Closes a tokenizer. NULL is a no-op.
#[no_mangle]
pub unsafe extern "C" fn ts_tokenizer_close(ptr: *mut TsTokenizer) {
    abi::guard_value((), || {
        if !ptr.is_null() {
            drop(Box::from_raw(ptr));
        }
    })
}

/// The token count of a raw string.
#[no_mangle]
pub unsafe extern "C" fn ts_tokenizer_count_text_tokens(
    ptr: *const TsTokenizer,
    text: *const c_char,
    add_special: bool,
    out_count: *mut u32,
) -> c_int {
    guard_result(|| {
        if out_count.is_null() {
            return Err((
                abi::TS_ERR_INVALID_ARGUMENT,
                "out_count must not be null".into(),
            ));
        }
        let handle = borrow(ptr)?;
        let raw = strings::required(text, "text").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        *out_count = handle.tokenizer.encode(raw, add_special).len() as u32;
        Ok(())
    })
}

/// A raw string as a JSON array of token ids.
#[no_mangle]
pub unsafe extern "C" fn ts_tokenizer_tokenize_json(
    ptr: *const TsTokenizer,
    text: *const c_char,
    add_special: bool,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        let handle = borrow(ptr)?;
        let raw = strings::required(text, "text").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let json = serde_json::to_string(&handle.tokenizer.encode(raw, add_special))
            .map_err(|e| (abi::TS_ERR_JSON, e.to_string()))?;
        strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// A JSON array of token ids back to text.
#[no_mangle]
pub unsafe extern "C" fn ts_tokenizer_detokenize_json(
    ptr: *const TsTokenizer,
    tokens_json: *const c_char,
    skip_special: bool,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        let handle = borrow(ptr)?;
        let raw = strings::required(tokens_json, "tokensJson")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let tokens: Vec<i32> = serde_json::from_str(raw)
            .map_err(|e| (abi::TS_ERR_JSON, format!("tokensJson: {e}")))?;
        let text = handle.tokenizer.decode(&tokens, skip_special);
        strings::emit(&text, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}
