//! Generation, streaming, tokenization, and context window fitting C ABI entry points.

use std::os::raw::{c_char, c_int, c_void};

use crate::abi::{self, guard_result, parse_json_or_default};
use crate::generate;
use crate::heavy::HeavyWorkGuard;
use crate::session;
use crate::strings;
use crate::wire;
use crate::{TsEventCallback, TsSession};

/// Generates one assistant turn.
///
/// `messages_json` is `[{"role":"user","content":"..."}]`, the same shape
/// `--messages-file` takes, rendered through the checkpoint's own chat
/// template. `options_json` may be null or `{}`. `cb` may be null, in which
/// case nothing streams and the whole turn arrives in `result_json`.
///
/// Blocks for the whole turn. Call it from a background thread.
#[no_mangle]
pub unsafe extern "C" fn ts_generate(
    ptr: *const TsSession,
    messages_json: *const c_char,
    options_json: *const c_char,
    cb: TsEventCallback,
    userdata: *mut c_void,
    result_json: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        // Checked before any work: a turn can run for minutes, and a null
        // out-pointer discovered only at the end means the whole generation
        // (and any side effect it had, e.g. a KV prefix advanced) already
        // happened for nothing.
        if result_json.is_null() {
            return Err((
                abi::TS_ERR_INVALID_ARGUMENT,
                "resultJson must not be null".to_string(),
            ));
        }
        let session = session::borrow(ptr).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let raw = strings::required(messages_json, "messagesJson")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let messages: Vec<wire::WireMessage> = serde_json::from_str(raw)
            .map_err(|e| (abi::TS_ERR_JSON, format!("messagesJson: {e}")))?;
        let options = parse_json_or_default::<wire::GenerateOptions>(options_json, "options")?;
        let _heavy_work = HeavyWorkGuard::acquire();

        let result = generate::generate(session, &messages, &options, |kind, text, a, b| {
            if let Some(f) = cb {
                // The pointer is into `text`'s own buffer and is valid only
                // for this call, which is why the header says the callee
                // must copy before returning.
                f(
                    userdata,
                    kind,
                    text.as_ptr() as *const c_char,
                    text.len(),
                    a,
                    b,
                );
            }
        })
        .map_err(|e| (abi::TS_ERR_GENERATE, e))?;

        let json = serde_json::to_string(&result).map_err(|e| (abi::TS_ERR_JSON, e.to_string()))?;
        strings::emit(&json, result_json).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// Evaluates the prompt token count without running generation.
#[no_mangle]
pub unsafe extern "C" fn ts_session_count_tokens(
    ptr: *const TsSession,
    messages_json: *const c_char,
    reasoning: *const c_char,
    out_count: *mut u32,
) -> c_int {
    guard_result(|| count_tokens_impl(ptr, messages_json, reasoning, std::ptr::null(), out_count))
}

/// [`ts_session_count_tokens`] with the functions a turn will OFFER counted
/// in. `tools_json` is the same array `ts_generate`'s `tools` option takes
/// (`[{"name","description","parameters"}]`); NULL or `[]` is no tools. The
/// definitions are rendered into the prompt by the checkpoint's template and
/// can run to thousands of tokens, which a count that ignored them would miss.
#[no_mangle]
pub unsafe extern "C" fn ts_session_count_tokens_with_tools(
    ptr: *const TsSession,
    messages_json: *const c_char,
    reasoning: *const c_char,
    tools_json: *const c_char,
    out_count: *mut u32,
) -> c_int {
    guard_result(|| count_tokens_impl(ptr, messages_json, reasoning, tools_json, out_count))
}

/// The offered tools from an optional JSON array, validated exactly as
/// `ts_generate` validates `options.tools`.
unsafe fn tools_from_json(
    tools_json: *const c_char,
) -> Result<crate::generate::ToolOffer, (c_int, String)> {
    let raw = strings::optional(tools_json, "toolsJson")
        .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
    let specs: Vec<wire::WireToolSpec> = match raw.map(str::trim).filter(|r| !r.is_empty()) {
        None => Vec::new(),
        Some(text) => {
            serde_json::from_str(text).map_err(|e| (abi::TS_ERR_JSON, format!("toolsJson: {e}")))?
        }
    };
    crate::generate::tool_offer(&specs).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
}

unsafe fn count_tokens_impl(
    ptr: *const TsSession,
    messages_json: *const c_char,
    reasoning: *const c_char,
    tools_json: *const c_char,
    out_count: *mut u32,
) -> Result<(), (c_int, String)> {
    if out_count.is_null() {
        return Err((
            abi::TS_ERR_INVALID_ARGUMENT,
            "out_count must not be null".into(),
        ));
    }
    let session = session::borrow(ptr).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
    let raw = strings::required(messages_json, "messagesJson")
        .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
    let messages: Vec<wire::WireMessage> =
        serde_json::from_str(raw).map_err(|e| (abi::TS_ERR_JSON, format!("messagesJson: {e}")))?;
    let reasoning_str = strings::optional(reasoning, "reasoning")
        .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?
        .unwrap_or("off");
    let offer = tools_from_json(tools_json)?;
    let count = generate::count_tokens(session, &messages, reasoning_str, &offer.definitions)
        .map_err(|e| (abi::TS_ERR_GENERATE, e))?;
    *out_count = count;
    Ok(())
}

/// Evaluates the token count of a raw text string using the session tokenizer.
#[no_mangle]
pub unsafe extern "C" fn ts_session_count_text_tokens(
    ptr: *const TsSession,
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
        let session = session::borrow(ptr).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let raw = strings::required(text, "text").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let count = generate::count_text_tokens(session, raw, add_special);
        *out_count = count;
        Ok(())
    })
}

/// Formats a conversation transcript into raw prompt text using the session's
/// chat template and reasoning effort setting.
#[no_mangle]
pub unsafe extern "C" fn ts_session_render_prompt(
    ptr: *const TsSession,
    messages_json: *const c_char,
    reasoning: *const c_char,
    out_prompt: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        let session = session::borrow(ptr).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let raw = strings::required(messages_json, "messagesJson")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let messages: Vec<wire::WireMessage> = serde_json::from_str(raw)
            .map_err(|e| (abi::TS_ERR_JSON, format!("messagesJson: {e}")))?;
        let reasoning_str = strings::optional(reasoning, "reasoning")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?
            .unwrap_or("off");
        let rendered = generate::render_prompt(session, &messages, reasoning_str)
            .map_err(|e| (abi::TS_ERR_GENERATE, e))?;
        strings::emit(&rendered, out_prompt).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// Tokenizes raw text into a JSON array of integer token IDs using the session tokenizer.
#[no_mangle]
pub unsafe extern "C" fn ts_session_tokenize_json(
    ptr: *const TsSession,
    text: *const c_char,
    add_special: bool,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        let session = session::borrow(ptr).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let raw = strings::required(text, "text").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let tokens = generate::tokenize(session, raw, add_special);
        let json = serde_json::to_string(&tokens).map_err(|e| (abi::TS_ERR_JSON, e.to_string()))?;
        strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// Detokenizes a JSON array of integer token IDs into text using the session tokenizer.
#[no_mangle]
pub unsafe extern "C" fn ts_session_detokenize_json(
    ptr: *const TsSession,
    tokens_json: *const c_char,
    skip_special: bool,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        let session = session::borrow(ptr).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let raw = strings::required(tokens_json, "tokensJson")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let tokens: Vec<i32> = serde_json::from_str(raw)
            .map_err(|e| (abi::TS_ERR_JSON, format!("tokensJson: {e}")))?;
        let text = generate::detokenize(session, &tokens, skip_special);
        strings::emit(&text, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// Fits a conversation transcript into a context token budget using `turbospark-window-fit`.
#[no_mangle]
pub unsafe extern "C" fn ts_session_fit_window_json(
    ptr: *const TsSession,
    messages_json: *const c_char,
    reasoning: *const c_char,
    max_tokens: u32,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        fit_window_impl(
            ptr,
            messages_json,
            reasoning,
            std::ptr::null(),
            max_tokens,
            out,
        )
    })
}

/// [`ts_session_fit_window_json`] measuring every candidate prompt WITH the
/// offered functions rendered in (see [`ts_session_count_tokens_with_tools`]),
/// so the turns kept are the turns that fit beside the tool definitions.
#[no_mangle]
pub unsafe extern "C" fn ts_session_fit_window_with_tools_json(
    ptr: *const TsSession,
    messages_json: *const c_char,
    reasoning: *const c_char,
    tools_json: *const c_char,
    max_tokens: u32,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| fit_window_impl(ptr, messages_json, reasoning, tools_json, max_tokens, out))
}

unsafe fn fit_window_impl(
    ptr: *const TsSession,
    messages_json: *const c_char,
    reasoning: *const c_char,
    tools_json: *const c_char,
    max_tokens: u32,
    out: *mut *mut c_char,
) -> Result<(), (c_int, String)> {
    let session = session::borrow(ptr).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
    let raw = strings::required(messages_json, "messagesJson")
        .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
    let messages: Vec<wire::WireMessage> =
        serde_json::from_str(raw).map_err(|e| (abi::TS_ERR_JSON, format!("messagesJson: {e}")))?;
    let reasoning_str = strings::optional(reasoning, "reasoning")
        .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?
        .unwrap_or("off");
    let offer = tools_from_json(tools_json)?;
    let outcome = generate::fit_window(
        session,
        &messages,
        reasoning_str,
        max_tokens,
        &offer.definitions,
    )
    .map_err(|e| (abi::TS_ERR_GENERATE, e))?;
    let json = serde_json::to_string(&outcome).map_err(|e| (abi::TS_ERR_JSON, e.to_string()))?;
    strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
}
