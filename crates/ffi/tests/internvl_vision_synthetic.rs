//! Image turns on a synthetic InternVL install through the C entry points
//! (`docs/INTERNVL_VISION_PHASE0.md`): `ts_session_open`, `ts_generate` and
//! the in-process server's own dispatch. The weights are untrained, so the
//! checks are on token counts and on pixels changing the output.
#![cfg(target_os = "macos")]

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;
use std::ptr;

use turbospark_ffi::{
    abi, ts_generate, ts_last_error, ts_server_info_json, ts_server_start, ts_server_stop,
    ts_session_close, ts_session_count_tokens, ts_session_info_json, ts_session_open,
    ts_string_free, Server, Session,
};

const TOKENIZER: &str =
    include_str!("../../tokenizer/tests/fixtures/ChatMLTokenizer/tokenizer.json");

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}

unsafe fn take(out: *mut c_char) -> serde_json::Value {
    let s = CStr::from_ptr(out).to_str().unwrap().to_string();
    ts_string_free(out);
    serde_json::from_str(&s).unwrap()
}

fn last_error() -> String {
    let mut buf = vec![0 as c_char; 4096];
    unsafe { ts_last_error(buf.as_mut_ptr(), buf.len()) };
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

unsafe extern "C" fn ignore(_: *mut c_void, _: c_int, _: *const c_char, _: usize, _: u32, _: u32) {}

/// A PNG with a stored (uncompressed) deflate stream: `ffi` has no image
/// encoder among its dev-dependencies and the decoder takes PNG or JPEG.
fn write_png(path: &Path, side: u32, shade: u8) {
    let mut raw = Vec::new();
    for y in 0..side {
        raw.push(0u8); // no row filter
        for x in 0..side {
            let v = shade.wrapping_add(((x / 7 + y / 5) * 23) as u8);
            raw.extend([v, v.wrapping_mul(3), 255 - v]);
        }
    }
    assert!(raw.len() < 65_536, "one stored block");
    let len = raw.len() as u16;
    let mut zlib = vec![0x78, 0x01, 0x01];
    zlib.extend(len.to_le_bytes());
    zlib.extend((!len).to_le_bytes());
    zlib.extend(&raw);
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in &raw {
        a = (a + u32::from(byte)) % 65_521;
        b = (b + a) % 65_521;
    }
    zlib.extend(((b << 16) | a).to_be_bytes());

    let mut out = vec![0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];
    let mut chunk = |kind: &[u8; 4], data: &[u8]| {
        out.extend((data.len() as u32).to_be_bytes());
        let body: Vec<u8> = kind.iter().chain(data).copied().collect();
        let mut crc = u32::MAX;
        for &byte in &body {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                crc = (crc >> 1) ^ (0xedb8_8320 & (crc & 1).wrapping_neg());
            }
        }
        out.extend(body);
        out.extend((!crc).to_be_bytes());
    };
    let mut header = Vec::new();
    header.extend(side.to_be_bytes());
    header.extend(side.to_be_bytes());
    header.extend([8, 2, 0, 0, 0]); // 8-bit RGB
    chunk(b"IHDR", &header);
    chunk(b"IDAT", &zlib);
    chunk(b"IEND", &[]);
    std::fs::write(path, out).expect("writes the image");
}

fn generate(session: *const Session, messages: &serde_json::Value) -> serde_json::Value {
    let messages = c(&messages.to_string());
    let options = c(r#"{"maxNewTokens":12,"temperature":0}"#);
    let mut result: *mut c_char = ptr::null_mut();
    let code = unsafe {
        ts_generate(
            session,
            messages.as_ptr(),
            options.as_ptr(),
            Some(ignore),
            ptr::null_mut(),
            &mut result,
        )
    };
    assert_eq!(code, abi::TS_OK, "generate: {}", last_error());
    unsafe { take(result) }
}

fn turn(path: &Path) -> serde_json::Value {
    serde_json::json!([{"role": "user", "content": [
        {"type": "image", "path": path},
        {"type": "text", "text": "hi"}
    ]}])
}

#[tokio::test(flavor = "multi_thread")]
async fn image_turns_work_on_ts_generate_and_the_in_process_server() {
    let dir = std::env::temp_dir().join(format!("turbospark-internvl-ffi-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    repack::build_synthetic_internvl_install(&dir, TOKENIZER, 2, "tiny-internvl").expect("builds");
    let (dark, light) = (dir.join("dark.png"), dir.join("light.png"));
    write_png(&dark, 56, 10);
    write_png(&light, 56, 200);

    let dir_c = c(dir.to_str().unwrap());
    let options = c(r#"{"maxContext":512}"#);
    let mut session: *mut Session = ptr::null_mut();
    let code = unsafe { ts_session_open(dir_c.as_ptr(), options.as_ptr(), &mut session) };
    assert_eq!(code, abi::TS_OK, "open: {}", last_error());

    let mut out: *mut c_char = ptr::null_mut();
    assert_eq!(
        unsafe { ts_session_info_json(session, &mut out) },
        abi::TS_OK
    );
    let info = unsafe { take(out) };
    assert_eq!(info["vision"]["active"], true, "{}", info["vision"]);
    assert_eq!(info["vision"]["maxPixels"], 12 * 56 * 56);
    assert_eq!(info["speculation"]["block"], serde_json::Value::Null);
    assert_eq!(info["prefixReuse"], true);

    // The token count renders the same flattened parts a turn does, so the
    // text-only template must not fail it.
    let messages = c(&turn(&dark).to_string());
    let reasoning = c("off");
    let mut counted = 0u32;
    let code = unsafe {
        ts_session_count_tokens(session, messages.as_ptr(), reasoning.as_ptr(), &mut counted)
    };
    assert_eq!(code, abi::TS_OK, "count: {}", last_error());

    let text_only = generate(
        session,
        &serde_json::json!([{"role": "user", "content": "hi"}]),
    );
    let text_only = text_only["promptTokens"].as_u64().unwrap();
    let first = generate(session, &turn(&dark));
    let other = generate(session, &turn(&light));
    let again = generate(session, &turn(&dark));
    // `<img>` + 4 context rows + `</img>`, then the flattened newline.
    assert_eq!(first["promptTokens"].as_u64().unwrap(), text_only + 7);
    assert_eq!(
        u64::from(counted),
        text_only + 2,
        "one marker before expansion"
    );
    assert_ne!(first["content"], other["content"], "pixels reach the model");
    assert_eq!(
        first["content"], again["content"],
        "no KV from the other image"
    );
    assert_eq!(again["reusedPrefixTokens"], 0);

    // The in-process server over the same session.
    let mut server: *mut Server = ptr::null_mut();
    let options = c(r#"{"port":0,"guardrails":"off"}"#);
    let code = unsafe { ts_server_start(session, options.as_ptr(), &mut server) };
    assert_eq!(code, abi::TS_OK, "server: {}", last_error());
    let mut out: *mut c_char = ptr::null_mut();
    assert_eq!(unsafe { ts_server_info_json(server, &mut out) }, abi::TS_OK);
    let info = unsafe { take(out) };
    let base = format!(
        "http://{}:{}",
        info["host"].as_str().unwrap(),
        info["port"].as_u64().unwrap()
    );
    let model = info["modelId"].clone();
    let ask = |shade_path: &Path| {
        // `data:` payload of the same PNG the direct turn read.
        let payload = base64(&std::fs::read(shade_path).unwrap());
        let body = serde_json::json!({
            "model": model, "max_tokens": 12, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{payload}")}},
                {"type": "text", "text": "hi"}
            ]}]
        });
        let url = format!("{base}/v1/chat/completions");
        async move {
            let response = reqwest::Client::new()
                .post(url)
                .json(&body)
                .send()
                .await
                .unwrap();
            let status = response.status();
            let text = response.text().await.unwrap();
            assert_eq!(status, 200, "{text}");
            serde_json::from_str::<serde_json::Value>(&text).unwrap()
        }
    };
    let served = ask(&dark).await;
    let served_other = ask(&light).await;
    assert_eq!(
        served["usage"]["prompt_tokens"].as_u64().unwrap(),
        text_only + 7
    );
    assert_ne!(
        served["choices"][0]["message"]["content"],
        served_other["choices"][0]["message"]["content"],
        "pixels reach the model through the in-process server"
    );

    unsafe {
        ts_server_stop(server);
        ts_session_close(session);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

fn base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let mut word = [0u8; 4];
        word[1..=chunk.len()].copy_from_slice(chunk);
        let word = u32::from_be_bytes(word);
        for i in 0..=chunk.len() {
            out.push(TABLE[(word >> (18 - 6 * i)) as usize & 63] as char);
        }
        out.extend(std::iter::repeat_n('=', 3 - chunk.len()));
    }
    out
}
