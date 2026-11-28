//! Text and image turns on a synthetic llama trunk with a LLaVA 1.6 sidecar
//! through the C entry points (`docs/LLAVA_NEXT_VISION_PHASE0.md`):
//! `ts_session_open` with `visionSidecar`, its vision info, and `ts_generate`.
//! The weights are untrained, so the checks are on token counts and on pixels
//! changing the output.
#![cfg(target_os = "macos")]

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;
use std::ptr;

use turbospark_ffi::{
    abi, ts_generate, ts_last_error, ts_session_close, ts_session_info_json, ts_session_open,
    ts_string_free, Session,
};

const TOKENIZER: &str =
    include_str!("../../tokenizer/tests/fixtures/ChatMLTokenizer/tokenizer.json");
/// `llava-hf/llava-v1.6-vicuna-7b-hf`'s `chat_template.json`, which renders
/// list content only.
const TEMPLATE: &str = "{% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'].upper() + ': '}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\\n' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] + ' '}}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] + ' '}}{% endgeneration %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}";

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
    let options = c(r#"{"maxNewTokens":8,"temperature":0}"#);
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

#[test]
fn a_llava_next_sidecar_serves_text_and_image_turns_through_the_c_surface() {
    let root =
        std::env::temp_dir().join(format!("turbospark-llava-next-ffi-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (trunk, sidecar) = (root.join("trunk"), root.join("tower"));
    std::fs::create_dir_all(&trunk).expect("trunk dir");
    std::fs::create_dir_all(&sidecar).expect("sidecar dir");

    // The ChatML fixture plus `<image>`, with the LLaVA 1.6 template.
    let mut json: serde_json::Value = serde_json::from_str(TOKENIZER).expect("tokenizer json");
    json["added_tokens"]
        .as_array_mut()
        .expect("added tokens")
        .push(serde_json::json!({
            "id": 248_100, "content": "<image>", "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": true
        }));
    std::fs::write(trunk.join("tokenizer.json"), json.to_string()).expect("tokenizer");
    std::fs::write(
        trunk.join("chat_template.json"),
        serde_json::json!({ "chat_template": TEMPLATE }).to_string(),
    )
    .expect("template");
    let loaded = tokenizer::MfTokenizer::load_from_dir(&trunk).expect("loads");
    let marker = loaded.token_to_id("<image>").expect("added");
    repack::build_synthetic_dense_llama_install(&trunk, i64::from(marker) + 1, 2, "tiny-llama")
        .expect("trunk");
    repack::build_synthetic_llava_next_sidecar(&sidecar, 64, i64::from(marker)).expect("sidecar");
    let (dark, light) = (root.join("dark.png"), root.join("light.png"));
    write_png(&dark, 56, 10);
    write_png(&light, 56, 180);

    let dir_c = c(trunk.to_str().unwrap());
    let options = c(&serde_json::json!({"maxContext": 4096, "visionSidecar": sidecar}).to_string());
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
    assert_eq!(info["vision"]["maxPixels"], 672 * 672);

    let text = |content: &str| {
        let done = generate(
            session,
            &serde_json::json!([{"role": "user", "content": content}]),
        );
        done["promptTokens"].as_u64().unwrap()
    };
    let (short, long) = (text("hi"), text("hi there, friend"));
    assert!(
        long > short + 5,
        "the question must reach the prompt: {long} {short}"
    );
    let image = |path: &Path| {
        generate(
            session,
            &serde_json::json!([{"role": "user", "content": [
                {"type": "image", "path": path},
                {"type": "text", "text": "hi"}
            ]}]),
        )
    };
    let (first, other) = (image(&dark), image(&light));
    // A square image takes the 1 x 2 grid (576 + 24 * 24 + 24), plus the
    // newline the template puts after `<image>`.
    assert_eq!(first["promptTokens"].as_u64().unwrap(), short + 1176 + 1);
    assert_ne!(first["content"], other["content"], "pixels reach the model");

    unsafe { ts_session_close(session) };
    let _ = std::fs::remove_dir_all(&root);
}
