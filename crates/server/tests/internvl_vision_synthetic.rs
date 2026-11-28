//! An image request to the real server over a synthetic InternVL install
//! (`docs/INTERNVL_VISION_PHASE0.md`): the text-only template, the splice, the
//! tower and the injection, through HTTP. The weights are untrained, so the
//! checks are on token counts and on pixels changing the output, never on what
//! the output says.
#![cfg(target_os = "macos")]

use std::io::Cursor;
use std::sync::Arc;

use turbospark_server::{build_router, RealChatModel};

const TOKENIZER: &str = include_str!("fixtures/ChatMLTokenizer/tokenizer.json");

fn png(width: u32, height: u32, shade: u8) -> Vec<u8> {
    let image = image::RgbImage::from_fn(width, height, |x, y| {
        let v = shade.wrapping_add(((x / 7 + y / 5) * 23) as u8);
        image::Rgb([v, v.wrapping_mul(3), 255 - v])
    });
    let mut out = Cursor::new(Vec::new());
    image
        .write_to(&mut out, image::ImageFormat::Png)
        .expect("encodes");
    out.into_inner()
}

fn data_url(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::from("data:image/png;base64,");
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

fn image_part(width: u32, height: u32, shade: u8) -> serde_json::Value {
    serde_json::json!({
        "type": "image_url",
        "image_url": {"url": data_url(&png(width, height, shade))}
    })
}

#[tokio::test(flavor = "multi_thread")]
async fn an_image_request_is_served_and_its_pixels_reach_the_model() {
    let dir =
        std::env::temp_dir().join(format!("turbospark-internvl-server-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    repack::build_synthetic_internvl_install(&dir, TOKENIZER, 2, "tiny-internvl").expect("builds");
    let model = RealChatModel::open(
        &dir,
        Some(512),
        None,
        Default::default(),
        runtime::Speculation::Off,
        runtime::SpeculativeDrafter::Auto,
        turbospark_server::GuardrailConfig::OFF,
        runtime::SteeringPolicy::off(),
        runtime::LoadPolicy::default(),
        tokenizer::ReasoningEffort::Off,
        None,
        // ON: an image turn must not continue from another image's KV.
        true,
        1,
        None,
        runtime::KvQuant::default(),
        runtime::ExpertResidency::default(),
    )
    .expect("opens");
    let model: Arc<dyn turbospark_server::ChatModel> = Arc::new(model);
    let router = build_router(model);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

    let client = reqwest::Client::new();
    let ask = |messages: serde_json::Value| {
        let request = client
            .post(format!("{base}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "m", "max_tokens": 12, "temperature": 0, "messages": messages
            }));
        async move {
            let response = request.send().await.unwrap();
            let status = response.status();
            let headers = format!("{:?}", response.headers());
            let text = response.text().await.unwrap();
            (status, headers, text)
        }
    };
    let ok = |(status, headers, text): (reqwest::StatusCode, String, String)| {
        assert_eq!(status, 200, "{text}");
        assert!(!headers.contains("no vision tower"), "{headers}");
        let body: serde_json::Value = serde_json::from_str(&text).unwrap();
        (
            body["usage"]["prompt_tokens"].as_u64().unwrap(),
            body["choices"][0]["message"]["content"].clone(),
        )
    };
    let turn = |parts: Vec<serde_json::Value>| {
        let mut content = parts;
        content.push(serde_json::json!({"type": "text", "text": "hi"}));
        serde_json::json!([{"role": "user", "content": content}])
    };

    let (text_only, _) = ok(ask(serde_json::json!([{"role": "user", "content": "hi"}])).await);
    let (square, first) = ok(ask(turn(vec![image_part(56, 56, 10)])).await);
    let (_, other) = ok(ask(turn(vec![image_part(56, 56, 200)])).await);
    let (_, again) = ok(ask(turn(vec![image_part(56, 56, 10)])).await);
    // `<img>` + 4 context rows + `</img>`, then the flattened newline.
    assert_eq!(square, text_only + 7, "one tile");
    assert_ne!(first, other, "the pixels must reach the model");
    assert_eq!(first, again, "no state may survive from the other image");

    // 2x1 grid plus the thumbnail is 3 runs; beside it one square image.
    let (two, _) = ok(ask(turn(vec![image_part(112, 56, 10), image_part(56, 56, 10)])).await);
    assert_eq!(two, text_only + 15 + 7);

    // A later text turn replays the picture with the history.
    let mut history = turn(vec![image_part(56, 56, 10)]);
    let messages = history.as_array_mut().unwrap();
    messages.push(serde_json::json!({"role": "assistant", "content": "a"}));
    messages.push(serde_json::json!({"role": "user", "content": "more"}));
    let (follow_up, _) = ok(ask(history).await);
    assert!(follow_up > square, "{follow_up} against {square}");

    // A literal marker in the text is a count mismatch, refused, not expanded.
    let (status, _, text) = ask(serde_json::json!([{"role": "user", "content": [
        image_part(56, 56, 10),
        {"type": "text", "text": "<img>"}
    ]}]))
    .await;
    assert_eq!(status, 400, "{text}");
    assert!(text.contains("cannot place 1 image(s)"), "{text}");

    let _ = std::fs::remove_dir_all(&dir);
}
