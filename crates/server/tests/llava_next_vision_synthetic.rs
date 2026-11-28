//! A text and an image request to the real server over a synthetic llama trunk
//! with a LLaVA 1.6 sidecar (`docs/LLAVA_NEXT_VISION_PHASE0.md`): the open
//! with the sidecar, the checkpoint template on both turn shapes, the anyres
//! placeholder count and the injection, through HTTP. The weights are
//! untrained, so the checks are on token counts and on pixels changing the
//! output, never on what the output says.
#![cfg(target_os = "macos")]

use std::io::Cursor;
use std::sync::Arc;

use turbospark_server::{build_router, RealChatModel};

const TOKENIZER: &str = include_str!("fixtures/ChatMLTokenizer/tokenizer.json");
/// `llava-hf/llava-v1.6-vicuna-7b-hf`'s `chat_template.json`, which renders
/// list content only.
const TEMPLATE: &str = "{% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'].upper() + ': '}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\\n' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] + ' '}}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] + ' '}}{% endgeneration %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}";

fn png(width: u32, height: u32, shade: u8) -> Vec<u8> {
    let image = image::RgbImage::from_fn(width, height, |x, y| {
        let v = shade.wrapping_add(((x / 37 + y / 29) * 41) as u8);
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

#[tokio::test(flavor = "multi_thread")]
async fn a_llava_next_sidecar_serves_text_and_image_turns() {
    let root = std::env::temp_dir().join(format!(
        "turbospark-llava-next-server-{}",
        std::process::id()
    ));
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

    let model = RealChatModel::open(
        &trunk,
        Some(4096),
        None,
        Default::default(),
        runtime::Speculation::Off,
        runtime::SpeculativeDrafter::Auto,
        turbospark_server::GuardrailConfig::OFF,
        runtime::SteeringPolicy::off(),
        runtime::LoadPolicy::default(),
        tokenizer::ReasoningEffort::Off,
        None,
        true,
        1,
        Some(&sidecar),
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
    let ask = |content: serde_json::Value| {
        let request = client
            .post(format!("{base}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "m", "max_tokens": 8, "temperature": 0,
                "messages": [{"role": "user", "content": content}]
            }));
        async move {
            let response = request.send().await.unwrap();
            let status = response.status();
            let text = response.text().await.unwrap();
            assert_eq!(status, 200, "{text}");
            let body: serde_json::Value = serde_json::from_str(&text).unwrap();
            (
                body["usage"]["prompt_tokens"].as_u64().unwrap(),
                body["choices"][0]["message"]["content"].clone(),
            )
        }
    };
    let image_turn = |width: u32, height: u32, shade: u8| {
        serde_json::json!([
            {"type": "image_url", "image_url": {"url": data_url(&png(width, height, shade))}},
            {"type": "text", "text": "hi"}
        ])
    };

    // A plain-string turn keeps its text: the template reads only lists.
    let (short, _) = ask(serde_json::json!("hi")).await;
    let (long, _) = ask(serde_json::json!("hi there, friend")).await;
    assert!(
        long > short + 5,
        "the question must reach the prompt: {long} {short}"
    );

    // 448 x 336 on a 1 x 2 grid: 576 + 24 * 32 + 24 rows, plus the newline
    // the template puts after `<image>`.
    let (with_image, first) = ask(image_turn(448, 336, 10)).await;
    assert_eq!(with_image, short + 1368 + 1);
    let (_, other) = ask(image_turn(448, 336, 180)).await;
    assert_ne!(first, other, "the pixels must reach the model");
    // A square 2 x 2 image: the count is per image.
    let (square, _) = ask(image_turn(448, 448, 10)).await;
    assert_eq!(square, short + 2928 + 1);

    let _ = std::fs::remove_dir_all(&root);
}
