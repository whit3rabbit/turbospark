//! DeepSeek-VL2's chat template writes `{{message['content']}}` and knows no
//! content-part list, so an image part is flattened to the reference's
//! `<image>\n` (mlx-vlm `IMAGE_TOKEN_NEWLINE`) before the template runs, and
//! the prompt opens with the BOS the template does not write.
//!
//! `#[ignore]`d and env-gated like the other real-install tests here: set
//! `TURBOSPARK_DSVL2_TOKENIZER_DIR` to a DeepSeek-VL2 checkpoint (or install)
//! directory holding `tokenizer.json` and its chat template.

use std::path::PathBuf;

use turbospark_tokenizer::{ContentPart, Message, MfTokenizer, Role};

const IMAGE_ID: i32 = 128_815;
const BOS_ID: i32 = 0;

fn load() -> Option<MfTokenizer> {
    let dir = std::env::var("TURBOSPARK_DSVL2_TOKENIZER_DIR").ok()?;
    Some(MfTokenizer::load_from_dir(&PathBuf::from(dir)).expect("tokenizer loads"))
}

#[test]
#[ignore = "needs a DeepSeek-VL2 checkpoint; set TURBOSPARK_DSVL2_TOKENIZER_DIR"]
fn an_image_part_renders_as_one_image_marker_before_the_text() {
    let Some(tok) = load() else { return };
    let messages = vec![Message::with_parts(
        Role::User,
        vec![
            ContentPart::Image,
            ContentPart::Text("Describe it.".to_string()),
        ],
    )];
    let rendered = tok.apply_chat_template(&messages).expect("renders");
    assert!(
        rendered.ends_with("<|User|>: <image>\nDescribe it.\n\n<|Assistant|>:"),
        "{rendered:?}"
    );
    let ids = tok.encode(&rendered, false);
    assert_eq!(ids[0], BOS_ID, "the prompt opens with BOS");
    assert_eq!(
        ids.iter().filter(|&&i| i == IMAGE_ID).count(),
        1,
        "one marker per image: {ids:?}"
    );
    // The same turn as plain text renders without a marker.
    let plain = tok
        .apply_chat_template(&[Message::new(Role::User, "Describe it.")])
        .expect("renders");
    assert!(!plain.contains("<image>"));
}

#[test]
#[ignore = "needs a DeepSeek-VL2 checkpoint; set TURBOSPARK_DSVL2_TOKENIZER_DIR"]
fn two_images_render_two_markers_in_order() {
    let Some(tok) = load() else { return };
    let messages = vec![Message::with_parts(
        Role::User,
        vec![
            ContentPart::Image,
            ContentPart::Image,
            ContentPart::Text("Compare.".to_string()),
        ],
    )];
    let rendered = tok.apply_chat_template(&messages).expect("renders");
    assert!(
        rendered.contains("<image>\n<image>\nCompare."),
        "{rendered:?}"
    );
    assert_eq!(tok.verify_bare_image_marker(IMAGE_ID), Ok(()));
}
