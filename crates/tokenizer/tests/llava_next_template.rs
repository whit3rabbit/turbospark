//! The LLaVA 1.6 chat templates read `content` only as a part list. The two
//! strings below are the `chat_template.json` of
//! `llava-hf/llava-v1.6-vicuna-7b-hf` and `llava-hf/llava-v1.6-mistral-7b-hf`;
//! the tokenizer is the vendored ChatML fixture, which only has to render.

use std::path::{Path, PathBuf};

use turbospark_tokenizer::{ContentPart, Message, MfTokenizer, Role};

const VICUNA: &str = "{% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'].upper() + ': '}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] + ' '}}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] + ' '}}{% endgeneration %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}";
const MISTRAL: &str = "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<<SYS>>\n' + message['content'][0]['text'] + '\n<</SYS>>\n\n' }}{% elif message['role'] == 'user' %}{{ '[INST] ' }}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] }}{% endfor %}{{' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'][0]['text'] + '</s> '}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}";

fn load(name: &str, template: &str) -> (PathBuf, MfTokenizer) {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    let dir = std::env::temp_dir().join(format!(
        "turbospark-llava-next-template-{name}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(&fixture).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), dir.join(entry.file_name())).unwrap();
    }
    let _ = std::fs::remove_file(dir.join("chat_template.jinja"));
    std::fs::write(
        dir.join("chat_template.json"),
        serde_json::json!({ "chat_template": template }).to_string(),
    )
    .unwrap();
    let tok = MfTokenizer::load_from_dir(&dir).expect("loads");
    (dir, tok)
}

fn image_turn() -> Message {
    Message::with_parts(
        Role::User,
        vec![ContentPart::Image, ContentPart::Text("What is it?".into())],
    )
}

#[test]
fn vicuna_renders_image_turns_and_plain_turns() {
    let (dir, mut tok) = load("vicuna", VICUNA);
    let image = tok.apply_chat_template(&[image_turn()]).expect("renders");
    assert_eq!(image, "USER: <image>\nWhat is it? ASSISTANT:");
    // Without the flag the plain turn loses its text, which is why the host
    // sets it.
    let plain = [Message::new(Role::User, "Hi")];
    assert!(!tok
        .apply_chat_template(&plain)
        .expect("renders")
        .contains("Hi"));
    tok.set_string_content_as_parts();
    assert_eq!(
        tok.apply_chat_template(&plain).expect("renders"),
        "USER: Hi ASSISTANT:"
    );
    assert_eq!(
        tok.apply_chat_template(&[image_turn()]).expect("renders"),
        image
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn mistral_renders_image_turns_and_a_plain_multi_turn_chat() {
    let (dir, mut tok) = load("mistral", MISTRAL);
    tok.set_string_content_as_parts();
    assert_eq!(
        tok.apply_chat_template(&[image_turn()]).expect("renders"),
        "[INST] <image>\nWhat is it? [/INST]"
    );
    let turns = [
        Message::new(Role::User, "Hi"),
        Message::new(Role::Assistant, "Hello"),
        Message::new(Role::User, "Bye"),
    ];
    assert_eq!(
        tok.apply_chat_template(&turns).expect("renders"),
        "[INST] Hi [/INST] Hello</s> [INST] Bye [/INST]"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
