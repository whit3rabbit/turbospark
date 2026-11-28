//! The Idefics3 / SmolVLM chat template indexes `content` as a part list, and
//! its turns end on a token no sidecar lists as an EOS. The template below is
//! `mlx-community/SmolVLM-256M-Instruct-4bit`'s `chat_template.json`; the
//! tokenizer is the ChatML fixture with the two added tokens appended.

use std::path::{Path, PathBuf};

use turbospark_tokenizer::{ContentPart, Message, MfTokenizer, Role};

const TEMPLATE: &str = "<|im_start|>{% for message in messages %}{{message['role'] | capitalize}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}";

fn install(extra_tokens: &[&str]) -> PathBuf {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    let dir = std::env::temp_dir().join(format!(
        "turbospark-idefics3-template-{}-{}",
        std::process::id(),
        extra_tokens.len()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(&fixture).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), dir.join(entry.file_name())).unwrap();
    }
    let path = dir.join("tokenizer.json");
    let mut json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let added = json["added_tokens"].as_array_mut().unwrap();
    let first = 1 + added
        .iter()
        .map(|t| t["id"].as_u64().unwrap())
        .max()
        .unwrap();
    for (next, token) in (first..).zip(extra_tokens) {
        added.push(serde_json::json!({
            "id": next, "content": token, "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": true
        }));
    }
    std::fs::write(&path, json.to_string()).unwrap();
    std::fs::write(dir.join("chat_template.jinja"), TEMPLATE).unwrap();
    dir
}

#[test]
fn text_and_image_turns_render_and_the_utterance_end_stops() {
    let dir = install(&["<end_of_utterance>", "<fake_token_around_image>"]);
    let tok = MfTokenizer::load_from_dir(&dir).expect("loads");
    let messages = vec![
        Message::new(Role::User, "Hi"),
        Message::new(Role::Assistant, "Hello"),
        Message::with_parts(
            Role::User,
            vec![ContentPart::Image, ContentPart::Text("What is it?".into())],
        ),
    ];
    let rendered = tok.apply_chat_template(&messages).expect("renders");
    assert_eq!(
        rendered,
        "<|im_start|>User: Hi<end_of_utterance>\nAssistant: Hello<end_of_utterance>\n\
         User:<image>What is it?<end_of_utterance>\nAssistant:"
    );
    let stop = tok.token_to_id("<end_of_utterance>").unwrap();
    assert!(tok.stop_token_ids.contains(&stop));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Without the family's image-padding token the stop set is left alone.
#[test]
fn another_vocabulary_does_not_gain_the_stop() {
    let dir = install(&["<end_of_utterance>"]);
    let tok = MfTokenizer::load_from_dir(&dir).expect("loads");
    let id = tok.token_to_id("<end_of_utterance>").unwrap();
    assert!(!tok.stop_token_ids.contains(&id));
    let _ = std::fs::remove_dir_all(&dir);
}
