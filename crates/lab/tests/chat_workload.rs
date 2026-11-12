#![cfg(target_os = "macos")]
//! The chat workload's offline half: the file format, the reply a client
//! that drops reasoning sends back, and the metric that reads a chat run.
//! The conversation itself needs a real install (`tslab ab` with a `chat:`
//! prompt, `crates/lab/program.md`).

use std::path::PathBuf;

use serde_json::json;
use tokenizer::{MfTokenizer, Role};
use turbospark_lab::stats::Direction;
use turbospark_lab::worker::{visible_content, ChatFile};
use turbospark_lab::workload::{Metric, PromptSource};

fn tokenizer() -> MfTokenizer {
    let dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../runtime/tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer loads")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tslab-chat-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn a_document_goes_before_the_first_user_message_cut_to_its_cap() {
    let dir = scratch("doc");
    std::fs::write(dir.join("doc.txt"), "alpha beta gamma delta ".repeat(200)).unwrap();
    let chat = dir.join("c.chat.json");
    std::fs::write(
        &chat,
        json!({
            "opening": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "What is it?"}
            ],
            "document": "doc.txt",
            "document_tokens": 12,
            "follow_ups": ["And then?", "Why?"]
        })
        .to_string(),
    )
    .unwrap();
    let tok = tokenizer();
    let c = ChatFile::load(&chat, &tok).unwrap();
    assert_eq!(c.follow_ups, vec!["And then?", "Why?"]);
    assert_eq!(c.opening[0].role, Role::System);
    let user = c.opening[1].content.clone().unwrap();
    assert!(user.ends_with("\n\nWhat is it?"), "{user:?}");
    let doc = user.trim_end_matches("\n\nWhat is it?");
    assert!(!doc.is_empty());
    assert!(
        tok.encode(doc, false).len() <= 12,
        "{doc:?} is over the cap"
    );
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn a_chat_without_follow_ups_is_refused_by_name() {
    let dir = scratch("bad");
    let chat = dir.join("c.chat.json");
    std::fs::write(&chat, json!({"opening": []}).to_string()).unwrap();
    let err = ChatFile::load(&chat, &tokenizer()).unwrap_err();
    assert!(err.contains("follow_ups"), "{err}");
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn the_reply_sent_back_is_what_follows_the_reasoning() {
    assert_eq!(
        visible_content("<think>\nhmm\n</think>\n\nAnswer.", true),
        "Answer."
    );
    assert_eq!(
        visible_content("still thinking, cut by the budget", true),
        ""
    );
    assert_eq!(visible_content("a</think>b</think> c ", true), "c");
}

/// A template with no thinking channel never writes `</think>`. Treating its
/// reply like a cut-off reasoning turn would send every assistant turn back
/// empty and quietly change what the chat measures.
#[test]
fn a_reply_without_a_thinking_channel_goes_back_whole() {
    assert_eq!(
        visible_content("  The answer is 4.\n", false),
        "The answer is 4."
    );
    assert_eq!(visible_content("a</think>b", false), "a</think>b");
}

#[test]
fn the_reasoning_flag_defaults_on_and_must_be_a_boolean() {
    let dir = scratch("reasoning");
    let write = |name: &str, extra: serde_json::Value| {
        let mut v = json!({
            "opening": [{"role": "user", "content": "Hi"}],
            "follow_ups": ["And?"]
        });
        if !extra.is_null() {
            v["reasoning"] = extra;
        }
        let path = dir.join(name);
        std::fs::write(&path, v.to_string()).unwrap();
        path
    };
    let tok = tokenizer();
    assert!(
        ChatFile::load(&write("a.json", json!(null)), &tok)
            .unwrap()
            .reasoning
    );
    assert!(
        ChatFile::load(&write("b.json", json!(true)), &tok)
            .unwrap()
            .reasoning
    );
    assert!(
        !ChatFile::load(&write("c.json", json!(false)), &tok)
            .unwrap()
            .reasoning
    );
    let err = ChatFile::load(&write("d.json", json!("no")), &tok).unwrap_err();
    assert!(err.contains("reasoning"), "{err}");
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn followup_ttft_is_read_from_totals_and_lower_is_better() {
    let m: Metric = serde_json::from_value(json!("followup_ttft_s")).unwrap();
    assert_eq!(m, Metric::FollowupTtftS);
    assert!(matches!(m.direction(), Direction::LowerIsBetter));
    let record = json!({"totals": {"followup_ttft_s": 1.25}});
    assert_eq!(m.read(&record), Some(1.25));
    assert_eq!(m.read(&json!({"totals": {}})), None);
}

#[test]
fn the_shipped_example_chat_parses() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("chats/long-doc.chat.json");
    let c = ChatFile::load(&path, &tokenizer()).unwrap();
    assert_eq!(c.follow_ups.len(), 4);
    assert!(PromptSource::Chat(path).is_chat());
}
