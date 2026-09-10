use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use turbospark_tokenizer::{ChatDialect, Message, MfTokenizer, Role, ToolCallSupport};
static COUNTER: AtomicU64 = AtomicU64::new(0);
fn fixture() -> (PathBuf, MfTokenizer) {
    let dir = std::env::temp_dir().join(format!(
        "minimax-tokenizer-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let marks = ["<unk>", "]~!b[", "]~b]", "[e~[", "<think>", "</think>"];
    let vocab: serde_json::Map<String, serde_json::Value> = marks
        .iter()
        .enumerate()
        .map(|(i, s)| (s.to_string(), serde_json::json!(i)))
        .collect();
    let tokens: Vec<_> = marks.iter().enumerate().map(|(i,s)| serde_json::json!({"id":i,"content":s,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true})).collect();
    let json = serde_json::json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":tokens,"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":vocab,"unk_token":"<unk>"}});
    std::fs::write(dir.join("tokenizer.json"), json.to_string()).unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"bos_token":"]~!b[","eos_token":"[e~["}"#,
    )
    .unwrap();
    let t = MfTokenizer::load_from_dir(&dir).unwrap();
    (dir, t)
}
#[test]
fn minimax_needs_no_pad_and_uses_checkpoint_eos_without_duplicate_bos() {
    let (dir, t) = fixture();
    assert_eq!(t.dialect, ChatDialect::MiniMax);
    let eos = t.token_to_id("[e~[").unwrap();
    assert_eq!(t.pad_id, eos);
    assert_eq!(t.end_of_turn_id, eos);
    assert_eq!(t.stop_token_ids, [eos].into_iter().collect());
    let bos = t.token_to_id("]~!b[").unwrap();
    assert_eq!(t.encode("]~!b[", true), [bos]);
    assert_eq!(t.dialect.tool_call_support(), ToolCallSupport::Prompted);
    assert!(t
        .apply_chat_template(&[Message::new(Role::User, "hi")])
        .unwrap_err()
        .to_string()
        .contains("checkpoint chat template"));
    std::fs::remove_dir_all(dir).unwrap();
}
#[test]
#[ignore = "requires pinned MiniMax tokenizer sidecars via TURBOSPARK_MINIMAX_INSTALL_DIR"]
fn real_minimax_template_renders_and_opens_thought() {
    let dir = std::env::var_os("TURBOSPARK_MINIMAX_INSTALL_DIR").expect("set install directory");
    let t = MfTokenizer::load_from_dir(&PathBuf::from(dir)).unwrap();
    let rendered = t
        .apply_chat_template(&[Message::new(Role::User, "hi")])
        .unwrap();
    assert!(
        rendered.trim_start().starts_with("]~!b[]~b]system\n"),
        "{rendered:?}"
    );
    assert!(rendered.contains("]~b]user\nhi[e~[\n"));
    assert!(rendered.ends_with("]~b]ai\n<think>\n"));
    let ids = t.encode(&rendered, true);
    assert_eq!(ids.iter().filter(|&&id| id == t.bos_id).count(), 1);
    assert!(t.stop_token_ids.contains(&t.token_to_id("[e~[").unwrap()));
}

#[test]
fn minimax_splits_prefilled_thought_and_leaves_tool_markup_as_text() {
    use turbospark_tokenizer::{StructuredAssistantDecoder, StructuredAssistantEvent};
    let (dir, t) = fixture();
    let start = t.token_to_id("<think>").unwrap();
    let end = t.token_to_id("</think>").unwrap();
    let mut decoder =
        StructuredAssistantDecoder::new(&t, Default::default(), || "unused".into(), &[start]);
    assert_eq!(
        decoder.consume(-1, "reasoning").unwrap(),
        [StructuredAssistantEvent::Reasoning("reasoning".into())]
    );
    assert!(decoder.consume(end, "</think>").unwrap().is_empty());
    let text = "answer <minimax:tool_call>unparsed</minimax:tool_call>";
    assert_eq!(
        decoder.consume(-1, text).unwrap(),
        [StructuredAssistantEvent::Content(text.into())]
    );
    assert!(!decoder.has_tool_calls());
    std::fs::remove_dir_all(dir).unwrap();
}
