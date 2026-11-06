//! A ChatML table without Qwen2.5's `<tool_call>` pair still loads.
//!
//! LLaVA-OneVision's trunk is Qwen2-7B-Instruct, whose table registers
//! `<|im_start|>`, `<|im_end|>` and `<image>` and nothing for tool calls.
//! `pull` refused it with "tokenizer missing required special token:
//! <tool_call>" before the pair became optional. The vendored ChatML fixture
//! is rewritten here with the pair removed, so the test needs no checkpoint.

use std::path::PathBuf;

use turbospark_tokenizer::{
    ChatDialect, MfTokenizer, StructuredAssistantDecoder, StructuredAssistantEvent,
    NO_SUCH_TOKEN_ID,
};

/// Copy the fixture into a scratch directory with `drop` removed from its
/// added tokens, and load it.
fn load_without(drop: &[&str], name: &str) -> Result<MfTokenizer, String> {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    let dst = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&dst).map_err(|e| e.to_string())?;
    for file in ["chat_template.jinja", "tokenizer_config.json"] {
        std::fs::copy(src.join(file), dst.join(file)).map_err(|e| e.to_string())?;
    }
    let text = std::fs::read_to_string(src.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let mut doc: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let added = doc["added_tokens"].as_array_mut().expect("added_tokens");
    let before = added.len();
    added.retain(|t| !drop.contains(&t["content"].as_str().unwrap_or("")));
    assert_eq!(before - added.len(), drop.len(), "the mutation must apply");
    std::fs::write(dst.join("tokenizer.json"), doc.to_string()).map_err(|e| e.to_string())?;
    MfTokenizer::load_from_dir(&dst).map_err(|e| e.to_string())
}

#[test]
fn a_table_with_no_tool_call_pair_resolves_as_chatml_with_no_such_ids() {
    let tok = load_without(&["<tool_call>", "</tool_call>"], "chatml_no_tool_call").unwrap();
    assert_eq!(tok.dialect, ChatDialect::ChatMl);
    assert_eq!(tok.tool_call_start_id, NO_SUCH_TOKEN_ID);
    assert_eq!(tok.tool_call_end_id, NO_SUCH_TOKEN_ID);
    // The ChatML turn markers are still resolved.
    assert!(tok.token_to_id("<|im_end|>").is_some());
}

#[test]
fn one_side_of_the_pair_without_the_other_is_still_refused() {
    let err = match load_without(&["</tool_call>"], "chatml_half_tool_call") {
        Ok(_) => panic!("a half pair must be refused"),
        Err(e) => e,
    };
    assert!(err.contains("</tool_call>"), "{err}");
}

/// Flushed text and withheld tails reach the decoder with the token id
/// `NO_SUCH_TOKEN_ID` (`consume_flushed_text`, and the runtime's
/// `RawDecodeProgress::Tail`). On this table the tool ids are that same
/// sentinel, so an unguarded comparison opened a phantom tool span on the
/// first flush (swallowing the text) and refused the second one.
#[test]
fn flushed_text_is_content_when_the_tool_ids_are_the_sentinel() {
    let tok = load_without(
        &["<tool_call>", "</tool_call>"],
        "chatml_no_tool_call_flush",
    )
    .unwrap();
    assert_eq!(tok.tool_call_start_id, NO_SUCH_TOKEN_ID);
    let mut decoder = StructuredAssistantDecoder::new(&tok, Default::default(), String::new, &[]);
    assert_eq!(
        decoder.consume_flushed_text("first").unwrap(),
        vec![StructuredAssistantEvent::Content("first".to_string())]
    );
    assert_eq!(
        decoder.consume(NO_SUCH_TOKEN_ID, "second").unwrap(),
        vec![StructuredAssistantEvent::Content("second".to_string())]
    );
    assert!(decoder.finish().unwrap().is_empty());
}
