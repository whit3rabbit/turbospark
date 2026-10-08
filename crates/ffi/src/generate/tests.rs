//! Unit tests for turn speculation gating, wire serialization, and image extraction.

use super::*;
use crate::generate::prompt::collect_image_parts;
use crate::wire::WireMessage;

/// The per-turn half of the speculation decision, in all eight states.
///
/// Cheap enough to be exhaustive, and worth being: seven of the eight
/// cells are "decode sequentially" and the one that is not is the only
/// path in this crate that reaches a batched verify.
#[test]
fn a_turn_speculates_only_when_the_session_can_and_the_turn_is_greedy() {
    assert_eq!(turn_block(Some(2), true, false), Some(2));
    // Sampled: the session's block is DISCARDED rather than honoured,
    // because acceptance is exact only at temperature 0.
    assert_eq!(turn_block(Some(2), false, false), None);
    // Rate-controlled: speculation cannot silently bypass the cap or its
    // thermal and memory-pressure probes.
    assert_eq!(turn_block(Some(2), true, true), None);
    assert_eq!(turn_block(Some(2), false, true), None);
    // No drafter: greedy does not conjure one.
    assert_eq!(turn_block(None, true, false), None);
    assert_eq!(turn_block(None, false, false), None);
    assert_eq!(turn_block(None, true, true), None);
    assert_eq!(turn_block(None, false, true), None);
}

fn parse(json: &str) -> Vec<WireMessage> {
    serde_json::from_str(json).expect("wire messages parse")
}

/// **THE REGRESSION GUARD FOR WIDENING `content`.** Every caller that
/// predates images sends a bare string, and `untagged` is what keeps that
/// decoding to the same thing. A parts-only shape would have been an ABI
/// break dressed as a field.
#[test]
fn a_bare_string_content_still_decodes_and_reserializes_as_one() {
    let messages = parse(r#"[{"role":"user","content":"hi"}]"#);
    assert_eq!(messages[0].text(), "hi");
    assert!(messages[0].parts().is_none());
    assert_eq!(messages[0].image_parts().count(), 0);
    // Round-trips as a STRING, not as an object: `WindowFitOutcome`
    // hands `retained` straight back to the caller, so a re-serialized
    // message that changed shape would break every existing consumer.
    let back = serde_json::to_string(&messages).expect("serializes");
    assert!(back.contains(r#""content":"hi""#), "{back}");
}

/// A missing `content` is still the empty string rather than an error,
/// which is what `#[serde(default)]` meant before this widening too.
#[test]
fn an_absent_content_is_the_empty_string() {
    let messages = parse(r#"[{"role":"user"}]"#);
    assert_eq!(messages[0].text(), "");
    assert!(messages[0].parts().is_none());
}

/// Parts keep the order they arrived in, and `text()` sees only the
/// prose. Order is what pairs the nth picture with the nth marker run.
#[test]
fn ordered_parts_decode_in_order_and_text_skips_the_images() {
    let messages = parse(
        r#"[{"role":"user","content":[
             {"type":"image","path":"/a.png"},
             {"type":"text","text":"one"},
             {"type":"image","base64":"QQ=="},
             {"type":"text","text":"two"}]}]"#,
    );
    assert_eq!(messages[0].text(), "onetwo");
    assert_eq!(messages[0].parts().expect("parts").len(), 4);

    let images: Vec<_> = collect_image_parts(&messages).expect("shapes are valid");
    assert_eq!(images.len(), 2);
    // The FIRST image is the path one, because that is the order sent.
    match images[0].image_source().expect("a source") {
        crate::wire::ImageSource::Path(p) => assert_eq!(p, "/a.png"),
        crate::wire::ImageSource::Base64(_) => panic!("images were reordered"),
    }
    match images[1].image_source().expect("a source") {
        crate::wire::ImageSource::Base64(b) => assert_eq!(b, "QQ=="),
        crate::wire::ImageSource::Path(_) => panic!("images were reordered"),
    }
}

/// Images are collected ACROSS messages, still in order: a conversation
/// can carry a picture in an earlier turn as well as the current one.
#[test]
fn image_parts_are_collected_across_messages_in_order() {
    let messages = parse(
        r#"[{"role":"user","content":[{"type":"image","path":"/first.png"}]},
            {"role":"assistant","content":"ok"},
            {"role":"user","content":[{"type":"image","path":"/second.png"}]}]"#,
    );
    let images = collect_image_parts(&messages).expect("shapes are valid");
    let paths: Vec<_> = images
        .iter()
        .map(|i| match i.image_source().expect("a source") {
            crate::wire::ImageSource::Path(p) => p.to_string(),
            crate::wire::ImageSource::Base64(_) => unreachable!(),
        })
        .collect();
    assert_eq!(paths, ["/first.png", "/second.png"]);
}

/// Both spellings, or neither, is REFUSED rather than resolved by
/// precedence -- and refused BEFORE the engine lock, which is what makes
/// it cheap. A caller that sent both meant one of them, and picking
/// silently runs the wrong picture with no error anywhere.
#[test]
fn an_image_part_needs_exactly_one_source() {
    let both =
        parse(r#"[{"role":"user","content":[{"type":"image","path":"/a","base64":"QQ=="}]}]"#);
    let err = collect_image_parts(&both).expect_err("both sources is refused");
    assert!(err.contains("both"), "{err}");

    let neither = parse(r#"[{"role":"user","content":[{"type":"image"}]}]"#);
    let err = collect_image_parts(&neither).expect_err("no source is refused");
    assert!(err.contains("neither"), "{err}");
}

/// A conversation with no image parts collects nothing, which is the
/// condition every text-only turn takes and the one that keeps its path
/// byte-identical.
#[test]
fn a_text_only_conversation_collects_no_images() {
    let messages = parse(
        r#"[{"role":"user","content":"hi"},
            {"role":"user","content":[{"type":"text","text":"still text"}]}]"#,
    );
    assert!(collect_image_parts(&messages)
        .expect("no shapes to get wrong")
        .is_empty());
}

/// **THE `TS_EVENT_TOOL` / `toolCalls` SHAPE, PINNED WHERE NOTHING ELSE CAN
/// SEE IT.** No `GenerateOptions` field offers tools, so no end-to-end case
/// can reach a non-empty tool call at all -- `the_finish_event_terminates_a_
/// successful_event_stream_exactly_once` asserts the EMPTY array, which is a
/// true statement about today and says nothing about the row shape. Swift's
/// `GenerationToolCall(parsingJSON:)` is already written against these three
/// spellings, so renaming one here would break a consumer with no gate
/// between the two sides. Calling the pure function directly is the only way
/// to assert it before the surface goes live.
///
/// **`arguments` MUST BE AN OBJECT AND NOT A STRING.** Those are the two
/// plausible encodings of a parsed call and they are not interchangeable: a
/// host reading `arguments.city` gets `nil` from the string form, with no
/// error, which reads as a model that sent no arguments.
#[test]
fn a_tool_call_row_carries_id_name_and_an_object_of_arguments() {
    let arguments = tokenizer::JsonValue::parse(r#"{"city":"Oslo","days":3}"#)
        .expect("fixture arguments parse");
    let call = tokenizer::ParsedToolCall {
        id: "toolu_0".to_string(),
        name: "get_weather".to_string(),
        arguments_json: arguments.encoded(),
        arguments,
    };

    let row = tool_call_json(&call);
    assert_eq!(row["id"], "toolu_0");
    assert_eq!(row["name"], "get_weather");
    // An OBJECT, addressable field by field, never the encoded string.
    assert!(row["arguments"].is_object(), "{row}");
    assert_eq!(row["arguments"]["city"], "Oslo");
    assert_eq!(row["arguments"]["days"], 3);
    // Exactly three keys: a host switching on this row must not have to
    // tolerate a fourth appearing without a version bump.
    assert_eq!(row.as_object().expect("an object").len(), 3, "{row}");

    // The whole row at once. Compared as a VALUE and not as a string: key
    // ORDER is not part of this contract (it follows whether serde_json is
    // built with `preserve_order`, which is a dependency's business), while
    // the key SET and the nesting are.
    assert_eq!(
        row,
        serde_json::json!({
            "id": "toolu_0",
            "name": "get_weather",
            "arguments": {"city": "Oslo", "days": 3},
        })
    );
}

// ------------------------------------------------------------------- tools

use crate::wire::WireToolSpec;
use tokenizer::{MfTokenizer, ReasoningEffort};

fn chatml() -> MfTokenizer {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tokenizer/tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer loads")
}

fn weather_spec() -> WireToolSpec {
    serde_json::from_value(serde_json::json!({
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }))
    .unwrap()
}

#[test]
fn an_offer_with_no_tools_is_empty_and_changes_nothing() {
    let offer = tool_offer(&[]).unwrap();
    assert!(offer.definitions.is_empty());
    assert!(offer.names.is_empty());
}

#[test]
fn a_valid_offer_carries_name_description_and_schema() {
    let offer = tool_offer(&[weather_spec()]).unwrap();
    assert_eq!(offer.definitions.len(), 1);
    assert_eq!(offer.definitions[0].name, "get_weather");
    assert_eq!(
        offer.definitions[0].description,
        "Current weather for a city"
    );
    assert!(offer.names.contains("get_weather"));
    assert!(offer.definitions[0].parameters.as_object().is_some());
}

#[test]
fn a_bad_offer_is_refused_by_name() {
    let mut blank = weather_spec();
    blank.name = "  ".into();
    assert!(tool_offer(&[blank]).unwrap_err().contains("non-empty name"));

    let twice = tool_offer(&[weather_spec(), weather_spec()]).unwrap_err();
    assert!(
        twice.contains("get_weather") && twice.contains("more than once"),
        "{twice}"
    );

    let mut scalar = weather_spec();
    scalar.parameters = Some(serde_json::json!("not a schema"));
    let e = tool_offer(&[scalar]).unwrap_err();
    assert!(e.contains("JSON Schema object"), "{e}");
}

/// The no-tools render must be exactly the render this crate always did: a
/// host that never offers a tool must not see its prompts move.
#[test]
fn rendering_without_tools_is_byte_identical_to_the_plain_template_path() {
    let tok = chatml();
    let messages = parse(r#"[{"role":"user","content":"Hello there"}]"#);
    let (ids, _) = render(&tok, &messages, ReasoningEffort::Off, &[]).unwrap();
    let plain = tok
        .apply_chat_template_with_reasoning(
            &[tokenizer::Message::new(
                tokenizer::Role::User,
                "Hello there",
            )],
            ReasoningEffort::Off,
        )
        .unwrap();
    assert_eq!(ids, tok.encode(&plain, false));
}

#[test]
fn offering_a_tool_puts_its_definition_in_the_prompt() {
    let tok = chatml();
    let messages = parse(r#"[{"role":"user","content":"Weather in Oslo?"}]"#);
    let offer = tool_offer(&[weather_spec()]).unwrap();
    let (with, _) = render(&tok, &messages, ReasoningEffort::Off, &offer.definitions).unwrap();
    let (without, _) = render(&tok, &messages, ReasoningEffort::Off, &[]).unwrap();
    let text = tok.decode(&with, false);
    assert!(text.contains("get_weather"), "{text}");
    assert!(text.contains("Current weather for a city"), "{text}");
    assert!(text.contains("city"), "{text}");
    assert_ne!(with, without);
    assert!(with.len() > without.len());
}

/// A call the model made earlier, and the result that answered it, must reach
/// the template: otherwise the model is re-asked a question it already
/// answered, with no memory of having called anything.
#[test]
fn a_replayed_tool_call_and_its_result_reach_the_rendered_prompt() {
    let tok = chatml();
    let messages = parse(
        r#"[{"role":"user","content":"Weather in Oslo?"},
            {"role":"assistant","content":"","toolCalls":[
                {"id":"toolu_0","name":"get_weather","arguments":{"city":"Oslo"}}]},
            {"role":"tool","toolCallId":"toolu_0","name":"get_weather","content":"RESULT-12C"}]"#,
    );
    let offer = tool_offer(&[weather_spec()]).unwrap();
    let (ids, _) = render(&tok, &messages, ReasoningEffort::Off, &offer.definitions).unwrap();
    let text = tok.decode(&ids, false);
    assert!(text.contains("Oslo"), "the call's arguments: {text}");
    assert!(text.contains("RESULT-12C"), "the tool result: {text}");
}

#[test]
fn snake_case_tool_fields_and_string_arguments_are_accepted() {
    let messages = parse(
        r#"[{"role":"assistant","content":"","tool_calls":[
                {"id":"a","name":"f","arguments":"{\"x\":1}"}]},
            {"role":"tool","tool_call_id":"a","content":"ok"}]"#,
    );
    assert_eq!(messages[0].tool_calls.len(), 1);
    assert_eq!(messages[1].tool_call_id.as_deref(), Some("a"));
    // And the serialized form (what `fitWindow` hands back) is camelCase, and
    // omits the fields on a message that has none.
    let out = serde_json::to_value(&messages[1]).unwrap();
    assert_eq!(out["toolCallId"], "a");
    let plain = serde_json::to_value(&parse(r#"[{"role":"user","content":"hi"}]"#)[0]).unwrap();
    assert!(
        plain.get("toolCalls").is_none() && plain.get("toolCallId").is_none(),
        "{plain}"
    );
}

#[test]
fn a_parsed_call_reports_tool_calls_whatever_closed_the_turn() {
    use runtime::StopReason as R;
    for raw in [
        R::EndOfTurn,
        R::Eos,
        R::StopString,
        R::MaxTokens,
        R::ToolCalls,
    ] {
        assert_eq!(reported_stop_reason(raw, true), "toolCalls", "{raw:?}");
    }
    // Without a call the raw reason is reported unchanged.
    assert_eq!(reported_stop_reason(R::EndOfTurn, false), "endOfTurn");
    assert_eq!(reported_stop_reason(R::Eos, false), "eos");
    assert_eq!(reported_stop_reason(R::MaxTokens, false), "maxTokens");
    // A Stop press is never "run the tool", even if a call had been parsed.
    assert_eq!(reported_stop_reason(R::Cancelled, true), "cancelled");
}

/// The exact shape `ChatMessage` in the Swift package encodes: camelCase keys
/// and `arguments` as the JSON TEXT of the object. Pinned here because no test
/// can drive the two sides together without a model, and a spelling drift
/// would silently replay no calls at all.
#[test]
fn the_swift_encoded_tool_message_shape_renders_the_call_and_result() {
    let tok = chatml();
    let messages = parse(
        r#"[{"role":"user","content":"Weather in Oslo?"},
            {"role":"assistant","content":"","toolCalls":[
                {"id":"toolu_0","name":"get_weather","arguments":"{\"city\":\"Oslo\"}"}]},
            {"role":"tool","content":"RESULT-9C","toolCallId":"toolu_0","name":"get_weather"}]"#,
    );
    assert_eq!(
        messages[1].tool_calls[0].arguments,
        serde_json::json!("{\"city\":\"Oslo\"}")
    );
    let offer = tool_offer(&[weather_spec()]).unwrap();
    let (ids, _) = render(&tok, &messages, ReasoningEffort::Off, &offer.definitions).unwrap();
    let text = tok.decode(&ids, false);
    assert!(text.contains("Oslo"), "{text}");
    assert!(text.contains("RESULT-9C"), "{text}");
}
