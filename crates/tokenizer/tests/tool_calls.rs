//! Unit tests for the three tool-call DSL parsers and the JSON value schema
//! normalization, independent of any loaded tokenizer.

use std::collections::HashSet;

use turbospark_tokenizer::{
    DeepseekToolCallParser, GemmaToolCallParser, JsonValue, QwenToolCallParser, ToolCallParserError,
};

fn tools(names: &[&str]) -> HashSet<String> {
    names.iter().map(|s| s.to_string()).collect()
}

#[test]
fn gemma_parses_simple_call() {
    let parser = GemmaToolCallParser::new();
    let call = parser
        .parse(
            "call:get_weather{city:\"NYC\",days:3}",
            &tools(&["get_weather"]),
            "id1",
        )
        .unwrap();
    assert_eq!(call.name, "get_weather");
    assert_eq!(call.id, "id1");
    let obj = call.arguments.as_object().unwrap();
    assert_eq!(obj["city"], JsonValue::String("NYC".to_string()));
    assert_eq!(obj["days"], JsonValue::Integer(3));
}

#[test]
fn gemma_rejects_unknown_tool() {
    let parser = GemmaToolCallParser::new();
    let err = parser
        .parse("call:evil{}", &tools(&["good"]), "id1")
        .unwrap_err();
    assert_eq!(err, ToolCallParserError::UnknownTool("evil".to_string()));
}

#[test]
fn gemma_rejects_malformed_call() {
    let parser = GemmaToolCallParser::new();
    let err = parser
        .parse("call:get_weather{city:}", &tools(&["get_weather"]), "id1")
        .unwrap_err();
    assert_eq!(err, ToolCallParserError::Malformed);
}

#[test]
fn gemma_custom_quoted_string_matches_json_quoted() {
    let parser = GemmaToolCallParser::new();
    let call = parser
        .parse("call:f{a:<|\"|>hi<|\"|>}", &tools(&["f"]), "id1")
        .unwrap();
    assert_eq!(
        call.arguments.as_object().unwrap()["a"],
        JsonValue::String("hi".to_string())
    );
}

#[test]
fn qwen_parses_function_with_parameters() {
    let parser = QwenToolCallParser::new();
    let text = "\n<function=get_weather>\n<parameter=city>\nNYC\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n";
    let call = parser.parse(text, &tools(&["get_weather"]), "id2").unwrap();
    assert_eq!(call.name, "get_weather");
    let obj = call.arguments.as_object().unwrap();
    assert_eq!(obj["city"], JsonValue::String("NYC".to_string()));
    assert_eq!(obj["days"], JsonValue::Integer(3));
}

#[test]
fn qwen_string_values_stay_raw_even_if_json_like() {
    let parser = QwenToolCallParser::new();
    let text = "<function=f>\n<parameter=note>\nhello \"world\"\n</parameter>\n</function>";
    let call = parser.parse(text, &tools(&["f"]), "id3").unwrap();
    assert_eq!(
        call.arguments.as_object().unwrap()["note"],
        JsonValue::String("hello \"world\"".to_string())
    );
}

#[test]
fn qwen_rejects_unknown_tool() {
    let parser = QwenToolCallParser::new();
    let text = "<function=evil>\n</function>";
    let err = parser.parse(text, &tools(&["good"]), "id").unwrap_err();
    assert_eq!(err, ToolCallParserError::UnknownTool("evil".to_string()));
}

#[test]
fn deepseek_parses_single_invoke() {
    let parser = DeepseekToolCallParser::new();
    let dsml = "\u{FF5C}DSML\u{FF5C}";
    let text = format!(
        "<{dsml}invoke name=\"get_weather\">\n<{dsml}parameter name=\"city\" string=\"true\">NYC</{dsml}parameter>\n<{dsml}parameter name=\"days\" string=\"false\">3</{dsml}parameter>\n</{dsml}invoke>"
    );
    let calls = parser
        .parse(&text, &tools(&["get_weather"]), || "id".to_string())
        .unwrap();
    assert_eq!(calls.len(), 1);
    let obj = calls[0].arguments.as_object().unwrap();
    assert_eq!(obj["city"], JsonValue::String("NYC".to_string()));
    assert_eq!(obj["days"], JsonValue::Integer(3));
}

#[test]
fn deepseek_parses_multiple_invokes() {
    let parser = DeepseekToolCallParser::new();
    let dsml = "\u{FF5C}DSML\u{FF5C}";
    let text = format!(
        "<{dsml}invoke name=\"a\">\n</{dsml}invoke>\n<{dsml}invoke name=\"b\">\n</{dsml}invoke>"
    );
    let mut n = 0;
    let calls = parser
        .parse(&text, &tools(&["a", "b"]), || {
            n += 1;
            format!("id{n}")
        })
        .unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].id, "id1");
    assert_eq!(calls[1].id, "id2");
}

#[test]
fn json_value_encoded_round_trips_through_parse() {
    let parsed = JsonValue::parse("{\"a\":1,\"b\":[true,null,\"x\"]}").unwrap();
    let text = parsed.encoded();
    let reparsed = JsonValue::parse(&text).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn gemma_schema_normalized_flattens_type_array() {
    let schema = JsonValue::parse("{\"type\":[\"string\",\"null\"]}").unwrap();
    let normalized = schema.gemma_schema_normalized();
    assert_eq!(
        normalized.as_object().unwrap()["type"],
        JsonValue::String("string".to_string())
    );
}

#[test]
fn gemma_schema_normalized_defaults_typeless_object_to_object_or_string() {
    let with_props = JsonValue::parse("{\"properties\":{}}").unwrap();
    let normalized = with_props.gemma_schema_normalized();
    assert_eq!(
        normalized.as_object().unwrap()["type"],
        JsonValue::String("object".to_string())
    );

    let bare = JsonValue::parse("{}").unwrap();
    let normalized = bare.gemma_schema_normalized();
    assert_eq!(
        normalized.as_object().unwrap()["type"],
        JsonValue::String("string".to_string())
    );
}

#[test]
fn gemma_tool_argument_body_sorts_keys() {
    let args = JsonValue::parse("{\"b\":1,\"a\":2}").unwrap();
    assert_eq!(args.gemma_tool_argument_body().unwrap(), "a:2,b:1");
}
