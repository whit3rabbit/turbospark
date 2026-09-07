//! Unit tests for the guardrail verdict, which is pure: no model, no I/O.
//!
//! Everything here drives [`inspect`] directly. The end-to-end path -- that a
//! rescued call reaches the wire as `tool_calls`, that a no-tools request
//! still streams -- is `tests/guardrails.rs`, which needs the scripted backend.

use super::*;
use runtime::{RawDecodeResult, StopReason};

/// A request offering one tool with a required `city` string.
fn weather_request(tool_choice: Option<&str>) -> ChatCompletionRequest {
    let body = serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": "weather in Oslo?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"}
                    },
                    "required": ["city"]
                }
            }
        }],
        "tool_choice": tool_choice.unwrap_or("auto"),
    });
    serde_json::from_value(body).expect("request fixture parses")
}

fn generated(text: &str, calls: Vec<ParsedToolCall>) -> Generated {
    Generated {
        text: text.to_string(),
        reasoning: String::new(),
        calls,
        decode: RawDecodeResult {
            reused_prefix_tokens: 0,
            session_slot_evicted: false,
            reason: StopReason::EndOfTurn,
            prompt_tokens: 1,
            new_tokens: 1,
            prefill_seconds: 0.0,
            decode_seconds: 0.0,
            kv_position: 0,
            kv_backed_token_ids: Vec::new(),
            peak_memory_pressure: runtime::MemoryPressure::Normal,
        },
    }
}

fn call(name: &str, arguments_json: &str) -> ParsedToolCall {
    ParsedToolCall {
        id: "toolu_0".to_string(),
        name: name.to_string(),
        arguments: JsonValue::parse(arguments_json).expect("fixture arguments parse"),
        arguments_json: arguments_json.to_string(),
    }
}

fn verdict_for(request: &ChatCompletionRequest, gen: &Generated) -> Verdict {
    inspect(
        gen,
        &tool_specs(request),
        &tool_names(request),
        requires_tool_call(request),
        &GuardrailConfig::default(),
    )
}

#[test]
fn a_well_formed_call_is_accepted() {
    let request = weather_request(None);
    let gen = generated("", vec![call("get_weather", r#"{"city":"Oslo"}"#)]);
    assert_eq!(verdict_for(&request, &gen), Verdict::Accept);
}

#[test]
fn a_request_with_no_tools_is_never_inspected() {
    // The guard that keeps ordinary chat traffic on its original path. The
    // text here IS a rescuable call, so only the empty offer set can be what
    // returns Accept.
    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
    }))
    .expect("request parses");
    let gen = generated(
        r#"{"name":"get_weather","arguments":{"city":"Oslo"}}"#,
        vec![],
    );
    assert_eq!(verdict_for(&request, &gen), Verdict::Accept);
}

#[test]
fn a_bare_json_call_the_decoder_missed_is_rescued() {
    let request = weather_request(None);
    let gen = generated(
        r#"Sure, let me check. {"name": "get_weather", "arguments": {"city": "Oslo"}}"#,
        vec![],
    );
    let Verdict::Rescued(calls) = verdict_for(&request, &gen) else {
        panic!("expected a rescue, got {:?}", verdict_for(&request, &gen));
    };
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert!(calls[0].arguments_json.contains("Oslo"));
}

#[test]
fn prose_with_no_call_is_accepted_under_auto() {
    // The model choosing to answer is a legitimate turn. Nudging it would
    // spend a whole generation to make a good answer worse.
    let request = weather_request(None);
    let gen = generated("It is currently 4 degrees and raining in Oslo.", vec![]);
    assert_eq!(verdict_for(&request, &gen), Verdict::Accept);
}

#[test]
fn prose_with_no_call_is_retried_when_the_request_required_one() {
    let request = weather_request(Some("required"));
    let gen = generated("It is currently 4 degrees and raining in Oslo.", vec![]);
    let Verdict::Retry(nudge) = verdict_for(&request, &gen) else {
        panic!("expected a retry under tool_choice=required");
    };
    assert!(nudge.contains("tool call"), "nudge was {nudge:?}");
}

#[test]
fn a_missing_required_argument_is_retried_and_the_nudge_names_it() {
    let request = weather_request(None);
    let gen = generated("", vec![call("get_weather", r#"{"days":3}"#)]);
    let Verdict::Retry(nudge) = verdict_for(&request, &gen) else {
        panic!("expected a retry for a missing required argument");
    };
    // The nudge has to name the tool AND the field, or the model cannot act
    // on it -- that is the whole difference between a retry and a re-roll.
    assert!(nudge.contains("get_weather"), "nudge was {nudge:?}");
    assert!(nudge.contains("city"), "nudge was {nudge:?}");
}

#[test]
fn a_wrongly_typed_argument_is_retried() {
    let request = weather_request(None);
    let gen = generated(
        "",
        vec![call("get_weather", r#"{"city":"Oslo","days":"three"}"#)],
    );
    let Verdict::Retry(nudge) = verdict_for(&request, &gen) else {
        panic!("expected a retry for a string where an integer belongs");
    };
    assert!(nudge.contains("days"), "nudge was {nudge:?}");
}

#[test]
fn arguments_that_are_not_an_object_read_as_missing_rather_than_panicking() {
    // `arguments_json` comes off the wire and is not trusted to be an object.
    let request = weather_request(None);
    let gen = generated("", vec![call("get_weather", "[1, 2, 3]")]);
    assert!(matches!(verdict_for(&request, &gen), Verdict::Retry(_)));
}

#[test]
fn the_off_config_accepts_everything() {
    let request = weather_request(None);
    let bad = generated("", vec![call("get_weather", r#"{"days":3}"#)]);
    let rescuable = generated(
        r#"{"name":"get_weather","arguments":{"city":"Oslo"}}"#,
        vec![],
    );
    for gen in [&bad, &rescuable] {
        assert_eq!(
            inspect(
                gen,
                &tool_specs(&request),
                &tool_names(&request),
                requires_tool_call(&request),
                &GuardrailConfig::OFF,
            ),
            Verdict::Accept
        );
    }
}

#[test]
fn rescue_and_validate_are_independent_switches() {
    let request = weather_request(None);
    let rescuable = generated(
        r#"{"name":"get_weather","arguments":{"city":"Oslo"}}"#,
        vec![],
    );
    let invalid = generated("", vec![call("get_weather", r#"{"days":3}"#)]);
    let validate_only = GuardrailConfig {
        rescue: false,
        ..GuardrailConfig::default()
    };
    let rescue_only = GuardrailConfig {
        validate: false,
        ..GuardrailConfig::default()
    };
    let run = |gen, config| {
        inspect(
            gen,
            &tool_specs(&request),
            &tool_names(&request),
            requires_tool_call(&request),
            &config,
        )
    };
    assert_eq!(run(&rescuable, validate_only), Verdict::Accept);
    assert!(matches!(run(&invalid, validate_only), Verdict::Retry(_)));
    assert!(matches!(run(&rescuable, rescue_only), Verdict::Rescued(_)));
    assert_eq!(run(&invalid, rescue_only), Verdict::Accept);
}

#[test]
fn tool_choice_none_suppresses_every_guardrail() {
    // `tool_names` returns empty for "none", so a call must not be rescued
    // into a request that asked for no tools at all.
    let request = weather_request(Some("none"));
    let gen = generated(
        r#"{"name":"get_weather","arguments":{"city":"Oslo"}}"#,
        vec![],
    );
    assert!(tool_names(&request).is_empty());
    assert_eq!(verdict_for(&request, &gen), Verdict::Accept);
}

#[test]
fn the_retry_turn_appends_an_assistant_turn_and_a_user_nudge() {
    let request = weather_request(None);
    let before = request.messages.len();
    let retried = with_retry_turn(&request, "some bad output", &[], "fix it");
    assert_eq!(retried.messages.len(), before + 2);
    assert!(matches!(retried.messages[before].role, ChatRole::Assistant));
    assert!(matches!(retried.messages[before + 1].role, ChatRole::User));
    // The nudge must be the LAST turn, or the template renders the model's
    // own failed output as the thing to respond to.
    assert_eq!(
        retried.messages[before + 1].effective_text().as_deref(),
        Some("fix it")
    );
}

#[test]
fn an_empty_assistant_turn_is_not_appended() {
    // A model that emitted nothing has nothing to show itself, and an empty
    // assistant turn renders as stray markup on several templates.
    let request = weather_request(None);
    let before = request.messages.len();
    let retried = with_retry_turn(&request, "   ", &[], "fix it");
    assert_eq!(retried.messages.len(), before + 1);
    assert!(matches!(retried.messages[before].role, ChatRole::User));
}

/// F19: a parsed call whose ARGUMENTS failed validation is the common case
/// `Verdict::Retry` fires for, and such a call's surrounding text is
/// typically empty (the whole reply was the call markup). Before this, an
/// empty `said` with a non-empty `calls` appended NOTHING -- the retry's
/// nudge followed the caller's own prior turn with no record at all of what
/// the model had just tried.
#[test]
fn a_tool_call_with_no_surrounding_text_still_appends_an_assistant_turn() {
    let request = weather_request(None);
    let before = request.messages.len();
    let call = tokenizer::ParsedToolCall {
        id: "toolu_0".to_string(),
        name: "get_weather".to_string(),
        arguments: tokenizer::JsonValue::parse(r#"{"city": "Oslo"}"#).unwrap(),
        arguments_json: r#"{"city": "Oslo"}"#.to_string(),
    };
    let retried = with_retry_turn(&request, "", std::slice::from_ref(&call), "fix it");
    assert_eq!(retried.messages.len(), before + 2, "{:?}", retried.messages);
    assert!(matches!(retried.messages[before].role, ChatRole::Assistant));
    let tool_calls = retried.messages[before]
        .tool_calls
        .as_ref()
        .expect("the assistant turn must carry the attempted call");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].function.name, "get_weather");
}

#[test]
fn requires_tool_call_reads_both_spellings() {
    assert!(!requires_tool_call(&weather_request(None)));
    assert!(!requires_tool_call(&weather_request(Some("auto"))));
    assert!(!requires_tool_call(&weather_request(Some("none"))));
    assert!(requires_tool_call(&weather_request(Some("required"))));

    let named: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    }))
    .expect("named tool_choice parses");
    assert!(requires_tool_call(&named));
}

#[test]
fn a_tool_offered_without_a_schema_is_skipped_rather_than_refused() {
    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "ping"}}],
    }))
    .expect("schema-less tool parses");
    assert!(tool_specs(&request).is_empty());
    // And a call to it is accepted rather than nudged: this server cannot
    // check a schema it was never sent.
    let gen = generated("", vec![call("ping", "{}")]);
    assert_eq!(verdict_for(&request, &gen), Verdict::Accept);
}

// ---------------------------------------------------------------------------
// The extra rescue formats (`guardrails/extra_formats.rs`): GLM, the
// invoke/parameter shape, Kimi K2, and Longcat. Every fixture below is the
// markup the family's own published chat template teaches, hand-built here
// because none of the four is installed on this machine -- these are
// RESCUE-tier formats, tested at the only place they run: [`inspect`].
// ---------------------------------------------------------------------------

fn object_field<'a>(value: &'a JsonValue, key: &str) -> Option<&'a JsonValue> {
    match value {
        JsonValue::Object(map) => map.get(key),
        _ => None,
    }
}

/// The same GLM call with the wrapper STRIPPED -- the shape that actually
/// reaches the rescue when a GLM vocabulary loads under a dialect this
/// engine knows: the `<tool_call>` pair is special, the detokenizer
/// renders it to nothing, the native decoder fails the span, and the
/// released body is all that is left. The pair markup is what this
/// strategy keys on, so the rescue still fires.
#[test]
fn a_glm_body_with_the_wrapper_stripped_is_still_rescued() {
    let request = weather_request(None);
    let gen = generated(
        "get_weather\n<arg_key>city</arg_key>\n<arg_value>Oslo</arg_value>\n",
        vec![],
    );
    let Verdict::Rescued(calls) = verdict_for(&request, &gen) else {
        panic!(
            "expected a wrapperless GLM rescue, got {:?}",
            verdict_for(&request, &gen)
        );
    };
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(
        object_field(&calls[0].arguments, "city"),
        Some(&JsonValue::String("Oslo".to_string()))
    );
}

/// GLM-4.6's shape: the function name inline after the opening tag, then
/// flat `arg_key`/`arg_value` pairs. Values coerce the way the native Qwen
/// parser's do, so `3` arrives as an integer and `Oslo` as a string.
#[test]
fn a_glm_call_with_arg_key_pairs_is_rescued() {
    let request = weather_request(None);
    let gen = generated(
        "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Oslo</arg_value>\n\
         <arg_key>days</arg_key>\n<arg_value>3</arg_value>\n</tool_call>",
        vec![],
    );
    let Verdict::Rescued(calls) = verdict_for(&request, &gen) else {
        panic!(
            "expected a GLM rescue, got {:?}",
            verdict_for(&request, &gen)
        );
    };
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(
        object_field(&calls[0].arguments, "city"),
        Some(&JsonValue::String("Oslo".to_string()))
    );
    assert_eq!(
        object_field(&calls[0].arguments, "days"),
        Some(&JsonValue::Integer(3))
    );
}

/// Multiple GLM blocks are multiple calls, each rescued with its own
/// arguments.
#[test]
fn two_glm_blocks_are_two_calls() {
    let request = weather_request(None);
    let gen = generated(
        "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Oslo</arg_value>\n</tool_call>\n\
         <tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Cairo</arg_value>\n</tool_call>",
        vec![],
    );
    let Verdict::Rescued(calls) = verdict_for(&request, &gen) else {
        panic!(
            "expected two rescues, got {:?}",
            verdict_for(&request, &gen)
        );
    };
    assert_eq!(calls.len(), 2);
    assert!(calls[0].arguments_json.contains("Oslo"));
    assert!(calls[1].arguments_json.contains("Cairo"));
}

/// MiniMax-M2's shape: the Anthropic/Claude invoke form, here inside its
/// `<minimax:tool_call>` wrapper. The wrapper is not what the strategy
/// matches -- the invoke blocks are -- so this also covers a model that
/// drops its own wrapper.
#[test]
fn a_minimax_invoke_block_is_rescued() {
    let request = weather_request(None);
    let gen = generated(
        "<minimax:tool_call>\n<invoke name=\"get_weather\">\n\
         <parameter name=\"city\">Oslo</parameter>\n</invoke>\n</minimax:tool_call>",
        vec![],
    );
    let Verdict::Rescued(calls) = verdict_for(&request, &gen) else {
        panic!(
            "expected an invoke rescue, got {:?}",
            verdict_for(&request, &gen)
        );
    };
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(
        object_field(&calls[0].arguments, "city"),
        Some(&JsonValue::String("Oslo".to_string()))
    );
}

/// The same invoke shape with no wrapper at all, which is the generic
/// half of the strategy's claim.
#[test]
fn an_unwrapped_invoke_block_is_rescued_too() {
    let request = weather_request(None);
    let gen = generated(
        "<invoke name=\"get_weather\"><parameter name=\"city\">Oslo</parameter></invoke>",
        vec![],
    );
    assert!(matches!(verdict_for(&request, &gen), Verdict::Rescued(_)));
}

/// A model that pretty-prints its invoke block puts the value on its own
/// line, indented: `<parameter name="city">\nOslo\n</parameter>`. The
/// recovered value must be trimmed to `"Oslo"`, or an enum/pattern check in
/// the request's own schema fails a call whose only fault is whitespace the
/// model added for readability.
#[test]
fn a_parameter_value_with_surrounding_newlines_is_trimmed() {
    let request = weather_request(None);
    let gen = generated(
        "<invoke name=\"get_weather\">\n<parameter name=\"city\">\n  Oslo\n</parameter>\n</invoke>",
        vec![],
    );
    let Verdict::Rescued(calls) = verdict_for(&request, &gen) else {
        panic!(
            "expected an invoke rescue, got {:?}",
            verdict_for(&request, &gen)
        );
    };
    assert_eq!(
        object_field(&calls[0].arguments, "city"),
        Some(&JsonValue::String("Oslo".to_string()))
    );
}

/// Kimi K2's shape: no name tag anywhere, only the id
/// `functions.NAME:IDX` Moonshot's own guidance documents. The name is
/// recovered from the id; the arguments are the JSON body.
#[test]
fn a_kimi_k2_call_is_rescued_with_the_name_recovered_from_its_id() {
    let request = weather_request(None);
    let gen = generated(
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0\
         <|tool_call_argument_begin|>{\"city\": \"Oslo\", \"days\": 3}<|tool_call_end|>\
         <|tool_calls_section_end|>",
        vec![],
    );
    let Verdict::Rescued(calls) = verdict_for(&request, &gen) else {
        panic!(
            "expected a Kimi rescue, got {:?}",
            verdict_for(&request, &gen)
        );
    };
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(
        object_field(&calls[0].arguments, "days"),
        Some(&JsonValue::Integer(3))
    );
}

/// Moonshot's documented anomaly: the model sometimes emits an opaque
/// `call_...` id instead of `functions.NAME:IDX`. There is no name in such
/// an id, so no call is rescued -- which under `auto` is prose, not a
/// retry. Inventing a name there would be worse than the refusal.
#[test]
fn a_kimi_anomaly_id_rescues_nothing() {
    let request = weather_request(None);
    let gen = generated(
        "<|tool_calls_section_begin|><|tool_call_begin|>call_59adf5614cfe4f4b8a71be54\
         <|tool_call_argument_begin|>{\"city\": \"Oslo\"}<|tool_call_end|>\
         <|tool_calls_section_end|>",
        vec![],
    );
    assert_eq!(verdict_for(&request, &gen), Verdict::Accept);
}

/// Longcat needs NO new code: its `{"name", "arguments"}` JSON inside the
/// `<longcat_tool_call>` wrapper is exactly what forge's JSON scan reads,
/// because that scan finds balanced braces anywhere in the text. This test
/// pins the zero-code claim so it stays measured rather than assumed.
#[test]
fn a_longcat_call_is_rescued_by_forges_existing_json_scan() {
    let request = weather_request(None);
    let gen = generated(
        "<longcat_tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}\n</longcat_tool_call>",
        vec![],
    );
    let Verdict::Rescued(calls) = verdict_for(&request, &gen) else {
        panic!(
            "expected a Longcat rescue, got {:?}",
            verdict_for(&request, &gen)
        );
    };
    assert_eq!(calls[0].name, "get_weather");
    assert!(calls[0].arguments_json.contains("Oslo"));
}

/// A rescued GLM call still goes through the schema check, and a call with
/// a missing required argument is re-asked rather than sent: the rescue
/// tier feeds the SAME validation the native tier gets.
#[test]
fn a_rescued_glm_call_with_a_missing_required_field_is_retried() {
    let request = weather_request(None);
    let gen = generated(
        "<tool_call>get_weather\n<arg_key>days</arg_key>\n<arg_value>3</arg_value>\n</tool_call>",
        vec![],
    );
    let Verdict::Retry(nudge) = verdict_for(&request, &gen) else {
        panic!("expected a retry, got {:?}", verdict_for(&request, &gen));
    };
    assert!(nudge.contains("city"), "nudge was {nudge:?}");
}

/// Markup naming a tool the request never offered rescues nothing, in this
/// tier exactly as in forge's: allowlist membership is the one gate every
/// strategy shares.
#[test]
fn glm_markup_for_an_unoffered_tool_is_not_rescued() {
    let request = weather_request(None);
    let gen = generated(
        "<tool_call>delete_everything\n<arg_key>city</arg_key>\n<arg_value>Oslo</arg_value>\n</tool_call>",
        vec![],
    );
    assert_eq!(verdict_for(&request, &gen), Verdict::Accept);
}

/// Gemma DSL shape: `call:NAME{key:value,...}`.
#[test]
fn a_gemma_call_is_rescued() {
    let request = weather_request(None);
    let gen = generated("call:get_weather{city: \"Oslo\", days: 3}", vec![]);
    let Verdict::Rescued(calls) = verdict_for(&request, &gen) else {
        panic!(
            "expected a Gemma rescue, got {:?}",
            verdict_for(&request, &gen)
        );
    };
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(
        object_field(&calls[0].arguments, "city"),
        Some(&JsonValue::String("Oslo".to_string()))
    );
    assert_eq!(
        object_field(&calls[0].arguments, "days"),
        Some(&JsonValue::Integer(3))
    );
}

/// Qwen ChatML parameter XML: `<function=NAME><parameter=KEY>VALUE</parameter>...</function>`.
#[test]
fn a_qwen_xml_parameter_call_is_rescued() {
    let request = weather_request(None);
    let gen = generated(
        "<function=get_weather>\n<parameter=city>Oslo</parameter>\n<parameter=days>3</parameter>\n</function>",
        vec![],
    );
    let Verdict::Rescued(calls) = verdict_for(&request, &gen) else {
        panic!(
            "expected a Qwen XML rescue, got {:?}",
            verdict_for(&request, &gen)
        );
    };
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(
        object_field(&calls[0].arguments, "city"),
        Some(&JsonValue::String("Oslo".to_string()))
    );
    assert_eq!(
        object_field(&calls[0].arguments, "days"),
        Some(&JsonValue::Integer(3))
    );
}
