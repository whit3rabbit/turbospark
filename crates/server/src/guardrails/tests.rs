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
    let retried = with_retry_turn(&request, "some bad output", "fix it");
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
    let retried = with_retry_turn(&request, "   ", "fix it");
    assert_eq!(retried.messages.len(), before + 1);
    assert!(matches!(retried.messages[before].role, ChatRole::User));
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
