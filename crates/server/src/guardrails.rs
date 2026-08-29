//! Tool-call guardrails: rescue parsing, argument validation, and one retry.
//!
//! A local model gets tool calling wrong in two ways this module can fix
//! without the caller noticing. It emits a call in a dialect its own template
//! did not teach it -- bare JSON, Qwen's `<function=>` XML, Mistral's
//! `[TOOL_CALLS]` -- which the [`tokenizer::StructuredAssistantDecoder`]
//! correctly declines to parse, so the markup reaches the client as prose. Or
//! it emits a well-formed call whose ARGUMENTS violate the schema the caller
//! sent: a missing required field, a string where a number belongs, an
//! invented enum member.
//!
//! `forge-guardrails` already solves both, and this module is the adapter:
//! type conversions in one direction, [`inspect`] to reach a verdict, and
//! [`run_guarded`] to act on it.
//!
//! **THE RETRY RE-RENDERS THE PROMPT, it does not append tokens.** A nudge has
//! to arrive as a `user` turn after the failed `assistant` turn, through the
//! checkpoint's own chat template, or it lands inside whatever channel the
//! model was last writing in. Appending nudge tokens to the `prompt_ids` the
//! first generation used is cheaper and wrong on every dialect, which is why
//! [`run_guarded`] takes the whole `ChatCompletionRequest` rather than the
//! planned prompt.
//!
//! **The budget is ONE retry and that is a throughput decision** (crate Gotcha
//! 1): the real backend serializes requests behind a mutex, so a second
//! generation doubles the worst-case hold. A ladder of escalating nudges is
//! what `forge`'s own `step_nudge` offers and is deliberately not taken here.

use std::collections::{HashMap, HashSet};

use anyllm_translate::openai::{
    ChatCompletionRequest, ChatContent, ChatMessage, ChatRole, ChatToolChoice,
};
use forge_guardrails::{
    rescue_tool_call, retry_nudge, unknown_tool_nudge, validate_tool_arguments, ToolSpec,
};
use tokenizer::{JsonValue, ParsedToolCall, ReasoningEffort};

use crate::handler::{plan, run_full, tool_names, AppState, GenError, Generated};

/// Which guardrails run, and how many times a bad turn may be re-asked.
///
/// Process-level, resolved once at startup and read off [`crate::ChatModel`],
/// for the reason the rate cap is (crate Gotcha 10): there is one runner per
/// process, and whether a deployment repairs tool calls is a property of the
/// deployment rather than of a caller's prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuardrailConfig {
    /// Recover a tool call the decoder did not parse out of the raw text.
    pub rescue: bool,
    /// Check a parsed call's arguments against the schema the request sent.
    pub validate: bool,
    /// How many times a failed turn may be re-generated with a nudge.
    pub retries: u8,
}

impl GuardrailConfig {
    /// Everything off: the exact path this server took before guardrails
    /// existed.
    pub const OFF: Self = Self {
        rescue: false,
        validate: false,
        retries: 0,
    };
}

impl Default for GuardrailConfig {
    fn default() -> Self {
        Self {
            rescue: true,
            validate: true,
            retries: 1,
        }
    }
}

impl GuardrailConfig {
    /// Whether any guardrail would run at all. Used to keep a request that
    /// cannot be affected on its original streaming path, and read by the
    /// binary for its startup line -- hence `pub` rather than `pub(crate)`.
    pub fn active(&self) -> bool {
        self.rescue || self.validate
    }
}

/// What [`inspect`] made of one generation.
#[derive(Debug, PartialEq)]
pub(crate) enum Verdict {
    /// Nothing to do: either the turn is fine or no guardrail applies.
    Accept,
    /// Calls recovered from text the decoder did not parse.
    Rescued(Vec<ParsedToolCall>),
    /// The turn is wrong in a way a re-ask might fix. Carries the nudge.
    Retry(String),
}

/// The tool schemas this request offered, keyed by name.
///
/// Built from the SAME `tool_choice` filtering [`tool_names`] applies, so a
/// request pinning one function cannot be validated against a schema it
/// suppressed. A tool whose `parameters` are absent or unparseable is skipped
/// rather than refused: the request was already accepted and generated
/// against, and refusing here would turn a schema this server merely cannot
/// CHECK into a failed request.
pub(crate) fn tool_specs(request: &ChatCompletionRequest) -> HashMap<String, ToolSpec> {
    let offered = tool_names(request);
    let mut specs = HashMap::new();
    for tool in request.tools.iter().flatten() {
        let name = &tool.function.name;
        if !offered.contains(name) {
            continue;
        }
        let Some(parameters) = tool.function.parameters.as_ref() else {
            continue;
        };
        let description = tool.function.description.clone().unwrap_or_default();
        if let Ok(spec) = ToolSpec::from_json_schema(name.clone(), description, parameters) {
            specs.insert(name.clone(), spec);
        }
    }
    specs
}

/// This port's parsed call as the type the guardrails speak.
///
/// Goes through `arguments_json` rather than through `JsonValue`, because
/// that string is what the decoder actually read off the wire and `serde_json`
/// deserializes it into forge's `IndexMap` directly. A call whose arguments
/// are not a JSON object yields an empty map, which the validator then reports
/// as every required field missing -- the right answer, and a diagnosis rather
/// than a panic.
fn to_forge_call(call: &ParsedToolCall) -> forge_guardrails::ToolCall {
    let args = serde_json::from_str(&call.arguments_json).unwrap_or_default();
    forge_guardrails::ToolCall::new(call.name.clone(), args).with_id(call.id.clone())
}

/// The inverse, for a call recovered by [`rescue_tool_call`].
///
/// Ids follow `stream_blocking`'s `toolu_N` shape so a rescued call is
/// indistinguishable on the wire from a parsed one.
fn from_forge_call(call: &forge_guardrails::ToolCall, index: usize) -> Option<ParsedToolCall> {
    let arguments_json = serde_json::to_string(&call.args).ok()?;
    let arguments = JsonValue::parse(&arguments_json).ok()?;
    Some(ParsedToolCall {
        id: call
            .id
            .clone()
            .unwrap_or_else(|| format!("toolu_rescued_{index}")),
        name: call.tool.clone(),
        arguments,
        arguments_json,
    })
}

/// Whether this request stated that prose is not an acceptable answer.
///
/// OpenAI spells it two ways and both mean it: `"required"`, and a named
/// function (which also pins WHICH tool). `"auto"`, `"none"` and an absent
/// field all leave the model free to answer directly.
fn requires_tool_call(request: &ChatCompletionRequest) -> bool {
    match &request.tool_choice {
        Some(ChatToolChoice::Simple(choice)) => choice == "required",
        Some(ChatToolChoice::Named(_)) => true,
        None => false,
    }
}

/// Read one generation against the schemas the request sent.
///
/// Pure: no I/O, no model, no clock. Every branch is unit-testable, which is
/// the whole reason the decision is split out of [`run_guarded`].
pub(crate) fn inspect(
    generated: &Generated,
    specs: &HashMap<String, ToolSpec>,
    offered: &HashSet<String>,
    requires_call: bool,
    config: &GuardrailConfig,
) -> Verdict {
    // A request that offered no tools cannot have a tool-call problem, and
    // this is the guard that keeps ordinary chat traffic on its original path.
    if offered.is_empty() {
        return Verdict::Accept;
    }

    let names: Vec<&str> = offered.iter().map(String::as_str).collect();

    // **RESCUE FIRST, THEN VALIDATE THE RESULT, and the composition is the
    // point rather than an ordering detail.** Returning a rescued call
    // unchecked was the first version of this function and it is exactly
    // backwards: a call recovered from markup the decoder could not parse is
    // the one MOST likely to have arguments the model also got wrong, so it
    // needs the schema check more than a cleanly parsed call does, not less.
    // Caught by `an_invalid_call_is_retried_and_the_second_answer_wins`, which
    // saw a call with empty arguments rescued straight onto the wire.
    let rescued = if generated.calls.is_empty() && config.rescue {
        rescue_tool_call(&generated.text, &names)
            .iter()
            .enumerate()
            .filter_map(|(i, c)| from_forge_call(c, i))
            .collect()
    } else {
        Vec::new()
    };
    let calls: &[ParsedToolCall] = if rescued.is_empty() {
        &generated.calls
    } else {
        &rescued
    };

    if calls.is_empty() {
        // NOTHING TO RESCUE IS ONLY A FAILURE IF THE REQUEST DEMANDED A CALL.
        // Under `auto` the model may legitimately have chosen to answer in
        // prose, and nudging that turns a good turn into a wasted generation;
        // under `required` or a named function the caller stated that prose is
        // not an acceptable answer, and this is the one place a re-ask is
        // clearly right. `forge`'s own `retry_nudge` is the wording.
        return if requires_call {
            Verdict::Retry(retry_nudge(&generated.text))
        } else {
            Verdict::Accept
        };
    }

    let mut messages = Vec::new();
    if config.validate {
        for call in calls {
            let Some(spec) = specs.get(&call.name) else {
                // Reachable for a call with no schema to check against: a tool
                // offered without `parameters`, or one whose schema this server
                // could not parse. Not a hard error -- it was already generated
                // against. A name that is not offered AT ALL is, though.
                if !offered.contains(&call.name) {
                    messages.push(unknown_tool_nudge(&call.name, &names));
                }
                continue;
            };
            let errors = validate_tool_arguments(&to_forge_call(call), spec);
            if !errors.is_empty() {
                messages.push(format!(
                    "Your call to `{}` had invalid arguments: {}.",
                    call.name,
                    errors
                        .iter()
                        .map(|e| e.message())
                        .collect::<Vec<_>>()
                        .join("; ")
                ));
            }
        }
    }

    // A rescued call that PASSES validation is the win; one that fails is
    // re-asked, and the rescue is discarded with it -- sending a client a
    // recovered call known to be malformed would be worse than the prose it
    // was recovered from.
    if messages.is_empty() {
        return if rescued.is_empty() {
            Verdict::Accept
        } else {
            Verdict::Rescued(rescued)
        };
    }

    Verdict::Retry(format!(
        "{} {}",
        messages.join(" "),
        "Call the tool again with corrected arguments."
    ))
}

/// The failed turn plus a nudge, as two more messages on a cloned request.
///
/// The assistant turn carries the text the model produced so it can see what
/// it got wrong; the nudge is a `user` turn because that is the only role
/// every checkpoint's template renders as an instruction to act on.
fn with_retry_turn(
    request: &ChatCompletionRequest,
    said: &str,
    nudge: &str,
) -> ChatCompletionRequest {
    let mut retried = request.clone();
    let message = |role: ChatRole, text: &str| ChatMessage {
        role,
        content: Some(ChatContent::Text(text.to_string())),
        name: None,
        tool_calls: None,
        tool_call_id: None,
        refusal: None,
        reasoning_content: None,
        thinking_blocks: None,
    };
    if !said.trim().is_empty() {
        retried.messages.push(message(ChatRole::Assistant, said));
    }
    retried.messages.push(message(ChatRole::User, nudge));
    retried
}

/// Generate, inspect, and re-ask once if the turn is repairable.
///
/// Falls back to [`run_full`]'s single generation whenever no guardrail
/// applies, so a request with no tools -- or a server started with
/// `--guardrails off` -- reaches exactly the code it always did.
pub(crate) async fn run_guarded(
    model: AppState,
    request: &ChatCompletionRequest,
    effort: ReasoningEffort,
) -> Result<Generated, GenError> {
    let config = model.guardrails();
    let offered = tool_names(request);
    let requires_call = requires_tool_call(request);
    let specs = if config.validate {
        tool_specs(request)
    } else {
        HashMap::new()
    };

    let mut attempt = request.clone();
    let mut budget = config.retries;
    loop {
        // Re-planned every round: the retry turn changes the messages, so the
        // prompt has to be re-rendered through the template rather than
        // patched. `plan` already succeeded on the caller's own request, so a
        // failure here belongs to the nudge turn and is reported as one.
        let planned =
            plan(&model, &attempt).map_err(|e| GenError::Join(format!("guardrail replan: {e}")))?;
        let generated = run_full(
            model.clone(),
            planned.prompt_ids,
            planned.config,
            planned.images,
            offered.clone(),
            effort,
        )
        .await?;

        if !config.active() {
            return Ok(generated);
        }

        match inspect(&generated, &specs, &offered, requires_call, &config) {
            Verdict::Accept => return Ok(generated),
            Verdict::Rescued(calls) => {
                // The raw text IS the markup the call was recovered from, so
                // it is dropped rather than sent as content. Leaking a bare
                // JSON blob or an `<function=>` tag to a client as prose is
                // the failure this whole path exists to prevent, and OpenAI's
                // own shape puts `content: null` beside `tool_calls`.
                return Ok(Generated {
                    text: String::new(),
                    calls,
                    ..generated
                });
            }
            Verdict::Retry(nudge) => {
                // Budget spent: return the last turn as it stands. A request
                // that the model could not get right is still a completed
                // generation, and failing it would be a worse answer than an
                // imperfect one.
                if budget == 0 {
                    return Ok(generated);
                }
                budget -= 1;
                attempt = with_retry_turn(&attempt, &generated.text, &nudge);
            }
        }
    }
}

#[cfg(test)]
mod tests;
