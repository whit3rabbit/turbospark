//! Rescue strategies for tool-call dialects [`forge_guardrails`] does not
//! know, tried BEFORE its own strategies.
//!
//! Four families emit calls in markup that neither this engine's native
//! per-dialect parsers nor forge's four strategies (bare JSON, rehearsal,
//! Qwen XML, Mistral bracket) match. Grammars verified 2026-09-06 against
//! each checkpoint's published `chat_template.jinja`, not against the
//! compressed table OLMX keeps (which also mislabels Gemma's real format --
//! this repo's `GemmaToolCallParser` is checked against the real installed
//! template and is correct):
//!
//! - **GLM** (GLM-4.6): a function name followed by flat
//!   `<arg_key>K</arg_key>` / `<arg_value>V</arg_value>` pairs. One run of
//!   pairs per call; values are JSON-encoded unless they are plain strings.
//!   The strategy keys on the pair markup rather than on the
//!   `<tool_call>` wrapper the template also teaches, because that wrapper
//!   does not reliably survive to the rescue: as ordinary text it is
//!   noise, and as a checkpoint's special token it is gone from the
//!   decoded text entirely (see `parse_glm`).
//! - **MiniMax** (MiniMax-M2): `<minimax:tool_call>` wrapping one or more
//!   `<invoke name="NAME"><parameter name="K">V</parameter>...</invoke>`
//!   blocks. This is the Anthropic/Claude invoke shape with a wrapper, so
//!   the strategy is generic and keys on neither wrapper nor family.
//! - **Kimi K2** (Kimi-K2-Instruct):
//!   `<|tool_call_begin|>functions.NAME:IDX<|tool_call_argument_begin|>
//!   {json}<|tool_call_end|>` -- there is NO name tag; the name is recovered
//!   from the id, whose format Moonshot documents as
//!   `functions.{func_name}:{idx}` (their `tool_call_guidance.md`, and what
//!   vLLM's `kimi_tool_parser` implements). The strategy is anchored on
//!   the section tags, which is the honest scope today: no family this
//!   engine loads carries Kimi's special tokens, so its markup can only
//!   arrive as literal text. The day a Kimi checkpoint loads under a known
//!   dialect, the detokenizer will strip those tags the way GLM's wrapper
//!   is stripped, and this strategy will need the same released-body
//!   treatment. A model emitting its documented anomaly id (a bare
//!   `call_...` string) yields a name the allowlist will refuse, which is
//!   the honest answer -- an opaque id states no name.
//! - **Longcat** (LongCat-Flash): `<longcat_tool_call>` wrapping a plain
//!   `{"name": ..., "arguments": {...}}` JSON object. Deliberately NOT
//!   implemented here: forge's JSON scan reads balanced braces anywhere in
//!   the text and accepts exactly that shape, so the wrapper tag is
//!   irrelevant to it. `guardrails/tests.rs` pins that with a fixture so the
//!   zero-code claim stays true instead of assumed.
//!
//! **THIS IS THE RESCUE TIER, AND IT IS DELIBERATELY NOT A NATIVE
//! DIALECT.** The repo's convention (Gemma, Qwen, DeepSeek, Harmony) is to
//! build a `ChatDialect` plus streaming decoder only against a real
//! installed checkpoint that can be smoke-tested; none of these four is
//! installed here. Regex rescue needs no dialect, no tokenizer, and no
//! checkpoint: it runs on the decoded text. It inherits the same limits as
//! Mistral's `[TOOL_CALLS]` coverage, the other rescue-tier entry: it only
//! fires on a COMPLETED turn (the buffered path), it cannot feed streaming
//! consumers, and it has never been validated against a real install.
//!
//! **THE ALLOWLIST IS LOAD BEARING HERE AS IN FORGE.** Every strategy
//! refuses a call whose name is not among the tools the request offered, so
//! recovered markup cannot mint a call to a tool the caller never granted.

use std::collections::BTreeMap;

use regex_lite::Regex;
use tokenizer::{JsonValue, ParsedToolCall};

/// One call recovered by this module, ready for the wire.
///
/// Ids follow the `toolu_rescued_N` shape [`super::inspect`]'s forge path
/// already uses, so a rescued call is indistinguishable on the wire no
/// matter which pass recovered it.
pub(super) fn rescue(text: &str, available_tools: &[&str]) -> Vec<ParsedToolCall> {
    let built = parse_glm(text, available_tools)
        .or_else(|| parse_invoke_parameters(text, available_tools))
        .or_else(|| parse_qwen_xml(text, available_tools))
        .or_else(|| parse_gemma(text, available_tools))
        .or_else(|| parse_kimi(text, available_tools));
    built
        .unwrap_or_default()
        .into_iter()
        .enumerate()
        .map(|(i, call)| {
            let arguments = JsonValue::Object(call.arguments);
            ParsedToolCall {
                id: format!("toolu_rescued_{i}"),
                name: call.name,
                arguments_json: arguments.encoded(),
                arguments,
            }
        })
        .collect()
}

/// A name plus its arguments as a map, the intermediate every strategy
/// builds before [`rescue`] turns it into a [`ParsedToolCall`].
struct RawCall {
    name: String,
    arguments: BTreeMap<String, JsonValue>,
}

/// The captured groups a block or pair regex yields, as plain strings.
///
/// `regex-lite` exposes captures through `Captures::get`, which answers an
/// `Option` per group; every pattern here owns both of its groups, so a
/// missing one collapses to the empty string rather than a skip.
struct Groups<'a> {
    first: &'a str,
    second: &'a str,
}

fn groups<'a>(caps: &regex_lite::Captures<'a>) -> Groups<'a> {
    let at = |i: usize| caps.get(i).map(|m| m.as_str()).unwrap_or_default();
    Groups {
        first: at(1),
        second: at(2),
    }
}

/// GLM: a function name directly followed by flat `<arg_key>K</arg_key>` /
/// `<arg_value>V</arg_value>` pairs, one call per run of pairs.
///
/// **THE `<tool_call>` WRAPPER THE TEMPLATE TEACHES IS DELIBERATELY NOT
/// PART OF THE PATTERN, and the two places this strategy runs strip or
/// keep it for different reasons.** Where the wrapper is ordinary text it
/// is skipped as noise. But a GLM vocabulary this engine loads under a
/// known dialect carries it as a SPECIAL token: the detokenizer renders
/// every special token to the empty string (AGENTS.md Gotcha 44), the
/// native decoder buffers the interior as a tool span, the dialect's own
/// parser refuses it, and what the rescue receives is the released span
/// body alone -- the name plus the pairs, wrapper gone in both directions.
/// The arg-pair markup is the signature that survives both routes, and the
/// allowlist is what stops a stray word preceding `<arg_key>` from
/// becoming a name. A call with no pairs at all matches nothing here;
/// forge's strategies find nothing either, and the turn stands as prose.
fn parse_glm(text: &str, available_tools: &[&str]) -> Option<Vec<RawCall>> {
    let call = Regex::new(
        r"([A-Za-z0-9_\-]{1,64})\s*((?:<arg_key>[\s\S]*?</arg_key>\s*<arg_value>[\s\S]*?</arg_value>\s*)+)",
    )
    .ok()?;
    let pair =
        Regex::new(r"<arg_key>([\s\S]*?)</arg_key>\s*<arg_value>([\s\S]*?)</arg_value>").ok()?;
    let mut calls = Vec::new();
    for caps in call.captures_iter(text) {
        let g = groups(&caps);
        if !available_tools.contains(&g.first) {
            continue;
        }
        let mut arguments = BTreeMap::new();
        for p in pair.captures_iter(g.second) {
            let kv = groups(&p);
            arguments.insert(kv.first.to_string(), coerce_value(kv.second));
        }
        calls.push(RawCall {
            name: g.first.to_string(),
            arguments,
        });
    }
    (!calls.is_empty()).then_some(calls)
}

/// MiniMax and anything else speaking the Anthropic/Claude invoke shape:
/// `<invoke name="NAME"><parameter name="K">V</parameter>...</invoke>`.
///
/// The `<minimax:tool_call>` wrapper is deliberately not part of the
/// pattern -- the invoke blocks are what carries the call, and matching
/// them alone costs nothing and covers a model that drops its own wrapper.
fn parse_invoke_parameters(text: &str, available_tools: &[&str]) -> Option<Vec<RawCall>> {
    let block = Regex::new(r#"<invoke\s+name="([A-Za-z0-9_\-]+)"\s*>([\s\S]*?)</invoke>"#).ok()?;
    let pair = Regex::new(r#"<parameter\s+name="([\s\S]*?)"\s*>([\s\S]*?)</parameter>"#).ok()?;
    collect_matches(text, &block, &pair, available_tools)
}

/// Qwen ChatML parameter XML: `<function=NAME><parameter=KEY>VALUE</parameter>...</function>`,
/// or JSON inside `<function=NAME>{...}</function>`.
fn parse_qwen_xml(text: &str, available_tools: &[&str]) -> Option<Vec<RawCall>> {
    let block = Regex::new(r"<function=([A-Za-z0-9_\-]+)>([\s\S]*?)</function>").ok()?;
    let pair = Regex::new(r"<parameter=([A-Za-z0-9_\-]+)>([\s\S]*?)</parameter>").ok()?;
    collect_matches(text, &block, &pair, available_tools)
}

/// Gemma DSL: `call:NAME{key:value,...}` or `call:NAME{"key":value,...}`.
///
/// The JSON-shaped body is found by a hand-rolled balanced-brace scan
/// rather than by capturing it with the regex alone: `regex_lite` has no
/// backreferences or recursion, so a lazy `\{[\s\S]*?\}` capture stops at
/// the FIRST closing brace it sees. For a call whose arguments contain a
/// nested object (`call:search{"query":"x","filters":{"category":"y"}}`,
/// a common shape for filters/options rather than an edge case) that
/// truncates the capture at the INNER object's `}`, producing an
/// unbalanced fragment that fails to parse -- silently dropping the whole
/// call, with no rescue and no error.
fn parse_gemma(text: &str, available_tools: &[&str]) -> Option<Vec<RawCall>> {
    let head = Regex::new(r"call:([A-Za-z0-9_\-]+)\s*\{").ok()?;
    let allowed: std::collections::HashSet<String> =
        available_tools.iter().map(|&s| s.to_string()).collect();
    let parser = tokenizer::GemmaToolCallParser::new();
    let mut calls = Vec::new();
    let mut cursor = 0usize;
    while cursor <= text.len() {
        let Some(caps) = head.captures(&text[cursor..]) else {
            break;
        };
        let whole_match = caps.get(0).map(|m| m.as_str()).unwrap_or_default();
        let name = caps
            .get(1)
            .map(|m| m.as_str())
            .unwrap_or_default()
            .to_string();
        let rel_start = caps.get(0).map(|m| m.start()).unwrap_or(0);
        // `whole_match` ends in the literal `{` the pattern requires, so
        // its last byte is that brace's own position.
        let brace_start = cursor + rel_start + whole_match.len() - 1;
        let Some(brace_len) = balanced_brace_len(&text[brace_start..]) else {
            // No matching close brace anywhere in the remaining text --
            // nothing else here can be a complete call either.
            break;
        };
        let body_end = brace_start + brace_len;
        let match_start = cursor + rel_start;
        cursor = body_end;

        if !available_tools.contains(&name.as_str()) {
            continue;
        }
        let matched = &text[match_start..body_end];
        if let Ok(parsed) = parser.parse(matched, &allowed, "id") {
            if let JsonValue::Object(map) = parsed.arguments {
                calls.push(RawCall {
                    name: parsed.name,
                    arguments: map,
                });
                continue;
            }
        }
        let body = &text[brace_start..body_end];
        if let Some(map) = json_object(body) {
            calls.push(RawCall {
                name,
                arguments: map,
            });
        }
    }
    (!calls.is_empty()).then_some(calls)
}

/// `s` must start with `{`. Returns the byte length of the balanced brace
/// span starting there (one past the matching `}`), or `None` if the
/// braces never balance before `s` ends. Tracks JSON string literals (a
/// `{`/`}` inside a quoted value does not perturb the depth count) and
/// backslash escapes within them; this needs no opinion on whether a key is
/// quoted, since the scan only ever looks at brace/quote/backslash bytes.
fn balanced_brace_len(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for (i, b) in s.bytes().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// Shared walk for the two tag-pair shapes: every non-overlapping block
/// contributes one call whose key/value pairs are read out of its body with
/// `pair` (group 1 the key, group 2 the value).
fn collect_matches(
    text: &str,
    block: &Regex,
    pair: &Regex,
    available_tools: &[&str],
) -> Option<Vec<RawCall>> {
    let mut calls = Vec::new();
    for caps in block.captures_iter(text) {
        let g = groups(&caps);
        if !available_tools.contains(&g.first) {
            continue;
        }
        let mut arguments = BTreeMap::new();
        for p in pair.captures_iter(g.second) {
            let kv = groups(&p);
            arguments.insert(kv.first.to_string(), coerce_value(kv.second));
        }
        // A body that never held a key/value pair is an empty argument
        // object, not a refusal: a no-argument tool is legal, and a call
        // with required arguments is better caught by the schema validator,
        // which reports exactly which field is missing.
        if arguments.is_empty() {
            if let Some(object) = json_object(g.second) {
                arguments = object;
            }
        }
        calls.push(RawCall {
            name: g.first.to_string(),
            arguments,
        });
    }
    (!calls.is_empty()).then_some(calls)
}

/// Kimi K2: an id and a JSON body, with the name INSIDE the id.
fn parse_kimi(text: &str, available_tools: &[&str]) -> Option<Vec<RawCall>> {
    let call = Regex::new(
        r"<\|tool_call_begin\|>\s*([\w.:\-]+)\s*<\|tool_call_argument_begin\|>([\s\S]*?)<\|tool_call_end\|>",
    )
    .ok()?;
    let mut calls = Vec::new();
    for caps in call.captures_iter(text) {
        let g = groups(&caps);
        let Some(name) = name_from_kimi_id(g.first) else {
            continue;
        };
        if !available_tools.contains(&name.as_str()) {
            continue;
        }
        let body = g.second.trim();
        let arguments = if body.is_empty() {
            BTreeMap::new()
        } else {
            // The body IS the arguments object; a body that will not parse
            // as one is a call this tier declines rather than guesses at.
            match json_object(body) {
                Some(object) => object,
                None => continue,
            }
        };
        calls.push(RawCall { name, arguments });
    }
    (!calls.is_empty()).then_some(calls)
}

/// `functions.get_weather:0` -> `get_weather`, per Moonshot's documented
/// id format. A documented-anomaly id (`call_ab12...`) yields that prefix
/// as the name, which the allowlist then refuses: there is no name in the
/// id to recover, and inventing one would be worse than a refusal.
fn name_from_kimi_id(id: &str) -> Option<String> {
    let without_namespace = id.strip_prefix("functions.").unwrap_or(id);
    let name = without_namespace.split(':').next()?;
    (!name.is_empty()).then(|| name.to_string())
}

/// A value is JSON-decoded only when it claims to be JSON, matching the
/// native Qwen parser's own rule: a quoted string value stays a string, a
/// bare word stays a string, and `3`, `{...}` and `true` become what they
/// say they are. Coercing here is what lets the schema validator check a
/// GLM `days` value as the integer the request asked for rather than as
/// the string that would always fail it.
fn coerce_value(raw: &str) -> JsonValue {
    let trimmed = raw.trim();
    let Some(first) = trimmed.chars().next() else {
        return JsonValue::String(raw.to_string());
    };
    if !"{[-0123456789tfn".contains(first) {
        return JsonValue::String(raw.to_string());
    }
    match JsonValue::parse(trimmed) {
        Ok(JsonValue::String(_)) => JsonValue::String(raw.to_string()),
        Ok(value) => value,
        Err(_) => JsonValue::String(raw.to_string()),
    }
}

fn json_object(text: &str) -> Option<BTreeMap<String, JsonValue>> {
    match JsonValue::parse(text.trim()) {
        Ok(JsonValue::Object(map)) => Some(map),
        _ => None,
    }
}
