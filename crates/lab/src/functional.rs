//! The functional half of the golden tripwire (LAB-08): a needle grid over
//! long context and strict tool calls. A ring that is too small or a pipeline
//! key that misses a constant passes short exact answers and fails these.
//!
//! Tool calls are rendered by the checkpoint's own template and read back by
//! the runtime's own decoder, so a pass means the shipped path parses them.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use tokenizer::{
    render_generic_chat_template, FunctionDefinition, JsonValue, Message, MfTokenizer,
    ReasoningEffort, Role, StructuredAssistantDecoder, StructuredAssistantEvent,
};

use crate::golden::Case;
use crate::workload::PromptSource;

/// Context lengths of the needle grid, tokens of filler.
pub const NEEDLE_SIZES: [usize; 3] = [4_096, 16_384, 65_536];
/// Where the needle sits, percent of the filler.
pub const NEEDLE_DEPTHS: [usize; 3] = [10, 50, 90];

/// What a case expects of the reply.
#[derive(Debug, Clone)]
pub enum Expect {
    /// An exact answer (`golden.rs`).
    Answer(Case),
    /// One call of `name` whose arguments hold exactly these values.
    Tool {
        name: &'static str,
        args: Vec<(&'static str, &'static str)>,
    },
}

#[derive(Debug, Clone)]
pub struct Check {
    pub id: String,
    pub source: PromptSource,
    pub expect: Expect,
}

/// The filler with the needle sentence at `depth` percent, cut between ids so
/// no word is split.
pub fn needle_ids(filler: &[i32], needle: &[i32], size: usize, depth: usize) -> Vec<i32> {
    let size = size.min(filler.len());
    let at = size * depth / 100;
    let mut out = Vec::with_capacity(size + needle.len());
    out.extend_from_slice(&filler[..at]);
    out.extend_from_slice(needle);
    out.extend_from_slice(&filler[at..size]);
    out
}

// The needle is a dull fact on purpose: an "access code" made a model on a
// real install deliberate about secrets until its token budget ran out.

/// A code per grid point, so a model cannot answer one from another.
pub fn needle_code(size: usize, depth: usize) -> String {
    format!("{}", 100_000 + (size * 7 + depth * 131) % 900_000)
}

/// The needle cases that fit `max_context` with `decode` steps to spare.
pub fn needle_checks(
    tokenizer: &MfTokenizer,
    filler: &str,
    dir: &Path,
    sizes: &[usize],
    max_context: usize,
    decode: usize,
) -> Result<(Vec<Check>, Vec<String>), String> {
    let ids = tokenizer.encode(filler, false);
    let mut checks = Vec::new();
    let mut skipped = Vec::new();
    for &size in sizes {
        // Room for the needle, the question and the template.
        if size + decode + 256 > max_context || size > ids.len() {
            skipped.push(format!(
                "needle {size}: needs --context over {} and {size} filler tokens (corpus has {})",
                size + decode + 256,
                ids.len()
            ));
            continue;
        }
        for depth in NEEDLE_DEPTHS {
            let code = needle_code(size, depth);
            let needle = tokenizer.encode(
                &format!("\n\nThe library shelf number is {code}.\n\n"),
                false,
            );
            let text = tokenizer.decode(&needle_ids(&ids, &needle, size, depth), false);
            let id = format!("needle_{size}_{depth}");
            let path = dir.join(format!("{id}.messages.json"));
            crate::atomic::write_new_json(
                &path,
                &json!([{"role": "user", "content": format!(
                    "{text}\n\nWhat is the library shelf number stated in the text above? Reply with the number only."
                )}]),
            )?;
            checks.push(Check {
                expect: Expect::Answer(Case {
                    id: id.clone(),
                    prompt: String::new(),
                    accept: vec![code],
                }),
                id,
                source: PromptSource::MessagesFile(path),
            });
        }
    }
    Ok((checks, skipped))
}

fn function(name: &str, description: &str, params: &[(&str, &str)]) -> FunctionDefinition {
    let properties: serde_json::Map<String, Value> = params
        .iter()
        .map(|(n, t)| (n.to_string(), json!({"type": t})))
        .collect();
    let required: Vec<&str> = params.iter().map(|(n, _)| *n).collect();
    let schema = json!({"type": "object", "properties": properties, "required": required});
    FunctionDefinition {
        name: name.into(),
        description: description.into(),
        parameters: JsonValue::parse(&schema.to_string()).unwrap_or(JsonValue::Null),
    }
}

fn tools() -> Vec<FunctionDefinition> {
    vec![
        function(
            "get_weather",
            "Current weather for a city.",
            &[("city", "string")],
        ),
        function(
            "add_numbers",
            "Add two integers.",
            &[("a", "integer"), ("b", "integer")],
        ),
        function("set_timer", "Start a timer.", &[("minutes", "integer")]),
        function(
            "search_files",
            "Search files for a text.",
            &[("query", "string")],
        ),
    ]
}

/// A request, the tool it should call and the argument values it should pass.
type ToolCase = (
    &'static str,
    &'static str,
    &'static [(&'static str, &'static str)],
);

const TOOL_CASES: [ToolCase; 4] = [
    (
        "What is the weather in Oslo right now?",
        "get_weather",
        &[("city", "Oslo")],
    ),
    (
        "Use the tool to add 4821 and 977.",
        "add_numbers",
        &[("a", "4821"), ("b", "977")],
    ),
    (
        "Set a timer for 25 minutes.",
        "set_timer",
        &[("minutes", "25")],
    ),
    (
        "Search my files for the word quarterly.",
        "search_files",
        &[("query", "quarterly")],
    ),
];

/// The tool cases as rendered prompts. `Err` for a checkpoint with no
/// template that takes tools.
pub fn tool_checks(tokenizer: &MfTokenizer, dir: &Path) -> Result<Vec<Check>, String> {
    let mut checks = Vec::new();
    for (i, (ask, name, args)) in TOOL_CASES.iter().enumerate() {
        let rendered = render_generic_chat_template(
            tokenizer,
            &[Message::new(Role::User, *ask)],
            &tools(),
            true,
            ReasoningEffort::default(),
        )
        .map_err(|e| format!("tool template: {e}"))?;
        let id = format!("tool_{i}_{name}");
        let path: PathBuf = dir.join(format!("{id}.prompt.txt"));
        // The template writes its own special tokens, and a text file is
        // encoded as is.
        crate::atomic::write_new(&path, rendered.as_bytes())?;
        checks.push(Check {
            id,
            source: PromptSource::TextFile(path),
            expect: Expect::Tool {
                name,
                args: args.to_vec(),
            },
        });
    }
    Ok(checks)
}

/// A JSON value as the text an expectation is written in.
fn plain(v: &JsonValue) -> Option<String> {
    match v {
        JsonValue::String(s) => Some(s.clone()),
        JsonValue::Integer(i) => Some(i.to_string()),
        JsonValue::UnsignedInteger(u) => Some(u.to_string()),
        _ => None,
    }
}

/// Exactly the expected tool with exactly the expected arguments; text
/// arguments match whatever their case.
pub fn call_matches(name: &str, arguments: &JsonValue, want: &str, args: &[(&str, &str)]) -> bool {
    let Some(got) = arguments.as_object() else {
        return false;
    };
    name == want
        && got.len() == args.len()
        && args.iter().all(|(k, v)| {
            got.get(*k)
                .and_then(plain)
                .is_some_and(|g| g.eq_ignore_ascii_case(v))
        })
}

/// Read the generated ids through the runtime's decoder and take the first
/// tool call. The run ignores end-of-turn, so anything after the first call
/// is not read.
pub fn first_tool_call(
    tokenizer: &MfTokenizer,
    prompt: &str,
    generated: &[i32],
) -> Result<Option<(String, JsonValue, String)>, String> {
    let allowed: HashSet<String> = tools().into_iter().map(|t| t.name).collect();
    let prompt_ids = tokenizer.encode(prompt, false);
    let mut next = 0;
    let mut decoder = StructuredAssistantDecoder::new(
        tokenizer,
        allowed,
        move || {
            next += 1;
            format!("call_{next}")
        },
        &prompt_ids,
    );
    let take = |events: Vec<StructuredAssistantEvent>| {
        events.into_iter().find_map(|e| match e {
            StructuredAssistantEvent::ToolCall(c) => Some((c.name, c.arguments, c.arguments_json)),
            _ => None,
        })
    };
    for &id in generated {
        let delta = tokenizer.decode(&[id], false);
        let events = decoder
            .consume(id, &delta)
            .map_err(|e| format!("tool parser: {e}"))?;
        if let Some(call) = take(events) {
            return Ok(Some(call));
        }
    }
    Ok(decoder.finish().ok().and_then(take))
}

/// Judge one reply; returns pass and what was read.
pub fn judge(check: &Check, tokenizer: &MfTokenizer, text: &str, ids: &[i32]) -> (bool, Value) {
    match &check.expect {
        Expect::Answer(case) => {
            let reply = crate::golden::visible_reply(text);
            let answer: String = reply.answer.chars().take(120).collect();
            (
                crate::golden::accepts(case, &reply.answer),
                json!({"answer": answer, "think_leak": reply.think_leak,
                       "garbage_flag": crate::golden::looks_like_garbage(&reply.answer)}),
            )
        }
        Expect::Tool { name, args } => {
            let prompt = match &check.source {
                PromptSource::TextFile(p) => std::fs::read_to_string(p).unwrap_or_default(),
                _ => String::new(),
            };
            match first_tool_call(tokenizer, &prompt, ids) {
                Ok(Some((got, arguments, raw))) => (
                    call_matches(&got, &arguments, name, args),
                    json!({"answer": format!("{got} {raw}")}),
                ),
                Ok(None) => (false, json!({"answer": "no tool call"})),
                Err(e) => (false, json!({"answer": e})),
            }
        }
    }
}

/// The cases the candidate fails that the baseline passed, and the reverse.
/// A class that "matches the pass vector" has neither.
pub fn pass_vector_difference(baseline: &[(String, bool)], candidate: &[(String, bool)]) -> Value {
    let differs = |a: &[(String, bool)], b: &[(String, bool)]| -> Vec<String> {
        a.iter()
            .filter(|(id, pass)| *pass && !b.iter().any(|(i, p)| i == id && *p))
            .map(|(id, _)| id.clone())
            .collect()
    };
    json!({"lost": differs(baseline, candidate), "gained": differs(candidate, baseline)})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_needle_lands_at_its_depth_and_codes_differ() {
        let filler: Vec<i32> = (0..100).collect();
        let out = needle_ids(&filler, &[-1, -2], 50, 10);
        assert_eq!(out.len(), 52);
        assert_eq!(&out[4..8], &[4, -1, -2, 5]);
        assert_eq!(needle_ids(&filler, &[-1], 50, 90)[45], -1);
        let codes: HashSet<String> = NEEDLE_SIZES
            .iter()
            .flat_map(|&s| NEEDLE_DEPTHS.iter().map(move |&d| needle_code(s, d)))
            .collect();
        assert_eq!(codes.len(), 9);
        assert!(codes.iter().all(|c| c.len() == 6));
    }

    #[test]
    fn a_tool_call_must_match_name_and_every_argument() {
        let args = JsonValue::parse(r#"{"a": 4821, "b": "977"}"#).unwrap();
        let want = [("a", "4821"), ("b", "977")];
        assert!(call_matches("add_numbers", &args, "add_numbers", &want));
        assert!(!call_matches("set_timer", &args, "add_numbers", &want));
        assert!(!call_matches(
            "add_numbers",
            &args,
            "add_numbers",
            &want[..1]
        ));
        let city = JsonValue::parse(r#"{"city": "oslo"}"#).unwrap();
        assert!(call_matches(
            "get_weather",
            &city,
            "get_weather",
            &[("city", "Oslo")]
        ));
        let wrong = JsonValue::parse(r#"{"city": "Bergen"}"#).unwrap();
        assert!(!call_matches(
            "get_weather",
            &wrong,
            "get_weather",
            &[("city", "Oslo")]
        ));
        assert!(!call_matches(
            "get_weather",
            &JsonValue::Null,
            "get_weather",
            &[]
        ));
    }

    #[test]
    fn the_pass_vector_names_what_was_lost_and_gained() {
        let v = |x: &[(&str, bool)]| -> Vec<(String, bool)> {
            x.iter().map(|(i, p)| (i.to_string(), *p)).collect()
        };
        let d = pass_vector_difference(
            &v(&[("a", true), ("b", false), ("c", true)]),
            &v(&[("a", true), ("b", true), ("c", false)]),
        );
        assert_eq!(d, json!({"lost": ["c"], "gained": ["b"]}));
        // A case the candidate never ran is lost.
        let d = pass_vector_difference(&v(&[("a", true)]), &[]);
        assert_eq!(d["lost"], json!(["a"]));
    }
}
