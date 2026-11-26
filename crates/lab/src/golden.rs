//! The golden-answer tripwire (LAB-08): a few prompts with exact answers.
//!
//! KL gates need a reference arm, which a new family does not have; a wrong
//! family can produce fluent, wrong output. Only the exact answers block.
//! The garbage flag is advisory.

use serde::{Deserialize, Serialize};

/// The embedded default set.
pub const DEFAULT_SET: &str = include_str!("../workloads/golden.json");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Case {
    pub id: String,
    pub prompt: String,
    /// Any of these as a whole word passes.
    pub accept: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Set {
    #[serde(default)]
    pub description: String,
    pub cases: Vec<Case>,
}

pub fn parse_set(text: &str) -> Result<Set, String> {
    let set: Set = serde_json::from_str(text).map_err(|e| format!("golden set: {e}"))?;
    if set.cases.is_empty() {
        return Err("golden set has no cases".into());
    }
    for c in &set.cases {
        crate::paths::validate_name(&c.id)?;
        if c.accept.is_empty() || c.accept.iter().any(|a| words(a).is_empty()) {
            return Err(format!("golden case {}: accept is empty", c.id));
        }
    }
    Ok(set)
}

/// Lowercase words; letters, digits and inner hyphens are kept.
fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !(c.is_alphanumeric() || c == '-'))
        .map(|w| w.trim_matches('-').to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

/// What a reply is judged on: the text after the reasoning, if the model
/// reasons. A reply that never closed its reasoning has no answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reply {
    pub answer: String,
    /// Reasoning markup that reached the answer, or never closed.
    pub think_leak: bool,
}

pub fn visible_reply(text: &str) -> Reply {
    let (answer, closed) = match text.rsplit_once("</think>") {
        Some((_, after)) => (after, true),
        None => (text, false),
    };
    // Opened and never closed: everything is reasoning.
    if !closed && answer.contains("<think>") {
        return Reply {
            answer: String::new(),
            think_leak: true,
        };
    }
    Reply {
        answer: answer.trim().to_string(),
        think_leak: answer.contains("<think>"),
    }
}

/// Whether the answer contains an accepted word (or word sequence).
pub fn accepts(case: &Case, answer: &str) -> bool {
    let got = words(answer);
    case.accept.iter().any(|a| {
        let want = words(a);
        got.windows(want.len()).any(|w| w == want.as_slice())
    })
}

/// Advisory: empty, replacement characters, no letters or digits, or one
/// short loop repeated. Never blocks.
pub fn looks_like_garbage(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() || t.contains('\u{fffd}') || !t.chars().any(char::is_alphanumeric) {
        return true;
    }
    let w: Vec<&str> = t.split_whitespace().collect();
    if w.len() >= 24 {
        let grams: std::collections::HashSet<&[&str]> = w.windows(3).collect();
        if grams.len() * 2 < w.len() - 2 {
            return true;
        }
    }
    // A run of one character with no spaces to split it.
    let chars: Vec<char> = t.chars().collect();
    chars.windows(40).any(|w| w.iter().all(|c| *c == w[0]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(accept: &str) -> Case {
        Case {
            id: "c".into(),
            prompt: "p".into(),
            accept: vec![accept.into()],
        }
    }

    #[test]
    fn the_embedded_set_parses() {
        assert!(parse_set(DEFAULT_SET).unwrap().cases.len() >= 8);
        assert!(parse_set(r#"{"cases":[]}"#).is_err());
        assert!(parse_set(r#"{"cases":[{"id":"a","prompt":"p","accept":["."]}]}"#).is_err());
    }

    #[test]
    fn answers_match_whole_words_only() {
        assert!(accepts(&case("391"), "The answer is **391**."));
        assert!(!accepts(&case("391"), "3910"));
        assert!(accepts(&case("tom"), "Tom!"));
        assert!(!accepts(&case("tom"), "tomorrow"));
        assert!(accepts(&case("zebra-7431"), "zebra-7431"));
        assert!(!accepts(&case("zebra-7431"), "zebra 74"));
        assert!(!accepts(&case("32"), ""));
    }

    #[test]
    fn reasoning_is_not_an_answer() {
        let r = visible_reply("<think>Paris?</think>\nLondon");
        assert_eq!(r.answer, "London");
        assert!(!r.think_leak);
        // The template renders the opening tag in the prompt.
        assert_eq!(visible_reply("maybe 391</think>391").answer, "391");
        let open = visible_reply("<think> the answer is paris");
        assert!(open.think_leak && open.answer.is_empty());
        assert!(visible_reply("</think> a <think> b").think_leak);
    }

    #[test]
    fn garbage_is_flagged_and_prose_is_not() {
        assert!(looks_like_garbage(""));
        assert!(looks_like_garbage("!!!!!!"));
        assert!(looks_like_garbage(&"a".repeat(50)));
        assert!(looks_like_garbage(&"the cat sat ".repeat(20)));
        assert!(!looks_like_garbage(
            "Paris is the capital of France, a city on the Seine with a long history."
        ));
    }
}
