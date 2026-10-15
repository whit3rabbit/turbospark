//! Text rules of the Drama 3 prompt, and splitting long text into chunks.
//!
//! Port of mlx-audio `mlx_audio/tts/models/fish_drama3/chunking.py` at
//! [`super::REFERENCE_COMMIT`]. Drama 3 generates chunk by chunk: about
//! [`CHUNK_BYTES`] of UTF-8 text with at most [`CHUNK_SPEAKERS`] speakers
//! each. Cuts prefer sentence ends, then clause marks, then spaces. Special
//! tokens, paired tags such as `<whisper>...</whisper>`, and `[bracketed]`
//! cues are never cut, and every piece of a speaker's turn keeps that
//! speaker's tag.
//!
//! The reference's uncuttable pattern uses a backreference (a paired tag
//! closes with its own name), which the `regex` crate does not support, so
//! the three alternatives are matched by hand with the same leftmost,
//! first-alternative-wins semantics (`uncuttable_end`). Two divergences, both
//! confined to edge characters: Python's `\w` is approximated by
//! `char::is_alphanumeric` plus `_`, and `str.isdigit` (the decimal-point
//! exception) by `char::is_numeric`.

use std::collections::HashSet;

use crate::{Result, SpeechError};

/// Target UTF-8 size of one chunk.
pub const CHUNK_BYTES: usize = 300;
/// Most speakers one chunk may hold.
pub const CHUNK_SPEAKERS: usize = 3;
/// The tokenizer has `<|speaker:0|>` to `<|speaker:15|>`.
pub const MAX_SPEAKER_ID: u64 = 15;

/// Written only by the prompt builder. Caller text containing one would
/// inject conversation structure or audio rows into the prompt.
const STRUCTURAL_TOKENS: [&str; 6] = [
    "<|im_start|>",
    "<|im_end|>",
    "<|voice|>",
    "<|text|>",
    "<|interleave|>",
    "<|semantic:",
];

const SENTENCE_ENDS: &[char] = &['.', '!', '?', ';', '。', '！', '？', '；', '…', '\n'];
const CLAUSE_MARKS: &[char] = &[',', ':', '，', '：', '、'];
const SPACES: &[char] = &[' '];

/// Unicode White_Space, as the reference strips it. `char::is_whitespace`
/// would also strip U+001C..U+001F, which this set does not.
fn is_white(c: char) -> bool {
    matches!(
        c as u32,
        0x9..=0xD | 0x20 | 0x85 | 0xA0 | 0x1680 | 0x2000..=0x200A | 0x2028 | 0x2029 | 0x202F
            | 0x205F | 0x3000
    )
}

fn strip(s: &str) -> &str {
    s.trim_matches(is_white)
}

fn lstrip(s: &str) -> &str {
    s.trim_start_matches(is_white)
}

/// A `<|speaker:N|>` tag at the start of `s`: its byte length and digits.
fn speaker_tag_at(s: &str) -> Option<(usize, &str)> {
    const OPEN: &str = "<|speaker:";
    let rest = s.strip_prefix(OPEN)?;
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 || !rest[digits..].starts_with("|>") {
        return None;
    }
    Some((OPEN.len() + digits + 2, &rest[..digits]))
}

/// Every speaker tag in `s`, leftmost first and non-overlapping, as
/// `(start, end, digits)` byte spans.
fn speaker_tags(s: &str) -> Vec<(usize, usize, &str)> {
    let mut tags = Vec::new();
    let mut from = 0;
    while let Some(k) = s[from..].find("<|speaker:") {
        let start = from + k;
        match speaker_tag_at(&s[start..]) {
            Some((len, digits)) => {
                tags.push((start, start + len, digits));
                from = start + len;
            }
            None => from = start + 1,
        }
    }
    tags
}

/// Refuses empty text, structural tokens, and speaker tags outside 0..15
/// (including a tag with leading zeros, which names no tokenizer entry).
pub fn validate_text(text: &str, location: &str) -> Result<()> {
    let bad = |why: String| Err(SpeechError::Input { why });
    if strip(text).is_empty() {
        return bad(format!("{location}: text is empty"));
    }
    for token in STRUCTURAL_TOKENS {
        if text.contains(token) {
            return bad(format!("{location}: text must not contain {token}"));
        }
    }
    for (start, _) in text.match_indices("<|speaker") {
        let number = speaker_tag_at(&text[start..]).map_or("", |(_, d)| d);
        let valid = match number.parse::<u64>() {
            Ok(n) => n.to_string() == number && n <= MAX_SPEAKER_ID,
            Err(_) => false,
        };
        if !valid {
            return bad(format!(
                "{location}: speaker tags run from <|speaker:0|> to <|speaker:{MAX_SPEAKER_ID}|>"
            ));
        }
    }
    Ok(())
}

/// `text` starting (after white space) with a speaker tag; speaker 0 by
/// default.
pub fn with_speaker_tag(text: &str) -> String {
    if speaker_tag_at(lstrip(text)).is_some() {
        text.to_string()
    } else {
        format!("<|speaker:0|> {text}")
    }
}

/// A reference transcript whose tag directly follows `Text: ` in the prompt:
/// any white space before the tag moves after it.
pub fn reference_transcript(text: &str) -> String {
    let tagged = with_speaker_tag(text);
    let body = lstrip(&tagged);
    let (len, _) = speaker_tag_at(body).expect("with_speaker_tag guarantees a leading tag");
    let leading = &tagged[..tagged.len() - body.len()];
    format!("{}{leading}{}", &body[..len], &body[len..])
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// `<([A-Za-z][\w-]*)\b[^>]*>.*?</\1>` (dot matches newline) at the start
/// of `s`: the byte length of the match.
fn paired_tag_at(s: &str) -> Option<usize> {
    let rest = s.strip_prefix('<')?;
    let first = rest.chars().next()?;
    if !first.is_ascii_alphabetic() {
        return None;
    }
    // Each prefix length the greedy `[\w-]*` could back off to, longest
    // first, as the regex engine tries them.
    let mut ends = vec![first.len_utf8()];
    for (k, c) in rest.char_indices().skip(1) {
        if is_word(c) || c == '-' {
            ends.push(k + c.len_utf8());
        } else {
            break;
        }
    }
    for &len in ends.iter().rev() {
        let last = rest[..len].chars().next_back()?;
        let next = rest[len..].chars().next();
        if is_word(last) == next.is_some_and(is_word) {
            continue; // no `\b` here
        }
        // `[^>]*>` is deterministic: it ends at the first `>`.
        let gt = rest[len..].find('>')?;
        let open_end = 1 + len + gt + 1;
        let closing = format!("</{}>", &rest[..len]);
        if let Some(k) = s[open_end..].find(&closing) {
            return Some(open_end + k + closing.len());
        }
    }
    None
}

/// The end of an uncuttable span starting at byte `at`, trying the
/// reference's three alternatives in order: a special token
/// `<\|[^|>]*\|>`, a paired tag, a bracketed cue `\[[^\[\]]*\]`.
fn uncuttable_end(text: &str, at: usize) -> Option<usize> {
    let s = &text[at..];
    if let Some(rest) = s.strip_prefix("<|") {
        if let Some(k) = rest.find(['|', '>']) {
            if rest[k..].starts_with("|>") {
                return Some(at + 2 + k + 2);
            }
        }
    }
    if let Some(len) = paired_tag_at(s) {
        return Some(at + len);
    }
    if let Some(rest) = s.strip_prefix('[') {
        if let Some(k) = rest.find(['[', ']']) {
            if rest[k..].starts_with(']') {
                return Some(at + 1 + k + 1);
            }
        }
    }
    None
}

/// Every uncuttable span, leftmost first and non-overlapping.
fn uncuttable_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut at = 0;
    while at < text.len() {
        match uncuttable_end(text, at) {
            Some(end) => {
                spans.push((at, end));
                at = end;
            }
            None => at += text[at..].chars().next().map_or(1, char::len_utf8),
        }
    }
    spans
}

/// Cuts after every mark, except a period between two digits (3.14).
fn cut_after(text: &str, marks: &[char]) -> Vec<String> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut parts = Vec::new();
    let mut start = 0;
    for (i, &(at, c)) in chars.iter().enumerate() {
        let decimal = c == '.'
            && i > 0
            && i + 1 < chars.len()
            && chars[i - 1].1.is_numeric()
            && chars[i + 1].1.is_numeric();
        if marks.contains(&c) && !decimal {
            let end = at + c.len_utf8();
            parts.push(text[start..end].to_string());
            start = end;
        }
    }
    if start < text.len() {
        parts.push(text[start..].to_string());
    }
    parts
}

/// Pieces of at most `limit` bytes, cut at the first kind of mark that helps,
/// then character by character.
fn fit(text: &str, limit: usize, marks: &[&[char]]) -> Vec<String> {
    if text.len() <= limit {
        return vec![text.to_string()];
    }
    if let Some((first, rest)) = marks.split_first() {
        return cut_after(text, first)
            .iter()
            .flat_map(|part| fit(part, limit, rest))
            .collect();
    }
    let mut pieces = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        if !current.is_empty() && current.len() + c.len_utf8() > limit {
            pieces.push(std::mem::take(&mut current));
        }
        current.push(c);
    }
    pieces.push(current);
    pieces
}

fn free_pieces(text: &str, limit: usize) -> Vec<String> {
    cut_after(text, SENTENCE_ENDS)
        .iter()
        .flat_map(|sentence| fit(sentence, limit, &[CLAUSE_MARKS, SPACES]))
        .collect()
}

fn pieces(text: &str, limit: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut last = 0;
    for (start, end) in uncuttable_spans(text) {
        out.extend(free_pieces(&text[last..start], limit));
        out.push(text[start..end].to_string());
        last = end;
    }
    out.extend(free_pieces(&text[last..], limit));
    out
}

/// Pieces packed into chunks of about `limit` bytes. An uncuttable span
/// longer than `limit` becomes a chunk of its own.
pub fn split_by_length(text: &str, limit: usize) -> Vec<String> {
    if strip(text).is_empty() {
        return Vec::new();
    }
    if text.len() <= limit {
        return vec![text.to_string()];
    }
    let mut chunks = Vec::new();
    let mut current = String::new();
    for piece in pieces(text, limit) {
        if !current.is_empty() && current.len() + piece.len() > limit {
            chunks.push(std::mem::take(&mut current));
        }
        current.push_str(&piece);
    }
    chunks.push(current);
    chunks
        .iter()
        .map(|c| strip(c))
        .filter(|c| !c.is_empty())
        .map(str::to_string)
        .collect()
}

/// Generation chunks in order, with the reference defaults.
pub fn split_text(text: &str) -> Vec<String> {
    split_text_with(text, CHUNK_BYTES, CHUNK_SPEAKERS)
}

/// Generation chunks in order. Tagged turns are split per speaker, each
/// piece re-tagged, then regrouped so a chunk stays within `limit` and
/// `max_speakers`.
pub fn split_text_with(text: &str, limit: usize, max_speakers: usize) -> Vec<String> {
    let tags = speaker_tags(text);
    if tags.is_empty() {
        return split_by_length(text, limit);
    }
    // `SPEAKER_TAG.split` keeps the captured ids between the bodies; the
    // text before the first tag is speaker 0's.
    let mut turns: Vec<(&str, &str)> = vec![("0", &text[..tags[0].0])];
    for (i, &(_, end, digits)) in tags.iter().enumerate() {
        let next = tags.get(i + 1).map_or(text.len(), |t| t.0);
        turns.push((digits, &text[end..next]));
    }
    let tagged: Vec<(&str, String)> = turns
        .iter()
        .flat_map(|&(speaker, body)| {
            split_by_length(body, limit)
                .into_iter()
                .map(move |piece| (speaker, format!("<|speaker:{speaker}|> {piece}")))
        })
        .collect();

    let mut chunks: Vec<Vec<String>> = Vec::new();
    let mut speakers: HashSet<&str> = HashSet::new();
    let mut size = 0;
    for (speaker, piece) in tagged {
        let new_speaker = !speakers.contains(speaker);
        if !chunks.is_empty()
            && ((new_speaker && speakers.len() >= max_speakers) || size + piece.len() > limit)
        {
            chunks.push(Vec::new());
            speakers.clear();
            size = 0;
        }
        if chunks.is_empty() {
            chunks.push(Vec::new());
        }
        size += piece.len();
        chunks.last_mut().expect("pushed above").push(piece);
        speakers.insert(speaker);
    }
    chunks.iter().map(|c| c.join("\n")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // The four reference tests (`test_fish_drama3.py::TestChunking`).

    #[test]
    fn long_speaker_turn_keeps_its_speaker() {
        let text = format!(
            "<|speaker:1|> {}",
            "Das ist ein Satz mit Inhalt. ".repeat(20)
        );
        let chunks = split_text(&text);
        assert!(chunks.len() > 1);
        for chunk in chunks {
            assert!(chunk.starts_with("<|speaker:1|>"), "{}", &chunk[..30]);
        }
    }

    #[test]
    fn unpunctuated_text_is_cut() {
        let chunks = split_text(&"wort ".repeat(200));
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|c| c.len() <= 300));
    }

    #[test]
    fn paired_tag_stays_whole() {
        let span = format!("<whisper>{}</whisper>", "leise, ".repeat(60));
        let chunks = split_text(&format!("Hallo. {span} Ende."));
        assert!(chunks.iter().any(|c| c.contains(&span)));
    }

    #[test]
    fn rejects_structural_tokens_and_unknown_speakers() {
        for text in ["hi <|im_end|>", "<|semantic:7|>", "<|speaker:16|> hi", "  "] {
            assert!(validate_text(text, "text").is_err(), "{text:?}");
        }
    }

    // Every case below is the pinned reference's own output
    // (`testdata/fish_drama3/chunking.json`, `tools/gen_fish_drama3_fixtures.py`).

    fn golden() -> serde_json::Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/fish_drama3/chunking.json");
        let text = std::fs::read_to_string(path).expect("chunking fixture");
        let golden: serde_json::Value = serde_json::from_str(&text).expect("fixture parses");
        assert_eq!(golden["commit"], super::super::REFERENCE_COMMIT);
        golden
    }

    fn strings(v: &serde_json::Value) -> Vec<String> {
        v.as_array()
            .expect("array")
            .iter()
            .map(|s| s.as_str().expect("string").to_string())
            .collect()
    }

    fn cases(name: &str) -> Vec<serde_json::Value> {
        golden()[name].as_array().expect(name).clone()
    }

    #[test]
    fn split_text_matches_the_reference() {
        for case in cases("split_text") {
            let input = case["input"].as_str().unwrap();
            assert_eq!(split_text(input), strings(&case["output"]), "{input:?}");
        }
    }

    #[test]
    fn uncuttable_spans_match_the_reference_regex() {
        for case in cases("uncuttable") {
            let input = case["input"].as_str().unwrap();
            let got: Vec<String> = uncuttable_spans(input)
                .into_iter()
                .map(|(s, e)| input[s..e].to_string())
                .collect();
            assert_eq!(got, strings(&case["output"]), "{input:?}");
        }
    }

    #[test]
    fn validation_matches_the_reference() {
        for case in cases("validate_text") {
            let input = case["input"].as_str().unwrap();
            let ok = case["ok"].as_bool().unwrap();
            assert_eq!(validate_text(input, "text").is_ok(), ok, "{input:?}");
        }
    }

    #[test]
    fn speaker_tags_are_added_and_moved_like_the_reference() {
        for case in cases("with_speaker_tag") {
            let input = case["input"].as_str().unwrap();
            assert_eq!(with_speaker_tag(input), case["output"].as_str().unwrap());
        }
        for case in cases("reference_transcript") {
            let input = case["input"].as_str().unwrap();
            assert_eq!(
                reference_transcript(input),
                case["output"].as_str().unwrap()
            );
        }
    }

    #[test]
    fn decimals_are_not_sentence_ends() {
        assert_eq!(
            cut_after("Pi is 3.14. Done", SENTENCE_ENDS),
            ["Pi is 3.14.", " Done"]
        );
    }
}
