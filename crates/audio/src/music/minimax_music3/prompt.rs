//! Caption and lyrics prompt assembly plus the tiny text encoder.
//!
//! Reference: `mlx_audio/music/models/minimax_music3/prompt.py` and
//! `_encode_tiny_text_pair` in `minimax_music3.py`. The reference uses
//! Python `re`; the patterns here are hand-ported, behavior-pinned by
//! fixture tests. Lowercasing in `normalize_lyrics` uses Rust's
//! `to_lowercase`, which matches Python `str.lower` on ASCII input (the
//! fixture corpus is ASCII).

use tokenizers::Tokenizer;

use crate::{Result, SpeechError};

/// `clean_caption`: rewrite `<|tag value|>` special tags to prose,
/// strip markdown headings, bullets, bold and italics, drop bullets
/// and horizontal rules, and collapse blank lines.
pub(crate) fn clean_caption(caption: &str) -> String {
    let text = rewrite_special_tags(caption);
    let mut lines = Vec::new();
    for line in text.split('\n') {
        let line = strip_heading(line);
        let line = strip_bullet(&line);
        let line = unmark_bold(&line);
        let line = unmark_italics(&line);
        lines.push(line.trim_end_matches([' ', '\t', '\r']).to_string());
    }
    let mut text = lines
        .join("\n")
        .replace("\u{2022} ", "")
        .replace("    ", "");
    text = strip_horizontal_rules(&text);
    collapse_blank_lines(&text)
}

/// `normalize_lyrics`: keep leading `[tag]` groups, split bracket
/// blocks and caret separators onto their own lines, lowercase the
/// bracket contents, and prefix `[start]`.
pub(crate) fn normalize_lyrics(lyrics: &str) -> String {
    let mut lines = Vec::new();
    for line in lyrics.split('\n') {
        match leading_tag_group(line) {
            Some(group) => lines.push(group.trim_matches([' ', '\t']).to_string()),
            None => lines.push(line.to_string()),
        }
    }
    let mut text = lines.join("\n");
    text = text
        .replace("] ", "]\n")
        .replace(" [", "\n[")
        .replace(" ^ ", "\n");
    text = lowercase_brackets(&text);
    format!("[start]\n{text}")
}

/// `assemble_prompt`: the caption-and-lyrics template the AR stage
/// consumes after tokenization.
pub(crate) fn assemble_prompt(caption: &str, lyrics: &str) -> String {
    format!(
        "<|im_start|><|caption_start|>{}<|caption_end|><|lyrics_start|>{}<|lyrics_end|><|im_end|><|audio_start|>",
        clean_caption(caption),
        normalize_lyrics(lyrics)
    )
}

/// The deterministic stand-in tokenizer used when no official
/// tokenizer is available: ids `(ord(c) * 17 + i * 3) % 180 + 1` over
/// the first 29 characters, wrapped in BOS 1 / EOS 2.
pub(crate) fn encode_tiny_ids(text: &str, max_length: usize) -> Vec<i32> {
    let mut ids = Vec::new();
    for (index, character) in text.chars().take(max_length - 3).enumerate() {
        ids.push((((character as u32) * 17 + index as u32 * 3) % 180 + 1) as i32);
    }
    let mut out = vec![1i32];
    if ids.is_empty() {
        out.push(1);
    } else {
        out.extend(ids);
    }
    out.push(2);
    out
}

/// Tokenize the assembled prompt with the checkpoint's Hugging Face
/// tokenizer. The model builds the unconditional CFG row from this
/// conditional row after tokenization.
pub(crate) fn encode_official_text(tokenizer: &Tokenizer, text: &str) -> Result<Vec<i32>> {
    let encoded = tokenizer
        .encode(text, true)
        .map_err(|error| SpeechError::Input {
            why: format!("MiniMax Music 3 tokenizer failed: {error}"),
        })?;
    encoded
        .get_ids()
        .iter()
        .map(|id| {
            i32::try_from(*id).map_err(|_| SpeechError::Input {
                why: format!("tokenizer id {id} exceeds the signed 32-bit model boundary"),
            })
        })
        .collect()
}

/// Build the conditional / unconditional id pair from one conditional
/// row: the unconditional row keeps the first and last two tokens and
/// fills the middle with the audio CFG token.
pub(crate) fn encode_tiny_text_pair(
    text: &str,
    audio_cfg_token_id: i32,
    max_length: usize,
) -> Vec<i32> {
    let conditional = encode_tiny_ids(text, max_length);
    let len = conditional.len();
    let mut unconditional = Vec::with_capacity(len);
    if len > 3 {
        unconditional.push(conditional[0]);
        unconditional.extend(std::iter::repeat_n(audio_cfg_token_id, len - 3));
        unconditional.extend_from_slice(&conditional[len - 2..]);
    } else {
        unconditional.extend_from_slice(&conditional);
    }
    let mut pair = conditional;
    pair.extend(unconditional);
    pair
}

/// `<|key value|>` -> `key is value`; `<|key|>` -> `key`.
fn rewrite_special_tags(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'<' && text[i..].starts_with("<|") {
            if let Some(close) = text[i + 2..].find("|>") {
                let inner = &text[i + 2..i + 2 + close];
                // split(None, 1): first token, then the remainder with
                // its leading whitespace dropped.
                let mut parts = inner.trim().splitn(2, char::is_whitespace);
                let first = parts.next().unwrap_or("");
                match parts.next().map(str::trim_start) {
                    Some(rest) if !rest.is_empty() => {
                        out.push_str(first);
                        out.push_str(" is ");
                        out.push_str(rest);
                    }
                    _ => out.push_str(first),
                }
                i += 2 + close + 2;
                continue;
            }
        }
        let ch = text[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// `^\s{0,3}#{1,6}\s+` -> "" (up to three spaces, 1-6 hashes, then
/// required whitespace).
fn strip_heading(line: &str) -> String {
    let leading = line.len() - line.trim_start_matches([' ', '\t']).len();
    if leading > 3 {
        return line.to_string();
    }
    let rest = &line[leading..];
    let hashes = rest.len() - rest.trim_start_matches('#').len();
    if hashes == 0 || hashes > 6 {
        return line.to_string();
    }
    let after = &rest[hashes..];
    match after.find(|c: char| !c.is_whitespace()) {
        Some(pos) if pos > 0 => after[pos..].to_string(),
        _ => line.to_string(),
    }
}

/// `^\s*[*+-]\s+` -> "".
fn strip_bullet(line: &str) -> String {
    let trimmed = line.trim_start_matches([' ', '\t']);
    let marker = trimmed.chars().next();
    if !matches!(marker, Some('*') | Some('+') | Some('-')) {
        return line.to_string();
    }
    let after = &trimmed[marker.unwrap().len_utf8()..];
    match after.find(|c: char| !c.is_whitespace()) {
        Some(pos) if pos > 0 => after[pos..].to_string(),
        _ => line.to_string(),
    }
}

/// `\*\*([^*]+)\*\*` -> inner.
fn unmark_bold(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(start) = rest.find("**") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        match after.find("**") {
            Some(end) if end > 0 && !after[..end].contains('*') => {
                out.push_str(&after[..end]);
                rest = &after[end + 2..];
            }
            _ => {
                out.push_str("**");
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// `(?<!\*)\*([^*\n]+)\*(?!\*)` -> inner: single-star emphasis not
/// adjacent to another star.
fn unmark_italics(line: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut out = String::with_capacity(line.len());
    let mut i = 0usize;
    while i < chars.len() {
        if chars[i] != '*' {
            out.push(chars[i]);
            i += 1;
            continue;
        }
        let prev_star = i > 0 && chars[i - 1] == '*';
        // Find the closing star.
        let mut j = i + 1;
        let mut closed = None;
        while j < chars.len() {
            if chars[j] == '*' {
                closed = Some(j);
                break;
            }
            if chars[j] == '\n' {
                break;
            }
            j += 1;
        }
        let next_star = closed.map(|j| j + 1 < chars.len() && chars[j + 1] == '*');
        match (closed, next_star) {
            (Some(j), Some(false)) if j > i + 1 && !prev_star => {
                out.extend(&chars[i + 1..j]);
                i = j + 1;
            }
            _ => {
                out.push('*');
                i += 1;
            }
        }
    }
    out
}

/// Blank out lines that are only `[-*_]{3,}` with optional whitespace.
/// The line separator survives the reference's MULTILINE sub, so a
/// trailing rule leaves one trailing newline behind.
fn strip_horizontal_rules(text: &str) -> String {
    let mut lines = Vec::new();
    for line in text.split('\n') {
        let stripped = line.trim_matches([' ', '\t', '\r']);
        let rule =
            stripped.len() >= 3 && stripped.chars().all(|c| c == '-' || c == '*' || c == '_');
        if rule {
            lines.push(String::new());
        } else {
            lines.push(line.to_string());
        }
    }
    lines.join("\n")
}

/// `\n{2,}` -> `\n`.
fn collapse_blank_lines(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut blank = 0usize;
    for ch in text.chars() {
        if ch == '\n' {
            blank += 1;
            if blank > 1 {
                continue;
            }
        } else {
            blank = 0;
        }
        out.push(ch);
    }
    out
}

/// `^[ \t]*((?:\[[^\]]+\][ \t]*)+)` -> the bracket group, if a line
/// starts with at least one `[...]` tag group.
fn leading_tag_group(line: &str) -> Option<String> {
    let trimmed = line.trim_start_matches([' ', '\t']);
    if !trimmed.starts_with('[') {
        return None;
    }
    let mut end = 0usize;
    let bytes = trimmed.as_bytes();
    let mut i = 0usize;
    loop {
        if i >= bytes.len() || bytes[i] != b'[' {
            break;
        }
        match trimmed[i + 1..].find(']') {
            Some(close) if close > 0 => {
                i = i + 1 + close + 1;
            }
            _ => return None,
        }
        // Optional separator whitespace before the next tag.
        let ws = trimmed[i..].len() - trimmed[i..].trim_start_matches([' ', '\t']).len();
        i += ws;
        end = i;
    }
    if end == 0 {
        return None;
    }
    Some(trimmed[..end].to_string())
}

/// `\[([^\]]+)\]` -> lowercased inner.
fn lowercase_brackets(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find('[') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        match after.find(']') {
            Some(close) => {
                out.push('[');
                out.push_str(&after[..close].to_lowercase());
                out.push(']');
                rest = &after[close + 1..];
            }
            None => {
                out.push('[');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}
