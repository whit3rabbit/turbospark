//! English text frontend of MegaTTS3: language gate, normalization, chunking.
//!
//! `MegaTTS3DiTInfer.forward` classifies the language (`langdetect`), runs a
//! WeTextProcessing normalizer, and splits the text into chunks that each fit
//! one DiT pass. This port is English-first:
//!
//! - [`ensure_english`] stands in for `langdetect`: Latin-script text is treated
//!   as English and any other script is a clear error (the Chinese path needs a
//!   Chinese normalizer that this port does not have).
//! - [`chunk_english`] is an exact port of `chunk_text_english` and is checked
//!   against the reference function.
//! - [`normalize_english`] is NOT the reference normalizer. WeTextProcessing's
//!   English grammar is NeMo's, compiled from large pynini FAR files that cannot
//!   run here (pynini ships no macOS wheel), so its output could not be
//!   compared. This is a conservative subset (whitespace, integers, decimals,
//!   percent, ordinals, dollar amounts; everything else is passed through
//!   unchanged) that follows NeMo's wording, for example no "and" in "one hundred
//!   twenty three". It is documented as unverified against the FST.

use crate::{Result, SpeechError};

/// Largest integer part the normalizer will spell out.
const MAX_INT_DIGITS: usize = 12;
/// `chunk_text_english` default.
pub const DEFAULT_MAX_CHARS: usize = 130;

/// Rejects text containing letters outside the Latin script.
pub fn ensure_english(text: &str) -> Result<()> {
    for c in text.chars() {
        if c.is_alphabetic() && !is_latin(c) {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "text contains {c:?}, which is outside the Latin script; this port covers English only"
                ),
            });
        }
    }
    Ok(())
}

fn is_latin(c: char) -> bool {
    matches!(c as u32, 0x41..=0x5A | 0x61..=0x7A | 0xAA | 0xBA | 0xC0..=0x24F | 0x1E00..=0x1EFF)
}

/// Python's `str.isspace` set, which is what `re`'s `\s` matches for text.
fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

fn is_ascii_stop(c: char) -> bool {
    matches!(c, ';' | ':' | ',' | '.' | '!' | '?')
}

fn is_wide_stop(c: char) -> bool {
    matches!(
        c,
        '\u{FF1B}' | '\u{FF1A}' | '\u{FF0C}' | '\u{3002}' | '\u{FF01}' | '\u{FF1F}'
    )
}

/// `re.split(r"(?<=[;:,.!?])\s+|(?<=[FW])", text)` with `FW` the full-width
/// semicolon, colon, comma, full stop, exclamation and question marks
/// (U+FF1B, U+FF1A, U+FF0C, U+3002, U+FF01, U+FF1F): whitespace after an
/// ASCII stop is consumed, a full-width stop splits with a zero-width match, and
/// a trailing empty piece is kept, like Python.
fn split_sentences(text: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut start = 0usize;
    let mut iter = text.char_indices().peekable();
    while let Some((idx, c)) = iter.next() {
        let end = idx + c.len_utf8();
        if is_wide_stop(c) {
            pieces.push(&text[start..end]);
            start = end;
        } else if is_ascii_stop(c) && iter.peek().is_some_and(|&(_, n)| py_space(n)) {
            pieces.push(&text[start..end]);
            let mut resume = end;
            while let Some(&(i, n)) = iter.peek() {
                if py_space(n) {
                    resume = i + n.len_utf8();
                    iter.next();
                } else {
                    break;
                }
            }
            start = resume;
        }
    }
    pieces.push(&text[start..]);
    pieces
}

/// `chunk_text_english`: greedy sentence packing under a UTF-8 byte budget.
pub fn chunk_english(text: &str, max_chars: usize) -> Vec<String> {
    // A sentence ending in a one-byte character gets a joining space.
    let spaced = |s: &str| -> String {
        match s.chars().next_back() {
            Some(last) if last.len_utf8() == 1 => format!("{s} "),
            _ => s.to_string(),
        }
    };
    let mut chunks = Vec::new();
    let mut current = String::new();
    for sentence in split_sentences(text) {
        if current.len() + sentence.len() <= max_chars {
            current.push_str(&spaced(sentence));
        } else {
            if !current.is_empty() {
                chunks.push(current.trim().to_string());
            }
            current = spaced(sentence);
        }
    }
    if !current.is_empty() {
        chunks.push(current.trim().to_string());
    }
    chunks
}

// ------------------------------------------------------------ normalizer

const ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];
const SCALES: [(u64, &str); 4] = [
    (1_000_000_000, "billion"),
    (1_000_000, "million"),
    (1_000, "thousand"),
    (1, ""),
];

fn below_thousand(n: u64, out: &mut Vec<&'static str>) {
    if n >= 100 {
        out.push(ONES[(n / 100) as usize]);
        out.push("hundred");
    }
    let r = n % 100;
    if r >= 20 {
        out.push(TENS[(r / 10) as usize]);
        if r % 10 != 0 {
            out.push(ONES[(r % 10) as usize]);
        }
    } else if r > 0 {
        out.push(ONES[r as usize]);
    }
}

/// Cardinal number words, NeMo style (no "and").
pub fn cardinal_words(n: u64) -> String {
    if n == 0 {
        return "zero".to_string();
    }
    let mut words: Vec<&'static str> = Vec::new();
    let mut rest = n;
    for (scale, name) in SCALES {
        let part = rest / scale;
        rest %= scale;
        if part > 0 {
            below_thousand(part, &mut words);
            if !name.is_empty() {
                words.push(name);
            }
        }
    }
    words.join(" ")
}

/// Ordinal number words: the cardinal with its last word inflected.
pub fn ordinal_words(n: u64) -> String {
    let card = cardinal_words(n);
    let (head, last) = match card.rfind(' ') {
        Some(i) => (&card[..=i], &card[i + 1..]),
        None => ("", card.as_str()),
    };
    let ord = match last {
        "one" => "first".to_string(),
        "two" => "second".to_string(),
        "three" => "third".to_string(),
        "five" => "fifth".to_string(),
        "eight" => "eighth".to_string(),
        "nine" => "ninth".to_string(),
        "twelve" => "twelfth".to_string(),
        w if w.ends_with('y') => format!("{}ieth", &w[..w.len() - 1]),
        w => format!("{w}th"),
    };
    format!("{head}{ord}")
}

/// A plain number token: optional comma grouping, optional fraction.
struct Plain {
    int: u64,
    frac: Option<String>,
}

/// `\d+` or `\d{1,3}(,\d{3})+`, then optionally `.\d+`; no leading zeros
/// unless the integer part is exactly "0".
fn parse_plain(run: &str) -> Option<Plain> {
    let (int_part, frac) = match run.split_once('.') {
        Some((i, f))
            if !f.is_empty()
                && f.bytes().all(|b| b.is_ascii_digit())
                && f.len() <= MAX_INT_DIGITS =>
        {
            (i, Some(f.to_string()))
        }
        Some(_) => return None,
        None => (run, None),
    };
    let groups: Vec<&str> = int_part.split(',').collect();
    let valid = if groups.len() == 1 {
        !int_part.is_empty() && int_part.bytes().all(|b| b.is_ascii_digit())
    } else {
        (1..=3).contains(&groups[0].len())
            && groups[0].bytes().all(|b| b.is_ascii_digit())
            && groups[1..]
                .iter()
                .all(|g| g.len() == 3 && g.bytes().all(|b| b.is_ascii_digit()))
    };
    if !valid {
        return None;
    }
    let digits: String = int_part.chars().filter(|c| *c != ',').collect();
    if digits.len() > MAX_INT_DIGITS || (digits.len() > 1 && digits.starts_with('0')) {
        return None;
    }
    Some(Plain {
        int: digits.parse().ok()?,
        frac,
    })
}

fn plain_words(p: &Plain) -> String {
    let mut out = cardinal_words(p.int);
    if let Some(f) = &p.frac {
        out.push_str(" point");
        for b in f.bytes() {
            out.push(' ');
            out.push_str(ONES[(b - b'0') as usize]);
        }
    }
    out
}

fn money_words(p: &Plain) -> Option<String> {
    let cents = match &p.frac {
        None => 0,
        Some(f) if f.len() == 2 => f.parse::<u64>().ok()?,
        Some(_) => return None,
    };
    let mut parts = Vec::new();
    if p.int > 0 || cents == 0 {
        parts.push(format!(
            "{} {}",
            cardinal_words(p.int),
            if p.int == 1 { "dollar" } else { "dollars" }
        ));
    }
    if cents > 0 {
        parts.push(format!(
            "{} {}",
            cardinal_words(cents),
            if cents == 1 { "cent" } else { "cents" }
        ));
    }
    Some(parts.join(" "))
}

/// Longest run of digits joined by `, . : / -` when each joiner is followed by a digit.
fn number_run(chars: &[char], start: usize) -> usize {
    let mut end = start;
    while end < chars.len() {
        let c = chars[end];
        // A digit, or a joiner that is itself followed by a digit.
        if c.is_ascii_digit()
            || (matches!(c, ',' | '.' | ':' | '/' | '-')
                && chars.get(end + 1).is_some_and(|n| n.is_ascii_digit()))
        {
            end += 1;
        } else {
            break;
        }
    }
    end
}

/// Conservative English normalization; see the module docs for its scope.
pub fn normalize_english(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len() + 16);
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let prev = if i == 0 { None } else { Some(chars[i - 1]) };
        // A number starts at a boundary: not glued to a word, to a symbol that
        // makes it a code ("#5", "US$5"), or to a hyphenated token ("COVID-19").
        let hyphen_joined = prev == Some('-') && i >= 2 && chars[i - 2].is_alphanumeric();
        let boundary =
            prev.is_none_or(|p| !p.is_alphanumeric() && !"$#@/\\_".contains(p)) && !hyphen_joined;
        let (sign_or_money, digit_at) = match c {
            '$' | '-' if boundary && chars.get(i + 1).is_some_and(|n| n.is_ascii_digit()) => {
                (Some(c), i + 1)
            }
            _ => (None, i),
        };
        let starts_number = chars.get(digit_at).is_some_and(|d| d.is_ascii_digit())
            && (sign_or_money.is_some() || boundary)
            && !(sign_or_money == Some('-')
                && prev.is_some_and(|p| !py_space(p) && p != '(' && p != '"' && p != '\''));
        if !starts_number {
            if c.is_ascii_digit() {
                // A digit glued to a word or code ("v1.2.3", "COVID-19"): keep the
                // whole token, so its tail is not mistaken for a new number.
                let mut end = number_run(&chars, i);
                while chars.get(end).is_some_and(|c| c.is_alphanumeric()) {
                    end += 1;
                }
                out.extend(&chars[i..end]);
                i = end;
            } else {
                out.push(c);
                i += 1;
            }
            continue;
        }
        let run_end = number_run(&chars, digit_at);
        let run: String = chars[digit_at..run_end].iter().collect();
        let next = chars.get(run_end).copied();
        let verbatim_end = {
            // Digits glued to letters ("3D", "5k") or odd joiners stay as written.
            let mut e = run_end;
            while chars.get(e).is_some_and(|c| c.is_alphanumeric()) {
                e += 1;
            }
            e
        };
        let plain = parse_plain(&run);
        let replaced: Option<(String, usize)> = match (sign_or_money, &plain) {
            (Some('$'), Some(p)) if !next.is_some_and(|n| n.is_alphanumeric()) => {
                money_words(p).map(|w| (w, run_end))
            }
            (Some('-'), Some(p)) if !next.is_some_and(|n| n.is_alphanumeric() || n == '%') => {
                Some((format!("minus {}", plain_words(p)), run_end))
            }
            (None, Some(p)) => match next {
                Some('%') => Some((format!("{} percent", plain_words(p)), run_end + 1)),
                Some(a) if a.is_ascii_alphabetic() && p.frac.is_none() => {
                    let suffix: String = chars[run_end..verbatim_end]
                        .iter()
                        .collect::<String>()
                        .to_ascii_lowercase();
                    if matches!(suffix.as_str(), "st" | "nd" | "rd" | "th")
                        && chars[digit_at..run_end].iter().all(|c| *c != ',')
                    {
                        Some((ordinal_words(p.int), verbatim_end))
                    } else {
                        None
                    }
                }
                Some(a) if a.is_alphanumeric() => None,
                _ => Some((plain_words(p), run_end)),
            },
            _ => None,
        };
        match replaced {
            Some((words, end)) => {
                out.push_str(&words);
                i = end;
            }
            None => {
                // Pass the whole token through untouched.
                let end = verbatim_end.max(run_end).max(i + 1);
                out.extend(&chars[i..end]);
                i = end;
            }
        }
    }
    // Collapse whitespace runs (including newlines) to single spaces.
    let mut collapsed = String::with_capacity(out.len());
    let mut last_space = false;
    for ch in out.trim().chars() {
        if py_space(ch) {
            if !last_space {
                collapsed.push(' ');
            }
            last_space = true;
        } else {
            collapsed.push(ch);
            last_space = false;
        }
    }
    collapsed
}
