//! The kitten_tts text front end (`kitten_tts.py` text helpers plus
//! `preprocess.py`).
//!
//! Port of `basic_english_tokenize`, `ensure_punctuation`, `chunk_text`,
//! `TextCleaner`, and the `TextPreprocessor` default pipeline at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/kitten_tts).
//! Every regex pass is a left-to-right scanner with sequential `re.sub`
//! semantics: a pass never rescans its own replacements, and matches start
//! at the earliest position the pattern accepts.
//!
//! The espeak grapheme-to-phoneme step is NOT ported (the reference calls
//! `phonemizer`'s EspeakBackend); the caller supplies the phonemized string
//! and everything from there on (tokenize, cleaner, ids) runs here. See the
//! family README for the intake contract.

use std::collections::HashMap;

use crate::{Result, SpeechError};

// ---------------------------------------------------------------------------
// TextCleaner symbol table
// ---------------------------------------------------------------------------

/// `_pad` (`TextCleaner.__init__`).
const PAD: &str = "$";
/// `_punctuation`; the exact characters the reference lists, including
/// the em dash, ellipsis, guillemets, and three ASCII double quotes.
const PUNCTUATION: &str = ";:,.!?\u{a1}\u{bf}\u{2014}\u{2026}\u{22}\u{ab}\u{bb}\u{22}\u{22} ";
/// `_letters`.
const LETTERS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
/// `_letters_ipa`: the exact reference string as unicode escapes so the
/// source stays ASCII. The Python dict built over the four strings keeps
/// the LAST index for a repeated character (the apostrophe appears twice).
const LETTERS_IPA: &str = "\u{251}\u{250}\u{252}\u{e6}\u{253}\u{299}\u{3b2}\u{254}\u{255}\u{e7}\u{257}\u{256}\u{f0}\u{2a4}\u{259}\u{258}\u{25a}\u{25b}\u{25c}\u{25d}\u{25e}\u{25f}\u{284}\u{261}\u{260}\u{262}\u{29b}\u{266}\u{267}\u{127}\u{265}\u{29c}\u{268}\u{26a}\u{29d}\u{26d}\u{26c}\u{26b}\u{26e}\u{29f}\u{271}\u{26f}\u{270}\u{14b}\u{273}\u{272}\u{274}\u{f8}\u{275}\u{278}\u{3b8}\u{153}\u{276}\u{298}\u{279}\u{27a}\u{27e}\u{27b}\u{280}\u{281}\u{27d}\u{282}\u{283}\u{288}\u{2a7}\u{289}\u{28a}\u{28b}\u{2c71}\u{28c}\u{263}\u{264}\u{28d}\u{3c7}\u{28e}\u{28f}\u{291}\u{290}\u{292}\u{294}\u{2a1}\u{295}\u{2a2}\u{1c0}\u{1c1}\u{1c2}\u{1c3}\u{2c8}\u{2cc}\u{2d0}\u{2d1}\u{2bc}\u{2b4}\u{2b0}\u{2b1}\u{2b2}\u{2b7}\u{2e0}\u{2e4}\u{2de}\u{2193}\u{2191}\u{2192}\u{2197}\u{2198}'\u{329}'\u{1d7b}";

fn reference_symbols() -> Vec<String> {
    let mut symbols: Vec<String> = Vec::new();
    symbols.push(PAD.to_string());
    symbols.extend(PUNCTUATION.chars().map(|c| c.to_string()));
    symbols.extend(LETTERS.chars().map(|c| c.to_string()));
    symbols.extend(LETTERS_IPA.chars().map(|c| c.to_string()));
    symbols
}

/// TextCleaner's `word_index_dictionary`: symbol -> id with the Python
/// dict's later-wins duplicate rule.
pub fn symbol_table() -> HashMap<String, u32> {
    let mut map: HashMap<String, u32> = HashMap::new();
    for (index, symbol) in reference_symbols().into_iter().enumerate() {
        map.insert(symbol, index as u32);
    }
    map
}

/// The `TextCleaner`: maps each character of the phonemized text to its
/// symbol id, skipping characters outside the table.
pub struct TextCleaner {
    map: HashMap<String, u32>,
}

impl Default for TextCleaner {
    fn default() -> Self {
        Self::new()
    }
}

impl TextCleaner {
    pub fn new() -> Self {
        Self {
            map: symbol_table(),
        }
    }

    /// The symbol list in reference order (first-insertion order with
    /// later-wins ids), for contract tests.
    pub fn symbols(&self) -> Vec<String> {
        let mut seen: Vec<String> = Vec::new();
        for symbol in reference_symbols() {
            if !seen.contains(&symbol) {
                seen.push(symbol);
            }
        }
        seen
    }

    pub fn id_of(&self, symbol: &str) -> Option<u32> {
        self.map.get(symbol).copied()
    }

    /// `TextCleaner.__call__`: per-character ids, unknowns dropped.
    pub fn clean(&self, text: &str) -> Vec<u32> {
        text.chars()
            .filter_map(|c| self.map.get(&c.to_string()).copied())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tokenization helpers
// ---------------------------------------------------------------------------

fn is_word_char(c: char) -> bool {
    // Python `\w`: letters, digits, underscore (unicode-aware).
    c.is_alphanumeric() || c == '_'
}

/// `basic_english_tokenize`: `\w+|[^\w\s]` over the phonemized string.
pub fn basic_english_tokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if is_word_char(chars[i]) {
            let start = i;
            while i < chars.len() && is_word_char(chars[i]) {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
        } else if !chars[i].is_whitespace() {
            tokens.push(chars[i].to_string());
            i += 1;
        } else {
            i += 1;
        }
    }
    tokens
}

/// `ensure_punctuation`: strip, then append a comma when the text does not
/// end in sentence punctuation.
pub fn ensure_punctuation(text: &str) -> String {
    let text = text.trim();
    if text.is_empty() {
        return text.to_string();
    }
    let last = text.chars().last().expect("nonempty");
    if ".!?,;:".contains(last) {
        text.to_string()
    } else {
        format!("{text},")
    }
}

/// `chunk_text`: split on `[.!?]+`, strip each sentence, then wrap
/// sentences longer than `max_len` on word boundaries.
pub fn chunk_text(text: &str, max_len: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    // re.split(r"[.!?]+", text): split positions are maximal separator
    // runs; empty pieces are skipped after the strip below.
    let mut piece = String::new();
    let mut separator_run = false;
    for c in text.chars() {
        if c == '.' || c == '!' || c == '?' {
            separator_run = true;
        } else {
            if separator_run && !piece.is_empty() {
                chunks.push(std::mem::take(&mut piece));
            }
            separator_run = false;
            piece.push(c);
        }
    }
    if !piece.is_empty() {
        chunks.push(piece);
    }

    let mut out = Vec::new();
    for sentence in chunks {
        let sentence = sentence.trim();
        if sentence.is_empty() {
            continue;
        }
        if sentence.chars().count() <= max_len {
            out.push(ensure_punctuation(sentence));
        } else {
            let mut temp = String::new();
            for word in sentence.split(' ') {
                let candidate_len = temp.chars().count() + word.chars().count() + 1;
                if temp.is_empty() || candidate_len <= max_len {
                    if !temp.is_empty() {
                        temp.push(' ');
                    }
                    temp.push_str(word);
                } else {
                    out.push(ensure_punctuation(temp.trim()));
                    temp = word.to_string();
                }
            }
            if !temp.is_empty() {
                out.push(ensure_punctuation(temp.trim()));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Number words
// ---------------------------------------------------------------------------

const ONES: [&str; 20] = [
    "",
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
// Covers the full i64 range (up to 9 quintillion). number_to_words takes
// an i64, so no value can exceed this table; before the last two names
// were added, anything >= 10^15 was silently dropped.
const SCALE: [&str; 7] = [
    "",
    "thousand",
    "million",
    "billion",
    "trillion",
    "quadrillion",
    "quintillion",
];

const ORDINAL_EXCEPTIONS: [(&str, &str); 10] = [
    ("one", "first"),
    ("two", "second"),
    ("three", "third"),
    ("four", "fourth"),
    ("five", "fifth"),
    ("six", "sixth"),
    ("seven", "seventh"),
    ("eight", "eighth"),
    ("nine", "ninth"),
    ("twelve", "twelfth"),
];

/// `number_to_words`: integers to English words with the reference's
/// "X hundred" reading for exact hundreds between 100 and 9999.
pub fn number_to_words(n: i64) -> String {
    if n == 0 {
        return "zero".to_string();
    }
    if n < 0 {
        return format!("negative {}", number_to_words(-n));
    }
    let n = n as u64;
    // X00-X999 read as "X hundred", excluding exact multiples of 1000.
    if (100..=9999).contains(&n) && n % 100 == 0 && n % 1000 != 0 {
        let hundreds = n / 100;
        if hundreds < 20 {
            return format!("{} hundred", ONES[hundreds as usize]);
        }
    }
    let mut parts: Vec<String> = Vec::new();
    let mut value = n;
    for scale in SCALE {
        let chunk = value % 1000;
        if chunk != 0 {
            let chunk_words = three_digits_to_words(chunk);
            if scale.is_empty() {
                parts.push(chunk_words);
            } else {
                parts.push(format!("{chunk_words} {scale}"));
            }
        }
        value /= 1000;
        if value == 0 {
            break;
        }
    }
    parts.reverse();
    parts.join(" ")
}

/// `_three_digits_to_words`: 0-999.
fn three_digits_to_words(n: u64) -> String {
    if n == 0 {
        return String::new();
    }
    let mut parts: Vec<String> = Vec::new();
    let hundreds = n / 100;
    let remainder = n % 100;
    if hundreds != 0 {
        parts.push(format!("{} hundred", ONES[hundreds as usize]));
    }
    if remainder < 20 {
        if remainder != 0 {
            parts.push(ONES[remainder as usize].to_string());
        }
    } else {
        let tens_word = TENS[(remainder / 10) as usize];
        let ones_word = ONES[(remainder % 10) as usize];
        parts.push(if ones_word.is_empty() {
            tens_word.to_string()
        } else {
            format!("{tens_word}-{ones_word}")
        });
    }
    parts.join(" ")
}

/// `float_to_words`: numeric string to words, decimal digits read
/// individually so trailing zeros survive.
pub fn float_to_words(value: &str, decimal_sep: &str) -> String {
    let mut text = value;
    let negative = text.starts_with('-');
    if negative {
        text = &text[1..];
    }
    let result = if let Some((int_part, dec_part)) = text.split_once('.') {
        let int_words = if int_part.is_empty() {
            "zero".to_string()
        } else {
            number_to_words(int_part.parse::<i64>().unwrap_or(0))
        };
        let mut digit_map = vec!["zero"];
        digit_map.extend(&ONES[1..]);
        let dec_words = dec_part
            .chars()
            .filter_map(|d| d.to_digit(10).map(|d| digit_map[d as usize]))
            .collect::<Vec<_>>()
            .join(" ");
        format!("{int_words} {decimal_sep} {dec_words}")
    } else {
        number_to_words(text.parse::<i64>().unwrap_or(0))
    };
    if negative {
        format!("negative {result}")
    } else {
        result
    }
}

/// `_ordinal_suffix`: ordinal word for n ("twenty-first", "hundredth").
pub fn ordinal_suffix(n: i64) -> String {
    let word = number_to_words(n);
    let (prefix, last, joiner) = if let Some((prefix, last)) = word.rsplit_once('-') {
        (prefix.to_string(), last.to_string(), "-")
    } else if let Some((prefix, last)) = word.rsplit_once(' ') {
        (prefix.to_string(), last.to_string(), " ")
    } else {
        (String::new(), word.clone(), "")
    };
    let last_ord = if let Some((_, ordinal)) = ORDINAL_EXCEPTIONS.iter().find(|(b, _)| *b == last) {
        ordinal.to_string()
    } else if last.ends_with('t') {
        format!("{last}h")
    } else if last.ends_with('e') {
        format!("{}th", &last[..last.len() - 1])
    } else {
        format!("{last}th")
    };
    if prefix.is_empty() {
        last_ord
    } else {
        format!("{prefix}{joiner}{last_ord}")
    }
}

// ---------------------------------------------------------------------------
// Scanner framework
// ---------------------------------------------------------------------------

/// Sequential `re.sub` replacement: walk `text` left to right; at each
/// position let `matcher` either consume a span (returns the replacement
/// and the length in chars) or pass one character through.
fn scan_replace(
    text: &str,
    mut matcher: impl FnMut(&[char], usize) -> Option<(String, usize)>,
) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if let Some((replacement, len)) = matcher(&chars, i) {
            out.push_str(&replacement);
            i += len.max(1);
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// Python `\b` at `index` over `chars`.
fn word_boundary(chars: &[char], index: usize) -> bool {
    let before = index > 0 && is_word_char(chars[index - 1]);
    let after = index < chars.len() && is_word_char(chars[index]);
    before != after
}

fn ascii_lower(c: char) -> char {
    c.to_ascii_lowercase()
}

fn slice_eq_ignore_ascii_case(chars: &[char], index: usize, pattern: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    if index + p.len() > chars.len() {
        return false;
    }
    p.iter()
        .zip(&chars[index..])
        .all(|(a, b)| a.to_ascii_lowercase() == ascii_lower(*b))
}

fn slice_eq(chars: &[char], index: usize, pattern: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    index + p.len() <= chars.len() && p.iter().zip(&chars[index..]).all(|(a, b)| a == b)
}

fn digits_at(chars: &[char], index: usize, max: usize) -> usize {
    let mut n = 0;
    while n < max && index + n < chars.len() && chars[index + n].is_ascii_digit() {
        n += 1;
    }
    n
}

/// A `[\\d,]+` run (commas and digits, at least one character) starting at
/// `index`; returns the length.
fn number_comma_run(chars: &[char], index: usize) -> usize {
    let mut n = 0;
    while index + n < chars.len() && (chars[index + n].is_ascii_digit() || chars[index + n] == ',')
    {
        n += 1;
    }
    n
}

fn word_run(chars: &[char], index: usize) -> usize {
    let mut n = 0;
    while index + n < chars.len() && is_word_char(chars[index + n]) {
        n += 1;
    }
    n
}

fn non_space_run(chars: &[char], index: usize) -> usize {
    let mut n = 0;
    while index + n < chars.len() && !chars[index + n].is_whitespace() {
        n += 1;
    }
    n
}

fn space_run(chars: &[char], index: usize) -> usize {
    let mut n = 0;
    while index + n < chars.len() && chars[index + n].is_whitespace() {
        n += 1;
    }
    n
}

/// `(\w+)(suffix)` with backtracking: the largest tail of the word run
/// ending at or before `run_end` that is followed by `suffix`.
fn word_then_suffix(chars: &[char], index: usize, suffix: &str) -> Option<(usize, usize)> {
    let run_end = index + word_run(chars, index);
    if run_end == index {
        return None;
    }
    let mut j = run_end;
    while j > index {
        if slice_eq_ignore_ascii_case(chars, j, suffix) {
            let total = (j - index) + suffix.chars().count();
            return Some((total, j - index));
        }
        j -= 1;
    }
    None
}

// ---------------------------------------------------------------------------
// TextPreprocessor passes
// ---------------------------------------------------------------------------

fn remove_html_tags(text: &str) -> String {
    scan_replace(text, |chars, i| {
        if chars[i] == '<' {
            let mut j = i + 1;
            while j < chars.len() && chars[j] != '>' {
                j += 1;
            }
            if j < chars.len() && j > i + 1 {
                return Some((" ".to_string(), j + 1 - i));
            }
        }
        None
    })
}

fn remove_urls(text: &str) -> String {
    let replaced = scan_replace(text, |chars, i| {
        if slice_eq(chars, i, "http://")
            || slice_eq(chars, i, "https://")
            || slice_eq(chars, i, "www.")
        {
            let run = non_space_run(chars, i);
            if run > 0 {
                return Some((String::new(), run));
            }
        }
        None
    });
    replaced.trim().to_string()
}

fn remove_emails(text: &str) -> String {
    let replaced = scan_replace(text, |chars, i| {
        if !word_boundary(chars, i) {
            return None;
        }
        // [\w.+-]+ then '@' then [\w-]+ '.' [a-z]{2,} then \b
        let mut j = i;
        while j < chars.len()
            && (is_word_char(chars[j]) || chars[j] == '.' || chars[j] == '+' || chars[j] == '-')
        {
            j += 1;
        }
        if j == i || j >= chars.len() || chars[j] != '@' {
            return None;
        }
        j += 1;
        let domain_start = j;
        while j < chars.len() && (is_word_char(chars[j]) || chars[j] == '-') {
            j += 1;
        }
        if j == domain_start || j >= chars.len() || chars[j] != '.' {
            return None;
        }
        j += 1;
        let tld_start = j;
        while j < chars.len() && chars[j].is_ascii_alphabetic() {
            j += 1;
        }
        if j - tld_start < 2 || !word_boundary(chars, j) {
            return None;
        }
        Some((String::new(), j - i))
    });
    replaced.trim().to_string()
}

const CONTRACTIONS: [&str; 5] = ["can't", "won't", "shan't", "ain't", "let's"];

fn expand_contractions(text: &str) -> String {
    let mut text = text.to_string();
    // Literal contractions first, in reference dict order.
    for literal in CONTRACTIONS {
        let replacement = match literal {
            "can't" => "cannot",
            "won't" => "will not",
            "shan't" => "shall not",
            "ain't" => "is not",
            _ => "let us",
        };
        text = scan_replace(&text, |chars, i| {
            if word_boundary(chars, i)
                && slice_eq_ignore_ascii_case(chars, i, literal)
                && word_boundary(chars, i + literal.chars().count())
            {
                return Some((replacement.to_string(), literal.chars().count()));
            }
            None
        });
    }
    // (\w+)n't -> "\1 not"
    text = scan_replace(&text, |chars, i| {
        if let Some((len, word_len)) = word_then_suffix(chars, i, "n't") {
            let word: String = chars[i..i + word_len].iter().collect();
            return Some((format!("{word} not"), len));
        }
        None
    });
    for (suffix, word) in [
        ("'re", "are"),
        ("'ve", "have"),
        ("'ll", "will"),
        ("'d", "would"),
        ("'m", "am"),
    ] {
        text = scan_replace(&text, |chars, i| {
            if let Some((len, word_len)) = word_then_suffix(chars, i, suffix) {
                let stem: String = chars[i..i + word_len].iter().collect();
                return Some((format!("{stem} {word}"), len));
            }
            None
        });
    }
    // it's -> "it is" (last in the reference dict).
    scan_replace(&text, |chars, i| {
        if word_boundary(chars, i)
            && slice_eq_ignore_ascii_case(chars, i, "it's")
            && word_boundary(chars, i + 4)
        {
            return Some(("it is".to_string(), 4));
        }
        None
    })
}

fn expand_ip_addresses(text: &str) -> String {
    scan_replace(text, |chars, i| {
        if !word_boundary(chars, i) {
            return None;
        }
        let mut groups = [0usize; 4];
        let mut j = i;
        #[allow(
            clippy::needless_range_loop,
            reason = "the four octet positions are protocol-indexed"
        )]
        for g in 0..4 {
            let len = digits_at(chars, j, 3);
            if len == 0 {
                return None;
            }
            groups[g] = j;
            j += len;
            if g < 3 {
                if j >= chars.len() || chars[j] != '.' {
                    return None;
                }
                j += 1;
            }
        }
        if !word_boundary(chars, j) {
            return None;
        }
        let words = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        ];
        let octets: Vec<String> = groups
            .iter()
            .map(|&g| {
                chars[g..]
                    .iter()
                    .take_while(|c| c.is_ascii_digit())
                    .map(|c| words[c.to_digit(10).expect("digit") as usize])
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect();
        Some((octets.join(" dot "), j - i))
    })
}

fn normalize_leading_decimals(text: &str) -> String {
    // (-)\.(\d) -> "-0.\2", then "\.(\d)" -> "0.\1".
    let with_minus = scan_replace(text, |chars, i| {
        if chars[i] == '-'
            && i > 0
            && !chars[i - 1].is_ascii_digit()
            && i + 2 < chars.len()
            && chars[i + 1] == '.'
            && chars[i + 2].is_ascii_digit()
        {
            return Some((format!("-0.{}", chars[i + 2]), 3));
        }
        None
    });
    scan_replace(&with_minus, |chars, i| {
        if chars[i] == '.'
            && (i == 0 || !chars[i - 1].is_ascii_digit())
            && i + 1 < chars.len()
            && chars[i + 1].is_ascii_digit()
        {
            // The reference sub is `(?<!\d)\.([\d])` -> "0.\1": one digit.
            return Some((format!("0.{}", chars[i + 1]), 2));
        }
        None
    })
}

const CURRENCY_SYMBOLS: [(&str, &str); 7] = [
    ("$", "dollar"),
    ("\u{20ac}", "euro"),
    ("\u{a3}", "pound"),
    ("\u{a5}", "yen"),
    ("\u{20b9}", "rupee"),
    ("\u{20a9}", "won"),
    ("\u{20bf}", "bitcoin"),
];

fn expand_currency(text: &str) -> String {
    scan_replace(text, |chars, i| {
        let symbol = CURRENCY_SYMBOLS
            .iter()
            .find(|(s, _)| s.starts_with(chars[i]));
        let (_, unit) = match symbol {
            Some(entry) => entry,
            None => return None,
        };
        let mut j = i + 1;
        j += space_run(chars, j);
        // [\d,]+ optionally followed by .digits
        let num_start = j;
        j += number_comma_run(chars, j);
        if j == num_start {
            return None;
        }
        if j < chars.len()
            && chars[j] == '.'
            && j + 1 < chars.len()
            && chars[j + 1].is_ascii_digit()
        {
            j += 1;
            j += digits_at(chars, j, usize::MAX);
        }
        let raw_number: String = chars[num_start..j].iter().collect();
        // \s*([KMBT])?(?![a-zA-Z\d]) with backtracking: the whitespace is
        // only consumed when the scale letter or the trailing lookahead
        // can still match after it.
        let spaces = space_run(chars, j);
        let mut scale: Option<char> = None;
        let mut matched_end: Option<usize> = None;
        let lookahead_ok = |pos: usize| pos >= chars.len() || !chars[pos].is_ascii_alphanumeric();
        for take in (0..=spaces).rev() {
            let p = j + take;
            if p < chars.len() && matches!(chars[p], 'K' | 'M' | 'B' | 'T') && lookahead_ok(p + 1) {
                scale = Some(chars[p]);
                matched_end = Some(p + 1);
                break;
            }
            if lookahead_ok(p) {
                matched_end = Some(p);
                break;
            }
        }
        let matched_end = matched_end?;
        j = matched_end;
        let raw_clean: String = raw_number.chars().filter(|c| *c != ',').collect();
        let scale_map = [
            ("K", "thousand"),
            ("M", "million"),
            ("B", "billion"),
            ("T", "trillion"),
        ];
        let expansion = if let Some(scale_char) = scale {
            let scale_word = scale_map
                .iter()
                .find(|(s, _)| s.starts_with(scale_char))
                .map(|(_, w)| *w)
                .unwrap_or("");
            let num = if raw_clean.contains('.') {
                float_to_words(&raw_clean, "point")
            } else {
                number_to_words(raw_clean.parse::<i64>().unwrap_or(0))
            };
            let plural = if unit.is_empty() { "" } else { "s" };
            format!("{num} {scale_word} {unit}{plural}")
                .trim()
                .to_string()
        } else if raw_clean.contains('.') {
            let (int_part, dec_part) = raw_clean.split_once('.').expect("dot");
            let dec_val: i64 = dec_part
                .chars()
                .take(2)
                .collect::<String>()
                .parse::<i64>()
                .unwrap_or(0)
                * if dec_part.chars().count() == 1 { 10 } else { 1 };
            let int_words = number_to_words(int_part.parse::<i64>().unwrap_or(0));
            let mut result = if unit.is_empty() {
                int_words
            } else {
                format!("{int_words} {unit}s")
            };
            if dec_val != 0 {
                let plural = if dec_val != 1 { "s" } else { "" };
                result.push_str(&format!(" and {} cent{plural}", number_to_words(dec_val)));
            }
            result
        } else {
            let val = raw_clean.parse::<i64>().unwrap_or(0);
            let words = number_to_words(val);
            if unit.is_empty() {
                words
            } else if val != 1 {
                format!("{words} {unit}s")
            } else {
                format!("{words} {unit}")
            }
        };
        Some((expansion, j - i))
    })
}

fn expand_percentages(text: &str) -> String {
    scan_replace(text, |chars, i| {
        // (-?[\d,]+(?:\.\d+)?)\s*%
        let mut j = i;
        if chars[j] == '-' {
            j += 1;
        }
        let num_start = j;
        j += number_comma_run(chars, j);
        if j == num_start {
            return None;
        }
        if j < chars.len()
            && chars[j] == '.'
            && j + 1 < chars.len()
            && chars[j + 1].is_ascii_digit()
        {
            j += 1;
            j += digits_at(chars, j, usize::MAX);
        }
        let raw: String = chars[i..j].iter().collect();
        let mut k = j;
        k += space_run(chars, k);
        if k >= chars.len() || chars[k] != '%' {
            return None;
        }
        let raw_clean: String = raw.chars().filter(|c| *c != ',').collect();
        let words = if raw_clean.contains('.') {
            format!("{} percent", float_to_words(&raw_clean, "point"))
        } else {
            format!(
                "{} percent",
                number_to_words(raw_clean.parse::<i64>().unwrap_or(0))
            )
        };
        Some((words, k + 1 - i))
    })
}

fn expand_scientific_notation(text: &str) -> String {
    scan_replace(text, |chars, i| {
        // (?<![a-zA-Z\d]) before, so position 0 or a non-alnum neighbor.
        if i > 0 && (chars[i - 1].is_ascii_alphanumeric()) {
            return None;
        }
        let mut j = i;
        if chars[j] == '-' {
            j += 1;
        }
        let coeff_start = j;
        j += digits_at(chars, j, usize::MAX);
        if j == coeff_start {
            return None;
        }
        if j < chars.len()
            && chars[j] == '.'
            && j + 1 < chars.len()
            && chars[j + 1].is_ascii_digit()
        {
            j += 1;
            j += digits_at(chars, j, usize::MAX);
        }
        let coeff: String = chars[i..j].iter().collect();
        if j >= chars.len() || (chars[j] != 'e' && chars[j] != 'E') {
            return None;
        }
        j += 1;
        let exp_start = j;
        if j < chars.len() && (chars[j] == '+' || chars[j] == '-') {
            j += 1;
        }
        let digits = digits_at(chars, j, usize::MAX);
        if digits == 0 {
            return None;
        }
        j += digits;
        if j < chars.len() && chars[j].is_ascii_alphanumeric() {
            return None;
        }
        let exp_str: String = chars[exp_start..j].iter().collect();
        let exp: i64 = exp_str.parse().unwrap_or(0);
        let coeff_words = if coeff.contains('.') {
            float_to_words(&coeff, "point")
        } else {
            number_to_words(coeff.parse::<i64>().unwrap_or(0))
        };
        let sign = if exp < 0 { "negative " } else { "" };
        Some((
            format!(
                "{coeff_words} times ten to the {sign}{}",
                number_to_words(exp.abs())
            ),
            j - i,
        ))
    })
}

fn expand_time(text: &str) -> String {
    scan_replace(text, |chars, i| {
        if !word_boundary(chars, i) {
            return None;
        }
        let h_len = digits_at(chars, i, 2);
        if h_len == 0 {
            return None;
        }
        let mut j = i + h_len;
        if j >= chars.len() || chars[j] != ':' {
            return None;
        }
        j += 1;
        let m_len = digits_at(chars, j, 2);
        if m_len != 2 {
            return None;
        }
        let h: i64 = chars[i..i + h_len]
            .iter()
            .collect::<String>()
            .parse()
            .unwrap_or(0);
        let mins: i64 = chars[j..j + 2]
            .iter()
            .collect::<String>()
            .parse()
            .unwrap_or(0);
        j += 2;
        // Optional seconds group, parsed but dropped by the replacement.
        let mut j2 = j;
        if j2 + 2 < chars.len() && chars[j2] == ':' && chars[j2 + 1].is_ascii_digit() {
            j2 += 1;
            let s_len = digits_at(chars, j2, 2);
            if s_len == 2 {
                j2 += 2;
            } else {
                j2 = j;
            }
        } else {
            j2 = j;
        }
        // \s*(am|pm)? then \b
        let mut k = j2;
        k += space_run(chars, k);
        let mut suffix = "";
        if slice_eq_ignore_ascii_case(chars, k, "am") {
            suffix = "am";
            k += 2;
        } else if slice_eq_ignore_ascii_case(chars, k, "pm") {
            suffix = "pm";
            k += 2;
        }
        if !word_boundary(chars, k) {
            return None;
        }
        let h_words = number_to_words(h);
        let suffix_text = if suffix.is_empty() {
            String::new()
        } else {
            format!(" {suffix}")
        };
        let replacement = if mins == 0 {
            if suffix.is_empty() {
                format!("{h_words} hundred")
            } else {
                format!("{h_words}{suffix_text}")
            }
        } else if mins < 10 {
            format!("{h_words} oh {}{suffix_text}", number_to_words(mins))
        } else {
            format!("{h_words} {}{suffix_text}", number_to_words(mins))
        };
        Some((replacement, k - i))
    })
}

fn expand_ordinals(text: &str) -> String {
    scan_replace(text, |chars, i| {
        if !word_boundary(chars, i) {
            return None;
        }
        let d_len = digits_at(chars, i, usize::MAX);
        if d_len == 0 {
            return None;
        }
        let mut j = i + d_len;
        if !slice_eq_ignore_ascii_case(chars, j, "st")
            && !slice_eq_ignore_ascii_case(chars, j, "nd")
            && !slice_eq_ignore_ascii_case(chars, j, "rd")
            && !slice_eq_ignore_ascii_case(chars, j, "th")
        {
            return None;
        }
        j += 2;
        if !word_boundary(chars, j) {
            return None;
        }
        let value: i64 = chars[i..i + d_len]
            .iter()
            .collect::<String>()
            .parse()
            .unwrap_or(0);
        Some((ordinal_suffix(value), j - i))
    })
}

const UNIT_MAP: [(&str, &str); 20] = [
    ("km", "kilometers"),
    ("kg", "kilograms"),
    ("mg", "milligrams"),
    ("ml", "milliliters"),
    ("gb", "gigabytes"),
    ("mb", "megabytes"),
    ("kb", "kilobytes"),
    ("tb", "terabytes"),
    ("hz", "hertz"),
    ("khz", "kilohertz"),
    ("mhz", "megahertz"),
    ("ghz", "gigahertz"),
    ("mph", "miles per hour"),
    ("kph", "kilometers per hour"),
    ("\u{b0}c", "degrees Celsius"),
    ("c\u{b0}", "degrees Celsius"),
    ("\u{b0}f", "degrees Fahrenheit"),
    ("f\u{b0}", "degrees Fahrenheit"),
    ("ms", "milliseconds"),
    ("ns", "nanoseconds"),
];

fn expand_units(text: &str) -> String {
    scan_replace(text, |chars, i| {
        let d_len = digits_at(chars, i, usize::MAX);
        if d_len == 0 {
            return None;
        }
        let mut j = i + d_len;
        if j < chars.len()
            && chars[j] == '.'
            && j + 1 < chars.len()
            && chars[j + 1].is_ascii_digit()
        {
            j += 1;
            j += digits_at(chars, j, usize::MAX);
        }
        let raw: String = chars[i..j].iter().collect();
        // \s*(unit)\b with backtracking over the whitespace run.
        let spaces = space_run(chars, j);
        let mut found: Option<(&str, usize)> = None;
        for take in (0..=spaces).rev() {
            let p = j + take;
            if let Some((u, _)) = UNIT_MAP
                .iter()
                .find(|(u, _)| slice_eq_ignore_ascii_case(chars, p, u))
            {
                let end = p + u.chars().count();
                if word_boundary(chars, end) {
                    found = Some((u, end));
                    break;
                }
            }
        }
        let (unit, after) = found?;
        let unit_lower: String = unit.chars().map(ascii_lower).collect();
        let expanded = UNIT_MAP
            .iter()
            .find(|(u, _)| *u == unit_lower)
            .map(|(_, w)| *w)
            .unwrap_or("");
        let words = if raw.contains('.') {
            float_to_words(&raw, "point")
        } else {
            number_to_words(raw.parse::<i64>().unwrap_or(0))
        };
        Some((format!("{words} {expanded}"), after - i))
    })
}

fn expand_scale_suffixes(text: &str) -> String {
    scan_replace(text, |chars, i| {
        // (?<![a-zA-Z])(\d+(?:\.\d+)?)\s*([KMBT])(?![a-zA-Z\d])
        if i > 0 && chars[i - 1].is_ascii_alphabetic() {
            return None;
        }
        let d_len = digits_at(chars, i, usize::MAX);
        if d_len == 0 {
            return None;
        }
        let mut j = i + d_len;
        if j < chars.len()
            && chars[j] == '.'
            && j + 1 < chars.len()
            && chars[j + 1].is_ascii_digit()
        {
            j += 1;
            j += digits_at(chars, j, usize::MAX);
        }
        let raw: String = chars[i..j].iter().collect();
        let mut k = j;
        k += space_run(chars, k);
        if k >= chars.len() || !matches!(chars[k], 'K' | 'M' | 'B' | 'T') {
            return None;
        }
        if k + 1 < chars.len() && chars[k + 1].is_ascii_alphanumeric() {
            return None;
        }
        let scale_word = match chars[k] {
            'K' => "thousand",
            'M' => "million",
            'B' => "billion",
            _ => "trillion",
        };
        let words = if raw.contains('.') {
            float_to_words(&raw, "point")
        } else {
            number_to_words(raw.parse::<i64>().unwrap_or(0))
        };
        Some((format!("{words} {scale_word}"), k + 1 - i))
    })
}

fn expand_fractions(text: &str) -> String {
    scan_replace(text, |chars, i| {
        if !word_boundary(chars, i) {
            return None;
        }
        let num_len = digits_at(chars, i, usize::MAX);
        if num_len == 0 {
            return None;
        }
        let mut j = i + num_len;
        j += space_run(chars, j);
        if j >= chars.len() || chars[j] != '/' {
            return None;
        }
        j += 1;
        j += space_run(chars, j);
        let den_len = digits_at(chars, j, usize::MAX);
        if den_len == 0 {
            return None;
        }
        let end = j + den_len;
        if !word_boundary(chars, end) {
            return None;
        }
        let num: i64 = chars[i..i + num_len]
            .iter()
            .collect::<String>()
            .parse()
            .unwrap_or(0);
        let den: i64 = chars[j..end]
            .iter()
            .collect::<String>()
            .parse()
            .unwrap_or(0);
        if den == 0 {
            let original: String = chars[i..end].iter().collect();
            return Some((original, end - i));
        }
        let num_words = number_to_words(num);
        let denom_word: String = if den == 2 {
            if num == 1 {
                "half".to_string()
            } else {
                "halves".to_string()
            }
        } else if den == 4 {
            if num == 1 {
                "quarter".to_string()
            } else {
                "quarters".to_string()
            }
        } else {
            let mut word = ordinal_suffix(den);
            if num != 1 {
                word.push('s');
            }
            word
        };
        Some((format!("{num_words} {denom_word}"), end - i))
    })
}

fn expand_decades(text: &str) -> String {
    scan_replace(text, |chars, i| {
        if !word_boundary(chars, i) {
            return None;
        }
        // \b(\d{1,3})0s\b: the digit run backtracks so the literal
        // trailing 0 can match ("80s" reads group 8, then "0s").
        let d_len = digits_at(chars, i, 3);
        if d_len == 0 {
            return None;
        }
        let mut group_len = 0usize;
        let mut j = i;
        for len in (1..=d_len).rev() {
            let candidate = i + len;
            if candidate + 1 < chars.len() && chars[candidate] == '0' {
                group_len = len;
                j = candidate + 1;
                break;
            }
        }
        if group_len == 0 {
            return None;
        }
        if j >= chars.len() || ascii_lower(chars[j]) != 's' {
            return None;
        }
        j += 1;
        if !word_boundary(chars, j) {
            return None;
        }
        let base: i64 = chars[i..i + group_len]
            .iter()
            .collect::<String>()
            .parse()
            .unwrap_or(0);
        let decade_word = match base % 10 {
            0 => "hundreds",
            1 => "tens",
            2 => "twenties",
            3 => "thirties",
            4 => "forties",
            5 => "fifties",
            6 => "sixties",
            7 => "seventies",
            8 => "eighties",
            _ => "nineties",
        };
        let replacement = if base < 10 {
            decade_word.to_string()
        } else {
            format!("{} {decade_word}", number_to_words(base / 10))
        };
        Some((replacement, j - i))
    })
}

const DIGIT_WORDS: [&str; 10] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
];

fn digit_group(chars: &[char], start: usize) -> String {
    chars[start..]
        .iter()
        .take_while(|c| c.is_ascii_digit())
        .map(|c| DIGIT_WORDS[c.to_digit(10).expect("digit") as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

fn expand_phone_numbers(text: &str) -> String {
    // Longest pattern first; the lookbehinds keep sub-matches from firing.
    let mut text = scan_replace(text, |chars, i| {
        // 11-digit: (?<!\d-)(?<!\d)\b(\d{1,2})-(\d{3})-(\d{3})-(\d{4})\b(?!-\d)
        if !word_boundary(chars, i)
            || (i > 0 && chars[i - 1].is_ascii_digit())
            || (i > 1 && chars[i - 1] == '-' && chars[i - 2].is_ascii_digit())
        {
            return None;
        }
        let mut j = i;
        let mut groups = Vec::new();
        let first = digits_at(chars, j, 2);
        if first == 0 {
            return None;
        }
        j += first;
        for expected in [3usize, 3, 4] {
            if j >= chars.len() || chars[j] != '-' {
                return None;
            }
            j += 1;
            let len = digits_at(chars, j, expected);
            if len != expected {
                return None;
            }
            groups.push(j);
            j += len;
        }
        if !word_boundary(chars, j)
            || (j + 1 < chars.len() && chars[j] == '-' && chars[j + 1].is_ascii_digit())
        {
            return None;
        }
        let mut words = vec![digit_group(chars, i)];
        for g in groups {
            words.push(digit_group(chars, g));
        }
        Some((words.join(" "), j - i))
    });
    // 10-digit: 555-123-4567
    text = scan_replace(&text, |chars, i| {
        if !word_boundary(chars, i)
            || (i > 0 && chars[i - 1].is_ascii_digit())
            || (i > 1 && chars[i - 1] == '-' && chars[i - 2].is_ascii_digit())
        {
            return None;
        }
        let mut j = i;
        let mut groups = Vec::new();
        let first = digits_at(chars, j, 3);
        if first != 3 {
            return None;
        }
        groups.push(j);
        j += first;
        for expected in [3usize, 4] {
            if j >= chars.len() || chars[j] != '-' {
                return None;
            }
            j += 1;
            let len = digits_at(chars, j, expected);
            if len != expected {
                return None;
            }
            groups.push(j);
            j += len;
        }
        if !word_boundary(chars, j)
            || (j + 1 < chars.len() && chars[j] == '-' && chars[j + 1].is_ascii_digit())
        {
            return None;
        }
        let words = groups
            .iter()
            .map(|g| digit_group(chars, *g))
            .collect::<Vec<_>>();
        Some((words.join(" "), j - i))
    });
    // 7-digit local: 555-1234
    scan_replace(&text, |chars, i| {
        if !word_boundary(chars, i)
            || (i > 1 && chars[i - 1] == '-' && chars[i - 2].is_ascii_digit())
        {
            return None;
        }
        let first = digits_at(chars, i, 3);
        if first != 3 {
            return None;
        }
        let mut j = i + 3;
        if j >= chars.len() || chars[j] != '-' {
            return None;
        }
        j += 1;
        let second = digits_at(chars, j, 4);
        if second != 4 {
            return None;
        }
        let end = j + 4;
        if !word_boundary(chars, end)
            || (end + 1 < chars.len() && chars[end] == '-' && chars[end + 1].is_ascii_digit())
        {
            return None;
        }
        Some((
            format!("{} {}", digit_group(chars, i), digit_group(chars, j)),
            end - i,
        ))
    })
}

fn expand_ranges(text: &str) -> String {
    scan_replace(text, |chars, i| {
        // (?<!\w)(\d+)-(\d+)(?!\w)
        if i > 0 && is_word_char(chars[i - 1]) {
            return None;
        }
        let lo_len = digits_at(chars, i, usize::MAX);
        if lo_len == 0 {
            return None;
        }
        let mut j = i + lo_len;
        if j >= chars.len() || chars[j] != '-' {
            return None;
        }
        j += 1;
        let hi_len = digits_at(chars, j, usize::MAX);
        if hi_len == 0 {
            return None;
        }
        let end = j + hi_len;
        if end < chars.len() && is_word_char(chars[end]) {
            return None;
        }
        let lo: i64 = chars[i..i + lo_len]
            .iter()
            .collect::<String>()
            .parse()
            .unwrap_or(0);
        let hi: i64 = chars[j..end]
            .iter()
            .collect::<String>()
            .parse()
            .unwrap_or(0);
        Some((
            format!("{} to {}", number_to_words(lo), number_to_words(hi)),
            end - i,
        ))
    })
}

fn expand_model_names(text: &str) -> String {
    scan_replace(text, |chars, i| {
        // \b([a-zA-Z][a-zA-Z0-9]*)-(\d[\d.]*)(?=[^\d.]|$)
        if !word_boundary(chars, i) || !chars[i].is_ascii_alphabetic() {
            return None;
        }
        let mut j = i + 1;
        while j < chars.len() && chars[j].is_ascii_alphanumeric() {
            j += 1;
        }
        if j >= chars.len() || chars[j] != '-' {
            return None;
        }
        j += 1;
        let num_start = j;
        if j >= chars.len() || !chars[j].is_ascii_digit() {
            return None;
        }
        while j < chars.len() && (chars[j].is_ascii_digit() || chars[j] == '.') {
            j += 1;
        }
        if j < chars.len() && (chars[j].is_ascii_digit() || chars[j] == '.') {
            return None;
        }
        let name: String = chars[i..num_start - 1].iter().collect();
        let number: String = chars[num_start..j].iter().collect();
        Some((format!("{name} {number}"), j - i))
    })
}

/// `replace_numbers`: bare numbers (never a minus after an ASCII letter)
/// to words, floats digit by digit.
fn replace_numbers(text: &str) -> String {
    scan_replace(text, |chars, i| {
        if i > 0 && chars[i - 1].is_ascii_alphabetic() {
            return None;
        }
        let mut j = i;
        if chars[j] == '-' {
            j += 1;
        }
        let num_start = j;
        j += number_comma_run(chars, j);
        if j == num_start {
            return None;
        }
        if j < chars.len()
            && chars[j] == '.'
            && j + 1 < chars.len()
            && chars[j + 1].is_ascii_digit()
        {
            j += 1;
            j += digits_at(chars, j, usize::MAX);
        }
        let raw: String = chars[i..j].iter().collect();
        let raw_clean: String = raw.chars().filter(|c| *c != ',').collect();
        let words = if raw_clean.contains('.') {
            float_to_words(&raw_clean, "point")
        } else {
            let parsed = raw_clean.parse::<f64>().unwrap_or(0.0);
            if parsed.is_finite() && parsed.abs() < 9.0e15 {
                number_to_words(parsed as i64)
            } else {
                return Some((raw, j - i));
            }
        };
        Some((words, j - i))
    })
}

fn remove_punctuation(text: &str) -> String {
    text.chars()
        .map(|c| {
            if is_word_char(c) || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect()
}

fn remove_extra_whitespace(text: &str) -> String {
    let mut out = String::new();
    let mut pending_space = false;
    for c in text.chars() {
        if c.is_whitespace() {
            pending_space = !out.is_empty();
        } else {
            if pending_space {
                out.push(' ');
                pending_space = false;
            }
            out.push(c);
        }
    }
    out
}

/// The `TextPreprocessor` default pipeline. The reference's
/// `expand_roman_numerals` (default False), `remove_hashtags`,
/// `remove_mentions`, `remove_accents`, and `remove_stopwords` (default
/// False) are not ported; `normalize_unicode` (NFC) is a no-op on ASCII
/// input and is skipped (see the README).
#[derive(Debug, Clone, Default)]
pub struct TextPreprocessor;

impl TextPreprocessor {
    pub fn new() -> Self {
        Self
    }

    pub fn process(&self, text: &str) -> Result<String> {
        let mut text = text.to_string();
        text = remove_html_tags(&text);
        text = remove_urls(&text);
        text = remove_emails(&text);
        // Checked after URL and email removal so ids inside them do not
        // trip the range check.
        check_integer_literals(&text)?;
        text = expand_contractions(&text);
        text = expand_ip_addresses(&text);
        text = normalize_leading_decimals(&text);
        text = expand_currency(&text);
        text = expand_percentages(&text);
        text = expand_scientific_notation(&text);
        text = expand_time(&text);
        text = expand_ordinals(&text);
        text = expand_units(&text);
        text = expand_scale_suffixes(&text);
        text = expand_fractions(&text);
        text = expand_decades(&text);
        text = expand_phone_numbers(&text);
        text = expand_ranges(&text);
        text = expand_model_names(&text);
        text = replace_numbers(&text);
        text = remove_punctuation(&text);
        text = text.to_lowercase();
        text = remove_extra_whitespace(&text);
        Ok(text)
    }
}

/// Bound used by `replace_numbers`: integers below it are spelled out.
/// Larger values pass through as raw digits, which the cleaner drops.
const SPELLED_INTEGER_LIMIT: i64 = 9_000_000_000_000_000;

/// Refuses integer literals the number helpers cannot spell out. The
/// helpers parse with `unwrap_or(0)`, so an oversized literal would
/// otherwise be read as "zero". Refusing matches the kokoro frontend,
/// which rejects numbers above 999999999. Digits after a '.' are a decimal
/// fraction (read one digit at a time) and are not checked.
fn check_integer_literals(text: &str) -> Result<()> {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let fraction = i > 0 && bytes[i - 1] == b'.';
        let mut digits = String::new();
        while i < bytes.len() {
            if bytes[i].is_ascii_digit() {
                digits.push(char::from(bytes[i]));
                i += 1;
            } else if bytes[i] == b',' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
                // Thousands separator inside a literal ("1,000,000").
                i += 1;
            } else {
                break;
            }
        }
        let spellable = digits
            .parse::<i64>()
            .is_ok_and(|v| v < SPELLED_INTEGER_LIMIT);
        if !fraction && !spellable {
            return Err(SpeechError::Input {
                why: format!(
                    "integer with {} digits is at or above {SPELLED_INTEGER_LIMIT}; write it in English",
                    digits.len()
                ),
            });
        }
    }
    Ok(())
}
