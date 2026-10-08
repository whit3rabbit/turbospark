// Tokenization/normalization adapted around the pinned Misaki lexicon contract.
// The Rust fork's integer-only recursive G2P split decimal/ordinal tokens and
// dropped unsupported symbols. This strict scanner keeps each token accountable.
use super::{input, Result, SpeechError};

pub(super) struct Token {
    pub word: String,
    pub start: usize,
    pub end: usize,
    pub spoken: Option<String>,
    pub punctuation: bool,
    pub tag: String,
    pub phones: String,
}

pub(super) fn validate_text(text: &str) -> Result<()> {
    if text.trim().is_empty() {
        return Err(input(
            "text must contain a US English word, number or punctuation",
        ));
    }
    if text.len() > 65_536 {
        return Err(input(
            "text exceeds 65536 UTF-8 bytes; submit shorter requests",
        ));
    }
    for (offset, ch) in text.char_indices() {
        if !(ch.is_ascii_alphanumeric()
            || matches!(
                ch,
                ' ' | '\t'
                    | '\n'
                    | '\r'
                    | '\''
                    | '"'
                    | '('
                    | ')'
                    | '.'
                    | ','
                    | '!'
                    | '?'
                    | ';'
                    | ':'
                    | '-'
                    | '$'
                    | '%'
                    | '&'
                    | '+'
                    | '@'
                    | '/'
                    | '\u{2018}'
                    | '\u{2019}'
                    | '\u{201c}'
                    | '\u{201d}'
                    | '\u{2013}'
                    | '\u{2014}'
                    | '\u{2026}'
                    | '\u{a0}'
            ))
        {
            return Err(SpeechError::Unsupported {why:format!("input character U+{:04X} at byte {offset} is unsupported; use US English ASCII letters and supported punctuation",ch as u32)});
        }
    }
    Ok(())
}
pub(super) fn text(text: &str) -> String {
    let normalized: String = text
        .chars()
        .map(|ch| match ch {
            '\u{2018}' | '\u{2019}' => '\'',
            '\u{201c}' | '\u{201d}' => '"',
            '\u{2013}' => '\u{2014}',
            '\u{a0}' => ' ',
            ch => ch,
        })
        .collect();
    normalized.trim().to_string()
}
pub(super) fn tokens(text: &str) -> Result<Vec<Token>> {
    let bytes = text.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < bytes.len() {
        if bytes[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        let start = i;
        let mut spoken = None;
        let mut punctuation = false;
        if bytes[i].is_ascii_alphabetic() {
            i += 1;
            while i < bytes.len()
                && (bytes[i].is_ascii_alphabetic()
                    || (bytes[i] == b'\''
                        && i + 1 < bytes.len()
                        && bytes[i + 1].is_ascii_alphabetic())
                    || (bytes[i] == b'-'
                        && i + 1 < bytes.len()
                        && bytes[i + 1].is_ascii_alphabetic()))
            {
                i += 1;
            }
            if bytes.get(i) == Some(&b'\'') {
                // A trailing possessive apostrophe is part of the lexical word.
                i += 1;
            }
            if i == start + 1 && bytes.get(i) == Some(&b'.') {
                let mut dotted_end = i;
                while dotted_end + 2 < bytes.len()
                    && bytes[dotted_end] == b'.'
                    && bytes[dotted_end + 1].is_ascii_alphabetic()
                    && bytes[dotted_end + 2] == b'.'
                {
                    dotted_end += 2;
                }
                if dotted_end > i {
                    i = dotted_end + 1;
                }
            }
            // Keep dictionary abbreviations atomic; the period is not a boundary.
            if i < bytes.len()
                && bytes[i] == b'.'
                && matches!(
                    &text[start..i],
                    "Dr" | "Mr" | "Mrs" | "Ms" | "Prof" | "Sr" | "Jr" | "St" | "vs"
                )
            {
                i += 1;
            }
        } else if bytes[i].is_ascii_digit()
            || bytes[i] == b'$'
            || (bytes[i] == b'-' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit())
        {
            let currency = bytes[i] == b'$';
            let negative = bytes[i] == b'-';
            if currency || negative {
                i += 1;
            }
            let num_start = i;
            while i < bytes.len()
                && (bytes[i].is_ascii_digit()
                    || (matches!(bytes[i], b',' | b'.')
                        && i + 1 < bytes.len()
                        && bytes[i + 1].is_ascii_digit()))
            {
                i += 1;
            }
            if num_start == i {
                return Err(input("$ must be followed by a numeric dollar amount"));
            }
            if i < bytes.len()
                && bytes[i] == b':'
                && i + 1 < bytes.len()
                && bytes[i + 1].is_ascii_digit()
            {
                return Err(SpeechError::Unsupported {why:"numeric time notation is unsupported; spell the time in US English (for example three forty-five)".into()});
            }
            let number = &text[num_start..i];
            let mut ordinal = false;
            if matches!(bytes.get(i..i + 2), Some(b"st" | b"nd" | b"rd" | b"th")) {
                i += 2;
                ordinal = true;
            }
            if i < bytes.len() && bytes[i].is_ascii_alphabetic() {
                return Err(input(
                    "number suffix is unsupported; separate the number and word",
                ));
            }
            spoken = Some(number_words(number, ordinal, currency, negative)?);
        } else {
            let ch = text[i..]
                .chars()
                .next()
                .expect("scanner is at a character boundary");
            i += ch.len_utf8();
            match ch {
                ';' | ':' | ',' | '.' | '!' | '?' | '"' | '(' | ')' | '\u{2014}' | '\u{2026}' => {
                    punctuation = true
                }
                '-' => {
                    // Hyphen is a word separator, represented by a supported pause.
                    spoken = Some(String::new());
                }
                '%' | '&' | '+' | '@' => {}
                '/' => {
                    return Err(SpeechError::Unsupported {why:"slash notation is unsupported; write slash or use separate US English words".into()});
                }
                '\'' => {
                    return Err(input(
                        "apostrophe must belong to an English word or contraction",
                    ));
                }
                _ => return Err(input("unrecognized text token")),
            }
        }
        let mut word = text[start..i].to_string();
        if word == "-" {
            word = "\u{2014}".into();
            punctuation = true;
            spoken = None;
        }
        out.push(Token {
            word,
            start,
            end: i,
            spoken,
            punctuation,
            tag: String::new(),
            phones: String::new(),
        });
    }
    Ok(out)
}

fn number_words(number: &str, ordinal: bool, currency: bool, negative: bool) -> Result<String> {
    let parts: Vec<_> = number.split('.').collect();
    if parts.len() > 2 || (ordinal && (currency || parts.len() != 1)) {
        return Err(input(
            "unsupported numeric format; spell the number in English",
        ));
    }
    let integer = parts[0];
    let groups: Vec<_> = integer.split(',').collect();
    if groups.len() > 1
        && (groups[0].is_empty()
            || groups[0].len() > 3
            || groups.iter().skip(1).any(|g| g.len() != 3))
    {
        return Err(input("numeric commas must separate groups of three digits"));
    }
    let digits = integer.replace(',', "");
    let n: u64 = digits
        .parse()
        .map_err(|_| input("number is outside the supported range"))?;
    if n > 999_999_999 {
        return Err(input("numbers above 999999999 must be written in English"));
    }
    let mut spoken = if ordinal {
        ordinal_words(n)
    } else if !currency
        && parts.len() == 1
        && digits.len() == 4
        && groups.len() == 1
        && (1000..=9999).contains(&n)
    {
        let high = n / 100;
        let low = n % 100;
        if high % 10 == 0 && low < 10 {
            cardinal(n)
        } else if low == 0 {
            format!("{} hundred", cardinal(high))
        } else if low < 10 {
            format!("{} oh {}", cardinal(high), cardinal(low))
        } else {
            format!("{} {}", cardinal(high), cardinal(low))
        }
    } else {
        cardinal(n)
    };
    if currency {
        let cents = if parts.len() == 2 {
            let s = parts[1];
            if s.is_empty() || s.len() > 2 {
                return Err(input("US dollar amounts support one or two decimal places"));
            }
            let c: u64 = s.parse().map_err(|_| input("invalid dollar cents"))?;
            if s.len() == 1 {
                c * 10
            } else {
                c
            }
        } else {
            0
        };
        spoken = format!("{} {}", spoken, if n == 1 { "dollar" } else { "dollars" });
        if cents != 0 {
            let cents = format!(
                "{} {}",
                cardinal(cents),
                if cents == 1 { "cent" } else { "cents" }
            );
            if n == 0 {
                spoken = cents;
            } else {
                spoken.push_str(&format!(" and {cents}"));
            }
        }
    } else if parts.len() == 2 {
        let fraction = parts[1];
        if fraction.is_empty()
            || fraction.len() > 9
            || !fraction.bytes().all(|c| c.is_ascii_digit())
        {
            return Err(input("decimal fractions support one through nine digits"));
        }
        spoken.push_str(" point");
        for digit in fraction.bytes() {
            spoken.push(' ');
            spoken.push_str(&cardinal((digit - b'0') as u64));
        }
    }
    if negative {
        spoken = format!("minus {spoken}");
    }
    Ok(spoken)
}
fn cardinal(n: u64) -> String {
    const SMALL: [&str; 20] = [
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
    if n < 20 {
        return SMALL[n as usize].into();
    }
    if n < 100 {
        return if n % 10 == 0 {
            TENS[(n / 10) as usize].into()
        } else {
            format!("{} {}", TENS[(n / 10) as usize], cardinal(n % 10))
        };
    }
    for (unit, name) in [(1_000_000, "million"), (1000, "thousand"), (100, "hundred")] {
        if n >= unit {
            return if n % unit == 0 {
                format!("{} {name}", cardinal(n / unit))
            } else {
                format!("{} {name} {}", cardinal(n / unit), cardinal(n % unit))
            };
        }
    }
    unreachable!("all integers covered")
}
fn ordinal_words(n: u64) -> String {
    let cardinal = cardinal(n);
    let (prefix, last) = cardinal.rsplit_once(' ').unwrap_or(("", &cardinal));
    let word = match last {
        "zero" => "zeroth".into(),
        "one" => "first".into(),
        "two" => "second".into(),
        "three" => "third".into(),
        "five" => "fifth".into(),
        "eight" => "eighth".into(),
        "nine" => "ninth".into(),
        "twelve" => "twelfth".into(),
        s if s.ends_with('y') => format!("{}ieth", &s[..s.len() - 1]),
        s => format!("{s}th"),
    };
    if prefix.is_empty() {
        word
    } else {
        format!("{prefix} {word}")
    }
}
