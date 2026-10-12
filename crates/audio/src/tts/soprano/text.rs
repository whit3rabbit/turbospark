//! The soprano text front end (`mlx_audio/tts/models/soprano/text.py`).
//!
//! Port of `clean_text` and its number, abbreviation, and punctuation
//! passes at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/soprano/text.py).
//! Each regex pass is re-implemented as a left-to-right scanner over the
//! whole string (sequential `re.sub` semantics: a pass never rescans its
//! own replacements). One documented deviation: `convert_to_ascii`
//! NFKD-decomposes Unicode before dropping non-ASCII, which needs the
//! `unicode-normalization` tables this crate does not carry; the port
//! drops non-ASCII characters outright, so callers that want the
//! reference behavior NFKD-normalize before calling (accented Latin
//! otherwise loses its letter instead of the diacritic).

/// `_expand_number` helpers (ONES/TENS/ORDINALS tables).
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

fn num_to_words(n: u64) -> String {
    if n == 0 {
        return "zero".into();
    }
    if n < 20 {
        return ONES[n as usize].into();
    }
    if n < 100 {
        let tens = TENS[(n / 10) as usize];
        let ones = n % 10;
        return if ones == 0 {
            tens.into()
        } else {
            format!("{tens} {}", ONES[ones as usize])
        };
    }
    if n < 1000 {
        let base = format!("{} hundred", ONES[(n / 100) as usize]);
        return if n % 100 == 0 {
            base
        } else {
            format!("{base} {}", num_to_words(n % 100))
        };
    }
    if n < 1_000_000 {
        let base = format!("{} thousand", num_to_words(n / 1000));
        return if n % 1000 == 0 {
            base
        } else {
            format!("{base} {}", num_to_words(n % 1000))
        };
    }
    if n < 1_000_000_000 {
        let base = format!("{} million", num_to_words(n / 1_000_000));
        return if n % 1_000_000 == 0 {
            base
        } else {
            format!("{base} {}", num_to_words(n % 1_000_000))
        };
    }
    let base = format!("{} billion", num_to_words(n / 1_000_000_000));
    if n % 1_000_000_000 == 0 {
        base
    } else {
        format!("{base} {}", num_to_words(n % 1_000_000_000))
    }
}

fn ordinal_to_words(n: u64) -> String {
    let exact: [(u64, &str); 19] = [
        (1, "first"),
        (2, "second"),
        (3, "third"),
        (4, "fourth"),
        (5, "fifth"),
        (6, "sixth"),
        (7, "seventh"),
        (8, "eighth"),
        (9, "ninth"),
        (10, "tenth"),
        (11, "eleventh"),
        (12, "twelfth"),
        (13, "thirteenth"),
        (14, "fourteenth"),
        (15, "fifteenth"),
        (16, "sixteenth"),
        (17, "seventeenth"),
        (18, "eighteenth"),
        (19, "nineteenth"),
    ];
    if let Some((_, word)) = exact.iter().find(|(k, _)| *k == n) {
        return (*word).into();
    }
    let tens_exact: [(u64, &str); 8] = [
        (20, "twentieth"),
        (30, "thirtieth"),
        (40, "fortieth"),
        (50, "fiftieth"),
        (60, "sixtieth"),
        (70, "seventieth"),
        (80, "eightieth"),
        (90, "ninetieth"),
    ];
    if let Some((_, word)) = tens_exact.iter().find(|(k, _)| *k == n) {
        return (*word).into();
    }
    if n < 100 {
        let tens = n / 10;
        let ones = n % 10;
        if ones == 0 {
            return format!("{}th", TENS[tens as usize]);
        }
        let ones_word = match exact.iter().find(|(k, _)| *k == ones) {
            Some((_, word)) => (*word).to_owned(),
            None => format!("{}th", ONES[ones as usize]),
        };
        return format!("{} {ones_word}", TENS[tens as usize]);
    }
    let base = num_to_words(n);
    if let Some(stripped) = base.strip_suffix('y') {
        return format!("{stripped}ieth");
    }
    format!("{base}th")
}

fn is_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// `re.sub(pattern, replacement, text)` with literal replacements, per
/// pass: scan for the next match with `find`, copy the skipped text,
/// append the replacement, and continue after it (a pass never rescans
/// its own output; the scan stops at end of input like `re.sub`).
fn replace_all(
    text: &[char],
    mut find: impl FnMut(&[char], usize) -> Option<(usize, usize, String)>,
) -> String {
    let mut out = String::new();
    let mut pos = 0usize;
    while pos < text.len() {
        match find(text, pos) {
            Some((start, end, replacement)) => {
                out.extend(text[pos..start].iter());
                out.push_str(&replacement);
                pos = end;
            }
            // No match at this position: copy the character and keep
            // scanning, exactly like `re.sub`.
            None => {
                out.push(text[pos]);
                pos += 1;
            }
        }
    }
    out
}

/// A digit run at `pos` returns (start, end, value). The value is `None`
/// when the run exceeds u64: saturating it would speak u64::MAX in place of
/// the caller's digits. Callers keep such runs verbatim, and must do so
/// for the whole run, since a suffix of it would still fit.
fn digit_run(chars: &[char], pos: usize) -> Option<(usize, usize, Option<u64>)> {
    let start = pos;
    let mut value: Option<u64> = Some(0);
    let mut end = pos;
    while end < chars.len() && chars[end].is_ascii_digit() {
        let digit = chars[end].to_digit(10).expect("ascii digit") as u64;
        value = value
            .and_then(|v| v.checked_mul(10))
            .and_then(|v| v.checked_add(digit));
        end += 1;
    }
    if end == start {
        None
    } else {
        Some((start, end, value))
    }
}

/// `expand_abbreviations`: the punctuation-terminated forms first (the
/// reference `\\bmrs\\.` style, case-insensitive, dot consumed), then
/// the cased whole-word forms.
fn expand_abbreviations(text: &str) -> String {
    const PUNCTUATED: &[(&str, &str)] = &[
        ("mrs", "misuss"),
        ("ms", "miss"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ];
    const CASED: &[(&str, &str)] = &[
        ("TTS", "text to speech"),
        ("Hz", "hertz"),
        ("kHz", "kilohertz"),
        ("KBs", "kilobytes"),
        ("KB", "kilobyte"),
        ("MBs", "megabytes"),
        ("MB", "megabyte"),
        ("GBs", "gigabytes"),
        ("GB", "gigabyte"),
        ("TBs", "terabytes"),
        ("TB", "terabyte"),
        ("APIs", "a p i's"),
        ("API", "a p i"),
        ("CLIs", "c l i's"),
        ("CLI", "c l i"),
        ("CPUs", "c p u's"),
        ("CPU", "c p u"),
        ("GPUs", "g p u's"),
        ("GPU", "g p u"),
        ("Ave", "avenue"),
        ("etc", "etcetera"),
    ];
    let chars: Vec<char> = text.chars().collect();
    let out = replace_all(&chars, |chars, pos| {
        for (abbrev, replacement) in PUNCTUATED {
            let word: Vec<char> = abbrev.chars().collect();
            if pos > 0 && is_word_char(chars[pos - 1]) {
                continue;
            }
            if chars[pos..].len() < word.len() + 1 {
                continue;
            }
            if !chars[pos..pos + word.len()]
                .iter()
                .zip(&word)
                .all(|(a, b)| a.to_ascii_lowercase() == *b)
            {
                continue;
            }
            if chars[pos + word.len()] != '.' {
                continue;
            }
            let end = pos + word.len() + 1;
            return Some((pos, end, (*replacement).to_owned()));
        }
        None
    });
    // Cased pass: exact-case whole word.
    let chars: Vec<char> = out.chars().collect();
    replace_all(&chars, |chars, pos| {
        for (abbrev, replacement) in CASED {
            let word: Vec<char> = abbrev.chars().collect();
            if pos > 0 && is_word_char(chars[pos - 1]) {
                continue;
            }
            if chars[pos..].len() < word.len() {
                continue;
            }
            if chars[pos..pos + word.len()].iter().ne(word.iter()) {
                continue;
            }
            let end = pos + word.len();
            if end < chars.len() && is_word_char(chars[end]) {
                continue;
            }
            return Some((pos, end, (*replacement).to_owned()));
        }
        None
    })
}

/// `normalize_numbers`: the six passes in reference order.
fn normalize_numbers(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    // 1. `#\\d` -> "number {d}".
    let out = replace_all(&chars, |chars, pos| {
        if chars[pos] == '#' && pos + 1 < chars.len() && chars[pos + 1].is_ascii_digit() {
            let digit = chars[pos + 1];
            Some((pos, pos + 2, format!("number {digit}")))
        } else {
            None
        }
    });
    // 2. `\\d(K|M|B|T)` (case-insensitive) -> "{d} {suffix word}".
    let chars: Vec<char> = out.chars().collect();
    let out = replace_all(&chars, |chars, pos| {
        let (_, end, _) = digit_run(chars, pos)?;
        if end != pos + 1 {
            return None; // the reference matches exactly one digit
        }
        let suffix = chars.get(end).copied()?;
        let word = match suffix {
            'K' | 'k' => "thousand",
            'M' | 'm' => "million",
            'B' | 'b' => "billion",
            'T' | 't' => "trillion",
            _ => return None,
        };
        let digit = chars[pos];
        Some((pos, end + 1, format!("{digit} {word}")))
    });
    // 3. `\\d[\\d,]+\\d` -> strip the commas.
    let chars: Vec<char> = out.chars().collect();
    let mut out = replace_all(&chars, |chars, pos| {
        if !chars[pos].is_ascii_digit() {
            return None;
        }
        // A digit, at least one (digit or comma), ending on a digit.
        let mut end = pos + 1;
        while end < chars.len() && (chars[end].is_ascii_digit() || chars[end] == ',') {
            end += 1;
        }
        if end == pos + 1 || !chars[end - 1].is_ascii_digit() {
            return None;
        }
        let cleaned: String = chars[pos..end].iter().filter(|&&c| c != ',').collect();
        Some((pos, end, cleaned))
    });
    // 4. `\\$([\\d.,]*\\d+)` -> the dollar/cents expansion.
    let chars: Vec<char> = out.chars().collect();
    out = replace_all(&chars, |chars, pos| {
        if chars[pos] != '$' {
            return None;
        }
        let mut end = pos + 1;
        while end < chars.len()
            && (chars[end].is_ascii_digit() || chars[end] == '.' || chars[end] == ',')
        {
            end += 1;
        }
        // The capture must end on a digit.
        while end > pos + 1 && !chars[end - 1].is_ascii_digit() {
            end -= 1;
        }
        if end == pos + 1 {
            return None;
        }
        let match_text: String = chars[pos + 1..end].iter().filter(|&&c| c != ',').collect();
        let parts: Vec<&str> = match_text.split('.').collect();
        if parts.len() > 2 {
            return Some((pos, end, format!("{match_text} dollars")));
        }
        // A value past u64 must not collapse to zero (it would be spoken
        // as "zero dollars"). Speak the digits, the same fallback as the
        // multi-decimal branch above.
        let dollars: Option<u64> = if parts[0].is_empty() {
            Some(0)
        } else {
            parts[0].parse().ok()
        };
        let cents: Option<u64> = match parts.get(1) {
            Some(cents) if !cents.is_empty() => cents.parse().ok(),
            _ => Some(0),
        };
        let (Some(dollars), Some(cents)) = (dollars, cents) else {
            return Some((pos, end, format!("{match_text} dollars")));
        };
        let replacement = match (dollars, cents) {
            (0, 0) => "zero dollars".to_owned(),
            (d, 0) => {
                let unit = if d == 1 { "dollar" } else { "dollars" };
                format!("{} {unit}", num_to_words(d))
            }
            (0, c) => {
                let unit = if c == 1 { "cent" } else { "cents" };
                format!("{} {unit}", num_to_words(c))
            }
            (d, c) => {
                let dollar_unit = if d == 1 { "dollar" } else { "dollars" };
                let cent_unit = if c == 1 { "cent" } else { "cents" };
                format!(
                    "{} {dollar_unit}, {} {cent_unit}",
                    num_to_words(d),
                    num_to_words(c)
                )
            }
        };
        Some((pos, end, replacement))
    });
    // 5. `\\d+(st|nd|rd|th)` -> ordinals.
    let chars: Vec<char> = out.chars().collect();
    let mut out = replace_all(&chars, |chars, pos| {
        let (_, end, value) = digit_run(chars, pos)?;
        let suffix: String = chars[end..].iter().take(2).collect();
        if !matches!(suffix.as_str(), "st" | "nd" | "rd" | "th") {
            return None;
        }
        let Some(value) = value else {
            return Some((pos, end, chars[pos..end].iter().collect()));
        };
        Some((pos, end + 2, ordinal_to_words(value)))
    });
    // 6. `\\d+` -> the year-aware number expansion.
    let chars: Vec<char> = out.chars().collect();
    out = replace_all(&chars, |chars, pos| {
        let (_, end, value) = digit_run(chars, pos)?;
        let Some(value) = value else {
            return Some((pos, end, chars[pos..end].iter().collect()));
        };
        if (1000..3000).contains(&value) {
            if value == 2000 {
                return Some((pos, end, "two thousand".to_owned()));
            }
            if (2000..2010).contains(&value) {
                return Some((
                    pos,
                    end,
                    format!("two thousand {}", num_to_words(value % 100)),
                ));
            }
            if value % 100 == 0 {
                return Some((pos, end, format!("{} hundred", num_to_words(value / 100))));
            }
            let first = value / 100;
            let second = value % 100;
            let replacement = if second < 10 {
                format!("{} oh {}", num_to_words(first), num_to_words(second))
            } else {
                format!("{} {}", num_to_words(first), num_to_words(second))
            };
            return Some((pos, end, replacement));
        }
        Some((pos, end, num_to_words(value)))
    });
    out
}

/// `expand_special_characters`: the literal symbol table in order.
fn expand_special_characters(text: &str) -> String {
    let table: [(&str, &str); 13] = [
        ("@", " at "),
        ("&", " and "),
        ("%", " percent "),
        (":", "."),
        (";", ","),
        ("+", " plus "),
        ("\\", " backslash "),
        ("~", " about "),
        ("<", " less than "),
        (">", " greater than "),
        ("=", " equals "),
        ("/", " slash "),
        ("_", " "),
    ];
    let mut out = text.to_owned();
    for (symbol, replacement) in table {
        // Sequential full-string replacement per entry (each reference
        // entry is its own `re.sub` on a literal).
        out = out.replace(symbol, replacement);
    }
    out
}

/// `remove_unknown_characters`: keep the recognized set, then drop the
/// placeholders `< > / _ +` that survive it.
fn remove_unknown_characters(text: &str) -> String {
    text.chars()
        .filter(|c| {
            c.is_ascii_alphanumeric()
                || matches!(
                    c,
                    ' ' | '!'
                        | '$'
                        | '%'
                        | '&'
                        | '\''
                        | '*'
                        | '+'
                        | ','
                        | '-'
                        | '.'
                        | '/'
                        | '<'
                        | '>'
                        | '?'
                        | '_'
                )
        })
        .filter(|c| !matches!(c, '<' | '>' | '/' | '_' | '+'))
        .collect()
}

/// `collapse_whitespace`: whitespace runs to single spaces, no space
/// before terminal punctuation, trimmed.
fn collapse_whitespace(text: &str) -> String {
    let mut collapsed = String::with_capacity(text.len());
    let mut last_space = false;
    for c in text.chars() {
        if c.is_whitespace() {
            if !last_space {
                collapsed.push(' ');
            }
            last_space = true;
        } else {
            collapsed.push(c);
            last_space = false;
        }
    }
    // r" ([.?!,])" -> r"\\1".
    let mut out = String::with_capacity(collapsed.len());
    let chars: Vec<char> = collapsed.chars().collect();
    for (index, &c) in chars.iter().enumerate() {
        if c == ' ' && index + 1 < chars.len() && matches!(chars[index + 1], '.' | '?' | '!' | ',')
        {
            continue;
        }
        out.push(c);
    }
    out.trim().to_owned()
}

/// `dedup_punctuation`: the four run-collapse passes in reference
/// order (each `re.sub` scans the result of the previous one).
fn dedup_punctuation(text: &str) -> String {
    // r"\.\.\.+" -> "..." (four or more dots).
    let mut out = replace_run(
        &text.chars().collect::<Vec<char>>(),
        |c| c == '.',
        |run| {
            if run.len() >= 4 {
                Some("...".to_owned())
            } else {
                None
            }
        },
    );
    // r",+" -> ",".
    let chars: Vec<char> = out.chars().collect();
    out = replace_run(&chars, |c| c == ',', |_run| Some(",".to_owned()));
    // r"[.,]*\.[.,]*" -> "." (any [.,] run holding a dot).
    let chars: Vec<char> = out.chars().collect();
    out = replace_run(
        &chars,
        |c| matches!(c, '.' | ','),
        |run| {
            if run.contains(&'.') {
                Some(".".to_owned())
            } else {
                None
            }
        },
    );
    // r"[.,!]*![.,!]*" -> "!".
    let chars: Vec<char> = out.chars().collect();
    out = replace_run(
        &chars,
        |c| matches!(c, '.' | ',' | '!'),
        |run| {
            if run.contains(&'!') {
                Some("!".to_owned())
            } else {
                None
            }
        },
    );
    // r"[.,!?]*\?[.,!?]*" -> "?".
    let chars: Vec<char> = out.chars().collect();
    replace_run(
        &chars,
        |c| matches!(c, '.' | ',' | '!' | '?'),
        |run| {
            if run.contains(&'?') {
                Some("?".to_owned())
            } else {
                None
            }
        },
    )
}

/// Longest-run scanner with the pass's candidate set: take the whole
/// maximal run of candidates, hand shrinking suffixes of it to
/// `collapse` until one accepts, and continue after the replacement
/// (sequential `re.sub` semantics).
fn replace_run(
    chars: &[char],
    candidate: impl Fn(char) -> bool,
    collapse: impl Fn(&[char]) -> Option<String>,
) -> String {
    let mut out = String::new();
    let mut pos = 0usize;
    while pos < chars.len() {
        if !candidate(chars[pos]) {
            out.push(chars[pos]);
            pos += 1;
            continue;
        }
        let mut run_end = pos;
        while run_end < chars.len() && candidate(chars[run_end]) {
            run_end += 1;
        }
        let mut end = run_end;
        let mut replaced = false;
        while end > pos {
            if let Some(replacement) = collapse(&chars[pos..end]) {
                out.push_str(&replacement);
                pos = end;
                replaced = true;
                break;
            }
            end -= 1;
        }
        if !replaced {
            out.push(chars[pos]);
            pos += 1;
        }
    }
    out
}

/// `clean_text`: the full pipeline in reference order.
pub fn clean_text(text: &str) -> String {
    let ascii = text.chars().filter(|c| c.is_ascii()).collect::<String>();
    let numbered = normalize_numbers(&ascii);
    let abbreviated = expand_abbreviations(&numbered);
    let expanded = expand_special_characters(&abbreviated);
    let lowered = expanded.to_lowercase();
    let unknown_removed = remove_unknown_characters(&lowered);
    let collapsed = collapse_whitespace(&unknown_removed);
    dedup_punctuation(&collapsed)
}
