//! The melotts text front end (`text.py`).
//!
//! Port of `text_normalize` and the pure post-G2P mapping (`_refine_ph`,
//! `_post_replace_ph`, `_distribute_phone`, `cleaned_text_to_sequence`,
//! the `add_blank` interleave) at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/melotts).
//! Every regex pass is a left-to-right scanner with sequential `re.sub`
//! semantics.
//!
//! The G2P itself (the bert-base-uncased wordpiece tokenizer, the CMU
//! dictionary, and the g2p_en neural fallback) is caller-owned: the
//! reference loads all three from external assets this crate does not
//! carry. The intake is the raw ARPAbet phone list per word plus the
//! word2ph alignment the reference g2p would return; `refine_ph`,
//! `post_replace_ph`, and `cleaned_text_to_sequence` then run here (see
//! the family README).

/// `pu_symbols`: punctuation plus the special symbols.
pub const PU_SYMBOLS: [&str; 9] = ["!", "?", "...", ",", ".", "'", "-", "SP", "UNK"];
/// `pad`.
pub const PAD: &str = "_";

/// `en_symbols`: the ARPAbet phones; the uppercase V is intentional.
pub const EN_SYMBOLS: [&str; 39] = [
    "aa", "ae", "ah", "ao", "aw", "ay", "b", "ch", "d", "dh", "eh", "er", "ey", "f", "g", "hh",
    "ih", "iy", "jh", "k", "l", "m", "n", "ng", "ow", "oy", "p", "r", "s", "sh", "t", "th", "uh",
    "uw", "V", "w", "y", "z", "zh",
];

/// The default symbol table (`[pad] + sorted(set(en_symbols)) +
/// pu_symbols`), the English-only fallback the reference overwrites from
/// the checkpoint config.json.
pub fn default_symbols() -> Vec<String> {
    let mut symbols: Vec<String> = vec![PAD.to_string()];
    let mut phones: Vec<&str> = EN_SYMBOLS.to_vec();
    phones.sort_unstable();
    symbols.extend(phones.into_iter().map(str::to_string));
    symbols.extend(PU_SYMBOLS.iter().map(|s| s.to_string()));
    symbols
}

/// `language_id_map`.
pub fn language_id(language: &str) -> Option<i32> {
    match language {
        "ZH" => Some(0),
        "JP" => Some(1),
        "EN" => Some(2),
        "ZH_MIX_EN" => Some(3),
        "KR" => Some(4),
        "ES" | "SP" => Some(5),
        "FR" => Some(6),
        _ => None,
    }
}

/// `language_tone_start_map`.
pub fn tone_start(language: &str) -> Option<i32> {
    match language {
        "ZH" | "ZH_MIX_EN" => Some(0),
        "JP" => Some(6),
        "EN" => Some(7),
        "KR" => Some(11),
        "ES" | "SP" => Some(12),
        "FR" => Some(13),
        _ => None,
    }
}

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

/// `_number_to_words`: integers to English words (simplified, up to
/// 999,999), "minus" for negatives.
pub fn number_to_words(n: i64) -> String {
    if n == 0 {
        return "zero".to_string();
    }
    if n < 0 {
        return format!("minus {}", number_to_words(-n));
    }
    let mut n = n as u64;
    let mut parts: Vec<String> = Vec::new();
    if n >= 1000 {
        parts.push(format!("{} thousand", number_to_words((n / 1000) as i64)));
        n %= 1000;
    }
    if n >= 100 {
        parts.push(format!("{} hundred", ONES[(n / 100) as usize]));
        n %= 100;
    }
    if n >= 20 {
        let mut word = TENS[(n / 10) as usize].to_string();
        if n % 10 != 0 {
            word.push(' ');
            word.push_str(ONES[(n % 10) as usize]);
        }
        parts.push(word);
    } else if n > 0 {
        parts.push(ONES[n as usize].to_string());
    }
    parts.join(" ")
}

/// `_expand_decimal`: "3.14" -> "three point one four"; digits 20 and up
/// read as their decimal string (the reference's quirk).
fn expand_decimal_number(text: &str) -> String {
    let (int_part, dec_part) = match text.split_once('.') {
        Some(parts) => parts,
        None => return text.to_string(),
    };
    let integer = number_to_words(int_part.parse::<i64>().unwrap_or(0));
    let decimal = dec_part
        .chars()
        .filter_map(|d| d.to_digit(10))
        .map(|d| {
            if d < 20 {
                ONES[d as usize].to_string()
            } else {
                d.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ");
    format!("{integer} point {decimal}")
}

const ABBREVIATIONS: [(&str, &str); 18] = [
    ("mrs", "missis"),
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

/// Python `\b` over ASCII word characters.
fn word_boundary(chars: &[char], index: usize) -> bool {
    let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let before = index > 0 && is_word(chars[index - 1]);
    let after = index < chars.len() && is_word(chars[index]);
    before != after
}

fn digit_run(chars: &[char], index: usize) -> usize {
    let mut n = 0;
    while index + n < chars.len() && chars[index + n].is_ascii_digit() {
        n += 1;
    }
    n
}

fn digit_run_max(chars: &[char], index: usize, max: usize) -> usize {
    let mut n = 0;
    while n < max && index + n < chars.len() && chars[index + n].is_ascii_digit() {
        n += 1;
    }
    n
}

fn comma_number_run(chars: &[char], index: usize) -> Option<usize> {
    // (\d{1,3}(,\d{3})+)
    let first = digit_run_max(chars, index, 3);
    if first == 0 {
        return None;
    }
    let mut j = index + first;
    let mut groups = 0;
    while j < chars.len() && chars[j] == ',' {
        let digits = digit_run(chars, j + 1);
        if digits != 3 {
            break;
        }
        j += 1 + digits;
        groups += 1;
    }
    if groups == 0 {
        None
    } else {
        Some(j - index)
    }
}

fn decimal_run(chars: &[char], index: usize) -> Option<usize> {
    // (\d+\.\d+)
    let int_len = digit_run(chars, index);
    if int_len == 0 {
        return None;
    }
    let j = index + int_len;
    if j >= chars.len() || chars[j] != '.' {
        return None;
    }
    let frac = digit_run(chars, j + 1);
    if frac == 0 {
        return None;
    }
    Some(int_len + 1 + frac)
}

/// `text_normalize`: lowercase, the abbreviation table, comma-number
/// removal, decimal and integer expansion.
pub fn text_normalize(text: &str) -> String {
    let mut text = text.to_lowercase();
    // Abbreviations, in reference table order.
    for (abbrev, expansion) in ABBREVIATIONS {
        let needle = format!("{abbrev}.");
        let needle_chars: Vec<char> = needle.chars().collect();
        text = scan_replace(&text, |chars, i| {
            if !word_boundary(chars, i) || i + needle_chars.len() > chars.len() {
                return None;
            }
            if chars[i..i + needle_chars.len()]
                .iter()
                .zip(&needle_chars)
                .all(|(a, b)| a.to_ascii_lowercase() == *b)
            {
                // The reference pattern is `\b%s\.`: a leading boundary
                // only, no trailing one.
                Some((expansion.to_string(), needle_chars.len()))
            } else {
                None
            }
        });
    }
    // Comma thousands: strip the commas.
    text = scan_replace(&text, |chars, i| {
        comma_number_run(chars, i).map(|len| {
            let kept: String = chars[i..i + len].iter().filter(|c| **c != ',').collect();
            (kept, len)
        })
    });
    // Decimals: "3.14" -> "three point one four".
    text = scan_replace(&text, |chars, i| {
        decimal_run(chars, i).map(|len| {
            let text: String = chars[i..i + len].iter().collect();
            (expand_decimal_number(&text), len)
        })
    });
    // Bare integers.
    text = scan_replace(&text, |chars, i| {
        let len = digit_run(chars, i);
        if len == 0 {
            return None;
        }
        let value: String = chars[i..i + len].iter().collect();
        // Python int() handles arbitrarily large values; the reference
        // reads the digits it can (the fixture stays in range).
        let words = value
            .parse::<i64>()
            .map(number_to_words)
            .unwrap_or_else(|_| value.clone());
        Some((words, len))
    });
    text
}

/// Sequential `re.sub` replacement over `text`.
fn scan_replace(text: &str, matcher: impl Fn(&[char], usize) -> Option<(String, usize)>) -> String {
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

/// `_refine_ph`: ARPAbet phone to (lowercase phone, tone).
pub fn refine_ph(phn: &str) -> (String, i32) {
    let last = phn.chars().last();
    match last.and_then(|c| c.to_digit(10)) {
        Some(digit) => (phn[..phn.len() - 1].to_lowercase(), digit as i32 + 1),
        None => (phn.to_lowercase(), 0),
    }
}

/// `_post_replace_ph`: map a phone to the symbol table, handling the
/// special cases (the uppercase V and unknowns).
pub fn post_replace_ph(ph: &str, symbols: &[String], punctuation: &[&str]) -> String {
    if ph == "v" {
        return "V".to_string();
    }
    if symbols.iter().any(|s| s == ph) {
        return ph.to_string();
    }
    if punctuation.contains(&ph) {
        return ph.to_string();
    }
    "UNK".to_string()
}

/// `_distribute_phone`: distribute `n_phone` phones over `n_word`
/// wordpiece sub-tokens, greedily filling the smallest buckets first.
pub fn distribute_phone(n_phone: usize, n_word: usize) -> Vec<usize> {
    let mut phones_per_word = vec![0usize; n_word];
    for _ in 0..n_phone {
        let (min_index, _) = phones_per_word
            .iter()
            .enumerate()
            .min_by_key(|(_, v)| **v)
            .expect("n_word > 0");
        phones_per_word[min_index] += 1;
    }
    phones_per_word
}

/// `cleaned_text_to_sequence`: phone symbols to integer ids with the tone
/// and language offsets, against the caller's symbol table.
pub fn cleaned_text_to_sequence(
    phones: &[String],
    tones: &[i32],
    language: &str,
    symbols: &[String],
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let tone_start = tone_start(language).unwrap_or(0);
    let lang_id = language_id(language).unwrap_or(0);
    let unk = symbols.iter().position(|s| s == "UNK").unwrap_or(0) as i32;
    let phone_ids = phones
        .iter()
        .map(|p| {
            symbols
                .iter()
                .position(|s| s == p)
                .map(|i| i as i32)
                .unwrap_or(unk)
        })
        .collect();
    let tone_ids = tones.iter().map(|t| t + tone_start).collect();
    let lang_ids = vec![lang_id; phones.len()];
    (phone_ids, tone_ids, lang_ids)
}

/// The reference `process_text` add_blank interleave: a pad before and
/// after every phone (including the leading/trailing pads the g2p adds).
pub fn add_blank_interleave(phones: &[String], tones: &[i32]) -> (Vec<String>, Vec<i32>) {
    let mut phones_out = vec![PAD.to_string()];
    let mut tones_out = vec![0];
    for (p, t) in phones.iter().zip(tones) {
        phones_out.push(p.clone());
        phones_out.push(PAD.to_string());
        tones_out.push(*t);
        tones_out.push(0);
    }
    (phones_out, tones_out)
}
