//! Echo-TTS text intake (`text.py`): the prompt normalization (speaker
//! markers, punctuation folding) and the byte-level tokenizer with a
//! zero BOS. The whole vocabulary is one UTF-8 byte, so no external
//! tokenizer is needed and the family owns its intake.

/// `normalize_text_prompt`: fold the fancy punctuation to ASCII, keep
/// the pre-marked dialogue tags, and default to speaker 1.
pub fn normalize_text_prompt(text: &str) -> String {
    let mut out = text.to_owned();
    out = out.replace('\u{2026}', "...");
    out = out.replace('\u{2019}', "'");
    out = out.replace('\u{201d}', "\"");
    out = out.replace('\n', " ");
    out = out.replace(':', ",");
    out = out.replace(';', ",");
    out = out.replace('\u{2014}', ", ");
    if !out.starts_with('[') && !out.starts_with('(') && !out.contains("S1") && !out.contains("S2")
    {
        out = format!("[S1] {out}");
    }
    out
}

/// `tokenizer_encode`: UTF-8 bytes with a zero BOS prepended.
pub fn tokenizer_encode(text: &str, append_bos: bool, normalize: bool) -> Vec<i32> {
    let text = if normalize {
        normalize_text_prompt(text)
    } else {
        text.to_owned()
    };
    let mut tokens: Vec<i32> = Vec::with_capacity(text.len() + 1);
    if append_bos {
        tokens.push(0);
    }
    tokens.extend(text.as_bytes().iter().map(|&b| b as i32));
    tokens
}

/// The padded id/mask batch of `get_text_input_ids_and_mask`. With
/// `pad_to_max = false` the width is the longest truncated encoding;
/// otherwise `max_length` itself.
pub struct TextBatch {
    pub ids: Vec<i32>,
    pub mask: Vec<bool>,
    pub rows: usize,
    pub width: usize,
}

/// `get_text_input_ids_and_mask`: encode every text (BOS, UTF-8 bytes),
/// truncate to `max_length`, pad, and mask. `max_length = None` uses the
/// longest encoding.
pub fn get_text_input_ids_and_mask(
    texts: &[&str],
    max_length: Option<usize>,
    normalize: bool,
    pad_to_max: bool,
) -> TextBatch {
    let encoded: Vec<Vec<i32>> = texts
        .iter()
        .map(|text| {
            let normalized = if normalize {
                normalize_text_prompt(text)
            } else {
                String::from(*text)
            };
            tokenizer_encode(&normalized, true, false)
        })
        .collect();
    let max_length = max_length.unwrap_or_else(|| encoded.iter().map(Vec::len).max().unwrap_or(0));
    let mut rows: Vec<(Vec<i32>, usize)> = encoded
        .into_iter()
        .map(|tokens| {
            let length = tokens.len().min(max_length);
            (tokens[..length].to_vec(), length)
        })
        .collect();
    let mut width = if pad_to_max {
        max_length
    } else {
        rows.iter().map(|(_, length)| *length).max().unwrap_or(0)
    };
    if !pad_to_max {
        width = width.max(1);
    }
    let mut ids = vec![0i32; texts.len() * width];
    let mut mask = vec![false; texts.len() * width];
    for (row, (tokens, length)) in rows.drain(..).enumerate() {
        ids[row * width..row * width + length].copy_from_slice(&tokens);
        for flag in &mut mask[row * width..row * width + length] {
            *flag = true;
        }
    }
    TextBatch {
        ids,
        mask,
        rows: texts.len(),
        width,
    }
}
