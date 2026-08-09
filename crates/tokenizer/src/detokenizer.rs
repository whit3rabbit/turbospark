//! Streaming detokenizer for generation loops. Ported from
//! `Tokenization/Detokenizer.swift`.
//!
//! Byte-fallback tokens (Gemma's SentencePiece `<0xXX>` tokens) can split a
//! multi-byte codepoint across several tokens; naively decoding each token
//! in isolation yields broken UTF-8. This holds back a trailing run of
//! byte-fallback IDs (and a trailing run of U+FFFD in the decoded text, the
//! renderer's placeholder for bytes still in flight) until a following
//! token resolves them, and reconciles a decode that contradicts text
//! already handed out (`clean_up_tokenization_spaces` can rewrite a space an
//! earlier decode already emitted) by emitting the divergent tail instead of
//! dropping it.

use crate::dialect::MfTokenizer;

/// Streaming detokenizer accumulating tokens into UTF-8 text chunks.
pub struct MfDetokenizer<'a> {
    tokenizer: &'a MfTokenizer,
    stable_ids: Vec<i32>,
    trailing_byte_ids: Vec<i32>,
    emitted: String,
}

impl<'a> MfDetokenizer<'a> {
    /// Creates a new streaming detokenizer wrapping the given tokenizer.
    pub fn new(tokenizer: &'a MfTokenizer) -> Self {
        Self {
            tokenizer,
            stable_ids: Vec::new(),
            trailing_byte_ids: Vec::new(),
            emitted: String::new(),
        }
    }

    /// Pushes a single token ID into the detokenizer, returning any newly emitted text.
    pub fn push(&mut self, id: i32) -> String {
        let token = self.tokenizer.id_to_token(id).unwrap_or_default();
        if is_byte_fallback(&token) {
            self.trailing_byte_ids.push(id);
            return String::new();
        }

        if !self.trailing_byte_ids.is_empty() {
            self.stable_ids.append(&mut self.trailing_byte_ids);
        }
        self.stable_ids.push(id);

        let current = self.tokenizer.decode(&self.stable_ids, true);
        self.commit_delta(&current, true)
    }

    /// Flushes any pending byte fallback tokens and returns all remaining text.
    pub fn flush(&mut self) -> String {
        let stable_text = if self.stable_ids.is_empty() {
            String::new()
        } else {
            self.tokenizer.decode(&self.stable_ids, true)
        };
        let trailing_text = self.assemble_byte_fallback();
        let full_text = stable_text + &trailing_text;
        self.commit_delta(&full_text, false)
    }

    fn commit_delta(&mut self, current: &str, holding_partial_scalar: bool) -> String {
        let cur = current.as_bytes();
        let withheld = if holding_partial_scalar {
            trailing_replacement_byte_count(current)
        } else {
            0
        };
        let held = withheld.min(cur.len().saturating_sub(self.emitted.len()));
        let committed = &cur[..cur.len() - held];
        let committed_text = || -> String {
            if held == 0 {
                current.to_string()
            } else {
                String::from_utf8_lossy(committed).into_owned()
            }
        };

        if !cur.starts_with(self.emitted.as_bytes()) {
            // Bounded by `committed.len()`: Swift's `dropFirst` clamps past
            // the end instead of trapping, and `shared` is a prefix match
            // against the *unclamped* `cur`, which can outrun `committed`
            // once `held` trims from the end.
            let shared =
                shared_scalar_aligned_prefix(cur, self.emitted.as_bytes()).min(committed.len());
            let new_tail = &committed[shared..];
            let overlap = overlap_count(&self.emitted.as_bytes()[shared..], new_tail);
            let tail = String::from_utf8_lossy(&new_tail[overlap..]).into_owned();
            self.emitted = committed_text();
            return tail;
        }
        let delta = String::from_utf8_lossy(&committed[self.emitted.len()..]).into_owned();
        self.emitted = committed_text();
        delta
    }

    fn assemble_byte_fallback(&self) -> String {
        let mut bytes = Vec::with_capacity(self.trailing_byte_ids.len());
        for &id in &self.trailing_byte_ids {
            let Some(tok) = self.tokenizer.id_to_token(id) else {
                continue;
            };
            if !is_byte_fallback(&tok) {
                continue;
            }
            let hex = &tok[3..tok.len() - 1];
            if let Ok(byte) = u8::from_str_radix(hex, 16) {
                bytes.push(byte);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

fn is_byte_fallback(token: &str) -> bool {
    let chars: Vec<char> = token.chars().collect();
    chars.len() == 6
        && token.starts_with("<0x")
        && token.ends_with('>')
        && chars[3..5].iter().all(|c| c.is_ascii_hexdigit())
}

/// UTF-8 length of the trailing run of U+FFFD in `text`.
fn trailing_replacement_byte_count(text: &str) -> usize {
    let mut bytes = 0;
    for c in text.chars().rev() {
        if c != '\u{FFFD}' {
            break;
        }
        bytes += 3; // U+FFFD is EF BF BD.
    }
    bytes
}

/// Byte length of the longest common prefix of `a` and `b`, backed off to a
/// UTF-8 scalar boundary so slicing `a` at it cannot split a codepoint.
fn shared_scalar_aligned_prefix(a: &[u8], b: &[u8]) -> usize {
    let mut matched = 0;
    let mut last_boundary = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if x != y {
            break;
        }
        if x & 0xC0 != 0x80 {
            last_boundary = matched;
        }
        matched += 1;
    }
    let splits_a_scalar = a.get(matched).is_some_and(|&b| b & 0xC0 == 0x80);
    if splits_a_scalar {
        last_boundary
    } else {
        matched
    }
}

/// Largest `k` where the first `k` bytes of `tail` are also the last `k`
/// bytes of `shown`, rejecting a `k` that would split a scalar in `tail`.
fn overlap_count(shown: &[u8], tail: &[u8]) -> usize {
    let mut k = shown.len().min(tail.len());
    while k > 0 {
        let splits_a_scalar = tail.get(k).is_some_and(|&b| b & 0xC0 == 0x80);
        if !splits_a_scalar && shown[shown.len() - k..] == tail[..k] {
            return k;
        }
        k -= 1;
    }
    0
}
