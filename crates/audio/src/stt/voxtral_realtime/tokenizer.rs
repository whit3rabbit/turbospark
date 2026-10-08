//! Tekken tokenizer for Voxtral Realtime, decode side only.
//!
//! Reference: `mlx_audio/stt/models/voxtral_realtime/tokenizer.py`
//! (`TekkenTokenizer`) at mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`.
//!
//! Token layout of `tekken.json`: ids `0..n_special` are special tokens
//! (BOS 1, EOS 2, STREAMING_PAD 32 for the pinned checkpoint); ids at and
//! above `n_special` index the BPE vocabulary, whose entries carry
//! base64-encoded UTF-8 bytes. The upstream decode concatenates the bytes
//! of every non-special id and decodes the result as UTF-8 with
//! replacement; this port matches that, including the empty-string
//! fallback for out-of-range ids.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::nn::bad_config;
use crate::{Result, SpeechError};

/// The decode-only Tekken tokenizer loaded from a model directory.
pub(crate) struct TekkenTokenizer {
    /// Base64 payload per BPE rank; index `id - n_special`.
    vocab: Vec<String>,
    n_special: usize,
    special_ids: BTreeSet<i64>,
}

impl TekkenTokenizer {
    /// Loads `tekken.json` from `model_dir`.
    pub(crate) fn from_model_path(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("tekken.json");
        let bytes = fs::read(&path).map_err(|error| SpeechError::Input {
            why: format!("cannot read {}: {error}", path.display()),
        })?;
        let root: Value = serde_json::from_slice(&bytes)
            .map_err(|error| bad_config("tekken.json", error.to_string()))?;
        let config = root
            .get("config")
            .ok_or_else(|| bad_config("tekken.json", "config block is missing"))?;
        let n_special = config
            .get("default_num_special_tokens")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                bad_config(
                    "tekken.json.config.default_num_special_tokens",
                    "must be an integer",
                )
            })? as usize;
        let vocab = root
            .get("vocab")
            .and_then(Value::as_array)
            .ok_or_else(|| bad_config("tekken.json.vocab", "must be an array"))?
            .iter()
            .map(|entry| {
                entry
                    .get("token_bytes")
                    .and_then(Value::as_str)
                    .ok_or_else(|| bad_config("tekken.json.vocab", "every entry needs token_bytes"))
                    .map(str::to_owned)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut special_ids = BTreeSet::new();
        if let Some(tokens) = root.get("special_tokens").and_then(Value::as_array) {
            for token in tokens {
                if let Some(rank) = token.get("rank").and_then(Value::as_i64) {
                    special_ids.insert(rank);
                }
            }
        }
        Ok(Self {
            vocab,
            n_special,
            special_ids,
        })
    }

    pub(crate) fn num_special_tokens(&self) -> usize {
        self.n_special
    }

    /// Raw bytes of one token id; empty for special and out-of-range ids
    /// (the upstream `token_bytes` fallback).
    pub(crate) fn token_bytes(&self, id: i32) -> Vec<u8> {
        if id < 0 || (id as i64) < self.n_special as i64 || self.special_ids.contains(&(id as i64))
        {
            return Vec::new();
        }
        let rank = id as usize - self.n_special;
        match self.vocab.get(rank) {
            Some(payload) => decode_base64(payload).unwrap_or_default(),
            None => Vec::new(),
        }
    }

    /// Decodes ids to text: special ids contribute nothing, the rest
    /// concatenate their bytes, decoded as UTF-8 with replacement.
    pub(crate) fn decode(&self, ids: &[i32]) -> String {
        let mut out = Vec::new();
        for &id in ids {
            out.extend_from_slice(&self.token_bytes(id));
        }
        String::from_utf8_lossy(&out).into_owned()
    }
}

/// Decodes standard-alphabet base64 with padding. Hand-rolled so the audio
/// crate keeps its dependency set; the Tekken payloads use exactly this
/// alphabet.
pub(crate) fn decode_base64(input: &str) -> Option<Vec<u8>> {
    fn value(byte: u8) -> Option<u32> {
        match byte {
            b'A'..=b'Z' => Some((byte - b'A') as u32),
            b'a'..=b'z' => Some((byte - b'a') as u32 + 26),
            b'0'..=b'9' => Some((byte - b'0') as u32 + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let clean: Vec<u8> = input
        .bytes()
        .filter(|&b| b != b'\n' && b != b'\r')
        .collect();
    if clean.len() % 4 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(clean.len() / 4 * 3);
    for chunk in clean.chunks_exact(4) {
        // Padding is a suffix run of at most two.
        let pad = chunk.iter().rev().take_while(|&&b| b == b'=').count();
        if pad > 2 || chunk[..4 - pad].iter().any(|&b| b == b'=') {
            return None;
        }
        let mut group = 0u32;
        for &byte in &chunk[..4 - pad] {
            group = (group << 6) | value(byte)?;
        }
        group <<= 6 * pad as u32;
        out.push((group >> 16) as u8);
        if pad < 2 {
            out.push((group >> 8) as u8);
        }
        if pad < 1 {
            out.push(group as u8);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::{decode_base64, TekkenTokenizer};
    use std::path::Path;

    #[test]
    fn base64_decoder_matches_known_vectors() {
        assert_eq!(decode_base64("").unwrap(), Vec::<u8>::new());
        assert_eq!(decode_base64("AA==").unwrap(), vec![0]);
        assert_eq!(decode_base64("IGA=").unwrap(), b" `".to_vec());
        assert_eq!(decode_base64("aGVsbG8=").unwrap(), b"hello".to_vec());
        assert_eq!(decode_base64("aGVsbG8h").unwrap(), b"hello!".to_vec());
        assert_eq!(
            decode_base64("SGVsbG8sIHdvcmxkIQ==").unwrap(),
            b"Hello, world!".to_vec()
        );
        // The generated-token payload the fixture records for " The".
        assert_eq!(decode_base64("IFRoZQ==").unwrap(), b" The".to_vec());
        // Non-alphabet bytes and impossible paddings refuse.
        assert!(decode_base64("aGVs*").is_none());
        assert!(decode_base64("A").is_none());
        assert!(decode_base64("AA=A").is_none());
        assert_eq!(decode_base64("AAA=").unwrap(), vec![0, 0]);
    }

    /// A tiny in-repo tekken.json exercising the special-id skip, the
    /// `n_special` offset, out-of-range fallback, and lossy UTF-8.
    #[test]
    fn tekken_decode_skips_specials_and_joins_bytes() {
        let dir = std::env::temp_dir().join(format!("voxtral-tekken-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("tekken.json"),
            serde_json::json!({
                "config": {"default_num_special_tokens": 3},
                "vocab": [
                    {"rank": 0, "token_bytes": "aGk="},
                    {"rank": 1, "token_bytes": "IQ=="},
                ],
                "special_tokens": [
                    {"rank": 0, "token_str": "<unk>"},
                    {"rank": 1, "token_str": "<s>"},
                    {"rank": 2, "token_str": "</s>"},
                ],
            })
            .to_string(),
        )
        .unwrap();
        let tokenizer = TekkenTokenizer::from_model_path(Path::new(&dir)).unwrap();
        assert_eq!(tokenizer.num_special_tokens(), 3);
        // ids 0..2 are special, id 3 is rank 0 ("hi"), id 4 is rank 1
        // ("!"), id 99 is out of range.
        assert_eq!(tokenizer.token_bytes(0), Vec::<u8>::new());
        assert_eq!(tokenizer.token_bytes(3), b"hi".to_vec());
        assert_eq!(tokenizer.token_bytes(4), b"!".to_vec());
        assert_eq!(tokenizer.token_bytes(99), Vec::<u8>::new());
        // Specials contribute nothing; bytes concatenate.
        let text = tokenizer.decode(&[0, 3, 4, 1, 99]);
        assert_eq!(text, "hi!");
        std::fs::remove_dir_all(&dir).ok();
    }
}
