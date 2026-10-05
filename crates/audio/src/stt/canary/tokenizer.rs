//! Canary SentencePiece-style tokenizer.
//!
//! Reference: `mlx_audio/stt/models/canary/tokenizer.py` (CanaryTokenizer)
//! and the `post_load_hook` path in `canary.py` at mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`. The pinned checkpoint embeds
//! the serialized SentencePiece `ModelProto` in `config.json` at
//! `tokenizer.model_base64` (the reference's third lookup), so the port
//! parses that protobuf directly: repeated field 1 holds one `SentencePiece`
//! message per piece with `piece` (field 1, string), `score` (field 2,
//! fixed32) and `type` (field 3, varint). Encoding is never needed (the
//! decoder only consumes model output), so the port keeps the id to piece
//! table plus the piece types and mirrors `SentencePieceProcessor.decode`
//! for the token id sequences the greedy loop produces.

use std::collections::HashMap;

use serde_json::Value;

use crate::{Result, SpeechError};

/// SentencePiece piece types from model.proto.
const TYPE_NORMAL: u64 = 1;
const TYPE_UNKNOWN: u64 = 2;
const TYPE_CONTROL: u64 = 3;
const TYPE_USER_DEFINED: u64 = 4;

/// The token id table of one Canary checkpoint.
#[derive(Debug, Clone)]
pub struct CanaryTokenizer {
    pieces: Vec<String>,
    /// SentencePiece type per id (normal, unknown, control, user defined).
    types: Vec<u64>,
    token_to_id: HashMap<String, i32>,
}

fn bad(field: &str, why: impl Into<String>) -> SpeechError {
    SpeechError::BadConfig {
        field: field.to_string(),
        why: why.into(),
    }
}

/// Decodes standard base64 with padding, matching Python's `base64.b64decode`.
fn decode_base64(input: &str) -> Result<Vec<u8>> {
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
    let mut out = Vec::with_capacity(input.len() / 4 * 3);
    let mut buffer = 0u32;
    let mut bits = 0u32;
    let mut padding = 0usize;
    let mut data_chars = 0usize;
    for byte in input.bytes() {
        match byte {
            b'=' => padding += 1,
            b'\n' | b'\r' | b' ' | b'\t' => {}
            other => {
                if padding > 0 {
                    return Err(bad("tokenizer.model_base64", "data after padding"));
                }
                data_chars += 1;
                let entry = value(other).ok_or_else(|| {
                    bad(
                        "tokenizer.model_base64",
                        format!("invalid base64 byte {other:#x}"),
                    )
                })?;
                buffer = (buffer << 6) | entry;
                bits += 6;
                if bits >= 8 {
                    bits -= 8;
                    out.push(((buffer >> bits) & 0xFF) as u8);
                }
            }
        }
    }
    if padding > 2 {
        return Err(bad(
            "tokenizer.model_base64",
            "more than two padding characters",
        ));
    }
    if data_chars % 4 == 1 {
        return Err(bad("tokenizer.model_base64", "invalid base64 length"));
    }
    Ok(out)
}

fn read_varint(bytes: &[u8], position: &mut usize) -> Result<u64> {
    let mut value = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *bytes
            .get(*position)
            .ok_or_else(|| bad("tokenizer.model_base64", "truncated protobuf varint"))?;
        *position += 1;
        value |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
        if shift > 63 {
            return Err(bad("tokenizer.model_base64", "varint too long"));
        }
    }
}

/// Skips one protobuf field value of the given wire type.
fn skip_field(bytes: &[u8], position: &mut usize, wire: u64) -> Result<()> {
    match wire {
        0 => {
            read_varint(bytes, position)?;
        }
        1 => *position += 8,
        2 => {
            let length = read_varint(bytes, position)? as usize;
            if (*position)
                .checked_add(length)
                .ok_or_else(|| bad("tokenizer.model_base64", "protobuf length overflow"))?
                > bytes.len()
            {
                return Err(bad(
                    "tokenizer.model_base64",
                    "protobuf field overruns input",
                ));
            }
            *position += length;
        }
        5 => *position += 4,
        other => {
            return Err(bad(
                "tokenizer.model_base64",
                format!("unsupported protobuf wire type {other}"),
            ))
        }
    }
    Ok(())
}

impl CanaryTokenizer {
    /// Builds the tokenizer from a parsed `config.json` carrying
    /// `tokenizer.model_base64` (the reference's embedded-proto lookup; the
    /// pinned checkpoint ships no tokenizer.model or tokens.txt).
    pub fn from_config(config: &Value) -> Result<Self> {
        let section = config.get("tokenizer").ok_or_else(|| {
            bad(
                "tokenizer",
                "missing from config.json (no tokenizer.model_base64)",
            )
        })?;
        let encoded = section
            .get("model_base64")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("tokenizer.model_base64", "missing or not a string"))?;
        let proto = decode_base64(encoded)?;
        Self::from_proto(&proto)
    }

    /// Parses a serialized SentencePiece ModelProto: repeated field 1.
    pub fn from_proto(proto: &[u8]) -> Result<Self> {
        let mut pieces = Vec::new();
        let mut types = Vec::new();
        let mut position = 0usize;
        while position < proto.len() {
            let tag = read_varint(proto, &mut position)?;
            let field = tag >> 3;
            let wire = tag & 0x7;
            if field == 1 && wire == 2 {
                let length = read_varint(proto, &mut position)? as usize;
                let end = position
                    .checked_add(length)
                    .filter(|&end| end <= proto.len())
                    .ok_or_else(|| bad("tokenizer.model_base64", "piece message overruns input"))?;
                let message = &proto[position..end];
                position = end;
                let mut piece = String::new();
                let mut kind = TYPE_NORMAL;
                let mut inner = 0usize;
                while inner < message.len() {
                    let tag = read_varint(message, &mut inner)?;
                    let field = tag >> 3;
                    let wire = tag & 0x7;
                    match (field, wire) {
                        (1, 2) => {
                            let length = read_varint(message, &mut inner)? as usize;
                            let stop = inner
                                .checked_add(length)
                                .filter(|&stop| stop <= message.len())
                                .ok_or_else(|| {
                                    bad("tokenizer.model_base64", "piece string overruns message")
                                })?;
                            piece =
                                String::from_utf8(message[inner..stop].to_vec()).map_err(|_| {
                                    bad("tokenizer.model_base64", "piece is not valid UTF-8")
                                })?;
                            inner = stop;
                        }
                        (2, 5) => inner += 4,
                        (3, 0) => kind = read_varint(message, &mut inner)?,
                        _ => skip_field(message, &mut inner, wire)?,
                    }
                }
                pieces.push(piece);
                types.push(kind);
            } else {
                // trainer_spec and the other ModelProto fields are unused.
                skip_field(proto, &mut position, wire)?;
            }
        }
        if pieces.is_empty() {
            return Err(bad("tokenizer.model_base64", "no pieces parsed"));
        }
        let token_to_id = pieces
            .iter()
            .enumerate()
            .map(|(id, piece)| (piece.clone(), id as i32))
            .collect();
        Ok(Self {
            pieces,
            types,
            token_to_id,
        })
    }

    /// Number of pieces in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// The piece string for one id.
    pub fn piece(&self, id: usize) -> &str {
        self.pieces.get(id).map(String::as_str).unwrap_or("")
    }

    /// Id for one literal token string, mirroring `token2id` lookups.
    pub fn token_to_id(&self, token: &str) -> Option<i32> {
        self.token_to_id.get(token).copied()
    }

    /// The end-of-text id the greedy loop stops at (the reference defaults
    /// the `<|endoftext|>` lookup to 0).
    pub fn eos_id(&self) -> i32 {
        self.token_to_id.get("<|endoftext|>").copied().unwrap_or(0)
    }

    fn require_id(&self, token: &str) -> Result<i32> {
        self.token_to_id(token).ok_or_else(|| {
            bad(
                "tokenizer",
                format!("prompt token {token} is not in the vocabulary"),
            )
        })
    }

    /// Builds the Canary prompt token sequence, mirroring
    /// `CanaryTokenizer.build_prompt_tokens`:
    /// `<|startofcontext|> <|startoftranscript|> <|emo:undefined|>
    /// <|{src}|> <|{tgt}|> (<|pnc|>|<|nopnc|>) <|noitn|> <|notimestamp|>
    /// <|nodiarize|>`.
    pub fn build_prompt_tokens(
        &self,
        source_lang: &str,
        target_lang: &str,
        use_pnc: bool,
    ) -> Result<Vec<i32>> {
        Ok(vec![
            self.require_id("<|startofcontext|>")?,
            self.require_id("<|startoftranscript|>")?,
            self.require_id("<|emo:undefined|>")?,
            self.require_id(&format!("<|{source_lang}|>"))?,
            self.require_id(&format!("<|{target_lang}|>"))?,
            if use_pnc {
                self.require_id("<|pnc|>")?
            } else {
                self.require_id("<|nopnc|>")?
            },
            self.require_id("<|noitn|>")?,
            self.require_id("<|notimestamp|>")?,
            self.require_id("<|nodiarize|>")?,
        ])
    }

    /// Decodes generated token ids to text, mirroring
    /// `SentencePieceProcessor.decode` for the sequences the greedy loop
    /// produces: control pieces are dropped, normal and user-defined pieces
    /// concatenate with the U+2581 marker becoming a space, and the caller's
    /// visible transcript is stripped exactly like the reference's final
    /// `text.strip()`. Unknown-type ids refuse: the reference raises when a
    /// decoded id has no piece. Byte-fallback pieces never occur in the
    /// pinned vocabulary and refuse as unsupported.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut text = String::new();
        for &id in ids {
            let id = usize::try_from(id)
                .map_err(|_| bad("generated ids", "token id must be non-negative"))?;
            let piece = self.pieces.get(id).ok_or_else(|| {
                bad(
                    "generated ids",
                    format!("token id {id} exceeds the vocabulary"),
                )
            })?;
            match self.types[id] {
                TYPE_CONTROL => {}
                TYPE_NORMAL | TYPE_USER_DEFINED => text.push_str(piece),
                TYPE_UNKNOWN => {
                    return Err(SpeechError::Input {
                        why: "cannot decode an unknown-type token id".into(),
                    })
                }
                other => {
                    return Err(SpeechError::Unsupported {
                        why: format!("cannot decode piece type {other}"),
                    })
                }
            }
        }
        Ok(text.replace('\u{2581}', " ").trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HELLO_WORLD: &str = "SGVsbG8sIHdvcmxkIQ==";

    #[test]
    fn base64_decodes_standard_padding() {
        assert_eq!(decode_base64(HELLO_WORLD).unwrap(), b"Hello, world!");
        assert_eq!(decode_base64("QQ==").unwrap(), b"A");
        assert_eq!(decode_base64("QUJD").unwrap(), b"ABC");
        assert!(decode_base64("A").is_err());
        assert!(decode_base64("QQ=A").is_err());
    }

    #[test]
    fn empty_proto_refuses() {
        assert!(CanaryTokenizer::from_proto(&[]).is_err());
    }
}
