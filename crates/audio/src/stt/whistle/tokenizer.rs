//! SentencePiece-style BPE attachment from the Needle export.

use crate::{Result, SpeechError};
use std::collections::HashMap;
use turbospark_core::TokenId;

#[allow(dead_code)]
#[derive(Debug)]
struct Piece {
    text: String,
    score: f32,
    kind: u8,
}

#[allow(dead_code)]
#[derive(Debug)]
pub(super) struct Tokenizer {
    pieces: Vec<Piece>,
    lookup: HashMap<String, TokenId>,
    byte_ids: [Option<TokenId>; 256],
    markers: Vec<String>,
    dummy_prefix: bool,
    byte_fallback: bool,
    pub bos: TokenId,
    pub eos: TokenId,
    unk: TokenId,
}

fn invalid(why: &str) -> SpeechError {
    SpeechError::Tensor {
        name: "whistle.tokenizer".into(),
        why: why.into(),
    }
}

impl Tokenizer {
    pub fn parse(bytes: &[u8], vocab: usize) -> Result<Self> {
        if bytes.len() < 24 {
            return Err(invalid("truncated tokenizer header"));
        }
        let header: [u32; 8] = std::array::from_fn(|i| match i {
            0..=4 => u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            5 | 6 => u32::from(bytes[20 + i - 5]),
            _ => u32::from(u16::from_le_bytes(bytes[22..24].try_into().unwrap())),
        });
        if header[0] as usize != vocab
            || vocab > 65_536
            || header[5] > 1
            || header[6] > 1
            || header[7] != 0
            || header[1..5].iter().any(|&id| id as usize >= vocab)
        {
            return Err(invalid("invalid tokenizer vocabulary or special IDs"));
        }
        let mut pieces = Vec::with_capacity(vocab);
        let mut lookup = HashMap::new();
        let mut byte_ids = [None; 256];
        let mut markers = Vec::new();
        let mut offset = 24;
        for id in 0..vocab {
            let record = bytes
                .get(offset..offset + 7)
                .ok_or_else(|| invalid("truncated piece record"))?;
            let score = f32::from_le_bytes(record[..4].try_into().unwrap());
            let kind = record[4];
            let len = u16::from_le_bytes(record[5..7].try_into().unwrap()) as usize;
            offset += 7;
            let text = std::str::from_utf8(
                bytes
                    .get(offset..offset + len)
                    .ok_or_else(|| invalid("truncated piece surface"))?,
            )
            .map_err(|_| invalid("invalid UTF-8 piece"))?
            .to_owned();
            offset += len;
            if !score.is_finite()
                || kind > 4
                || text.is_empty()
                || lookup.insert(text.clone(), id as TokenId).is_some()
            {
                return Err(invalid("invalid or duplicate piece"));
            }
            if kind == 4 {
                let byte = text
                    .strip_prefix("<0x")
                    .and_then(|s| s.strip_suffix('>'))
                    .filter(|s| s.len() == 2)
                    .and_then(|s| u8::from_str_radix(s, 16).ok())
                    .ok_or_else(|| invalid("malformed byte piece"))?;
                if byte_ids[byte as usize].replace(id as TokenId).is_some() {
                    return Err(invalid("duplicate byte piece"));
                }
            }
            if kind == 3 {
                markers.push(text.clone());
            }
            pieces.push(Piece { text, score, kind });
        }
        if offset != bytes.len() || (header[6] != 0 && byte_ids.iter().any(Option::is_none)) {
            return Err(invalid("trailing data or incomplete byte fallback"));
        }
        if pieces[header[2] as usize].kind != 2
            || pieces[header[3] as usize].kind != 2
            || pieces[header[4] as usize].kind != 1
            || header[2] == header[3]
        {
            return Err(invalid("invalid BOS, EOS, or unknown piece type"));
        }
        markers.sort_by_key(|s| std::cmp::Reverse(s.chars().count()));
        Ok(Self {
            pieces,
            lookup,
            byte_ids,
            markers,
            dummy_prefix: header[5] != 0,
            byte_fallback: header[6] != 0,
            bos: header[3] as TokenId,
            eos: header[2] as TokenId,
            unk: header[4] as TokenId,
        })
    }

    pub fn marker(&self, text: &str) -> Result<TokenId> {
        self.lookup
            .get(text)
            .copied()
            .filter(|&id| self.pieces[id as usize].kind == 3)
            .ok_or_else(|| invalid("missing language marker"))
    }

    #[allow(dead_code)]
    fn bpe(&self, segment: &str, ids: &mut Vec<TokenId>) {
        let mut symbols: Vec<String> = segment.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best: Option<(usize, f32)> = None;
            for j in 0..symbols.len().saturating_sub(1) {
                if let Some(&id) = self.lookup.get(&(symbols[j].clone() + &symbols[j + 1])) {
                    // Control pieces must not be made by merging ordinary text.
                    if self.pieces[id as usize].kind != 0 {
                        continue;
                    }
                    let score = self.pieces[id as usize].score;
                    if best.is_none_or(|(_, previous)| score > previous) {
                        best = Some((j, score));
                    }
                }
            }
            let Some((j, _)) = best else { break };
            let right = symbols.remove(j + 1);
            symbols[j].push_str(&right);
        }
        for symbol in symbols {
            if let Some(&id) = self.lookup.get(&symbol) {
                ids.push(id);
            } else if self.byte_fallback {
                ids.extend(symbol.bytes().map(|b| self.byte_ids[b as usize].unwrap()));
            } else {
                ids.push(self.unk);
            }
        }
    }

    #[allow(dead_code)]
    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut escaped = text.replace(' ', "\u{2581}");
        if self.dummy_prefix {
            escaped.insert(0, '\u{2581}');
        }
        let mut ids = Vec::new();
        let mut start = 0;
        let mut cursor = 0;
        while cursor < escaped.len() {
            if let Some(marker) = self
                .markers
                .iter()
                .find(|m| escaped[cursor..].starts_with(m.as_str()))
            {
                self.bpe(&escaped[start..cursor], &mut ids);
                ids.push(self.lookup[marker]);
                cursor += marker.len();
                start = cursor;
            } else {
                cursor += escaped[cursor..].chars().next().unwrap().len_utf8();
            }
        }
        self.bpe(&escaped[start..], &mut ids);
        ids
    }

    pub fn decode(&self, ids: &[TokenId]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            let piece = usize::try_from(id)
                .ok()
                .and_then(|id| self.pieces.get(id))
                .ok_or_else(|| invalid("token ID outside vocabulary"))?;
            match piece.kind {
                1 | 2 => (),
                4 => bytes.push(u8::from_str_radix(&piece.text[3..5], 16).unwrap()),
                _ => bytes.extend_from_slice(piece.text.as_bytes()),
            }
        }
        let mut text = String::from_utf8_lossy(&bytes).replace('\u{2581}', " ");
        if self.dummy_prefix && text.starts_with(' ') {
            text.remove(0);
        }
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny(ab: f32, bc: f32) -> Vec<u8> {
        let entries = [
            ("<pad>", 0.0, 2),
            ("</s>", 0.0, 2),
            ("<s>", 0.0, 2),
            ("<unk>", 0.0, 1),
            ("a", 0.0, 0),
            ("b", 0.0, 0),
            ("c", 0.0, 0),
            ("ab", ab, 0),
            ("bc", bc, 0),
            ("<|en|>", 0.0, 3),
        ];
        let mut bytes = Vec::new();
        for value in [entries.len() as u32, 0, 1, 2, 3] {
            bytes.extend(value.to_le_bytes());
        }
        bytes.extend([0, 0, 0, 0]);
        for (text, score, kind) in entries {
            bytes.extend(score.to_le_bytes());
            bytes.push(kind);
            bytes.extend((text.len() as u16).to_le_bytes());
            bytes.extend(text.as_bytes());
        }
        bytes
    }

    #[test]
    fn bpe_uses_highest_score_leftmost_ties_and_protects_markers() {
        for (ab, bc, expected) in [(1.0, 2.0, vec!["a", "bc"]), (2.0, 2.0, vec!["ab", "c"])] {
            let tokenizer = Tokenizer::parse(&tiny(ab, bc), 10).unwrap();
            let actual = tokenizer.encode("abc");
            let surfaces: Vec<_> = actual
                .iter()
                .map(|&id| tokenizer.pieces[id as usize].text.as_str())
                .collect();
            assert_eq!(surfaces, expected);
            assert_eq!(
                tokenizer.encode("a<|en|>b"),
                [
                    tokenizer.lookup["a"],
                    tokenizer.marker("<|en|>").unwrap(),
                    tokenizer.lookup["b"]
                ]
            );
        }
    }

    #[test]
    fn refuses_special_type_changes_and_incomplete_byte_fallback() {
        let original = tiny(1.0, 2.0);
        let mut changed = original.clone();
        changed[21] = 1;
        assert!(Tokenizer::parse(&changed, 10).is_err());
        let mut changed = original.clone();
        changed[12..16].copy_from_slice(&4u32.to_le_bytes());
        assert!(Tokenizer::parse(&changed, 10).is_err());
        let mut changed = original;
        changed[8..12].copy_from_slice(&2u32.to_le_bytes());
        assert!(Tokenizer::parse(&changed, 10).is_err());
    }
}
