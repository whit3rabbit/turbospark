//! The word vocabulary: strings (`reader_words.json`) and unit-norm FP16
//! text-embedding rows (`word_unit_f16.bin`).
//!
//! Both files index the same word ids. `word_unit_f16.bin` is headerless,
//! row-major `[words][EMBEDDING_DIM]` little-endian IEEE binary16; the
//! pinned file is exactly `229457 * 768 * 2` bytes and its rows have unit
//! L2 norm to FP16 precision. The list holds 153 strings twice under
//! distinct ids and ends with the `<rep>` marker; see the family README.

use std::path::Path;

use half::f16;

use crate::nn::bad_config;
use crate::{Result, SpeechError};

/// Width of the text-embedding rows and the reader's `vec` output.
pub const EMBEDDING_DIM: usize = 768;
/// The vocabulary entry the pinned list ends with. This port renders it as
/// a repeat of the preceding word (an unverified reading; see the README).
pub const REPEAT_MARKER: &str = "<rep>";
pub const WORDS_FILE: &str = "reader_words.json";
pub const WORD_TABLE_FILE: &str = "word_unit_f16.bin";

/// Word strings by id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordList {
    words: Vec<String>,
    repeat_id: Option<u32>,
}

impl WordList {
    /// Parses the JSON string array the reference ships.
    pub fn from_json_str(text: &str) -> Result<Self> {
        let words: Vec<String> =
            serde_json::from_str(text).map_err(|error| bad_config(WORDS_FILE, error))?;
        Self::new(words)
    }

    pub fn new(words: Vec<String>) -> Result<Self> {
        if words.is_empty() {
            return Err(bad_config(WORDS_FILE, "the word list is empty"));
        }
        if u32::try_from(words.len()).is_err() || i32::try_from(words.len()).is_err() {
            return Err(bad_config(WORDS_FILE, "too many words for 32-bit ids"));
        }
        let repeat_id = words
            .iter()
            .position(|word| word == REPEAT_MARKER)
            .map(|id| id as u32);
        Ok(Self { words, repeat_id })
    }

    pub fn load(asset_dir: &Path) -> Result<Self> {
        let path = asset_dir.join(WORDS_FILE);
        let text = std::fs::read_to_string(&path).map_err(|error| bad_config(WORDS_FILE, error))?;
        Self::from_json_str(&text)
    }

    pub fn len(&self) -> usize {
        self.words.len()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    pub fn word(&self, id: u32) -> Option<&str> {
        self.words.get(id as usize).map(String::as_str)
    }

    /// Id of the `<rep>` marker, when the list carries one.
    pub fn repeat_id(&self) -> Option<u32> {
        self.repeat_id
    }
}

/// FP16 embedding rows kept as raw bits and widened on use, so the pinned
/// 352 MB table is not doubled in memory.
#[derive(Debug, Clone, PartialEq)]
pub struct WordTable {
    rows: usize,
    dim: usize,
    bits: Vec<u16>,
}

impl WordTable {
    pub fn from_f16_le_bytes(bytes: &[u8], dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(bad_config(WORD_TABLE_FILE, "row width must be positive"));
        }
        let row_bytes = dim * 2;
        if bytes.is_empty() || bytes.len() % row_bytes != 0 {
            return Err(bad_config(
                WORD_TABLE_FILE,
                format!(
                    "{} bytes is not a whole number of {dim}-wide FP16 rows",
                    bytes.len()
                ),
            ));
        }
        let bits = bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        Ok(Self {
            rows: bytes.len() / row_bytes,
            dim,
            bits,
        })
    }

    /// Builds a table from f32 rows, rounding to FP16 (tests and dynamic
    /// vocabulary experiments).
    pub fn from_f32_rows(values: &[f32], dim: usize) -> Result<Self> {
        if dim == 0 || values.is_empty() || values.len() % dim != 0 {
            return Err(bad_config(
                WORD_TABLE_FILE,
                "rows must be a non-empty whole number of dim-wide rows",
            ));
        }
        Ok(Self {
            rows: values.len() / dim,
            dim,
            bits: values.iter().map(|&v| f16::from_f32(v).to_bits()).collect(),
        })
    }

    pub fn load(asset_dir: &Path) -> Result<Self> {
        let bytes = std::fs::read(asset_dir.join(WORD_TABLE_FILE))
            .map_err(|error| bad_config(WORD_TABLE_FILE, error))?;
        Self::from_f16_le_bytes(&bytes, EMBEDDING_DIM)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Row `id` widened to f32.
    pub fn row(&self, id: u32) -> Option<Vec<f32>> {
        let id = id as usize;
        (id < self.rows).then(|| {
            self.bits[id * self.dim..(id + 1) * self.dim]
                .iter()
                .map(|&bits| f16::from_bits(bits).to_f32())
                .collect()
        })
    }

    /// Dot product of row `id` with `query`, accumulated in f32.
    pub fn dot(&self, id: u32, query: &[f32]) -> Option<f32> {
        let id = id as usize;
        if id >= self.rows || query.len() != self.dim {
            return None;
        }
        Some(
            self.bits[id * self.dim..(id + 1) * self.dim]
                .iter()
                .zip(query)
                .map(|(&bits, &q)| f16::from_bits(bits).to_f32() * q)
                .sum(),
        )
    }

    /// Checks that the strings and rows describe the same vocabulary.
    pub fn check_matches(&self, words: &WordList) -> Result<()> {
        if self.rows != words.len() {
            return Err(SpeechError::BadConfig {
                field: WORD_TABLE_FILE.into(),
                why: format!(
                    "{} embedding rows but {} words in {WORDS_FILE}",
                    self.rows,
                    words.len()
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_list_finds_the_repeat_marker() {
        let list = WordList::from_json_str(r#"["a","the","a","<rep>"]"#).unwrap();
        assert_eq!(list.len(), 4);
        assert_eq!(list.repeat_id(), Some(3));
        assert_eq!(list.word(2), Some("a"));
        assert_eq!(list.word(4), None);
        assert_eq!(
            WordList::from_json_str(r#"["a"]"#).unwrap().repeat_id(),
            None
        );
        assert!(WordList::from_json_str("[]").is_err());
        assert!(WordList::from_json_str("{}").is_err());
    }

    #[test]
    fn fp16_rows_decode_little_endian() {
        // 1.0 = 0x3C00, -2.0 = 0xC000, 0.5 = 0x3800, 0 = 0x0000.
        let bytes = [0x00, 0x3C, 0x00, 0xC0, 0x00, 0x38, 0x00, 0x00];
        let table = WordTable::from_f16_le_bytes(&bytes, 2).unwrap();
        assert_eq!(table.rows(), 2);
        assert_eq!(table.row(0), Some(vec![1.0, -2.0]));
        assert_eq!(table.row(1), Some(vec![0.5, 0.0]));
        assert_eq!(table.row(2), None);
        assert_eq!(table.dot(0, &[3.0, 1.0]), Some(1.0));
        assert_eq!(table.dot(0, &[3.0]), None);
        assert!(WordTable::from_f16_le_bytes(&bytes[..6], 2).is_err());
        assert!(WordTable::from_f16_le_bytes(&[], 2).is_err());
    }

    #[test]
    fn table_and_list_sizes_must_agree() {
        let table = WordTable::from_f32_rows(&[1.0, 0.0, 0.0, 1.0], 2).unwrap();
        let two = WordList::from_json_str(r#"["a","b"]"#).unwrap();
        let three = WordList::from_json_str(r#"["a","b","c"]"#).unwrap();
        assert!(table.check_matches(&two).is_ok());
        assert!(table.check_matches(&three).is_err());
    }
}
