//! Two-stage vocabulary search: a 128-dimensional block shortlist followed
//! by exact 768-dimensional reranking.
//!
//! The reference shortlist is the compiled `scorer_blk_d128_k16_b4_t500`
//! Core ML program (`model.mil`, coremltools 9.0): `vec [1,500,768]` times
//! a `[768,128]` projection, times a `[128,229460]` projected vocabulary,
//! reshaped to `[1,500,57365,4]`, `reduce_max` over the last axis, and a
//! sorted descending `topk(k=16)` returning block ids. Block `b` covers word
//! ids `4b..4b+3`; the pinned projected vocabulary is the word table in
//! identity order times the projection (checked column by column, see the
//! README), and its three trailing padding columns (ids 229457..229459) are
//! non-zero copies of one vector, so reranking must skip ids past the table.
//!
//! [`CpuBlockScorer`] reproduces that program in f32 on the CPU from the
//! compiled weights. Core ML evaluates it in FP16, so near-tied blocks can
//! order differently; the CPU path is a portable reference, not a parity
//! claim against the ANE/GPU partition.

use std::path::Path;

use crate::nn::bad_config;
use crate::{Result, SpeechError};

use super::vocabulary::{WordTable, EMBEDDING_DIM};

/// Words per shortlist block.
pub const BLOCK_SIZE: usize = 4;
/// Blocks the shortlist keeps per frame.
pub const TOP_BLOCKS: usize = 16;
/// Width of the compressed search space.
pub const SEARCH_DIM: usize = 128;
/// Projected vocabulary columns in the pinned scorer (229457 words padded
/// up to a whole number of blocks).
pub const SCORER_COLUMNS: usize = 229_460;
/// Relative path of the compiled scorer weights inside an asset directory.
pub const SCORER_WEIGHTS: &str = "scorer_blk_d128_k16_b4_t500.mlmodelc/weights/weight.bin";
/// Blob offsets named by the pinned `model.mil` (`matmul_0_y_0` and
/// `matmul_1_y_0`).
const PROJECTION_BLOB_OFFSET: usize = 64;
const VOCAB_BLOB_OFFSET: usize = 196_736;
/// MIL blob-storage metadata sentinel and the FP16 data-type code the
/// pinned weights carry.
const BLOB_SENTINEL: u32 = 0xDEAD_BEEF;
const BLOB_DTYPE_FP16: u32 = 1;

/// The best word of a frame and its similarity to the reader vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    pub word_id: u32,
    /// Dot product with the unit-norm word row. The reader L2-normalizes
    /// `vec` inside its graph, so this is the cosine up to FP16 rounding.
    pub cosine: f32,
}

/// Per-frame block ids, row-major `[frames][k]`, best block first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockShortlist {
    pub frames: usize,
    pub k: usize,
    pub blocks: Vec<u32>,
}

impl BlockShortlist {
    pub fn new(frames: usize, k: usize, blocks: Vec<u32>) -> Result<Self> {
        if k == 0 || blocks.len() != frames * k {
            return Err(SpeechError::Input {
                why: format!(
                    "shortlist holds {} block ids, expected {frames} frames x {k}",
                    blocks.len()
                ),
            });
        }
        Ok(Self { frames, k, blocks })
    }

    pub fn row(&self, frame: usize) -> &[u32] {
        &self.blocks[frame * self.k..(frame + 1) * self.k]
    }
}

/// Produces block shortlists for reader vectors (the Core ML scorer on
/// macOS, [`CpuBlockScorer`] anywhere).
pub trait BlockShortlister {
    /// `vectors` is row-major `[frames][EMBEDDING_DIM]`.
    fn shortlist(&mut self, vectors: &[f32], frames: usize) -> Result<BlockShortlist>;
}

/// CPU reference of the compiled block scorer.
#[derive(Debug, Clone)]
pub struct CpuBlockScorer {
    /// `[EMBEDDING_DIM][SEARCH_DIM]`.
    projection: Vec<f32>,
    /// Column-major projected vocabulary, `[columns][SEARCH_DIM]`.
    columns: Vec<f32>,
    column_count: usize,
    k: usize,
}

impl CpuBlockScorer {
    /// Builds a scorer from a `[EMBEDDING_DIM][SEARCH_DIM]` projection and a
    /// `[SEARCH_DIM][columns]` projected vocabulary (the MIL layouts).
    pub fn from_parts(projection: &[f32], vocab: &[f32], columns: usize, k: usize) -> Result<Self> {
        if projection.len() != EMBEDDING_DIM * SEARCH_DIM {
            return Err(bad_config(
                "scorer projection",
                format!("holds {} values, expected 768 x 128", projection.len()),
            ));
        }
        if columns == 0 || columns % BLOCK_SIZE != 0 || vocab.len() != SEARCH_DIM * columns {
            return Err(bad_config(
                "scorer vocabulary",
                format!(
                    "{} values for {columns} columns; columns must be a positive multiple of {BLOCK_SIZE}",
                    vocab.len()
                ),
            ));
        }
        if k == 0 || k > columns / BLOCK_SIZE {
            return Err(bad_config(
                "scorer k",
                format!("{k} blocks requested of {}", columns / BLOCK_SIZE),
            ));
        }
        let mut transposed = vec![0.0f32; columns * SEARCH_DIM];
        for row in 0..SEARCH_DIM {
            for column in 0..columns {
                transposed[column * SEARCH_DIM + row] = vocab[row * columns + column];
            }
        }
        Ok(Self {
            projection: projection.to_vec(),
            columns: transposed,
            column_count: columns,
            k,
        })
    }

    /// Parses the pinned compiled weights (`weights/weight.bin`).
    pub fn from_coreml_weights(bytes: &[u8]) -> Result<Self> {
        let projection = read_fp16_blob(bytes, PROJECTION_BLOB_OFFSET, EMBEDDING_DIM * SEARCH_DIM)?;
        let vocab = read_fp16_blob(bytes, VOCAB_BLOB_OFFSET, SEARCH_DIM * SCORER_COLUMNS)?;
        Self::from_parts(&projection, &vocab, SCORER_COLUMNS, TOP_BLOCKS)
    }

    pub fn load(asset_dir: &Path) -> Result<Self> {
        let bytes = std::fs::read(asset_dir.join(SCORER_WEIGHTS))
            .map_err(|error| bad_config(SCORER_WEIGHTS, error))?;
        Self::from_coreml_weights(&bytes)
    }

    pub fn column_count(&self) -> usize {
        self.column_count
    }

    /// Projects one reader vector into the search space.
    pub fn project(&self, vector: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; SEARCH_DIM];
        for (row, &value) in vector.iter().enumerate() {
            let weights = &self.projection[row * SEARCH_DIM..(row + 1) * SEARCH_DIM];
            for (slot, &weight) in out.iter_mut().zip(weights) {
                *slot += value * weight;
            }
        }
        out
    }

    /// Projected-vocabulary column `column` (`[SEARCH_DIM]`).
    pub fn column(&self, column: usize) -> &[f32] {
        &self.columns[column * SEARCH_DIM..(column + 1) * SEARCH_DIM]
    }

    fn shortlist_frame(&self, vector: &[f32], out: &mut Vec<u32>) {
        let query = self.project(vector);
        // Sorted descending; ties keep the lower block id first.
        let mut best: Vec<(f32, u32)> = Vec::with_capacity(self.k + 1);
        for block in 0..self.column_count / BLOCK_SIZE {
            let mut score = f32::NEG_INFINITY;
            for column in block * BLOCK_SIZE..(block + 1) * BLOCK_SIZE {
                let dot: f32 = self
                    .column(column)
                    .iter()
                    .zip(&query)
                    .map(|(&a, &b)| a * b)
                    .sum();
                score = score.max(dot);
            }
            if best.len() == self.k && best.last().is_some_and(|&(low, _)| score <= low) {
                // Equal scores keep the earlier block.
                continue;
            }
            let at = best.partition_point(|&(kept, _)| kept >= score);
            best.insert(at, (score, block as u32));
            best.truncate(self.k);
        }
        // `f32::max` drops NaN dots, so every block scores at least -inf and
        // the list always fills to k (construction guarantees k <= blocks).
        out.extend(best.iter().map(|&(_, block)| block));
    }
}

impl BlockShortlister for CpuBlockScorer {
    fn shortlist(&mut self, vectors: &[f32], frames: usize) -> Result<BlockShortlist> {
        if vectors.len() != frames * EMBEDDING_DIM {
            return Err(SpeechError::Input {
                why: format!(
                    "{} reader values for {frames} frames of {EMBEDDING_DIM}",
                    vectors.len()
                ),
            });
        }
        let mut blocks = Vec::with_capacity(frames * self.k);
        for frame in vectors.chunks_exact(EMBEDDING_DIM) {
            self.shortlist_frame(frame, &mut blocks);
        }
        BlockShortlist::new(frames, self.k, blocks)
    }
}

/// Exact reranking of a frame's shortlisted blocks against the full table.
///
/// Word ids past the table (the scorer's padding columns) are skipped. Ties
/// keep the first candidate in shortlist order. Returns `None` when no
/// shortlisted id names a table row.
pub fn rerank(vector: &[f32], blocks: &[u32], table: &WordTable) -> Option<Candidate> {
    let mut best: Option<Candidate> = None;
    for &block in blocks {
        let first = block as usize * BLOCK_SIZE;
        for word in first..first + BLOCK_SIZE {
            let Ok(word_id) = u32::try_from(word) else {
                continue;
            };
            let Some(cosine) = table.dot(word_id, vector) else {
                continue;
            };
            if best.is_none_or(|kept| cosine > kept.cosine) {
                best = Some(Candidate { word_id, cosine });
            }
        }
    }
    best
}

/// Exhaustive search over every table row, the ground truth for measuring
/// how often the shortlist keeps the exact winner.
pub fn exhaustive_best(vector: &[f32], table: &WordTable) -> Option<Candidate> {
    let mut best: Option<Candidate> = None;
    for word_id in 0..table.rows() as u32 {
        let cosine = table.dot(word_id, vector)?;
        if best.is_none_or(|kept| cosine > kept.cosine) {
            best = Some(Candidate { word_id, cosine });
        }
    }
    best
}

/// Reads one FP16 blob from MIL blob storage: a 64-byte metadata record at
/// `offset` (sentinel, data type, byte size, data offset) pointing at the
/// little-endian payload.
fn read_fp16_blob(bytes: &[u8], offset: usize, values: usize) -> Result<Vec<f32>> {
    let field = |at: usize, width: usize| -> Result<&[u8]> {
        bytes.get(offset + at..offset + at + width).ok_or_else(|| {
            bad_config(
                SCORER_WEIGHTS,
                format!("blob metadata at {offset} is truncated"),
            )
        })
    };
    let u32_at = |at: usize| -> Result<u32> {
        let raw = field(at, 4)?;
        Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
    };
    let u64_at = |at: usize| -> Result<u64> {
        let raw = field(at, 8)?;
        let mut word = [0u8; 8];
        word.copy_from_slice(raw);
        Ok(u64::from_le_bytes(word))
    };
    if u32_at(0)? != BLOB_SENTINEL {
        return Err(bad_config(
            SCORER_WEIGHTS,
            format!("no blob metadata sentinel at offset {offset}"),
        ));
    }
    let dtype = u32_at(4)?;
    if dtype != BLOB_DTYPE_FP16 {
        return Err(SpeechError::Unsupported {
            why: format!(
                "scorer blob at {offset} has data type {dtype}; only FP16 (1) is verified"
            ),
        });
    }
    let size = usize::try_from(u64_at(8)?).unwrap_or(usize::MAX);
    let data = usize::try_from(u64_at(16)?).unwrap_or(usize::MAX);
    if size != values * 2 {
        return Err(bad_config(
            SCORER_WEIGHTS,
            format!(
                "blob at {offset} holds {size} bytes, expected {}",
                values * 2
            ),
        ));
    }
    let payload = data
        .checked_add(size)
        .and_then(|end| bytes.get(data..end))
        .ok_or_else(|| {
            bad_config(
                SCORER_WEIGHTS,
                format!("blob at {offset} runs past the file"),
            )
        })?;
    Ok(payload
        .chunks_exact(2)
        .map(|pair| half::f16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny scorer whose projection keeps the first 128 coordinates.
    fn identity_projection() -> Vec<f32> {
        let mut projection = vec![0.0f32; EMBEDDING_DIM * SEARCH_DIM];
        for i in 0..SEARCH_DIM {
            projection[i * SEARCH_DIM + i] = 1.0;
        }
        projection
    }

    fn basis(index: usize, scale: f32) -> Vec<f32> {
        let mut row = vec![0.0f32; EMBEDDING_DIM];
        row[index] = scale;
        row
    }

    #[test]
    fn shortlist_ranks_blocks_by_their_best_word() {
        // 8 columns = 2 blocks; column c is the basis vector e_c.
        let columns = 8;
        let mut vocab = vec![0.0f32; SEARCH_DIM * columns];
        for c in 0..columns {
            vocab[c * columns + c] = 1.0;
        }
        let mut scorer =
            CpuBlockScorer::from_parts(&identity_projection(), &vocab, columns, 2).unwrap();
        // Frame 0 points at word 6 (block 1); frame 1 at word 1 (block 0).
        let mut vectors = basis(6, 1.0);
        vectors.extend(basis(1, 1.0));
        let shortlist = scorer.shortlist(&vectors, 2).unwrap();
        assert_eq!(shortlist.row(0), &[1, 0]);
        assert_eq!(shortlist.row(1), &[0, 1]);
        // The k = 1 variant keeps only the winning block.
        let mut top1 =
            CpuBlockScorer::from_parts(&identity_projection(), &vocab, columns, 1).unwrap();
        assert_eq!(top1.shortlist(&vectors, 2).unwrap().blocks, vec![1, 0]);
    }

    #[test]
    fn rerank_skips_padding_ids_past_the_table() {
        // Five words: block 1 holds word 4 plus three padding ids.
        let mut rows = Vec::new();
        for i in 0..5 {
            rows.extend(basis(i, 1.0));
        }
        let table = WordTable::from_f32_rows(&rows, EMBEDDING_DIM).unwrap();
        let query = basis(4, 1.0);
        let best = rerank(&query, &[1, 0], &table).unwrap();
        assert_eq!(best.word_id, 4);
        assert!((best.cosine - 1.0).abs() < 1e-6);
        // A shortlist that names only padding finds nothing.
        assert_eq!(rerank(&query, &[7], &table), None);
        assert_eq!(exhaustive_best(&query, &table).unwrap().word_id, 4);
    }

    #[test]
    fn rerank_keeps_the_first_of_tied_candidates() {
        let mut rows = basis(0, 1.0);
        rows.extend(basis(0, 1.0));
        let table = WordTable::from_f32_rows(&rows, EMBEDDING_DIM).unwrap();
        assert_eq!(rerank(&basis(0, 1.0), &[0], &table).unwrap().word_id, 0);
    }

    #[test]
    fn coreml_blob_parser_validates_metadata() {
        let mut bytes = vec![0u8; 64];
        let mut record = vec![0u8; 64];
        record[..4].copy_from_slice(&BLOB_SENTINEL.to_le_bytes());
        record[4..8].copy_from_slice(&BLOB_DTYPE_FP16.to_le_bytes());
        record[8..16].copy_from_slice(&4u64.to_le_bytes());
        record[16..24].copy_from_slice(&128u64.to_le_bytes());
        bytes.extend(&record);
        bytes.extend([0x00, 0x3C, 0x00, 0xC0]);
        assert_eq!(read_fp16_blob(&bytes, 64, 2).unwrap(), vec![1.0, -2.0]);
        assert!(read_fp16_blob(&bytes, 64, 3).is_err());
        assert!(read_fp16_blob(&bytes, 0, 2).is_err());
        let mut wrong_type = bytes.clone();
        wrong_type[68] = 2;
        assert!(matches!(
            read_fp16_blob(&wrong_type, 64, 2),
            Err(SpeechError::Unsupported { .. })
        ));
        assert!(read_fp16_blob(&bytes[..130], 64, 2).is_err());
    }

    #[test]
    fn refuses_misshaped_parts() {
        let projection = identity_projection();
        assert!(
            CpuBlockScorer::from_parts(&projection[1..], &[0.0; SEARCH_DIM * 4], 4, 1).is_err()
        );
        assert!(CpuBlockScorer::from_parts(&projection, &[0.0; SEARCH_DIM * 6], 6, 1).is_err());
        assert!(CpuBlockScorer::from_parts(&projection, &[0.0; SEARCH_DIM * 4], 4, 2).is_err());
    }
}
