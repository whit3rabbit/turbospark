//! `qwen4_exp`'s hashed n-gram PLE table: the on-disk layout and its reader.
//!
//! 30.8% of the checkpoint, 32.0 GB at 4 bits, and the SECOND streamed store
//! in an install after `packed_experts/`. It holds 320,001,536 rows of 160
//! values, addressed by a hash of the token bigram and trigram, and a decode
//! reads exactly 16 of them per token.
//!
//! **IT DOES NOT REUSE `PackedExpertsLayout`, WHICH THE VISION TOWER DID.**
//! That schema is layer-by-fixed-stride-blob and the tower fits it exactly
//! (one layer of `depth` blobs). This is a flat array of 100-byte rows with no
//! layer axis and no blob boundary that means anything -- the shard split is
//! an artifact of how the checkpoint was written, not a unit anything reads.
//! Borrowing the schema would put two fictional axes on a table that has one.
//!
//! **AND IT IS READ BY `mmap` RATHER THAN THROUGH THE `pread` STREAMER, WHICH
//! IS THE OPPOSITE OF WHAT THE ROUTED EXPERTS DO.** `docs/EXPERT_RESIDENCY.md`
//! records mapping-in-place as a measured NEGATIVE, and that result does not
//! transfer here, because the two differ by three orders of magnitude in
//! record size. An expert blob is 2.76 MB, many whole pages, and a cold
//! `pread` of one reaches 9.46 GB/s at QD1. A row here is 100 BYTES, far under
//! a page, and the same sweep measures 4 KiB random reads at 0.08 GB/s at QD1
//! -- about 120x worse per byte. Sub-page random access is the worst case for
//! the streamer and the ordinary case for the page cache, so the mapping wins
//! for the same reason it lost there. Re-measure before quoting either result
//! about the other.
//!
//! ## The record
//!
//! The source stores three separate tensors per shard (`weight`, `scales`,
//! `biases`), so a naive reader issues THREE reads per row and 48 per token.
//! The writer interleaves each row's three planes into one contiguous record,
//! which takes that to 16.
//!
//! **The interleave is byte-neutral**: 320,001,536 x 100 B is 32,000,153,600,
//! exactly the sum of the source planes. There is no padding and the record is
//! deliberately not aligned to anything. Padding 100 up to 128 would cost 28%
//! of 32 GB for an alignment no reader needs, because the dequant is on the
//! HOST -- 16 rows a token is not worth a kernel until a phase table says so.
//! A future GPU reader is what would make alignment worth its 9 GB.

use std::path::Path;

use serde::Deserialize;

use crate::error::ModelError;

/// The subdirectory an install keeps its n-gram table in.
pub const NGRAM_TABLE_DIR: &str = "ngram_table";

/// The header file inside [`NGRAM_TABLE_DIR`].
pub const NGRAM_TABLE_HEADER: &str = "header.json";

/// The record file inside [`NGRAM_TABLE_DIR`].
pub const NGRAM_TABLE_BLOB: &str = "rows.bin";

/// A sanity ceiling on the header, which is a few KB of scalars plus three
/// short int64 arrays. Guards against a truncated or hostile install turning
/// into an unbounded read.
pub const NGRAM_HEADER_MAX_BYTES: u64 = 1 << 20;

/// `ngram_table/header.json`.
///
/// **THE THREE INT64 BUFFERS LIVE HERE RATHER THAN IN THE RESIDENT INDEX, AND
/// THAT IS THE POINT OF HAVING A HEADER AT ALL.** `multipliers`,
/// `head_vocab_sizes` and `head_offsets` are the table's own statement of how
/// it is addressed. The resident walk NARROWS every unquantized tensor to
/// BF16 (`crates/repack` Gotcha 9), and a 20-million-entry prime vocabulary
/// size rounded through BF16 is not a near miss -- it is a different modulus,
/// so every hash lands on the wrong row, with no error, because the bytes are
/// the right width for the tensor they were written as.
///
/// Carrying them also means nothing here ever DERIVES the hashing. The
/// reference recomputes them from a seed as a fallback, and a derived
/// convention that reads plausibly and is wrong is exactly what the control
/// vector's `direction.N` numbering cost (`crates/repack` Gotcha 11).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NgramTableLayout {
    /// Schema version. Bumped when the record layout changes.
    pub version: u32,
    /// Total addressable rows: `rows_per_shard * shards`.
    pub rows: u64,
    /// Rows in one source shard. Kept because it is what maps a global row id
    /// back to the checkpoint tensor it came from, which is the only way to
    /// re-derive a row from the source.
    pub rows_per_shard: u64,
    /// Source shard count (`split_ngram_parts`).
    pub shards: u64,
    /// Values in one row (`ple_embed_dim / ngram_heads`).
    pub head_dim: u64,
    /// Affine quantization group size. 32 here, against 64 everywhere else in
    /// this checkpoint -- the config states it per tensor and this table is
    /// the exception, so it is recorded rather than inherited.
    pub group_size: u64,
    /// Bits per value.
    pub bits: u64,
    /// Bytes per interleaved record: `weight_bytes + scale_bytes + bias_bytes`.
    pub record_bytes: u64,
    /// Packed-weight bytes at the front of a record.
    pub weight_bytes: u64,
    /// Scale-plane bytes, immediately after the weights.
    pub scale_bytes: u64,
    /// Bias-plane bytes, immediately after the scales.
    pub bias_bytes: u64,
    /// Companion plane dtype, `bf16` here. Recorded because the FP16/BF16 axis
    /// is the one that fails silently: the planes are the same width, so a
    /// wrong reading passes every length check and decodes the scales as
    /// values orders of magnitude off (`crates/repack` Gotcha 9's measured
    /// case, 0.0271 read as 1.7e-16).
    pub companion_dtype: String,
    /// ZERO-BASED index of the layer carrying this table.
    pub layer_index: u64,
    /// Hash multipliers, one per n-gram order.
    pub multipliers: Vec<i64>,
    /// Each hash head's prime vocabulary size.
    pub head_vocab_sizes: Vec<i64>,
    /// Each hash head's base offset into the global row space.
    pub head_offsets: Vec<i64>,
}

impl NgramTableLayout {
    /// The current schema version.
    pub const VERSION: u32 = 1;

    /// Byte offset of a global row id in `rows.bin`.
    ///
    /// The shards concatenate in id order, so the mapping is linear and the
    /// shard boundary never appears in the arithmetic. That is deliberate:
    /// the reference resolves `shard = gid / rows` and `row = gid % rows`
    /// only because ITS tensors are still separate, and reproducing that split
    /// here would reintroduce a boundary the writer exists to remove.
    pub fn row_offset(&self, gid: u64) -> Option<u64> {
        if gid >= self.rows {
            return None;
        }
        gid.checked_mul(self.record_bytes)
    }

    /// Total size `rows.bin` must be.
    pub fn blob_bytes(&self) -> Option<u64> {
        self.rows.checked_mul(self.record_bytes)
    }

    /// Groups per row, which is how many scale and bias entries a record has.
    pub fn groups_per_row(&self) -> u64 {
        if self.group_size == 0 {
            return 0;
        }
        self.head_dim / self.group_size
    }

    /// Checks the header describes a table this port can read.
    ///
    /// Every clause is a shape some reader strides by, and each would produce
    /// a WRONG ROW rather than an error if it were merely trusted. A header is
    /// install metadata, so it is exactly as trustworthy as the walk that
    /// wrote it -- and the walk that wrote it is the thing being changed.
    pub fn validate(&self) -> Result<(), ModelError> {
        let bad = |detail: String| {
            Err(ModelError::IndexCorrupt {
                detail: format!("ngram_table/header.json: {detail}"),
            })
        };
        if self.version != Self::VERSION {
            return bad(format!(
                "version {} is not the {} this port reads",
                self.version,
                Self::VERSION
            ));
        }
        if self.bits != 4 {
            return bad(format!("{}-bit rows have no dequantizer here", self.bits));
        }
        if self.companion_dtype != "bf16" {
            return bad(format!(
                "companion planes are {:?}; only bf16 is read, and fp16 is the \
                 same width so it would be misread rather than refused",
                self.companion_dtype
            ));
        }
        if self.group_size == 0 || self.head_dim == 0 || self.head_dim % self.group_size != 0 {
            return bad(format!(
                "head_dim {} is not a whole number of {}-value groups",
                self.head_dim, self.group_size
            ));
        }
        // The three plane widths are DERIVED from the shape, and checking them
        // against it is what catches a writer that packed at a different width
        // than it declared -- which is a record stride that reads plausible
        // bytes from the wrong place.
        let want_weight = self.head_dim * self.bits / 8;
        if self.weight_bytes != want_weight {
            return bad(format!(
                "weight_bytes {} but {} values at {} bits is {want_weight}",
                self.weight_bytes, self.head_dim, self.bits
            ));
        }
        let want_companion = self.groups_per_row() * 2;
        if self.scale_bytes != want_companion || self.bias_bytes != want_companion {
            return bad(format!(
                "scale/bias bytes {}/{} but {} bf16 groups is {want_companion} each",
                self.scale_bytes,
                self.bias_bytes,
                self.groups_per_row()
            ));
        }
        if self.record_bytes != self.weight_bytes + self.scale_bytes + self.bias_bytes {
            return bad(format!(
                "record_bytes {} is not {} + {} + {}",
                self.record_bytes, self.weight_bytes, self.scale_bytes, self.bias_bytes
            ));
        }
        if self.rows != self.rows_per_shard.saturating_mul(self.shards) {
            return bad(format!(
                "rows {} is not rows_per_shard {} x shards {}",
                self.rows, self.rows_per_shard, self.shards
            ));
        }
        if self.rows == 0 {
            return bad("the table is empty".to_string());
        }
        // The hashing buffers. `head_vocab_sizes` and `head_offsets` are one
        // per hash head and must agree; `multipliers` is one per n-gram order
        // and is a different length on purpose, so a check that required all
        // three to match would refuse every real table.
        if self.head_vocab_sizes.len() != self.head_offsets.len() {
            return bad(format!(
                "{} head vocab sizes against {} offsets",
                self.head_vocab_sizes.len(),
                self.head_offsets.len()
            ));
        }
        if self.head_vocab_sizes.is_empty() || self.multipliers.is_empty() {
            return bad("the hashing buffers are empty".to_string());
        }
        if self.head_vocab_sizes.iter().any(|v| *v <= 0) {
            return bad("a head vocabulary size is not positive".to_string());
        }
        // Every offset must be non-negative -- `ple_ngram_rows` casts it
        // straight to `u64` with no check of its own, so a negative value
        // would wrap into a huge row index rather than erroring -- and the
        // array must be ASCENDING, which is what lets the `used` check below
        // examine only the LAST entry instead of every one.
        if self.head_offsets.iter().any(|&o| o < 0) {
            return bad("a head offset is negative".to_string());
        }
        if !self.head_offsets.windows(2).all(|w| w[0] <= w[1]) {
            return bad("head_offsets is not ascending".to_string());
        }
        // The offsets partition the row space, so the last head's offset plus
        // its size is the used height. It may be UNDER `rows` -- the total is
        // padded up to `make_ngram_vocab_size_divisible_by` and then split
        // into equal shards, so trailing rows are real and unaddressed -- but
        // never over, which would index past the blob.
        let used = self
            .head_offsets
            .last()
            .copied()
            .unwrap_or(0)
            .saturating_add(self.head_vocab_sizes.last().copied().unwrap_or(0));
        if used < 0 || used as u64 > self.rows {
            return bad(format!(
                "the hash heads address {used} rows but the table holds {}",
                self.rows
            ));
        }
        Ok(())
    }
}

/// Reads and validates `<dir>/ngram_table/header.json`.
///
/// `Ok(None)` when the directory is absent, which is every install of every
/// other family and every `qwen4_exp` install written before this landed.
pub fn load_ngram_table_layout(dir: &Path) -> Result<Option<NgramTableLayout>, ModelError> {
    let path = dir.join(NGRAM_TABLE_DIR).join(NGRAM_TABLE_HEADER);
    if !path.exists() {
        return Ok(None);
    }
    let size = std::fs::metadata(&path)
        .map_err(|e| ModelError::IoFailed {
            call: "metadata".to_string(),
            detail: format!("{}: {e}", path.display()),
        })?
        .len();
    if size > NGRAM_HEADER_MAX_BYTES {
        return Err(ModelError::IndexCorrupt {
            detail: format!(
                "{} is {size} bytes, past the {NGRAM_HEADER_MAX_BYTES} ceiling",
                path.display()
            ),
        });
    }
    let raw = std::fs::read_to_string(&path).map_err(|e| ModelError::IoFailed {
        call: "read".to_string(),
        detail: format!("{}: {e}", path.display()),
    })?;
    let layout: NgramTableLayout =
        serde_json::from_str(&raw).map_err(|e| ModelError::IndexCorrupt {
            detail: format!("{}: {e}", path.display()),
        })?;
    layout.validate()?;
    Ok(Some(layout))
}

#[cfg(test)]
#[path = "ngram_table_tests.rs"]
mod tests;
