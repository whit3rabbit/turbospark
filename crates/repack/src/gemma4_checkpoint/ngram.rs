//! Writes `qwen4_exp`'s hashed n-gram PLE table into an install.
//!
//! The source keeps three tensors per shard (`weight`, `scales`, `biases`) and
//! 128 shards; the install keeps ONE flat array of interleaved records
//! addressed by global row id. See `model_io::ngram_table` for why that layout
//! and why it is read by `mmap`.
//!
//! **THE WRITER STREAMS SHARD BY SHARD AND NEVER HOLDS THE TABLE.** At 32.0 GB
//! it could not; `write_packed_vision` next door buffers its whole store and
//! is right to, because a tower is 879 MiB. Peak here is one shard in and one
//! shard out, about 500 MB, whatever the table's size.
//!
//! **SHARDS MUST ARRIVE IN ORDER, AND THAT IS ENFORCED RATHER THAN ASSUMED.**
//! The install's addressing is linear in the global row id precisely because
//! the shards are concatenated in id order. A walk that emitted them in
//! `BTreeMap` order would be fine and one that emitted them in HashMap order
//! would write a table whose every row is real, correctly formed, and at the
//! wrong id -- which no length check, checksum or structural validation can
//! see, and which reads as a model that produces fluent nonsense.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use super::config::Gemma4Error;

/// What the table's shape is, before any bytes are written.
///
/// Derived by the caller from the checkpoint's `config.json` and the source
/// tensor shapes, so this module never guesses a width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramTableSpec {
    /// Rows in one source shard.
    pub rows_per_shard: u64,
    /// Source shard count.
    pub shards: u64,
    /// Values in one row.
    pub head_dim: u64,
    /// Affine group size (32 for this table, against 64 elsewhere in the file).
    pub group_size: u64,
    /// Bits per value.
    pub bits: u64,
    /// ZERO-BASED index of the layer carrying the table.
    pub layer_index: u64,
}

impl NgramTableSpec {
    /// Packed-weight bytes in one row.
    pub fn weight_bytes(&self) -> u64 {
        self.head_dim * self.bits / 8
    }

    /// Scale (and, separately, bias) bytes in one row: one BF16 per group.
    pub fn companion_bytes(&self) -> u64 {
        if self.group_size == 0 {
            return 0;
        }
        self.head_dim / self.group_size * 2
    }

    /// Bytes in one interleaved record.
    pub fn record_bytes(&self) -> u64 {
        self.weight_bytes() + 2 * self.companion_bytes()
    }

    /// Total rows.
    pub fn rows(&self) -> u64 {
        self.rows_per_shard * self.shards
    }

    /// Refuses a shape this writer cannot express, before it writes anything.
    ///
    /// Mirrors `NgramTableLayout::validate`, which is the READER's copy, and
    /// they are twins rather than one shared call on purpose: this one has to
    /// fire before a 32 GB write and that one has to fire against an install
    /// whose writer may not have been this code at all.
    pub fn validate(&self) -> Result<(), Gemma4Error> {
        let bad = |detail: String| {
            Err(Gemma4Error::ShapeMismatch {
                tensor: "ngram_table".to_string(),
                detail,
            })
        };
        if self.bits != 4 {
            return bad(format!("{}-bit rows have no dequantizer here", self.bits));
        }
        if self.group_size == 0 || self.head_dim == 0 || self.head_dim % self.group_size != 0 {
            return bad(format!(
                "head_dim {} is not a whole number of {}-value groups",
                self.head_dim, self.group_size
            ));
        }
        // A row's packed run must be a whole number of bytes, or the record
        // boundary falls mid-byte and every row after the first is shifted.
        if self.head_dim * self.bits % 8 != 0 {
            return bad(format!(
                "{} values at {} bits is not a whole number of bytes",
                self.head_dim, self.bits
            ));
        }
        if self.rows_per_shard == 0 || self.shards == 0 {
            return bad("the table is empty".to_string());
        }
        Ok(())
    }
}

/// Streams the interleaved table into `<dir>/ngram_table/`.
pub struct NgramTableWriter {
    spec: NgramTableSpec,
    dir: PathBuf,
    blob: BufWriter<File>,
    /// The next shard index this writer will accept. See the module header.
    next_shard: u64,
    written_rows: u64,
}

impl NgramTableWriter {
    /// Creates the directory and opens `rows.bin`.
    pub fn create(install_dir: &Path, spec: NgramTableSpec) -> Result<Self, Gemma4Error> {
        spec.validate()?;
        let dir = install_dir.join(model_io::NGRAM_TABLE_DIR);
        std::fs::create_dir_all(&dir).map_err(|e| Gemma4Error::Config(e.to_string()))?;
        let path = dir.join(model_io::NGRAM_TABLE_BLOB);
        let blob = File::create(&path).map_err(|e| Gemma4Error::Config(e.to_string()))?;
        Ok(Self {
            spec,
            dir,
            blob: BufWriter::with_capacity(1 << 20, blob),
            next_shard: 0,
            written_rows: 0,
        })
    }

    /// Interleaves one shard's three planes and appends them.
    ///
    /// The three slices are the source tensors' raw bytes, each `rows` long in
    /// its own row width. They are copied verbatim: this is a REARRANGEMENT
    /// and not a transcode, so no value is read, rounded or reinterpreted, and
    /// the written bytes are the checkpoint's own.
    pub fn write_shard(
        &mut self,
        shard: u64,
        weight: &[u8],
        scales: &[u8],
        biases: &[u8],
    ) -> Result<(), Gemma4Error> {
        if shard != self.next_shard {
            return Err(Gemma4Error::ShapeMismatch {
                tensor: format!("ngram_table shard {shard}"),
                detail: format!(
                    "shards must be written in id order and the next is {}; out of order \
                     writes every row correctly formed and at the wrong id, which nothing \
                     downstream can detect",
                    self.next_shard
                ),
            });
        }
        let rows = self.spec.rows_per_shard as usize;
        let w = self.spec.weight_bytes() as usize;
        let c = self.spec.companion_bytes() as usize;
        for (plane, name, stride) in [
            (weight, "weight", w),
            (scales, "scales", c),
            (biases, "biases", c),
        ] {
            let want = rows * stride;
            if plane.len() != want {
                return Err(Gemma4Error::ShapeMismatch {
                    tensor: format!("ngram_table shard {shard} {name}"),
                    detail: format!("{} bytes, expected {rows} rows x {stride}", plane.len()),
                });
            }
        }

        let mut record = vec![0u8; self.spec.record_bytes() as usize];
        for r in 0..rows {
            record[..w].copy_from_slice(&weight[r * w..(r + 1) * w]);
            record[w..w + c].copy_from_slice(&scales[r * c..(r + 1) * c]);
            record[w + c..].copy_from_slice(&biases[r * c..(r + 1) * c]);
            self.blob
                .write_all(&record)
                .map_err(|e| Gemma4Error::Config(e.to_string()))?;
        }
        self.next_shard += 1;
        self.written_rows += rows as u64;
        Ok(())
    }

    /// Writes `header.json` and closes the blob.
    ///
    /// The three int64 buffers come from the CHECKPOINT rather than being
    /// recomputed from the seed. See `model_io::NgramTableLayout`: they are the
    /// table's own statement of how it is addressed, and the resident index
    /// cannot hold them because that walk narrows to BF16.
    pub fn finish(
        mut self,
        multipliers: Vec<i64>,
        head_vocab_sizes: Vec<i64>,
        head_offsets: Vec<i64>,
    ) -> Result<(), Gemma4Error> {
        if self.next_shard != self.spec.shards {
            return Err(Gemma4Error::ShapeMismatch {
                tensor: "ngram_table".to_string(),
                detail: format!(
                    "{} of {} shards written; a short table addresses rows that are not there",
                    self.next_shard, self.spec.shards
                ),
            });
        }
        self.blob
            .flush()
            .map_err(|e| Gemma4Error::Config(e.to_string()))?;
        drop(self.blob);

        let header = serde_json::json!({
            "version": model_io::NgramTableLayout::VERSION,
            "rows": self.written_rows,
            "rowsPerShard": self.spec.rows_per_shard,
            "shards": self.spec.shards,
            "headDim": self.spec.head_dim,
            "groupSize": self.spec.group_size,
            "bits": self.spec.bits,
            "recordBytes": self.spec.record_bytes(),
            "weightBytes": self.spec.weight_bytes(),
            "scaleBytes": self.spec.companion_bytes(),
            "biasBytes": self.spec.companion_bytes(),
            // The source's companions are BF16, and this is recorded rather
            // than assumed because FP16 is the same width: a wrong reading
            // passes every length check and decodes the scales as values
            // orders of magnitude off (`crates/repack` Gotcha 9).
            "companionDtype": "bf16",
            "layerIndex": self.spec.layer_index,
            "multipliers": multipliers,
            "headVocabSizes": head_vocab_sizes,
            "headOffsets": head_offsets,
        });
        let path = self.dir.join(model_io::NGRAM_TABLE_HEADER);
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&header).unwrap_or_default(),
        )
        .map_err(|e| Gemma4Error::Config(e.to_string()))?;

        // **THE WRITER VALIDATES ITS OWN OUTPUT THROUGH THE READER'S CHECK.**
        // The two are separate implementations by design, so running one over
        // the other here is what stops them drifting -- a header this walk
        // wrote and the runtime then refuses is a 68 GiB stream wasted, and
        // the refusal would arrive at open rather than at write.
        match model_io::load_ngram_table_layout(self.dir.parent().unwrap_or(Path::new("."))) {
            Ok(Some(_)) => Ok(()),
            Ok(None) => Err(Gemma4Error::Config(
                "the n-gram header was written and did not read back".to_string(),
            )),
            Err(e) => Err(Gemma4Error::Config(format!(
                "the n-gram header this walk wrote is one the reader refuses: {e:?}"
            ))),
        }
    }
}
