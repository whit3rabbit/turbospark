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

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use model_io::ArchConfig;

use super::config::Gemma4Error;
use super::shards::Gemma4Shards;

/// The n-gram table's own affine group size: 32, against `AFFINE_GROUP_SIZE`
/// (64) everywhere else in this checkpoint. 160 values (one head's row) is
/// not divisible by 64; it is by 32.
pub const NGRAM_GROUP_SIZE: u64 = 32;

/// Bits per value. [`NgramTableSpec::validate`] already refuses anything
/// else; named here so [`write_ngram_table`] states the constraint rather
/// than repeating the literal.
pub const NGRAM_BITS: u64 = 4;

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
    fn checked_layout(&self) -> Result<(u64, u64, u64, u64), Gemma4Error> {
        let overflow = || Gemma4Error::ShapeMismatch {
            tensor: "ngram_table".to_string(),
            detail: "table dimensions overflow the install format".to_string(),
        };
        let weight = self
            .head_dim
            .checked_mul(self.bits)
            .and_then(|bits| bits.checked_div(8))
            .ok_or_else(overflow)?;
        let companion = self
            .head_dim
            .checked_div(self.group_size)
            .and_then(|groups| groups.checked_mul(2))
            .ok_or_else(overflow)?;
        let record = companion
            .checked_mul(2)
            .and_then(|bytes| weight.checked_add(bytes))
            .ok_or_else(overflow)?;
        let rows = self
            .rows_per_shard
            .checked_mul(self.shards)
            .ok_or_else(overflow)?;
        Ok((weight, companion, record, rows))
    }

    /// Packed-weight bytes in one row.
    pub fn weight_bytes(&self) -> Result<u64, Gemma4Error> {
        self.checked_layout().map(|layout| layout.0)
    }

    /// Scale (and, separately, bias) bytes in one row: one BF16 per group.
    pub fn companion_bytes(&self) -> Result<u64, Gemma4Error> {
        self.checked_layout().map(|layout| layout.1)
    }

    /// Bytes in one interleaved record.
    pub fn record_bytes(&self) -> Result<u64, Gemma4Error> {
        self.checked_layout().map(|layout| layout.2)
    }

    /// Total rows.
    pub fn rows(&self) -> Result<u64, Gemma4Error> {
        self.checked_layout().map(|layout| layout.3)
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
        if self.head_dim.checked_mul(self.bits).map(|bits| bits % 8) != Some(0) {
            return bad(format!(
                "{} values at {} bits is not a whole number of bytes",
                self.head_dim, self.bits
            ));
        }
        if self.rows_per_shard == 0 || self.shards == 0 {
            return bad("the table is empty".to_string());
        }
        self.checked_layout()?;
        Ok(())
    }
}

/// Streams the interleaved table into `<dir>/ngram_table/`.
pub struct NgramTableWriter {
    spec: NgramTableSpec,
    /// The INSTALL directory (what `create` was called with), stored
    /// verbatim so `finish`'s read-back check reads from the directory
    /// this walk actually wrote into rather than guessing it back from
    /// `dir` via `.parent()`. On a bare relative single-segment path
    /// `.parent()` returns `""`, which `load_ngram_table_layout` reads
    /// relative to the CWD -- a directory this writer never touched.
    install_dir: PathBuf,
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
            install_dir: install_dir.to_path_buf(),
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
        let rows =
            usize::try_from(self.spec.rows_per_shard).map_err(|_| Gemma4Error::ShapeMismatch {
                tensor: "ngram_table".to_string(),
                detail: "rows per shard do not fit this platform".to_string(),
            })?;
        let w =
            usize::try_from(self.spec.weight_bytes()?).map_err(|_| Gemma4Error::ShapeMismatch {
                tensor: "ngram_table".to_string(),
                detail: "weight row width does not fit this platform".to_string(),
            })?;
        let c = usize::try_from(self.spec.companion_bytes()?).map_err(|_| {
            Gemma4Error::ShapeMismatch {
                tensor: "ngram_table".to_string(),
                detail: "companion row width does not fit this platform".to_string(),
            }
        })?;
        for (plane, name, stride) in [
            (weight, "weight", w),
            (scales, "scales", c),
            (biases, "biases", c),
        ] {
            let want = rows
                .checked_mul(stride)
                .ok_or_else(|| Gemma4Error::ShapeMismatch {
                    tensor: format!("ngram_table shard {shard} {name}"),
                    detail: "plane byte length overflows this platform".to_string(),
                })?;
            if plane.len() != want {
                return Err(Gemma4Error::ShapeMismatch {
                    tensor: format!("ngram_table shard {shard} {name}"),
                    detail: format!("{} bytes, expected {rows} rows x {stride}", plane.len()),
                });
            }
        }

        let record_bytes =
            usize::try_from(self.spec.record_bytes()?).map_err(|_| Gemma4Error::ShapeMismatch {
                tensor: "ngram_table".to_string(),
                detail: "record width does not fit this platform".to_string(),
            })?;
        let mut record = Vec::new();
        record
            .try_reserve_exact(record_bytes)
            .map_err(|e| Gemma4Error::ShapeMismatch {
                tensor: "ngram_table".to_string(),
                detail: format!("cannot allocate {record_bytes}-byte row record: {e}"),
            })?;
        record.resize(record_bytes, 0);
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
            "recordBytes": self.spec.record_bytes()?,
            "weightBytes": self.spec.weight_bytes()?,
            "scaleBytes": self.spec.companion_bytes()?,
            "biasBytes": self.spec.companion_bytes()?,
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
        let bytes = serde_json::to_vec_pretty(&header).map_err(|e| {
            Gemma4Error::Config(format!("serializing ngram_table/header.json: {e}"))
        })?;
        std::fs::write(&path, bytes).map_err(|e| Gemma4Error::Config(e.to_string()))?;

        // **THE WRITER VALIDATES ITS OWN OUTPUT THROUGH THE READER'S CHECK.**
        // The two are separate implementations by design, so running one over
        // the other here is what stops them drifting -- a header this walk
        // wrote and the runtime then refuses is a 68 GiB stream wasted, and
        // the refusal would arrive at open rather than at write.
        match model_io::load_ngram_table_layout(&self.install_dir) {
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

/// The n-gram table's classified names, OWNED so they outlive the borrowed
/// `Gemma4Shards<'a>` a `Gemma4RepackOutput` is built from and built once by
/// [`orchestrate::classify_all`](super::orchestrate::classify_all)'s caller.
///
/// Carries names, not bytes: see [`Gemma4RepackOutput::ngram`]'s doc for why.
/// [`write_ngram_table`] is what turns this into an install, and it is called
/// from both writer entry points in `mod.rs` rather than from here, because
/// only they have the install directory this struct does not.
#[derive(Debug, Clone, Default)]
pub struct NgramPlan {
    /// Shard index -> role (`weight`/`scales`/`biases`) -> source tensor name.
    pub shards: BTreeMap<usize, BTreeMap<&'static str, String>>,
    /// Field name -> source tensor name, for the three hashing buffers.
    pub meta: BTreeMap<&'static str, String>,
}

impl NgramPlan {
    /// Owns a copy of [`super::orchestrate::ClassifiedNames`]'s borrowed
    /// `ngram_shards`/`ngram_meta` maps.
    pub fn from_classified(
        shards: &BTreeMap<usize, BTreeMap<&'static str, &str>>,
        meta: &BTreeMap<&'static str, &str>,
    ) -> Self {
        Self {
            shards: shards
                .iter()
                .map(|(&shard, roles)| {
                    let roles = roles
                        .iter()
                        .map(|(&role, &name)| (role, name.to_string()))
                        .collect();
                    (shard, roles)
                })
                .collect(),
            meta: meta
                .iter()
                .map(|(&field, &name)| (field, name.to_string()))
                .collect(),
        }
    }
}

/// Streams `qwen4_exp`'s hashed n-gram PLE table from the checkpoint into
/// `<dir>/ngram_table/`, one shard at a time. A no-op when `plan` classified
/// no n-gram tensors, which is every family but `qwen4_exp`.
///
/// **CALLED FROM BOTH WRITER ENTRY POINTS IN `mod.rs`, FOR A DIFFERENT REASON
/// THAN THE MTP HEAD AND THE VISION TOWER ARE.** Those two are read into
/// memory once inside `orchestrate_gemma4_checkpoint_sharded` and carried in
/// `Gemma4RepackOutput`, so the non-streamed writer gets them for free. This
/// table is 32 GB and this module's whole point is that nothing holds it --
/// so the read and the write are the SAME loop, and that loop needs `dir`,
/// which `orchestrate_gemma4_checkpoint_sharded`'s pure, in-memory contract
/// does not carry. `write_gemma4_install` and `write_gemma4_install_streamed`
/// both call this directly instead, immediately after they have a directory
/// to write into.
pub fn write_ngram_table(
    shards: &Gemma4Shards<'_>,
    arch: &ArchConfig,
    plan: &NgramPlan,
    dir: &Path,
    mut progress: impl FnMut(&str),
) -> Result<(), Gemma4Error> {
    if plan.shards.is_empty() {
        return Ok(());
    }
    let layer_indices = arch.ple.layer_indices();
    let [layer_index] = layer_indices[..] else {
        return Err(Gemma4Error::Config(format!(
            "arch.ple declares {} PLE layers but this walk found an n-gram table on disk; \
             placing it needs exactly one",
            layer_indices.len()
        )));
    };

    // `rows_per_shard` read off the ARTIFACT (the first shard's own tensor
    // shape) rather than derived from config arithmetic -- Gotcha 62's rule,
    // read the artifact rather than the note about it. Every shard shares the
    // same row count: the concatenated table is padded to a multiple of
    // `split_ngram_parts` before the split, precisely so it divides evenly.
    let (&first_shard, first_roles) = plan.shards.iter().next().expect("checked non-empty above");
    let first_weight = ngram_tensor_name(first_roles, first_shard, "weight")?;
    let rows_per_shard =
        *shards
            .info(first_weight)?
            .shape
            .first()
            .ok_or_else(|| Gemma4Error::ShapeMismatch {
                tensor: first_weight.to_string(),
                detail: "ngram shard weight tensor has no rows dimension".to_string(),
            })?;

    let spec = NgramTableSpec {
        rows_per_shard,
        shards: u64::try_from(plan.shards.len()).map_err(|_| Gemma4Error::ShapeMismatch {
            tensor: "ngram_table".to_string(),
            detail: "shard count does not fit the install format".to_string(),
        })?,
        head_dim: u64::try_from(arch.ple.head_dim()).map_err(|_| {
            Gemma4Error::Config(format!(
                "PLE head dimension {} is not positive",
                arch.ple.head_dim()
            ))
        })?,
        group_size: NGRAM_GROUP_SIZE,
        bits: NGRAM_BITS,
        layer_index: u64::try_from(layer_index).map_err(|_| {
            Gemma4Error::Config(format!("PLE layer index {layer_index} is negative"))
        })?,
    };

    let mut writer = NgramTableWriter::create(dir, spec)?;
    for (&shard, roles) in &plan.shards {
        let weight_name = ngram_tensor_name(roles, shard, "weight")?;
        let scale_name = ngram_tensor_name(roles, shard, "scales")?;
        let bias_name = ngram_tensor_name(roles, shard, "biases")?;
        validate_plane_metadata(
            shards,
            weight_name,
            "U32",
            rows_per_shard,
            spec.weight_bytes()?,
            4,
        )?;
        validate_plane_metadata(
            shards,
            scale_name,
            "BF16",
            rows_per_shard,
            spec.companion_bytes()?,
            2,
        )?;
        validate_plane_metadata(
            shards,
            bias_name,
            "BF16",
            rows_per_shard,
            spec.companion_bytes()?,
            2,
        )?;
        let weight = shards.read(weight_name)?;
        let scales = shards.read(scale_name)?;
        let biases = shards.read(bias_name)?;
        writer.write_shard(
            u64::try_from(shard).map_err(|_| Gemma4Error::ShapeMismatch {
                tensor: format!("ngram_table shard {shard}"),
                detail: "shard index does not fit the install format".to_string(),
            })?,
            &weight,
            &scales,
            &biases,
        )?;
        progress(&format!("n-gram shard {shard} of {} written", spec.shards));
    }

    let multipliers = read_i64_buffer(shards, plan, "layer_multipliers")?;
    let head_vocab_sizes = read_i64_buffer(shards, plan, "ngram_heads_vocab_sizes")?;
    let head_offsets = read_i64_buffer(shards, plan, "ngram_heads_offsets")?;
    let rows = spec.rows()?;
    writer.finish(multipliers, head_vocab_sizes, head_offsets)?;
    progress(&format!("n-gram table written ({rows} rows)"));
    Ok(())
}

fn validate_plane_metadata(
    shards: &Gemma4Shards<'_>,
    name: &str,
    dtype: &str,
    rows: u64,
    row_bytes: u64,
    element_bytes: u64,
) -> Result<(), Gemma4Error> {
    let info = shards.info(name)?;
    if info.dtype != dtype {
        return Err(Gemma4Error::UnsupportedDtype {
            tensor: name.to_string(),
            dtype: format!("{} in n-gram table (expected {dtype})", info.dtype),
        });
    }
    let cols = row_bytes
        .checked_div(element_bytes)
        .ok_or_else(|| Gemma4Error::ShapeMismatch {
            tensor: name.to_string(),
            detail: "element width is zero".to_string(),
        })?;
    if info.shape.as_slice() != [rows, cols] {
        return Err(Gemma4Error::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!("shape {:?}, expected [{rows}, {cols}]", info.shape),
        });
    }
    let expected = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| Gemma4Error::ShapeMismatch {
            tensor: name.to_string(),
            detail: "tensor byte length overflows the install format".to_string(),
        })?;
    let actual = info
        .data_offsets
        .1
        .checked_sub(info.data_offsets.0)
        .ok_or_else(|| Gemma4Error::ShapeMismatch {
            tensor: name.to_string(),
            detail: "tensor data range is inverted".to_string(),
        })?;
    if actual != expected {
        return Err(Gemma4Error::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!("data range is {actual} bytes, expected {expected}"),
        });
    }
    Ok(())
}

fn ngram_tensor_name<'a>(
    roles: &'a BTreeMap<&'static str, String>,
    shard: usize,
    role: &str,
) -> Result<&'a str, Gemma4Error> {
    roles
        .get(role)
        .map(String::as_str)
        .ok_or_else(|| Gemma4Error::MissingTensor(format!("ngram_table shard {shard} {role}")))
}

/// Decodes one of the table's three int64 hashing buffers to `Vec<i64>`.
///
/// These are read VERBATIM, never derived: `NgramTableWriter::finish`'s doc
/// says why (a checkpoint that ever ships them wrong and a port that only
/// derives them are the same silent failure from opposite directions).
fn read_i64_buffer(
    shards: &Gemma4Shards<'_>,
    plan: &NgramPlan,
    field: &'static str,
) -> Result<Vec<i64>, Gemma4Error> {
    let name = plan
        .meta
        .get(field)
        .ok_or_else(|| Gemma4Error::MissingTensor(format!("ngram_table {field}")))?;
    let info = shards.info(name)?;
    if info.dtype != "I64" {
        return Err(Gemma4Error::UnsupportedDtype {
            tensor: name.clone(),
            dtype: format!("{} in n-gram buffer {field} (expected I64)", info.dtype),
        });
    }
    let bytes = shards.read(name)?;
    if bytes.len() % 8 != 0 {
        return Err(Gemma4Error::ShapeMismatch {
            tensor: name.clone(),
            detail: format!("{} bytes is not a whole number of i64 values", bytes.len()),
        });
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("chunks_exact(8)")))
        .collect())
}
