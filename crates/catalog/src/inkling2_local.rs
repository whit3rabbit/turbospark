//! Local installs of Inkling 2 Mini from the sources that need whole files on
//! disk rather than the range-streamed walk: a LookAtMyQuant packed release
//! directory (`docs/LAMQ.md`), an Inkling 2 GGUF, the original Hugging Face
//! checkpoint and an MLX conversion of it (`docs/INKLING2_PHASE0.md`). The CLI (`turbospark model pull-local`), the
//! FFI and the Hugging Face route (which downloads the files first) all come
//! through [`install_inkling2_local`], so they share one order of steps:
//!
//! 1. refuse an existing destination; check free space;
//! 2. write the install into a sibling `.partial` staging directory;
//! 3. copy the tokenizer and chat files (from the LAMQ directory, or for a
//!    GGUF, which carries none, from a directory the caller names);
//! 4. read the result back as a run would (tokenizer renders a turn; manifest,
//!    resident index and expert layout load);
//! 5. rename into place and record the install.
//!
//! A failure at any step removes the staging directory and records nothing.

use std::path::{Path, PathBuf};

use crate::install::{
    record_into, today, verify_install, verify_tokenizer, InstallTarget, Installed,
};
use crate::library::LocationEnv;
use crate::store::{directory_bytes, InstalledModel, ModelModality, Store, StoreError};

/// The local Inkling sources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Inkling2LocalSource {
    /// A LookAtMyQuant packed release directory.
    Lamq(PathBuf),
    /// An Inkling 2 GGUF file (`inkling2_loop` or `inkling2`).
    Gguf(PathBuf),
    /// A directory holding the original checkpoint: `config.json` (no MLX
    /// `quantization` block) and one `.safetensors` weights file.
    Hf(PathBuf),
    /// A directory holding an MLX conversion: `config.json` with an MLX
    /// `quantization` block and its safetensors shards.
    Mlx(PathBuf),
}

impl Inkling2LocalSource {
    pub fn path(&self) -> &Path {
        match self {
            Self::Lamq(p) | Self::Gguf(p) | Self::Hf(p) | Self::Mlx(p) => p,
        }
    }

    /// `lamq`, `gguf`, `hf` or `mlx`, for messages and the receipt.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Lamq(_) => "lamq",
            Self::Gguf(_) => "gguf",
            Self::Hf(_) => "hf",
            Self::Mlx(_) => "mlx",
        }
    }
}

/// Which safetensors Inkling source a `config.json` describes: None unless
/// it declares `model_type` `inkling2_loop`; MLX when it carries mlx-lm's
/// `quantization` block (the original checkpoint's is null and its
/// `quantization_config` names modelopt NVFP4), else the original.
pub(crate) fn inkling2_safetensors_kind(config_json: &str) -> Option<bool> {
    let v: serde_json::Value = serde_json::from_str(config_json).ok()?;
    if v.get("model_type").and_then(|t| t.as_str()) != Some("inkling2_loop") {
        return None;
    }
    Some(v.get("quantization").is_some_and(|q| q.is_object()))
}

/// The one weights file of an original-checkpoint directory: every
/// `.safetensors` file except the DSpark drafter (`mtp.safetensors`), which
/// the intake does not read. More than one is refused, since the intake
/// reads a single file.
fn hf_weights_file(dir: &Path) -> Result<PathBuf, String> {
    let mut found = Vec::new();
    for e in std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let name = e.map_err(|e| e.to_string())?.file_name();
        let name = name.to_string_lossy();
        if name.ends_with(".safetensors") && name != "mtp.safetensors" {
            found.push(dir.join(name.as_ref()));
        }
    }
    found.sort();
    match found.len() {
        1 => Ok(found.remove(0)),
        0 => Err(format!(
            "{} has no .safetensors weights file",
            dir.display()
        )),
        n => Err(format!(
            "{} has {n} .safetensors weights files; the Inkling 2 checkpoint intake reads one",
            dir.display()
        )),
    }
}

/// Which Inkling intake `path` needs, or None when it is neither a
/// LookAtMyQuant Inkling release directory nor an Inkling 2 GGUF. Reads only
/// `manifest.json` or the GGUF header.
pub fn detect_inkling2_local(path: &Path) -> Result<Option<Inkling2LocalSource>, String> {
    let meta = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if meta.is_dir() {
        let manifest = path.join("manifest.json");
        let Ok(text) = std::fs::read_to_string(&manifest) else {
            return Ok(detect_safetensors_dir(path));
        };
        // A `.gturbo` install also has a manifest.json; only the LAMQ
        // release's top-level `format` string selects this intake.
        let format = serde_json::from_str::<serde_json::Value>(&text)
            .ok()
            .and_then(|v| v.get("format").and_then(|f| f.as_str()).map(str::to_string));
        if format.as_deref() == Some(repack::LAMQ_INKLING2_FORMAT) {
            return Ok(Some(Inkling2LocalSource::Lamq(path.to_path_buf())));
        }
        // A `.gturbo` install has no config.json; a checkpoint directory
        // that also ships a manifest.json still falls through to it.
        return Ok(detect_safetensors_dir(path));
    }
    if path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
    {
        let arch = repack::inkling2_gguf_architecture(path).map_err(|e| e.0)?;
        return Ok(arch.map(|_| Inkling2LocalSource::Gguf(path.to_path_buf())));
    }
    Ok(None)
}

/// An original-checkpoint or MLX directory, from its `config.json`.
fn detect_safetensors_dir(dir: &Path) -> Option<Inkling2LocalSource> {
    let text = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let mlx = inkling2_safetensors_kind(&text)?;
    Some(if mlx {
        Inkling2LocalSource::Mlx(dir.to_path_buf())
    } else {
        Inkling2LocalSource::Hf(dir.to_path_buf())
    })
}

/// Bytes a source occupies (a file, or every file in a directory).
fn source_bytes(source: &Inkling2LocalSource) -> u64 {
    match source {
        Inkling2LocalSource::Gguf(p) => std::fs::metadata(p).map_or(0, |m| m.len()),
        Inkling2LocalSource::Lamq(p) | Inkling2LocalSource::Hf(p) | Inkling2LocalSource::Mlx(p) => {
            directory_bytes(p)
        }
    }
}

/// Free space an install needs beside its source, from the source's bytes:
/// the written install plus a margin. The LAMQ expert padding makes an
/// install about 1.43x the release (a GGUF install is about the file's
/// size, under the same bound); the original checkpoint and an MLX
/// conversion keep their expert bytes as published, so their install is
/// about the source's size. `crate::inkling2_repo` estimates from listed
/// sizes with this same function.
pub(crate) fn inkling2_install_estimate(whole_file_format: bool, source_bytes: u64) -> u64 {
    if whole_file_format {
        source_bytes.saturating_mul(3) / 2 + (1 << 30)
    } else {
        source_bytes + (1 << 30)
    }
}

/// [`inkling2_install_estimate`] for a source on disk.
pub fn inkling2_local_required_bytes(source: &Inkling2LocalSource) -> u64 {
    let packed = matches!(
        source,
        Inkling2LocalSource::Lamq(_) | Inkling2LocalSource::Gguf(_)
    );
    inkling2_install_estimate(packed, source_bytes(source))
}

/// Where a source's DSpark block drafter is, if it ships one: a `dspark/`
/// folder holding `mtp.safetensors` and its `config.json` (what the Hub
/// route assembles from a row's `mtp` repository), or, for the original
/// checkpoint, its own top-level `mtp.safetensors` beside the `config.json`
/// whose `mtp_config` describes it. The runtime loads `<install>/dspark/`
/// (`runtime::families::inkling2::dspark`). A GGUF carries none of its own,
/// so for one the `dspark/` folder beside the file counts: where the Hub
/// route puts a row's `mtp` drafter, the original release's BF16 weights
/// (unsloth's own drafter GGUFs are a Q8_0 copy of those, or a different
/// drafter; `docs/INKLING2_PHASE0.md`).
fn drafter_source(source: &Inkling2LocalSource) -> Option<(PathBuf, PathBuf)> {
    let gguf_dir;
    let dir = match source {
        Inkling2LocalSource::Gguf(f) => {
            gguf_dir = f.parent()?.to_path_buf();
            &gguf_dir
        }
        Inkling2LocalSource::Lamq(d) | Inkling2LocalSource::Hf(d) | Inkling2LocalSource::Mlx(d) => {
            d
        }
    };
    let pair = |d: &Path| {
        let (w, c) = (d.join("mtp.safetensors"), d.join("config.json"));
        (w.is_file() && c.is_file()).then_some((w, c))
    };
    pair(&dir.join("dspark")).or_else(|| match source {
        Inkling2LocalSource::Hf(d) => pair(d),
        _ => None,
    })
}

/// A DSpark drafter GGUF beside a GGUF source (`dspark-*.gguf` in the same
/// folder; the old layout first, since only it converts), or None.
fn drafter_gguf_beside(source: &Inkling2LocalSource) -> Option<PathBuf> {
    let Inkling2LocalSource::Gguf(f) = source else {
        return None;
    };
    let mut found: Vec<PathBuf> = std::fs::read_dir(f.parent()?)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("dspark") && n.ends_with(".gguf"))
        })
        .collect();
    found.sort_by_key(|p| {
        !p.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.contains("_Old"))
    });
    found.into_iter().next()
}

/// Copies the drafter pair into `dest` (`<staging>/dspark`). A copy, not a
/// move: a `pull-local` source is the user's own folder.
fn copy_drafter((weights, config): &(PathBuf, PathBuf), dest: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dest).map_err(|e| format!("creating {}: {e}", dest.display()))?;
    for (from, name) in [(weights, "mtp.safetensors"), (config, "config.json")] {
        std::fs::copy(from, dest.join(name))
            .map_err(|e| format!("copying {} into {}: {e}", from.display(), dest.display()))?;
    }
    Ok(())
}

fn read_config(dir: &Path) -> Result<String, String> {
    let p = dir.join("config.json");
    std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))
}

/// A sibling of `dest`, so the final rename stays on one filesystem.
fn staging_dir(dest: &Path) -> Result<PathBuf, String> {
    let name = dest
        .file_name()
        .ok_or_else(|| format!("{} has no directory name", dest.display()))?;
    let parent = dest.parent().filter(|p| !p.as_os_str().is_empty());
    let mut staging = std::ffi::OsString::from(".");
    staging.push(name);
    staging.push(".partial");
    Ok(parent.unwrap_or_else(|| Path::new(".")).join(staging))
}

fn refuse_existing(dest: &Path) -> Result<(), String> {
    if std::fs::symlink_metadata(dest).is_ok() {
        return Err(format!(
            "{} already exists; this never overwrites. Remove it or pick another alias",
            dest.display()
        ));
    }
    Ok(())
}

/// Writes, verifies and records an install of `source` at `target`.
/// `tokenizer_from` is required for a GGUF (a directory holding at least
/// `tokenizer.json`, such as an HF install or a download of the base repo)
/// and ignored for a LAMQ directory, which ships its own tokenizer files.
/// `alias` is the installed name; `repo` and `revision` are what the
/// receipt records as the origin.
#[allow(clippy::too_many_arguments)]
pub fn install_inkling2_local(
    source: &Inkling2LocalSource,
    alias: &str,
    target: &InstallTarget,
    tokenizer_from: Option<&Path>,
    repo: &str,
    revision: &str,
    available_bytes: u64,
    mut progress: impl FnMut(&str),
    store: &Store,
    env: &LocationEnv,
) -> Result<Installed, StoreError> {
    let failed = |e: String| StoreError::Io(e);
    let dest = &target.dir;
    refuse_existing(dest).map_err(failed)?;
    let need = inkling2_local_required_bytes(source);
    if available_bytes < need {
        return Err(failed(format!(
            "installing {} needs about {} free; {} is available where {} will be created",
            source.path().display(),
            crate::install::human_bytes(need),
            crate::install::human_bytes(available_bytes),
            dest.display()
        )));
    }
    let tokenizer_dir = match source {
        Inkling2LocalSource::Lamq(dir)
        | Inkling2LocalSource::Hf(dir)
        | Inkling2LocalSource::Mlx(dir) => dir.clone(),
        Inkling2LocalSource::Gguf(_) => tokenizer_from
            .ok_or_else(|| {
                failed(
                    "a GGUF carries no tokenizer.json: name a directory holding the Inkling 2 \
                     tokenizer files (tokenizer.json, chat_template.jinja), such as an HF \
                     install or a download of thinkingmachines/Inkling-2-Mini"
                        .to_string(),
                )
            })?
            .to_path_buf(),
    };
    if !tokenizer_dir.join("tokenizer.json").is_file() {
        return Err(failed(format!(
            "{} has no tokenizer.json",
            tokenizer_dir.display()
        )));
    }
    let staging = staging_dir(dest).map_err(failed)?;
    if std::fs::symlink_metadata(&staging).is_ok() {
        return Err(failed(format!(
            "{} is left over from an interrupted install; remove it and run again",
            staging.display()
        )));
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| failed(format!("creating {}: {e}", parent.display())))?;
    }
    std::fs::create_dir_all(&staging)
        .map_err(|e| failed(format!("creating {}: {e}", staging.display())))?;

    let model_id = format!("{repo}@{revision}");
    let built = (|| -> Result<(), String> {
        match source {
            Inkling2LocalSource::Lamq(dir) => {
                let r =
                    repack::write_inkling2_lamq_install(dir, &staging, &model_id, |l| progress(l))
                        .map_err(|e| e.0)?;
                progress(&format!(
                    "wrote {} resident tensors, experts {:.2} GB",
                    r.resident_tensors,
                    r.expert_bytes_written as f64 / 1e9
                ));
            }
            Inkling2LocalSource::Gguf(file) => {
                let r =
                    repack::write_inkling2_gguf_install(file, &staging, &model_id, |l| progress(l))
                        .map_err(|e| e.0)?;
                progress(&format!(
                    "wrote {} resident tensors from a {} GGUF",
                    r.resident_tensors, r.arch
                ));
            }
            Inkling2LocalSource::Hf(dir) => {
                let config = read_config(dir)?;
                let file = hf_weights_file(dir)?;
                let src = model_io::SafetensorsFile::open(&file)
                    .map_err(|e| format!("{}: {e}", file.display()))?;
                let r = repack::write_inkling2_install(&src, &config, &staging, &model_id, |l| {
                    progress(l)
                })
                .map_err(|e| e.0)?;
                progress(&format!(
                    "wrote {} resident tensors, {} NVFP4 and {} MXFP8 expert layers",
                    r.resident_tensors, r.nvfp4_layers, r.mxfp8_layers
                ));
            }
            Inkling2LocalSource::Mlx(dir) => {
                let config = read_config(dir)?;
                let (arch, side) = repack::parse_inkling2_config(&config).map_err(|e| e.0)?;
                let src = repack::Inkling2MlxSource::open(
                    dir,
                    arch.num_layers as usize,
                    (side.core_end - side.core_start) as usize,
                    arch.hidden_size as usize,
                )
                .map_err(|e| e.0)?;
                // The manifest declares one dense bit width.
                let bits = match src.affine_bits.iter().collect::<Vec<_>>().as_slice() {
                    [b] => **b,
                    other => {
                        return Err(format!(
                            "{}: the MLX conversion mixes affine widths {other:?}; the Inkling \
                             MLX intake takes one",
                            dir.display()
                        ))
                    }
                };
                let r = repack::write_inkling2_install_from(
                    &src,
                    &config,
                    &staging,
                    &model_id,
                    repack::inkling2_mlx_manifest_quant(bits),
                    |l| progress(l),
                )
                .map_err(|e| e.0)?;
                progress(&format!(
                    "wrote {} resident tensors ({bits}-bit affine dense weights)",
                    r.resident_tensors
                ));
            }
        }
        repack::copy_inkling2_tokenizer_files(&tokenizer_dir, &staging).map_err(|e| e.0)?;
        if let Some(from) = drafter_source(source) {
            copy_drafter(&from, &staging.join("dspark"))?;
            progress("copied the DSpark drafter (mtp.safetensors, config.json) into dspark/");
        } else if let Some(gguf) = drafter_gguf_beside(source) {
            // unsloth ships the drafter as its own GGUF beside the model; the
            // old layout converts into the pair the runtime loads, the new
            // one is a different drafter and the install goes on without it.
            match repack::write_inkling2_dspark_from_gguf(&gguf, &staging.join("dspark")) {
                Ok(r) => progress(&format!(
                    "converted the DSpark drafter {} into dspark/ ({} tensors, {} dequantized \
                     from Q8_0)",
                    gguf.display(),
                    r.tensors,
                    r.dequantized_tensors
                )),
                Err(e) => {
                    let _ = std::fs::remove_dir_all(staging.join("dspark"));
                    progress(&format!("no DSpark drafter: {e}"));
                }
            }
        }
        verify_tokenizer(&staging, &mut progress)?;
        verify_install(&staging, &model_io::inkling2_mini(), &mut progress)?;
        refuse_existing(dest)?;
        std::fs::rename(&staging, dest).map_err(|e| format!("moving the install into place: {e}"))
    })();
    if let Err(e) = built {
        // The staging directory is this run's own; nothing else writes it.
        let _ = std::fs::remove_dir_all(&staging);
        return Err(store.install_failure(&target.root, &target.root, &target.library_name, e));
    }
    let model = InstalledModel {
        alias: alias.to_string(),
        repo: repo.to_string(),
        revision: revision.to_string(),
        path: dest.canonicalize().unwrap_or_else(|_| dest.clone()),
        family: model_io::ModelFamily::Inkling2Loop.as_str().to_string(),
        install_bytes: directory_bytes(dest),
        installed_on: today(),
        status: "unlisted".to_string(),
        kind: None,
        modality: ModelModality::Text,
        variant: Some(source.kind().to_string()),
        library: None,
        relative_path: None,
    };
    let model = record_into(store, &model, target, env)?;
    Ok(Installed {
        model,
        arch: model_io::inkling2_mini(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("turbospark-ink-local-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn detects_a_lamq_release_by_its_manifest_format_only() {
        let d = tmp("lamq");
        std::fs::write(
            d.join("manifest.json"),
            r#"{"format": "inkling2-packed-v1-sqyfr"}"#,
        )
        .unwrap();
        assert_eq!(
            detect_inkling2_local(&d).unwrap(),
            Some(Inkling2LocalSource::Lamq(d.clone()))
        );
        // A .gturbo install's manifest is not a LAMQ release.
        std::fs::write(
            d.join("manifest.json"),
            r#"{"magic": "GTURBO", "arch": {}}"#,
        )
        .unwrap();
        assert_eq!(detect_inkling2_local(&d).unwrap(), None);
        let empty = tmp("empty");
        assert_eq!(detect_inkling2_local(&empty).unwrap(), None);
    }

    #[test]
    fn a_non_inkling_gguf_is_not_detected() {
        let d = tmp("gguf");
        let p = d.join("x.gguf");
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        let key = b"general.architecture";
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key);
        b.extend_from_slice(&8u32.to_le_bytes());
        for (arch, want) in [
            ("llama", false),
            ("inkling2_loop", true),
            ("inkling2", true),
        ] {
            let mut f = b.clone();
            f.extend_from_slice(&(arch.len() as u64).to_le_bytes());
            f.extend_from_slice(arch.as_bytes());
            std::fs::write(&p, &f).unwrap();
            assert_eq!(detect_inkling2_local(&p).unwrap().is_some(), want, "{arch}");
        }
        assert!(detect_inkling2_local(&d.join("missing.gguf")).is_err());
    }

    #[test]
    fn a_config_selects_the_original_or_the_mlx_intake() {
        assert_eq!(
            inkling2_safetensors_kind(
                r#"{"model_type": "inkling2_loop", "quantization": {"bits": 4}}"#
            ),
            Some(true)
        );
        assert_eq!(
            inkling2_safetensors_kind(
                r#"{"model_type": "inkling2_loop", "quantization": null, "quantization_config": {"quant_method": "modelopt"}}"#
            ),
            Some(false)
        );
        assert_eq!(
            inkling2_safetensors_kind(r#"{"model_type": "inkling2_loop"}"#),
            Some(false)
        );
        assert_eq!(
            inkling2_safetensors_kind(r#"{"model_type": "llama", "quantization": {}}"#),
            None
        );
        assert_eq!(inkling2_safetensors_kind("not json"), None);

        let d = tmp("ckpt");
        std::fs::write(d.join("config.json"), r#"{"model_type": "inkling2_loop"}"#).unwrap();
        assert_eq!(
            detect_inkling2_local(&d).unwrap(),
            Some(Inkling2LocalSource::Hf(d.clone()))
        );
        std::fs::write(
            d.join("config.json"),
            r#"{"model_type": "inkling2_loop", "quantization": {"bits": 4}}"#,
        )
        .unwrap();
        assert_eq!(
            detect_inkling2_local(&d).unwrap(),
            Some(Inkling2LocalSource::Mlx(d.clone()))
        );
    }

    #[test]
    fn the_checkpoint_intake_reads_exactly_one_weights_file_besides_the_drafter() {
        let d = tmp("weights");
        assert!(hf_weights_file(&d).is_err());
        std::fs::write(d.join("mtp.safetensors"), b"x").unwrap();
        assert!(
            hf_weights_file(&d).is_err(),
            "the drafter alone is not the weights"
        );
        std::fs::write(d.join("model-00001-of-00001.safetensors"), b"x").unwrap();
        assert_eq!(
            hf_weights_file(&d).unwrap(),
            d.join("model-00001-of-00001.safetensors")
        );
        std::fs::write(d.join("model-extra.safetensors"), b"x").unwrap();
        assert!(hf_weights_file(&d).unwrap_err().contains("reads one"));
    }

    #[test]
    fn checkpoint_installs_are_sized_at_the_source_and_packed_ones_at_1_5x() {
        assert_eq!(
            inkling2_install_estimate(false, 10 << 30),
            (10 << 30) + (1 << 30)
        );
        assert_eq!(
            inkling2_install_estimate(true, 10 << 30),
            (15 << 30) + (1 << 30)
        );
    }

    #[test]
    fn the_drafter_comes_from_a_dspark_folder_or_the_checkpoints_own_file() {
        let d = tmp("drafter");
        std::fs::write(d.join("config.json"), r#"{"model_type": "inkling2_loop"}"#).unwrap();
        let hf = Inkling2LocalSource::Hf(d.clone());
        let mlx = Inkling2LocalSource::Mlx(d.clone());
        assert_eq!(drafter_source(&hf), None, "no mtp.safetensors yet");
        std::fs::write(d.join("mtp.safetensors"), b"w").unwrap();
        assert_eq!(
            drafter_source(&hf),
            Some((d.join("mtp.safetensors"), d.join("config.json")))
        );
        // Only the original checkpoint's own file counts at the top level.
        assert_eq!(drafter_source(&mlx), None);
        let ds = d.join("dspark");
        std::fs::create_dir_all(&ds).unwrap();
        std::fs::write(ds.join("mtp.safetensors"), b"x").unwrap();
        assert_eq!(
            drafter_source(&mlx),
            None,
            "a dspark folder needs its config too"
        );
        std::fs::write(ds.join("config.json"), b"{}").unwrap();
        let pair = (ds.join("mtp.safetensors"), ds.join("config.json"));
        assert_eq!(drafter_source(&mlx), Some(pair.clone()));
        assert_eq!(
            drafter_source(&hf),
            Some(pair.clone()),
            "the dspark folder wins"
        );
        // A GGUF's drafter is the dspark folder beside the file, never a
        // top-level mtp.safetensors (that rule is the original checkpoint's).
        assert_eq!(
            drafter_source(&Inkling2LocalSource::Gguf(d.join("m.gguf"))),
            Some(pair.clone())
        );
        let lone = tmp("drafter-gguf-alone");
        std::fs::write(lone.join("mtp.safetensors"), b"w").unwrap();
        std::fs::write(lone.join("config.json"), b"{}").unwrap();
        assert_eq!(
            drafter_source(&Inkling2LocalSource::Gguf(lone.join("m.gguf"))),
            None
        );

        // A drafter GGUF beside a GGUF source: the old layout sorts first.
        let g = tmp("drafter-gguf-beside");
        assert_eq!(
            drafter_gguf_beside(&Inkling2LocalSource::Gguf(g.join("m.gguf"))),
            None
        );
        std::fs::write(g.join("dspark-Inkling-2-Mini-Q8_0.gguf"), b"n").unwrap();
        std::fs::write(g.join("dspark-Inkling-2-Mini-Q8_0_Old.gguf"), b"o").unwrap();
        assert_eq!(
            drafter_gguf_beside(&Inkling2LocalSource::Gguf(g.join("m.gguf"))),
            Some(g.join("dspark-Inkling-2-Mini-Q8_0_Old.gguf"))
        );
        assert_eq!(
            drafter_gguf_beside(&Inkling2LocalSource::Hf(g.clone())),
            None
        );

        let out = d.join("out/dspark");
        copy_drafter(&pair, &out).unwrap();
        assert_eq!(std::fs::read(out.join("mtp.safetensors")).unwrap(), b"x");
        assert_eq!(std::fs::read(out.join("config.json")).unwrap(), b"{}");
    }

    #[test]
    fn staging_is_a_hidden_sibling() {
        let s = staging_dir(Path::new("/a/b/ink.gturbo")).unwrap();
        assert_eq!(s, Path::new("/a/b/.ink.gturbo.partial"));
    }
}
