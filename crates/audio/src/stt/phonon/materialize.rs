//! Checkpoint materialization for Phonon-1.
//!
//! The packed decoder keeps its weights private inside the shared
//! `qwen3_asr` decoder, so, exactly like the Parakeet Redux reader, this port
//! materializes every checkpoint tensor to f32 in one temporary safetensors
//! file at open time and hands that file to the shared audio tower and text
//! decoder loaders. The file is removed as soon as the weights are resident.
//!
//! Conversion is strict: every tensor of the shard must be classified by the
//! packed manifest (decoder modules, hybrid embedding, hybrid audio linears)
//! or be one of the expected plain decoder norms. A checkpoint carrying any
//! other tensor is refused rather than partially loaded. The shard is
//! verified against the manifest SHA-256 before conversion.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Map, Value};
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::nn::tensor_error;
use crate::quant::{load_quantized, QuantScheme};
use crate::stt::qwen3_asr::decoder::Decoder;
use crate::stt::qwen3_asr::encoder::AudioEncoder;
use crate::{Result, SpeechError};

use super::config::{PhononConfig, ShardSpec};
use super::packed::{materialize_weight, slim_metadata, unpack_codes};

/// One converted f32 tensor: row-major values plus the shape the shared
/// loaders expect.
pub(super) struct ConvertedTensor {
    shape: Vec<usize>,
    values: Vec<f32>,
}

fn insert_tensor(
    tensors: &mut BTreeMap<String, ConvertedTensor>,
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
) -> Result<()> {
    let count = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
    if count != Some(values.len()) {
        return Err(tensor_error(
            &name,
            format!("shape {shape:?} does not match {} values", values.len()),
        ));
    }
    if tensors
        .insert(name.clone(), ConvertedTensor { shape, values })
        .is_some()
    {
        return Err(tensor_error(&name, "duplicate converted tensor name"));
    }
    Ok(())
}

/// Loads a 2-D uint8 byte-plane tensor.
fn load_u8_plane(file: &SafetensorsFile, name: &str, shape: [usize; 2]) -> Result<Vec<u8>> {
    let desc = file
        .descriptor(name)
        .ok_or_else(|| tensor_error(name, "required tensor is missing"))?;
    if desc.dtype != "U8" || desc.shape != shape {
        return Err(tensor_error(
            name,
            format!(
                "expected U8 shape {shape:?}, got {} {:?}",
                desc.dtype, desc.shape
            ),
        ));
    }
    Ok(file.raw_bytes(name)?.to_vec())
}

/// Copies one plain (F32/F16/BF16) tensor to the converted set unchanged in
/// shape.
fn plain_copy(
    file: &SafetensorsFile,
    name: &str,
    tensors: &mut BTreeMap<String, ConvertedTensor>,
) -> Result<()> {
    let desc = file
        .descriptor(name)
        .ok_or_else(|| tensor_error(name, "required tensor is missing"))?;
    if !matches!(desc.dtype.as_str(), "F16" | "BF16" | "F32") {
        return Err(tensor_error(
            name,
            format!("unexpected dtype {} for a plain tensor", desc.dtype),
        ));
    }
    let values = file.load_as_f32(name)?;
    insert_tensor(tensors, name.to_owned(), desc.shape.clone(), values)
}

/// Converts one packed decoder linear into its fused f32 weight. The
/// manifest parse already restricted the layout to the verified slim
/// quint5 form.
fn convert_module(
    file: &SafetensorsFile,
    name: &str,
    in_features: usize,
    out_features: usize,
    consumed: &mut HashSet<String>,
) -> Result<Vec<f32>> {
    let bytes_per_row = super::packed::packed_bytes_per_row(in_features)?;
    let quint5_name = format!("{name}.quint5_q");
    let plane = load_u8_plane(file, &quint5_name, [out_features, bytes_per_row])?;
    consumed.insert(quint5_name);
    let (base_q, residual_q) = unpack_codes(&plane, out_features, in_features)?;

    let groups = in_features / super::packed::GROUP_SIZE;
    let alpha_name = format!("{name}.base_alpha");
    let scale_name = format!("{name}.residual_scale");
    let alpha = file.load_as_f32(&alpha_name)?;
    let residual = file.load_as_f32(&scale_name)?;
    consumed.insert(alpha_name);
    consumed.insert(scale_name);
    let metadata = slim_metadata(&alpha, &residual, out_features, groups)?;

    materialize_weight(&base_q, &residual_q, &metadata, out_features, in_features)
}

/// Converts every shard tensor into the f32 layout the shared loaders read.
/// The manifest classifies the packed modules, the tied embedding, and the
/// audio tower linears; the remaining decoder norms are expected by name and
/// shape. Anything unclassified fails the conversion.
pub(super) fn convert_checkpoint(
    file: &SafetensorsFile,
    config: &PhononConfig,
) -> Result<BTreeMap<String, ConvertedTensor>> {
    let text = &config.backbone.text;
    let manifest = &config.manifest;
    let mut converted = BTreeMap::new();
    let mut consumed = HashSet::new();

    for module in &manifest.modules {
        let weight = convert_module(
            file,
            &module.name,
            module.in_features,
            module.out_features,
            &mut consumed,
        )?;
        insert_tensor(
            &mut converted,
            format!("{}.weight", module.name),
            vec![module.out_features, module.in_features],
            weight,
        )?;
    }

    let embedding_name = "model.embed_tokens.weight";
    match &manifest.embedding {
        Some(spec) => {
            for suffix in ["weight", "scales", "biases"] {
                let tensor = format!("model.embed_tokens.{suffix}");
                if file.contains_tensor(&tensor) {
                    consumed.insert(tensor);
                }
            }
            let (weight, _) = load_quantized(
                file,
                "model.embed_tokens",
                QuantScheme {
                    bits: spec.bits,
                    group_size: spec.group_size,
                },
            )?;
            insert_tensor(
                &mut converted,
                embedding_name.to_owned(),
                vec![spec.num_embeddings, spec.dims],
                weight,
            )?;
        }
        None => plain_copy(file, embedding_name, &mut converted).map_err(|error| {
            tensor_error(
                embedding_name,
                format!(
                    "the manifest declares no hybrid embedding and the plain load failed: {error}"
                ),
            )
        })?,
    }
    consumed.insert(embedding_name.to_owned());

    for linear in &manifest.audio_linears {
        for suffix in ["weight", "scales", "biases", "bias"] {
            let tensor = format!("{}.{}", linear.name, suffix);
            if file.contains_tensor(&tensor) {
                consumed.insert(tensor);
            }
        }
        let has_bias = file.contains_tensor(&format!("{}.bias", linear.name));
        if has_bias != linear.bias {
            return Err(tensor_error(
                &linear.name,
                "manifest bias flag does not match the checkpoint tensors",
            ));
        }
        let (weight, bias) = load_quantized(
            file,
            &linear.name,
            QuantScheme {
                bits: linear.bits,
                group_size: linear.group_size,
            },
        )?;
        insert_tensor(
            &mut converted,
            format!("{}.weight", linear.name),
            vec![linear.out_features, linear.in_features],
            weight,
        )?;
        if let Some(bias) = bias {
            insert_tensor(
                &mut converted,
                format!("{}.bias", linear.name),
                vec![linear.out_features],
                bias,
            )?;
        }
    }

    // Remaining audio tower tensors are the plain convolutions and layer
    // norms the shared encoder loads by name.
    let mut plain_audio = BTreeSet::new();
    for name in file.tensor_names() {
        if consumed.contains(name) || plain_audio.contains(name) {
            continue;
        }
        if name.starts_with("audio_tower.") {
            let base = name
                .strip_suffix(".weight")
                .or_else(|| name.strip_suffix(".bias"));
            let base = base.ok_or_else(|| {
                tensor_error(
                    name,
                    "unrecognized audio tower tensor (expected .weight or .bias)",
                )
            })?;
            if manifest
                .audio_linears
                .iter()
                .any(|linear| linear.name == base)
            {
                return Err(tensor_error(
                    name,
                    "audio linear tensor was not consumed by the manifest conversion",
                ));
            }
            plain_copy(file, name, &mut converted)?;
            plain_audio.insert(name.to_owned());
        }
    }

    // Plain decoder tensors, expected by name and width. QK norms are per
    // head dimension; the layer norms span the hidden width.
    let mut expected_plain: BTreeMap<String, usize> = BTreeMap::new();
    expected_plain.insert("model.norm.weight".to_owned(), text.hidden_size);
    for layer in 0..text.num_hidden_layers {
        for (suffix, width) in [
            ("input_layernorm.weight", text.hidden_size),
            ("post_attention_layernorm.weight", text.hidden_size),
            ("self_attn.q_norm.weight", text.head_dim),
            ("self_attn.k_norm.weight", text.head_dim),
        ] {
            expected_plain.insert(format!("model.layers.{layer}.{suffix}"), width);
        }
    }
    for (name, width) in &expected_plain {
        if converted.contains_key(name) {
            return Err(tensor_error(name, "converted twice"));
        }
        let desc = file
            .descriptor(name)
            .ok_or_else(|| tensor_error(name, "required decoder tensor is missing"))?;
        if desc.shape.len() != 1 || desc.shape[0] != *width {
            return Err(tensor_error(
                name,
                format!("expected shape [{width}], got {:?}", desc.shape),
            ));
        }
        plain_copy(file, name, &mut converted)?;
        consumed.insert(name.clone());
    }

    let unclassified: Vec<&str> = file
        .tensor_names()
        .filter(|name| !consumed.contains(*name) && !plain_audio.contains(*name))
        .collect();
    if let Some(name) = unclassified.first() {
        return Err(tensor_error(
            name,
            format!(
                "the checkpoint carries {} tensor(s) outside the verified Phonon layout, e.g. \
                 an untied output head or an unexpected quantization",
                unclassified.len()
            ),
        ));
    }

    Ok(converted)
}

/// Verifies the single weight shard against the manifest digest and opens it.
pub(super) fn open_shard(model_dir: &Path, shard: &ShardSpec) -> Result<SafetensorsFile> {
    if shard.name.ends_with(".tar.zst") || shard.name.ends_with(".bps") {
        return Err(SpeechError::Unsupported {
            why: format!(
                "shard {} is a transport archive, not a materialized weight file; materialize \
                 the release with the mlx-audio reference first",
                shard.name
            ),
        });
    }
    let path = model_dir.join(&shard.name);
    let digest = sha256_file(&path)?;
    let expected = file_digest(model_dir, shard)?;
    if digest != expected {
        return Err(tensor_error(
            &shard.name,
            format!("SHA-256 mismatch: materialized file {digest} != manifest {expected}"),
        ));
    }
    Ok(SafetensorsFile::open(&path)?)
}

/// Reads the manifest sha256 for one shard. The digest key is part of the
/// pinned manifest schema; a shard row without one is refused.
fn file_digest(model_dir: &Path, shard: &ShardSpec) -> Result<String> {
    let manifest: Value = crate::quant::read_json(&model_dir.join("packed_manifest.json"))?;
    for row in manifest
        .get("shards")
        .and_then(Value::as_array)
        .ok_or_else(|| bad_manifest("shards missing"))?
    {
        if row.get("name").and_then(Value::as_str) == Some(shard.name.as_str()) {
            return row
                .get("sha256")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| bad_manifest("shard row has no sha256"));
        }
    }
    Err(bad_manifest(format!(
        "shard {} not in manifest",
        shard.name
    )))
}

fn bad_manifest(why: impl std::fmt::Display) -> SpeechError {
    SpeechError::BadConfig {
        field: "packed_manifest.json".into(),
        why: why.to_string(),
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    turbospark_model_io::hash_file(path, 1 << 20).map_err(|error| {
        tensor_error(
            &path.display().to_string(),
            format!("cannot hash weight shard: {error}"),
        )
    })
}

pub(super) fn temporary_path() -> Result<PathBuf> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| SpeechError::BadConfig {
            field: "temporary file".into(),
            why: error.to_string(),
        })?
        .as_nanos();
    Ok(std::env::temp_dir().join(format!(
        "turbospark-phonon-{}-{nonce}.safetensors",
        std::process::id()
    )))
}

/// Serializes the converted tensors as one F32 safetensors file (the parakeet
/// redux layout).
pub(super) fn write_safetensors(
    path: &Path,
    tensors: &BTreeMap<String, ConvertedTensor>,
) -> Result<()> {
    let mut header = Map::new();
    let mut data = Vec::new();
    let total_bytes = tensors
        .values()
        .try_fold(0usize, |n, tensor| {
            n.checked_add(tensor.values.len().checked_mul(4)?)
        })
        .ok_or_else(|| SpeechError::BadConfig {
            field: "temporary file".into(),
            why: "converted tensor size overflow".into(),
        })?;
    data.try_reserve_exact(total_bytes)
        .map_err(|error| SpeechError::BadConfig {
            field: "temporary file".into(),
            why: format!("cannot allocate converted weights: {error}"),
        })?;
    for (name, tensor) in tensors {
        let start = data.len();
        for value in &tensor.values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            name.clone(),
            json!({
                "dtype": "F32",
                "shape": tensor.shape,
                "data_offsets": [start, data.len()]
            }),
        );
    }
    let mut header =
        serde_json::to_vec(&Value::Object(header)).map_err(|error| SpeechError::BadConfig {
            field: "temporary file".into(),
            why: format!("serialize header: {error}"),
        })?;
    while (8 + header.len()) % 8 != 0 {
        header.push(b' ');
    }
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| SpeechError::BadConfig {
            field: "temporary file".into(),
            why: format!("create {}: {error}", path.display()),
        })?;
    out.write_all(&(header.len() as u64).to_le_bytes())
        .and_then(|_| out.write_all(&header))
        .and_then(|_| out.write_all(&data))
        .and_then(|_| out.sync_all())
        .map_err(|error| SpeechError::BadConfig {
            field: "temporary file".into(),
            why: format!("write converted weights: {error}"),
        })?;
    Ok(())
}

/// Loads the audio tower and text decoder from the converted file and drops
/// it. The shared loaders read plain f32 tensors, so the quantization scheme
/// argument never engages (the converted file never carries `.scales`).
pub(super) fn load_models(
    converted_path: &Path,
    config: &PhononConfig,
) -> Result<(AudioEncoder, Decoder)> {
    let file = SafetensorsFile::open(converted_path)?;
    let encoder = AudioEncoder::load(&file, &config.backbone.audio)?;
    let decoder = Decoder::load(
        &file,
        &config.backbone.text,
        QuantScheme {
            bits: config.backbone.quant_bits,
            group_size: config.backbone.quant_group_size,
        },
    )?;
    Ok((encoder, decoder))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converted_safetensors_round_trips_through_the_shared_reader() {
        let tensors = BTreeMap::from([
            (
                "a.weight".to_owned(),
                ConvertedTensor {
                    shape: vec![2, 2],
                    values: vec![1.0, 2.0, 3.0, 4.0],
                },
            ),
            (
                "b.weight".to_owned(),
                ConvertedTensor {
                    shape: vec![3],
                    values: vec![0.5, -0.5, 1.5],
                },
            ),
        ]);
        let path = temporary_path().unwrap();
        write_safetensors(&path, &tensors).unwrap();
        let file = SafetensorsFile::open(&path).unwrap();
        assert_eq!(
            file.load_as_f32("a.weight").unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(file.descriptor("b.weight").unwrap().shape, vec![3]);
        std::fs::remove_file(&path).unwrap();
    }
}
