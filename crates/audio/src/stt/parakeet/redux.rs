//! Reader for Moondream's pinned `thrush-ternary-v2` Parakeet Redux export.
//!
//! The source stores each five ternary weights in one U8 plus one scale per
//! 128 values. The Rust CPU encoder currently expands these values to F32 in
//! a temporary safetensors file at open time; the file is removed as soon as
//! the model weights are resident.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Map, Value};
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::{Result, SpeechError};

fn bad(field: &str, why: impl Into<String>) -> SpeechError {
    SpeechError::BadConfig {
        field: field.to_string(),
        why: why.into(),
    }
}

fn tensor_error(name: &str, why: impl Into<String>) -> SpeechError {
    SpeechError::Tensor {
        name: name.to_string(),
        why: why.into(),
    }
}

fn positive(v: &Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(Value::as_u64)
        .and_then(|x| usize::try_from(x).ok())
        .filter(|&x| x > 0)
        .ok_or_else(|| bad(key, "must be a positive integer"))
}

/// Adapts the source Redux config and BPE vocabulary to the common TDT
/// config accepted by the shared Parakeet model implementation.
pub(super) fn normalized_config(model_dir: &Path, source: &Value) -> Result<Value> {
    let enc = source
        .get("encoder_config")
        .ok_or_else(|| bad("encoder_config", "missing from Redux config"))?;
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer: Value = serde_json::from_slice(&fs::read(&tokenizer_path).map_err(|e| {
        bad(
            "tokenizer.json",
            format!("failed to read {}: {e}", tokenizer_path.display()),
        )
    })?)
    .map_err(|e| bad("tokenizer.json", format!("invalid JSON: {e}")))?;
    let vocab_map = tokenizer
        .get("model")
        .and_then(|v| v.get("vocab"))
        .and_then(Value::as_object)
        .ok_or_else(|| bad("tokenizer.json", "model.vocab must be an object"))?;
    let vocab_size = positive(source, "blank_token_id")?;
    if vocab_map.len() != vocab_size {
        return Err(bad(
            "tokenizer.json",
            format!(
                "expected {vocab_size} non-blank tokens, got {}",
                vocab_map.len()
            ),
        ));
    }
    let mut vocabulary = vec![String::new(); vocab_size];
    for (piece, id) in vocab_map {
        let id = id
            .as_u64()
            .and_then(|x| usize::try_from(x).ok())
            .filter(|&x| x < vocab_size)
            .ok_or_else(|| {
                bad(
                    "tokenizer.json",
                    "vocabulary IDs must be contiguous before blank",
                )
            })?;
        if !vocabulary[id].is_empty() {
            return Err(bad("tokenizer.json", "duplicate vocabulary ID"));
        }
        vocabulary[id] = piece.clone();
    }
    if vocabulary.iter().any(String::is_empty) {
        return Err(bad("tokenizer.json", "vocabulary IDs are not contiguous"));
    }
    let added = tokenizer
        .get("added_tokens")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("tokenizer.json", "added_tokens must be an array"))?;
    let blank_id = added
        .iter()
        .find(|token| token.get("content").and_then(Value::as_str) == Some("<blank>"))
        .and_then(|token| token.get("id"))
        .and_then(Value::as_u64)
        .and_then(|x| usize::try_from(x).ok())
        .ok_or_else(|| bad("tokenizer.json", "missing <blank> token"))?;
    if blank_id != vocab_size || positive(source, "vocab_size")? != vocab_size + 1 {
        return Err(bad(
            "blank_token_id",
            "blank must immediately follow the text vocabulary",
        ));
    }

    let hidden = positive(enc, "hidden_size")?;
    let intermediate = positive(enc, "intermediate_size")?;
    let factor = positive(enc, "subsampling_factor")?;
    if intermediate % hidden != 0 {
        return Err(bad(
            "encoder_config.intermediate_size",
            "must be divisible by hidden_size",
        ));
    }
    if enc.get("model_type").and_then(Value::as_str) != Some("parakeet_encoder")
        || enc.get("hidden_act").and_then(Value::as_str) != Some("silu")
        || enc.get("attention_bias").and_then(Value::as_bool) != Some(false)
        || enc.get("convolution_bias").and_then(Value::as_bool) != Some(false)
        || enc.get("num_attention_heads") != enc.get("num_key_value_heads")
        || !matches!(factor, 2 | 4 | 8)
        || !matches!(positive(enc, "num_mel_bins")?, 80 | 128)
        || positive(enc, "num_mel_bins")? % factor != 0
        || positive(enc, "subsampling_conv_kernel_size")? != 3
        || positive(enc, "subsampling_conv_stride")? != 2
        || source.get("hidden_act").and_then(Value::as_str) != Some("relu")
    {
        return Err(SpeechError::Unsupported {
            why: "unsupported Parakeet Redux encoder or ternary architecture".into(),
        });
    }
    let sample_rate = 16_000usize;
    let layers = positive(enc, "num_hidden_layers")?;
    let heads = positive(enc, "num_attention_heads")?;
    let mel_bins = positive(enc, "num_mel_bins")?;
    let decoder_hidden = positive(source, "decoder_hidden_size")?;
    let decoder_layers = positive(source, "num_decoder_layers")?;
    let durations = source
        .get("durations")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("durations", "must be an array"))?;
    let max_symbols = positive(source, "max_symbols_per_step")?;
    let config = json!({
        "preprocessor": {
            "sample_rate": sample_rate,
            "normalize": "per_feature",
            "window_size": 0.025,
            "window_stride": 0.01,
            "window": "hann",
            "features": mel_bins,
            "n_fft": 512,
            "log": true,
            "frame_splicing": 1,
            "dither": 0.0,
            "normalize_valid_frames": true
        },
        "encoder": {
            "feat_in": mel_bins,
            "n_layers": layers,
            "d_model": hidden,
            "use_bias": false,
            "subsampling": "dw_striding",
            "subsampling_factor": factor,
            "subsampling_conv_channels": positive(enc, "subsampling_conv_channels")?,
            "subsampling_conv_kernel_size": 3,
            "subsampling_conv_stride": 2,
            "causal_downsampling": false,
            "reduction": null,
            "reduction_position": null,
            "reduction_factor": 1,
            "ff_expansion_factor": intermediate / hidden,
            "self_attention_model": "rel_pos",
            "n_heads": heads,
            "att_context_size": [-1, -1],
            "att_context_style": "regular",
            "xscaling": enc.get("scale_input").and_then(Value::as_bool).unwrap_or(false),
            "untie_biases": true,
            "pos_emb_max_len": positive(enc, "max_position_embeddings")?,
            "conv_kernel_size": positive(enc, "conv_kernel_size")?,
            "conv_norm_type": "batch_norm",
            "conv_context_size": null
        },
        "decoder": {
            "blank_as_pad": true,
            "vocab_size": vocab_size,
            "prednet": {
                "pred_hidden": decoder_hidden,
                "pred_rnn_layers": decoder_layers
            }
        },
        "joint": {
            "num_classes": vocab_size,
            "num_extra_outputs": durations.len(),
            "vocabulary": vocabulary,
            "jointnet": {
                "joint_hidden": decoder_hidden,
                "activation": "relu",
                "encoder_hidden": hidden,
                "pred_hidden": decoder_hidden
            }
        },
        "decoding": {
            "model_type": "tdt",
            "durations": durations,
            "greedy": {"max_symbols": max_symbols}
        }
    });
    Ok(config)
}

#[derive(Clone)]
struct ConvertedTensor {
    shape: Vec<usize>,
    values: Vec<f32>,
}

/// Converts the pinned raw HF safetensors into the shared loader's MLX-like
/// tensor names. It verifies the format manifest before expanding any codes.
pub(super) fn convert_checkpoint(model_dir: &Path) -> Result<PathBuf> {
    let source_config: Value = serde_json::from_slice(
        &fs::read(model_dir.join("config.json"))
            .map_err(|e| bad("config.json", format!("failed to read Redux config: {e}")))?,
    )
    .map_err(|e| bad("config.json", format!("invalid JSON: {e}")))?;
    let group_size = positive(&source_config, "ternary_group_size")?;
    if group_size != 128 {
        return Err(SpeechError::Unsupported {
            why: "Redux ternary group size must be 128".into(),
        });
    }
    let modules = source_config
        .get("ternary_modules")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("ternary_modules", "must be an array"))?;
    let mut module_shapes = HashMap::new();
    for module in modules {
        let name = module
            .as_str()
            .ok_or_else(|| bad("ternary_modules", "entries must be strings"))?;
        let name_owned = name.to_string();
        if module_shapes.contains_key(name) {
            return Err(bad("ternary_modules", "duplicate module name"));
        }
        module_shapes.insert(name_owned, (0usize, 0usize, false));
    }

    let manifest_path = model_dir.join("ternary.json");
    let manifest: Value = serde_json::from_slice(&fs::read(&manifest_path).map_err(|e| {
        bad(
            "ternary.json",
            format!("failed to read {}: {e}", manifest_path.display()),
        )
    })?)
    .map_err(|e| bad("ternary.json", format!("invalid JSON: {e}")))?;
    let quant = manifest
        .get("quant")
        .ok_or_else(|| bad("ternary.json", "missing quant metadata"))?;
    let packing = manifest
        .get("packing")
        .ok_or_else(|| bad("ternary.json", "missing packing metadata"))?;
    if manifest.get("format").and_then(Value::as_str) != Some("thrush-ternary-v2")
        || manifest.get("names").and_then(Value::as_str) != Some("hf")
        || quant.get("mode").and_then(Value::as_str) != Some("ternary")
        || quant.get("group_size").and_then(Value::as_u64) != Some(128)
        || packing.get("base").and_then(Value::as_u64) != Some(3)
        || packing.get("elements_per_byte").and_then(Value::as_u64) != Some(5)
        || packing.get("code_offset").and_then(Value::as_u64) != Some(1)
    {
        return Err(SpeechError::Unsupported {
            why: "unsupported Parakeet Redux ternary manifest".into(),
        });
    }
    let entries = manifest
        .get("quantized_modules")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("ternary.json", "quantized_modules must be an array"))?;
    if entries.len() != module_shapes.len() {
        return Err(bad(
            "ternary.json",
            "module list length disagrees with config",
        ));
    }
    let mut manifest_seen = HashSet::new();
    for entry in entries {
        let name = entry
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("ternary.json", "module entry missing name"))?;
        if !manifest_seen.insert(name.to_string()) {
            return Err(bad("ternary.json", "duplicate quantized module name"));
        }
        let slot = module_shapes
            .get_mut(name)
            .ok_or_else(|| bad("ternary.json", format!("unexpected module {name}")))?;
        let out = positive(entry, "out_features")?;
        let input = positive(entry, "in_features")?;
        let entry_group = positive(entry, "group_size")?;
        if entry_group != group_size
            || entry.get("has_bias").and_then(Value::as_bool) != Some(false)
            || input % group_size != 0
        {
            return Err(bad(
                "ternary.json",
                format!("unsupported shape or bias for {name}"),
            ));
        }
        *slot = (
            out,
            input,
            entry.get("as_conv1d").and_then(Value::as_bool) == Some(true),
        );
    }
    if manifest_seen.len() != module_shapes.len() {
        return Err(bad("ternary.json", "module list does not match config"));
    }

    let input_path = model_dir.join("model.safetensors");
    let file = SafetensorsFile::open(&input_path)?;
    let mut tensors = BTreeMap::<String, ConvertedTensor>::new();
    let mut quant_seen = HashSet::new();
    let mut scales_seen = HashSet::new();
    let mut lstm_biases = HashMap::<(usize, bool), Vec<f32>>::new();
    for name in file.tensor_names() {
        if is_ignored(name) {
            continue;
        }
        if let Some(module) = name.strip_suffix(".qweight") {
            let &(out, input, as_conv1d) = module_shapes
                .get(module)
                .ok_or_else(|| tensor_error(name, "not listed in the ternary config"))?;
            let scales_name = format!("{module}.scales");
            let values = dequantize_ternary(&file, name, &scales_name, out, input, group_size)?;
            quant_seen.insert(module.to_string());
            let mapped = map_name(module)?;
            let mut shape = vec![out, input];
            if as_conv1d {
                shape.push(1);
            }
            insert_tensor(&mut tensors, format!("{mapped}.weight"), shape, values)?;
            continue;
        }
        if name.ends_with(".scales") {
            let module = name.strip_suffix(".scales").unwrap_or_default();
            if !module_shapes.contains_key(module) {
                return Err(tensor_error(name, "not listed in the ternary config"));
            }
            scales_seen.insert(module.to_string());
            continue;
        }
        if let Some((layer, input_bias)) = parse_lstm_bias(name) {
            lstm_biases.insert(
                (layer, input_bias),
                file.load_as_f32(name)
                    .map_err(|e| tensor_error(name, format!("failed to read LSTM bias: {e}")))?,
            );
            continue;
        }
        let desc = file
            .descriptor(name)
            .ok_or_else(|| tensor_error(name, "missing descriptor"))?;
        if !matches!(desc.dtype.as_str(), "F16" | "BF16" | "F32") {
            return Err(tensor_error(
                name,
                format!("unexpected dtype {}", desc.dtype),
            ));
        }
        let mapped = map_name(name)?;
        let values = file
            .load_as_f32(name)
            .map_err(|e| tensor_error(name, format!("failed to convert to F32: {e}")))?;
        insert_tensor(&mut tensors, mapped, desc.shape.clone(), values)?;
    }
    if quant_seen.len() != module_shapes.len()
        || scales_seen != quant_seen
        || quant_seen
            .iter()
            .any(|name| !module_shapes.contains_key(name))
    {
        return Err(bad(
            "model.safetensors",
            "ternary weights do not match the manifest module list",
        ));
    }
    let decoder_layers = positive(&source_config, "num_decoder_layers")?;
    for layer in 0..decoder_layers {
        let input_bias = lstm_biases
            .remove(&(layer, true))
            .ok_or_else(|| tensor_error("decoder.lstm", "missing input bias"))?;
        let recurrent_bias = lstm_biases
            .remove(&(layer, false))
            .ok_or_else(|| tensor_error("decoder.lstm", "missing recurrent bias"))?;
        if input_bias.len() != recurrent_bias.len() {
            return Err(tensor_error("decoder.lstm", "bias shapes disagree"));
        }
        let bias: Vec<f32> = input_bias
            .into_iter()
            .zip(recurrent_bias)
            .map(|(a, b)| a + b)
            .collect();
        insert_tensor(
            &mut tensors,
            format!("decoder.prediction.dec_rnn.lstm.{layer}.bias"),
            vec![bias.len()],
            bias,
        )?;
    }
    if !lstm_biases.is_empty() {
        return Err(tensor_error("decoder.lstm", "unexpected extra LSTM biases"));
    }
    let output = temporary_path()?;
    if let Err(error) = write_safetensors(&output, tensors) {
        let _ = fs::remove_file(&output);
        return Err(error);
    }
    Ok(output)
}

fn is_ignored(name: &str) -> bool {
    name.starts_with("vad_head.") || name.ends_with(".conv.norm.num_batches_tracked")
}

fn parse_lstm_bias(name: &str) -> Option<(usize, bool)> {
    for (suffix, input) in [("bias_ih_l", true), ("bias_hh_l", false)] {
        if let Some(layer) = name.strip_prefix("decoder.lstm.")?.strip_prefix(suffix) {
            return layer.parse().ok().map(|i| (i, input));
        }
    }
    None
}

fn map_name(source: &str) -> Result<String> {
    if let Some(rest) = source.strip_prefix("decoder.lstm.weight_ih_l") {
        let (layer, tail) = split_layer_suffix(rest, source)?;
        if !tail.is_empty() {
            return Err(tensor_error(
                source,
                "unexpected suffix after LSTM input weight",
            ));
        }
        return Ok(format!("decoder.prediction.dec_rnn.lstm.{layer}.Wx"));
    }
    if let Some(rest) = source.strip_prefix("decoder.lstm.weight_hh_l") {
        let (layer, tail) = split_layer_suffix(rest, source)?;
        if !tail.is_empty() {
            return Err(tensor_error(
                source,
                "unexpected suffix after LSTM recurrent weight",
            ));
        }
        return Ok(format!("decoder.prediction.dec_rnn.lstm.{layer}.Wh"));
    }
    let mut name = source
        .replace("encoder.subsampling.layers.", "encoder.pre_encode.conv.")
        .replace("encoder.subsampling.linear", "encoder.pre_encode.out")
        .replace(".conv.norm.", ".conv.batch_norm.")
        .replace(".self_attn.q_proj", ".self_attn.linear_q")
        .replace(".self_attn.k_proj", ".self_attn.linear_k")
        .replace(".self_attn.v_proj", ".self_attn.linear_v")
        .replace(".self_attn.o_proj", ".self_attn.linear_out")
        .replace(".self_attn.relative_k_proj", ".self_attn.linear_pos")
        .replace(".self_attn.bias_u", ".self_attn.pos_bias_u")
        .replace(".self_attn.bias_v", ".self_attn.pos_bias_v")
        .replace("decoder.embedding.", "decoder.prediction.embed.")
        .replace("decoder.decoder_projector.", "joint.pred.")
        .replace("encoder_projector.", "joint.enc.")
        .replace("joint.head.", "joint.joint_net.2.");
    if name == source {
        name = source.to_string();
    }
    Ok(name)
}

fn split_layer_suffix<'a>(value: &'a str, full: &str) -> Result<(usize, &'a str)> {
    let digits = value.bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 {
        return Err(tensor_error(full, "invalid LSTM layer index"));
    }
    let layer = value[..digits]
        .parse::<usize>()
        .map_err(|_| tensor_error(full, "invalid LSTM layer index"))?;
    Ok((layer, &value[digits..]))
}

fn dequantize_ternary(
    file: &SafetensorsFile,
    qweight_name: &str,
    scales_name: &str,
    out: usize,
    input: usize,
    group_size: usize,
) -> Result<Vec<f32>> {
    let qdesc = file
        .descriptor(qweight_name)
        .ok_or_else(|| tensor_error(qweight_name, "missing descriptor"))?;
    if group_size == 0 {
        return Err(tensor_error(scales_name, "group size must be positive"));
    }
    let row_bytes = input.div_ceil(5);
    if qdesc.dtype != "U8" || qdesc.shape != [out, row_bytes] {
        return Err(tensor_error(
            qweight_name,
            format!(
                "expected U8 [{out}, {row_bytes}], found {} {:?}",
                qdesc.dtype, qdesc.shape
            ),
        ));
    }
    let scales_desc = file
        .descriptor(scales_name)
        .ok_or_else(|| tensor_error(scales_name, "missing descriptor"))?;
    let groups = input / group_size;
    if input % group_size != 0 || scales_desc.shape != [out, groups] {
        return Err(tensor_error(
            scales_name,
            format!(
                "expected scales [{out}, {groups}], found {:?}",
                scales_desc.shape
            ),
        ));
    }
    let bytes = file.raw_bytes(qweight_name)?;
    let expected_bytes = out
        .checked_mul(row_bytes)
        .ok_or_else(|| tensor_error(qweight_name, "packed weight size overflows"))?;
    if bytes.len() != expected_bytes {
        return Err(tensor_error(
            qweight_name,
            format!(
                "expected {expected_bytes} packed bytes, found {}",
                bytes.len()
            ),
        ));
    }
    let scales = file
        .load_as_f32(scales_name)
        .map_err(|e| tensor_error(scales_name, format!("failed to read scales: {e}")))?;
    let expected_scales = out
        .checked_mul(groups)
        .ok_or_else(|| tensor_error(scales_name, "scale count overflows"))?;
    if scales.len() != expected_scales {
        return Err(tensor_error(
            scales_name,
            format!("expected {expected_scales} scales, found {}", scales.len()),
        ));
    }
    if scales.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return Err(tensor_error(
            scales_name,
            "scales must be finite and nonnegative",
        ));
    }
    let value_count = out
        .checked_mul(input)
        .ok_or_else(|| tensor_error(qweight_name, "dequantized weight size overflows"))?;
    let mut values = Vec::new();
    values.try_reserve_exact(value_count).map_err(|e| {
        tensor_error(
            qweight_name,
            format!("cannot allocate {value_count} dequantized weights: {e}"),
        )
    })?;
    values.resize(value_count, 0.0f32);
    const POWERS: [u16; 5] = [1, 3, 9, 27, 81];
    for row in 0..out {
        let base = row * row_bytes;
        for (byte_i, &byte) in bytes[base..base + row_bytes].iter().enumerate() {
            if byte > 242 {
                return Err(tensor_error(qweight_name, "base-3 byte exceeds 242"));
            }
            for (digit, &power) in POWERS.iter().enumerate() {
                let column = byte_i * 5 + digit;
                let code = (u16::from(byte) / power) % 3;
                if column >= input {
                    if code != 0 {
                        return Err(tensor_error(qweight_name, "nonzero padding trit"));
                    }
                    continue;
                }
                let scale = scales[row * groups + column / group_size];
                values[row * input + column] = code as f32 * scale - scale;
            }
        }
    }
    Ok(values)
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

fn temporary_path() -> Result<PathBuf> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| bad("temporary file", e.to_string()))?
        .as_nanos();
    Ok(std::env::temp_dir().join(format!(
        "turbospark-parakeet-redux-{}-{nonce}.safetensors",
        std::process::id()
    )))
}

fn write_safetensors(path: &Path, tensors: BTreeMap<String, ConvertedTensor>) -> Result<()> {
    let mut header = Map::new();
    let mut data = Vec::<u8>::new();
    let total_bytes = tensors
        .values()
        .try_fold(0usize, |n, tensor| {
            n.checked_add(tensor.values.len().checked_mul(4)?)
        })
        .ok_or_else(|| bad("temporary file", "converted tensor size overflow"))?;
    data.try_reserve_exact(total_bytes).map_err(|e| {
        bad(
            "temporary file",
            format!("cannot allocate converted weights: {e}"),
        )
    })?;
    for (name, tensor) in tensors {
        let start = data.len();
        for value in tensor.values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            name,
            json!({
                "dtype": "F32",
                "shape": tensor.shape,
                "data_offsets": [start, data.len()]
            }),
        );
    }
    let mut header = serde_json::to_vec(&Value::Object(header))
        .map_err(|e| bad("temporary file", format!("serialize header: {e}")))?;
    while (8 + header.len()) % 8 != 0 {
        header.push(b' ');
    }
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| bad("temporary file", format!("create {}: {e}", path.display())))?;
    out.write_all(&(header.len() as u64).to_le_bytes())
        .and_then(|_| out.write_all(&header))
        .and_then(|_| out.write_all(&data))
        .and_then(|_| out.sync_all())
        .map_err(|e| bad("temporary file", format!("write converted weights: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn maps_hf_names_to_mlx_names() {
        assert_eq!(
            map_name("encoder.layers.0.self_attn.q_proj.weight").unwrap(),
            "encoder.layers.0.self_attn.linear_q.weight"
        );
        assert_eq!(
            map_name("encoder.subsampling.layers.3.weight").unwrap(),
            "encoder.pre_encode.conv.3.weight"
        );
        assert_eq!(
            map_name("decoder.lstm.weight_ih_l1").unwrap(),
            "decoder.prediction.dec_rnn.lstm.1.Wx"
        );
        assert_eq!(
            map_name("decoder.lstm.weight_hh_l1").unwrap(),
            "decoder.prediction.dec_rnn.lstm.1.Wh"
        );
    }

    #[test]
    fn lstm_bias_names_are_distinguished() {
        assert_eq!(parse_lstm_bias("decoder.lstm.bias_ih_l0"), Some((0, true)));
        assert_eq!(parse_lstm_bias("decoder.lstm.bias_hh_l1"), Some((1, false)));
    }

    fn safetensors_fixture(
        qweight: &[u8],
        scales: &[f32],
        qshape: &[usize],
    ) -> (PathBuf, SafetensorsFile) {
        let mut data = qweight.to_vec();
        let scale_start = data.len();
        for scale in scales {
            data.extend_from_slice(&scale.to_le_bytes());
        }
        let mut header = Map::new();
        header.insert(
            "proj.qweight".into(),
            json!({
                "dtype": "U8",
                "shape": qshape,
                "data_offsets": [0, scale_start]
            }),
        );
        header.insert(
            "proj.scales".into(),
            json!({
                "dtype": "F32",
                "shape": [1, scales.len()],
                "data_offsets": [scale_start, data.len()]
            }),
        );
        let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
        while (8 + header.len()) % 8 != 0 {
            header.push(b' ');
        }
        let mut contents = (header.len() as u64).to_le_bytes().to_vec();
        contents.extend_from_slice(&header);
        contents.extend_from_slice(&data);

        let nonce = FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "turbospark-redux-ternary-test-{}-{nonce}.safetensors",
            std::process::id()
        ));
        fs::write(&path, contents).unwrap();
        assert!(
            fs::metadata(&path).unwrap().len() >= 8,
            "fixture is empty: {}",
            path.display()
        );
        let file = SafetensorsFile::open(&path).unwrap();
        (path, file)
    }

    #[test]
    fn dequantizes_base_three_trits_with_per_group_scales() {
        // [0, 1, 2, 0, 1] packs to 102; [2, 1, 0] packs to 5.
        let (path, file) = safetensors_fixture(&[102, 5], &[0.5, 2.0], &[1, 2]);
        let values = dequantize_ternary(&file, "proj.qweight", "proj.scales", 1, 8, 4).unwrap();
        drop(file);
        fs::remove_file(path).unwrap();

        assert_eq!(values, [-0.5, 0.0, 0.5, -0.5, 0.0, 2.0, 0.0, -2.0]);
    }

    #[test]
    fn rejects_nonzero_ternary_padding_trits() {
        // Six live trits are followed by a nonzero seventh trit in padding.
        let (path, file) = safetensors_fixture(&[102, 5], &[0.5, 2.0], &[1, 2]);
        let result = dequantize_ternary(&file, "proj.qweight", "proj.scales", 1, 6, 3);
        drop(file);
        fs::remove_file(path).unwrap();

        assert!(result
            .unwrap_err()
            .to_string()
            .contains("nonzero padding trit"));
    }
}
