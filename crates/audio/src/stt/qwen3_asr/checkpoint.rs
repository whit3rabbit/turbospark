//! Descriptor-only checkpoint validation for Qwen3-ASR installs.
//!
//! The catalog probe runs this before any receipt is written: every tensor
//! name, dtype, and shape the loaders consume is checked without
//! dequantizing or loading values, so a foreign or truncated distribution
//! is refused with the offending tensor instead of failing mid-load. The
//! tensor list must stay in step with `encoder.rs`, `decoder.rs`, and the
//! packed Metal loader, which still own the authoritative checks at open;
//! this module is the fast pass that needs no allocation.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::models::stt::qwen3_asr::config::{AudioEncoderConfig, Qwen3Config};
use crate::{Result, SpeechError};

/// The audio tower is unquantized; the CPU loader reads any float dtype
/// `load_as_f32` converts.
const FLOAT_DTYPES: [&str; 3] = ["F32", "F16", "BF16"];
/// Packed linear companions ride the same affine loader as the weights.
const COMPANION_DTYPES: [&str; 3] = ["F32", "F16", "BF16"];

fn resolve_base(file: &SafetensorsFile, base: &str) -> String {
    for candidate in [
        base.to_owned(),
        format!("thinker.{base}"),
        format!("llm.{base}"),
    ] {
        if file.contains_tensor(&format!("{candidate}.weight")) {
            return candidate;
        }
    }
    base.to_owned()
}

fn expect(file: &SafetensorsFile, base: &str, dtypes: &[&str], shape: &[usize]) -> Result<()> {
    let desc = file.descriptor(base).ok_or_else(|| SpeechError::Tensor {
        name: base.to_owned(),
        why: "required tensor is missing".into(),
    })?;
    if !dtypes.contains(&desc.dtype.as_str()) {
        return Err(SpeechError::Tensor {
            name: base.to_owned(),
            why: format!("dtype {} is not one of {dtypes:?}", desc.dtype),
        });
    }
    if desc.shape.as_slice() != shape {
        return Err(SpeechError::Tensor {
            name: base.to_owned(),
            why: format!("expected shape {shape:?}, got {:?}", desc.shape),
        });
    }
    Ok(())
}

/// One affine-INT8 linear in exactly the layout the loaders and the packed
/// Metal GEMV consume: U32 packed bytes `[rows, cols / 4]` plus per-group
/// scale and bias companions `[rows, cols / group]`, and no bias tensor.
fn expect_packed_linear(
    file: &SafetensorsFile,
    config: &Qwen3Config,
    base: &str,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let group = config.quant_group_size;
    if cols % 4 != 0 || cols % group != 0 {
        return Err(SpeechError::Tensor {
            name: base.to_owned(),
            why: format!(
                "column width {cols} does not divide into U32 words and groups of {group}"
            ),
        });
    }
    let base = resolve_base(file, base);
    if file.contains_tensor(&format!("{base}.bias")) {
        return Err(SpeechError::Tensor {
            name: base.to_owned(),
            why: "quantized linear carries a bias, which the affine loader and the packed Metal GEMV refuse".into(),
        });
    }
    expect(file, &format!("{base}.weight"), &["U32"], &[rows, cols / 4])?;
    expect(
        file,
        &format!("{base}.scales"),
        &COMPANION_DTYPES,
        &[rows, cols / group],
    )?;
    expect(
        file,
        &format!("{base}.biases"),
        &COMPANION_DTYPES,
        &[rows, cols / group],
    )?;
    Ok(())
}

fn expect_norm(file: &SafetensorsFile, base: &str, width: usize) -> Result<()> {
    // The RMS norm weights upload as raw BF16 on the Metal path and the CPU
    // loader reads them through the same dtype; refuse anything else.
    let base = resolve_base(file, base);
    expect(file, &format!("{base}.weight"), &["BF16"], &[width])
}

fn validate_audio_tower(file: &SafetensorsFile, audio: &AudioEncoderConfig) -> Result<()> {
    let width = audio.downsample_hidden_size;
    // Checkpoint conv weights ship either the MLX [out, kh, kw, in] order
    // or the PyTorch [out, in, kh, kw] order; the loader transposes the
    // former at load.
    for (index, inputs) in [(1usize, 1usize), (2, width), (3, width)] {
        let prefix = format!("audio_tower.conv2d{index}");
        let name = format!("{prefix}.weight");
        let desc = file.descriptor(&name).ok_or_else(|| SpeechError::Tensor {
            name: name.clone(),
            why: "required convolution weight is missing".into(),
        })?;
        let mlx = [width, 3, 3, inputs];
        let pytorch = [width, inputs, 3, 3];
        if desc.shape.as_slice() != mlx && desc.shape.as_slice() != pytorch {
            return Err(SpeechError::Tensor {
                name,
                why: format!(
                    "expected MLX shape {mlx:?} or PyTorch shape {pytorch:?}, got {:?}",
                    desc.shape
                ),
            });
        }
        expect(file, &format!("{prefix}.bias"), &FLOAT_DTYPES, &[width])?;
    }
    // Three stride-2 convolutions turn the 128 mel bands into 16 frequency
    // rows, so conv_out sees width * 16 features per time step.
    let frequency = 128usize.div_ceil(2).div_ceil(2).div_ceil(2);
    expect(
        file,
        "audio_tower.conv_out.weight",
        &FLOAT_DTYPES,
        &[audio.d_model, width * frequency],
    )?;
    for layer in 0..audio.encoder_layers {
        let prefix = format!("audio_tower.layers.{layer}");
        let attention = format!("{prefix}.self_attn");
        for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            expect(
                file,
                &format!("{attention}.{projection}.weight"),
                &FLOAT_DTYPES,
                &[audio.d_model, audio.d_model],
            )?;
            expect(
                file,
                &format!("{attention}.{projection}.bias"),
                &FLOAT_DTYPES,
                &[audio.d_model],
            )?;
        }
        expect(
            file,
            &format!("{prefix}.self_attn_layer_norm.weight"),
            &FLOAT_DTYPES,
            &[audio.d_model],
        )?;
        expect(
            file,
            &format!("{prefix}.self_attn_layer_norm.bias"),
            &FLOAT_DTYPES,
            &[audio.d_model],
        )?;
        expect(
            file,
            &format!("{prefix}.fc1.weight"),
            &FLOAT_DTYPES,
            &[audio.encoder_ffn_dim, audio.d_model],
        )?;
        expect(
            file,
            &format!("{prefix}.fc1.bias"),
            &FLOAT_DTYPES,
            &[audio.encoder_ffn_dim],
        )?;
        expect(
            file,
            &format!("{prefix}.fc2.weight"),
            &FLOAT_DTYPES,
            &[audio.d_model, audio.encoder_ffn_dim],
        )?;
        expect(
            file,
            &format!("{prefix}.fc2.bias"),
            &FLOAT_DTYPES,
            &[audio.d_model],
        )?;
        expect(
            file,
            &format!("{prefix}.final_layer_norm.weight"),
            &FLOAT_DTYPES,
            &[audio.d_model],
        )?;
        expect(
            file,
            &format!("{prefix}.final_layer_norm.bias"),
            &FLOAT_DTYPES,
            &[audio.d_model],
        )?;
    }
    expect(
        file,
        "audio_tower.ln_post.weight",
        &FLOAT_DTYPES,
        &[audio.d_model],
    )?;
    expect(
        file,
        "audio_tower.ln_post.bias",
        &FLOAT_DTYPES,
        &[audio.d_model],
    )?;
    expect(
        file,
        "audio_tower.proj1.weight",
        &FLOAT_DTYPES,
        &[audio.d_model, audio.d_model],
    )?;
    expect(
        file,
        "audio_tower.proj1.bias",
        &FLOAT_DTYPES,
        &[audio.d_model],
    )?;
    expect(
        file,
        "audio_tower.proj2.weight",
        &FLOAT_DTYPES,
        &[audio.output_dim, audio.d_model],
    )?;
    expect(
        file,
        "audio_tower.proj2.bias",
        &FLOAT_DTYPES,
        &[audio.output_dim],
    )?;
    Ok(())
}

fn validate_text_decoder(file: &SafetensorsFile, config: &Qwen3Config) -> Result<()> {
    let text = &config.text;
    let q_width = text.num_attention_heads * text.head_dim;
    let kv_width = text.num_key_value_heads * text.head_dim;
    expect_packed_linear(
        file,
        config,
        "model.embed_tokens",
        text.vocab_size,
        text.hidden_size,
    )?;
    for index in 0..text.num_hidden_layers {
        let prefix = format!("model.layers.{index}");
        let attn = format!("{prefix}.self_attn");
        let mlp = format!("{prefix}.mlp");
        expect_norm(file, &format!("{prefix}.input_layernorm"), text.hidden_size)?;
        expect_norm(
            file,
            &format!("{prefix}.post_attention_layernorm"),
            text.hidden_size,
        )?;
        expect_norm(file, &format!("{attn}.q_norm"), text.head_dim)?;
        expect_norm(file, &format!("{attn}.k_norm"), text.head_dim)?;
        expect_packed_linear(
            file,
            config,
            &format!("{attn}.q_proj"),
            q_width,
            text.hidden_size,
        )?;
        expect_packed_linear(
            file,
            config,
            &format!("{attn}.k_proj"),
            kv_width,
            text.hidden_size,
        )?;
        expect_packed_linear(
            file,
            config,
            &format!("{attn}.v_proj"),
            kv_width,
            text.hidden_size,
        )?;
        expect_packed_linear(
            file,
            config,
            &format!("{attn}.o_proj"),
            text.hidden_size,
            q_width,
        )?;
        expect_packed_linear(
            file,
            config,
            &format!("{mlp}.gate_proj"),
            text.intermediate_size,
            text.hidden_size,
        )?;
        expect_packed_linear(
            file,
            config,
            &format!("{mlp}.up_proj"),
            text.intermediate_size,
            text.hidden_size,
        )?;
        expect_packed_linear(
            file,
            config,
            &format!("{mlp}.down_proj"),
            text.hidden_size,
            text.intermediate_size,
        )?;
    }
    expect_norm(file, "model.norm", text.hidden_size)?;
    Ok(())
}

/// Validates every tensor the Qwen3-ASR loaders consume, by descriptor
/// only. `config` must already parse and pass `Qwen3Config::from_json`.
pub fn validate_checkpoint(file: &SafetensorsFile, config: &Qwen3Config) -> Result<()> {
    validate_audio_tower(file, &config.audio)?;
    validate_text_decoder(file, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn test_config() -> Qwen3Config {
        // Same miniature geometry the runtime Metal tests synthesize: the
        // decoder widths all divide into groups of 64.
        let json: serde_json::Value = serde_json::from_str(
            r#"{
            "model_type": "qwen3_asr",
            "audio_token_id": 259, "audio_start_token_id": 258, "audio_end_token_id": 260,
            "audio_config": {"num_mel_bins":128,"encoder_layers":1,"encoder_attention_heads":4,
                "encoder_ffn_dim":64,"d_model":64,"max_source_positions":64,"n_window":2,
                "n_window_infer":4,"downsample_hidden_size":8,"output_dim":64,
                "scale_embedding":false,"activation_function":"gelu"},
            "text_config": {"vocab_size":261,"hidden_size":64,"intermediate_size":128,
                "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
                "head_dim":64,"rms_norm_eps":1e-5,"rope_theta":10000.0,
                "tie_word_embeddings":true,"attention_bias":false,"hidden_act":"silu"},
            "quantization_config": {"bits":8,"group_size":64,"mode":"affine"}
        }"#,
        )
        .unwrap();
        Qwen3Config::from_json(&json).unwrap()
    }

    /// Every required tensor name with its shape; dtypes are assigned by
    /// the writer (F32 tower, BF16 norms, U32/BF16 packed decoder).
    fn required_tensors(config: &Qwen3Config) -> BTreeMap<String, Vec<usize>> {
        let audio = &config.audio;
        let width = audio.downsample_hidden_size;
        let frequency = 128usize.div_ceil(2).div_ceil(2).div_ceil(2);
        let mut tensors = BTreeMap::new();
        fn push_packed(
            tensors: &mut BTreeMap<String, Vec<usize>>,
            group: usize,
            name: String,
            rows: usize,
            cols: usize,
        ) {
            tensors.insert(format!("{name}.weight"), vec![rows, cols / 4]);
            tensors.insert(format!("{name}.scales"), vec![rows, cols / group]);
            tensors.insert(format!("{name}.biases"), vec![rows, cols / group]);
        }
        fn push_conv(
            tensors: &mut BTreeMap<String, Vec<usize>>,
            width: usize,
            name: String,
            inputs: usize,
        ) {
            tensors.insert(format!("{name}.weight"), vec![width, inputs, 3, 3]);
            tensors.insert(format!("{name}.bias"), vec![width]);
        }
        push_conv(&mut tensors, width, "audio_tower.conv2d1".to_owned(), 1);
        push_conv(&mut tensors, width, "audio_tower.conv2d2".to_owned(), width);
        push_conv(&mut tensors, width, "audio_tower.conv2d3".to_owned(), width);
        tensors.insert(
            "audio_tower.conv_out.weight".to_owned(),
            vec![audio.d_model, width * frequency],
        );
        for layer in 0..audio.encoder_layers {
            let prefix = format!("audio_tower.layers.{layer}");
            for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                tensors.insert(
                    format!("{prefix}.self_attn.{projection}.weight"),
                    vec![audio.d_model, audio.d_model],
                );
                tensors.insert(
                    format!("{prefix}.self_attn.{projection}.bias"),
                    vec![audio.d_model],
                );
            }
            for norm in ["self_attn_layer_norm", "final_layer_norm"] {
                tensors.insert(format!("{prefix}.{norm}.weight"), vec![audio.d_model]);
                tensors.insert(format!("{prefix}.{norm}.bias"), vec![audio.d_model]);
            }
            tensors.insert(
                format!("{prefix}.fc1.weight"),
                vec![audio.encoder_ffn_dim, audio.d_model],
            );
            tensors.insert(format!("{prefix}.fc1.bias"), vec![audio.encoder_ffn_dim]);
            tensors.insert(
                format!("{prefix}.fc2.weight"),
                vec![audio.d_model, audio.encoder_ffn_dim],
            );
            tensors.insert(format!("{prefix}.fc2.bias"), vec![audio.d_model]);
        }
        tensors.insert("audio_tower.ln_post.weight".to_owned(), vec![audio.d_model]);
        tensors.insert("audio_tower.ln_post.bias".to_owned(), vec![audio.d_model]);
        tensors.insert(
            "audio_tower.proj1.weight".to_owned(),
            vec![audio.d_model, audio.d_model],
        );
        tensors.insert("audio_tower.proj1.bias".to_owned(), vec![audio.d_model]);
        tensors.insert(
            "audio_tower.proj2.weight".to_owned(),
            vec![audio.output_dim, audio.d_model],
        );
        tensors.insert("audio_tower.proj2.bias".to_owned(), vec![audio.output_dim]);

        let text = &config.text;
        let q_width = text.num_attention_heads * text.head_dim;
        let kv_width = text.num_key_value_heads * text.head_dim;
        let group = config.quant_group_size;
        push_packed(
            &mut tensors,
            group,
            "model.embed_tokens".to_owned(),
            text.vocab_size,
            text.hidden_size,
        );
        for layer in 0..text.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            let attn = format!("{prefix}.self_attn");
            let mlp = format!("{prefix}.mlp");
            tensors.insert(
                format!("{prefix}.input_layernorm.weight"),
                vec![text.hidden_size],
            );
            tensors.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![text.hidden_size],
            );
            tensors.insert(format!("{attn}.q_norm.weight"), vec![text.head_dim]);
            tensors.insert(format!("{attn}.k_norm.weight"), vec![text.head_dim]);
            push_packed(
                &mut tensors,
                group,
                format!("{attn}.q_proj"),
                q_width,
                text.hidden_size,
            );
            push_packed(
                &mut tensors,
                group,
                format!("{attn}.k_proj"),
                kv_width,
                text.hidden_size,
            );
            push_packed(
                &mut tensors,
                group,
                format!("{attn}.v_proj"),
                kv_width,
                text.hidden_size,
            );
            push_packed(
                &mut tensors,
                group,
                format!("{attn}.o_proj"),
                text.hidden_size,
                q_width,
            );
            push_packed(
                &mut tensors,
                group,
                format!("{mlp}.gate_proj"),
                text.intermediate_size,
                text.hidden_size,
            );
            push_packed(
                &mut tensors,
                group,
                format!("{mlp}.up_proj"),
                text.intermediate_size,
                text.hidden_size,
            );
            push_packed(
                &mut tensors,
                group,
                format!("{mlp}.down_proj"),
                text.hidden_size,
                text.intermediate_size,
            );
        }
        tensors.insert("model.norm.weight".to_owned(), vec![text.hidden_size]);
        tensors
    }

    fn dtype_of(name: &str) -> &'static str {
        if name.starts_with("audio_tower") {
            return "F32";
        }
        if name.ends_with(".scales") || name.ends_with(".biases") {
            return "BF16";
        }
        if name.ends_with(".weight") && name.starts_with("model.") {
            // embed_tokens and every *_proj are packed; every remaining
            // model.* weight is an RMS norm.
            if name.contains("embed_tokens") || name.ends_with("_proj.weight") {
                "U32"
            } else {
                "BF16"
            }
        } else {
            "F32"
        }
    }

    fn write_safetensors(path: &std::path::Path, tensors: &BTreeMap<String, Vec<usize>>) {
        let mut header = serde_json::Map::new();
        let mut blob = 0usize;
        for (name, shape) in tensors {
            let elements: usize = shape.iter().product();
            let bytes = match dtype_of(name) {
                "U32" => elements * 4,
                "BF16" | "F16" => elements * 2,
                _ => elements * 4,
            };
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": dtype_of(name),
                    "shape": shape,
                    "data_offsets": [blob, blob + bytes],
                }),
            );
            blob += bytes;
        }
        let header_json = serde_json::Value::Object(header).to_string();
        let mut file = (header_json.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header_json.as_bytes());
        file.resize(file.len() + blob, 0);
        std::fs::write(path, file).unwrap();
    }

    #[test]
    fn accepts_the_full_layout_and_refuses_shape_mutations() {
        let config = test_config();
        let dir = std::env::temp_dir().join(format!("qwen3-checkpoint-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");

        let tensors = required_tensors(&config);
        write_safetensors(&path, &tensors);
        let file = turbospark_model_io::safetensors::SafetensorsFile::open(&path).unwrap();
        validate_checkpoint(&file, &config).expect("complete layout must validate");

        // Mutation: a wrong shape must be refused with the tensor named.
        let mut broken = tensors.clone();
        broken.insert(
            "audio_tower.conv_out.weight".to_owned(),
            vec![config.audio.d_model, 8],
        );
        write_safetensors(&path, &broken);
        let file = turbospark_model_io::safetensors::SafetensorsFile::open(&path).unwrap();
        let error = validate_checkpoint(&file, &config).unwrap_err().to_string();
        assert!(error.contains("conv_out"), "{error}");

        // Mutation: a missing decoder tensor must be refused too.
        let mut missing = tensors;
        missing.remove(&"model.layers.1.mlp.down_proj.scales".to_owned());
        write_safetensors(&path, &missing);
        let file = turbospark_model_io::safetensors::SafetensorsFile::open(&path).unwrap();
        let error = validate_checkpoint(&file, &config).unwrap_err().to_string();
        assert!(error.contains("down_proj.scales"), "{error}");

        std::fs::remove_dir_all(&dir).ok();
    }
}
