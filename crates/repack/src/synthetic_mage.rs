//! A tiny `mage_vl` checkpoint under the MLX conversions' names, for the
//! intake, runtime and FFI tests (`docs/MAGE_VL_VISION_PHASE0.md`).
//!
//! Only the sizes are shrunk. The patch of 16, the merge of 2, one frame per
//! patch and a head width of 32 (so all three rope axes exist) match what the
//! port assumes.

use std::path::Path;

use crate::gemma4_checkpoint::{
    parse_gemma4_quantization, parse_mage_vl_vision_config, write_gemma4_install,
};
use crate::ranged_download::MemoryRangeSource;
use crate::safetensors_header::{parse_header, DEFAULT_MAX_HEADER_BYTES};
use crate::synthetic_llama::SyntheticTowerTensor;
use crate::synthetic_tensors::{
    assemble_safetensors, bf16_vector, deterministic_row, int4_triple, u16_le, Tensor,
};

/// The tower's own width.
pub const SYNTHETIC_MAGE_HIDDEN: usize = 64;
/// The tower's attention heads (head width 32).
pub const SYNTHETIC_MAGE_HEADS: usize = 2;
/// Tower blocks.
pub const SYNTHETIC_MAGE_DEPTH: usize = 2;
/// The trunk width the merger writes into.
pub const SYNTHETIC_MAGE_TEXT_HIDDEN: usize = 64;

const MLP: usize = 96;
const TRUNK_LAYERS: usize = 2;

/// The published conversions' template, reduced to the arms they have.
const CHAT_TEMPLATE: &str = "{% for message in messages %}{% if loop.first and message['role'] \
    != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}\
    <|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}\
    {{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}\
    {% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}\
    <|vision_start|><|image_pad|><|vision_end|>{% elif 'text' in content %}\
    {{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}\
    {% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

/// The files a pull of the checkpoint reads.
pub struct SyntheticMageVl {
    pub config_json: String,
    pub preprocessor_json: String,
    /// One `model.safetensors`.
    pub safetensors: Vec<u8>,
    /// The tower tensors as written, for an independent CPU forward pass.
    pub tensors: Vec<SyntheticTowerTensor>,
}

/// Build the checkpoint. `ids` are `[vision_start, vision_end, image,
/// video]`; `vocab` must cover them.
pub fn synthetic_mage_vl_checkpoint(vocab: usize, ids: [i64; 4]) -> SyntheticMageVl {
    const H: usize = SYNTHETIC_MAGE_HIDDEN;
    const T: usize = SYNTHETIC_MAGE_TEXT_HIDDEN;
    const HEAD: usize = 32;

    let mut tensors: Vec<SyntheticTowerTensor> = Vec::new();
    let mut raw: Vec<Tensor> = Vec::new();
    let mut add = |name: String, shape: Vec<usize>, center: f32, amplitude: f32| {
        let name = format!("vision_tower.{name}");
        let n: usize = shape.iter().product();
        let bits: Vec<u16> = deterministic_row(13_000 + raw.len() as u64, n)
            .iter()
            .map(|&j| compute::f32_to_bf16(center + j * amplitude))
            .collect();
        tensors.push(SyntheticTowerTensor {
            name: name.clone(),
            shape: shape.clone(),
            values: bits.iter().map(|&b| compute::bf16_to_f32(b)).collect(),
        });
        raw.push(Tensor {
            name,
            dtype: "BF16",
            shape: shape.iter().map(|&d| d as u64).collect(),
            bytes: u16_le(&bits),
        });
    };

    add(
        "embeddings.patch_embedding.weight".into(),
        vec![H, 16, 16, 3],
        0.0,
        0.05,
    );
    add("layernorm_pre.weight".into(), vec![H], 1.0, 0.3);
    add("layernorm_pre.bias".into(), vec![H], 0.0, 0.3);
    for block in 0..SYNTHETIC_MAGE_DEPTH {
        let p = format!("encoder.layers.{block}.");
        for norm in ["layer_norm1", "layer_norm2"] {
            add(format!("{p}{norm}.weight"), vec![H], 1.0, 0.2);
            add(format!("{p}{norm}.bias"), vec![H], 0.0, 0.2);
        }
        add(format!("{p}self_attn.qkv.weight"), vec![3 * H, H], 0.0, 0.4);
        add(format!("{p}self_attn.qkv.bias"), vec![3 * H], 0.0, 0.2);
        add(format!("{p}self_attn.proj.weight"), vec![H, H], 0.0, 0.3);
        add(format!("{p}self_attn.proj.bias"), vec![H], 0.0, 0.2);
        add(format!("{p}mlp.fc1.weight"), vec![MLP, H], 0.0, 0.6);
        add(format!("{p}mlp.fc1.bias"), vec![MLP], 0.0, 0.2);
        add(format!("{p}mlp.fc2.weight"), vec![H, MLP], 0.0, 0.8);
        add(format!("{p}mlp.fc2.bias"), vec![H], 0.0, 0.2);
    }
    add("merger.ln_q.weight".into(), vec![H], 1.0, 0.2);
    add("merger.ln_q.bias".into(), vec![H], 0.0, 0.2);
    add("merger.mlp.0.weight".into(), vec![4 * H, 4 * H], 0.0, 0.1);
    add("merger.mlp.0.bias".into(), vec![4 * H], 0.0, 0.2);
    add("merger.mlp.2.weight".into(), vec![T, 4 * H], 0.0, 0.1);
    add("merger.mlp.2.bias".into(), vec![T], 0.0, 0.2);

    raw.extend(int4_triple(
        "language_model.model.embed_tokens.weight",
        vocab,
        T,
        1,
    ));
    raw.extend(int4_triple("language_model.lm_head.weight", vocab, T, 2));
    for l in 0..TRUNK_LAYERS {
        let p = format!("language_model.model.layers.{l}");
        let seed = 1000 * (l as u64 + 1);
        for (name, len) in [
            ("input_layernorm", T),
            ("post_attention_layernorm", T),
            ("self_attn.q_norm", HEAD),
            ("self_attn.k_norm", HEAD),
        ] {
            raw.push(bf16_vector(
                &format!("{p}.{name}.weight"),
                len,
                1.0,
                seed + len as u64,
            ));
        }
        for (i, (name, rows, cols)) in [
            ("self_attn.q_proj", 4 * HEAD, T),
            ("self_attn.k_proj", 2 * HEAD, T),
            ("self_attn.v_proj", 2 * HEAD, T),
            ("self_attn.o_proj", T, 4 * HEAD),
            ("mlp.gate_proj", 64, T),
            ("mlp.up_proj", 64, T),
            ("mlp.down_proj", T, 64),
        ]
        .into_iter()
        .enumerate()
        {
            raw.extend(int4_triple(
                &format!("{p}.{name}.weight"),
                rows,
                cols,
                seed + i as u64,
            ));
        }
    }
    raw.push(bf16_vector("language_model.model.norm.weight", T, 1.0, 7));

    let config_json = serde_json::json!({
        "model_type": "mage_vl",
        "quantization": {"bits": 4, "group_size": 64},
        "tie_word_embeddings": false,
        "vision_start_token_id": ids[0],
        "vision_end_token_id": ids[1],
        "image_token_id": ids[2],
        "video_token_id": ids[3],
        "text_config": {
            "model_type": "qwen3", "hidden_size": T, "intermediate_size": 64,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": HEAD,
            "num_hidden_layers": TRUNK_LAYERS, "vocab_size": vocab,
            "max_position_embeddings": 4096, "rms_norm_eps": 1e-6, "hidden_act": "silu",
            "rope_parameters": {"rope_theta": 5_000_000, "rope_type": "default"}
        },
        "vision_config": {
            "model_type": "mage_vl_vision", "hidden_act": "gelu",
            "layer_norm_type": "layer_norm", "layer_norm_eps": 1e-6, "rope_theta": 10000.0,
            "num_hidden_layers": SYNTHETIC_MAGE_DEPTH, "hidden_size": H,
            "num_attention_heads": SYNTHETIC_MAGE_HEADS, "intermediate_size": MLP,
            "out_hidden_size": T, "patch_size": 16, "temporal_patch_size": 1,
            "spatial_merge_size": 2, "num_channels": 3, "use_head": false
        }
    })
    .to_string();
    let preprocessor_json = serde_json::json!({
        "patch_size": 16, "temporal_patch_size": 1, "merge_size": 2,
        "min_pixels": 1024, "max_pixels": 36864,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "rescale_factor": 0.00392156862745098
    })
    .to_string();
    SyntheticMageVl {
        config_json,
        preprocessor_json,
        safetensors: assemble_safetensors(&raw),
        tensors,
    }
}

/// Write the checkpoint as an install through the real config parsers and
/// repack walk. With `tokenizer_json` (a small ChatML `tokenizer.json`) the
/// vision markers are appended to it and the sidecars an image request needs
/// are written too; without one the ids are 100..104 over a 128-token vocab.
pub fn build_synthetic_mage_vl_install(
    dir: &Path,
    tokenizer_json: Option<&str>,
) -> Result<(model_io::ArchConfig, Vec<SyntheticTowerTensor>), Box<dyn std::error::Error>> {
    let mut vocab = 128usize;
    let mut ids = [100i64, 101, 102, 103];
    let mut tokenizer = None;
    if let Some(source) = tokenizer_json {
        let mut json: serde_json::Value = serde_json::from_str(source)?;
        let base = json["model"]["vocab"]
            .as_object()
            .and_then(|v| v.values().filter_map(serde_json::Value::as_u64).max())
            .ok_or("tokenizer.json has no model.vocab")?
            + 1;
        let added = json["added_tokens"]
            .as_array_mut()
            .ok_or("tokenizer.json has no added_tokens")?;
        let template = added.first().cloned().ok_or("added_tokens is empty")?;
        // The ChatML fixture already carries some of the markers.
        const MARKERS: [&str; 4] = [
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
            "<|video_pad|>",
        ];
        for content in MARKERS {
            if !added.iter().any(|t| t["content"] == content) {
                let mut token = template.clone();
                token["content"] = content.into();
                added.push(token);
            }
        }
        for (offset, token) in added.iter_mut().enumerate() {
            token["id"] = (base + offset as u64).into();
        }
        for (content, id) in MARKERS.into_iter().zip(&mut ids) {
            let at = added.iter().position(|t| t["content"] == content);
            *id = (base + at.ok_or("marker was just added")? as u64) as i64;
        }
        vocab = base as usize + added.len();
        tokenizer = Some(json);
    }

    let fixture = synthetic_mage_vl_checkpoint(vocab, ids);
    let mut arch = crate::parse_qwen3_config(&fixture.config_json)?;
    arch.vision = parse_mage_vl_vision_config(&fixture.config_json)?;
    let header = parse_header(&fixture.safetensors, DEFAULT_MAX_HEADER_BYTES)?;
    let quant = parse_gemma4_quantization(&fixture.config_json)?;
    write_gemma4_install(
        dir,
        &arch,
        "tiny-mage-vl",
        &header,
        &MemoryRangeSource::new(&fixture.safetensors),
        &quant,
    )?;
    std::fs::write(
        dir.join("preprocessor_config.json"),
        &fixture.preprocessor_json,
    )?;
    if let Some(tokenizer) = tokenizer {
        std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string())?;
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"eos_token":"<|im_end|>","pad_token":"<|endoftext|>"}"#,
        )?;
        std::fs::write(dir.join("chat_template.jinja"), CHAT_TEMPLATE)?;
    }
    Ok((arch, fixture.tensors))
}
