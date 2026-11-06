use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use compute::rms_norm::rms_norm;
use compute::rope::rope_neox;
use compute::vision::matmul_bias;
use model_io::safetensors::SafetensorsFile;

use crate::packed::PackedTensorStore;

pub const HIDDEN_SIZE: usize = 2560;
pub const INTERMEDIATE_SIZE: usize = 9728;
pub const NUM_LAYERS: usize = 36;
pub const EXTRACT_LAYER_COUNT: usize = 35; // layers 0..=34 for hidden_states[-2]
pub const NUM_Q_HEADS: usize = 32;
pub const NUM_KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const RMS_NORM_EPS: f32 = 1e-6;
pub const ROPE_THETA: f32 = 1000000.0;

/// Common tensor-loading contract for source and packed image components.
pub trait TensorLoader {
    fn contains_tensor(&self, name: &str) -> bool;
    fn load_tensor(&self, name: &str) -> Result<Vec<f32>, String>;
}

/// Helper to load weights across multiple safetensors shards.
pub struct ShardedSafetensors {
    pub(crate) shards: BTreeMap<String, SafetensorsFile>,
    pub(crate) weight_map: BTreeMap<String, String>,
    packed: Option<PackedTensorStore>,
}

impl ShardedSafetensors {
    pub fn open(model_dir: &Path) -> Result<Self, String> {
        Self::open_indexed(model_dir, "model.safetensors.index.json")
    }

    /// Open a component whose safetensors index has a non-text-encoder name.
    pub fn open_indexed(model_dir: &Path, index_name: &str) -> Result<Self, String> {
        let index_path = model_dir.join(index_name);
        let mut index_file = match File::open(&index_path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let shard_name = index_name.strip_suffix(".index.json").ok_or_else(|| {
                    format!(
                        "indexed safetensors file {} is missing and has no single-file name",
                        index_path.display()
                    )
                })?;
                let shard_path = model_dir.join(shard_name);
                let shard = SafetensorsFile::open(&shard_path).map_err(|e| {
                    format!(
                        "failed to open single-file safetensors component {}: {e}",
                        shard_path.display()
                    )
                })?;
                let weight_map = shard
                    .tensor_names()
                    .map(|name| (name.to_string(), shard_name.to_string()))
                    .collect();
                let mut shards = BTreeMap::new();
                shards.insert(shard_name.to_string(), shard);
                return Ok(Self {
                    shards,
                    weight_map,
                    packed: None,
                });
            }
            Err(error) => {
                return Err(format!("failed to open {}: {error}", index_path.display()));
            }
        };
        let mut index_str = String::new();
        index_file
            .read_to_string(&mut index_str)
            .map_err(|e| format!("failed to read {}: {e}", index_path.display()))?;
        let val: serde_json::Value =
            serde_json::from_str(&index_str).map_err(|e| format!("failed to parse index: {e}"))?;
        let weight_map_val = val
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| "missing weight_map in index".to_string())?;

        let mut weight_map = BTreeMap::new();
        for (k, v) in weight_map_val {
            if let Some(s) = v.as_str() {
                weight_map.insert(k.clone(), s.to_string());
            }
        }

        let mut shards = BTreeMap::new();
        for shard_name in weight_map.values() {
            if !shards.contains_key(shard_name) {
                let shard_path = model_dir.join(shard_name);
                let sf = SafetensorsFile::open(&shard_path)
                    .map_err(|e| format!("failed to open shard {}: {e}", shard_path.display()))?;
                shards.insert(shard_name.clone(), sf);
            }
        }

        Ok(Self {
            shards,
            weight_map,
            packed: None,
        })
    }

    /// Open a component written by [`crate::packed::pack_component`].
    pub fn open_packed(model_dir: &Path) -> Result<Self, String> {
        Ok(Self {
            shards: BTreeMap::new(),
            weight_map: BTreeMap::new(),
            packed: Some(PackedTensorStore::open(model_dir)?),
        })
    }

    pub(crate) fn source_tensors(&self) -> impl Iterator<Item = (&str, &str)> {
        self.weight_map
            .iter()
            .map(|(name, shard)| (name.as_str(), shard.as_str()))
    }

    pub fn load_tensor(&self, name: &str) -> Result<Vec<f32>, String> {
        if let Some(packed) = &self.packed {
            return packed.load_tensor(name);
        }
        let shard_name = self
            .weight_map
            .get(name)
            .ok_or_else(|| format!("tensor {name} not found in index"))?;
        let shard = self
            .shards
            .get(shard_name)
            .ok_or_else(|| format!("shard {shard_name} not loaded"))?;
        shard
            .load_as_f32(name)
            .map_err(|e| format!("failed to load tensor {name}: {e}"))
    }

    /// Load a tensor only after confirming the checkpoint's declared shape.
    pub fn load_tensor_shape(
        &self,
        name: &str,
        expected_shape: &[usize],
    ) -> Result<Vec<f32>, String> {
        if let Some(packed) = &self.packed {
            let actual_shape = packed
                .shape(name)
                .ok_or_else(|| format!("tensor {name} not found in packed index"))?;
            if actual_shape != expected_shape {
                return Err(format!(
                    "tensor {name} has shape {actual_shape:?}, expected {expected_shape:?}"
                ));
            }
            return packed.load_tensor(name);
        }
        let shard_name = self
            .weight_map
            .get(name)
            .ok_or_else(|| format!("tensor {name} not found in index"))?;
        let shard = self
            .shards
            .get(shard_name)
            .ok_or_else(|| format!("shard {shard_name} not loaded"))?;
        let descriptor = shard
            .descriptor(name)
            .ok_or_else(|| format!("tensor {name} missing from shard {shard_name}"))?;
        if descriptor.shape != expected_shape {
            return Err(format!(
                "tensor {name} has shape {:?}, expected {:?}",
                descriptor.shape, expected_shape
            ));
        }
        self.load_tensor(name)
    }

    pub fn load_token_embeddings(&self, token_ids: &[i64]) -> Result<Vec<f32>, String> {
        let name = "model.embed_tokens.weight";
        if let Some(packed) = &self.packed {
            let shape = packed
                .shape(name)
                .ok_or_else(|| format!("tensor {name} not found in packed index"))?;
            if shape.len() != 2 || shape[1] != HIDDEN_SIZE {
                return Err(format!("tensor {name} has invalid shape {shape:?}"));
            }
            let mut out = Vec::with_capacity(token_ids.len() * HIDDEN_SIZE);
            for &id in token_ids {
                let row_idx = usize::try_from(id).map_err(|_| format!("invalid token id {id}"))?;
                if row_idx >= shape[0] {
                    return Err(format!("token id {id} out of embedding table bounds"));
                }
                out.extend_from_slice(&packed.load_row(name, row_idx)?);
            }
            return Ok(out);
        }
        let shard_name = self
            .weight_map
            .get(name)
            .ok_or_else(|| format!("tensor {name} not found in index"))?;
        let shard = self
            .shards
            .get(shard_name)
            .ok_or_else(|| format!("shard {shard_name} not loaded"))?;
        let raw_bytes = shard
            .raw_bytes(name)
            .map_err(|e| format!("failed to get raw bytes for {name}: {e}"))?;

        let mut out = Vec::with_capacity(token_ids.len() * HIDDEN_SIZE);
        for &id in token_ids {
            let row_idx = usize::try_from(id).map_err(|_| format!("invalid token id {id}"))?;
            // Checked so a huge i64 id cannot wrap onto a valid row in release
            // (or panic in debug) before the end-of-table check sees it.
            let row_bytes_len = HIDDEN_SIZE * 2;
            let end = row_idx
                .checked_add(1)
                .and_then(|rows| rows.checked_mul(row_bytes_len));
            let Some(end) = end.filter(|&end| end <= raw_bytes.len()) else {
                return Err(format!("token id {id} out of embedding table bounds"));
            };
            let start = end - row_bytes_len;
            let row_bytes = &raw_bytes[start..end];
            for chunk in row_bytes.chunks_exact(2) {
                let u = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(f32::from_bits((u as u32) << 16));
            }
        }
        Ok(out)
    }
}

impl TensorLoader for ShardedSafetensors {
    fn contains_tensor(&self, name: &str) -> bool {
        self.packed.as_ref().map_or_else(
            || self.weight_map.contains_key(name),
            |packed| packed.contains_tensor(name),
        )
    }

    fn load_tensor(&self, name: &str) -> Result<Vec<f32>, String> {
        ShardedSafetensors::load_tensor(self, name)
    }
}

impl TensorLoader for SafetensorsFile {
    fn contains_tensor(&self, name: &str) -> bool {
        SafetensorsFile::contains_tensor(self, name)
    }

    fn load_tensor(&self, name: &str) -> Result<Vec<f32>, String> {
        self.load_as_f32(name)
            .map_err(|e| format!("failed to load tensor {name}: {e}"))
    }
}

/// Weights for one Qwen3 decoder layer.
pub struct LayerWeights {
    pub input_layernorm: Vec<f32>,
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub gate_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    pub down_proj: Vec<f32>,
}

impl LayerWeights {
    /// Every weight is checked against the Qwen3 layer geometry at load
    /// time, matching the Metal path, so a crafted but self-consistent
    /// checkpoint returns an error instead of tripping a length assert in
    /// `forward_layer`.
    pub fn load(shards: &ShardedSafetensors, layer_idx: usize) -> Result<Self, String> {
        let p = format!("model.layers.{layer_idx}");
        let q_dim = NUM_Q_HEADS * HEAD_DIM;
        let kv_dim = NUM_KV_HEADS * HEAD_DIM;
        let load = |suffix: &str, shape: &[usize]| {
            shards.load_tensor_shape(&format!("{p}.{suffix}"), shape)
        };
        Ok(Self {
            input_layernorm: load("input_layernorm.weight", &[HIDDEN_SIZE])?,
            q_proj: load("self_attn.q_proj.weight", &[q_dim, HIDDEN_SIZE])?,
            k_proj: load("self_attn.k_proj.weight", &[kv_dim, HIDDEN_SIZE])?,
            v_proj: load("self_attn.v_proj.weight", &[kv_dim, HIDDEN_SIZE])?,
            o_proj: load("self_attn.o_proj.weight", &[HIDDEN_SIZE, q_dim])?,
            q_norm: load("self_attn.q_norm.weight", &[HEAD_DIM])?,
            k_norm: load("self_attn.k_norm.weight", &[HEAD_DIM])?,
            post_attention_layernorm: load("post_attention_layernorm.weight", &[HIDDEN_SIZE])?,
            gate_proj: load("mlp.gate_proj.weight", &[INTERMEDIATE_SIZE, HIDDEN_SIZE])?,
            up_proj: load("mlp.up_proj.weight", &[INTERMEDIATE_SIZE, HIDDEN_SIZE])?,
            down_proj: load("mlp.down_proj.weight", &[HIDDEN_SIZE, INTERMEDIATE_SIZE])?,
        })
    }
}

/// Execute Qwen3 layer forward pass in FP32 on CPU.
pub fn forward_layer(x: &mut [f32], seq_len: usize, w: &LayerWeights) {
    let mut normed_x = Vec::with_capacity(seq_len * HIDDEN_SIZE);
    for t in 0..seq_len {
        let token_slice = &x[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
        normed_x.extend(rms_norm(token_slice, &w.input_layernorm, RMS_NORM_EPS));
    }

    let mut q = matmul_bias(
        &normed_x,
        &w.q_proj,
        None,
        seq_len,
        HIDDEN_SIZE,
        NUM_Q_HEADS * HEAD_DIM,
    );
    let mut k = matmul_bias(
        &normed_x,
        &w.k_proj,
        None,
        seq_len,
        HIDDEN_SIZE,
        NUM_KV_HEADS * HEAD_DIM,
    );
    let v = matmul_bias(
        &normed_x,
        &w.v_proj,
        None,
        seq_len,
        HIDDEN_SIZE,
        NUM_KV_HEADS * HEAD_DIM,
    );

    // Q/K normalization and RoPE
    for t in 0..seq_len {
        for h in 0..NUM_Q_HEADS {
            let offset = t * (NUM_Q_HEADS * HEAD_DIM) + h * HEAD_DIM;
            let slice = &mut q[offset..offset + HEAD_DIM];
            let normed = rms_norm(slice, &w.q_norm, RMS_NORM_EPS);
            slice.copy_from_slice(&normed);
            let rotated = rope_neox(slice, 1, 1, HEAD_DIM, HEAD_DIM / 2, t, ROPE_THETA);
            slice.copy_from_slice(&rotated);
        }
        for h in 0..NUM_KV_HEADS {
            let offset = t * (NUM_KV_HEADS * HEAD_DIM) + h * HEAD_DIM;
            let slice = &mut k[offset..offset + HEAD_DIM];
            let normed = rms_norm(slice, &w.k_norm, RMS_NORM_EPS);
            slice.copy_from_slice(&normed);
            let rotated = rope_neox(slice, 1, 1, HEAD_DIM, HEAD_DIM / 2, t, ROPE_THETA);
            slice.copy_from_slice(&rotated);
        }
    }

    // Causal attention with GQA (4 Q heads per KV head)
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let mut attn_out = vec![0.0f32; seq_len * NUM_Q_HEADS * HEAD_DIM];

    for i in 0..seq_len {
        for qh in 0..NUM_Q_HEADS {
            let kv_h = qh / (NUM_Q_HEADS / NUM_KV_HEADS);
            let q_vec = &q[i * (NUM_Q_HEADS * HEAD_DIM) + qh * HEAD_DIM
                ..(i * (NUM_Q_HEADS * HEAD_DIM) + (qh + 1) * HEAD_DIM)];

            // Compute causal logits for j in 0..=i
            let mut logits = Vec::with_capacity(i + 1);
            let mut max_logit = f32::NEG_INFINITY;
            for j in 0..=i {
                let k_vec = &k[j * (NUM_KV_HEADS * HEAD_DIM) + kv_h * HEAD_DIM
                    ..(j * (NUM_KV_HEADS * HEAD_DIM) + (kv_h + 1) * HEAD_DIM)];
                let dot: f32 = q_vec.iter().zip(k_vec).map(|(&a, &b)| a * b).sum();
                let score = dot * scale;
                if score > max_logit {
                    max_logit = score;
                }
                logits.push(score);
            }

            // Softmax
            let mut exp_sum = 0.0f32;
            for s in &mut logits {
                *s = (*s - max_logit).exp();
                exp_sum += *s;
            }
            let inv_sum = 1.0f32 / exp_sum;
            for s in &mut logits {
                *s *= inv_sum;
            }

            // Weighted sum of V vectors
            let out_head = &mut attn_out[i * (NUM_Q_HEADS * HEAD_DIM) + qh * HEAD_DIM
                ..(i * (NUM_Q_HEADS * HEAD_DIM) + (qh + 1) * HEAD_DIM)];
            for (j, &weight) in logits.iter().enumerate() {
                let v_vec = &v[j * (NUM_KV_HEADS * HEAD_DIM) + kv_h * HEAD_DIM
                    ..(j * (NUM_KV_HEADS * HEAD_DIM) + (kv_h + 1) * HEAD_DIM)];
                for d in 0..HEAD_DIM {
                    out_head[d] += weight * v_vec[d];
                }
            }
        }
    }

    // Output projection + residual add
    let o = matmul_bias(
        &attn_out,
        &w.o_proj,
        None,
        seq_len,
        NUM_Q_HEADS * HEAD_DIM,
        HIDDEN_SIZE,
    );
    for idx in 0..x.len() {
        x[idx] += o[idx];
    }

    // Post-attention norm
    let mut post_norm = Vec::with_capacity(seq_len * HIDDEN_SIZE);
    for t in 0..seq_len {
        let token_slice = &x[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
        post_norm.extend(rms_norm(
            token_slice,
            &w.post_attention_layernorm,
            RMS_NORM_EPS,
        ));
    }

    // SwiGLU MLP
    let gate = matmul_bias(
        &post_norm,
        &w.gate_proj,
        None,
        seq_len,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
    );
    let up = matmul_bias(
        &post_norm,
        &w.up_proj,
        None,
        seq_len,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
    );

    let mut act = Vec::with_capacity(seq_len * INTERMEDIATE_SIZE);
    for idx in 0..gate.len() {
        let g = gate[idx];
        let silu = g / (1.0f32 + (-g).exp());
        act.push(silu * up[idx]);
    }

    let down = matmul_bias(
        &act,
        &w.down_proj,
        None,
        seq_len,
        INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
    );
    for idx in 0..x.len() {
        x[idx] += down[idx];
    }
}

/// Compute text encoder conditioning forward pass for token_ids (up to retained unpadded tokens).
///
/// Runs embedding lookup and layers 0..=34 (35 layers), returning layer 34 output pre-final-norm.
pub fn encode_tokens(token_ids: &[i64], shards: &ShardedSafetensors) -> Result<Vec<f32>, String> {
    encode_tokens_with_cancel(token_ids, shards, |_, _| true)
}

/// Execute the text encoder and allow the caller to stop between layers.
pub fn encode_tokens_with_cancel<F>(
    token_ids: &[i64],
    shards: &ShardedSafetensors,
    mut should_continue: F,
) -> Result<Vec<f32>, String>
where
    F: FnMut(usize, usize) -> bool,
{
    let seq_len = token_ids.len();
    if seq_len == 0 {
        return Ok(Vec::new());
    }

    let mut x = shards.load_token_embeddings(token_ids)?;

    for layer_idx in 0..EXTRACT_LAYER_COUNT {
        let layer_weights = LayerWeights::load(shards, layer_idx)?;
        forward_layer(&mut x, seq_len, &layer_weights);
        if !should_continue(layer_idx + 1, EXTRACT_LAYER_COUNT) {
            return Err("image generation cancelled".to_string());
        }
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    const LAYER_TENSORS: [&str; 11] = [
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ];

    fn temporary_directory(label: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        let sequence = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "turbospark-image-text-encoder-{label}-{}-{nanos}-{sequence}",
            std::process::id()
        ));
        fs::create_dir_all(&root).expect("create test directory");
        root
    }

    /// A self-consistent single-file component: a valid one-row embedding
    /// table and every layer-0 tensor present but one element long. Header
    /// shapes match payload bytes, so only a load-time shape check against the
    /// Qwen3 layer geometry can refuse it before the CPU forward asserts.
    fn write_undersized_layer_component(directory: &Path) {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        header.insert(
            "model.embed_tokens.weight".to_string(),
            serde_json::json!({
                "dtype": "BF16",
                "shape": [1, HIDDEN_SIZE],
                "data_offsets": [0, HIDDEN_SIZE * 2]
            }),
        );
        payload.resize(HIDDEN_SIZE * 2, 0);
        for suffix in LAYER_TENSORS {
            let start = payload.len();
            payload.extend_from_slice(&1.0f32.to_le_bytes());
            header.insert(
                format!("model.layers.0.{suffix}"),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [start, payload.len()]
                }),
            );
        }
        let header_bytes = serde_json::to_vec(&header).expect("serialize header");
        let mut file = Vec::with_capacity(8 + header_bytes.len() + payload.len());
        file.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        file.extend_from_slice(&header_bytes);
        file.extend_from_slice(&payload);
        fs::write(directory.join("model.safetensors"), file).expect("write component");
    }

    #[test]
    fn undersized_layer_weights_are_refused_instead_of_panicking() {
        let root = temporary_directory("undersized");
        let source = root.join("source");
        fs::create_dir_all(&source).expect("create source");
        write_undersized_layer_component(&source);
        let packed_dir = root.join("packed");
        crate::packed::pack_component(&source, "model.safetensors.index.json", &packed_dir)
            .expect("pack undersized component");

        for shards in [
            ShardedSafetensors::open(&source).expect("open source component"),
            ShardedSafetensors::open_packed(&packed_dir).expect("open packed component"),
        ] {
            let error =
                encode_tokens(&[0], &shards).expect_err("undersized layer weights must be refused");
            assert!(
                error.contains("input_layernorm.weight") && error.contains("expected"),
                "unexpected error: {error}"
            );
        }
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn huge_token_id_is_refused_without_offset_wrap() {
        let root = temporary_directory("huge-id");
        write_undersized_layer_component(&root);
        let shards = ShardedSafetensors::open(&root).expect("open component");
        // (2^64 / 5120) + 1 rows wraps start to a small in-bounds offset.
        let wrapping_id = (u64::MAX / (HIDDEN_SIZE as u64 * 2) + 1) as i64;
        for id in [wrapping_id, i64::MAX, 1] {
            let error = shards
                .load_token_embeddings(&[id])
                .expect_err("out-of-table id must be refused");
            assert!(error.contains("out of embedding table bounds"), "{error}");
        }
        assert_eq!(
            shards.load_token_embeddings(&[0]).unwrap().len(),
            HIDDEN_SIZE
        );
        let _ = fs::remove_dir_all(&root);
    }
}
