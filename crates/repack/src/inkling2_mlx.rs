//! An MLX conversion of Inkling 2 (mlx-lm PR 2086, `mlx-community/Inkling-2-
//! Mini-4bit` and its kind) presented to the Inkling intake under the HF
//! checkpoint's names, dtypes and shapes (`inkling2_checkpoint::Inkling2Source`).
//!
//! The conversion is mlx-lm's `inkling2/checkpoint.py::sanitize`, and every
//! step it takes is undone here, by name:
//! - Renames (`model.llm.attn.wq_du` -> `model.language_model.self_attn.q_proj`
//!   and the rest) are inverted by [`mlx_to_hf`]. The attention-input queries
//!   shift by one layer (`attn_queries.{i}` is layer `i + 1`'s).
//! - Routed experts keep their source bytes: `gate_up_proj.weight` is the HF
//!   `w13_weight` viewed as u32 (same interleaved gate/up rows), `scales` the
//!   HF `w13_weight_scale`. NVFP4's per-expert `global_scale` is
//!   `amax * 6 * 448 / denominator`, so it is presented as the `amax` with a
//!   denominator of 2688, which makes the intake's `amax / denominator` equal
//!   `global_scale / 2688`, the HF outer scale up to f32 rounding.
//! - `conv1d.weight` is `[C, taps, 1]` from the HF `[C, 1, taps]` by a swap
//!   with a size-1 axis, so the bytes are already in HF order.
//! - The router and the loop gates' `forget`/`gain` projections were widened
//!   BF16 to F32; they are narrowed back, and refused unless every value is
//!   exactly a BF16.
//! - The two cross-pass query tensors were merged as `[q0, kv0, kv1, q1]` per
//!   core layer into `message_input.query`; they are split back.
//! - Dense projections, the embedding and the output head are MLX affine
//!   (`weight` u32, BF16 `scales`/`biases`, group 64). They are passed through
//!   as INT4/INT8 resident entries ([`Inkling2Source::affine`]).
//!
//! The vision tower and any MTP draft are not read: the install is text-only.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use model_io::SafetensorsFile;

use crate::inkling2_checkpoint::{AffineTensor, Inkling2RepackError, Inkling2Source, SourceTensor};

/// `6 * 448`: E2M1's largest value times E4M3's, the factor mlx-lm folds into
/// `global_scale`.
pub const MLX_NVFP4_GLOBAL_FACTOR: f32 = 2688.0;

#[path = "inkling2_mlx_oq.rs"]
pub(crate) mod oq;

fn err<T>(msg: impl Into<String>) -> Result<T, Inkling2RepackError> {
    Err(Inkling2RepackError(msg.into()))
}

/// The HF name of one MLX tensor name, or None for a name the text install
/// does not read (vision tower, the merged message queries, expert planes,
/// which the source maps by hand). `.scales`/`.biases` companions map to None.
pub fn mlx_to_hf(name: &str) -> Option<String> {
    if name.starts_with("model.vision_tower.")
        || name.ends_with(".scales")
        || name.ends_with(".biases")
        || name.contains(".mlp.experts.")
        || name == "model.language_model.message_input.query"
    {
        return None;
    }
    if let Some(rest) = name.strip_prefix("lm_head.") {
        return Some(format!("model.llm.unembed.{rest}"));
    }
    // `model.language_model.layers.{L}.{attention_input|mlp_input}.query`.
    if let Some(rest) = name.strip_prefix("model.language_model.layers.") {
        if let Some((l, tail)) = rest.split_once('.') {
            if let Ok(l) = l.parse::<usize>() {
                match tail {
                    "attention_input.query" => {
                        return Some(format!(
                            "model.llm.attn_res.attn_queries.{}",
                            l.checked_sub(1)?
                        ));
                    }
                    "mlp_input.query" => {
                        return Some(format!("model.llm.attn_res.mlp_queries.{l}"));
                    }
                    _ => {}
                }
            }
        }
    }
    if name == "model.language_model.final_input.query" {
        return Some("model.llm.attn_res.output_query".into());
    }
    let mut n = if let Some(rest) = name.strip_prefix("model.language_model.embed_tokens.") {
        format!("model.llm.embed.{rest}")
    } else {
        format!("model.llm.{}", name.strip_prefix("model.language_model.")?)
    };
    // Applied in this order: specific names before the generic `self_attn`.
    for (from, to) in [
        (".self_attn.q_proj.", ".attn.wq_du."),
        (".self_attn.k_proj.", ".attn.wk_dv."),
        (".self_attn.v_proj.", ".attn.wv_dv."),
        (".self_attn.r_proj.", ".attn.wr_du."),
        (".self_attn.o_proj.", ".attn.wo_ud."),
        (".self_attn.gate_proj.", ".attn.gate_attn."),
        (".self_attn.sink_proj.", ".attn.attn_sink_proj."),
        (".q_message_gate.", ".controllers.attn."),
        (".k_message_gate.", ".controllers.k."),
        (".v_message_gate.", ".controllers.v."),
        (".mlp_message_gate.", ".controllers.mlp."),
        (".self_attn.", ".attn."),
        (
            ".post_attention_convolution.conv1d.weight",
            ".attn_sconv.weight",
        ),
        (".post_mlp_convolution.conv1d.weight", ".mlp_sconv.weight"),
        ("sconv.conv1d.weight", "sconv.weight"),
        ("sconv.static_weight", "sconv.static_w"),
        ("sconv.weight_proj.weight", "sconv.weight_proj"),
        (".post_attention_layernorm.", ".mlp_norm."),
        (".input_layernorm.", ".attn_norm."),
        (".q_input_layernorm.", ".q_input_norm."),
        (".k_input_layernorm.", ".k_input_norm."),
        (".v_input_layernorm.", ".v_input_norm."),
        (".mlp.gate.e_score_correction_bias", ".mlp.gate.bias"),
    ] {
        n = n.replace(from, to);
    }
    // The loop controllers' planes, after the `_message_gate` rename above.
    if n.contains(".controllers.") {
        for (from, to) in [
            (".down_proj.weight", ".down_weight"),
            (".forget_gate_proj.weight", ".forget_weight"),
            (".message_gate_proj.weight", ".gain_weight"),
        ] {
            n = n.replace(from, to);
        }
    }
    Some(n)
}

/// How one HF name is produced from the MLX shards.
enum Mapped {
    /// An MLX tensor, possibly relabeled (dtype, shape) or narrowed F32->BF16.
    Plain {
        mlx: String,
        dtype: Option<&'static str>,
        shape: Option<Vec<usize>>,
        narrow: bool,
    },
    /// An affine matrix: the MLX base name (without `.weight`), and the
    /// `(first, count)` rows of it when the HF matrix is one slice.
    Affine {
        base: String,
        bits: u8,
        rows: Option<(usize, usize)>,
    },
    /// Bytes the source builds itself.
    Synth {
        dtype: &'static str,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    },
}

/// The MLX shards of one conversion, read as the HF checkpoint.
pub struct Inkling2MlxSource {
    shards: Vec<SafetensorsFile>,
    at: HashMap<String, usize>,
    map: BTreeMap<String, Mapped>,
    /// Affine bit widths seen on dense matrices, for the manifest.
    pub affine_bits: std::collections::BTreeSet<u8>,
}

impl Inkling2MlxSource {
    /// Opens every `*.safetensors` shard in `dir` listed by its index (or the
    /// one `model.safetensors`) and builds the HF name map. `layers`, `experts`,
    /// `core` (looped layers) and `hidden` come from the parsed config.
    pub fn open(
        dir: &Path,
        layers: usize,
        core: usize,
        hidden: usize,
    ) -> Result<Self, Inkling2RepackError> {
        let mut files: Vec<String> =
            match std::fs::read_to_string(dir.join("model.safetensors.index.json")) {
                Ok(text) => {
                    let v: serde_json::Value = serde_json::from_str(&text)
                        .map_err(|e| Inkling2RepackError(e.to_string()))?;
                    let mut f: Vec<String> = v["weight_map"]
                        .as_object()
                        .ok_or_else(|| Inkling2RepackError("index has no weight_map".into()))?
                        .values()
                        .filter_map(|x| x.as_str().map(str::to_string))
                        .collect();
                    f.sort();
                    f.dedup();
                    f
                }
                Err(_) => vec!["model.safetensors".into()],
            };
        files.retain(|f| f.ends_with(".safetensors") && !f.contains('/'));
        let mut shards = Vec::with_capacity(files.len());
        let mut at = HashMap::new();
        for (i, f) in files.iter().enumerate() {
            let s = SafetensorsFile::open(&dir.join(f))
                .map_err(|e| Inkling2RepackError(format!("{f}: {e}")))?;
            for n in s.tensor_names() {
                at.insert(n.to_string(), i);
            }
            shards.push(s);
        }
        let mut src = Self {
            shards,
            at,
            map: BTreeMap::new(),
            affine_bits: Default::default(),
        };
        if oq::is_oq(&src) {
            src.build_oq_map(layers, core, hidden)?;
        } else {
            src.build_map(layers, core, hidden)?;
        }
        Ok(src)
    }

    fn desc(&self, mlx: &str) -> Option<&model_io::safetensors::TensorDescriptor> {
        self.at
            .get(mlx)
            .and_then(|&i| self.shards[i].descriptor(mlx))
    }

    fn bytes(&self, mlx: &str) -> Result<&[u8], Inkling2RepackError> {
        let i = *self
            .at
            .get(mlx)
            .ok_or_else(|| Inkling2RepackError(format!("MLX tensor {mlx} missing")))?;
        self.shards[i]
            .raw_bytes(mlx)
            .map_err(|e| Inkling2RepackError(format!("{mlx}: {e}")))
    }

    fn build_map(
        &mut self,
        layers: usize,
        core: usize,
        hidden: usize,
    ) -> Result<(), Inkling2RepackError> {
        let names: Vec<String> = self.at.keys().cloned().collect();
        for mlx in &names {
            let Some(hf) = mlx_to_hf(mlx) else { continue };
            let d = self.desc(mlx).expect("listed").clone();
            let base = mlx.strip_suffix(".weight");
            let quantized = d.dtype == "U32"
                && base.is_some_and(|b| self.at.contains_key(&format!("{b}.scales")));
            let mapped = if quantized {
                let base = base.expect("checked").to_string();
                let s = self.desc(&format!("{base}.scales")).expect("checked");
                let words = d.shape[1];
                let groups = s.shape[1];
                let bits = (words * 32 / (groups * 64)) as u8;
                self.affine_bits.insert(bits);
                Mapped::Affine {
                    base,
                    bits,
                    rows: None,
                }
            } else if mlx.ends_with("conv1d.weight") {
                // [C, taps, 1] -> [C, 1, taps]: a swap with a size-1 axis.
                if d.shape.len() != 3 || d.shape[2] != 1 {
                    return err(format!("{mlx}: expected [C, taps, 1], got {:?}", d.shape));
                }
                Mapped::Plain {
                    mlx: mlx.clone(),
                    dtype: None,
                    shape: Some(vec![d.shape[0], 1, d.shape[1]]),
                    narrow: false,
                }
            } else {
                let narrow = d.dtype == "F32"
                    && (hf.ends_with(".mlp.gate.weight")
                        || hf.ends_with(".forget_weight")
                        || hf.ends_with(".gain_weight"));
                Mapped::Plain {
                    mlx: mlx.clone(),
                    dtype: None,
                    shape: None,
                    narrow,
                }
            };
            self.map.insert(hf, mapped);
        }
        // Routed experts, per layer.
        for l in 0..layers {
            let m = format!("model.language_model.layers.{l}.mlp.experts");
            let h = format!("model.llm.layers.{l}.mlp.experts");
            let nvfp4 = self
                .at
                .contains_key(&format!("{m}.gate_up_proj.global_scale"));
            for (hf_p, mlx_p) in [("w13", "gate_up_proj"), ("w2", "down_proj")] {
                let w = format!("{m}.{mlx_p}.weight");
                let s = format!("{m}.{mlx_p}.scales");
                let (wd, sd) = (
                    self.desc(&w)
                        .ok_or_else(|| Inkling2RepackError(format!("{w} missing")))?
                        .clone(),
                    self.desc(&s)
                        .ok_or_else(|| Inkling2RepackError(format!("{s} missing")))?
                        .clone(),
                );
                if wd.dtype != "U32" || sd.dtype != "U8" || wd.shape.len() != 3 {
                    return err(format!(
                        "{w}: expected U32 weights and U8 scales, got {} {:?} and {}",
                        wd.dtype, wd.shape, sd.dtype
                    ));
                }
                let (wdt, sdt) = if nvfp4 {
                    ("U8", "F8_E4M3")
                } else {
                    ("F8_E4M3", "U8")
                };
                let row_bytes = wd.shape[2] * 4;
                self.map.insert(
                    format!("{h}.{hf_p}_weight"),
                    Mapped::Plain {
                        mlx: w,
                        dtype: Some(wdt),
                        shape: Some(vec![wd.shape[0], wd.shape[1], row_bytes]),
                        narrow: false,
                    },
                );
                self.map.insert(
                    format!("{h}.{hf_p}_weight_scale"),
                    Mapped::Plain {
                        mlx: s,
                        dtype: Some(sdt),
                        shape: None,
                        narrow: false,
                    },
                );
                if nvfp4 {
                    let g = format!("{m}.{mlx_p}.global_scale");
                    if !self.at.contains_key(&g) {
                        return err(format!("{g} missing on an NVFP4 layer"));
                    }
                    self.map.insert(
                        format!("{h}.{hf_p}_weight_amax"),
                        Mapped::Plain {
                            mlx: g,
                            dtype: None,
                            shape: None,
                            narrow: false,
                        },
                    );
                    self.map.insert(
                        format!("{h}.{hf_p}_weight_outer_scale_denominator"),
                        Mapped::Synth {
                            dtype: "F32",
                            shape: vec![1],
                            bytes: MLX_NVFP4_GLOBAL_FACTOR.to_le_bytes().to_vec(),
                        },
                    );
                }
            }
        }
        // The cross-pass queries, split back from `[q0, kv0, kv1, q1]` rows.
        let mq = "model.language_model.message_input.query";
        let d = self
            .desc(mq)
            .ok_or_else(|| Inkling2RepackError(format!("{mq} missing")))?
            .clone();
        if d.dtype != "F32" || d.shape != [core, 4, hidden] {
            return err(format!(
                "{mq}: expected F32 [{core}, 4, {hidden}], got {} {:?}",
                d.dtype, d.shape
            ));
        }
        let b = self.bytes(mq)?;
        let row = |i: usize, j: usize| &b[(i * 4 + j) * hidden * 4..(i * 4 + j + 1) * hidden * 4];
        let (mut qs, mut kvs) = (Vec::new(), Vec::new());
        for i in 0..core {
            qs.extend_from_slice(row(i, 0));
            qs.extend_from_slice(row(i, 3));
            kvs.extend_from_slice(row(i, 1));
            kvs.extend_from_slice(row(i, 2));
        }
        for (name, bytes) in [
            ("model.llm.middle_cycle_cross_pass_queries", qs),
            ("model.llm.middle_cycle_cross_pass_kv_queries", kvs),
        ] {
            self.map.insert(
                name.into(),
                Mapped::Synth {
                    dtype: "F32",
                    shape: vec![2 * core, hidden],
                    bytes,
                },
            );
        }
        Ok(())
    }
}

/// F32 bytes to BF16, refusing any value that is not exactly a BF16.
fn narrow_f32(name: &str, b: &[u8]) -> Result<Vec<u8>, Inkling2RepackError> {
    let mut out = Vec::with_capacity(b.len() / 2);
    for c in b.chunks_exact(4) {
        let v = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        if v & 0xFFFF != 0 {
            return err(format!(
                "{name}: an F32 value is not exactly a BF16, so it cannot be narrowed losslessly"
            ));
        }
        out.extend_from_slice(&((v >> 16) as u16).to_le_bytes());
    }
    Ok(out)
}

impl Inkling2Source for Inkling2MlxSource {
    fn tensor_names(&self) -> Vec<String> {
        self.map.keys().cloned().collect()
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    fn tensor(&self, name: &str) -> Result<SourceTensor<'_>, Inkling2RepackError> {
        match self.map.get(name) {
            None => err(format!("missing tensor {name}")),
            Some(Mapped::Affine { .. }) => err(format!(
                "{name} is affine-quantized in this MLX conversion; read it through affine()"
            )),
            Some(Mapped::Synth {
                dtype,
                shape,
                bytes,
            }) => Ok((
                dtype.to_string(),
                shape.clone(),
                Cow::Borrowed(bytes.as_slice()),
            )),
            Some(Mapped::Plain {
                mlx,
                dtype,
                shape,
                narrow,
            }) => {
                let d = self.desc(mlx).expect("mapped");
                let b = self.bytes(mlx)?;
                if *narrow {
                    return Ok((
                        "BF16".into(),
                        d.shape.clone(),
                        Cow::Owned(narrow_f32(mlx, b)?),
                    ));
                }
                Ok((
                    dtype.map_or_else(|| d.dtype.clone(), str::to_string),
                    shape.clone().unwrap_or_else(|| d.shape.clone()),
                    Cow::Borrowed(b),
                ))
            }
        }
    }

    fn affine(&self, name: &str) -> Option<AffineTensor<'_>> {
        let Some(Mapped::Affine { base, bits, rows }) = self.map.get(name) else {
            return None;
        };
        let w = self.desc(&format!("{base}.weight"))?;
        // A stacked expert tensor is `[E, R, words]`: its rows are flattened.
        let words = *w.shape.last()?;
        let total: usize = w.shape[..w.shape.len() - 1].iter().product();
        let (first, n) = rows.unwrap_or((0, total));
        let cols = words * 32 / *bits as usize;
        let slice = |suffix: &str, row_bytes: usize| {
            self.bytes(&format!("{base}.{suffix}"))
                .ok()?
                .get(first * row_bytes..(first + n) * row_bytes)
        };
        Some(AffineTensor {
            bits: *bits,
            rows: n,
            cols,
            packed: slice("weight", words * 4)?,
            scales: slice("scales", cols / 64 * 2)?,
            biases: slice("biases", cols / 64 * 2)?,
        })
    }
}

/// The manifest quant section for an MLX conversion: affine attention and
/// embedding at the conversion's dense bit width (one width only), the F32
/// router, the Inkling routed scheme.
pub fn inkling2_mlx_manifest_quant(bits: u8) -> serde_json::Value {
    let affine = serde_json::json!({
        "weightBits": bits, "scheme": "affine", "scaleType": "bf16", "biasType": "bf16", "groupSize": 64
    });
    let mut q = crate::inkling2_checkpoint::inkling2_manifest_quant();
    q["embedding"] = affine.clone();
    q["attention"] = affine;
    q
}

#[cfg(test)]
mod tests {
    use super::mlx_to_hf;

    /// One name per rule of mlx-lm's `sanitize`, inverted. The full set was
    /// checked against a real install's 646 resident names
    /// (`examples/inkling2_mlx_install.rs`); these hold the rules in CI.
    #[test]
    fn mlx_names_map_back_to_the_hf_checkpoint() {
        for (mlx, hf) in [
            ("lm_head.weight", "model.llm.unembed.weight"),
            (
                "model.language_model.embed_tokens.weight",
                "model.llm.embed.weight",
            ),
            (
                "model.language_model.embed_norm.weight",
                "model.llm.embed_norm.weight",
            ),
            (
                "model.language_model.final_input.query",
                "model.llm.attn_res.output_query",
            ),
            // Layer L's attention input is HF query L - 1.
            (
                "model.language_model.layers.5.attention_input.query",
                "model.llm.attn_res.attn_queries.4",
            ),
            (
                "model.language_model.layers.5.mlp_input.query",
                "model.llm.attn_res.mlp_queries.5",
            ),
            (
                "model.language_model.layers.3.self_attn.q_proj.weight",
                "model.llm.layers.3.attn.wq_du.weight",
            ),
            (
                "model.language_model.layers.3.self_attn.o_proj.weight",
                "model.llm.layers.3.attn.wo_ud.weight",
            ),
            (
                "model.language_model.layers.3.self_attn.gate_proj.weight",
                "model.llm.layers.3.attn.gate_attn.weight",
            ),
            (
                "model.language_model.layers.3.self_attn.sink_proj.weight",
                "model.llm.layers.3.attn.attn_sink_proj.weight",
            ),
            (
                "model.language_model.layers.3.self_attn.q_l2_clip.scale",
                "model.llm.layers.3.attn.q_l2_clip.scale",
            ),
            (
                "model.language_model.layers.5.self_attn.k_sconv.weight_proj.weight",
                "model.llm.layers.5.attn.k_sconv.weight_proj",
            ),
            (
                "model.language_model.layers.5.self_attn.k_sconv.static_weight",
                "model.llm.layers.5.attn.k_sconv.static_w",
            ),
            (
                "model.language_model.layers.3.self_attn.v_sconv.conv1d.weight",
                "model.llm.layers.3.attn.v_sconv.weight",
            ),
            (
                "model.language_model.layers.3.post_attention_convolution.conv1d.weight",
                "model.llm.layers.3.attn_sconv.weight",
            ),
            (
                "model.language_model.layers.3.post_mlp_convolution.conv1d.weight",
                "model.llm.layers.3.mlp_sconv.weight",
            ),
            (
                "model.language_model.layers.3.input_layernorm.weight",
                "model.llm.layers.3.attn_norm.weight",
            ),
            (
                "model.language_model.layers.5.q_input_layernorm.weight",
                "model.llm.layers.5.q_input_norm.weight",
            ),
            (
                "model.language_model.layers.3.post_attention_layernorm.weight",
                "model.llm.layers.3.mlp_norm.weight",
            ),
            (
                "model.language_model.layers.3.mlp.gate.e_score_correction_bias",
                "model.llm.layers.3.mlp.gate.bias",
            ),
            (
                "model.language_model.layers.7.q_message_gate.down_proj.weight",
                "model.llm.layers.7.controllers.attn.down_weight",
            ),
            (
                "model.language_model.layers.7.mlp_message_gate.forget_gate_proj.weight",
                "model.llm.layers.7.controllers.mlp.forget_weight",
            ),
            (
                "model.language_model.layers.7.k_message_gate.message_gate_proj.weight",
                "model.llm.layers.7.controllers.k.gain_weight",
            ),
        ] {
            assert_eq!(mlx_to_hf(mlx).as_deref(), Some(hf), "{mlx}");
        }
        for skipped in [
            "model.vision_tower.embeddings.patch_embedding.weight",
            "model.language_model.layers.3.self_attn.q_proj.scales",
            "model.language_model.layers.3.mlp.experts.gate_up_proj.weight",
            "model.language_model.message_input.query",
            // Layer 0 has no attention input.
            "model.language_model.layers.0.attention_input.query",
        ] {
            assert_eq!(mlx_to_hf(skipped), None, "{skipped}");
        }
    }
}
