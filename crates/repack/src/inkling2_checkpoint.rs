//! Inkling 2 (`inkling2_loop`) intake: the HF safetensors checkpoint to a
//! streamed-MoE `.gturbo` install. Facts: `docs/INKLING2_PHASE0.md`.
//!
//! Its own writer rather than an arm of the Gemma-shaped walk, because almost
//! nothing about the checkpoint matches that walk: routed experts are stacked
//! per projection (`mlp.experts.w13_weight [E, 2F, H/2]`) in NVFP4 or MXFP8
//! with per-expert outer scales, there is no shared expert, and roughly a
//! third of the resident tensors are F32.
//!
//! What the install holds:
//! - `model_weights.bin`: every `model.llm.*` tensor except the routed
//!   experts, verbatim (BF16 or F32) with two conventions undone here, keyed by
//!   name: the router (`mlp.gate.weight`) is widened BF16 to F32, exactly,
//!   because llama.cpp runs this router in F32 and the raw-F32 router is the
//!   manifest shape the loader admits; and the full-attention
//!   `{k,v}_sconv.static_w` (`[taps, ch]`, indexed by LAG) is rewritten as
//!   `{k,v}_sconv.weight` (`[ch, 1, taps]`, tap `taps - 1` on the current row),
//!   the layout every other conv already has. Vision and MTP tensors are left
//!   out: the install is text-only.
//! - `packed_experts/layer_NN.bin`: per expert, a 16-byte record of F32 outer
//!   scales (gate, up, down, pad; `amax / outer_scale_denominator`, or 1.0 for
//!   MXFP8), then the gate rows (even `w13` rows), up rows (odd), and down rows,
//!   each transcoded losslessly by `compute::inkling2`. This is the blob
//!   contract `gpu::moe_inkling2` reads.
//! - `inkling2.json`: the sidecar (`model_io::inkling2_config`).

use std::path::Path;

use compute::inkling2::{
    mxfp8_row_to_blocks, nvfp4_row_to_blocks, MXFP8_BLOCK_BYTES, MXFP8_BLOCK_ELEMS,
    NVFP4_BLOCK_BYTES, NVFP4_BLOCK_ELEMS,
};
use model_io::inkling2_config::{Inkling2Config, INKLING2_CONFIG_FILE};
use model_io::{ArchConfig, SafetensorsFile, INKLING2_ROUTED_SCHEME};
use serde_json::Value;

use crate::gturbo_writer::{ExpertBlob, LayerBlobs, StreamingGturboWriter, SubTensor};
use crate::resident_writer::{
    write_resident_weights_bin_mixed, RawTensorSpec, ResidentEntrySpec, DTYPE_BF16, DTYPE_FP32,
};

/// Bytes of the per-expert outer-scale record at the front of every blob.
pub const INKLING2_OUTER_RECORD_BYTES: usize = 16;

/// Sub-tensor dtype of an NVFP4 projection inside an expert blob.
pub const INKLING2_NVFP4_DTYPE: &str = "inkling2_nvfp4";
/// Sub-tensor dtype of an MXFP8 projection inside an expert blob.
pub const INKLING2_MXFP8_DTYPE: &str = "inkling2_mxfp8";

/// Files the tokenizer and chat path need, copied beside the install.
/// `tokenizer.json` is required; the rest are copied when present.
pub const INKLING2_TOKENIZER_FILES: &[&str] = &[
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
];

/// A refused or failed Inkling intake.
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2RepackError(pub String);

impl std::fmt::Display for Inkling2RepackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "inkling2 intake: {}", self.0)
    }
}

impl std::error::Error for Inkling2RepackError {}

fn err<T>(msg: impl Into<String>) -> Result<T, Inkling2RepackError> {
    Err(Inkling2RepackError(msg.into()))
}

/// Parses `config.json` into the install's `ArchConfig` and sidecar.
///
/// Refuses every value the flow does not implement instead of installing it:
/// a non-sigmoid router, attention or output biases, a shared expert, a V norm
/// with a gamma, unnormalized residual values, a variable loop, and SWA heads
/// that differ from the full-attention heads.
pub fn parse_inkling2_config(
    json: &str,
) -> Result<(ArchConfig, Inkling2Config), Inkling2RepackError> {
    let top: Value = serde_json::from_str(json).map_err(|e| Inkling2RepackError(e.to_string()))?;
    if top.get("model_type").and_then(Value::as_str) != Some("inkling2_loop") {
        return err(format!(
            "model_type {:?} is not inkling2_loop",
            top.get("model_type")
        ));
    }
    let t = top
        .get("text_config")
        .ok_or_else(|| Inkling2RepackError("missing text_config".into()))?;
    let int = |k: &str| -> Result<i64, Inkling2RepackError> {
        t.get(k).and_then(Value::as_i64).ok_or_else(|| {
            Inkling2RepackError(format!("text_config.{k} missing or not an integer"))
        })
    };
    let float = |k: &str| -> Result<f64, Inkling2RepackError> {
        t.get(k)
            .and_then(Value::as_f64)
            .ok_or_else(|| Inkling2RepackError(format!("text_config.{k} missing or not a number")))
    };
    let flag = |k: &str, want: bool| -> Result<(), Inkling2RepackError> {
        match t.get(k).and_then(Value::as_bool) {
            Some(v) if v == want => Ok(()),
            Some(v) => err(format!(
                "text_config.{k} is {v}; only {want} is implemented"
            )),
            None => err(format!("text_config.{k} missing")),
        }
    };
    flag("q_bias", false)?;
    flag("o_bias", false)?;
    flag("use_value_norm_no_gamma", true)?;
    flag("attn_res_normalize_values", true)?;
    flag("use_gate_bias", true)?;
    if int("n_shared_experts")? != 0 {
        return err("a shared expert is not implemented");
    }
    if let Some(f) = t.get("scoring_func").and_then(Value::as_str) {
        if f != "sigmoid" {
            return err(format!("router scoring_func {f} is not implemented"));
        }
    }
    let heads = int("num_attention_heads")?;
    let kv = int("num_key_value_heads")?;
    let head_dim = int("head_dim")?;
    for (k, want) in [
        ("swa_head_dim", head_dim),
        ("swa_num_attention_heads", heads),
        ("swa_num_key_value_heads", kv),
    ] {
        if int(k)? != want {
            return err(format!("{k} differs from the full-attention value {want}"));
        }
    }
    let layers = int("num_hidden_layers")?;
    let local: Vec<i64> = t
        .get("local_layer_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| Inkling2RepackError("local_layer_ids missing".into()))?
        .iter()
        .map(|v| {
            v.as_i64()
                .ok_or_else(|| Inkling2RepackError("bad local_layer_ids".into()))
        })
        .collect::<Result<_, _>>()?;
    let mask: Vec<u8> = (0..layers).map(|i| u8::from(!local.contains(&i))).collect();

    let lc = t
        .get("loop_config")
        .ok_or_else(|| Inkling2RepackError("loop_config missing".into()))?;
    if lc.get("is_varloop").and_then(Value::as_bool) != Some(false) {
        return err("a variable loop (is_varloop) is not implemented");
    }
    let lint = |k: &str| -> Result<u32, Inkling2RepackError> {
        lc.get(k)
            .and_then(Value::as_u64)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| Inkling2RepackError(format!("loop_config.{k} missing")))
    };
    let u32_of = |k: &str| -> Result<u32, Inkling2RepackError> {
        u32::try_from(int(k)?).map_err(|_| Inkling2RepackError(format!("{k} out of range")))
    };

    let mut arch = model_io::inkling2_mini();
    arch.hidden_size = int("hidden_size")?;
    arch.moe_intermediate_size = int("intermediate_size")?;
    arch.num_heads = heads;
    arch.num_kv_heads = kv;
    arch.num_full_kv_heads = kv;
    arch.head_dim = head_dim;
    arch.full_head_dim = head_dim;
    arch.vocab_size = int("vocab_size")?;
    arch.sliding_window = int("sliding_window_size")?;
    arch.final_logit_softcap = float("final_logit_softcapping")?;
    arch.num_layers = layers;
    arch.num_experts = int("n_routed_experts")?;
    arch.top_k_experts = int("num_experts_per_tok")?;
    arch.full_attention_layer_mask = mask;
    // `powf(-0.5)` is the correctly rounded 2^-3.5 the other head_dim-128
    // baselines carry; `1.0 / sqrt` lands 2 ULP away and fails the manifest's
    // exact float comparison.
    arch.attention_scale = (head_dim as f64).powf(-0.5);
    arch.routed_scaling_factor = float("route_scale")?;
    arch.short_conv_kernel_size = int("sconv_kernel_size")?;

    let side = Inkling2Config {
        core_start: lint("core_start")?,
        core_end: lint("core_end")?,
        passes: lint("num_passes")?,
        gate_rank: lint("gate_rank")?,
        d_rel: u32_of("d_rel")?,
        rel_extent: u32_of("rel_extent")?,
        qk_clip: float("qk_l2_clip_threshold")? as f32,
        rel_clip: float("rel_l2_clip_threshold")? as f32,
        log_floor: float("log_scaling_n_floor")? as f32,
        log_alpha: float("log_scaling_alpha")? as f32,
        gate_softcap: float("mlp_gate_softcap")? as f32,
        up_softcap: float("mlp_up_softcap")? as f32,
        res_block: u32_of("n_layers_per_attn_res_block")?,
        unpadded_vocab: u32_of("unpadded_vocab_size")?,
        rms_eps: float("rms_norm_eps")? as f32,
        eos_token_id: top
            .get("eos_token_id")
            .and_then(Value::as_u64)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| Inkling2RepackError("eos_token_id missing".into()))?,
    };
    side.validate_against(&arch)
        .map_err(|e| Inkling2RepackError(e.to_string()))?;
    Ok((arch, side))
}

/// The routed-expert scheme of one layer, from its `w13_weight` dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scheme {
    Nvfp4,
    Mxfp8,
}

impl Scheme {
    fn row_bytes(self, cols: usize) -> usize {
        match self {
            Scheme::Nvfp4 => cols / NVFP4_BLOCK_ELEMS * NVFP4_BLOCK_BYTES,
            Scheme::Mxfp8 => cols / MXFP8_BLOCK_ELEMS * MXFP8_BLOCK_BYTES,
        }
    }
    fn dtype(self) -> &'static str {
        match self {
            Scheme::Nvfp4 => INKLING2_NVFP4_DTYPE,
            Scheme::Mxfp8 => INKLING2_MXFP8_DTYPE,
        }
    }
    /// Source bytes of one row of `cols` elements, and of its scales.
    fn src_bytes(self, cols: usize) -> (usize, usize) {
        match self {
            Scheme::Nvfp4 => (cols / 2, cols / 16),
            Scheme::Mxfp8 => (cols, cols / 32),
        }
    }
}

/// One tensor as the intake reads it: safetensors dtype string, shape, bytes.
pub type SourceTensor<'a> = (String, Vec<usize>, std::borrow::Cow<'a, [u8]>);

/// An MLX affine-quantized dense matrix (`weight` packed u32, BF16 `scales`
/// and `biases` per group of 64), passed through to the install as is.
pub struct AffineTensor<'a> {
    pub bits: u8,
    pub rows: usize,
    /// Logical columns (packed u32 words x 32 / bits).
    pub cols: usize,
    pub packed: &'a [u8],
    pub scales: &'a [u8],
    pub biases: &'a [u8],
}

/// Where the intake reads tensors from, always by their HF checkpoint names
/// (`model.llm.*`) and in the HF dtypes and shapes. The HF checkpoint is one
/// safetensors file read directly; an MLX conversion is presented under the
/// same names by `inkling2_mlx` so everything below has one path.
pub trait Inkling2Source {
    fn tensor_names(&self) -> Vec<String>;
    fn contains_tensor(&self, name: &str) -> bool;
    fn tensor(&self, name: &str) -> Result<SourceTensor<'_>, Inkling2RepackError>;
    /// A dense matrix the source holds affine-quantized, which the intake
    /// writes as an INT4/INT8 resident entry instead of reading `tensor`.
    fn affine(&self, _name: &str) -> Option<AffineTensor<'_>> {
        None
    }
}

impl Inkling2Source for SafetensorsFile {
    fn tensor_names(&self) -> Vec<String> {
        SafetensorsFile::tensor_names(self)
            .map(str::to_string)
            .collect()
    }
    fn contains_tensor(&self, name: &str) -> bool {
        SafetensorsFile::contains_tensor(self, name)
    }
    fn tensor(&self, name: &str) -> Result<SourceTensor<'_>, Inkling2RepackError> {
        let d = self
            .descriptor(name)
            .ok_or_else(|| Inkling2RepackError(format!("missing tensor {name}")))?;
        let bytes = self
            .raw_bytes(name)
            .map_err(|e| Inkling2RepackError(format!("{name}: {e}")))?;
        Ok((
            d.dtype.clone(),
            d.shape.clone(),
            std::borrow::Cow::Borrowed(bytes),
        ))
    }
}

/// One tensor's dtype, shape and bytes, checked.
fn tensor<'a>(
    src: &'a dyn Inkling2Source,
    name: &str,
) -> Result<SourceTensor<'a>, Inkling2RepackError> {
    src.tensor(name)
}

fn f32_scalars(
    src: &dyn Inkling2Source,
    name: &str,
    n: usize,
) -> Result<Vec<f32>, Inkling2RepackError> {
    let (dt, shape, b) = tensor(src, name)?;
    if dt != "F32" || shape.iter().product::<usize>() != n {
        return err(format!(
            "{name}: expected F32 with {n} elements, got {dt} {shape:?}"
        ));
    }
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Bytes one expert's blob uses (before the page round-up).
fn expert_blob_bytes(scheme: Scheme, hidden: usize, ffn: usize) -> usize {
    INKLING2_OUTER_RECORD_BYTES
        + 2 * ffn * scheme.row_bytes(hidden)
        + hidden * scheme.row_bytes(ffn)
}

fn layer_scheme(src: &dyn Inkling2Source, layer: usize) -> Result<Scheme, Inkling2RepackError> {
    let name = format!("model.llm.layers.{layer}.mlp.experts.w13_weight");
    let (dt, _, _) = tensor(src, &name)?;
    match dt.as_str() {
        "U8" => Ok(Scheme::Nvfp4),
        "F8_E4M3" => Ok(Scheme::Mxfp8),
        other => err(format!(
            "{name}: dtype {other} is neither NVFP4 (U8) nor MXFP8 (F8_E4M3)"
        )),
    }
}

/// Builds one layer's 128 expert blobs.
fn build_layer(
    src: &dyn Inkling2Source,
    layer: usize,
    experts: usize,
    hidden: usize,
    ffn: usize,
) -> Result<LayerBlobs, Inkling2RepackError> {
    let scheme = layer_scheme(src, layer)?;
    let p = format!("model.llm.layers.{layer}.mlp.experts");
    let (w13_rb, s13_rb) = scheme.src_bytes(hidden);
    let (w2_rb, s2_rb) = scheme.src_bytes(ffn);
    let check = |name: &str,
                 want_dt: &str,
                 want: [usize; 3]|
     -> Result<std::borrow::Cow<'_, [u8]>, Inkling2RepackError> {
        let (dt, shape, b) = tensor(src, name)?;
        if dt != want_dt || shape != want {
            return err(format!(
                "{name}: expected {want_dt} {want:?}, got {dt} {shape:?}"
            ));
        }
        Ok(b)
    };
    let (wdt, sdt) = match scheme {
        Scheme::Nvfp4 => ("U8", "F8_E4M3"),
        Scheme::Mxfp8 => ("F8_E4M3", "U8"),
    };
    let w13 = check(&format!("{p}.w13_weight"), wdt, [experts, 2 * ffn, w13_rb])?;
    let s13 = check(
        &format!("{p}.w13_weight_scale"),
        sdt,
        [experts, 2 * ffn, s13_rb],
    )?;
    let w2 = check(&format!("{p}.w2_weight"), wdt, [experts, hidden, w2_rb])?;
    let s2 = check(
        &format!("{p}.w2_weight_scale"),
        sdt,
        [experts, hidden, s2_rb],
    )?;
    let outers: Option<(Vec<f32>, Vec<f32>)> = match scheme {
        Scheme::Nvfp4 => {
            let outer = |n: &str| -> Result<Vec<f32>, Inkling2RepackError> {
                let amax = f32_scalars(src, &format!("{p}.{n}_weight_amax"), experts)?;
                let den =
                    f32_scalars(src, &format!("{p}.{n}_weight_outer_scale_denominator"), 1)?[0];
                if !(den.is_finite() && den > 0.0) {
                    return err(format!(
                        "{p}.{n}: denominator {den} must be finite and positive"
                    ));
                }
                amax.iter()
                    .enumerate()
                    .map(|(e, &a)| {
                        if a.is_finite() && a >= 0.0 {
                            // The reference's order: np.float32(a) / np.float32(d).
                            Ok(a / den)
                        } else {
                            err(format!("{p}.{n}: amax of expert {e} is {a}"))
                        }
                    })
                    .collect()
            };
            Some((outer("w13")?, outer("w2")?))
        }
        Scheme::Mxfp8 => {
            for n in ["w13", "w2"] {
                for suffix in ["_weight_amax", "_weight_outer_scale_denominator"] {
                    if src.contains_tensor(&format!("{p}.{n}{suffix}")) {
                        return err(format!("{p}.{n}: MXFP8 must not carry {suffix}"));
                    }
                }
            }
            None
        }
    };

    let transcode = |w: &[u8], s: &[u8]| -> Result<Vec<u8>, Inkling2RepackError> {
        match scheme {
            Scheme::Nvfp4 => nvfp4_row_to_blocks(w, s),
            Scheme::Mxfp8 => mxfp8_row_to_blocks(w, s),
        }
        .map_err(|m| Inkling2RepackError(format!("layer {layer}: {m}")))
    };
    let one = |e: usize| -> Result<ExpertBlob, Inkling2RepackError> {
        let (o13, o2) = outers.as_ref().map_or((1.0, 1.0), |(a, b)| (a[e], b[e]));
        let mut record = Vec::with_capacity(INKLING2_OUTER_RECORD_BYTES);
        for v in [o13, o13, o2, 0.0f32] {
            record.extend_from_slice(&v.to_le_bytes());
        }
        let rows = |w: &[u8],
                    s: &[u8],
                    n_rows: usize,
                    rb: usize,
                    sb: usize,
                    pick: &dyn Fn(usize) -> usize| {
            let mut out = Vec::with_capacity(n_rows * scheme.row_bytes(rb));
            for k in 0..n_rows {
                let r = pick(k);
                out.extend(transcode(
                    &w[r * rb..(r + 1) * rb],
                    &s[r * sb..(r + 1) * sb],
                )?);
            }
            Ok::<_, Inkling2RepackError>(out)
        };
        let w13e = &w13[e * 2 * ffn * w13_rb..(e + 1) * 2 * ffn * w13_rb];
        let s13e = &s13[e * 2 * ffn * s13_rb..(e + 1) * 2 * ffn * s13_rb];
        let w2e = &w2[e * hidden * w2_rb..(e + 1) * hidden * w2_rb];
        let s2e = &s2[e * hidden * s2_rb..(e + 1) * hidden * s2_rb];
        // w13 interleaves rows: even rows are gate, odd rows are up.
        let gate = rows(w13e, s13e, ffn, w13_rb, s13_rb, &|k| 2 * k)?;
        let up = rows(w13e, s13e, ffn, w13_rb, s13_rb, &|k| 2 * k + 1)?;
        let down = rows(w2e, s2e, hidden, w2_rb, s2_rb, &|k| k)?;
        let sub = |role: &str, bytes: Vec<u8>, dtype: &str, shape: Vec<u64>| SubTensor {
            role: role.into(),
            bytes,
            dtype: dtype.into(),
            shape,
        };
        Ok(ExpertBlob {
            expert: e,
            sub_tensors: vec![
                sub("outer", record, "F32", vec![4]),
                sub(
                    "gate",
                    gate,
                    scheme.dtype(),
                    vec![ffn as u64, hidden as u64],
                ),
                sub("up", up, scheme.dtype(), vec![ffn as u64, hidden as u64]),
                sub(
                    "down",
                    down,
                    scheme.dtype(),
                    vec![hidden as u64, ffn as u64],
                ),
            ],
        })
    };

    // Experts are independent; split them across threads.
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = experts.div_ceil(threads).max(1);
    let mut blobs: Vec<Result<ExpertBlob, Inkling2RepackError>> = Vec::with_capacity(experts);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..experts)
            .step_by(chunk)
            .map(|start| {
                let one = &one;
                s.spawn(move || {
                    (start..(start + chunk).min(experts))
                        .map(one)
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            blobs.extend(h.join().expect("expert transcode thread panicked"));
        }
    });
    Ok(LayerBlobs {
        layer,
        experts: blobs.into_iter().collect::<Result<_, _>>()?,
    })
}

/// Pads a shape to the resident index's rank 4.
fn rank4(shape: &[usize]) -> Result<(u32, u32, u32, u32), Inkling2RepackError> {
    if shape.len() > 4 {
        return err(format!("rank {} exceeds 4", shape.len()));
    }
    let mut d = [0u32; 4];
    for (i, &v) in shape.iter().enumerate() {
        d[i] = u32::try_from(v).map_err(|_| Inkling2RepackError(format!("dim {v} too large")))?;
    }
    Ok((d[0], d[1], d[2], d[3]))
}

fn is_routed_expert(name: &str) -> bool {
    name.contains(".mlp.experts.")
}

/// The resident entries: every `model.llm.*` tensor that is not a routed
/// expert, with the router widened and `static_w` rewritten.
fn resident_entries(
    src: &dyn Inkling2Source,
    taps: usize,
) -> Result<Vec<ResidentEntrySpec>, Inkling2RepackError> {
    let mut names: Vec<String> = src
        .tensor_names()
        .into_iter()
        .filter(|n| n.starts_with("model.llm.") && !is_routed_expert(n))
        .collect();
    names.sort_unstable();
    let mut out = Vec::with_capacity(names.len());
    for name in &names {
        let name = name.as_str();
        if let Some(a) = src.affine(name) {
            out.push(affine_entry(name, &a)?);
            continue;
        }
        let (dt, shape, bytes) = tensor(src, name)?;
        let (dt, shape) = (dt.as_str(), shape.as_slice());
        let raw = |name: String, dtype: u8, bytes: Vec<u8>, shape: &[usize]| {
            Ok::<_, Inkling2RepackError>(ResidentEntrySpec::Raw(RawTensorSpec {
                name,
                dtype,
                bytes,
                shape: rank4(shape)?,
            }))
        };
        if name.ends_with(".mlp.gate.weight") {
            if dt != "BF16" {
                return err(format!("{name}: router expected BF16, got {dt}"));
            }
            let widened: Vec<u8> = bytes
                .chunks_exact(2)
                .flat_map(|b| {
                    f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16).to_le_bytes()
                })
                .collect();
            out.push(raw(name.to_string(), DTYPE_FP32, widened, shape)?);
            continue;
        }
        if let Some(stem) = name.strip_suffix("_sconv.static_w") {
            if dt != "BF16" || shape.len() != 2 || shape[0] != taps {
                return err(format!(
                    "{name}: expected BF16 [{taps}, ch], got {dt} {shape:?}"
                ));
            }
            let ch = shape[1];
            let rewritten = rewrite_static_w(&bytes, taps, ch);
            out.push(raw(
                format!("{stem}_sconv.weight"),
                DTYPE_BF16,
                rewritten,
                &[ch, 1, taps],
            )?);
            continue;
        }
        let dtype = match dt {
            "BF16" => DTYPE_BF16,
            "F32" => DTYPE_FP32,
            other => return err(format!("{name}: unexpected resident dtype {other}")),
        };
        out.push(raw(name.to_string(), dtype, bytes.to_vec(), shape)?);
    }
    Ok(out)
}

/// A full-attention `static_w` (BF16 `[taps, ch]`, indexed by lag with lag 0
/// the current row) as the conv kernel's `w[ch, j]` with tap `taps - 1` the
/// current row: the converter's `flip(0).T`. Callers check dtype and shape.
pub(crate) fn rewrite_static_w(bytes: &[u8], taps: usize, ch: usize) -> Vec<u8> {
    let mut rewritten = vec![0u8; bytes.len()];
    for c in 0..ch {
        for j in 0..taps {
            let s = ((taps - 1 - j) * ch + c) * 2;
            let d = (c * taps + j) * 2;
            rewritten[d..d + 2].copy_from_slice(&bytes[s..s + 2]);
        }
    }
    rewritten
}

/// An affine resident entry, checked against the reader's contract (logical
/// columns a multiple of 64, BF16 companions of `rows * cols / 64`).
fn affine_entry(
    name: &str,
    a: &AffineTensor<'_>,
) -> Result<ResidentEntrySpec, Inkling2RepackError> {
    let groups = a.rows * a.cols / 64;
    if a.cols % 64 != 0
        || a.packed.len() != a.rows * a.cols * a.bits as usize / 8
        || a.scales.len() != groups * 2
        || a.biases.len() != groups * 2
    {
        return err(format!(
            "{name}: affine {}-bit [{}, {}] does not match its packed and companion sizes",
            a.bits, a.rows, a.cols
        ));
    }
    let u16s = |b: &[u8]| -> Vec<u16> {
        b.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    };
    let spec = crate::resident_writer::ResidentTensorSpec {
        name: name.to_string(),
        packed: a.packed.to_vec(),
        scales: u16s(a.scales),
        biases: u16s(a.biases),
        rows: a.rows as u32,
        cols: a.cols as u32,
    };
    match a.bits {
        4 => Ok(ResidentEntrySpec::Int4(spec)),
        8 => Ok(ResidentEntrySpec::Int8(spec)),
        b => err(format!("{name}: affine {b}-bit is not supported")),
    }
}

/// The `manifest.quant` section: raw BF16 embedding and attention, the raw
/// F32 router, no shared expert (mirrors attention, i.e. defaulted), and the
/// Inkling routed scheme.
pub fn inkling2_manifest_quant() -> Value {
    let raw16 = serde_json::json!({
        "weightBits": 16, "scheme": "raw", "scaleType": "none", "biasType": "none", "groupSize": 0
    });
    serde_json::json!({
        "embedding": raw16,
        "attention": raw16,
        "router": {"weightBits": 32, "scheme": "raw", "scaleType": "none", "biasType": "none", "groupSize": 0},
        "sharedExpert": raw16,
        "routedExpert": {
            "weightBits": 0, "scheme": INKLING2_ROUTED_SCHEME, "scaleType": "none",
            "biasType": "none", "groupSize": 0
        },
    })
}

/// What an intake wrote.
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2InstallReport {
    pub layers: usize,
    pub nvfp4_layers: usize,
    pub mxfp8_layers: usize,
    pub resident_tensors: usize,
    pub expert_stride: u64,
}

/// Writes an install from a checkpoint already on disk. `progress` hears one
/// line per layer. Peak memory is the resident region plus one layer's blobs.
pub fn write_inkling2_install(
    src: &SafetensorsFile,
    config_json: &str,
    out: &Path,
    model_id: &str,
    progress: impl FnMut(&str),
) -> Result<Inkling2InstallReport, Inkling2RepackError> {
    write_inkling2_install_from(
        src,
        config_json,
        out,
        model_id,
        inkling2_manifest_quant(),
        progress,
    )
}

/// [`write_inkling2_install`] from any source, with the manifest's quant
/// section given (an MLX conversion declares its affine attention and
/// embedding there).
pub fn write_inkling2_install_from(
    src: &dyn Inkling2Source,
    config_json: &str,
    out: &Path,
    model_id: &str,
    quant: Value,
    mut progress: impl FnMut(&str),
) -> Result<Inkling2InstallReport, Inkling2RepackError> {
    let (arch, side) = parse_inkling2_config(config_json)?;
    let layers = arch.num_layers as usize;
    let (hidden, ffn, experts) = (
        arch.hidden_size as usize,
        arch.moe_intermediate_size as usize,
        arch.num_experts as usize,
    );
    let schemes: Vec<Scheme> = (0..layers)
        .map(|l| layer_scheme(src, l))
        .collect::<Result<_, _>>()?;
    let page = crate::GTURBO_PAGE_BYTES;
    let stride = schemes
        .iter()
        .map(|&s| expert_blob_bytes(s, hidden, ffn) as u64)
        .max()
        .unwrap_or(0)
        .div_ceil(page)
        * page;

    let resident = resident_entries(src, arch.short_conv_kernel_size as usize)?;
    let io = |e: crate::gturbo_writer::WriterError| Inkling2RepackError(e.to_string());
    let mut writer = StreamingGturboWriter::new(out, stride, experts).map_err(io)?;
    writer.set_quant(quant);
    for (l, scheme) in schemes.iter().enumerate() {
        let blobs = build_layer(src, l, experts, hidden, ffn)?;
        writer.write_layer(&blobs).map_err(io)?;
        progress(&format!(
            "layer {l:2}: {experts} {scheme:?} experts written"
        ));
    }
    let side_path = out.join(INKLING2_CONFIG_FILE);
    std::fs::write(
        &side_path,
        serde_json::to_vec_pretty(&side).expect("sidecar serializes"),
    )
    .map_err(|e| Inkling2RepackError(format!("{}: {e}", side_path.display())))?;
    let n_resident = resident.len();
    writer
        .finish_streaming(&arch, model_id, |w| {
            write_resident_weights_bin_mixed(&resident, w)
        })
        .map_err(io)?;
    Ok(Inkling2InstallReport {
        layers,
        nvfp4_layers: schemes.iter().filter(|&&s| s == Scheme::Nvfp4).count(),
        mxfp8_layers: schemes.iter().filter(|&&s| s == Scheme::Mxfp8).count(),
        resident_tensors: n_resident,
        expert_stride: stride,
    })
}

/// Copies the tokenizer and chat files beside the install.
pub fn copy_inkling2_tokenizer_files(
    src_dir: &Path,
    out: &Path,
) -> Result<(), Inkling2RepackError> {
    for name in INKLING2_TOKENIZER_FILES {
        let from = src_dir.join(name);
        if !from.exists() {
            if *name == "tokenizer.json" {
                return err(format!("{} is required", from.display()));
            }
            continue;
        }
        std::fs::copy(&from, out.join(name))
            .map_err(|e| Inkling2RepackError(format!("{}: {e}", from.display())))?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "inkling2_checkpoint_tests.rs"]
mod tests;
