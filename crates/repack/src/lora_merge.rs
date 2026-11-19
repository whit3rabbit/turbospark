//! PEFT LoRA merge at intake: present a base checkpoint with an adapter's
//! deltas already added, as the BF16 checkpoint `merge_and_unload` would save.
//!
//! **Why a source layer.** Like `quantize_on_install`, this rewrites the
//! SOURCE (a [`RangeSource`] that serves merged bytes under the base shard's
//! unchanged header) instead of adding a writer arm, so every writer, the MTP
//! path and the vision path stay untouched. Stack it BEFORE
//! `quantize_on_install_shards`: the quantizer then sees merged weights, which
//! is the order `mlx_lm.convert` applies to a merged checkpoint.
//!
//! The delta of a module is `scaling * (B @ A)` with `A [r, in]`, `B [out, r]`,
//! added to the base `[out, in]` weight in f32 and rounded back to the base
//! dtype. Anything this module cannot merge exactly (DoRA, rank or alpha
//! patterns, trained bias, transposed weights, extra saved modules) is
//! REFUSED by name: a silent partial merge reads as a fluent, wrong model.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::ranged_download::{DownloadError, RangeSource};
use crate::safetensors_header::SafetensorsHeader;

/// PEFT prefix on every adapter tensor name.
const PEFT_PREFIX: &str = "base_model.model.";
const A_SUFFIX: &str = ".lora_A.weight";
const B_SUFFIX: &str = ".lora_B.weight";

/// The fields of `adapter_config.json` the merge depends on.
#[derive(Debug, Clone, PartialEq)]
pub struct AdapterConfig {
    pub base_model: Option<String>,
    pub rank: usize,
    pub alpha: f32,
    pub use_rslora: bool,
}

impl AdapterConfig {
    /// PEFT's scaling: `alpha / r`, or `alpha / sqrt(r)` for rsLoRA.
    pub fn scaling(&self) -> f32 {
        if self.use_rslora {
            self.alpha / (self.rank as f32).sqrt()
        } else {
            self.alpha / self.rank as f32
        }
    }
}

/// Parse `adapter_config.json`, refusing every feature the merge cannot
/// reproduce.
pub fn parse_adapter_config(text: &str) -> Result<AdapterConfig, String> {
    let v: Value =
        serde_json::from_str(text).map_err(|e| format!("parsing adapter_config.json: {e}"))?;
    let flag = |k: &str| v.get(k).and_then(Value::as_bool).unwrap_or(false);
    let non_empty = |k: &str| match v.get(k) {
        None | Some(Value::Null) => false,
        Some(Value::Object(m)) => !m.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(_) => true,
    };
    if let Some(t) = v.get("peft_type").and_then(Value::as_str) {
        if t != "LORA" {
            return Err(format!("adapter peft_type {t:?} is not LORA"));
        }
    }
    for (key, why) in [
        ("use_dora", flag("use_dora")),
        ("lora_bias", flag("lora_bias")),
        ("fan_in_fan_out", flag("fan_in_fan_out")),
        ("rank_pattern", non_empty("rank_pattern")),
        ("alpha_pattern", non_empty("alpha_pattern")),
        ("modules_to_save", non_empty("modules_to_save")),
    ] {
        if why {
            return Err(format!(
                "adapter_config.json sets {key}, which this merge does not implement"
            ));
        }
    }
    let rank = v
        .get("r")
        .and_then(Value::as_u64)
        .filter(|r| *r > 0)
        .ok_or("adapter_config.json has no positive integer r")? as usize;
    let alpha = v
        .get("lora_alpha")
        .and_then(Value::as_f64)
        .ok_or("adapter_config.json has no lora_alpha")? as f32;
    Ok(AdapterConfig {
        base_model: v
            .get("base_model_name_or_path")
            .and_then(Value::as_str)
            .map(str::to_string),
        rank,
        alpha,
        use_rslora: flag("use_rslora"),
    })
}

/// One module's factors, `delta = scaling * (B @ A)`.
#[derive(Debug, Clone, PartialEq)]
struct Module {
    out: usize,
    inp: usize,
    rank: usize,
    /// `[rank, inp]` row-major.
    a: Vec<f32>,
    /// `[out, rank]` row-major.
    b: Vec<f32>,
}

/// A loaded adapter, keyed by the BASE tensor name it merges into
/// (`...q_proj.weight`).
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    scaling: f32,
    modules: BTreeMap<String, Module>,
}

fn decode(dtype: &str, raw: &[u8]) -> Result<Vec<f32>, String> {
    match dtype {
        "F32" => Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        "BF16" => Ok(raw
            .chunks_exact(2)
            .map(|c| compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        "F16" => Ok(raw
            .chunks_exact(2)
            .map(|c| compute::quant_1bit::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        other => Err(format!("unsupported tensor dtype {other}")),
    }
}

fn encode(dtype: &str, vals: &[f32]) -> Vec<u8> {
    match dtype {
        "F32" => vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
        "BF16" => vals
            .iter()
            .flat_map(|v| compute::f32_to_bf16(*v).to_le_bytes())
            .collect(),
        "F16" => vals
            .iter()
            .flat_map(|v| compute::quant_1bit::f32_to_f16(*v).to_le_bytes())
            .collect(),
        _ => unreachable!("dtype checked by decode"),
    }
}

impl LoraAdapter {
    /// Load the factors from the adapter's safetensors file. `header` is its
    /// parsed header and `src` serves its bytes.
    pub fn load(
        config: &AdapterConfig,
        header: &SafetensorsHeader,
        src: &dyn RangeSource,
    ) -> Result<Self, String> {
        let mut modules = BTreeMap::new();
        let mut seen_b = 0usize;
        for name in header.tensors.keys() {
            let Some(stem) = name
                .strip_prefix(PEFT_PREFIX)
                .and_then(|n| n.strip_suffix(A_SUFFIX))
            else {
                if name.ends_with(B_SUFFIX) {
                    seen_b += 1;
                    continue;
                }
                return Err(format!(
                    "{name}: not a lora_A/lora_B weight (bias, DoRA magnitude or an extra \
                     module); this merge refuses rather than drop it"
                ));
            };
            let b_name = format!("{PEFT_PREFIX}{stem}{B_SUFFIX}");
            let (a_info, b_info) = (
                &header.tensors[name],
                header
                    .tensors
                    .get(&b_name)
                    .ok_or(format!("{name} has no matching {b_name}"))?,
            );
            let (a_shape, b_shape) = (&a_info.shape, &b_info.shape);
            if a_shape.len() != 2 || b_shape.len() != 2 || a_shape[0] != b_shape[1] {
                return Err(format!(
                    "{stem}: factor shapes {a_shape:?} x {b_shape:?} do not compose"
                ));
            }
            if a_shape[0] as usize != config.rank {
                return Err(format!(
                    "{stem}: factor rank {} differs from adapter_config r {}",
                    a_shape[0], config.rank
                ));
            }
            let read = |n: &str, dtype: &str| -> Result<Vec<f32>, String> {
                let (s, e) = header.absolute_range(n).ok_or(format!("{n}: bad range"))?;
                let raw = src.read_range(s, e).map_err(|e| format!("{n}: {e:?}"))?;
                decode(dtype, &raw).map_err(|e| format!("{n}: {e}"))
            };
            modules.insert(
                format!("{stem}.weight"),
                Module {
                    out: b_shape[0] as usize,
                    inp: a_shape[1] as usize,
                    rank: config.rank,
                    a: read(name, &a_info.dtype)?,
                    b: read(&b_name, &b_info.dtype)?,
                },
            );
        }
        if modules.len() != seen_b {
            return Err(format!(
                "{} lora_A tensors but {seen_b} lora_B tensors",
                modules.len()
            ));
        }
        if modules.is_empty() {
            return Err("the adapter carries no lora_A/lora_B tensors".to_string());
        }
        Ok(Self {
            scaling: config.scaling(),
            modules,
        })
    }

    /// Base tensor names this adapter merges into.
    pub fn targets(&self) -> impl Iterator<Item = &str> {
        self.modules.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.modules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Check every module against the base shards' headers. A module with no
    /// base tensor, or with a different shape, is an error: the install would
    /// otherwise be silently under-merged.
    pub fn check_against(&self, headers: &[SafetensorsHeader]) -> Result<(), String> {
        for (name, m) in &self.modules {
            let info = headers
                .iter()
                .find_map(|h| h.tensors.get(name))
                .ok_or_else(|| format!("adapter targets {name}, which the base does not have"))?;
            if info.shape != [m.out as u64, m.inp as u64] {
                return Err(format!(
                    "{name}: base shape {:?} differs from the adapter's [{}, {}]",
                    info.shape, m.out, m.inp
                ));
            }
            if !matches!(info.dtype.as_str(), "BF16" | "F16" | "F32") {
                return Err(format!(
                    "{name}: base dtype {} cannot be merged",
                    info.dtype
                ));
            }
        }
        Ok(())
    }

    /// `base + scaling * (B @ A)` rounded to `dtype`.
    fn merge(&self, name: &str, dtype: &str, raw: &[u8]) -> Result<Vec<u8>, String> {
        let m = &self.modules[name];
        let mut w = decode(dtype, raw)?;
        if w.len() != m.out * m.inp {
            return Err(format!(
                "{name}: {} elements, expected {}",
                w.len(),
                m.out * m.inp
            ));
        }
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(m.out.max(1));
        let per = m.out.div_ceil(threads);
        let scale = self.scaling;
        std::thread::scope(|s| {
            for (ci, rows) in w.chunks_mut(per * m.inp).enumerate() {
                s.spawn(move || {
                    let mut acc = vec![0f32; m.inp];
                    for (ri, row) in rows.chunks_exact_mut(m.inp).enumerate() {
                        let o = ci * per + ri;
                        acc.iter_mut().for_each(|x| *x = 0.0);
                        for r in 0..m.rank {
                            let b = m.b[o * m.rank + r];
                            let a = &m.a[r * m.inp..(r + 1) * m.inp];
                            for (x, av) in acc.iter_mut().zip(a) {
                                *x += b * av;
                            }
                        }
                        for (x, d) in row.iter_mut().zip(&acc) {
                            *x += scale * d;
                        }
                    }
                });
            }
        });
        Ok(encode(dtype, &w))
    }
}

/// Where one shard's merged tensors live, as absolute byte ranges. Computed
/// from the shard's header BEFORE any tensor renaming (the adapter speaks the
/// HF spelling), then handed to [`LoraMergedSource::new`] later: byte ranges
/// do not change when the header is canonicalized.
#[derive(Debug, Clone, Default)]
pub struct MergeTargets(Vec<(u64, u64, String, String)>);

impl MergeTargets {
    /// Tensors of this shard that get a delta.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl LoraAdapter {
    /// The tensors of `header` this adapter merges into.
    pub fn locate(&self, header: &SafetensorsHeader) -> Result<MergeTargets, String> {
        let mut targets = Vec::new();
        for name in self.targets() {
            if let Some(info) = header.tensors.get(name) {
                let (s, e) = header
                    .absolute_range(name)
                    .ok_or_else(|| format!("{name}: range overflows"))?;
                targets.push((s, e, name.to_string(), info.dtype.clone()));
            }
        }
        targets.sort();
        Ok(MergeTargets(targets))
    }
}

/// One base shard with the adapter merged in. The header is the base's own.
pub struct LoraMergedSource<'a> {
    inner: &'a dyn RangeSource,
    adapter: &'a LoraAdapter,
    /// `(absolute start, absolute end, name, dtype)`, sorted by start.
    targets: Vec<(u64, u64, String, String)>,
}

impl<'a> LoraMergedSource<'a> {
    pub fn new(
        inner: &'a dyn RangeSource,
        adapter: &'a LoraAdapter,
        targets: MergeTargets,
    ) -> Self {
        Self {
            inner,
            adapter,
            targets: targets.0,
        }
    }

    /// Tensors of this shard that get a delta.
    pub fn merged_count(&self) -> usize {
        self.targets.len()
    }
}

impl RangeSource for LoraMergedSource<'_> {
    fn read_range(&self, start: u64, end_exclusive: u64) -> Result<Vec<u8>, DownloadError> {
        let mut out = self.inner.read_range(start, end_exclusive)?;
        // Targets are disjoint and sorted, so the overlapping ones are a
        // contiguous run.
        let first = self.targets.partition_point(|t| t.1 <= start);
        for (ts, te, name, dtype) in self.targets[first..]
            .iter()
            .take_while(|t| t.0 < end_exclusive)
        {
            // Whole tensor in hand? Merge from it; otherwise fetch it.
            let whole = if start <= *ts && *te <= end_exclusive {
                out[(*ts - start) as usize..(*te - start) as usize].to_vec()
            } else {
                self.inner.read_range(*ts, *te)?
            };
            let merged = self
                .adapter
                .merge(name, dtype, &whole)
                .map_err(DownloadError::Request)?;
            let (lo, hi) = (start.max(*ts), end_exclusive.min(*te));
            out[(lo - start) as usize..(hi - start) as usize]
                .copy_from_slice(&merged[(lo - ts) as usize..(hi - ts) as usize]);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ranged_download::MemoryRangeSource;
    use crate::safetensors_header::TensorInfo;

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn bf16_bytes(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|&x| compute::f32_to_bf16(x).to_le_bytes())
            .collect()
    }

    /// Empty-header blob (data region at byte 8), as quantize_on_install's tests do.
    fn blob(tensors: &[(&str, &str, Vec<u64>, Vec<u8>)]) -> (SafetensorsHeader, Vec<u8>) {
        let mut map = BTreeMap::new();
        let mut data = Vec::new();
        for (n, d, s, b) in tensors {
            let start = data.len() as u64;
            data.extend_from_slice(b);
            map.insert(
                n.to_string(),
                TensorInfo {
                    dtype: d.to_string(),
                    shape: s.clone(),
                    data_offsets: (start, data.len() as u64),
                },
            );
        }
        let mut file = vec![0u8; 8];
        file.extend(data);
        (
            SafetensorsHeader {
                tensors: map,
                metadata: None,
                header_len: 0,
            },
            file,
        )
    }

    const CFG: &str = r#"{"peft_type":"LORA","r":2,"lora_alpha":4,"use_rslora":false,
        "rank_pattern":{},"alpha_pattern":{},"modules_to_save":null,
        "base_model_name_or_path":"Qwen/Qwen3.8-27B"}"#;

    fn adapter() -> (LoraAdapter, AdapterConfig) {
        let cfg = parse_adapter_config(CFG).unwrap();
        // out=3, in=4, r=2
        let a: Vec<f32> = (0..8).map(|i| i as f32 * 0.5 - 1.0).collect();
        let b: Vec<f32> = (0..6).map(|i| (i as f32 - 2.0) * 0.25).collect();
        let (h, file) = blob(&[
            (
                "base_model.model.model.layers.0.q_proj.lora_A.weight",
                "F32",
                vec![2, 4],
                f32_bytes(&a),
            ),
            (
                "base_model.model.model.layers.0.q_proj.lora_B.weight",
                "F32",
                vec![3, 2],
                f32_bytes(&b),
            ),
        ]);
        let ad = LoraAdapter::load(&cfg, &h, &MemoryRangeSource::new(&file)).unwrap();
        (ad, cfg)
    }

    #[test]
    fn scaling_is_alpha_over_rank_and_rslora_over_sqrt() {
        let mut c = parse_adapter_config(CFG).unwrap();
        assert_eq!(c.scaling(), 2.0);
        c.use_rslora = true;
        assert!((c.scaling() - 4.0 / 2f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn unmergeable_features_are_refused_by_name() {
        for (k, v) in [
            ("use_dora", "true"),
            ("lora_bias", "true"),
            ("fan_in_fan_out", "true"),
            ("rank_pattern", r#"{"q_proj":4}"#),
            ("alpha_pattern", r#"{"q_proj":4}"#),
            ("modules_to_save", r#"["lm_head"]"#),
        ] {
            let t = format!(r#"{{"r":2,"lora_alpha":4,"{k}":{v}}}"#);
            let e = parse_adapter_config(&t).unwrap_err();
            assert!(e.contains(k), "{e}");
        }
    }

    #[test]
    fn merge_matches_a_naive_reference() {
        let (ad, _) = adapter();
        let base: Vec<f32> = (0..12).map(|i| i as f32 * 0.125).collect();
        let (h, file) = blob(&[(
            "model.layers.0.q_proj.weight",
            "BF16",
            vec![3, 4],
            bf16_bytes(&base),
        )]);
        ad.check_against(std::slice::from_ref(&h)).unwrap();
        let src = MemoryRangeSource::new(&file);
        let merged = LoraMergedSource::new(&src, &ad, ad.locate(&h).unwrap());
        assert_eq!(merged.merged_count(), 1);
        let (s, e) = h.absolute_range("model.layers.0.q_proj.weight").unwrap();
        let got = decode("BF16", &merged.read_range(s, e).unwrap()).unwrap();
        let m = &ad.modules["model.layers.0.q_proj.weight"];
        for o in 0..3 {
            for i in 0..4 {
                let mut d = 0.0;
                for r in 0..2 {
                    d += m.b[o * 2 + r] * m.a[r * 4 + i];
                }
                let want = compute::bf16_to_f32(compute::f32_to_bf16(
                    compute::bf16_to_f32(compute::f32_to_bf16(base[o * 4 + i])) + 2.0 * d,
                ));
                assert_eq!(got[o * 4 + i], want, "[{o},{i}]");
            }
        }
        // The delta is non-zero, so a merge that did nothing would fail above.
        assert_ne!(got, decode("BF16", &bf16_bytes(&base)).unwrap());
    }

    #[test]
    fn partial_and_spanning_ranges_see_the_same_merged_bytes() {
        let (ad, _) = adapter();
        let other = bf16_bytes(&[1.0, 2.0]);
        let base = bf16_bytes(&(0..12).map(|i| i as f32).collect::<Vec<_>>());
        let (h, file) = blob(&[
            ("model.layers.0.q_proj.weight", "BF16", vec![3, 4], base),
            ("model.norm.weight", "BF16", vec![2], other),
        ]);
        let src = MemoryRangeSource::new(&file);
        let merged = LoraMergedSource::new(&src, &ad, ad.locate(&h).unwrap());
        let all = merged.read_range(8, file.len() as u64).unwrap();
        // Untouched tensor stays verbatim.
        assert_eq!(&all[24..], &file[8 + 24..]);
        // Every sub-range agrees with the full read.
        for (s, e) in [(0usize, 5usize), (5, 20), (3, 26), (24, 28), (0, 28)] {
            let got = merged.read_range(8 + s as u64, 8 + e as u64).unwrap();
            assert_eq!(got, all[s..e], "range {s}..{e}");
        }
    }

    #[test]
    fn missing_or_mismatched_targets_are_refused() {
        let (ad, _) = adapter();
        let (h, _) = blob(&[(
            "model.norm.weight",
            "BF16",
            vec![2],
            bf16_bytes(&[1.0, 2.0]),
        )]);
        assert!(ad
            .check_against(std::slice::from_ref(&h))
            .unwrap_err()
            .contains("does not have"));
        let (h, _) = blob(&[(
            "model.layers.0.q_proj.weight",
            "BF16",
            vec![4, 3],
            bf16_bytes(&[0.0; 12]),
        )]);
        assert!(ad
            .check_against(std::slice::from_ref(&h))
            .unwrap_err()
            .contains("differs"));
    }

    #[test]
    fn a_stray_adapter_tensor_is_refused() {
        let cfg = parse_adapter_config(CFG).unwrap();
        let (h, file) = blob(&[(
            "base_model.model.model.layers.0.q_proj.lora_magnitude_vector",
            "F32",
            vec![3],
            f32_bytes(&[1.0; 3]),
        )]);
        let e = LoraAdapter::load(&cfg, &h, &MemoryRangeSource::new(&file)).unwrap_err();
        assert!(e.contains("not a lora_A/lora_B"), "{e}");
    }
}
