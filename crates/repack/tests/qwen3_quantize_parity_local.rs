//! Byte parity of quantize-on-install for plain dense `qwen3`, against a
//! LOCAL copy of the published pair. Never downloads; skipped (ignored) in CI.
//!
//! ```sh
//! TS_QWEN3_BF16_DIR=~/models/qwen3-parity/Qwen3-4B \
//! TS_QWEN3_Q4_DIR=~/models/qwen3-parity/Qwen3-4B-4bit \
//!   cargo test -p turbospark-repack --test qwen3_quantize_parity_local --release \
//!   -- --ignored --nocapture
//! ```
//!
//! `mlx-community/Qwen3-4B-4bit` is `mlx_lm.convert` of `Qwen/Qwen3-4B`, so
//! the whole install view (not only the row quantizer, which
//! `mlx_quantize_parity_network` already pins) must reproduce it tensor for
//! tensor: the weight/scales/biases triples, and every tensor that stays
//! BF16 (norms, `q_norm`/`k_norm`). The view is built through the same
//! entry points `catalog::stream_mlx` uses (canonicalize, tied-head drop,
//! `quantize_on_install_shards_for`), so this tests the install path and not
//! a copy of it.
//!
//! **Tie rule.** The published 4-bit file was made by the CPU `mx.quantize`
//! kernel (round half to even), not the Metal one (half away from zero) that
//! installs use (`Rounding::HalfAway`, `crates/repack/AGENTS.md`). Against the
//! published bytes the view therefore differs only on exact ties: a few
//! hundredths of a percent of scales and about 0.4 percent of weight bytes.
//! The test pins that precisely: (1) names, dtypes, shapes, every bias plane
//! and every raw BF16 tensor (norms, `q_norm`, `k_norm`) are byte-identical
//! through the real view; (2) every weight and scale plane of the published
//! file equals `quantize_*_affine_mlx_with(.., HalfEven)` of the BF16 source,
//! with ZERO differing bytes; (3) the view's own difference from the
//! published file is counted and bounded.
//!
//! Revisions this was run against are recorded in `crates/repack/AGENTS.md`.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use model_io::ModelFamily;
use turbospark_repack::quantize_on_install::{
    canonicalize_for_family, drop_tied_lm_head, family_admits_quantize_on_install,
    quantize_on_install_shards_for, shifts_norms,
};
use turbospark_repack::{
    canonicalize_qwen2_header, parse_header, DownloadError, RangeSource, SafetensorsHeader,
    DEFAULT_MAX_HEADER_BYTES,
};

struct FileSource(File);

impl RangeSource for FileSource {
    fn read_range(&self, start: u64, end: u64) -> Result<Vec<u8>, DownloadError> {
        let mut buf = vec![0u8; (end - start) as usize];
        self.0
            .read_exact_at(&mut buf, start)
            .map_err(|e| DownloadError::Request(e.to_string()))?;
        Ok(buf)
    }
}

fn dir(var: &str) -> Option<PathBuf> {
    std::env::var(var).ok().map(PathBuf::from)
}

fn shards(d: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(d)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    v.sort();
    v
}

fn header_of(src: &FileSource) -> SafetensorsHeader {
    let lead = src.read_range(0, 8).unwrap();
    let n = u64::from_le_bytes(lead[..8].try_into().unwrap());
    let bytes = src.read_range(0, 8 + n).unwrap();
    parse_header(&bytes, DEFAULT_MAX_HEADER_BYTES).expect("header")
}

#[test]
#[ignore = "needs TS_QWEN3_BF16_DIR and TS_QWEN3_Q4_DIR (local copies of the published pair)"]
fn qwen3_4b_bf16_installs_to_the_published_4bit_bytes() {
    let (Some(bf_dir), Some(q_dir)) = (dir("TS_QWEN3_BF16_DIR"), dir("TS_QWEN3_Q4_DIR")) else {
        panic!("set TS_QWEN3_BF16_DIR and TS_QWEN3_Q4_DIR");
    };
    let family = ModelFamily::Qwen3Dense;
    let bf_srcs: Vec<FileSource> = shards(&bf_dir)
        .iter()
        .map(|p| FileSource(File::open(p).unwrap()))
        .collect();
    let q_srcs: Vec<FileSource> = shards(&q_dir)
        .iter()
        .map(|p| FileSource(File::open(p).unwrap()))
        .collect();

    // The same steps stream_mlx runs, in its order.
    let mut headers: Vec<_> = bf_srcs.iter().map(header_of).collect();
    let orig_headers = headers.clone();
    let layout = canonicalize_for_family(family, &mut headers).expect("canonicalize");
    for h in &mut headers {
        canonicalize_qwen2_header(h).expect("qwen2 header canonicalization");
    }
    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(bf_dir.join("config.json")).unwrap()).unwrap();
    if cfg["tie_word_embeddings"].as_bool() == Some(true) {
        drop_tied_lm_head(&mut headers);
    }
    let refs: Vec<&dyn RangeSource> = bf_srcs.iter().map(|s| s as &dyn RangeSource).collect();
    let view = quantize_on_install_shards_for(family, &headers, &refs, 4, layout).expect("view");
    assert!(!shifts_norms(family), "qwen3 norms are plain x*w, no +1");

    // Index the view by tensor name.
    let mut ours = std::collections::BTreeMap::new();
    for (i, s) in view.iter().enumerate() {
        for (name, info) in &s.header().tensors {
            ours.insert(name.clone(), (i, info.clone()));
        }
    }

    // Walk every published tensor. Published names are `model.*`; ours are
    // `language_model.model.*` after canonicalization.
    let mut classes: std::collections::BTreeMap<String, (usize, usize, u64)> = Default::default();
    let mut matched = std::collections::BTreeSet::new();
    let mut bad = Vec::new();
    let (mut tie_diff_bytes, mut tie_total_bytes, mut tie_total_all) = (0u64, 0u64, 0u64);
    let mut quantized_bases: Vec<String> = Vec::new();
    for q in &q_srcs {
        let qh = header_of(q);
        for (name, info) in &qh.tensors {
            let want = q
                .read_range(
                    qh.absolute_range(name).unwrap().0,
                    qh.absolute_range(name).unwrap().1,
                )
                .unwrap();
            let key = format!("language_model.{name}");
            let class = if name.ends_with(".scales") {
                "scales"
            } else if name.ends_with(".biases") {
                "biases"
            } else if info.dtype == "U32" {
                "quantized weight"
            } else {
                "raw (bf16) tensor"
            };
            let e = classes.entry(class.to_string()).or_default();
            e.0 += 1;
            e.2 += want.len() as u64;
            let Some((i, oi)) = ours.get(&key) else {
                bad.push(format!("{name}: missing from our view"));
                continue;
            };
            matched.insert(key.clone());
            if oi.dtype != info.dtype || oi.shape != info.shape {
                bad.push(format!(
                    "{name}: ours {} {:?} vs published {} {:?}",
                    oi.dtype, oi.shape, info.dtype, info.shape
                ));
                continue;
            }
            let (a, b) = view[*i].header().absolute_range(&key).unwrap();
            let got = view[*i].read_range(a, b).unwrap();
            if got != want {
                let diff = got.iter().zip(&want).filter(|(x, y)| x != y).count();
                if class == "quantized weight" || class == "scales" {
                    // Tie-rule differences only; step 2 proves the published
                    // bytes against the half-to-even oracle, step 3 bounds this.
                    tie_diff_bytes += diff as u64;
                    tie_total_bytes += want.len() as u64;
                    e.1 += 1;
                } else {
                    bad.push(format!(
                        "{name} [{class}]: {diff} of {} bytes differ",
                        want.len()
                    ));
                    e.1 += 1;
                }
            }
            if class == "quantized weight" || class == "scales" {
                tie_total_all += want.len() as u64;
            }
            if class == "quantized weight" {
                let base = name.strip_suffix(".weight").unwrap().to_string();
                quantized_bases.push(base);
            }
        }
    }
    for k in ours.keys() {
        if !matched.contains(k) {
            bad.push(format!("{k}: in our view, not in the published file"));
        }
    }
    for (class, (n, differing, bytes)) in &classes {
        println!(
            "{class:20} tensors {n:5}  view != published {differing:5} (weights and scales: tie rule only)  bytes {bytes}"
        );
    }
    println!(
        "view vs published (HalfAway installs): {tie_diff_bytes} differing bytes in \
         {tie_total_bytes} bytes of the differing tensors, of {tie_total_all} weight+scale bytes"
    );
    assert!(bad.is_empty(), "parity failures:\n{}", bad.join("\n"));

    // Step 2: the published weight and scale planes are the half-to-even
    // quantization of the BF16 source, byte for byte, for EVERY matrix.
    let q = &q_srcs[0];
    let qh = header_of(q);
    let mut even_diffs = 0usize;
    for base in &quantized_bases {
        let wname = format!("{base}.weight");
        let (si, info) = orig_headers
            .iter()
            .enumerate()
            .find_map(|(i, h)| h.tensors.get(&wname).map(|t| (i, t)))
            .unwrap_or_else(|| panic!("{wname} not in the bf16 source"));
        assert_eq!(info.dtype, "BF16", "{wname}");
        let (rows, cols) = (info.shape[0] as usize, info.shape[1] as usize);
        let (a, b) = orig_headers[si].absolute_range(&wname).unwrap();
        let raw = bf_srcs[si].read_range(a, b).unwrap();
        let rd = |n: &str| {
            let (x, y) = qh.absolute_range(n).unwrap();
            q.read_range(x, y).unwrap()
        };
        let (pw, ps, pb) = (
            rd(&wname),
            rd(&format!("{base}.scales")),
            rd(&format!("{base}.biases")),
        );
        let (rw, rg) = (cols / 2, cols / 64 * 2);
        for r in 0..rows {
            let row: Vec<f32> = raw[r * cols * 2..(r + 1) * cols * 2]
                .chunks_exact(2)
                .map(|c| compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let o = compute::quant_mlx::quantize_int4_affine_mlx_with(
                &row,
                compute::quant_mlx::Rounding::HalfEven,
            );
            let sb: Vec<u8> = o.scales.iter().flat_map(|v| v.to_le_bytes()).collect();
            let bb: Vec<u8> = o.biases.iter().flat_map(|v| v.to_le_bytes()).collect();
            even_diffs += (o.packed != pw[r * rw..(r + 1) * rw]) as usize
                + (sb != ps[r * rg..(r + 1) * rg]) as usize
                + (bb != pb[r * rg..(r + 1) * rg]) as usize;
        }
    }
    println!(
        "half-to-even oracle over {} matrices: {even_diffs} differing rows (weight, scales, biases)",
        quantized_bases.len()
    );
    assert_eq!(
        even_diffs, 0,
        "the published file is not the half-to-even quantization of the bf16 source"
    );
    // Step 3: the installs' own (Metal tie rule) difference stays a tie-sized
    // fraction of the weight and scale bytes.
    assert!(
        (tie_diff_bytes as f64) < 0.01 * tie_total_all as f64,
        "view differs from the published file by more than ties can explain"
    );
    assert!(
        family_admits_quantize_on_install(family),
        "parity holds, so the family must be admitted"
    );
}
