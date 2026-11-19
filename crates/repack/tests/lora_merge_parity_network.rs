//! Does merging the pi-adapted LoRA into the BF16 base reproduce what tozuo
//! published? Read off real bytes with ranged reads (two base tensors of
//! ~180 MB each, the 637 MB adapter, and a few rows of the MLX copy):
//!
//! ```sh
//! cargo test -p turbospark-repack --test lora_merge_parity_network --release -- --ignored --nocapture
//! ```
//!
//! `tozuo/Qwen3.8-27B-pi-adapted-MLX-4bit` is `harness-hub/Qwen3.8-27B-pi-adapted`
//! merged into `Qwen/Qwen3.8-27B` and quantized with `mx.quantize` (its card).
//! So `quantize(base + scale * B @ A)` over the first rows of a targeted tensor
//! must match its `.weight`/`.scales`/`.biases` rows. Equality is not
//! guaranteed (the merge may round to BF16 at a different point), so the test
//! reports the match fraction and bounds it, with the UNMERGED base as the
//! control: if merging did nothing useful the two would score alike.

use turbospark_repack::lora_merge::{parse_adapter_config, LoraAdapter, LoraMergedSource};
use turbospark_repack::{
    fetch_safetensors_header, HttpRangeSource, MemoryRangeSource, RangeSource,
};

const BASE: &str = "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main";
const ADAPTER: &str = "https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted/resolve/main";
const MLX: &str = "https://huggingface.co/tozuo/Qwen3.8-27B-pi-adapted-MLX-4bit/resolve/main";

const ROWS: usize = 64;

fn shard_of(index_url: &str, name: &str) -> String {
    let text = reqwest::blocking::get(index_url)
        .and_then(|r| r.text())
        .expect("index");
    let v: serde_json::Value = serde_json::from_str(&text).expect("index json");
    v["weight_map"][name]
        .as_str()
        .unwrap_or_else(|| panic!("{name} not in {index_url}"))
        .to_string()
}

fn bf16_rows(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|c| compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// Fraction of equal bytes between two buffers of the same length.
fn same(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / a.len() as f64
}

/// `(packed, scales, biases)` for `rows` rows of `cols` BF16 columns.
fn quantize(rows: &[f32], cols: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (mut w, mut s, mut b) = (Vec::new(), Vec::new(), Vec::new());
    for row in rows.chunks_exact(cols) {
        let q = compute::quant_mlx::quantize_int4_affine_mlx(row);
        w.extend_from_slice(&q.packed);
        s.extend(q.scales.iter().flat_map(|v| v.to_le_bytes()));
        b.extend(q.biases.iter().flat_map(|v| v.to_le_bytes()));
    }
    (w, s, b)
}

#[test]
#[ignore = "network: ~1.1 GB of ranged reads"]
fn merged_base_quantizes_to_the_published_mlx_rows() {
    let cfg = parse_adapter_config(
        &reqwest::blocking::get(format!("{ADAPTER}/adapter_config.json"))
            .and_then(|r| r.text())
            .expect("adapter config"),
    )
    .expect("adapter config parses");
    let asrc = HttpRangeSource::new(format!("{ADAPTER}/adapter_model.safetensors"));
    let ah = fetch_safetensors_header(&asrc).expect("adapter header");
    let end = ah
        .tensors
        .keys()
        .filter_map(|n| ah.absolute_range(n))
        .map(|(_, e)| e)
        .max()
        .unwrap();
    let bytes = asrc.read_range(0, end).expect("adapter bytes");
    let adapter = LoraAdapter::load(&cfg, &ah, &MemoryRangeSource::new(&bytes)).expect("load");
    assert_eq!(adapter.len(), 256, "r=32 on q/k/v/o (16 layers) + mlp (64)");

    let (mut merged_total, mut control_total) = (0.0f64, 0.0f64);
    for stem in [
        "layers.0.mlp.down_proj",
        "layers.3.self_attn.o_proj",
        "layers.0.mlp.gate_proj",
    ] {
        let hf = format!("model.language_model.{stem}.weight");
        let mlx = format!("language_model.model.{stem}");

        let base_shard = shard_of(&format!("{BASE}/model.safetensors.index.json"), &hf);
        let base = HttpRangeSource::new(format!("{BASE}/{base_shard}"));
        let bh = fetch_safetensors_header(&base).expect("base header");
        let cols = bh.tensors[&hf].shape[1] as usize;
        let (ts, _) = bh.absolute_range(&hf).unwrap();
        // The merge reads the whole tensor through the source; the rows are
        // then cut from its output.
        let merged_src = LoraMergedSource::new(&base, &adapter, adapter.locate(&bh).unwrap());
        let row_bytes = (cols * 2) as u64;
        let merged_raw = merged_src
            .read_range(ts, ts + ROWS as u64 * row_bytes)
            .expect("merged rows");
        let plain_raw = base
            .read_range(ts, ts + ROWS as u64 * row_bytes)
            .expect("base rows");
        assert_ne!(merged_raw, plain_raw, "{stem}: the merge changed nothing");

        let mlx_shard = shard_of(
            &format!("{MLX}/model.safetensors.index.json"),
            &format!("{mlx}.weight"),
        );
        let msrc = HttpRangeSource::new(format!("{MLX}/{mlx_shard}"));
        let mh = fetch_safetensors_header(&msrc).expect("mlx header");
        let rows_of = |suffix: &str, per_row: u64| {
            let (s, _) = mh.absolute_range(&format!("{mlx}.{suffix}")).unwrap();
            msrc.read_range(s, s + ROWS as u64 * per_row).unwrap()
        };
        let want_w = rows_of("weight", (cols / 2) as u64);
        let want_s = rows_of("scales", (cols / 64 * 2) as u64);
        let want_b = rows_of("biases", (cols / 64 * 2) as u64);

        let (mw, ms, mb) = quantize(&bf16_rows(&merged_raw), cols);
        let (pw, ps, pb) = quantize(&bf16_rows(&plain_raw), cols);
        let merged = (same(&mw, &want_w), same(&ms, &want_s), same(&mb, &want_b));
        let control = (same(&pw, &want_w), same(&ps, &want_s), same(&pb, &want_b));
        println!("{stem}: merged {merged:?} vs unmerged control {control:?}");
        // The delta is small (r=32 against a 27B trunk), so the unmerged
        // control already agrees on most bytes; the merge must be (near)
        // exact, never worse than the control, and better in aggregate.
        assert!(
            merged.0 >= 0.999,
            "{stem}: packed weight agreement {merged:?}"
        );
        assert!(merged.0 >= control.0, "{stem}: control {control:?}");
        merged_total += merged.0;
        control_total += control.0;
    }
    assert!(
        merged_total > control_total,
        "merged {merged_total} must beat the unmerged control {control_total} in aggregate"
    );
}
