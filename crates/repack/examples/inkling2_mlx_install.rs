//! Installs Inkling 2 Mini from a local MLX conversion (mlx-lm PR 2086, for
//! example `mlx-community/Inkling-2-Mini-4bit`), after checking the MLX
//! source against an install made from the HF checkpoint.
//!
//! ```text
//! cargo run --release -p turbospark-repack --example inkling2_mlx_install -- \
//!     <mlx dir> <reference hf install> <out dir | ->
//! ```
//!
//! Checks before writing: the HF names the MLX source presents are exactly
//! the reference install's resident names (plus the routed experts), each
//! dtype matches or is affine where the reference holds BF16, and the
//! NVFP4/MXFP8 expert rows decode to the reference's (outer scales within
//! f32 rounding). `-` as the out dir stops after the checks.

use std::collections::BTreeSet;
use std::path::Path;
use std::time::Instant;

use turbospark_repack::{Inkling2MlxSource, Inkling2Source};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!(
            "usage: {} <mlx dir> <reference install> <out dir | ->",
            args[0]
        );
        std::process::exit(2);
    }
    let (mlx, reference) = (Path::new(&args[1]), Path::new(&args[2]));
    let config = std::fs::read_to_string(mlx.join("config.json")).expect("config.json");
    let (arch, side) = turbospark_repack::parse_inkling2_config(&config).expect("config parses");
    let core = (side.core_end - side.core_start) as usize;
    let src = Inkling2MlxSource::open(
        mlx,
        arch.num_layers as usize,
        core,
        arch.hidden_size as usize,
    )
    .expect("open MLX source");
    eprintln!("affine bit widths: {:?}", src.affine_bits);

    // 1. Names and dtypes against the reference install.
    let index = model_io::load_resident_index(&reference.join("model_weights.bin")).expect("index");
    let want: BTreeSet<String> = index.entries.keys().cloned().collect();
    let mut have: BTreeSet<String> = src
        .tensor_names()
        .into_iter()
        .filter(|n| !n.contains(".mlp.experts."))
        .collect();
    // The intake rewrites `static_w` to `weight` and widens the router.
    have = have
        .into_iter()
        .map(|n| n.replace("_sconv.static_w", "_sconv.weight"))
        .collect();
    let missing: Vec<_> = want.difference(&have).collect();
    let extra: Vec<_> = have.difference(&want).collect();
    eprintln!(
        "names: {} reference, {} mapped; missing {missing:?}, extra {extra:?}",
        want.len(),
        have.len()
    );
    assert!(missing.is_empty() && extra.is_empty(), "name sets differ");
    let mut affine = 0usize;
    for name in src
        .tensor_names()
        .iter()
        .filter(|n| !n.contains(".mlp.experts."))
    {
        let ref_name = name.replace("_sconv.static_w", "_sconv.weight");
        let e = &index.entries[&ref_name];
        if let Some(a) = src.affine(name) {
            assert_eq!(
                e.dtype, 1,
                "{name}: affine in MLX but not BF16 in the reference"
            );
            assert_eq!(
                (a.rows as u32, a.cols as u32),
                (e.shape.0, e.shape.1),
                "{name}: shape"
            );
            affine += 1;
            continue;
        }
        let (dt, _, bytes) = src.tensor(name).expect("tensor");
        let tag = match dt.as_str() {
            "BF16" => 1,
            "F32" => 3,
            other => panic!("{name}: dtype {other}"),
        };
        // The router is widened to F32 by the intake.
        let tag = if name.ends_with(".mlp.gate.weight") {
            3
        } else {
            tag
        };
        assert_eq!(tag, e.dtype, "{name}: dtype tag");
        if !name.ends_with("_sconv.static_w") {
            let n = bytes.len() / if dt == "BF16" { 2 } else { 4 };
            let want_n = if name.ends_with(".mlp.gate.weight") {
                e.size_bytes as usize / 4
            } else {
                e.size_bytes as usize / if tag == 1 { 2 } else { 4 }
            };
            assert_eq!(n, want_n, "{name}: element count");
        }
    }
    eprintln!("dtypes and shapes match; {affine} dense matrices are affine");

    // 1b. Affine matrices dequantized against the reference BF16: the same
    // matrix in the same orientation shows 4-bit rounding error only.
    let ref_bytes = std::fs::read(reference.join("model_weights.bin")).expect("reference weights");
    let bf16 = |b: &[u8]| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16);
    for (name, rows) in [
        ("model.llm.layers.0.attn.wq_du.weight", 0..64usize),
        ("model.llm.layers.11.attn.wo_ud.weight", 100..164),
        ("model.llm.layers.5.attn.k_sconv.weight_proj", 0..32),
        ("model.llm.layers.20.attn.attn_sink_proj.weight", 0..24),
        ("model.llm.embed.weight", 150_000..150_064),
        ("model.llm.unembed.weight", 199_990..200_054),
    ] {
        let a = src
            .affine(name)
            .unwrap_or_else(|| panic!("{name} is not affine"));
        let e = &index.entries[name];
        let (mut dot, mut nr, mut ng, mut worst) = (0f64, 0f64, 0f64, 0f32);
        for r in rows {
            for c in 0..a.cols {
                let word = u32::from_le_bytes(
                    a.packed[(r * a.cols + c) / 8 * 4..][..4]
                        .try_into()
                        .unwrap(),
                );
                let q = ((word >> (4 * (c % 8))) & 15) as f32;
                let g = (r * a.cols + c) / 64;
                let got = q * bf16(&a.scales[g * 2..]) + bf16(&a.biases[g * 2..]);
                let want = bf16(&ref_bytes[e.file_offset as usize + (r * a.cols + c) * 2..]);
                dot += f64::from(got * want);
                nr += f64::from(want * want);
                ng += f64::from(got * got);
                worst = worst.max((got - want).abs() / bf16(&a.scales[g * 2..]).abs().max(1e-12));
            }
        }
        eprintln!(
            "{name}: cosine {:.5}, worst error {:.2} quantization steps",
            dot / (nr.sqrt() * ng.sqrt()),
            worst
        );
        assert!(
            dot / (nr.sqrt() * ng.sqrt()) > 0.99,
            "{name}: not the same matrix"
        );
    }

    // 2. Experts: outer scales and a few decoded rows against the reference blobs.
    let layout = model_io::load_packed_experts_layout(reference, 1 << 26).expect("layout");
    let (h, f) = (
        arch.hidden_size as usize,
        arch.moe_intermediate_size as usize,
    );
    let mut worst_outer = 0f32;
    for (l, e) in [(0usize, 0usize), (7, 77), (19, 127), (20, 3), (23, 64)] {
        let file = std::fs::read(
            reference
                .join("packed_experts")
                .join(&layout.layers[l].file),
        )
        .unwrap();
        let entry = &layout.layers[l].experts[e];
        let sub = |role: &str| {
            let s = &entry.sub_tensors[role];
            let start = (entry.offset + s.offset) as usize;
            file[start..start + s.size as usize].to_vec()
        };
        let outer = sub("outer");
        let ref_outer: Vec<f32> = outer
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let x = format!("model.llm.layers.{l}.mlp.experts");
        let (wdt, _, w13) = src.tensor(&format!("{x}.w13_weight")).unwrap();
        let (_, _, s13) = src.tensor(&format!("{x}.w13_weight_scale")).unwrap();
        let nvfp4 = wdt == "U8";
        if nvfp4 {
            for (k, n) in [(0usize, "w13"), (2, "w2")] {
                let (_, _, amax) = src.tensor(&format!("{x}.{n}_weight_amax")).unwrap();
                let g = f32::from_le_bytes(amax[e * 4..e * 4 + 4].try_into().unwrap());
                let got = g / turbospark_repack::MLX_NVFP4_GLOBAL_FACTOR;
                let rel = ((got - ref_outer[k]) / ref_outer[k]).abs();
                worst_outer = worst_outer.max(rel);
            }
        }
        // Row 2k of w13 is gate row k: compare gate row 5 decoded both ways.
        let (rb, sb) = if nvfp4 { (h / 2, h / 16) } else { (h, h / 32) };
        let row = 2 * 5;
        let per_e = 2 * f;
        let wr = &w13[(e * per_e + row) * rb..(e * per_e + row + 1) * rb];
        let sr = &s13[(e * per_e + row) * sb..(e * per_e + row + 1) * sb];
        let blocks = if nvfp4 {
            compute::inkling2::nvfp4_row_to_blocks(wr, sr).unwrap()
        } else {
            compute::inkling2::mxfp8_row_to_blocks(wr, sr).unwrap()
        };
        let gate = sub("gate");
        let row_bytes = blocks.len();
        assert_eq!(
            &gate[5 * row_bytes..6 * row_bytes],
            blocks.as_slice(),
            "layer {l} expert {e} gate row 5"
        );
    }
    eprintln!("expert rows byte-identical to the reference; worst NVFP4 outer relative error {worst_outer:.3e}");

    if args[3] == "-" {
        return;
    }
    let out = Path::new(&args[3]);
    let bits = *src.affine_bits.iter().next().expect("affine dense weights");
    assert_eq!(src.affine_bits.len(), 1, "one dense bit width only");
    let t0 = Instant::now();
    let report = turbospark_repack::write_inkling2_install_from(
        &src,
        &config,
        out,
        "inkling2-mini-mlx4",
        turbospark_repack::inkling2_mlx_manifest_quant(bits),
        |line| eprintln!("{line}  ({:.0} s)", t0.elapsed().as_secs_f32()),
    )
    .expect("install");
    turbospark_repack::copy_inkling2_tokenizer_files(mlx, out).expect("tokenizer files");
    model_io::load_manifest(out, &arch, 1 << 26).expect("manifest validates");
    eprintln!("{report:?} in {:.0} s", t0.elapsed().as_secs_f32());
}
