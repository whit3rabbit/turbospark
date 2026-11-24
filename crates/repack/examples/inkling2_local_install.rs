//! Installs Inkling 2 Mini from a local HF snapshot, then spot-checks the
//! install against the checkpoint.
//!
//! ```text
//! cargo run --release -p turbospark-repack --example inkling2_local_install -- \
//!     <snapshot dir> <install dir>
//! ```
//!
//! The snapshot needs `config.json`, `model-00001-of-00001.safetensors` and
//! `tokenizer.json`. The spot check decodes gate, up and down rows of a few
//! experts on an NVFP4 layer and an MXFP8 layer from the written blobs and
//! compares them with the HF bytes decoded by the reference decoders, and
//! re-validates the manifest and sidecar through the loaders.

use compute::inkling2::{dequant_mxfp8_row, dequant_nvfp4_row, dequantize_mxfp8_blocks};
use model_io::inkling2_config::Inkling2Config;
use model_io::SafetensorsFile;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <snapshot dir> <install dir>", args[0]);
        std::process::exit(2);
    }
    let (snap, out) = (Path::new(&args[1]), Path::new(&args[2]));
    let config = std::fs::read_to_string(snap.join("config.json")).expect("config.json");
    let src = SafetensorsFile::open(&snap.join("model-00001-of-00001.safetensors"))
        .expect("open checkpoint");
    let t0 = Instant::now();
    let report =
        turbospark_repack::write_inkling2_install(&src, &config, out, "inkling2-mini", |line| {
            eprintln!("{line}  ({:.0} s)", t0.elapsed().as_secs_f32())
        })
        .expect("install");
    turbospark_repack::copy_inkling2_tokenizer_files(snap, out).expect("tokenizer files");
    eprintln!("{report:?} in {:.0} s", t0.elapsed().as_secs_f32());

    let (arch, side) = turbospark_repack::parse_inkling2_config(&config).unwrap();
    model_io::load_manifest(out, &arch, 1 << 26).expect("manifest validates");
    assert_eq!(Inkling2Config::load(out, &arch).unwrap(), side);
    let layout = model_io::load_packed_experts_layout(out, 1 << 26).expect("layout loads");
    let (h, f) = (
        arch.hidden_size as usize,
        arch.moe_intermediate_size as usize,
    );
    let mut checked = 0usize;
    for (l, e) in [(0usize, 0usize), (7, 77), (19, 127), (20, 3), (23, 64)] {
        let x = format!("model.llm.layers.{l}.mlp.experts");
        let raw = |n: &str| src.raw_bytes(&format!("{x}.{n}")).unwrap();
        let nvfp4 = src.descriptor(&format!("{x}.w13_weight")).unwrap().dtype == "U8";
        let entry = &layout.layers[l].experts[e];
        let file = std::fs::read(out.join("packed_experts").join(&layout.layers[l].file)).unwrap();
        let part = |role: &str| {
            let s = &entry.sub_tensors[role];
            let start = (entry.offset + s.offset) as usize;
            file[start..start + s.size as usize].to_vec()
        };
        for (role, rows, cols, w, s, pick) in [
            (
                "gate",
                f,
                h,
                raw("w13_weight"),
                raw("w13_weight_scale"),
                2usize,
            ),
            ("up", f, h, raw("w13_weight"), raw("w13_weight_scale"), 3),
            ("down", h, f, raw("w2_weight"), raw("w2_weight_scale"), 1),
        ] {
            let blocks = part(role);
            let got = if nvfp4 {
                compute::dequantize_nvfp4(&blocks, rows * cols)
            } else {
                dequantize_mxfp8_blocks(&blocks, rows * cols)
            };
            let src_rows = if role == "down" { h } else { 2 * f };
            for k in [0usize, 1, rows / 2, rows - 1] {
                // pick: 2 = even w13 rows, 3 = odd w13 rows, 1 = straight.
                let r = e * src_rows
                    + match pick {
                        2 => 2 * k,
                        3 => 2 * k + 1,
                        _ => k,
                    };
                let mut want = vec![0.0f32; cols];
                if nvfp4 {
                    let (rb, sb) = (cols / 2, cols / 16);
                    dequant_nvfp4_row(
                        &w[r * rb..(r + 1) * rb],
                        &s[r * sb..(r + 1) * sb],
                        1.0,
                        &mut want,
                    );
                } else {
                    let (rb, sb) = (cols, cols / 32);
                    dequant_mxfp8_row(
                        &w[r * rb..(r + 1) * rb],
                        &s[r * sb..(r + 1) * sb],
                        &mut want,
                    );
                }
                assert_eq!(
                    &got[k * cols..(k + 1) * cols],
                    &want[..],
                    "layer {l} expert {e} {role} row {k}"
                );
                checked += 1;
            }
        }
        let rec = part("outer");
        let o: Vec<f32> = rec
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if nvfp4 {
            let scalar = |n: &str, i: usize| {
                let b = raw(n);
                f32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])
            };
            let o13 =
                scalar("w13_weight_amax", e) / scalar("w13_weight_outer_scale_denominator", 0);
            let o2 = scalar("w2_weight_amax", e) / scalar("w2_weight_outer_scale_denominator", 0);
            assert_eq!(o, vec![o13, o13, o2, 0.0], "layer {l} expert {e} outer");
        } else {
            assert_eq!(o, vec![1.0, 1.0, 1.0, 0.0], "layer {l} expert {e} outer");
        }
    }
    eprintln!("spot check: {checked} rows across NVFP4 and MXFP8 layers decode exactly; manifest, sidecar and layout load");
}
