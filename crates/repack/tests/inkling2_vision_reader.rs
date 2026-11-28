//! The Inkling 2 vision tower's config parse and ingest
//! (`docs/INKLING2_VISION_PHASE0.md`).
//!
//! The config test runs on the checked-in fixture (the pinned `config.json`).
//! The reader test reads the real tower and is ignored by default: point
//! `TURBOSPARK_INKLING2_VISION_SAFETENSORS` at the file
//! `scripts/inkling2_vision_fetch.py` writes (636 MB, BF16, HF names) and run
//! with `--ignored`.

use model_io::VisionTowerKind;
use turbospark_repack::{
    canonicalize_vision_header, parse_header, parse_vision_config, read_inkling2_vision_entries,
    Gemma4Shards, MemoryRangeSource, VISION_PREFIX,
};

const FIXTURE: &str = include_str!("fixtures/inkling2/config.json");

#[test]
fn the_pinned_config_declares_the_inkling2_tower() {
    let v = parse_vision_config(FIXTURE).unwrap();
    assert_eq!(v.tower, VisionTowerKind::Inkling2);
    assert!(v.is_active());
    assert_eq!((v.depth, v.hidden_size, v.num_heads), (23, 1024, 16));
    assert_eq!(v.out_hidden_size, 2304);
    assert_eq!(v.image_token_id, 200_054);
}

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn f16(bits: u16) -> f32 {
    half_to_f32(bits)
}

/// IEEE half to f32, written out so the test does not share the crate's
/// converter.
fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let frac = f32::from(h & 0x3ff);
    match exp {
        0 => sign * frac * 2f32.powi(-24),
        31 => sign * f32::INFINITY,
        e => sign * (1.0 + frac / 1024.0) * 2f32.powi(e - 15),
    }
}

#[test]
#[ignore = "reads the real 636 MB tower; set TURBOSPARK_INKLING2_VISION_SAFETENSORS"]
fn the_real_tower_ingests_with_the_fold_and_the_permutation() {
    let Ok(path) = std::env::var("TURBOSPARK_INKLING2_VISION_SAFETENSORS") else {
        panic!("set TURBOSPARK_INKLING2_VISION_SAFETENSORS to the fetched vision.safetensors");
    };
    let bytes = std::fs::read(&path).unwrap();
    let source = MemoryRangeSource::new(&bytes);
    let mut header = parse_header(&bytes, 1 << 26).unwrap();
    let base = header.data_region_start();
    // The raw source values, before canonicalization renames them.
    let raw = |name: &str| -> Vec<f32> {
        let t = &header.tensors[name];
        let (s, e) = (base + t.data_offsets.0, base + t.data_offsets.1);
        bytes[s as usize..e as usize]
            .chunks_exact(2)
            .map(|c| bf16(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    };
    let ls1 = raw("model.visual.backbone.layers.0.ls_1.gamma");
    let out_b = raw("model.visual.backbone.layers.0.self_attn.out_proj.bias");
    let q_w = raw("model.visual.backbone.layers.0.self_attn.q_proj.weight");
    let k_b = raw("model.visual.backbone.layers.0.self_attn.k_proj.bias");

    canonicalize_vision_header(&mut header).unwrap();
    let names: Vec<String> = header.tensors.keys().cloned().collect();
    let bases: Vec<&str> = names
        .iter()
        .filter(|n| n.starts_with(VISION_PREFIX))
        .map(String::as_str)
        .collect();
    assert_eq!(bases.len(), 422);
    let shards = Gemma4Shards::single(&header, &source);
    let vision = parse_vision_config(FIXTURE).unwrap();
    let read = read_inkling2_vision_entries(&shards, &bases, &vision).unwrap();

    assert_eq!(
        read.entries.len(),
        7,
        "5 residents, proj_out, the class row"
    );
    assert_eq!(read.blocks.experts.len(), 23);
    let block0 = &read.blocks.experts[0].sub_tensors;
    assert_eq!(block0.len(), 12);
    let role = |r: &str| block0.iter().find(|s| s.role == r).unwrap();
    let vals = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(2)
            .map(|c| f16(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    };

    // The fold: proj_b == ls_1 * out_proj.bias, to FP16 rounding.
    let proj_b = vals(&role("proj_b").bytes);
    for i in [0usize, 1, 511, 1023] {
        let want = ls1[i] * out_b[i];
        assert!(
            (proj_b[i] - want).abs() <= want.abs() * 1e-3 + 1e-7,
            "proj_b[{i}] {} vs {want}",
            proj_b[i]
        );
    }

    // The permutation: fused row 1 of head 0 is source q row 2; row 32 is
    // source row 1; head 1 starts at row 64 with source row 64.
    let qkv_w = vals(&role("qkv_w").bytes);
    let row = |m: &[f32], r: usize| m[r * 1024..r * 1024 + 8].to_vec();
    let close = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= y.abs() * 1e-3)
    };
    assert!(close(&row(&qkv_w, 1), &row(&q_w, 2)));
    assert!(close(&row(&qkv_w, 32), &row(&q_w, 1)));
    assert!(close(&row(&qkv_w, 64), &row(&q_w, 64)));
    assert!(close(&row(&qkv_w, 65), &row(&q_w, 66)));
    // k's bias is permuted the same way, at offset 1024 of the fused bias.
    let qkv_b = vals(&role("qkv_b").bytes);
    assert!((qkv_b[1024 + 1] - k_b[2]).abs() <= k_b[2].abs() * 1e-3);
    assert!((qkv_b[1024 + 32] - k_b[1]).abs() <= k_b[1].abs() * 1e-3);

    // Nothing in the real tower overflows FP16 or lands subnormal in bulk.
    let lossy: usize = read.lossy_conversion.iter().map(|(_, n)| n).sum();
    eprintln!(
        "block stride {} bytes; {} tensors lost bits ({lossy} values)",
        read.block_stride,
        read.lossy_conversion.len()
    );
}
