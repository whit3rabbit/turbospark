#![cfg(target_os = "macos")]
use half::f16;
use std::sync::atomic::{AtomicU64, Ordering};
use turbospark_repack::{
    build_synthetic_minimax_gguf, parse_gguf_header, write_gguf_install_streamed, MemoryRangeSource,
};
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};

static COUNTER: AtomicU64 = AtomicU64::new(0);
fn install() -> (std::path::PathBuf, model_io::ArchConfig) {
    let dir = std::env::temp_dir().join(format!(
        "minimax-fixture-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let (bytes, _) = build_synthetic_minimax_gguf();
    let header = parse_gguf_header(&bytes, bytes.len() as u64).unwrap();
    let arch = write_gguf_install_streamed(
        &dir,
        &header,
        &MemoryRangeSource::new(&bytes),
        "minimax-fixture",
        |_| {},
    )
    .unwrap();
    (dir, arch)
}
fn run(r: &mut RealForwardRunner, chunk: bool) -> Vec<u16> {
    r.reset();
    let prompt = [5, 9, 2, 7, 1];
    let mut out = vec![f16::ZERO; 64];
    if chunk {
        r.prefill_chunk(&prompt, 0, &mut out).unwrap();
    } else {
        for (p, &t) in prompt.iter().enumerate() {
            r.produce(t, p, &mut out).unwrap();
        }
    }
    assert!(out.iter().all(|v| v.is_finite()));
    out.iter().map(|v| v.to_bits()).collect()
}
#[test]
fn minimax_mixed_gguf_decode_prefill_reset_and_slots_agree() {
    let (dir, arch) = install();
    let mut small = RealForwardRunner::open_with_options(&dir, arch.clone(), 32, 8).unwrap();
    let expected = run(&mut small, false);
    assert_eq!(expected, run(&mut small, false));
    assert_eq!(expected, run(&mut small, true));
    let mut large = RealForwardRunner::open_with_options(&dir, arch, 32, 16).unwrap();
    assert_eq!(expected, run(&mut large, false));
    assert_eq!(expected, run(&mut large, true));
    let digest = expected
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .fold(0xcbf29ce484222325u64, |h, b| {
            (h ^ b as u64).wrapping_mul(0x100000001b3)
        });
    assert_eq!(digest, 0x9422bdb22e2f00f2);
    drop((small, large));
    std::fs::remove_dir_all(dir).unwrap();
}
#[test]
fn minimax_router_and_correction_bytes_survive_repack() {
    let (dir, _) = install();
    let (bytes, _) = build_synthetic_minimax_gguf();
    let header = parse_gguf_header(&bytes, bytes.len() as u64).unwrap();
    let index = model_io::load_resident_index(&dir.join("model_weights.bin")).unwrap();
    let resident = std::fs::read(dir.join("model_weights.bin")).unwrap();
    let size = turbospark_repack::minimax_gguf_sizing(&header).unwrap();
    assert_eq!(size.resident_bytes, resident.len() as u64);
    let expert_bytes: u64 = std::fs::read_dir(dir.join("packed_experts"))
        .unwrap()
        .map(|e| e.unwrap())
        .filter(|e| e.path().extension().is_some_and(|s| s == "bin"))
        .map(|e| e.metadata().unwrap().len())
        .sum();
    assert_eq!(size.expert_file_bytes, expert_bytes);
    for (source, tail) in [
        ("ffn_gate_inp.weight", "mlp.gate.weight"),
        ("exp_probs_b.bias", "mlp.e_score_correction_bias"),
    ] {
        let e = &index.entries[&format!("language_model.model.layers.0.{tail}")];
        assert_eq!(e.dtype, 3);
        let (start, end) = header
            .absolute_range(&format!("blk.0.{source}"))
            .unwrap()
            .unwrap();
        assert_eq!(
            &resident[e.file_offset as usize..(e.file_offset + e.size_bytes) as usize],
            &bytes[start as usize..end as usize]
        );
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn minimax_manifest_cannot_opt_into_a_tied_head() {
    let (dir, mut arch) = install();
    let path = dir.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(manifest["arch"]["tieWordEmbeddings"], false);
    manifest["arch"]["tieWordEmbeddings"] = true.into();
    std::fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    arch.tie_word_embeddings = true;
    let error = RealForwardRunner::open_with_options(&dir, arch, 32, 8).err();
    std::fs::remove_dir_all(dir).unwrap();
    assert!(matches!(error,
        Some(turbospark_runtime::RealForwardError::Unsupported(ref reason))
            if reason == "MiniMax requires an untied output head"
    ));
}
