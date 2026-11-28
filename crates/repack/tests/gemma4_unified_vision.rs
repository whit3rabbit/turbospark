//! Intake of the `gemma4_unified` encoder-free embedder
//! (`docs/GEMMA4_UNIFIED_VISION_PHASE0.md`): what the walk ingests, what it
//! refuses by name, and what stays unclassifiable for other installs.

use model_io::ModelFamily;
use turbospark_repack::{
    build_synthetic_gemma4_unified_install, classify_for_family, drop_gemma4_unified_audio,
    fetch_safetensors_header, parse_gemma4_quantization, parse_gemma4_unified_vision_config,
    read_vision_entries_quantized, synthetic_gemma4_unified_checkpoint, Gemma4Bucket, Gemma4Shards,
    MemoryRangeSource, RangeSource, GEMMA4_UNIFIED_ROLES,
};

#[test]
fn the_install_carries_one_blob_with_every_role_dequantized() {
    let dir = std::env::temp_dir().join(format!("tsp_g4_unified_repack_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (arch, _) = build_synthetic_gemma4_unified_install(&dir, 128, 6, 8, "t").unwrap();
    let layout: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("packed_vision/layout.json")).unwrap())
            .unwrap();
    let blob = &layout["layers"][0]["experts"][0]["tensors"];
    // 64 x 128 projector and 128 x 6912 dense matrix at FP16, not packed.
    assert_eq!(blob["proj_w"]["size"], 64 * 128 * 2);
    assert_eq!(blob["dense_w"]["size"], 128 * 6912 * 2);
    assert_eq!(blob.as_object().unwrap().len(), GEMMA4_UNIFIED_ROLES.len());
    assert_eq!(arch.vision.depth, 1);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_reader_refuses_missing_unknown_and_mis_declared_tensors() {
    let (config, blob, _) = synthetic_gemma4_unified_checkpoint(128, 6, 8);
    let vision = parse_gemma4_unified_vision_config(&config).unwrap();
    let quant = parse_gemma4_quantization(&config).unwrap();
    let source = MemoryRangeSource::new(&blob);
    let header = fetch_safetensors_header(&source).unwrap();
    let shards = Gemma4Shards::new(vec![(&header, &source as &dyn RangeSource)]).unwrap();
    let all: Vec<&str> = GEMMA4_UNIFIED_ROLES.iter().map(|(_, n)| *n).collect();
    let read = |names: &[&str], quant| {
        read_vision_entries_quantized(&shards, names, &vision, quant).map(|r| r.entries.len())
    };
    assert_eq!(read(&all, Some(&quant)).unwrap(), 0);

    let err = read(&all[1..], Some(&quant)).unwrap_err().to_string();
    assert!(err.contains("patch_ln1.weight"), "{err}");

    let mut extra = all.clone();
    extra.push("vision_embedder.patch_ln3.weight");
    let err = read(&extra, Some(&quant)).unwrap_err().to_string();
    assert!(err.contains("patch_ln3"), "{err}");

    // The 4-bit planes read as the 8-bit default unpack to half the width.
    let mut wrong = quant.clone();
    wrong.bits_overrides.clear();
    let err = read(&all, Some(&wrong)).unwrap_err().to_string();
    assert!(
        err.contains("patch_dense") || err.contains("scales"),
        "{err}"
    );
}

#[test]
fn only_the_gemma4_family_claims_the_embedder_and_audio_is_not_a_walk_drop() {
    let embedder = "vision_embedder.patch_dense.weight";
    assert!(matches!(
        classify_for_family(embedder, 6, ModelFamily::Gemma4),
        Gemma4Bucket::VisionTower
    ));
    for family in [ModelFamily::QwenGdnDense, ModelFamily::Llama] {
        assert!(matches!(
            classify_for_family(embedder, 6, family),
            Gemma4Bucket::Unknown
        ));
    }
    // E2B/E4B spell their audio projector the same way: the classifier must
    // keep refusing it. Only the unified-gated header pass removes it.
    let audio = "embed_audio.embedding_projection.weight";
    assert!(matches!(
        classify_for_family(audio, 6, ModelFamily::Gemma4),
        Gemma4Bucket::Unknown
    ));
    let (_, blob, _) = synthetic_gemma4_unified_checkpoint(128, 6, 8);
    let mut header = fetch_safetensors_header(&MemoryRangeSource::new(&blob)).unwrap();
    let before = header.tensors.len();
    assert_eq!(
        drop_gemma4_unified_audio(&mut header),
        [
            "embed_audio.embedding_projection.weight",
            "embed_audio.embedding_projection.scales",
            "embed_audio.embedding_projection.biases"
        ]
    );
    assert_eq!(header.tensors.len(), before - 3);
}
