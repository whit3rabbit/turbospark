//! ROADMAP Phase G Stage 2's boundary, enforced rather than documented.
//!
//! Stage 1 refused every GGUF install, because no kernel in this port read a
//! block layout. Stage 2 moved that line rather than erasing it, and ROADMAP
//! Phase S moved it again. Q8_0 and Q4_K each have a resident GEMV, an
//! embedding lookup and the routed-expert decode pair behind them; Q6_K has a
//! resident GEMV and an embedding lookup; IQ3_XXS, IQ4_XS and IQ4_NL have a
//! resident GEMV each plus the half of the routed pair their real file asks
//! for. Those six OPEN. Q4_0 has nothing and still refuses.
//!
//! Both directions are asserted here, and the refusal is checked twice over,
//! because a single gate is a single point of failure for a whole class of
//! silently-wrong numbers:
//!
//! 1. `load_manifest` refuses a `scheme: "gguf"` slot whose `ggmlType` is
//!    outside `model_io::EXECUTABLE_GGUF_TYPES`.
//! 2. `RealForwardRunner::open` refuses a resident-index dtype tag outside
//!    the same set, which is the backstop for an install whose manifest was
//!    edited to get past (1) -- exactly what someone trying to force one open
//!    would do. It reads the bytes rather than a claim about them.

#![cfg(target_os = "macos")]

use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_repack::{
    build_synthetic_gemma4_gguf, parse_gguf_header, write_gguf_install_streamed, MemoryRangeSource,
    SyntheticGgufShape, GGUF_DEFAULT_MAX_HEADER_BYTES,
};
use turbospark_runtime::RealForwardRunner;

fn tempdir() -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "turbospark-gguf-refused-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}

/// A shape whose every quantized row is a whole number of 32-element Q8_0
/// blocks, which is what real ggml files always are: the routed
/// `moe_intermediate` rows and the `hidden`-length rows both have to tile.
/// The default fixture shape does NOT (its `moe_intermediate` is 16), and it
/// stays that way because the repack-side tests want the smallest file.
fn executable_shape() -> SyntheticGgufShape {
    SyntheticGgufShape {
        moe_intermediate: 32,
        ..SyntheticGgufShape::default()
    }
}

/// Writes a GGUF-sourced install and returns its directory plus the arch it
/// declares.
fn gguf_install(shape: SyntheticGgufShape) -> (std::path::PathBuf, model_io::ArchConfig) {
    let (bytes, _) = build_synthetic_gemma4_gguf(shape);
    let header = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).expect("parse");
    let dir = tempdir();
    let arch = write_gguf_install_streamed(
        &dir,
        &header,
        &MemoryRangeSource::new(&bytes),
        "gguf-stage2",
        |_| {},
    )
    .expect("write install");
    (dir, arch)
}

/// Rewrites every resident-index entry carrying `from` to carry `to`, in
/// place, leaving the bytes those entries point at untouched. That is exactly
/// the state a hand-forged install would be in: a block type this port cannot
/// execute, claiming to be one it can, or the reverse.
fn retag_dtypes(dir: &std::path::Path, from: u8, to: u8) -> usize {
    let path = dir.join("model_weights.bin");
    let mut bytes = std::fs::read(&path).unwrap();
    let index = model_io::load_resident_index(&path).expect("index");
    let mut changed = 0;
    for i in 0..index.header.entry_count as usize {
        let at = model_io::HEADER_BYTES + i * model_io::ENTRY_BYTES + 6;
        if bytes[at] == from {
            bytes[at] = to;
            changed += 1;
        }
    }
    std::fs::write(&path, &bytes).unwrap();
    changed
}

/// The Stage 2 deliverable: a Q8_0 GGUF install opens, and the kernels behind
/// it are the ones this test is really about (the embedding lookup, the
/// resident GEMV, and the routed-expert decode pair).
#[test]
fn a_q8_0_gguf_install_opens() {
    let (dir, arch) = gguf_install(executable_shape());

    match RealForwardRunner::open(&dir, arch) {
        Ok(_) => {}
        Err(e) => panic!("a Q8_0 GGUF install must open now that its kernels exist: {e}"),
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// The K-quant mixture a real `Q4_K_M` carries, end to end: routed experts
/// and the embedding table at Q4_K, the attention projections at Q6_K, the
/// rest Q8_0. Three block types in one install, which is the case a
/// single-type fixture cannot make: every dispatch site has to pick from the
/// TENSOR rather than from one decision made at open.
#[test]
fn a_mixed_k_quant_gguf_install_opens() {
    let (dir, arch) = gguf_install(SyntheticGgufShape::k_quant());

    match RealForwardRunner::open(&dir, arch) {
        Ok(_) => {}
        Err(e) => panic!("a Q4_K/Q6_K GGUF install must open now that its kernels exist: {e}"),
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// The manifest gate, on a block type with no kernel. Q4_0 is the one the
/// parser knows and nothing decodes: it has no CPU reference, no GEMV and no
/// expert pair, so an install of one must not open and the refusal must say
/// which type.
#[test]
fn a_block_type_without_kernels_is_refused_by_the_manifest() {
    let (dir, arch) = gguf_install(executable_shape());

    let path = dir.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    for slot in ["embedding", "attention", "sharedExpert", "routedExpert"] {
        manifest["quant"][slot]["ggmlType"] = serde_json::json!("Q4_0");
    }
    std::fs::write(&path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();

    let text = match RealForwardRunner::open(&dir, arch) {
        Ok(_) => panic!("a Q4_0 install must not open"),
        Err(e) => e.to_string(),
    };
    assert!(
        text.contains("Q4_0") || text.contains("q4_0"),
        "the refusal must name the block type, got: {text}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The dtype backstop on its own: the manifest still says Q8_0, but the
/// bytes on disk are tagged Q4_0. `open` has to believe the index.
#[test]
fn the_dtype_backstop_fires_even_if_the_manifest_is_forged() {
    let (dir, arch) = gguf_install(executable_shape());

    let changed = retag_dtypes(&dir, 6, 9);
    assert!(changed > 0, "the fixture carries no Q8_0 resident tensors");

    let text = match RealForwardRunner::open(&dir, arch) {
        Ok(_) => panic!("forging the manifest must not make a Q4_0 install openable"),
        Err(e) => e.to_string(),
    };
    assert!(
        text.contains("GGUF block dtype"),
        "expected the resident-index backstop, got: {text}"
    );
    assert!(
        text.contains("Phase G"),
        "the refusal should point at the work that lifts it, got: {text}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// Opening is not running. This drives real decode steps through the whole
/// Q8_0 path on real Metal hardware: the embedding lookup, the attention and
/// shared-expert GEMVs, the routed-expert decode pair reading streamed
/// blobs, and the output head. The fixture's weights are deterministic
/// patterns rather than trained ones, so nothing about the TOKENS means
/// anything; what is asserted is that every logit is finite and that the
/// distribution is not degenerate, which a kernel reading a block layout
/// wrongly does not satisfy for long.
#[test]
fn a_q8_0_gguf_install_decodes() {
    decodes(executable_shape());
}

/// The same drive through the mixed K-quant install, which is where the Q4_K
/// routed pair, the Q4_K embedding lookup and the Q6_K resident GEMV all run
/// on real hardware inside one forward pass.
#[test]
fn a_mixed_k_quant_gguf_install_decodes() {
    decodes(SyntheticGgufShape::k_quant());
}

/// ROADMAP Phase S's mixture, and the first install in this suite that is
/// mixed along TWO axes at once: IQ3_XXS gate/up over IQ4_NL down (the two
/// phases of one expert reading different layouts), with the LAST LAYER
/// different again at IQ4_XS over Q8_0.
///
/// Both axes are new, and each defeats a different piece of the old
/// plumbing. `RoutedBlobLayout` was one value for a whole install, read off
/// the manifest's single `ggmlType`; `MoeExpertOffsets` was one struct read
/// off layer 0 expert 0, on the reasoning that the writer packs every blob
/// identically. Both were true of every install that existed and neither
/// survives a mixture: the manifest cannot name one type, and the offsets
/// follow the byte sizes, which differ per layer.
///
/// The odd layer is LAST rather than first on purpose. A bug that resolved
/// everything from layer 0 would still produce a working install if layer 0
/// were the odd one out -- it would just be wrong everywhere else, and this
/// fixture's logits would still be finite. Putting the difference at the end
/// means layer 0's answer is the majority answer, which is what a
/// resolve-once bug would use.
#[test]
fn the_phase_s_iq_mixture_decodes() {
    decodes(SyntheticGgufShape::iq_mixed());
}

/// ROADMAP M5's block type, inside a whole forward pass.
///
/// What this adds over `crates/gpu/tests/moe_gguf_parity.rs`, which already
/// holds the MXFP4 pair against the CPU reference: the parity test hands the
/// kernels a hand-built blob at hand-chosen offsets, where this one makes the
/// REPACK WALK produce the blob, the manifest and the layout, and then makes
/// `RealForwardRunner` resolve a dispatch from them. Every hole the real files
/// have exposed in this port was in that resolution rather than in a block
/// format (`crates/repack/CLAUDE.md` Gotcha 5 lists four of them), and the
/// cheapest place to find the next one is here rather than after a 12 GB
/// stream.
///
/// It is also the first install whose expert type has NO resident GEMV, which
/// is the asymmetry `EXECUTABLE_GGUF_TYPES` and `EXECUTABLE_GGUF_DTYPES` now
/// differ over: this install has to OPEN (its routed slot names a type with a
/// routed pair) while an MXFP4 attention tensor would still be refused.
#[test]
fn the_mxfp4_expert_mixture_decodes() {
    decodes(SyntheticGgufShape::mxfp4());
}

/// The other half of that asymmetry, which no other block type can express:
/// MXFP4 is executable in a routed slot and NOT as a resident tensor, so an
/// install that moves it into the resident index must be refused even though
/// the very same type decodes fine two lines above.
///
/// Retags the Q8_0 resident tensors rather than the routed blobs, because the
/// routed experts are not IN the resident index -- which is precisely why one
/// type can be executable in one place and not the other.
#[test]
fn mxfp4_is_refused_as_a_resident_tensor_though_its_experts_run() {
    let (dir, arch) = gguf_install(SyntheticGgufShape::mxfp4());

    let changed = retag_dtypes(&dir, 6, 14);
    assert!(changed > 0, "the fixture carries no Q8_0 resident tensors");

    let text = match RealForwardRunner::open(&dir, arch) {
        Ok(_) => panic!("MXFP4 has no resident GEMV; a resident MXFP4 tensor must not open"),
        Err(e) => e.to_string(),
    };
    assert!(
        text.contains("RESIDENT kernel"),
        "the refusal must say which kind of kernel is missing, got: {text}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The offsets really are per layer, checked on the fixture rather than
/// through the decode above. A resolve-once bug is not guaranteed to produce
/// a NaN -- it reads valid bytes at the wrong place -- so the structural
/// claim is worth asserting directly.
#[test]
fn a_mixed_install_gives_its_layers_different_expert_layouts() {
    let (dir, _) = gguf_install(SyntheticGgufShape::iq_mixed());
    let layout = model_io::load_packed_experts_layout(
        &dir,
        model_io::PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES,
    )
    .expect("layout.json");

    let dtypes = |layer: usize| {
        let subs = &layout.expert(layer, 0).sub_tensors;
        (
            subs["gate"].dtype.clone(),
            subs["up"].dtype.clone(),
            subs["down"].dtype.clone(),
        )
    };
    assert_eq!(
        dtypes(0),
        ("iq3_xxs".into(), "iq3_xxs".into(), "iq4_nl".into())
    );
    let last = layout.layers.len() - 1;
    assert_eq!(
        dtypes(last),
        ("iq4_xs".into(), "iq4_xs".into(), "q8_0".into())
    );
    // Different types mean different byte sizes mean different offsets, which
    // is the concrete reason one struct read off layer 0 cannot serve.
    assert_ne!(
        layout.expert(0, 0).sub_tensors["down"].offset,
        layout.expert(last, 0).sub_tensors["down"].offset
    );
    // And the manifest names every type it carries, not just the dominant one.
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let mut declared: Vec<String> = manifest["quant"]["routedExpert"]["ggmlTypes"]
        .as_array()
        .expect("a mixed routed slot declares ggmlTypes")
        .iter()
        .map(|v| v.as_str().unwrap().to_lowercase())
        .collect();
    declared.sort();
    assert_eq!(declared, ["iq3_xxs", "iq4_nl", "iq4_xs", "q8_0"]);

    std::fs::remove_dir_all(&dir).ok();
}

fn decodes(shape: SyntheticGgufShape) {
    use half::f16;
    use turbospark_runtime::LogitProducer;

    let vocab = shape.vocab as usize;
    let (dir, arch) = gguf_install(shape);
    let mut runner = RealForwardRunner::open(&dir, arch).expect("opens");

    runner.reset();
    let mut token = 5i32;
    for position in 0..4usize {
        let mut logits = vec![f16::from_f32(0.0); vocab];
        runner
            .produce(token, position, &mut logits)
            .expect("produce succeeds");

        let bad: Vec<usize> = logits
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.to_f32().is_finite())
            .map(|(i, _)| i)
            .collect();
        assert!(
            bad.is_empty(),
            "non-finite logit at position {position}: {} of {vocab}, first {:?}",
            bad.len(),
            &bad[..bad.len().min(8)]
        );
        let first = logits[0].to_f32();
        assert!(
            logits.iter().any(|v| v.to_f32() != first),
            "every logit is {first} at position {position}: the head produced nothing"
        );

        token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
            .map(|(i, _)| i as i32)
            .unwrap();
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// **THE FP16 EXCEPTION IS SCOPED BY NAME, AND BOTH DIRECTIONS MATTER**
/// (ROADMAP M-V3).
///
/// `readable_resident_dtype` accepts tag 2 (FP16) only for `vision.`-prefixed
/// tensors, because the `qwen3_5` vision tower is FP16 end to end -- its Metal
/// kernels bind `half` where every other kernel in `crates/gpu` binds
/// `bfloat` -- while every TEXT consumer of an unquantized tensor
/// (`norm_view`, `read_bf16_host`) is dtype-BLIND and decodes by byte width.
/// So an FP16 tensor either of those reaches is MISREAD rather than rejected:
/// same width, every length check passes, values wrong by up to 2^112
/// (AGENTS.md Gotcha 45).
///
/// This test is the half that could rot silently. The accepting half is
/// exercised by every vision install; nothing but this says the exception did
/// not widen into a blanket permission, and a blanket one is not a narrower
/// bug than the one the refusal was written to prevent -- it is the same bug.
///
/// Retagging BF16 (1) to FP16 (2) on a GGUF install touches only text
/// tensors, since this fixture has no vision tower at all.
#[test]
fn an_fp16_resident_tensor_is_refused_unless_it_is_the_vision_tower() {
    let (dir, arch) = gguf_install(executable_shape());
    let changed = retag_dtypes(&dir, 1, 2);
    assert!(
        changed > 0,
        "the fixture carries no BF16 resident tensor to retag, so this proves nothing"
    );

    // `expect_err` does not compile here: it needs the OK type to be `Debug`
    // and `RealForwardRunner` is not (`crates/runtime` Gotcha 17).
    let Err(err) = RealForwardRunner::open(&dir, arch) else {
        panic!("an FP16 text tensor must be refused: it would be decoded as BF16");
    };
    let text = err.to_string();
    assert!(
        text.contains("resident dtype 2"),
        "the refusal should name the tag it refused: {text}"
    );
    assert!(
        text.contains("narrowed to BF16"),
        "the refusal should say what the writer owes: {text}"
    );

    std::fs::remove_dir_all(&dir).ok();
}
