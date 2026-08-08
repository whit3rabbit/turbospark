//! Ranged-header check against the real community GGUFs of both supported
//! architectures. `#[ignore]`d and opt-in, like the other `*_network` tests.
//!
//! This deliberately does NOT download a checkpoint. A GGUF header is a few
//! MB of a 20-27 GB file, and reading just that is the whole point of
//! `fetch_gguf_header`. What it proves is the part a synthetic fixture
//! cannot: that this port's parser agrees with what llama.cpp's converter
//! actually writes, and that every tensor name in a real file is one the
//! repack walk knows how to place.
//!
//! ```sh
//! cargo test -p turbospark-repack --test gguf_checkpoint_network --release -- --ignored --nocapture
//! ```

use model_io::ModelFamily;
use turbospark_repack::{
    arch_from_gguf, fetch_gguf_header, ggml_type_name, map_gguf_name, peek_manifest_arch,
    GgufHeader, GgufMapping, HttpRangeSource,
};

const GEMMA4_Q8_0: &str = "https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-Q8_0.gguf";
const QWEN36_Q4_K_M: &str =
    "https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-Q4_K_M.gguf";

/// Collapses `blk.<N>.` to `blk.*.` so a 30-layer model prints 20 shapes
/// rather than 600 names.
fn shape_key(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut parts = name.split('.').peekable();
    while let Some(p) = parts.next() {
        out.push_str(if p.chars().all(|c| c.is_ascii_digit()) && !p.is_empty() {
            "*"
        } else {
            p
        });
        if parts.peek().is_some() {
            out.push('.');
        }
    }
    out
}

fn report(label: &str, h: &GgufHeader) {
    println!("\n=== {label}");
    println!("architecture      {:?}", h.architecture());
    println!("alignment         {}", h.alignment);
    println!("data region start {}", h.data_region_start);
    println!("tensors           {}", h.tensors.len());
    println!("metadata keys     {}", h.metadata.len());

    let mut arch_keys: Vec<&String> = h
        .metadata
        .keys()
        .filter(|k| !k.starts_with("tokenizer."))
        .collect();
    arch_keys.sort();
    println!("-- non-tokenizer metadata");
    for k in arch_keys {
        println!("   {k} = {:?}", h.metadata[k]);
    }

    // Keyed on dims and type as well as name, so a tensor whose SHAPE varies
    // by layer shows up as two rows instead of hiding behind the first one
    // encountered. Gemma 4's global-attention layers differ from its
    // sliding-window layers exactly this way.
    let mut shapes: std::collections::BTreeMap<(String, u32, Vec<u64>), usize> =
        std::collections::BTreeMap::new();
    for (name, info) in &h.tensors {
        *shapes
            .entry((shape_key(name), info.ggml_type, info.dims.clone()))
            .or_insert(0) += 1;
    }
    println!("-- tensor shapes (count, type, dims as stored)");
    for ((key, ty, dims), count) in &shapes {
        println!(
            "   {count:>4}x {key}  {}  {dims:?}",
            ggml_type_name(*ty).unwrap_or("?")
        );
    }

    // A per-layer tensor that is not present on EVERY layer is the single
    // most dangerous thing to map by pattern: the walk would silently place
    // the wrong layer's weights, or drop a layer. Name the exceptions.
    let blocks = h
        .metadata
        .iter()
        .find(|(k, _)| k.ends_with(".block_count"))
        .and_then(|(_, v)| v.as_u64())
        .unwrap_or(0) as usize;
    println!("-- per-layer tensors NOT present on all {blocks} layers");
    let mut by_name: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for name in h.tensors.keys() {
        *by_name.entry(shape_key(name)).or_insert(0) += 1;
    }
    for (key, count) in &by_name {
        if !key.starts_with("blk.*.") || *count == blocks {
            continue;
        }
        let present: Vec<usize> = (0..blocks)
            .filter(|i| {
                h.tensors
                    .contains_key(&key.replacen("blk.*.", &format!("blk.{i}."), 1))
            })
            .collect();
        println!("   {key}: {count} of {blocks}, layers {present:?}");
    }
}

/// Every tensor in a real file must map. An unmapped name is a hole in the
/// table, and the only place it can be found is against a real converter's
/// output: a synthetic fixture only contains names the author already knew.
fn assert_every_name_maps(h: &GgufHeader, family: ModelFamily) {
    let mut unmapped: Vec<&String> = h
        .tensors
        .keys()
        .filter(|n| map_gguf_name(n, family).is_err())
        .collect();
    unmapped.sort();
    unmapped.dedup_by_key(|n| shape_key(n));
    assert!(
        unmapped.is_empty(),
        "{} unmapped tensor names, e.g. {:?}",
        unmapped.len(),
        unmapped.iter().take(10).collect::<Vec<_>>()
    );
}

/// The other half of the check, and the half a mapping table cannot verify
/// on its own: every canonical name produced must be one the runtime
/// actually looks up. The authority for that is the resident index of an
/// install this port already runs.
///
/// Skips with a note when the install is not present, like the other
/// env-gated tests.
fn assert_mapped_names_exist_in_install(h: &GgufHeader, family: ModelFamily, env: &str) {
    let Ok(dir) = std::env::var(env) else {
        println!("(skipped install cross-check: {env} is unset)");
        return;
    };
    let index =
        model_io::load_resident_index(&std::path::Path::new(&dir).join("model_weights.bin"))
            .expect("resident index");

    let mut missing: Vec<String> = Vec::new();
    for name in h.tensors.keys() {
        if let Ok(GgufMapping::Resident(canonical)) = map_gguf_name(name, family) {
            if !index.entries.contains_key(&canonical) {
                missing.push(format!("{name} -> {canonical}"));
            }
        }
    }
    missing.sort();
    missing.dedup_by_key(|m| shape_key(m));
    assert!(
        missing.is_empty(),
        "{} mapped names are absent from the install at {dir}: {:?}",
        missing.len(),
        missing.iter().take(10).collect::<Vec<_>>()
    );
    println!("(install cross-check passed against {dir})");
}

/// The strongest check this file can make. `arch_from_gguf` and
/// `peek_manifest_arch` start from completely different inputs -- GGUF
/// metadata written by llama.cpp's converter, versus a `manifest.json`
/// written by this port's own repack of an MLX checkpoint -- and must land
/// on the same `ArchConfig`. Anything that disagrees is either a mapping
/// bug or a real difference between the two conversions, and both are worth
/// failing on.
fn assert_arch_matches_install(h: &GgufHeader, env: &str) {
    let Ok(dir) = std::env::var(env) else {
        println!("(skipped arch cross-check: {env} is unset)");
        return;
    };
    let expected = peek_manifest_arch(std::path::Path::new(&dir)).expect("manifest arch");
    let derived = arch_from_gguf(h).expect("arch from GGUF metadata");
    assert_eq!(
        derived, expected,
        "ArchConfig derived from GGUF metadata disagrees with the install at {dir}"
    );
    println!("(arch cross-check passed against {dir})");
}

fn fetch(url: &str) -> GgufHeader {
    let source = HttpRangeSource::new(url);
    fetch_gguf_header(&source).expect("fetch GGUF header")
}

#[test]
#[ignore = "network: reads a few MB off a 27 GB remote checkpoint"]
fn reads_the_real_gemma4_q8_0_header() {
    let h = fetch(GEMMA4_Q8_0);
    report("gemma-4-26B-A4B-it-Q8_0.gguf", &h);
    assert_eq!(h.architecture(), Some("gemma4"));
    assert!(h.tensors.len() > 100);
    assert_every_name_maps(&h, ModelFamily::Gemma4);
    assert_mapped_names_exist_in_install(&h, ModelFamily::Gemma4, "TURBOSPARK_GEMMA4_INSTALL_DIR");
    assert_arch_matches_install(&h, "TURBOSPARK_GEMMA4_INSTALL_DIR");
}

#[test]
#[ignore = "network: reads a few MB off a 20 GB remote checkpoint"]
fn reads_the_real_qwen36_q4_k_m_header() {
    let h = fetch(QWEN36_Q4_K_M);
    report("Qwen3.6-35B-A3B-Q4_K_M.gguf", &h);
    // llama.cpp converts Qwen 3.6 under the 3.5 series' name.
    assert_eq!(h.architecture(), Some("qwen35moe"));
    assert!(h.tensors.len() > 100);
    assert_every_name_maps(&h, ModelFamily::Qwen36);
    assert_mapped_names_exist_in_install(&h, ModelFamily::Qwen36, "TURBOSPARK_QWEN36_INSTALL_DIR");
    assert_arch_matches_install(&h, "TURBOSPARK_QWEN36_INSTALL_DIR");
}
