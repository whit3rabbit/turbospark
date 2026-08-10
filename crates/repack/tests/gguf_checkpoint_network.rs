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
    arch_from_gguf, fetch_gguf_header, ggml_type_block, ggml_type_name, map_gguf_name,
    peek_manifest_arch, GgufHeader, GgufMapping, HttpRangeSource,
};

const GEMMA4_Q8_0: &str = "https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-Q8_0.gguf";
const QWEN36_Q4_K_M: &str =
    "https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-Q4_K_M.gguf";

// ROADMAP Phase S scoping. `ggml-org` publishes only BF16/Q4_0/Q8_0 for
// these two models, so the sub-4-bit checkpoints Phase S would ingest come
// from `unsloth`, whose "UD" (Unsloth Dynamic) builds mix block types per
// tensor rather than applying one everywhere. WHICH types, and what share
// of the bytes each carries, is the entire scoping question for Phase S,
// and it is a header read rather than a 13 GB download.
const GEMMA4_UD_Q3_K_M: &str = "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf";
const QWEN36_UD_Q3_K_M: &str =
    "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf";

// The other side of the Phase S fork: a STATIC (no imatrix) Q3_K_M of the
// same base model. The name is the same and the block types are not, which
// is the whole reason both are probed rather than one being assumed to
// stand for "3-bit".
const GEMMA4_STATIC_Q3_K_M: &str = "https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it.Q3_K_M.gguf";

// ROADMAP Phase M2's first bring-up candidate. `docs/NEW_MODEL.md` Phase 0
// says decide the scope in writing BEFORE touching code, and for a GGUF
// source the header is where most of that writing comes from: which
// `ArchConfig` fields the file determines, which tensor names have no row
// yet, and what the layer graph looks like. A few MB off a 26 GB file.
// TWO conversions of the SAME model, because they do not agree on how the
// experts are laid out and the difference decides the repack walk. The 2023
// TheBloke build predates llama.cpp merging per-expert tensors into one 3-D
// `ffn_*_exps`; the 2025 mradermacher build is post-merge. Its quantization
// matters too: Q4_0 experts are a type this port REFUSES, Q4_K ones it runs.
const MIXTRAL_8X7B_LEGACY_Q4_0: &str = "https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_0.gguf";
const MIXTRAL_8X7B_Q4_K_M: &str = "https://huggingface.co/mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf";

// The candidate the ROADMAP Phase M2 granularity finding points at: a
// FINE-GRAINED MoE with the same dense-GQA-plus-MoE layer graph the `llama`
// flow already runs. Probed header-only, to do the multiplication that Phase
// M2 skipped BEFORE anyone downloads 17 GB.
const QWEN3_30B_A3B_Q4_K_M: &str =
    "https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf";

// The dense half of the SAME architecture string, probed beside it because
// the whole reason Phase M2 takes `llama` MoE-first is that one string
// covers both and only one of them keeps the memory ceiling.
const LLAMA31_8B_Q6_K: &str = "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf";

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
    // The histogram, in BYTES rather than tensor count, is what scopes
    // kernel work on a mixed file: a type carrying 0.1% of the weights and
    // one carrying 80% cost the same single row above and are not the same
    // decision. Routed experts are ~90% of an MoE's bytes, so the share
    // column is effectively "is this the expert type or a bystander".
    // A type with no `ggml_type_block` row must print as UNSIZED, never as
    // zero bytes. Zero would rank it last in the share column, which is the
    // exact opposite of the truth: an unlisted type is one this port has
    // never handled, and on a mixed file the unhandled type is likely to be
    // the routed experts, i.e. ~90% of the weights. This bit once already
    // (the first run of this probe read "Q8_0 75.6%" on a file whose
    // experts are IQ3_XXS and were being counted as 0.000 GiB).
    let mut by_type: std::collections::BTreeMap<&str, (usize, Option<u64>)> =
        std::collections::BTreeMap::new();
    for info in h.tensors.values() {
        let name = ggml_type_name(info.ggml_type).unwrap_or("?");
        let elems: u64 = info.dims.iter().product();
        let bytes = ggml_type_block(info.ggml_type).map(|(blk, sz)| elems / blk.max(1) * sz);
        let e = by_type.entry(name).or_insert((0, Some(0)));
        e.0 += 1;
        e.1 = match (e.1, bytes) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };
    }
    let sized: u64 = by_type.values().filter_map(|(_, b)| *b).sum();
    let unsized_types = by_type.values().filter(|(_, b)| b.is_none()).count();
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    println!("-- ggml type histogram (tensors, bytes, share of SIZED bytes)");
    for (name, (count, bytes)) in &by_type {
        match bytes {
            Some(b) => println!(
                "   {name:<8} {count:>4} tensors  {:>9.3} GiB  {:>5.1}%",
                gib(*b),
                *b as f64 * 100.0 / sized.max(1) as f64
            ),
            None => println!(
                "   {name:<8} {count:>4} tensors     UNSIZED  (no ggml_type_block row: \
                 unhandled by this port, and NOT zero)"
            ),
        }
    }
    println!(
        "   {:<8} {:>4} tensors  {:>9.3} GiB sized{}",
        "TOTAL",
        h.tensors.len(),
        gib(sized),
        if unsized_types > 0 {
            format!(", {unsized_types} type(s) UNSIZED and excluded")
        } else {
            String::new()
        }
    );

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

/// ROADMAP Phase S, the scoping step. Deliberately asserts almost nothing:
/// its output is the type histogram, which says which kernels a 3-bit
/// install would need and how much of the model each one carries. The one
/// thing it DOES assert is that the names still map, because an unsloth
/// build is a different converter run from the `ggml-org` one and a
/// name-table hole there would be found here or not at all.
#[test]
#[ignore = "network: reads a few MB off a 13 GB remote checkpoint"]
fn scopes_phase_s_from_the_gemma4_ud_q3_k_m_header() {
    let h = fetch(GEMMA4_UD_Q3_K_M);
    report("gemma-4-26B-A4B-it-UD-Q3_K_M.gguf", &h);
    assert_eq!(h.architecture(), Some("gemma4"));
    assert_every_name_maps(&h, ModelFamily::Gemma4);
    assert_arch_matches_install(&h, "TURBOSPARK_GEMMA4_INSTALL_DIR");
}

/// The static counterpart to the UD probe above, and the one that decides
/// what Phase S costs. Two files both called "Q3_K_M" of the same base
/// model do not carry the same block types: the imatrix build spends its
/// expert bytes on codebook types, the static one on K-quants this port
/// already has most of. Run both before scoping any kernel work.
#[test]
#[ignore = "network: reads a few MB off a 12 GB remote checkpoint"]
fn scopes_phase_s_from_the_gemma4_static_q3_k_m_header() {
    let h = fetch(GEMMA4_STATIC_Q3_K_M);
    report("gemma-4-26B-A4B-it.Q3_K_M.gguf (static, mradermacher)", &h);
    assert_eq!(h.architecture(), Some("gemma4"));
    assert_every_name_maps(&h, ModelFamily::Gemma4);
    assert_arch_matches_install(&h, "TURBOSPARK_GEMMA4_INSTALL_DIR");
}

/// The Qwen sibling. Worth probing separately rather than assuming it
/// mirrors Gemma: the two families already differ in whether the routed
/// experts are fused, and a UD mix is chosen per tensor.
#[test]
#[ignore = "network: reads a few MB off a 16 GB remote checkpoint"]
fn scopes_phase_s_from_the_qwen36_ud_q3_k_m_header() {
    let h = fetch(QWEN36_UD_Q3_K_M);
    report("Qwen3.6-35B-A3B-UD-Q3_K_M.gguf", &h);
    assert_eq!(h.architecture(), Some("qwen35moe"));
    assert_every_name_maps(&h, ModelFamily::Qwen36);
    assert_arch_matches_install(&h, "TURBOSPARK_QWEN36_INSTALL_DIR");
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

/// ROADMAP Phase M2, `docs/NEW_MODEL.md` Phase 0: scope the first bring-up
/// candidate from its header, before any code.
///
/// Prints, rather than asserts, most of what it finds -- Phase 0's output is
/// a WRITTEN scope and this is the instrument for it. The three assertions
/// are the claims the phase's ORDERING rests on, so they are the ones that
/// must fail loudly if a republished file ever moves: Mixtral is the `llama`
/// architecture, its MoE is expressed in metadata, and this port classes
/// that architecture as recognized-but-unported rather than runnable.
///
/// The unmapped-name list is the deliverable for `gguf_names.rs`: there is
/// no `ModelFamily` for `llama` yet, so `assert_every_name_maps` cannot be
/// called here and the shape rows in the report above are the inventory.
#[test]
#[ignore = "network: reads a few MB off two ~26 GB remote checkpoints"]
fn scopes_phase_m2_from_the_mixtral_header() {
    let h = fetch(MIXTRAL_8X7B_Q4_K_M);
    report(
        "Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf (2025 conversion)",
        &h,
    );

    // The same model converted in 2023, kept as the evidence that ONE
    // architecture string has two on-disk expert layouts. A walk that
    // assumes either one must refuse the other by name rather than
    // misreading it: 256 separate `ffn_down.<e>.weight` tensors and one
    // 3-D `ffn_down_exps` hold the same weights in the same order, so a
    // mistake here is silent.
    let legacy = fetch(MIXTRAL_8X7B_LEGACY_Q4_0);
    report(
        "mixtral-8x7b-instruct-v0.1.Q4_0.gguf (2023 conversion, pre-merge)",
        &legacy,
    );
    let merged = |hh: &GgufHeader| {
        hh.tensors
            .keys()
            .any(|n| n.ends_with("ffn_down_exps.weight"))
    };
    println!(
        "\n-- expert layout: 2025 merged={} ({} tensors), 2023 merged={} ({} tensors)",
        merged(&h),
        h.tensors.len(),
        merged(&legacy),
        legacy.tensors.len()
    );

    assert_eq!(h.architecture(), Some("llama"));
    assert_eq!(legacy.architecture(), Some("llama"));
    assert!(
        matches!(
            turbospark_repack::gguf_arch_support("llama"),
            Some(turbospark_repack::ArchSupport::Supported(_))
        ),
        "llama gained a decode flow in ROADMAP Phase M2"
    );
    let experts = h
        .metadata
        .get("llama.expert_count")
        .and_then(|v| v.as_u64())
        .expect("Mixtral publishes an expert count");
    let used = h
        .metadata
        .get("llama.expert_used_count")
        .and_then(|v| v.as_u64())
        .expect("Mixtral publishes a top-k");
    println!("\n-- MoE shape: {experts} experts, top-{used}");
    assert!(experts > 1 && used >= 1 && used < experts);
    assert!(
        merged(&h) && !merged(&legacy),
        "the two conversions are supposed to differ in expert layout; if they \
         no longer do, the walk's refusal of the legacy shape is dead code"
    );
    // Neither file publishes a separate expert FFN width, which both other
    // families do and `arch_from_gguf` currently REQUIRES. It has to fall
    // back to `feed_forward_length` for this architecture.
    assert!(
        !h.metadata.contains_key("llama.expert_feed_forward_length"),
        "if this key appeared, arch_from_gguf needs no fallback after all"
    );
}

/// The dense half of the same architecture string, for the contrast that
/// decides Phase M2's order: same `general.architecture`, no expert count,
/// so every byte of it would be resident (AGENTS.md Gotcha 19) and the
/// memory ceiling the engine is built around does not apply.
#[test]
#[ignore = "network: reads a few MB off a 7 GB remote checkpoint"]
fn scopes_phase_m2_from_the_dense_llama_header() {
    let h = fetch(LLAMA31_8B_Q6_K);
    report("Meta-Llama-3.1-8B-Instruct-Q6_K.gguf", &h);

    assert_eq!(h.architecture(), Some("llama"));
    let experts = h
        .metadata
        .get("llama.expert_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    assert!(
        experts <= 1,
        "a dense Llama must not carry a routed expert count, got {experts}"
    );

    // The RoPE frequency scaling Llama 3.1 added ships as a TENSOR, not as
    // metadata, and there is no `ArchConfig` field or kernel input for it
    // here. Mixtral has none, which is part of why the MoE half goes first.
    assert!(
        h.tensors.contains_key("rope_freqs.weight"),
        "expected Llama 3.1's rope_freqs tensor"
    );
}

/// Diagnostic for the ROADMAP Phase M2 install failure: both streamed runs
/// died writing layer 21 with "error decoding response body", which is
/// DETERMINISTIC and therefore not CDN flakiness. The first question is
/// whether the byte ranges the walk asks for are inside the file at all.
///
/// Prints the tensor whose data ends furthest into the file and compares it
/// with the published length, plus the span of layer 21 specifically.
#[test]
#[ignore = "network: reads a header off a 26 GB remote checkpoint"]
fn checks_mixtral_tensor_ranges_against_the_file_length() {
    let h = fetch(MIXTRAL_8X7B_Q4_K_M);
    let file_len: u64 = 28_448_468_384;

    let end_of = |info: &turbospark_repack::GgufTensorInfo| -> u64 {
        let elems: u64 = info.dims.iter().product();
        let bytes = ggml_type_block(info.ggml_type)
            .map(|(blk, sz)| elems / blk.max(1) * sz)
            .unwrap_or(0);
        h.data_region_start + info.offset + bytes
    };

    let mut furthest = ("", 0u64);
    for (name, info) in &h.tensors {
        let end = end_of(info);
        if end > furthest.1 {
            furthest = (name.as_str(), end);
        }
    }
    println!("file length      {file_len}");
    println!("furthest tensor  {} ends at {}", furthest.0, furthest.1);
    println!(
        "slack            {} bytes",
        file_len as i64 - furthest.1 as i64
    );

    for name in [
        "blk.21.ffn_gate_exps.weight",
        "blk.21.ffn_up_exps.weight",
        "blk.21.ffn_down_exps.weight",
    ] {
        if let Some(info) = h.tensors.get(name) {
            println!(
                "{name}: type {} dims {:?} -> [{}, {})",
                ggml_type_name(info.ggml_type).unwrap_or("?"),
                info.dims,
                h.data_region_start + info.offset,
                end_of(info)
            );
        }
    }
    assert!(
        furthest.1 <= file_len,
        "the walk would read past EOF: {} ends at {} in a {file_len}-byte file",
        furthest.0,
        furthest.1
    );
}

/// Phase 0 for whatever MoE family comes after `llama`: the expert
/// GRANULARITY multiplication, off the header, before any download
/// (AGENTS.md Gotcha 36).
///
/// Prints the blob size and the slot-cache working set rather than asserting
/// a threshold, because what counts as "fits" depends on the machine. What IS
/// asserted is the comparison that decides it: this candidate's expert must be
/// far smaller than Mixtral's 108.9 MiB, or it is the same dead end again.
#[test]
#[ignore = "network: reads a header off a 17 GB remote checkpoint"]
fn scopes_the_next_moe_candidate_by_expert_granularity() {
    let h = fetch(QWEN3_30B_A3B_Q4_K_M);
    report("Qwen3-30B-A3B-Q4_K_M.gguf", &h);
    let arch = h.architecture().expect("architecture");
    assert_eq!(arch, "qwen3moe");

    let u64_at = |key: &str| -> u64 {
        h.metadata
            .get(&format!("{arch}.{key}"))
            .and_then(turbospark_repack::GgufValue::as_u64)
            .unwrap_or_else(|| panic!("missing {arch}.{key}"))
    };
    let experts = u64_at("expert_count");
    let top_k = u64_at("expert_used_count");
    let layers = u64_at("block_count");
    let hidden = u64_at("embedding_length");
    let expert_ff = u64_at("expert_feed_forward_length");

    // Q4_K: 144 bytes per 256 elements, over gate + up + down.
    let blob = 3 * expert_ff * hidden * 144 / 256;
    let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    println!("\n-- granularity: {experts} experts (top-{top_k}), {layers} layers");
    println!("   expert ffn {expert_ff} x hidden {hidden}");
    println!("   one expert blob   {:.1} MiB", mib(blob));
    println!(
        "   whole table       {:.1} GiB",
        gib(blob * experts * layers)
    );
    for slots in [8u64, 16, 32] {
        println!(
            "   slot cache at {slots:>2}: {:.2} GiB",
            gib(blob * slots * layers)
        );
    }

    const MIXTRAL_BLOB: u64 = 108 * 1024 * 1024;
    assert!(
        blob < MIXTRAL_BLOB / 8,
        "expert blob {:.1} MiB is not fine-grained; this is Mixtral's problem again",
        mib(blob)
    );
}
