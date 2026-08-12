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

/// [`fetch`] without the panic, for probes that SURVEY candidates rather
/// than assert about one known-good file. A header this parser refuses is
/// itself a Phase 0 finding.
fn try_fetch(url: &str) -> Result<GgufHeader, String> {
    let source = HttpRangeSource::new(url);
    fetch_gguf_header(&source).map_err(|e| format!("{e:?}"))
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

/// Phase 1's admission check for `qwen3moe`, off the same header the
/// granularity probe below reads: EVERY tensor name maps, and the derived
/// `ArchConfig` is the baseline this port declares.
///
/// **This is the only place a name-mapping hole can surface for this
/// family.** `crates/repack/tests/gguf_names.rs` pins the rows, but it can
/// only pin rows someone already wrote; a real converter's output is the
/// only input that can contain a name nobody thought of. There is no MLX
/// install of this model here, so unlike Gemma and Qwen 3.6 there is no
/// second, independently-produced side to cross-check the config against --
/// the assertion is against the hand-entered baseline, and the baseline was
/// entered FROM this header, so what this really proves is that the two have
/// not drifted apart since.
#[test]
#[ignore = "network: reads a header off a 17 GB remote checkpoint"]
fn qwen3moe_maps_every_name_and_derives_its_baseline() {
    let h = fetch(QWEN3_30B_A3B_Q4_K_M);
    assert_every_name_maps(&h, ModelFamily::Qwen3Moe);

    let derived = arch_from_gguf(&h).expect("arch from GGUF metadata");
    let baseline = model_io::qwen3_30b_a3b();
    assert_eq!(
        derived, baseline,
        "the derived ArchConfig has drifted from the qwen3_30b_a3b baseline"
    );

    // The two rows that make this NOT the `llama` architecture, on EVERY
    // layer rather than on layer 0. A per-head norm present on some layers
    // and absent on others would open, decode, and be wrong only sometimes.
    let layers = derived.num_layers as usize;
    for norm in ["attn_q_norm", "attn_k_norm"] {
        let present = (0..layers)
            .filter(|i| h.tensors.contains_key(&format!("blk.{i}.{norm}.weight")))
            .count();
        assert_eq!(present, layers, "{norm} on {present} of {layers} layers");
    }

    // Absences that are load-bearing: no shared expert (Qwen 3.6 has one and
    // this model does not), no dense FFN, no sliding window, untied head.
    for absent in [
        "blk.0.ffn_gate_shexp.weight",
        "blk.0.ffn_gate_inp_shexp.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    ] {
        assert!(!h.tensors.contains_key(absent), "unexpected {absent}");
    }
    assert!(
        !h.metadata.contains_key("qwen3moe.attention.sliding_window"),
        "a sliding-window key would make the all-full-attention mask a lie"
    );
    assert!(h.tensors.contains_key("output.weight"));
    assert!(!derived.tie_word_embeddings);

    // No new kernels: every block type in the file is one this port already
    // executes. That is the claim that made this family the cheap next one,
    // and it is one header read rather than an install.
    let mut types: Vec<&str> = h
        .tensors
        .values()
        .filter_map(|i| ggml_type_name(i.ggml_type))
        .collect();
    types.sort_unstable();
    types.dedup();
    println!("-- block types: {types:?}");
    for t in &types {
        let lower = t.to_lowercase();
        // F32 never reaches an install: `transcode_f32` narrows the norms to
        // BF16 and quantizes the router to INT8 affine before writing.
        assert!(
            lower == "f32" || model_io::EXECUTABLE_GGUF_TYPES.contains(&lower.as_str()),
            "{t} has no kernel here, so this checkpoint is not the cheap bring-up it looks like"
        );
    }
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

// ROADMAP Phase M2 step 2: the DENSE half of the `llama` architecture.
//
// M2's finding 3 says choosing the CHECKPOINT is part of step 1 rather
// than a detail, and the dense half has two gates that a header answers
// outright: does the file carry `rope_freqs.weight` (Llama 3.1's RoPE
// frequency scaling, which ships as a TENSOR and has neither an
// `ArchConfig` field nor a kernel input here), and is every block type in
// it one this port can already execute. Three real published files, a few
// MB read off each.
const MISTRAL_7B_V03_Q4_K_M: &str = "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf";
const LLAMA2_7B_CHAT_Q4_K_M: &str =
    "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf";
const LLAMA31_8B_Q4_K_M: &str = "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";

/// What a dense `llama` candidate would cost this port, off its header
/// alone: `(carries rope_freqs, block types with no kernel here, sized
/// bytes)`.
fn dense_candidate_gates(h: &GgufHeader) -> (bool, Vec<String>, u64) {
    let rope_freqs = h.tensors.contains_key("rope_freqs.weight");
    let mut missing: Vec<String> = h
        .tensors
        .values()
        .filter_map(|i| ggml_type_name(i.ggml_type))
        // The UNQUANTIZED widths never reach a kernel, so they must not be
        // read against `EXECUTABLE_GGUF_TYPES` -- that list is the set of
        // BLOCK types a dispatch can decode. `transcode_f32` narrows F32
        // norms to BF16 and INT8s the router at repack time, so nothing F32
        // is in an install at all (Gotcha 29). The first run of this probe
        // reported `["F32"]` against every candidate, which reads as three
        // blocked checkpoints and is three false positives.
        .filter(|t| !matches!(*t, "F32" | "F16" | "BF16"))
        .filter(|t| !model_io::EXECUTABLE_GGUF_TYPES.contains(&t.to_lowercase().as_str()))
        .map(|t| t.to_string())
        .collect();
    missing.sort();
    missing.dedup();
    let bytes = h
        .tensors
        .values()
        .filter_map(|i| {
            let elems: u64 = i.dims.iter().product();
            ggml_type_block(i.ggml_type).map(|(blk, sz)| elems / blk.max(1) * sz)
        })
        .sum();
    (rope_freqs, missing, bytes)
}

#[test]
#[ignore = "network: reads three real dense-llama GGUF headers (a few MB each)"]
fn scopes_the_dense_llama_candidates() {
    // GOTCHA 36'S MULTIPLICATION DOES NOT APPLY HERE, AND THAT IS THE
    // FINDING. `slots x layers x expert_stride` sizes a slot cache, and a
    // dense model has no routed experts to put in one. Routed experts are
    // the ONLY thing this engine streams; every other tensor is mapped AND
    // PINNED (Gotcha 19). So a dense model's working-set floor is its
    // WHOLE weight file, and the GiB column below IS that floor rather
    // than an estimate of it. No engineering moves it, which is why the
    // ROADMAP says this half ships without the memory ceiling and the
    // parity row has to say so per row.
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    let mut clean: Vec<&str> = Vec::new();
    for (label, url) in [
        ("Mistral-7B-Instruct-v0.3 Q4_K_M", MISTRAL_7B_V03_Q4_K_M),
        ("Llama-2-7B-Chat Q4_K_M", LLAMA2_7B_CHAT_Q4_K_M),
        ("Meta-Llama-3.1-8B-Instruct Q4_K_M", LLAMA31_8B_Q4_K_M),
    ] {
        // NOT `fetch`, which panics. A candidate this port cannot even
        // PARSE is a Phase 0 result and belongs in the table beside the
        // ones it can; killing the run on the second of three would hide
        // the third. TheBloke's 2023-era Llama 2 is the live case: it is
        // GGUF v2 and this parser accepts v3 only, which is the same shape
        // of problem as M2 finding 1 (that era's Mixtral carries the
        // pre-merge per-expert layout). Both say the 2023 conversions are
        // a separate ingest question, not a cheaper way in.
        let h = match try_fetch(url) {
            Ok(h) => h,
            Err(e) => {
                println!("\n=== {label}\n   UNREADABLE by this port's parser: {e}");
                continue;
            }
        };
        let experts = h
            .metadata
            .get("llama.expert_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let (rope_freqs, missing, bytes) = dense_candidate_gates(&h);
        println!(
            "\n=== {label}\n   architecture    {:?}\n   tensors         {}\n   \
             expert_count    {experts}\n   rope_freqs      {}\n   \
             types w/o kernel{}\n   RESIDENT FLOOR  {:.2} GiB (all of it pinned; \
             nothing streams)",
            h.architecture(),
            h.tensors.len(),
            if rope_freqs {
                "PRESENT -- needs a per-pair frequency input the rope kernels do not take"
            } else {
                "absent -- the existing scalar-theta rope kernel is enough"
            },
            if missing.is_empty() {
                " none".to_string()
            } else {
                format!(" {missing:?}")
            },
            gib(bytes),
        );

        assert_eq!(h.architecture(), Some("llama"), "{label}");
        assert!(experts <= 1, "{label} is not dense: expert_count {experts}");
        // Every dense candidate is well clear of the 1.6-2.2 GiB band the
        // MoE families hold. Asserted rather than described, so the claim
        // cannot rot into prose while the numbers move.
        assert!(
            gib(bytes) > 3.0,
            "{label} resident floor {:.2} GiB -- if a dense llama ever fits \
             the MoE band, this comment is wrong and needs rewriting",
            gib(bytes)
        );
        if !rope_freqs && missing.is_empty() {
            clean.push(label);
        }
    }

    // THE DECISION, stated as an assertion rather than left to the reader.
    // A candidate that needs neither a new kernel nor a new rope input is
    // the one the dense bring-up targets; Llama 3.1 is deliberately NOT it,
    // for the same reason Mixtral went before the dense half at all.
    println!("\n-- clears both gates with no new kernel work: {clean:?}");
    assert!(
        !clean.is_empty(),
        "no dense candidate avoids both new-kernel gates; the dense half \
         cannot land without rope frequency scaling after all"
    );
}

// ROADMAP M5 Phase 0: which MoE family comes after `llama` and `qwen3moe`.
//
// The registry's remaining planned rows in the order Phase M states them
// (`llama4`, `gpt-oss`, `deepseek2`), probed by header rather than argued
// about. Phase M's own correction is why this test exists at all: being MoE
// is necessary and not sufficient, and the multiplication that decides it
// (AGENTS.md Gotcha 36) is two metadata keys and a tensor size. Mixtral cost
// a 26 GB download to learn that; this costs a few MB per candidate.
//
// `deepseek2` is probed but is not a candidate: it needs the MLA kernels
// that DeepSeek-V4-Flash is also blocked on, which is a kernel FAMILY rather
// than a block type. It is here so the granularity column has its number
// beside the others rather than an assumption.
const GPT_OSS_20B_MXFP4: &str =
    "https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-MXFP4.gguf";
const GPT_OSS_120B_MXFP4: &str =
    "https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/main/gpt-oss-120b-MXFP4.gguf";
// Shard 1 of 2. A split GGUF puts the whole header in the first shard, so
// this stays a header read like every other row here.
const LLAMA4_SCOUT_Q4_K_M: &str = "https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF/resolve/main/Q4_K_M/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf";
const DEEPSEEK_V3_Q6_K: &str = "https://huggingface.co/unsloth/DeepSeek-V3-GGUF/resolve/main/DeepSeek-V3-Q6_K/DeepSeek-V3-Q6_K-00001-of-00012.gguf";

/// The size of ONE routed expert, read off the file rather than computed
/// from an assumed quantization.
///
/// Two reasons not to multiply a block constant by a shape the way
/// [`scopes_the_next_moe_candidate_by_expert_granularity`] does. The
/// candidates here are not all Q4_K (gpt-oss ships MXFP4 experts), so one
/// constant would be wrong for at least one row. And ROADMAP Phase S made
/// the stride PER LAYER, so the honest number is a maximum over layers, not
/// layer 0's -- padding every layer to the widest is exactly the 35% size
/// regression Phase S caught, and a Phase 0 estimate that reads layer 0
/// alone would under-report it.
///
/// Returns `(layer0_blob, max_layer_blob)` in bytes, both per expert.
fn routed_expert_blob(h: &GgufHeader, experts: u64) -> Option<(u64, u64)> {
    if experts == 0 {
        return None;
    }
    let mut per_layer: std::collections::BTreeMap<&str, u64> = std::collections::BTreeMap::new();
    for (name, info) in &h.tensors {
        if !name.ends_with("_exps.weight") && !name.ends_with("_exps") {
            continue;
        }
        // `blk.<N>.<role>` -- the layer index is the second dotted field.
        let mut parts = name.split('.');
        let (Some("blk"), Some(layer)) = (parts.next(), parts.next()) else {
            continue;
        };
        let elems: u64 = info.dims.iter().product();
        // An UNSIZED type contributes nothing and would silently shrink the
        // blob, which on an MXFP4 file is the whole expert table. Bail to
        // `None` instead: "cannot size this candidate" is a Phase 0 result
        // and a quietly small number is not (Gotcha 29's UNSIZED rule, one
        // layer out).
        let (blk, sz) = ggml_type_block(info.ggml_type)?;
        *per_layer.entry(layer).or_insert(0) += elems / blk.max(1) * sz;
    }
    if per_layer.is_empty() {
        return None;
    }
    let layer0 = *per_layer.get("0").unwrap_or(&0) / experts;
    let max = per_layer.values().copied().max().unwrap_or(0) / experts;
    Some((layer0, max))
}

/// The Phase 0 facts an MoE candidate is decided on, all off the header.
fn report_moe_candidate(label: &str, h: &GgufHeader) -> Option<u64> {
    let arch = h.architecture().unwrap_or("?").to_string();
    let key = |k: &str| -> Option<u64> {
        h.metadata
            .get(&format!("{arch}.{k}"))
            .and_then(turbospark_repack::GgufValue::as_u64)
    };
    let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);

    let experts = key("expert_count").unwrap_or(0);
    let top_k = key("expert_used_count").unwrap_or(0);
    let layers = key("block_count").unwrap_or(0);
    let hidden = key("embedding_length").unwrap_or(0);
    let q_heads = key("attention.head_count").unwrap_or(0);
    let kv_heads = key("attention.head_count_kv").unwrap_or(0);

    println!("\n=== {label}");
    println!("   architecture    {arch:?}");
    println!("   tensors         {}", h.tensors.len());
    println!("   layers          {layers}, hidden {hidden}, {q_heads} q heads over {kv_heads} kv");
    println!(
        "   experts         {experts} (top-{top_k}), expert ffn {:?}, dense ffn {:?}",
        key("expert_feed_forward_length"),
        key("feed_forward_length")
    );

    // GOTCHA 36, THE WHOLE REASON THIS PROBE RUNS BEFORE A DOWNLOAD.
    let blob = routed_expert_blob(h, experts);
    match blob {
        None if experts == 0 => println!("   granularity     n/a (dense)"),
        None => println!(
            "   granularity     UNSIZABLE -- a routed expert uses a block type with no \
             `ggml_type_block` row, so its bytes cannot be counted here at all"
        ),
        Some((l0, max)) => {
            println!(
                "   one expert      {:.1} MiB (layer 0) / {:.1} MiB (widest layer)",
                mib(l0),
                mib(max)
            );
            println!("   whole table     {:.1} GiB", gib(max * experts * layers));
            for slots in [8u64, 16, 32] {
                println!(
                    "   slot cache @{slots:>2}  {:.2} GiB",
                    gib(max * slots * layers)
                );
            }
        }
    }

    // The block-type gate, read the same way the dense probe reads it: the
    // unquantized widths are transcoded at repack and never reach a dispatch.
    let (_, missing, bytes) = dense_candidate_gates(h);
    println!(
        "   types w/o kernel{}",
        if missing.is_empty() {
            " none".to_string()
        } else {
            format!(" {missing:?}")
        }
    );
    println!("   sized bytes     {:.2} GiB", gib(bytes));

    // The layer-graph keys that name NEW DECODE WORK rather than new
    // kernels. Each of these is a flow question, and a flow question is the
    // expensive kind: M3 was cheap precisely because its answer to all of
    // them was "same graph as `llama`".
    let mut graph: Vec<String> = Vec::new();
    for k in [
        "attention.sliding_window",
        "attention.sliding_window_pattern",
        "rope.scaling.type",
        "rope.scaling.factor",
        "expert_shared_count",
        "expert_shared_feed_forward_length",
        "attention.key_length",
        "leading_dense_block_count",
        "expert_gating_func",
        "attention.q_lora_rank",
        "attention.kv_lora_rank",
    ] {
        if let Some(v) = h.metadata.get(&format!("{arch}.{k}")) {
            graph.push(format!("{k}={v:?}"));
        }
    }
    println!(
        "   graph keys      {}",
        if graph.is_empty() {
            "none of the ones that would mean a new flow".to_string()
        } else {
            graph.join(", ")
        }
    );

    // Tensors with no analogue in either name table. Printed as SHAPE KEYS
    // rather than counted, because the point is to read them.
    let mut novel: Vec<String> = h
        .tensors
        .keys()
        .map(|n| shape_key(n))
        .filter(|k| {
            !k.starts_with("blk.*.attn_")
                && !k.starts_with("blk.*.ffn_")
                && !matches!(
                    k.as_str(),
                    "token_embd.weight" | "output.weight" | "output_norm.weight"
                )
        })
        .collect();
    novel.sort();
    novel.dedup();
    println!("   novel tensors   {novel:?}");
    // `attn_sinks` hides inside the `blk.*.attn_` prefix above, and it is
    // the one gpt-oss tensor with no counterpart in any flow here, so it is
    // named explicitly rather than filtered away with the rest of attention.
    let sinks: Vec<String> = h
        .tensors
        .keys()
        .map(|n| shape_key(n))
        .filter(|k| k.contains("sink"))
        .collect();
    if !sinks.is_empty() {
        println!("   attention sinks {:?} -- no kernel input here", {
            let mut s = sinks;
            s.sort();
            s.dedup();
            s
        });
    }

    blob.map(|(_, max)| max)
}

/// ROADMAP M5 Phase 0. Four real headers, no download.
///
/// Asserts nothing about which candidate wins -- that is a judgement about
/// cost, and the phase records it in prose. What it DOES assert is the thing
/// a future edit could quietly break: that each candidate is still the
/// architecture its registry row claims, so a survey cannot rot into folklore
/// the way `arch_registry_network.rs` exists to prevent for the strings.
#[test]
#[ignore = "network: reads four real MoE GGUF headers (a few MB each)"]
fn scopes_phase_m5_moe_candidates() {
    let mut blobs: Vec<(&str, Option<u64>)> = Vec::new();
    for (label, url, expect_arch) in [
        ("gpt-oss-20b MXFP4", GPT_OSS_20B_MXFP4, "gpt-oss"),
        ("gpt-oss-120b MXFP4", GPT_OSS_120B_MXFP4, "gpt-oss"),
        (
            "Llama-4-Scout-17B-16E Q4_K_M",
            LLAMA4_SCOUT_Q4_K_M,
            "llama4",
        ),
        ("DeepSeek-V3 Q6_K", DEEPSEEK_V3_Q6_K, "deepseek2"),
    ] {
        // `try_fetch`, for the same reason the dense survey uses it: a
        // candidate this parser refuses is a row, not the end of the run.
        let h = match try_fetch(url) {
            Ok(h) => h,
            Err(e) => {
                println!("\n=== {label}\n   UNREADABLE by this port's parser: {e}");
                blobs.push((label, None));
                continue;
            }
        };
        assert_eq!(
            h.architecture(),
            Some(expect_arch),
            "{label} no longer reports the architecture its registry row was admitted on"
        );
        blobs.push((label, report_moe_candidate(label, &h)));
    }

    // The comparison the phase turns on, printed as one table so the
    // granularity finding is read rather than re-derived.
    println!("\n-- expert blob, against the two families that already run");
    println!("   Gemma 4 26B-A4B   ~3.2 MiB   (streams, 1.5 GiB slot cache at 16)");
    println!("   Qwen3-30B-A3B      2.5 MiB   (streams, 1.90 GiB at 16 over 48 layers)");
    println!("   Mixtral 8x7B     108.9 MiB   (does NOT stream here: 54.5 GiB at 16)");
    for (label, blob) in &blobs {
        match blob {
            Some(b) => println!("   {label:<30} {:.1} MiB", *b as f64 / (1024.0 * 1024.0)),
            None => println!("   {label:<30} unsized or unreadable"),
        }
    }

    // THE TWO DECISIONS, asserted rather than left in prose, for the same
    // reason `scopes_the_dense_llama_candidates` asserts its resident floor:
    // a number that only ever appears in a comment rots silently when the
    // published files move.
    let blob_of = |needle: &str| -> u64 {
        blobs
            .iter()
            .find(|(l, _)| l.contains(needle))
            .and_then(|(_, b)| *b)
            .unwrap_or_else(|| panic!("no sized expert blob for {needle}"))
    };
    const MIB: u64 = 1024 * 1024;
    assert!(
        blob_of("Llama-4-Scout") > 64 * MIB,
        "Llama 4 Scout's expert got small enough to stream here; the M5 refusal \
         clause in arch_registry.rs is now wrong and needs rewriting"
    );
    assert!(
        blob_of("gpt-oss-20b") < 16 * MIB,
        "gpt-oss stopped being the fine-grained survivor of this survey; M5 \
         picked its target on this number"
    );
}
