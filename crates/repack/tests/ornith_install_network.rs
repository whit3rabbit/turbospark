//! Streams both Ornith-1.5 GGUFs into `.gturbo` installs.
//!
//! ```sh
//! TURBOSPARK_ORNITH9B_INSTALL_DIR=~/models/ornith9b.gturbo \
//!   cargo test -p turbospark-repack --test ornith_install_network --release -- --ignored --nocapture installs_the_real_ornith_9b
//!
//! TURBOSPARK_ORNITH35B_GGUF_INSTALL_DIR=~/models/ornith35b-gguf.gturbo \
//!   cargo test -p turbospark-repack --test ornith_install_network --release -- --ignored --nocapture installs_the_real_ornith_35b
//! ```
//!
//! Neither checkpoint is materialized locally: `write_gguf_install_streamed`
//! reads it a layer at a time through `HttpRangeSource`, so the only disk
//! either needs is its install (~9 GB and ~35 GB).
//!
//! **Q8_0, NOT `Q4_K_M`, AND THAT IS FORCED RATHER THAN PREFERRED.** The
//! V-head de-interleave for `linear_attn.out_proj` runs on COLUMNS, and a
//! 128-column head has to be a whole number of blocks (`crates/repack`
//! Gotcha 7). Q8_0's block is 32 elements and 128 is four of them; Q4_K's and
//! Q6_K's are 256, so a head is HALF a superblock and the permutation cuts
//! through the packed 6-bit scale header. `Q4_K_M` puts `ssm_out` at Q4_K on
//! every layer of both models, so both are refused by name in seconds --
//! `ornith_gguf_network.rs` reads that type off the real header, which is
//! where the evidence for this paragraph lives.
//!
//! Nothing is given up by taking Q8_0. Both published Q8_0 files are Q8_0 and
//! F32 THROUGHOUT (no K-quant anywhere), so every block type already has the
//! kernels its slot needs and neither install needs a line of new code.
//!
//! **RUN THE 9B FIRST.** It is a quarter of the wall clock and it exercises
//! the path with no prior coverage at all -- `qwen35` dense, whose name table
//! and `SUPPORTED_GGUF` row are both new, and whose FFN is the one part of
//! the layer graph the MoE half does not share. That is ROADMAP M4's
//! `head_dim` lesson (the cheap model is what caught a bug the three
//! expensive ones passed over) applied before the expensive stream rather
//! than after it.
//!
//! **THE SIDECAR LISTS ARE PER CHECKPOINT AND DELIBERATELY NOT SHARED.** The
//! 35B ships `generation_config.json` and the 9B does not, so a list copied
//! from one to the other 404s AFTER the stream has written a perfectly good
//! install -- which is exactly what a copied list cost `qwen38` (AGENTS.md
//! Gotcha 47). Read each repo's own file list.

use std::path::PathBuf;

use turbospark_repack::{fetch_gguf_header, write_gguf_install_streamed, HttpRangeSource};

/// Pinned by revision. `curl -sI` on a `resolve/main` URL returns
/// `x-repo-commit`, which costs no bytes (repack Gotcha 5).
const MOE_Q8_0: &str = "https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF/resolve/5ae357e3eaf951ae221e8d784c71a8a3cdb6aa5f/Ornith-1.5-35B-Q8_0.gguf";
const DENSE_Q8_0: &str = "https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF/resolve/0677a38f331a214c4e5e7bd07ecab04c14ac52f1/Ornith-1.5-9B-Q8_0.gguf";

/// The published byte sizes, from `x-linked-size` on a HEAD of the URLs above.
///
/// **`x-linked-etag` on the same HEAD is the file's SHA-256** and is pinned
/// beside them (AGENTS.md Gotcha 46 records finding that and declining to use
/// it; this is the occasion, because a size alone cannot tell a re-quantized
/// file of the same shape from the one these numbers were measured against).
/// Neither costs a byte, and the walk reads the header before it reads a
/// weight, so a moved file is caught in seconds rather than at some offset.
const MOE_BYTES: u64 = 37_802_149_120;
const DENSE_BYTES: u64 = 9_527_501_248;
const MOE_SHA256: &str = "854cf83f80cd37a061ed86df1fa7201162e4e1fb820b91068cc12a11d2746c9e";
const DENSE_SHA256: &str = "6874eeb25c71081dc8f0bbe88f3ebb786312447132745371cd980bce95d259b9";

const MOE_MODEL_ID: &str = "ornith-ai/Ornith-1.5-35B-A3B-GGUF";
const DENSE_MODEL_ID: &str = "ornith-ai/Ornith-1.5-9B-GGUF";

/// A GGUF carries its tokenizer as llama.cpp metadata; this port loads an HF
/// `tokenizer.json`. Both sidecar bases are the SAFETENSORS repo the GGUF was
/// converted from, pinned by revision.
const MOE_SIDECAR_BASE: &str = "https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/resolve/fbb995a79eedd569a5edc5f2af9644c0fa1124fc";
const DENSE_SIDECAR_BASE: &str = "https://huggingface.co/ornith-ai/Ornith-1.5-9B/resolve/98db59be66b580b0395b3dc8237b32eefcdfec22";

/// The 35B's. It publishes a `generation_config.json`; the 9B does not.
const MOE_SIDECARS: &[&str] = &[
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
];

/// The 9B's. NO `generation_config.json` -- its stop set comes from the
/// dialect and `tokenizer_config.json` alone.
const DENSE_SIDECARS: &[&str] = &[
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
];

fn get(url: &str) -> Vec<u8> {
    let response = reqwest::blocking::Client::builder()
        .timeout(None)
        .build()
        .expect("client")
        .get(url)
        .send()
        .unwrap_or_else(|e| panic!("GET {url}: {e}"));
    assert!(
        response.status().is_success(),
        "GET {url}: HTTP {}",
        response.status()
    );
    response.bytes().expect("body").to_vec()
}

/// Asserts the published file is still the one these constants were measured
/// against, from HEADERS alone: `x-linked-size` is its byte length and
/// `x-linked-etag` is its SHA-256.
///
/// Run BEFORE the walk. The revision pin already prevents the repository from
/// moving under us; what this catches is the case a revision cannot, which is
/// a file re-uploaded at the same path with the same shape and different
/// bytes. Costs one round trip against a ~35 GB stream.
fn assert_published_file(url: &str, bytes: u64, sha256: &str) {
    let response = reqwest::blocking::Client::builder()
        .timeout(None)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("client")
        .head(url)
        .send()
        .unwrap_or_else(|e| panic!("HEAD {url}: {e}"));
    let header = |name: &str| {
        response
            .headers()
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(|v| v.trim_matches('"').to_string())
            .unwrap_or_else(|| panic!("HEAD {url}: no {name} header"))
    };
    assert_eq!(
        header("x-linked-size").parse::<u64>().expect("size"),
        bytes,
        "{url}: published size moved"
    );
    assert_eq!(
        header("x-linked-etag"),
        sha256,
        "{url}: published bytes moved"
    );
    eprintln!("pinned: {bytes} bytes, sha256 {sha256}");
}

fn install_dir(env: &str, fallback: &str) -> PathBuf {
    let dir = match std::env::var_os(env) {
        Some(dir) => PathBuf::from(dir),
        None => std::env::temp_dir().join(format!("{fallback}-{}", std::process::id())),
    };
    std::fs::create_dir_all(&dir).expect("create install dir");
    dir
}

fn fetch_sidecars(dir: &std::path::Path, base: &str, names: &[&str]) {
    for name in names {
        std::fs::write(dir.join(name), get(&format!("{base}/{name}")))
            .unwrap_or_else(|e| panic!("write sidecar {name}: {e}"));
    }
}

/// The DENSE half, and the first `qwen35` GGUF this port has ever installed.
#[test]
#[ignore = "network: streams the real 9.5 GB Ornith-1.5-9B Q8_0 GGUF and writes a ~9 GB install"]
fn installs_the_real_ornith_9b() {
    assert_published_file(DENSE_Q8_0, DENSE_BYTES, DENSE_SHA256);
    let source = HttpRangeSource::new(DENSE_Q8_0);
    let header = fetch_gguf_header(&source).expect("fetch GGUF header");
    // The converter's name, not the family's (AGENTS.md Gotcha 29). Note it
    // is a PREFIX of the MoE half's `qwen35moe`, which is why the registry
    // lookup is exact equality.
    assert_eq!(header.architecture(), Some("qwen35"));
    eprintln!(
        "header: {} tensors, alignment {}, data region starts at {}",
        header.tensors.len(),
        header.alignment,
        header.data_region_start
    );

    let dir = install_dir("TURBOSPARK_ORNITH9B_INSTALL_DIR", "turbospark-ornith9b");
    eprintln!("installing to {}", dir.display());
    let arch = write_gguf_install_streamed(&dir, &header, &source, DENSE_MODEL_ID, |stage| {
        eprintln!("[repack] {stage}");
    })
    .expect("streamed GGUF install");

    assert_eq!(arch.family, model_io::ModelFamily::QwenGdnDense);
    assert_eq!(arch.num_layers, 32);
    assert_eq!(arch.hidden_size, 4096);
    assert_eq!(arch.intermediate_size, 12288);
    assert_eq!(arch.num_experts, 0);

    fetch_sidecars(&dir, DENSE_SIDECAR_BASE, DENSE_SIDECARS);

    model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect("manifest validates: q8_0 is an executable block type");
    let resident =
        model_io::load_resident_index(&dir.join("model_weights.bin")).expect("resident index");

    // A DENSE install has NO packed experts at all, which is the thing this
    // family's MoE half cannot check. `crates/repack` Gotcha 8 records what
    // the walk used to get wrong when `plan.routed` is empty.
    let packed = dir.join("packed_experts");
    let layer_files = std::fs::read_dir(&packed)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .filter(|e| e.file_name().to_string_lossy().starts_with("layer_"))
                .count()
        })
        .unwrap_or(0);
    assert_eq!(layer_files, 0, "a dense install writes no expert blobs");

    // The dense FFN triple, which is the only part of the layer graph the
    // MoE half does not share, on a FULL-attention layer and a LINEAR one.
    for name in [
        "language_model.model.layers.3.mlp.gate_proj.weight",
        "language_model.model.layers.3.mlp.up_proj.weight",
        "language_model.model.layers.3.mlp.down_proj.weight",
        "language_model.model.layers.0.mlp.gate_proj.weight",
    ] {
        assert!(
            resident.entries.contains_key(name),
            "{name} missing from the resident index"
        );
    }
    // The gated-DeltaNet family on a linear layer, including the two tensors
    // that carry no `.weight` suffix on either side (Gotcha 26).
    for name in [
        "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
        "language_model.model.layers.0.linear_attn.in_proj_z.weight",
        "language_model.model.layers.0.linear_attn.out_proj.weight",
        "language_model.model.layers.0.linear_attn.A_log",
        "language_model.model.layers.0.linear_attn.dt_bias",
    ] {
        assert!(
            resident.entries.contains_key(name),
            "{name} missing from the resident index"
        );
    }
    // Layer 0 is LINEAR, so it has no self-attention projections at all.
    assert!(
        !resident
            .entries
            .contains_key("language_model.model.layers.0.self_attn.q_proj.weight"),
        "layer 0 is a gated-DeltaNet layer and must carry no q_proj"
    );

    eprintln!("installed {} resident tensors", resident.entries.len());
}

/// The MoE half, and the first install whose source declares a
/// multi-token-prediction block.
#[test]
#[ignore = "network: streams the real 37.8 GB Ornith-1.5-35B-A3B Q8_0 GGUF and writes a ~35 GB install"]
fn installs_the_real_ornith_35b() {
    assert_published_file(MOE_Q8_0, MOE_BYTES, MOE_SHA256);
    let source = HttpRangeSource::new(MOE_Q8_0);
    let header = fetch_gguf_header(&source).expect("fetch GGUF header");
    assert_eq!(header.architecture(), Some("qwen35moe"));
    eprintln!(
        "header: {} tensors, alignment {}, data region starts at {}",
        header.tensors.len(),
        header.alignment,
        header.data_region_start
    );

    let dir = install_dir(
        "TURBOSPARK_ORNITH35B_GGUF_INSTALL_DIR",
        "turbospark-ornith35b-gguf",
    );
    eprintln!("installing to {}", dir.display());
    let arch = write_gguf_install_streamed(&dir, &header, &source, MOE_MODEL_ID, |stage| {
        eprintln!("[repack] {stage}");
    })
    .expect("streamed GGUF install");

    // The headline again, now from the WALK rather than from a header read:
    // Ornith-1.5-35B-A3B is Qwen 3.6's architecture retrained.
    assert_eq!(arch, model_io::qwen_gdn_moe_35b_a3b());

    fetch_sidecars(&dir, MOE_SIDECAR_BASE, MOE_SIDECARS);

    model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect("manifest validates: q8_0 is an executable block type");
    let resident =
        model_io::load_resident_index(&dir.join("model_weights.bin")).expect("resident index");

    // **THE MTP BLOCK MUST NOT HAVE BECOME A 41st LAYER.** `block_count` is
    // 41 and the trunk is 40; the head is skipped by `plan::classify`, so
    // there are exactly 40 expert-blob files and no `layers.40` anywhere in
    // the resident index. Getting this wrong writes an install that loads and
    // decodes with a phantom layer, which is the silent failure the whole
    // `nextn_predict_layers` subtraction exists to prevent.
    let packed = dir.join("packed_experts");
    let layer_files = std::fs::read_dir(&packed)
        .expect("packed_experts")
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().starts_with("layer_"))
        .count();
    assert_eq!(layer_files, 40, "40 trunk layers of experts, not 41");
    assert!(
        !resident.entries.keys().any(|k| k.contains("layers.40")),
        "the MTP block must not reach the resident index as a trunk layer"
    );
    assert!(
        !resident.entries.keys().any(|k| k.contains("mtp")),
        "this walk skips the head; it is ingested from the safetensors \
         checkpoint instead (docs/MTP.md)"
    );

    // The shared expert and its sigmoid gate, which the dense half has not.
    for name in [
        "language_model.model.layers.3.mlp.shared_expert.gate_proj.weight",
        "language_model.model.layers.3.mlp.shared_expert_gate.weight",
        "language_model.model.layers.3.mlp.gate.weight",
    ] {
        assert!(
            resident.entries.contains_key(name),
            "{name} missing from the resident index"
        );
    }

    eprintln!("installed {} resident tensors", resident.entries.len());
}
