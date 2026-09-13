//! Real-artifact gates for the Qwen2/Qwen2.5 bring-up.
//!
//! These tests are ignored because one streams the pinned 4.28 GB MLX
//! checkpoint and the other reads a remote GGUF header. The synthetic tests
//! cover the same control flow without making the default suite network- or
//! disk-heavy.
//!
//! ```text
//! TURBOSPARK_QWEN2_INSTALL_DIR=~/models/qwen25-7b-4bit.gturbo \
//!   cargo test -p turbospark-repack --test qwen2_checkpoint_network \
//!   --release -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use turbospark_repack::{
    arch_from_gguf, canonicalize_qwen2_header, fetch_gguf_header, fetch_safetensors_header,
    parse_gemma4_quantization, parse_qwen2_config, write_qwen2_dense_install_streamed,
    Gemma4Shards, HttpRangeSource,
};

const GGUF_URL: &str = "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/74ef91efd0899612867d6bb080ce5a2788ef6aa1/qwen2.5-7b-instruct-q3_k_m.gguf";
const MLX_BASE: &str = "https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit/resolve/c8e9187488f846965507bfc2b3957d59fd0d5a27";
const MODEL_ID: &str = "mlx-community/Qwen2.5-7B-Instruct-4bit";

fn get(path: &str) -> Vec<u8> {
    let url = format!("{MLX_BASE}/{path}");
    let response = reqwest::blocking::Client::builder()
        .timeout(None)
        .build()
        .expect("client")
        .get(&url)
        .send()
        .unwrap_or_else(|e| panic!("GET {url}: {e}"));
    assert!(
        response.status().is_success(),
        "GET {url}: {}",
        response.status()
    );
    response.bytes().expect("body").to_vec()
}

fn install_dir() -> PathBuf {
    let dir = std::env::var_os("TURBOSPARK_QWEN2_INSTALL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("turbospark-qwen2-real-{}", std::process::id()))
        });
    std::fs::create_dir_all(&dir).expect("create install directory");
    dir
}

/// Header-only proof for the exact GGUF named in the roadmap. It records the
/// current quantization boundary explicitly: `qwen2` is recognized, but this
/// port parses Q3_K and does not yet execute it.
#[test]
#[ignore = "network: reads the pinned Qwen2.5 Q3_K_M GGUF header"]
fn the_pinned_qwen25_q3_k_m_header_resolves_and_exposes_the_quant_boundary() {
    let source = HttpRangeSource::new(GGUF_URL);
    let header = fetch_gguf_header(&source).expect("fetch Qwen2.5 GGUF header");
    assert_eq!(header.architecture(), Some("qwen2"));
    let arch = arch_from_gguf(&header).expect("Qwen2.5 GGUF metadata parses");
    assert_eq!(arch, model_io::qwen2_5_7b());
    for name in [
        "blk.0.attn_q.bias",
        "blk.0.attn_k.bias",
        "blk.0.attn_v.bias",
    ] {
        assert!(
            header.tensors.contains_key(name),
            "missing Qwen2 tensor {name}"
        );
    }
    assert!(
        header.tensors.values().any(|t| t.ggml_type == 11),
        "the Q3_K_M artifact should carry at least one Q3_K tensor"
    );
    assert!(
        !model_io::EXECUTABLE_GGUF_TYPES.contains(&"q3_k"),
        "the test should stay red if Q3_K is accidentally claimed executable"
    );
}

/// Full MLX/HF intake gate for the pinned 4-bit Qwen2.5 checkpoint.
#[test]
#[ignore = "network: streams the pinned 4.28 GB Qwen2.5 MLX checkpoint"]
fn repacks_the_pinned_qwen25_mlx_checkpoint() {
    let config = String::from_utf8(get("config.json")).expect("config utf8");
    let arch = parse_qwen2_config(&config).expect("Qwen2 config parses");
    assert_eq!(arch, model_io::qwen2_5_7b());
    let quant = parse_gemma4_quantization(&config).expect("quantization parses");
    assert_eq!(quant.default_bits, 4);
    assert_eq!(quant.group_size, 64);

    let source = HttpRangeSource::new(format!("{MLX_BASE}/model.safetensors"));
    let mut header = fetch_safetensors_header(&source).expect("fetch safetensors header");
    let source_names = header
        .tensors
        .keys()
        .filter(|name| name.starts_with("model.") || name.starts_with("lm_head."))
        .count();
    assert!(
        source_names > 0,
        "MLX Qwen2 checkpoint uses the source namespace"
    );
    canonicalize_qwen2_header(&mut header).expect("canonicalize MLX Qwen2 names");
    assert!(!header.tensors.keys().any(|name| name.starts_with("model.")));

    let shards = Gemma4Shards::single(&header, &source);
    let dir = install_dir();
    write_qwen2_dense_install_streamed(&dir, &arch, MODEL_ID, &shards, &quant, |stage| {
        eprintln!("[repack] {stage}");
    })
    .expect("streamed Qwen2 install");

    model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect("Qwen2 manifest validates");
    let index = model_io::load_resident_index(&dir.join("model_weights.bin"))
        .expect("Qwen2 resident index loads");
    for layer in [0, 27] {
        for projection in ["q", "k", "v"] {
            let name =
                format!("language_model.model.layers.{layer}.self_attn.{projection}_proj.bias");
            assert_eq!(index.entries[&name].dtype, 1, "{name} is BF16");
        }
    }
    assert!(
        index
            .entries
            .keys()
            .all(|name| !name.starts_with("model.") && !name.starts_with("lm_head.")),
        "source MLX namespaces must not leak into the install"
    );
}

/// Cheap companion to the full stream: validate the pinned config and header
/// without downloading the 4.28 GB tensor payload.
#[test]
#[ignore = "network: reads the pinned Qwen2.5 MLX config and safetensors header"]
fn the_pinned_qwen25_mlx_header_matches_the_qwen2_contract() {
    let config = String::from_utf8(get("config.json")).expect("config utf8");
    assert_eq!(
        parse_qwen2_config(&config).expect("Qwen2 config parses"),
        model_io::qwen2_5_7b()
    );

    let source = HttpRangeSource::new(format!("{MLX_BASE}/model.safetensors"));
    let mut header = fetch_safetensors_header(&source).expect("fetch safetensors header");
    assert!(header
        .tensors
        .keys()
        .any(|name| name == "model.layers.0.self_attn.q_proj.bias"));
    canonicalize_qwen2_header(&mut header).expect("canonicalize MLX Qwen2 names");
    for name in [
        "language_model.model.layers.0.self_attn.q_proj.bias",
        "language_model.model.layers.0.self_attn.k_proj.bias",
        "language_model.model.layers.0.self_attn.v_proj.bias",
    ] {
        assert!(
            header.tensors.contains_key(name),
            "missing canonical tensor {name}"
        );
    }
}
