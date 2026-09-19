//! Network-gated proof that the HADAMARD-FOLDED repack path works against
//! the REAL published checkpoint:
//! `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` (the Bonsai-2 line,
//! `docs/BONSAI2.md`). Sibling of `ternary_checkpoint_network.rs`; not run
//! by default:
//!
//! ```sh
//! TURBOSPARK_BONSAI2_INSTALL_DIR=~/.turbospark/models/text/bonsai2.gturbo \
//!   cargo test -p turbospark-repack --test bonsai2_checkpoint_network --release -- --ignored --nocapture
//! ```
//!
//! **STILL A FOURTH CHECKPOINT OF ONE ARCHITECTURE, NOT A NEW FAMILY** --
//! `tests/qwen35_config.rs` says so without the network: this file's
//! `text_config` parses to `model_io::qwen_gdn_dense_27b()` exactly. What
//! IS new is the WEIGHT BASIS: every quantized matrix is stored as
//! `W * diag(signs) * H / sqrt(block)`, so the walk has to carry the
//! contract (`hadamard.bin` plus the manifest section) beside the weights,
//! and the runtime transforms activations. See that page for why the
//! transform is NOT folded back into the weights at repack time.
//!
//! The artifact is 8.0 GiB, one shard, NO `model.safetensors.index.json` --
//! the original ternary checkpoint's cheapest whole-file fingerprint does
//! not exist here -- so the PIN is the safetensors header itself: its
//! 303,110 leading bytes (8-byte length + 303,102-byte JSON, naming all
//! 2,390 tensors) hash to the constant below. `model.safetensors` is
//! 8,595,477,990 bytes; the walk's own last-tensor check confirms the
//! header against the real file size.
//!
//! **THE SIDECAR LIST IS THIS REPO'S, NOT CARRIED OVER** (the near-miss the
//! ternary file's header warns about, third time around): it ships
//! `generation_config.json` and NO `vocab.json`/`merges.txt` -- the inverse
//! of the ternary repo and a strict subset of Qwen3.8's.

use std::path::PathBuf;

use turbospark_repack::{
    fetch_safetensors_header, parse_gemma4_quantization, parse_prism_hadamard,
    parse_qwen_gdn_dense_config, write_qwen_gdn_dense_install_streamed_with_hadamard, Gemma4Shards,
    HttpRangeSource, RangeSource,
};

const REPO_BASE: &str = "https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit/resolve/3f926b415992eaa2ae9dd7b573706494d6bbf787";
const MODEL_ID: &str = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit";
/// The safetensors header, bytes 0..303110 of `model.safetensors`, hashed so
/// a silently re-uploaded artifact fails HERE rather than a quarter of an
/// hour into a stream. Includes the 8-byte big-endian header-length prefix,
/// so it is the hash of the file's actual leading bytes and nothing else.
const PINNED_HEADER_SHA256: &str =
    "6260cd46303cbb72428f16a827e70962aa7a2d98b7ec6124804cd774db49f7ac";
const HEADER_BYTES: u64 = 303_110;
/// `model.safetensors`, bytes. Asserted via the header's last tensor below.
const MODEL_BYTES: u64 = 8_595_477_990;

/// The 2-bit-affine resident dtype tag; 15 would be 1-bit, 4 INT4, 1 raw
/// BF16 (`resident_writer`'s constant, restated because it is private).
const DTYPE_INT2_AFFINE: u8 = 16;

fn get(path: &str) -> Vec<u8> {
    let url = format!("{REPO_BASE}/{path}");
    let response = reqwest::blocking::Client::builder()
        .timeout(None)
        .build()
        .expect("client")
        .get(&url)
        .send()
        .unwrap_or_else(|e| panic!("GET {url}: {e}"));
    assert!(
        response.status().is_success(),
        "GET {url}: HTTP {}",
        response.status()
    );
    response.bytes().expect("body").to_vec()
}

fn install_dir() -> PathBuf {
    match std::env::var_os("TURBOSPARK_BONSAI2_INSTALL_DIR") {
        Some(dir) => {
            let dir = PathBuf::from(dir);
            std::fs::create_dir_all(&dir).expect("create install dir");
            dir
        }
        None => {
            let dir = std::env::temp_dir()
                .join(format!("turbospark-bonsai2-real-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            dir
        }
    }
}

#[test]
#[ignore = "streams the real 8.0 GB Hadamard-folded Bonsai-2 27B checkpoint over the network"]
fn repacks_the_real_hadamard_folded_bonsai2_checkpoint() {
    // Config FIRST, cross-checked against the pinned baseline: failing here
    // (or in the offline `qwen35_config.rs` twin) costs seconds, not a
    // stream.
    let config = String::from_utf8(get("config.json")).expect("config utf8");
    let arch = parse_qwen_gdn_dense_config(&config).expect("config parses");
    assert_eq!(
        arch,
        model_io::qwen_gdn_dense_27b(),
        "parsed config does not match the pinned qwen3_5 baseline"
    );
    assert_eq!(arch.num_experts, 0, "this family is DENSE");

    // TWO BITS AT GROUP 128, now declared at the CONFIG ROOT (the ternary
    // file carried the object inside text_config) and with an explicit
    // affine mode -- parse_gemma4_quantization skips that key by name.
    let quant = parse_gemma4_quantization(&config).expect("quantization parses");
    assert_eq!(quant.default_bits, 2);
    assert_eq!(quant.group_size, 128);

    // THE CONTRACT, off the same file: 402 packed modules, every one folded
    // at block 1024, and the embedding alone marked for the inverse.
    let contract = parse_prism_hadamard(&config)
        .expect("contract parses")
        .expect("the hadamard modules list is present");
    assert_eq!(contract.block, 1024);
    assert_eq!(contract.folded.len(), 401, "lm_head + 400 layer matrices");
    assert_eq!(
        contract.inverse,
        vec!["language_model.model.embed_tokens.weight".to_string()],
        "the embedding is the one inverse-transformed module"
    );

    // The header, fingerprinted: fetch the pinned leading bytes and hash
    // them before anything else looks at the file.
    let header_url = format!("{REPO_BASE}/model.safetensors");
    let source = HttpRangeSource::new(header_url);
    let head = source
        .read_range(0, HEADER_BYTES)
        .expect("header range read");
    assert_eq!(
        model_io::hash_data(&head),
        PINNED_HEADER_SHA256,
        "model.safetensors' leading bytes do not match the pinned fingerprint"
    );
    let header = fetch_safetensors_header(&source).expect("safetensors header");
    let declared: u64 = header
        .tensors
        .values()
        .map(|t| t.data_offsets.1)
        .max()
        .expect("tensors")
        + header.data_region_start();
    assert_eq!(
        declared, MODEL_BYTES,
        "the header's last tensor does not end at the published file size"
    );

    // THE INVENTORY: 2,390 tensors = the ternary checkpoint's 2,180 minus
    // nothing, plus 402 sign vectors and the quantization-freed aux
    // projections. The config comparison and this count share no input.
    assert_eq!(header.tensors.len(), 2_390, "the tensor inventory moved");
    let vision = header
        .tensors
        .keys()
        .filter(|n| n.starts_with("vision_tower."))
        .count();
    assert_eq!(vision, 333, "the vision tower moved");
    let signs = header
        .tensors
        .keys()
        .filter(|n| n.ends_with(".signs"))
        .count();
    assert_eq!(signs, 402, "one sign vector per packed module");
    // THE UNQUANTIZED PAIR: in_proj_a/b are raw F32 matrices here, where
    // every earlier checkpoint quantized them -- the reason the runtime
    // gained a BF16 GEMV. Their install entries are the walk's BF16 tag.
    let in_proj_a = &header.tensors["language_model.model.layers.0.linear_attn.in_proj_a.weight"];
    assert_eq!(
        in_proj_a.dtype, "F32",
        "in_proj_a moved; the BF16 GEMV's first caller is this tensor"
    );

    let shards = Gemma4Shards::single(&header, &source);
    let dir = install_dir();
    eprintln!("installing to {}", dir.display());
    write_qwen_gdn_dense_install_streamed_with_hadamard(
        &dir,
        &arch,
        MODEL_ID,
        &shards,
        &quant,
        &contract,
        |stage| eprintln!("[repack] {stage}"),
    )
    .expect("streamed install");

    // Tokenizer sidecars so turbospark-check can open the dir directly. See
    // the module header: this list is read off THIS repo. ChatML dialect, so
    // a raw `--prompt` babbles; use `--messages-file` or `--chat`.
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ] {
        std::fs::write(dir.join(name), get(name)).expect("tokenizer sidecar");
    }

    // The contract file, beside the weights, and the manifest section that
    // points at it. (5120 + 6144 + 17408) F32 values = 114,688 bytes.
    let hadamard_bytes = std::fs::read(dir.join("hadamard.bin")).expect("hadamard.bin");
    assert_eq!(hadamard_bytes.len(), 114_688, "the sign blob's size moved");
    let manifest = std::fs::read_to_string(dir.join("manifest.json")).expect("manifest");
    let manifest_json: serde_json::Value = serde_json::from_str(&manifest).expect("manifest json");
    let section = manifest_json
        .get("hadamard")
        .expect("the manifest carries the hadamard section");
    assert_eq!(section["block"], 1024);
    let mut widths: Vec<u64> = section["signs"]
        .as_array()
        .expect("signs list")
        .iter()
        .map(|s| s["width"].as_u64().unwrap())
        .collect();
    widths.sort_unstable();
    assert_eq!(
        widths,
        vec![5120, 6144, 17408],
        "one sign vector per distinct input width"
    );

    // Read the install back through the real loaders.
    model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect("manifest validates against the production baseline");
    let resident_index =
        model_io::load_resident_index(&dir.join("model_weights.bin")).expect("resident index");
    for name in [
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
        // Layer 0 is LINEAR: it has no self_attn at all.
        "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
        "language_model.model.layers.0.linear_attn.conv1d.weight",
        "language_model.model.layers.0.linear_attn.A_log",
        "language_model.model.layers.0.linear_attn.dt_bias",
        // Layer 3 is the first FULL-attention layer.
        "language_model.model.layers.3.self_attn.q_proj.weight",
        "language_model.model.layers.3.self_attn.q_norm.weight",
        // The DENSE FFN, on the last layer.
        "language_model.model.layers.63.mlp.gate_proj.weight",
        "language_model.model.layers.63.mlp.down_proj.weight",
        // THE UNQUANTIZED PAIR, resident at BF16.
        "language_model.model.layers.0.linear_attn.in_proj_a.weight",
        "language_model.model.layers.0.linear_attn.in_proj_b.weight",
    ] {
        assert!(
            resident_index.entries.contains_key(name),
            "resident index is missing {name}"
        );
    }
    let a_entry =
        &resident_index.entries["language_model.model.layers.0.linear_attn.in_proj_a.weight"];
    assert_eq!(
        a_entry.dtype, 1,
        "the unquantized in_proj_a must land as BF16 (tag 1), the tag the GEMV arm reads"
    );
    assert_eq!(
        a_entry.size_bytes,
        48 * 5120 * 2,
        "a BF16 [48, 5120] matrix is rows*cols*2 bytes"
    );
    for name in [
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
        "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
        "language_model.model.layers.3.self_attn.q_proj.weight",
    ] {
        assert_eq!(
            resident_index.entries[name].dtype, DTYPE_INT2_AFFINE,
            "{name} should carry the 2-bit affine tag"
        );
    }
    // The width, on real bytes: identical affine geometry to the ternary
    // install (16 elements a u32 word, 40 FP16 companions a row at group
    // 128) -- the rotation changed no plane's SHAPE.
    let embed = &resident_index.entries["language_model.model.embed_tokens.weight"];
    let (rows, cols) = (arch.vocab_size as u64, arch.hidden_size as u64);
    assert_eq!(embed.size_bytes, rows * cols / 4);
    assert_eq!(embed.scale_size, rows * (cols / 128) * 2);
    assert_eq!(embed.bias_size, embed.scale_size);

    // EVERY unquantized tensor is narrowed to BF16 (AGENTS.md Gotcha 45);
    // the checkpoint ships its norms, conv and aux projections F32.
    for e in resident_index.entries.values() {
        if e.scale_size == 0 && e.bias_size == 0 {
            assert_eq!(e.dtype, 1, "{} was not narrowed to BF16", e.name);
        }
    }
    // A dense install: no router, no shared expert, no routed experts.
    for marker in [".mlp.gate.weight", ".mlp.shared_expert", ".switch_mlp."] {
        assert!(
            !resident_index.entries.keys().any(|k| k.contains(marker)),
            "a dense install carries {marker}"
        );
    }
    // The vision tower is excluded outright, and no sign vector leaked into
    // the resident set.
    assert!(
        !resident_index
            .entries
            .keys()
            .any(|k| k.starts_with("vision_tower.")),
        "vision tower leaked into the resident set"
    );
    assert!(
        !resident_index.entries.keys().any(|k| k.ends_with(".signs")),
        "a sign vector leaked into the resident set"
    );

    let layout =
        model_io::load_packed_experts_layout(&dir, 64 * 1024 * 1024).expect("experts layout");
    assert_eq!(layout.layers.len(), 0, "a dense model streams nothing");

    eprintln!(
        "SUCCESS: real Hadamard-folded Bonsai-2 repacked into {}, run \
         `cargo run -p turbospark-cli --bin turbospark-check --release -- --model {} --messages-file /tmp/p.json`",
        dir.display(),
        dir.display()
    );
}
