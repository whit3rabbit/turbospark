//! ROADMAP M5 step 5: install the real published `gpt-oss-20b` MXFP4 GGUF.
//!
//! The SIXTH family, and the first whose expert block type has no resident
//! GEMV at all: MXFP4 appears in `ffn_{gate,up,down}_exps` and nowhere else,
//! with attention, `token_embd` and `output` all Q8_0. That partition is the
//! file's own shape rather than a scoping decision, and it is why the type
//! cost two kernels where the usual rule is three (AGENTS.md Gotcha 29).
//!
//! It is also the first file to put a BIAS beside every projection, a sink
//! vector on every attention block, and a per-expert bias INSIDE the routed
//! blob -- the last of which is a RANK-2 routed tensor, where every routed
//! tensor the walk had ever seen was rank 3. All three were exercised against
//! `SyntheticGptOssShape` before this test was written, which is the order
//! `crates/repack/CLAUDE.md` Gotcha 8 asks for: the fixture found the rank-2
//! refusal and the F32-bias-counted-as-a-block-type refusal in milliseconds
//! each, where the same two discoveries against this file would have cost a
//! 25-minute re-stream apiece.
//!
//! The 12.1 GB file is NEVER written to disk: `write_gguf_install_streamed`
//! takes a `RangeSource` and reads a layer at a time.

use std::path::PathBuf;

use turbospark_repack::{fetch_gguf_header, write_gguf_install_streamed, HttpRangeSource};

const GPT_OSS_20B_MXFP4: &str =
    "https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-MXFP4.gguf";
const MODEL_ID: &str = "ggml-org/gpt-oss-20b-GGUF";

/// The GGUF carries its tokenizer as llama.cpp metadata; this port loads an
/// HF `tokenizer.json`. Take the sidecars from the checkpoint the GGUF was
/// converted from, as every sibling does.
///
/// `chat_template.jinja` is NOT optional here, unlike for the four families
/// before it. `ChatDialect::Harmony` has no fallback renderer on purpose --
/// Harmony is a 17 KB template with a system preamble, a reasoning-effort
/// knob and a TypeScript tool namespace, and a partial re-implementation is
/// AGENTS.md Gotcha 41's failure mode exactly -- so an install without it
/// refuses to render rather than inventing a frame.
const SIDECAR_BASE: &str = "https://huggingface.co/openai/gpt-oss-20b/resolve/main";

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

fn install_dir() -> PathBuf {
    let dir = match std::env::var_os("TURBOSPARK_GPTOSS_INSTALL_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => std::env::temp_dir().join(format!("turbospark-gptoss-{}", std::process::id())),
    };
    for pinned in [
        "TURBOSPARK_GEMMA4_INSTALL_DIR",
        "TURBOSPARK_QWEN36_INSTALL_DIR",
        "TURBOSPARK_QWEN3MOE_INSTALL_DIR",
        "TURBOSPARK_GEMMA4_IQ_INSTALL_DIR",
        "TURBOSPARK_MISTRAL_INSTALL_DIR",
    ] {
        assert_ne!(
            std::env::var_os(pinned).map(PathBuf::from),
            Some(dir.clone()),
            "refusing to overwrite {pinned}, which the standing gates measure against"
        );
    }
    std::fs::create_dir_all(&dir).expect("create install dir");
    dir
}

#[test]
#[ignore = "network: streams the real ~12.1 GB gpt-oss-20b MXFP4 GGUF and writes a ~12 GB install"]
fn repacks_the_real_gpt_oss_20b_mxfp4_gguf() {
    let source = HttpRangeSource::new(GPT_OSS_20B_MXFP4);
    let header = fetch_gguf_header(&source).expect("fetch GGUF header");
    assert_eq!(header.architecture(), Some("gpt-oss"));
    // The merged expert layout, as `gguf_names.rs` requires, and NOT fused:
    // Gemma packs gate and up into one `ffn_gate_up_exps`, this file keeps
    // them apart, and the walk splits or does not split on that basis.
    for name in ["ffn_gate_exps.weight", "ffn_up_exps.weight"] {
        assert!(
            header.tensors.keys().any(|n| n.ends_with(name)),
            "expected an unfused {name}"
        );
    }
    // THE RANK-2 ROUTED TENSORS, asserted off the header before a byte of
    // expert data is fetched. This is the shape that refused the walk until
    // ROADMAP M5, and `blk.0` is enough to see it.
    for role in ["gate", "up", "down"] {
        let name = format!("blk.0.ffn_{role}_exps.bias");
        let info = header
            .tensors
            .get(&name)
            .unwrap_or_else(|| panic!("{name} missing"));
        assert_eq!(
            info.dims.len(),
            2,
            "{name} should be [width, experts], got {:?}",
            info.dims
        );
    }
    eprintln!(
        "header: {} tensors, alignment {}, data region starts at {}",
        header.tensors.len(),
        header.alignment,
        header.data_region_start
    );

    let dir = install_dir();
    eprintln!("installing to {}", dir.display());
    let arch = write_gguf_install_streamed(&dir, &header, &source, MODEL_ID, |stage| {
        eprintln!("[repack] {stage}");
    })
    .expect("streamed GGUF install");
    assert_eq!(arch, model_io::gpt_oss_20b());

    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ] {
        std::fs::write(dir.join(name), get(&format!("{SIDECAR_BASE}/{name}")))
            .expect("tokenizer sidecar");
    }

    model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect("manifest validates: mxfp4 is executable in the routed slot");
    let resident =
        model_io::load_resident_index(&dir.join("model_weights.bin")).expect("resident index");

    let tag = |ggml: u32| turbospark_repack::dtype_tag_for_ggml_type(ggml).expect("tag");
    let dtype = |name: &str| resident.entries[name].dtype;
    let l0 = "language_model.model.layers.0";
    // Q8_0 EVERYWHERE OUTSIDE THE EXPERTS. If any of these came back MXFP4
    // the install would pass the manifest gate and be stopped by the resident
    // dtype backstop, which is the layering working -- but it would also mean
    // the file is not the one this family was scoped on.
    assert_eq!(dtype("language_model.model.embed_tokens.weight"), tag(8));
    assert_eq!(
        dtype("language_model.lm_head.weight"),
        tag(8),
        "gpt-oss ships an untied Q8_0 head"
    );
    assert_eq!(dtype(&format!("{l0}.self_attn.q_proj.weight")), tag(8));

    // THE BIASES AND THE SINKS, which no family before this had. GGUF ships
    // them F32 and `norm_view` reads BF16, so carrying them verbatim would
    // fail at open with a byte-size error and dropping them would skip an
    // addition silently.
    for (tail, elems) in [
        ("self_attn.q_proj.bias", arch.num_heads * arch.full_head_dim),
        (
            "self_attn.k_proj.bias",
            arch.num_full_kv_heads * arch.full_head_dim,
        ),
        (
            "self_attn.v_proj.bias",
            arch.num_full_kv_heads * arch.full_head_dim,
        ),
        ("self_attn.o_proj.bias", arch.hidden_size),
        // ONE PER QUERY HEAD, not per KV head: 64 against 8 on this model.
        ("self_attn.sinks.weight", arch.num_heads),
    ] {
        let entry = resident
            .entries
            .get(&format!("{l0}.{tail}"))
            .unwrap_or_else(|| panic!("{tail} is missing from the install"));
        assert_eq!(
            entry.dtype,
            turbospark_repack::DTYPE_BF16,
            "{tail} should be BF16 after the transcode"
        );
        assert_eq!(entry.size_bytes as i64, elems * 2, "{tail} width");
    }
    // The router lands INT8 affine, as on every other GGUF family, and its
    // own bias narrows to BF16 beside it.
    assert_eq!(dtype(&format!("{l0}.mlp.gate.weight")), 5);
    assert_eq!(
        resident.entries[&format!("{l0}.mlp.gate.bias")].size_bytes as i64,
        arch.num_experts * 2
    );

    let layout = model_io::load_packed_experts_layout(&dir, 64 * 1024 * 1024).expect("layout");
    assert_eq!(layout.num_layers, 24);
    assert_eq!(layout.experts_per_layer, 32);

    // SIX SUB-TENSORS PER EXPERT, not three. The per-expert biases ride in
    // the blob beside the weights they belong to, so the streamer reads one
    // contiguous run per miss and the kernel never needs to know which expert
    // a slot holds.
    let expert0 = layout
        .layers
        .first()
        .and_then(|layer| layer.experts.first())
        .expect("one expert");
    for role in [
        "gate",
        "up",
        "down",
        "gate_biases",
        "up_biases",
        "down_biases",
    ] {
        assert!(
            expert0.sub_tensors.contains_key(role),
            "expert blob is missing `{role}`; got {:?}",
            expert0.sub_tensors.keys().collect::<Vec<_>>()
        );
    }
    // Gate and up output the expert width; down outputs `hidden`. Both are
    // 2880 on this model, which is exactly why `SyntheticGptOssShape` keeps
    // them unequal -- the real file cannot tell the two apart.
    for (role, elems) in [
        ("gate_biases", arch.moe_intermediate_size),
        ("up_biases", arch.moe_intermediate_size),
        ("down_biases", arch.hidden_size),
    ] {
        assert_eq!(
            expert0.sub_tensors[role].size as i64,
            elems * 4,
            "{role} rides F32 verbatim from the GGUF, which is what the kernel reads"
        );
    }

    // THE GRANULARITY CLAIM, measured on the artifact rather than predicted
    // from the header (Gotcha 36). 12.6 MiB per expert is 4x Gemma's and 5x
    // Qwen3-30B-A3B's, but 8.6x SMALLER than Mixtral's 108.9, which is the
    // whole reason this candidate survived the M5 survey.
    let stride: u64 = expert0.sub_tensors.values().map(|sub| sub.size).sum();
    let mib = stride as f64 / (1024.0 * 1024.0);
    let slot_cache_gib =
        stride as f64 * 16.0 * layout.num_layers as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!("one expert blob {mib:.2} MiB, slot cache at 16 slots {slot_cache_gib:.2} GiB");
    assert!(
        mib < 16.0,
        "expert blob {mib:.2} MiB is not fine-grained; Mixtral's is 108.9 and cannot stream here"
    );

    eprintln!(
        "SUCCESS: real gpt-oss-20b MXFP4 GGUF installed to {}",
        dir.display()
    );
}
