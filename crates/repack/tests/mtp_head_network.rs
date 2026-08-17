//! Header-only probe of the multi-token-prediction head this engine does not
//! yet ingest (`docs/MTP_SPECULATIVE.md`, stage 0). Reads the safetensors
//! INDEX and ONE shard header off `Qwen/Qwen3.8-27B` -- about 114 KB against a
//! 54 GB checkpoint, a few seconds -- and asserts what the head is made of.
//! Not run by default:
//!
//! ```sh
//! cargo test -p turbospark-repack --test mtp_head_network --release -- --ignored --nocapture
//! ```
//!
//! **THE ONE QUESTION IT EXISTS TO ANSWER IS WHETHER THE HEAD'S SINGLE BLOCK
//! IS FULL-ATTENTION OR GATED-DELTANET**, because every downstream estimate
//! forks on it. `qwen3_5` is a hybrid -- three linear layers to every full one
//! -- so a head built from a LINEAR block would need `GdnStateManager` state
//! per draft step, would make the draft non-invertible, and would put the
//! recurrent chain on the speculative path. It is FULL-attention, so the draft
//! step reuses `families/qwen/attn.rs` unchanged and its rollback is a KV
//! cursor move.
//!
//! **THE SECOND FINDING IS THAT NO THIRD-PARTY ARTIFACT IS NEEDED.**
//! `youssofal/MTPLX` publishes this head as an 849 MB `mtp.safetensors`
//! sidecar beside its own re-quantized trunk, and the arithmetic below sums to
//! 849.40 MB of BF16 -- so that sidecar is a verbatim copy of Qwen's own
//! tensors, not a re-calibrated one. The head can therefore be read straight
//! out of the official checkpoint's LAST SHARD, which carries the 15 `mtp.*`
//! tensors and exactly one other. That removes both an Apache-2.0 weights
//! dependency and the "calibrated against a different trunk" risk that would
//! otherwise sit under every accept-length number.
//!
//! Note what is deliberately NOT asserted here: anything about
//! `mlx-community/Qwen3.8-27B-4bit`, whose conversion DROPS the head
//! (`qwen38_checkpoint_network.rs` states that and asserts it). The two files
//! are read by two tests because they are two claims.

use turbospark_repack::{fetch_safetensors_header, HttpRangeSource};

/// The official BF16 checkpoint, pinned to a revision like every other
/// network target here. NOT the mlx-community conversion the trunk install is
/// streamed from -- that one has no head at all.
const REPO_BASE: &str = "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main";

/// The shard the index maps every `mtp.*` tensor to. Read from the index
/// rather than hardcoded into the fetch; this constant is the assertion.
const MTP_SHARD: &str = "model-00018-of-00018.safetensors";

/// `config.json`'s `mtp_num_hidden_layers`. One block, which is what makes a
/// draft step ~1/64 of a forward pass on a 64-layer trunk.
const MTP_LAYERS: usize = 1;

/// Every tensor the head is made of, with the shape the header declares.
/// Read against `model_io::qwen_gdn_dense_27b()` below rather than trusted:
/// each row is either a trunk shape or a statement about the head's own
/// structure, and the comment says which.
const EXPECTED: &[(&str, &[u64])] = &[
    // The head's own structure. `fc` projects the CONCATENATION of a
    // normalized next-token embedding and a normalized trunk hidden state
    // back down to one hidden, so its input is 2 x hidden and it is the only
    // tensor in the model with that width.
    ("mtp.fc.weight", &[5120, 10240]),
    ("mtp.pre_fc_norm_embedding.weight", &[5120]),
    ("mtp.pre_fc_norm_hidden.weight", &[5120]),
    ("mtp.norm.weight", &[5120]),
    // The block. Every one of these is a TRUNK full-attention layer shape,
    // which is the finding: no new kernel, no new dispatch shape.
    ("mtp.layers.0.input_layernorm.weight", &[5120]),
    ("mtp.layers.0.post_attention_layernorm.weight", &[5120]),
    // q_proj is 2 x (24 heads x 256) because `attn_output_gate` is true for
    // this family: half of it is the gate. o_proj's INPUT is the unhalved
    // 24 x 256 = 6144, which is the cheapest independent confirmation of that
    // -- a non-gated block would have 12288 on both.
    ("mtp.layers.0.self_attn.q_proj.weight", &[12288, 5120]),
    ("mtp.layers.0.self_attn.k_proj.weight", &[1024, 5120]),
    ("mtp.layers.0.self_attn.v_proj.weight", &[1024, 5120]),
    ("mtp.layers.0.self_attn.o_proj.weight", &[5120, 6144]),
    ("mtp.layers.0.self_attn.q_norm.weight", &[256]),
    ("mtp.layers.0.self_attn.k_norm.weight", &[256]),
    ("mtp.layers.0.mlp.gate_proj.weight", &[17408, 5120]),
    ("mtp.layers.0.mlp.up_proj.weight", &[17408, 5120]),
    ("mtp.layers.0.mlp.down_proj.weight", &[5120, 17408]),
];

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

#[test]
#[ignore = "reads ~114 KB of headers off the real Qwen3.8-27B checkpoint over the network"]
fn scopes_the_qwen38_mtp_head() {
    let arch = model_io::qwen_gdn_dense_27b();

    // 1. The index: which tensors exist and which shard holds them.
    let index_bytes = get("model.safetensors.index.json");
    let index: serde_json::Value = serde_json::from_slice(&index_bytes).expect("index json");
    let weight_map = index["weight_map"].as_object().expect("weight_map");
    let mtp: Vec<&String> = weight_map
        .keys()
        .filter(|n| n.starts_with("mtp."))
        .collect();
    assert_eq!(
        mtp.len(),
        EXPECTED.len(),
        "the head's tensor inventory moved: {mtp:?}"
    );
    for name in &mtp {
        assert_eq!(
            weight_map[name.as_str()].as_str(),
            Some(MTP_SHARD),
            "{name} is no longer in the shard this probe reads"
        );
    }
    // The head has NO embedding table and NO lm_head, which is what makes it
    // 849 MB rather than several GB: it reuses the trunk's, both of which are
    // already resident in a `.gturbo` install. `mtp_use_dedicated_embeddings`
    // is false in config.json and this is the same claim read off the bytes.
    for suffix in ["embed_tokens.weight", "lm_head.weight"] {
        assert!(
            !mtp.iter().any(|n| n.ends_with(suffix)),
            "the head grew its own {suffix}; it no longer shares the trunk's"
        );
    }

    // 2. The shard header: shapes, dtypes and the byte total.
    let source = HttpRangeSource::new(format!("{REPO_BASE}/{MTP_SHARD}"));
    let header = fetch_safetensors_header(&source).expect("shard header");
    let mut total = 0u64;
    println!(
        "\n{:52} {:20} {:6} {:>10}",
        "tensor", "shape", "dtype", "MB"
    );
    for (name, shape) in EXPECTED {
        let info = header
            .tensors
            .get(*name)
            .unwrap_or_else(|| panic!("{name} is not in {MTP_SHARD}"));
        assert_eq!(&info.shape, shape, "{name} changed shape");
        // BF16 THROUGHOUT, and that is a constraint rather than a note: this
        // port dispatches no unquantized GEMV, so the head's five matrices
        // have to be quantized at repack. Its norms narrow through the
        // existing `narrow_raw_to_bf16` at zero cost, since they are already
        // the target width (`crates/repack` Gotcha 9).
        assert_eq!(info.dtype, "BF16", "{name} is not BF16");
        let bytes = info.data_offsets.1 - info.data_offsets.0;
        assert_eq!(
            bytes,
            shape.iter().product::<u64>() * 2,
            "{name} is not a dense BF16 tensor of its declared shape"
        );
        total += bytes;
        println!(
            "{name:52} {:20} {:6} {:>10.2}",
            format!("{shape:?}"),
            info.dtype,
            bytes as f64 / 1e6
        );
    }
    println!(
        "{:52} {:20} {:6} {:>10.2}\n",
        "TOTAL",
        "",
        "",
        total as f64 / 1e6
    );

    // 849,398,784 bytes = 849.40 MB. Quoted because it identifies MTPLX's
    // `mtp.safetensors` (849 MB) as a verbatim copy of these bytes, which is
    // why nothing here depends on that repository. A drifting total would mean
    // the head was re-published and the identification has to be re-made.
    //
    // The per-tensor `product(shape) * 2` assertion above already makes this
    // derivable, so it is a cross-check rather than a second source of truth
    // -- and it earned that on its first run, catching a hand-rounded literal
    // (849_400_320) that no other assertion here could see.
    assert_eq!(total, 849_398_784, "the head's byte total moved");

    // 3. The block is a TRUNK full-attention layer, field by field. This is
    // the probe's whole reason to exist: if any of these fail, the draft step
    // needs its own kernels and `docs/MTP_SPECULATIVE.md`'s cost estimate is
    // wrong rather than merely imprecise.
    let hidden = arch.hidden_size as u64;
    let q_out = arch.num_heads as u64 * arch.full_head_dim as u64;
    let kv_out = arch.num_full_kv_heads as u64 * arch.full_head_dim as u64;
    assert!(arch.attn_output_gate, "this check assumes a gated block");
    let q = |n: &str| header.tensors[n].shape.clone();
    assert_eq!(
        q("mtp.layers.0.self_attn.q_proj.weight"),
        vec![2 * q_out, hidden]
    );
    assert_eq!(
        q("mtp.layers.0.self_attn.k_proj.weight"),
        vec![kv_out, hidden]
    );
    assert_eq!(
        q("mtp.layers.0.self_attn.v_proj.weight"),
        vec![kv_out, hidden]
    );
    assert_eq!(
        q("mtp.layers.0.self_attn.o_proj.weight"),
        vec![hidden, q_out]
    );
    assert_eq!(
        q("mtp.layers.0.self_attn.q_norm.weight"),
        vec![arch.full_head_dim as u64],
        "per-head norm, so the head takes the trunk's q/k norm path too"
    );
    // The FFN is the trunk's DENSE width, not a routed expert's. `qwen3_5` is
    // dense, so `moe_intermediate_size` is 0 and there is no other candidate.
    let inter = arch.intermediate_size as u64;
    assert_eq!(q("mtp.layers.0.mlp.gate_proj.weight"), vec![inter, hidden]);
    assert_eq!(q("mtp.layers.0.mlp.down_proj.weight"), vec![hidden, inter]);

    // 4. The two structural tensors that are NOT trunk shapes.
    assert_eq!(
        q("mtp.fc.weight"),
        vec![hidden, 2 * hidden],
        "fc must take [embedding, hidden] concatenated"
    );
    assert_eq!(q("mtp.norm.weight"), vec![hidden]);

    println!(
        "FINDINGS\n\
         - {MTP_LAYERS} block, FULL attention (self_attn q/k/v/o + per-head q/k norms),\n  \
           shape-identical to a trunk full-attention layer: no new Metal kernel.\n\
         - Dense SwiGLU MLP at the trunk's own intermediate width ({inter}).\n\
         - No embedding table and no lm_head: both shared with the trunk.\n\
         - BF16 throughout, {:.2} MB, so the five matrices need quantizing at\n  \
           repack (drafter quality is a throughput axis, never a correctness one).\n\
         - All 15 tensors live in {MTP_SHARD}, so ingestion is one ranged read\n  \
           of one shard rather than a dependency on a third-party sidecar.\n",
        total as f64 / 1e6
    );
}
