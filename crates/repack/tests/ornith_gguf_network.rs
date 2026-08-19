//! Ornith-1.5's two published GGUFs, read HEADER-ONLY.
//!
//! A few MB off a 21.7 GB and a 5.6 GB file, seconds each, no checkpoint
//! downloaded. This is the check `crates/repack/CLAUDE.md` Gotcha 5 argues
//! for and the one AGENTS.md Gotcha 47's workflow turns into a habit: settle
//! the family, the shape, the name table and the block types BEFORE budgeting
//! a stream, because every one of those is knowable from the header.
//!
//! ```sh
//! cargo test -p turbospark-repack --test ornith_gguf_network --release -- --ignored --nocapture
//! ```
//!
//! **The 35B case is an INDEPENDENT re-derivation of `ornith_config.rs`'s
//! headline.** That file parses the HF `config.json` and asserts the result
//! equals `qwen_gdn_moe_35b_a3b()`; this one parses llama.cpp's GGUF METADATA
//! for the same model and asserts the same equality. The two sources share no
//! bytes, no parser and no key names, so agreeing is worth more than either
//! alone -- the same reasoning `qwen38_checkpoint_network.rs` uses when it
//! pairs a config diff with a tensor inventory.

use model_io::ModelFamily;
use turbospark_repack::{
    arch_from_gguf, family_for_architecture, fetch_gguf_header, ggml_type_block, ggml_type_name,
    map_gguf_name, GgufHeader, HttpRangeSource,
};

/// Pinned by revision rather than floating at `main`: `curl -sI` on a
/// `resolve/main` URL returns `x-repo-commit`, which costs no bytes.
const MOE_Q4_K_M: &str = "https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF/resolve/5ae357e3eaf951ae221e8d784c71a8a3cdb6aa5f/Ornith-1.5-35B-Q4_K_M.gguf";
const DENSE_Q4_K_M: &str = "https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF/resolve/0677a38f331a214c4e5e7bd07ecab04c14ac52f1/Ornith-1.5-9B-Q4_K_M.gguf";

fn fetch(url: &str) -> GgufHeader {
    let source = HttpRangeSource::new(url);
    fetch_gguf_header(&source).expect("fetch GGUF header")
}

/// The trunk/head split, restated here rather than reached through
/// `plan::head_block_index`.
///
/// A test that asserts through the function it is testing cannot see that
/// function being wrong, and this rule is two lines. It must agree with
/// `plan.rs`'s by inspection.
fn is_head_block(name: &str, trunk_layers: usize) -> bool {
    let Some(rest) = name.strip_prefix("blk.") else {
        return false;
    };
    let Some((idx, _)) = rest.split_once('.') else {
        return false;
    };
    idx.parse::<usize>()
        .is_ok_and(|layer| layer >= trunk_layers)
}

/// Every TRUNK tensor must map. The head's block is excluded by the same
/// rule the walk uses, and counted separately below so its size is a
/// reported number rather than an unexamined exclusion.
fn assert_every_trunk_name_maps(h: &GgufHeader, family: ModelFamily, trunk_layers: usize) {
    let mut unmapped: Vec<&String> = h
        .tensors
        .keys()
        .filter(|n| !is_head_block(n, trunk_layers))
        .filter(|n| map_gguf_name(n, family).is_err())
        .collect();
    unmapped.sort();
    assert!(
        unmapped.is_empty(),
        "{} unmapped trunk tensor names, e.g. {:?}",
        unmapped.len(),
        unmapped.iter().take(10).collect::<Vec<_>>()
    );
}

/// Block-type histogram in BYTES, plus the UNSIZED rows.
///
/// Read the unsized rows before the percentages (Phase S's trap, Gotcha 5): a
/// type with no `ggml_type_block` row is one this port cannot ingest, and on
/// a mixed file that is usually the routed experts -- printing it as 0 bytes
/// once made an imatrix file look 76% Q8_0. The sized total matching the
/// published file size is what says the accounting closed.
fn report_block_types(h: &GgufHeader) -> Vec<u32> {
    let mut sized: std::collections::BTreeMap<u32, (usize, u64)> = Default::default();
    let mut unsized_types: std::collections::BTreeMap<u32, usize> = Default::default();
    for info in h.tensors.values() {
        let elements: u64 = info.dims.iter().product();
        match ggml_type_block(info.ggml_type) {
            Some((block, bytes)) => {
                let e = sized.entry(info.ggml_type).or_insert((0, 0));
                e.0 += 1;
                e.1 += elements / block * bytes;
            }
            None => *unsized_types.entry(info.ggml_type).or_insert(0) += 1,
        }
    }
    let total: u64 = sized.values().map(|(_, b)| *b).sum();
    println!("   block types, {} tensors:", h.tensors.len());
    for (ty, (count, bytes)) in &sized {
        println!(
            "     {:<10} x{count:<4} {:>8.3} GiB  {:>5.1}%",
            ggml_type_name(*ty).unwrap_or("?"),
            *bytes as f64 / (1 << 30) as f64,
            *bytes as f64 / total as f64 * 100.0
        );
    }
    for (ty, count) in &unsized_types {
        println!("     UNSIZED type {ty} x{count}  <- no ggml_type_block row: NOT ingestible");
    }
    println!(
        "     sized total {:.3} GiB",
        total as f64 / (1 << 30) as f64
    );
    assert!(
        unsized_types.is_empty(),
        "unsized block types present: {unsized_types:?}"
    );
    sized.keys().copied().collect()
}

/// **THE HEADLINE, RE-DERIVED FROM A SECOND SOURCE.**
///
/// `ornith_config.rs` asserts this off the HF `config.json`; this asserts it
/// off llama.cpp's GGUF metadata. Two independent producers, two parsers, no
/// shared input, one `ArchConfig`.
#[test]
#[ignore = "network: reads a few MB off a 21.7 GB remote checkpoint"]
fn the_moe_header_derives_the_pinned_qwen36_baseline() {
    let h = fetch(MOE_Q4_K_M);
    let architecture = h.architecture().expect("general.architecture");
    println!("== Ornith-1.5-35B-A3B Q4_K_M ==");
    println!("   general.architecture = {architecture}");
    assert_eq!(architecture, "qwen35moe");
    assert_eq!(
        family_for_architecture(architecture),
        Some(ModelFamily::QwenGdnMoe)
    );

    let arch = arch_from_gguf(&h).expect("arch derives");

    // `block_count` is 41 and the trunk is 40: llama.cpp counts the
    // multi-token-prediction block. If this ever reads 41 the subtraction in
    // `arch_from_gguf` regressed, and the mask would call the head LINEAR.
    println!("   derived num_layers = {} (trunk)", arch.num_layers);
    assert_eq!(
        arch.num_layers, 40,
        "the MTP block must not count as a layer"
    );

    assert_eq!(
        arch,
        model_io::qwen_gdn_moe_35b_a3b(),
        "the GGUF-derived ArchConfig must equal the pinned qwen3_5_moe baseline"
    );

    assert_every_trunk_name_maps(&h, ModelFamily::QwenGdnMoe, arch.num_layers as usize);

    // The head is REPORTED rather than silently excluded, and its shape is
    // the reason `docs/MTP.md`'s head is not this one: its FFN is MoE.
    let head: Vec<&String> = h
        .tensors
        .keys()
        .filter(|n| is_head_block(n, arch.num_layers as usize))
        .collect();
    println!("   MTP head block: {} tensors", head.len());
    assert!(
        !head.is_empty(),
        "this checkpoint declares nextn_predict_layers 1 and must carry the block"
    );
    assert!(
        head.iter().any(|n| n.contains("nextn.eh_proj")),
        "the head must carry its concat projection"
    );
    assert!(
        head.iter().any(|n| n.contains("ffn_gate_exps")),
        "this head's FFN is MoE, which is what makes it a different shape \
         from the dense qwen3_5 head docs/MTP.md describes"
    );

    let types = report_block_types(&h);
    for ty in types {
        let name = ggml_type_name(ty).unwrap_or("?");
        assert!(
            matches!(name, "F32" | "Q4_K" | "Q6_K"),
            "unexpected block type {name}: this file was scoped as Q4_K/Q6_K/F32"
        );
    }

    // Gotcha 36: the multiplication that decides whether a checkpoint can
    // stream at all, done from the header BEFORE any download.
    let one_expert = 3 * arch.moe_intermediate_size * arch.hidden_size;
    println!(
        "   expert arithmetic: {} experts of ~{:.3} MiB (3 x {} x {} at ~4.5 bpw), \
         16 slots x {} layers ~= {:.2} GiB",
        arch.num_experts,
        one_expert as f64 * 4.5 / 8.0 / (1 << 20) as f64,
        arch.moe_intermediate_size,
        arch.hidden_size,
        arch.num_layers,
        16.0 * arch.num_layers as f64 * one_expert as f64 * 4.5 / 8.0 / (1 << 30) as f64
    );
}

/// The 9B: the FIRST published GGUF of the dense half, and therefore the
/// reason `("qwen35", QwenGdnDense)` is in `SUPPORTED_GGUF` at all.
#[test]
#[ignore = "network: reads a few MB off a 5.6 GB remote checkpoint"]
fn the_dense_header_derives_the_published_9b_shape() {
    let h = fetch(DENSE_Q4_K_M);
    let architecture = h.architecture().expect("general.architecture");
    println!("== Ornith-1.5-9B Q4_K_M ==");
    println!("   general.architecture = {architecture}");

    // The pair that must NOT collapse. `qwen35` is a prefix of `qwen35moe`,
    // and a `starts_with` lookup here would hand a dense file the MoE
    // baseline: 256 experts and a router in the flow.
    assert_eq!(architecture, "qwen35");
    assert_eq!(
        family_for_architecture(architecture),
        Some(ModelFamily::QwenGdnDense)
    );

    let arch = arch_from_gguf(&h).expect("arch derives");
    println!(
        "   {} layers, hidden {}, ffn {}, {} q over {} kv at head_dim {}",
        arch.num_layers,
        arch.hidden_size,
        arch.intermediate_size,
        arch.num_heads,
        arch.num_kv_heads,
        arch.head_dim
    );

    assert_eq!(arch.num_layers, 32);
    assert_eq!(arch.hidden_size, 4096);
    assert_eq!(arch.intermediate_size, 12288, "the DENSE FFN width");
    assert_eq!(arch.num_heads, 16);
    assert_eq!(arch.num_kv_heads, 4);
    assert_eq!(arch.head_dim, 256);
    assert_eq!(arch.vocab_size, 248_320);
    assert_eq!(arch.num_experts, 0, "the dense half has no routed experts");
    assert_eq!(arch.moe_intermediate_size, 0);

    // **THE GATED-DELTANET BLOCK, FIELD BY FIELD, AND THIS IS THE ASSERTION
    // THE FIRST DRAFT OF THIS FILE LACKED.** `arch_from_gguf` used to assign
    // `linear_attention` for the MoE half alone, so a dense file kept
    // `qwen_gdn_dense_27b()`'s `num_v_heads: 48` -- Bonsai-27B's, not this
    // checkpoint's 32. Nothing above catches it: the layer count, the widths
    // and the mask are all still right. It surfaced as a GEMV shape mismatch
    // on layer 0 after a five-minute stream, when it was knowable from these
    // five metadata keys.
    //
    // The MoE half cannot see this bug at all, because ITS baseline value
    // happens to equal the real one (AGENTS.md Gotcha 37).
    assert_eq!(arch.linear_attention.num_k_heads, 16, "ssm.group_count");
    assert_eq!(arch.linear_attention.num_v_heads, 32, "ssm.time_step_rank");
    assert_eq!(arch.linear_attention.key_head_dim, 128, "ssm.state_size");
    assert_eq!(
        arch.linear_attention.value_head_dim, 128,
        "ssm.inner_size / num_v_heads"
    );
    assert_eq!(arch.linear_attention.conv_kernel_size, 4, "ssm.conv_kernel");
    // What the dispatch actually reads: `2*Hk*Dk + Hv*Dv`. The real file's
    // `attn_qkv` is 8192 rows wide, and 48 v-heads would demand 10240.
    let qkv = 2 * arch.linear_attention.num_k_heads * arch.linear_attention.key_head_dim
        + arch.linear_attention.num_v_heads * arch.linear_attention.value_head_dim;
    println!("   derived gdn qkv_dim = {qkv}");
    assert_eq!(qkv, 8192, "must match the file's attn_qkv row count");

    // 24 gated-DeltaNet layers to 8 full ones. The mask comes from
    // `full_attention_interval`, the same key and rule the MoE half uses.
    assert_eq!(
        arch.full_attention_layer_mask
            .iter()
            .filter(|&&m| m == 2)
            .count(),
        24,
        "24 linear layers"
    );
    assert_eq!(
        arch.full_attention_layer_mask
            .iter()
            .filter(|&&m| m == 1)
            .count(),
        8,
        "8 full-attention layers"
    );

    // This checkpoint declares `mtp_num_hidden_layers: 1` in its HF config
    // and ships NO head: `block_count` is 32 with no `nextn_predict_layers`,
    // so the trunk is the whole file. A config key is a claim about the
    // architecture; the bytes are the authority.
    assert!(
        !h.tensors.keys().any(|n| n.contains("nextn")),
        "the 9B publishes no MTP head"
    );
    assert!(
        !h.tensors
            .keys()
            .any(|n| is_head_block(n, arch.num_layers as usize)),
        "no block above the trunk"
    );

    assert_every_trunk_name_maps(&h, ModelFamily::QwenGdnDense, arch.num_layers as usize);

    let types = report_block_types(&h);
    for ty in types {
        let name = ggml_type_name(ty).unwrap_or("?");
        assert!(
            matches!(name, "F32" | "Q4_K" | "Q6_K"),
            "unexpected block type {name}"
        );
    }
}
