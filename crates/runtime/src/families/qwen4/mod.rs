//! The real-checkpoint `qwen4_exp` (Qwen3.8-Flash-Next) decode flow for
//! [`RealForwardRunner`](crate::real_forward::RealForwardRunner).
//!
//! **A SEVENTH FLOW, not a variant of `families/qwen/`.** The two share a
//! GDN-or-attention-per-layer shape and a shared-expert-gated MoE, and
//! diverge in everything else that matters: this family's residual stream
//! is `hc_count * hidden_size` wide (10240, not 2560) and every layer
//! replaces its plain residual add with a hyper-connection read/inject
//! pair; there is no `input_layernorm` or `post_attention_layernorm`
//! tensor anywhere (the hyper-connection's own `hc_norm` is the pre-norm
//! for its sublayer); one layer carries a 51.2B-parameter hashed n-gram
//! embedding table read by 16 host-side row lookups a token; every norm in
//! the model is CENTERED except the GDN gated norm's SIGMOID variant
//! (`docs/QWEN4_PHASE0.md` item 9 -- the inverse of `muse_glimmer`'s
//! assignment). Sharing `families/qwen/`'s flow for the parts that
//! coincide would mean threading a `hc: Option<...>` and a wide-vs-narrow
//! residual width through code four OTHER families and the MTP head
//! already depend on staying exactly as it is (`crates/runtime` Gotcha
//! 11's argument, one family over).
//!
//! **VERIFICATION NOTE, READ BEFORE TRUSTING ANY FORMULA BELOW.** This
//! family's Phase 0 fact-finding (`docs/QWEN4_PHASE0.md`) itself carried a
//! wrong pseudocode for the PLE hash's multiplier pairing until it was
//! checked against `transformers`' actual `modular_qwen4_exp.py` source
//! directly -- a plausible-looking formula that silently dropped the
//! current token from every bigram hash. The hyper-connection formula
//! below WAS cross-checked the same way and matches its reference exactly;
//! nothing else in this module header has been. Re-verify a formula
//! against source before extending code built from it, rather than
//! trusting this file's own summary the way the original PLE section
//! could not be trusted.
//!
//! ## The decoder layer, as pseudocode
//!
//! `H = hidden_size` (2560), `C = hc_count` (4), so the residual carried
//! between layers is `C * H = 10240` wide. Ported field-for-field from
//! `docs/QWEN4_PHASE0.md` section 2, which is itself transcribed from
//! `modular_qwen4_exp.py`'s `Qwen4ExpDecoderLayer.forward` and
//! `Qwen4ExpModel.forward`.
//!
//! ```text
//! # entry, once: hidden = embed_tokens(ids).repeat(1, 1, C)   // replicated, not padded
//!
//! def layer(hidden):                        // hidden: [.., 10240]
//!     if layer_idx == 1:                    // the PLE layer, and only this one
//!         hidden = hidden + ple(hidden, input_ids)            // "## PLE" below
//!
//!     mixed, raw, inject_w = attn_hc(hidden)                  // "## Hyper-connections"
//!     out = gdn(mixed) if linear else qsa(mixed)               // "## GDN" / "## QSA" below
//!     hidden = raw + (out[.., None, :] * inject_w[.., :, None]).flatten()
//!
//!     mixed, raw, inject_w = mlp_hc(hidden)
//!     out = moe(mixed)                                         // "## MoE" below
//!     hidden = raw + (out[.., None, :] * inject_w[.., :, None]).flatten()
//!     return hidden
//!
//! # exit, once: hidden = hyper_connection_mixer(hidden)   // collapses 10240 -> 2560,
//! #                                                        // INCLUDES its own hc_norm
//! #             logits = lm_head(hidden)                  // no softcap, no separate final norm
//! ```
//!
//! **THE INJECTION TRAP.** `raw` is the UN-normalized hyper input (the wide
//! stream as it stood before this sublayer's `hc_norm`); the mix that
//! produced `out` read the NORMALIZED one. Both come back from one
//! `attn_hc`/`mlp_hc` call for exactly that reason. Swapping them decodes
//! fluently and is a different model (`docs/QWEN4_PHASE0.md` section 2).
//!
//! ## Hyper-connections (`attn_hc` / `mlp_hc`, `hc.rs`)
//!
//! ```text
//! raw          = hyper_input                                    // [.., C*H]
//! normed       = hc_norm(raw)                       // grouped RMS, group=H, CENTERED
//! w = silu(input_mix_weight_down(normed) / C)                   // C*H -> hc_lowrank (320)
//! w = sigmoid(input_mix_weight_up(w))                           // 320 -> C*H
//! mixed        = (w.view(C,H) * normed.view(C,H)).mean(dim=C)   // [.., H], MEAN not sum
//! inject_w     = 2 * sigmoid(block_inject_weight(normed) / C)   // C*H -> C
//! return mixed, raw, inject_w
//! ```
//!
//! The FINAL `hyper_connection_mixer` (once, after layer 47) is the same
//! module with no `block_inject_weight`: it returns `mixed` alone, which
//! is already RMS-normed by the `hc_norm` inside it -- there is no separate
//! tensor for a final norm anywhere in this checkpoint.
//!
//! ## PLE (`ple.rs`, layer index 1 only)
//!
//! ```text
//! rows   = ngram_hash(token_history)              // 16 row ids, model_io::ngram_hash
//! emb    = concat(table[rows[h]] for h in 0..16)  // [.., 2560], host mmap + dequant
//! key    = norm_key(key_proj(emb)).view(C, H)     // 2560 -> 10240, grouped centered norm
//! value  = value_proj(emb)                        // 2560 -> 2560
//! query  = norm_query(hidden).view(C, H)          // hidden is the 10240 residual
//! gate   = (key * query).sum(-1) / sqrt(H)        // [.., C, 1]
//! gate   = sign(gate) * sqrt(max(|gate|, 1e-6))   // SIGNED sqrt, ple_gate kernel
//! gv     = sigmoid(gate) * value                  // broadcast to [.., C, H] = 10240
//! out    = gv.flatten() + silu(dilated_conv(norm_conv(gv.flatten())))  // dilation=3, k=4
//! ```
//!
//! ## GDN (`attn.rs`, mask-2 layers, 36 of 48)
//!
//! Identical shape to `families/qwen/attn.rs::encode_linear_block` --
//! fused in-proj, causal conv, per-head qk norm, delta recurrence -- with
//! TWO differences: `num_v_heads` is 48 here against 32, and the gated
//! output norm uses the SIGMOID variant (`FC_GDN_GATE_SIGMOID`,
//! `docs/QWEN4_PHASE0.md` item 0 finding 4) rather than silu. Not shared
//! code with that function: a different state type, and Gotcha 11's
//! argument against touching a verified shared flow for what looks like a
//! parameter change.
//!
//! ## QSA-as-dense-attention (`attn.rs`, mask-1 layers, 12 of 48)
//!
//! **NO INDEXER CODE IN THIS PORT.** `docs/QWEN4_PHASE0.md` section 5
//! proves from source (two independent references, same conclusion) that
//! below `indexer_budget` (2,048 tokens) QSA's block selection is a no-op
//! and the layer is bit-for-bit dense causal attention. This flow REFUSES
//! any context above that budget at open, which is what licenses never
//! reading `self_attn.indexer.*` at all -- not an approximation of QSA,
//! the exact function it computes under the budget. What runs is
//! `families/qwen/attn.rs::encode_full_attention_block`'s exact shape
//! (packed `[query; gate]` in `q_proj`, per-head `q_norm`/`k_norm`,
//! `rope_neox_subdim` at `rotary_dim = 64`), CENTERED norms, 24 q heads
//! over 2 kv, `head_dim` 256, `theta` 1e7.
//!
//! ## MoE (`moe.rs`, every layer)
//!
//! Softmax-before-topk with top-10 renormalization
//! (`docs/QWEN4_PHASE0.md` item 7) is algebraically IDENTICAL to this
//! port's existing `router_topk_gemma4` (top-k on raw logits, softmax over
//! only the selected set -- the two differ only in which order the
//! softmax normalizer and the top-k selection happen, and the normalizer
//! cancels in the renormalization either way). No router bias here (unlike
//! `gpt-oss`, where a bias WOULD break that equivalence), so the existing
//! function is reused unchanged. The gated shared expert SEEDS phase 2's
//! accumulator rather than being added to a finished routed sum, matching
//! `families/qwen/moe.rs`'s existing choice for the identical reason (FP
//! addition is not associative).
//!
//! ## Deliberately unsupported in this cut, refused by name at open
//!
//! Context above `indexer_budget` (QSA math above), an image prompt (this
//! checkpoint is a VLM and only the text tower is ingested, as `qwen3_5`
//! already does), chunked prefill, and speculative decoding. Sequential
//! decode only.

pub(crate) mod attn;
pub(crate) mod hc;
pub(crate) mod moe;
pub(crate) mod ple;
mod produce;
mod state;

pub(crate) use state::RealQwen4State;

/// The trunk's tensor-name prefix. Unlike `families/qwen`, there is no MTP
/// prefix constant here: this family's MTP head (`docs/QWEN4_PHASE0.md`
/// item 8) is out of scope for this cut, so nothing under `mtp.` is read.
pub(crate) const TRUNK_PREFIX: &str = "language_model.model";

pub(crate) const RMS_EPS: f32 = 1e-6;

pub(crate) fn layer_tensor(layer: usize, suffix: &str) -> String {
    format!("{TRUNK_PREFIX}.layers.{layer}.{suffix}")
}
