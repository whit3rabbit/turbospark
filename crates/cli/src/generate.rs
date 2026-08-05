//! Real generation, wired to `RealForwardRunner` (macOS/GPU only). Loads a
//! `.gturbo` install from `request.model`, reconstructs its full
//! `ArchConfig` from `manifest.json`'s own `arch` object (shape fields
//! read as written; family-extension fields fall back to Gemma 4's
//! baseline values, the same fallback rule `arch_validation` applies to
//! omitted manifest fields), and opens it with `RealForwardRunner`. This
//! covers both the synthetic short-name installs and real Gemma 4
//! installs repacked by `repack::write_gemma4_install` (verbatim
//! checkpoint tensor naming, MoE, mixed SWA/full attention); anything the
//! runner does not support (linear/compressed layers, non-Gemma families)
//! is rejected with a clear error, not a crash. The tokenizer is expected
//! to live alongside the `.gturbo` files in the same directory (the usual
//! HF checkpoint bundling convention). Only `Mode::Prompt` is supported;
//! `MessagesFile` and `Chat` need chat-template rendering this
//! integration does not do yet.

use std::io::Write;
use std::path::Path;

use invocation::{InvocationRequest, Mode};
use runtime::{run_raw_completion, GenerationConfig, RawDecodeProgress, RealForwardRunner};
use selection::ShapingConfig;
use tokenizer::MfTokenizer;

pub fn try_generate(request: &InvocationRequest) {
    let prompt = match &request.mode {
        Mode::Prompt(text) => text,
        Mode::MessagesFile(_) | Mode::Chat => {
            eprintln!(
                "note: real generation only supports --prompt mode in this build; \
                 messages-file and chat modes need chat-template wiring not done yet"
            );
            return;
        }
    };

    let model_dir = Path::new(&request.model);
    let arch = match peek_arch(model_dir) {
        Ok(arch) => arch,
        Err(e) => {
            eprintln!("note: not attempting real generation: {e}");
            return;
        }
    };

    let tokenizer = match MfTokenizer::load_from_dir(model_dir) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "note: not attempting real generation: failed to load a tokenizer \
                 from {}: {e}",
                model_dir.display()
            );
            return;
        }
    };

    let mut runner = match RealForwardRunner::open(model_dir, arch) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("note: not attempting real generation: {e}");
            return;
        }
    };

    let shaping = match ShapingConfig::new(
        request.temperature,
        request.top_k,
        Some(request.top_p),
        request.repetition_penalty,
        request.seed,
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("note: not attempting real generation: {e}");
            return;
        }
    };
    let config = GenerationConfig {
        shaping,
        max_new_tokens: request.max_new,
        stop_strings: request.stop.clone(),
        extra_stop_tokens: Vec::new(),
    };

    let prompt_ids = tokenizer.encode(prompt, true);
    let vocab_size = tokenizer.vocab_size;

    println!("generating (real forward pass, synthetic/untrained weights):");
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let result = run_raw_completion(
        &mut runner,
        &tokenizer,
        &prompt_ids,
        &config,
        request.max_context,
        vocab_size,
        |event| {
            if let RawDecodeProgress::Token { delta, .. } = event {
                let _ = write!(out, "{delta}");
                let _ = out.flush();
            }
        },
    );
    println!();

    match result {
        Ok(r) => println!(
            "note: {} prompt tokens, {} generated, stop reason {:?}",
            r.prompt_tokens, r.new_tokens, r.reason
        ),
        Err(e) => eprintln!("generation failed: {e}"),
    }
}

fn peek_arch(model_dir: &Path) -> Result<model_io::ArchConfig, String> {
    let manifest_path = model_dir.join("manifest.json");
    let bytes = std::fs::read(&manifest_path)
        .map_err(|e| format!("no manifest.json at {}: {e}", manifest_path.display()))?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("manifest.json: {e}"))?;
    let m: model_io::ManifestArch = serde_json::from_value(value["arch"].clone())
        .map_err(|e| format!("manifest.json arch: {e}"))?;

    // Start from the Gemma 4 baseline (the manifest's own fallback rule
    // for omitted family-extension fields) and overwrite every shape
    // field with what the manifest actually says.
    let mut arch = repack::tiny_gemma4_arch(m.vocab_size, m.num_layers);
    arch.hidden_size = m.hidden_size;
    arch.intermediate_size = m.ffn_intermediate;
    arch.moe_intermediate_size = m.moe_intermediate_size;
    arch.num_heads = m.num_heads;
    arch.num_kv_heads = m.num_kv_heads;
    arch.num_full_kv_heads = m.num_full_kv_heads;
    arch.head_dim = m.head_dim;
    arch.full_head_dim = m.full_head_dim;
    arch.sliding_window = m.sliding_window;
    arch.final_logit_softcap = m.final_logit_softcap;
    arch.rope_theta = m.rope_theta;
    arch.full_rope_theta = m.full_rope_theta;
    arch.partial_rotary_factor = m.partial_rotary_factor;
    arch.num_experts = m.num_experts;
    arch.top_k_experts = m.top_k_experts;
    arch.tie_word_embeddings = m.tie_word_embeddings;
    arch.attention_k_eq_v = m.attention_k_eq_v;
    arch.hidden_activation = m.hidden_activation.clone();
    arch.full_attention_layer_mask = m
        .full_attention_layer_mask
        .iter()
        .map(|&v| v as u8)
        .collect();
    if let Some(scale) = m.attention_scale {
        arch.attention_scale = scale;
    }
    if m.family.as_deref().is_some_and(|f| f != "gemma4") {
        return Err(format!(
            "manifest family {:?} is not supported by real generation yet",
            m.family
        ));
    }
    Ok(arch)
}
