//! Real generation, wired to `RealForwardRunner` (macOS/GPU only). Loads a
//! `.gturbo` install from `request.model`, peeks its `manifest.json` for
//! `vocabSize`/`numLayers` (the only two dimensions
//! `repack::tiny_gemma4_arch` needs — every other field is pinned to
//! Gemma 4's own baseline values, see that function's docs), and opens it
//! with `RealForwardRunner`. Only works for installs shaped like
//! `repack::build_synthetic_gemma4_install`'s output (dense, all-full-
//! attention, Gemma-4-baseline non-shape fields) — the same scope
//! restriction `RealForwardRunner::open` itself enforces; a production
//! checkpoint (MoE and/or hybrid-attention) is rejected with a clear
//! error, not a crash. The tokenizer is expected to live alongside the
//! `.gturbo` files in the same directory (the usual HF checkpoint
//! bundling convention). Only `Mode::Prompt` is supported; `MessagesFile`
//! and `Chat` need chat-template rendering this integration does not do
//! yet.

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
    let vocab_size = value["arch"]["vocabSize"]
        .as_i64()
        .ok_or("manifest.json: arch.vocabSize missing or not an integer")?;
    let num_layers = value["arch"]["numLayers"]
        .as_i64()
        .ok_or("manifest.json: arch.numLayers missing or not an integer")?;
    Ok(repack::tiny_gemma4_arch(vocab_size, num_layers))
}
