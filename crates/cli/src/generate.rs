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
    model_io::arch_from_manifest_dir(model_dir).map_err(|e| e.to_string())
}
