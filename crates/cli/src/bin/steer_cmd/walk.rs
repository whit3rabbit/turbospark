//! Model resolution and the capture corpus walk for `turbospark-steer`.
//!
//! The walk is the documented extraction recipe (`docs/OBLITERATION.md`,
//! Reproducing) driven in one process: `TURBOSPARK_RESID_CAPTURE` is set
//! before the open, `run_raw_completion` resets the producer at the start of
//! every generation (which re-arms the capture), and the capture keeps one
//! snapshot per generation at the last prompt token. A corpus of N prompts
//! therefore costs ONE model open, not N.

use std::path::{Path, PathBuf};

use compute::steering_extract::CaptureSet;
use model_io::ArchConfig;
use runtime::{ExpertCacheSlots, GenerationConfig, RateControl, RealForwardRunner};
use selection::ShapingConfig;
use tokenizer::{Message, MfTokenizer, ReasoningEffort, Role};

use super::capture_io;

/// The env var the capture hook reads at open
/// (`crates/runtime/src/resid_capture.rs`). Set around each side's walk and
/// removed again, so no later open in this process captures by accident.
const CAPTURE_ENV: &str = "TURBOSPARK_RESID_CAPTURE";

/// Ceiling for the capture walk's KV window. Extraction prompts are single
/// turns of one or two sentences, so the checkpoint's trained context
/// clamped here sizes the KV comfortably and keeps the open cheap.
const MAX_CAPTURE_CONTEXT: u32 = 4096;

/// A model resolved for capture or evaluation work.
pub struct ResolvedModel {
    pub dir: PathBuf,
    pub arch: ArchConfig,
    pub tokenizer: MfTokenizer,
    pub max_context: u32,
}

/// Resolve `--model` (an install path or a catalog alias) the way
/// `crates/cli/src/generate/session.rs` does: a path that exists wins, a
/// disconnected drive is named as such, and the arch comes from the
/// install's own manifest.
pub fn resolve_model(arg: &str) -> Result<ResolvedModel, String> {
    if let Some(message) = catalog::not_connected_message(arg, catalog::ModelModality::Text) {
        return Err(message);
    }
    let dir = catalog::resolve_model_arg(arg);
    let arch = repack::peek_manifest_arch(&dir)?;
    if !runtime::family_dispatches_steering(arch.family) {
        return Err(format!(
            "family {:?} does not dispatch steering (and its flows do not feed the \
             residual capture); refusing rather than extracting a direction from a \
             place nothing steers",
            arch.family
        ));
    }
    let tokenizer = MfTokenizer::load_from_dir(&dir)
        .map_err(|e| format!("failed to load a tokenizer from {}: {e}", dir.display()))?;
    let trained = repack::trained_context_meta::peek(&dir);
    let max_context = trained
        .unwrap_or(MAX_CAPTURE_CONTEXT)
        .min(MAX_CAPTURE_CONTEXT);
    Ok(ResolvedModel {
        dir,
        arch,
        tokenizer,
        max_context,
    })
}

/// Render one corpus prompt through the checkpoint's own chat template.
///
/// The capture MUST run through the template: a direction extracted from raw
/// text residuals would be a direction for a framing the chat session never
/// runs (AGENTS.md: judge a chat checkpoint only through its template).
pub fn render_prompt(
    model: &ResolvedModel,
    system: Option<&str>,
    prompt: &str,
) -> Result<Vec<i32>, String> {
    let messages = build_messages(system, prompt);
    let rendered = model
        .tokenizer
        .apply_chat_template_with_reasoning(&messages, ReasoningEffort::Off)
        .map_err(|e| format!("chat template: {e}"))?;
    Ok(model.tokenizer.encode(&rendered, false))
}

/// The turn list one corpus prompt renders from: the pack's optional system
/// message FIRST, the prompt second. The sycophancy pack conditions both
/// sides through system prompts, so a reordered list would extract the
/// direction of the wrong contrast.
fn build_messages(system: Option<&str>, prompt: &str) -> Vec<Message> {
    let mut messages = Vec::new();
    if let Some(system) = system {
        messages.push(Message::new(Role::System, system));
    }
    messages.push(Message::new(Role::User, prompt));
    messages
}

/// Refuse a rendered prompt at or above the capture window. Checked per
/// prompt before its forward pass (the open has already happened), so an
/// over-long corpus prompt fails with its index instead of an admission
/// error from deep inside the generation loop.
fn check_window(index: usize, tokens: usize, window: u32) -> Result<(), String> {
    if tokens >= window as usize {
        return Err(format!(
            "prompt {index} renders to {tokens} tokens, at or above the capture window \
             {window}; shorten the corpus or raise the trained context"
        ));
    }
    Ok(())
}

/// Capture one side's corpus: open once, run every prompt for one greedy
/// token, and return what the capture wrote.
///
/// Greedy at a near-zero temperature matches the documented capture command
/// (`--temperature 0.0001 --top-k 1`); the snapshot is taken at the last
/// prompt token before sampling, so the sampled token cannot change it, and
/// deterministic settings make a rerun comparable.
pub fn capture_side(
    model: &ResolvedModel,
    system: Option<&str>,
    prompts: &[String],
    capture_path: &Path,
    label: &str,
) -> Result<CaptureSet, String> {
    std::env::set_var(CAPTURE_ENV, capture_path);
    let result = capture_side_inner(model, system, prompts, capture_path, label);
    std::env::remove_var(CAPTURE_ENV);
    result
}

fn capture_side_inner(
    model: &ResolvedModel,
    system: Option<&str>,
    prompts: &[String],
    capture_path: &Path,
    label: &str,
) -> Result<CaptureSet, String> {
    // Pinned OFF for drafting and steering and OFF for KV quant, the same
    // pins `open_with_options` documents for measuring callers: an edit
    // running during a capture would extract a direction from the edited
    // model and hand it back as the model's own.
    let mut runner = RealForwardRunner::open_with_slot_policy(
        &model.dir,
        model.arch.clone(),
        model.max_context as usize,
        ExpertCacheSlots::Auto,
    )
    .map_err(|e| e.to_string())?;
    let shaping = ShapingConfig::new(0.0001, 1, Some(1.0), 1.0, None).map_err(|e| e.to_string())?;
    let config = GenerationConfig {
        shaping,
        max_new_tokens: 1,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: RateControl::default(),
    };
    let vocab_size = runner.vocab_size();
    for (i, prompt) in prompts.iter().enumerate() {
        let ids = render_prompt(model, system, prompt)?;
        check_window(i, ids.len(), model.max_context)?;
        if i % 8 == 0 {
            eprintln!("  [{label}] capturing prompt {}/{}", i + 1, prompts.len());
        }
        runtime::run_raw_completion(
            &mut runner,
            &model.tokenizer,
            &ids,
            &config,
            model.max_context,
            vocab_size,
            |_| {},
        )
        .map_err(|e| format!("prompt {i}: {e}"))?;
    }
    // The capture writes on drop, and writes NOTHING when no non-prefill
    // pass ran (a family guard or a fully-prefilled corpus), so dropping
    // here is the flush point, not a leak.
    drop(runner);
    let set = capture_io::read_capture(capture_path)
        .map_err(|e| format!("capture for {label} wrote nothing readable: {e}"))?;
    check_snapshot_count(
        label,
        set.rows.len(),
        prompts.len(),
        model.arch.family.as_str(),
    )?;
    Ok(set)
}

/// One snapshot per prompt, or refuse. The capture keys on the last prompt
/// token's non-prefill pass, so a family whose generation loop prefills the
/// whole prompt records nothing for some prompts; averaging a short set
/// would silently extract from a different corpus than the one named.
fn check_snapshot_count(
    label: &str,
    snapshots: usize,
    prompts: usize,
    family: &str,
) -> Result<(), String> {
    if snapshots != prompts {
        return Err(format!(
            "[{label}] capture holds {snapshots} snapshots for {prompts} prompts on family \
             {family}; the capture keys on the last prompt token's non-prefill pass, and \
             this family's generation loop did not take that pass for every prompt"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The walk itself is real-install territory (it opens Metal and a
    /// model). What is testable here is the pure half: the turn list
    /// `render_prompt` renders must put the pack's system message first.
    #[test]
    fn messages_put_system_first_and_prompt_second() {
        let system = "You are a helpful assistant.";
        let prompt = "What is the capital of Australia?";
        let messages = build_messages(Some(system), prompt);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[0].content.as_deref(), Some(system));
        assert_eq!(messages[1].content.as_deref(), Some(prompt));

        let bare = build_messages(None, prompt);
        assert_eq!(bare.len(), 1);
        assert_eq!(bare[0].role, Role::User);
    }

    /// A prompt AT the window is refused (the generated token needs a
    /// slot), one below it passes, and the error names the prompt index.
    #[test]
    fn window_refuses_at_and_above_the_capture_window() {
        assert!(check_window(0, 4095, 4096).is_ok());
        let err = check_window(7, 4096, 4096).expect_err("at the window");
        assert!(err.contains("prompt 7"), "names the index: {err}");
        assert!(check_window(0, 5000, 4096).is_err());
    }

    /// A short capture set is refused rather than averaged.
    #[test]
    fn snapshot_shortfall_is_refused() {
        assert!(check_snapshot_count("positive", 24, 24, "llama").is_ok());
        let err = check_snapshot_count("negative", 23, 24, "llama").expect_err("short set");
        assert!(err.contains("23 snapshots for 24 prompts"), "{err}");
    }
}
