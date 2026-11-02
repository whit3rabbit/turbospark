//! The `check` verb: run a trait pack's held-out prompts twice (unsteered,
//! then steered with the extracted vector) and print the keyword-marker rate
//! both times.
//!
//! This is a before/after readout, not a judge: no second model scores
//! anything, and the marker lists are word lists rather than evaluations.
//! Heretic's KeywordRate scorer is the idea; the lists here are written for
//! this port. A response counts once if ANY marker matches, case
//! insensitively. Greedy decoding makes the two arms reproducible.

use std::path::PathBuf;

use runtime::{
    DraftPolicies, ExpertCacheSlots, GenerationConfig, RateControl, RealForwardRunner,
    SteeringPolicy, SteeringVector,
};
use selection::ShapingConfig;

use super::packs;
use super::walk;

pub struct CheckOptions {
    pub model: String,
    pub trait_id: String,
    pub vector: PathBuf,
    pub mode: Option<String>,
    pub scale: Option<f32>,
    pub layers: Option<(u32, u32)>,
    pub limit: Option<usize>,
    pub max_new: u32,
    pub prompts_file: Option<PathBuf>,
    pub system_file: Option<PathBuf>,
}

/// Lowercased once at load; scoring is a plain substring match.
fn score(text: &str, markers: &[String]) -> bool {
    let lower = text.to_lowercase();
    markers.iter().any(|m| lower.contains(m.as_str()))
}

/// Generate one prompt's response greedily and return the text.
fn generate_text(
    runner: &mut RealForwardRunner,
    model: &walk::ResolvedModel,
    system: Option<&str>,
    prompt: &str,
    max_new: u32,
) -> Result<(String, usize), String> {
    let ids = walk::render_prompt(model, system, prompt)?;
    if ids.len() + max_new as usize >= model.max_context as usize {
        return Err(format!(
            "prompt renders to {} tokens; with max-new {max_new} it exceeds the \
             capture window {}",
            ids.len(),
            model.max_context
        ));
    }
    let shaping = ShapingConfig::new(0.0001, 1, Some(1.0), 1.0, None).map_err(|e| e.to_string())?;
    let config = GenerationConfig {
        shaping,
        max_new_tokens: max_new,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: RateControl::default(),
    };
    let vocab_size = runner.vocab_size();
    let mut text = String::new();
    let mut new_tokens = 0usize;
    runtime::run_raw_completion(
        runner,
        &model.tokenizer,
        &ids,
        &config,
        model.max_context,
        vocab_size,
        |ev| {
            if let runtime::RawDecodeProgress::Token { delta, .. } = ev {
                text.push_str(&delta);
                new_tokens += 1;
            }
        },
    )
    .map_err(|e| e.to_string())?;
    Ok((text, new_tokens))
}

/// Run one arm (one open) over every eval prompt.
fn run_arm(
    model: &walk::ResolvedModel,
    steering: Option<SteeringPolicy>,
    system: Option<&str>,
    prompts: &[String],
    max_new: u32,
) -> Result<Vec<(String, usize)>, String> {
    let mut runner = match steering {
        Some(policy) => RealForwardRunner::open_with_slot_policy_speculation_and_steering(
            &model.dir,
            model.arch.clone(),
            model.max_context as usize,
            ExpertCacheSlots::Auto,
            DraftPolicies::off(),
            policy,
        ),
        None => RealForwardRunner::open_with_slot_policy(
            &model.dir,
            model.arch.clone(),
            model.max_context as usize,
            ExpertCacheSlots::Auto,
        ),
    }
    .map_err(|e| e.to_string())?;
    if let Some(line) = runner.steering_line() {
        eprintln!("{line}");
    }
    let mut out = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        out.push(generate_text(&mut runner, model, system, prompt, max_new)?);
    }
    Ok(out)
}

pub fn run(opts: CheckOptions) -> Result<(), String> {
    let pack = packs::by_id(&opts.trait_id)?;
    let mut prompts = match &opts.prompts_file {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| format!("--prompts-file {}: {e}", path.display()))?;
            text.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>()
        }
        None => pack.eval.prompts.clone(),
    };
    if prompts.len() < 3 {
        return Err(format!(
            "{} eval prompts found; at least 3 are required for a rate worth reading",
            prompts.len()
        ));
    }
    if let Some(limit) = opts.limit.filter(|n| *n > 0) {
        prompts.truncate(limit);
    }
    let markers = pack
        .eval
        .markers
        .iter()
        .map(|m| m.to_lowercase())
        .collect::<Vec<_>>();
    let system = match &opts.system_file {
        Some(path) => Some(
            std::fs::read_to_string(path)
                .map_err(|e| format!("--system-file {}: {e}", path.display()))?,
        ),
        None => None,
    };

    let model = walk::resolve_model(&opts.model)?;
    let mut set = repack::control_vector::load_control_vector(&opts.vector)
        .map_err(|e| format!("--steering {}: {e}", opts.vector.display()))?;
    let declared = set.declared_mode;
    if let Some((start, end)) = opts.layers {
        set.restrict_to_range(start as usize, end as usize);
    }
    // The flag wins over the file's declared mode; the pack's recommendation
    // is the default when neither says anything.
    let mode = opts
        .mode
        .as_deref()
        .and_then(foundation::SteeringMode::parse)
        .or(declared)
        .or_else(|| foundation::SteeringMode::parse(&pack.recommended_mode))
        .expect("pack validation guarantees a parseable recommendation");
    let scale = opts.scale.unwrap_or(pack.recommended_scale);
    let steering = SteeringPolicy {
        vectors: vec![SteeringVector {
            set,
            mode,
            alpha: scale,
        }],
        target: 0.0,
        gate_threshold: 0.0,
    };

    println!(
        "checking '{}' against {} ({}, {} at scale {})",
        pack.id,
        model.dir.display(),
        opts.vector.display(),
        mode.as_str(),
        scale
    );

    let off_arm = run_arm(&model, None, system.as_deref(), &prompts, opts.max_new)
        .map_err(|e| format!("unsteered arm: {e}"))?;
    let on_arm = run_arm(
        &model,
        Some(steering),
        system.as_deref(),
        &prompts,
        opts.max_new,
    )
    .map_err(|e| format!("steered arm: {e}"))?;

    println!(
        "\n  {:>3}  {:<44}  {:>5}  {:>5}",
        "#", "prompt", "OFF", "ON"
    );
    for (i, prompt) in prompts.iter().enumerate() {
        let off_hit = score(&off_arm[i].0, &markers);
        let on_hit = score(&on_arm[i].0, &markers);
        println!(
            "  {i:>3}  {:<44}  {:>5}  {:>5}",
            truncate(prompt, 44),
            if off_hit { "yes" } else { "no" },
            if on_hit { "yes" } else { "no" }
        );
    }
    let off_count = off_arm.iter().filter(|(t, _)| score(t, &markers)).count();
    let on_count = on_arm.iter().filter(|(t, _)| score(t, &markers)).count();
    println!(
        "\nunsteered: {off_count}/{} prompts matched {} markers ({:.0}%)",
        prompts.len(),
        pack.id,
        100.0 * off_count as f64 / prompts.len() as f64
    );
    println!(
        "steered:   {on_count}/{} prompts matched ({:.0}%)",
        prompts.len(),
        100.0 * on_count as f64 / prompts.len() as f64
    );

    // The collapse proxy from docs/OBLITERATION.md: a steered run that stops
    // almost immediately is the documented failure mode of too-strong
    // ablation, and a marker rate of zero on a collapsed arm is a PERFECT
    // score on a destroyed model. REPORTED, never asserted away.
    let short: Vec<usize> = on_arm
        .iter()
        .enumerate()
        .filter(|(_, (_, n))| *n < 5)
        .map(|(i, _)| i)
        .collect();
    if !short.is_empty() {
        eprintln!(
            "warning: steered responses {} produced fewer than 5 tokens; the edit is \
             likely too strong at this scale (the documented collapse mode). Step the \
             scale down and re-run.",
            short
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    if off_count == 0 {
        println!(
            "\nthe unsteered arm already matches no markers, so this check cannot \
             show improvement; judge by reading the two texts instead"
        );
    } else if on_count < off_count {
        println!(
            "\nmarker rate fell {off_count} -> {on_count}. Read a few paired \
             responses before trusting the direction; a word list is a smoke \
             signal, not a verdict."
        );
    } else if on_count == off_count {
        println!(
            "\nmarker rate unchanged at this scale. Raise --scale toward {:.1} and \
             re-run, or accept that this direction does not move the trait on \
             this checkpoint (an ordinary outcome).",
            (scale + 0.2).min(0.6)
        );
    }
    Ok(())
}

fn truncate(text: &str, width: usize) -> String {
    let mut out: String = text.chars().take(width).collect();
    if text.chars().count() > width {
        out.push_str("...");
    }
    out.replace('\n', " ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scorer is a case-insensitive substring match over any marker;
    /// pin both directions because the trait rows in the output table are
    /// this function's verdicts.
    #[test]
    fn scoring_is_case_insensitive_substring() {
        let markers: Vec<String> = vec!["great question".to_string(), "sorry".to_string()];
        assert!(score("What a GREAT question!", &markers));
        assert!(score("I'm SORRY, but...", &markers));
        assert!(!score("Here is the answer: 42.", &markers));
        assert!(!score("", &markers));
        assert!(score("sorry", &markers));
    }

    /// A prompt longer than the display width truncates with an ellipsis and
    /// never panics on multibyte input (prompts are ASCII today, but the
    /// --prompts-file override is user text).
    #[test]
    fn truncation_is_char_safe() {
        assert_eq!(truncate("short", 44), "short");
        let long = "x".repeat(60);
        let cut = truncate(&long, 44);
        assert!(cut.ends_with("..."));
        assert_eq!(cut.chars().count(), 47);
        let multibyte = "é".repeat(50);
        assert!(truncate(&multibyte, 44).ends_with("..."));
    }
}
