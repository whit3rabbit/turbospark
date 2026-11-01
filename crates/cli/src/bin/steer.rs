//! `turbospark-steer`: trait steering. Extracts a directional control
//! vector from a model's own activations against a named contrast, and
//! checks the result before it is used.
//!
//! A SECOND binary rather than subcommands on `turbospark-check`, for the
//! same reason `turbospark-model` is one (`src/bin/model.rs`'s header):
//! `crates/invocation` is a pure, flat parser whose contract is "`--model`
//! and exactly one mode flag", and this verb set needs none of that shape.
//!
//! Nothing here ships or labels a direction. A trait pack (`steer_data/`)
//! ships PROMPTS; the direction is extracted from the user's own checkpoint
//! at run time and lands in a file the user chose, keeping the provenance
//! rule the app's steering surface states (`swift/TurboSparkApp`,
//! `docs/OBLITERATION.md`).
//!
//! Exit status: 0 success, 1 a valid command whose work was refused or
//! failed, 2 an invalid command line.

use std::path::PathBuf;
use std::process::ExitCode;

#[path = "../ui/mod.rs"]
mod ui;

mod steer_cmd;

const USAGE: &str = "\
turbospark-steer: trait steering (heretic-style presets, extracted on your machine)

USAGE:
    turbospark-steer packs
    turbospark-steer extract --model MODEL --trait ID [OPTIONS]
    turbospark-steer check --model MODEL --trait ID --vector PATH [OPTIONS]

COMMANDS:
    packs            list the built-in trait packs (contrastive prompt sets)
    extract          capture two prompt corpora and reduce them to one
                     control vector (.gguf) drawn from this model's own
                     activations; no weight byte is modified
    check            run the pack's held-out prompts unsteered and steered,
                     and print the keyword-marker rate both times

EXTRACT OPTIONS:
    --model M              install path or catalog alias (required)
    --trait ID             a built-in pack id (see `packs`)
    --positive-file PATH   custom corpus instead of a pack (one prompt per
    --negative-file PATH   line; both are required together)
    --out PATH             output .gguf (default <trait>-direction.gguf)
    --limit N              cap prompts per side (default: all)
    --alpha-budget X       stream-share budget for the printed ceiling
                           diagnostic (default 0.10)

CHECK OPTIONS:
    --model M              install path or catalog alias (required)
    --trait ID             the pack that supplies prompts and markers
    --vector PATH          the .gguf `extract` wrote (required)
    --mode MODE            ablate|add|clamp|renorm (default: the pack's
                           recommendation, or the file's declared mode)
    --scale X              steering strength (default: the pack's
                           recommendation, 0.3)
    --layers START:END     restrict to a 0-based inclusive layer band
    --limit N              cap eval prompts
    --max-new N            response budget per prompt (default 200)
    --prompts-file PATH    custom eval prompts (markers still come from the
                           pack)
    --system-file PATH     system prompt for the eval turns

EXAMPLES:
    turbospark-steer packs
    turbospark-steer extract --model qwen38-27b --trait refusals
    turbospark-steer check --model qwen38-27b --trait refusals \\
        --vector refusals-direction.gguf --scale 0.3

A trait pack ships prompts, never a direction: the vector is extracted
from YOUR checkpoint, so judge it by `check` and your own prompts. Start
at scale 0.3; past 0.6 an ablation can collapse generation entirely.
See docs/OBLITERATION.md for the measured story.
";

/// What went wrong, split the way `turbospark-check`'s exit statuses split:
/// a bad command line is 2, work that was refused or failed is 1.
enum Failure {
    Usage(String),
    Failed(String),
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    match dispatch(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(Failure::Usage(message)) => {
            eprintln!("{message}");
            eprintln!("\nturbospark-steer --help for usage");
            ExitCode::from(2)
        }
        Err(Failure::Failed(message)) => {
            ui::error(&message);
            ExitCode::FAILURE
        }
    }
}

fn dispatch(args: &[String]) -> Result<(), Failure> {
    let verb = args[0].as_str();
    let rest = &args[1..];
    match verb {
        "packs" => run_packs(rest).map_err(Failure::Usage),
        "extract" => steer_cmd::extract::run(parse_extract(rest).map_err(Failure::Usage)?)
            .map_err(Failure::Failed),
        "check" => steer_cmd::check::run(parse_check(rest).map_err(Failure::Usage)?)
            .map_err(Failure::Failed),
        other => Err(Failure::Usage(format!(
            "unknown verb '{other}'; expected packs, extract or check"
        ))),
    }
}

fn run_packs(args: &[String]) -> Result<(), String> {
    let mut json = false;
    for arg in args {
        match arg.as_str() {
            "--json" => json = true,
            other => return Err(format!("packs: unknown option '{other}'")),
        }
    }
    let packs = steer_cmd::packs::all()?;
    if json {
        // Hand-rolled rather than a serde serialize: the display shape is a
        // stable interface and the packs carry prompts that must not be
        // re-escaped into a different form than they ship in.
        let items: Vec<String> = packs
            .iter()
            .map(|p| {
                format!(
                    "  {{\n    \"id\": {},\n    \"title\": {},\n    \"positive_prompts\": {},\
                     \n    \"negative_prompts\": {},\n    \"eval_prompts\": {},\n    \
                     \"recommended_mode\": {},\n    \"recommended_scale\": {}\n  }}",
                    quote(&p.id),
                    quote(&p.title),
                    p.positive.prompts.len(),
                    p.negative.prompts.len(),
                    p.eval.prompts.len(),
                    quote(&p.recommended_mode),
                    p.recommended_scale
                )
            })
            .collect();
        println!("[\n{}\n]", items.join(",\n"));
    } else {
        for p in &packs {
            println!(
                "{}  {}\n    {}\n    positive {} / negative {} prompts, {} eval \
                 prompts, {} markers\n    recommended: {} at {}\n",
                p.id,
                p.title,
                text_wrap(&p.summary, 4),
                p.positive.prompts.len(),
                p.negative.prompts.len(),
                p.eval.prompts.len(),
                p.eval.markers.len(),
                p.recommended_mode,
                p.recommended_scale
            );
        }
    }
    Ok(())
}

fn quote(text: &str) -> String {
    serde_json::to_string(text).unwrap_or_else(|_| format!("\"{text}\""))
}

/// Indent-wrapped plain text for the packs listing.
fn text_wrap(text: &str, indent: usize) -> String {
    let width = 96usize.saturating_sub(indent);
    let mut out = String::new();
    let mut line_len = 0usize;
    for word in text.split_whitespace() {
        if line_len + word.len() + 1 > width {
            out.push_str("\n");
            out.push_str(&" ".repeat(indent));
            line_len = 0;
        } else if !out.is_empty() && !out.ends_with('\n') {
            out.push(' ');
            line_len += 1;
        }
        out.push_str(word);
        line_len += word.len();
    }
    out
}

fn pop_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .position(|w| w[0] == flag)
        .map(|i| args[i + 1].clone())
}

/// Every arg must be a known flag followed by its value. Bare values and
/// unknown flags are usage errors; a known flag at the end with no value
/// reaches `require_value`/`pop_value`, whose message names the flag.
fn check_flags(args: &[String], verb: &str, allowed: &[&str]) -> Result<(), String> {
    let mut i = 0usize;
    while i < args.len() {
        let arg = &args[i];
        if !arg.starts_with("--") {
            return Err(format!("{verb}: unexpected argument '{arg}'"));
        }
        if !allowed.contains(&arg.as_str()) {
            return Err(format!("{verb}: unknown option '{arg}'"));
        }
        i += 2;
    }
    Ok(())
}

fn require_value(args: &[String], flag: &str) -> Result<String, String> {
    pop_value(args, flag).ok_or_else(|| format!("{flag} needs a value"))
}

/// Parse and range-check a float flag.
fn parse_f32(value: &str, flag: &str, lo: f32, hi: f32) -> Result<f32, String> {
    let v: f32 = value
        .parse()
        .map_err(|_| format!("{flag}: '{value}' is not a number"))?;
    if !v.is_finite() || v < lo || v > hi {
        return Err(format!("{flag}: {v} is outside {lo}..={hi}"));
    }
    Ok(v)
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("{flag}: '{value}' is not a non-negative integer"))
}

fn parse_u32(value: &str, flag: &str) -> Result<u32, String> {
    value
        .parse::<u32>()
        .map_err(|_| format!("{flag}: '{value}' is not a non-negative integer"))
}

/// START:END, 0-based inclusive, START < END.
fn parse_layers(value: &str) -> Result<(u32, u32), String> {
    let (start, end) = value
        .split_once(':')
        .ok_or_else(|| "--layers: expected START:END".to_string())?;
    let start = parse_u32(start.trim(), "--layers")?;
    let end = parse_u32(end.trim(), "--layers")?;
    if start >= end {
        return Err(format!("--layers: START {start} must be below END {end}"));
    }
    Ok((start, end))
}

fn parse_extract(args: &[String]) -> Result<steer_cmd::extract::ExtractOptions, String> {
    check_flags(
        args,
        "extract",
        &[
            "--model",
            "--trait",
            "--out",
            "--limit",
            "--positive-file",
            "--negative-file",
            "--alpha-budget",
        ],
    )?;
    let model = require_value(args, "--model")?;
    let trait_id = pop_value(args, "--trait");
    let out = pop_value(args, "--out").map(PathBuf::from);
    let limit = pop_value(args, "--limit")
        .map(|v| parse_usize(&v, "--limit"))
        .transpose()?;
    let positive_file = pop_value(args, "--positive-file").map(PathBuf::from);
    let negative_file = pop_value(args, "--negative-file").map(PathBuf::from);
    let alpha_budget = match pop_value(args, "--alpha-budget") {
        Some(v) => parse_f32(&v, "--alpha-budget", 0.01, 1.0)? as f64,
        None => 0.10,
    };
    if trait_id.is_none() && (positive_file.is_none() || negative_file.is_none()) {
        return Err(
            "extract needs --trait ID, or both --positive-file and --negative-file".to_string(),
        );
    }
    if trait_id.is_some() && (positive_file.is_some() || negative_file.is_some()) {
        return Err(
            "extract: --trait and --positive-file/--negative-file are alternatives, \
             not a combination"
                .to_string(),
        );
    }
    Ok(steer_cmd::extract::ExtractOptions {
        model,
        trait_id,
        out,
        limit,
        positive_file,
        negative_file,
        alpha_budget,
    })
}

fn parse_check(args: &[String]) -> Result<steer_cmd::check::CheckOptions, String> {
    check_flags(
        args,
        "check",
        &[
            "--model",
            "--trait",
            "--vector",
            "--mode",
            "--scale",
            "--layers",
            "--limit",
            "--max-new",
            "--prompts-file",
            "--system-file",
        ],
    )?;
    let model = require_value(args, "--model")?;
    let trait_id = require_value(args, "--trait")?;
    let vector = PathBuf::from(require_value(args, "--vector")?);
    let mode = pop_value(args, "--mode");
    if let Some(mode) = &mode {
        if foundation::SteeringMode::parse(mode).is_none() {
            return Err(format!(
                "--mode: '{}' is not one of {}",
                mode,
                foundation::STEERING_MODE_NAMES.join(", ")
            ));
        }
    }
    let scale = pop_value(args, "--scale")
        .map(|v| parse_f32(&v, "--scale", 0.0, 2.0))
        .transpose()?;
    let layers = pop_value(args, "--layers")
        .map(|v| parse_layers(&v))
        .transpose()?;
    let limit = pop_value(args, "--limit")
        .map(|v| parse_usize(&v, "--limit"))
        .transpose()?;
    let max_new = match pop_value(args, "--max-new") {
        Some(v) => {
            let n = parse_u32(&v, "--max-new")?;
            if n < 8 {
                return Err("--max-new: at least 8 so a response can carry a marker".to_string());
            }
            n.min(2048)
        }
        None => 200,
    };
    let prompts_file = pop_value(args, "--prompts-file").map(PathBuf::from);
    let system_file = pop_value(args, "--system-file").map(PathBuf::from);
    Ok(steer_cmd::check::CheckOptions {
        model,
        trait_id,
        vector,
        mode,
        scale,
        layers,
        limit,
        max_new,
        prompts_file,
        system_file,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two-verb grammar's guards: a missing required flag, an unknown
    /// option, a bad range, and the trait-versus-files exclusivity must all
    /// be USAGE failures (exit 2) rather than runtime refusals (exit 1),
    /// matching `turbospark-invocation`'s split.
    #[test]
    fn usage_errors_are_named_before_any_model_is_touched() {
        // Slices are verb-less: dispatch strips the verb before parsing.
        assert!(parse_extract(&[]).is_err());
        assert!(parse_extract(&["--model".into(), "m".into()]).is_err());
        assert!(parse_extract(&[
            "--model".into(),
            "m".into(),
            "--trait".into(),
            "refusals".into(),
            "--frobnicate".into(),
        ])
        .is_err());
        assert!(parse_extract(&[
            "--model".into(),
            "m".into(),
            "--trait".into(),
            "refusals".into(),
            "--positive-file".into(),
            "p.txt".into(),
        ])
        .is_err());
        // A bare value with no flag is a usage error, not a silent model name.
        assert!(parse_extract(&["--model".into(), "m".into(), "stray".into()]).is_err());
        assert!(parse_check(&[
            "--model".into(),
            "m".into(),
            "--trait".into(),
            "refusals".into(),
            "--layers".into(),
            "40:20".into(),
        ])
        .is_err());
        assert!(parse_check(&[
            "--model".into(),
            "m".into(),
            "--trait".into(),
            "refusals".into(),
            "--scale".into(),
            "9.0".into(),
        ])
        .is_err());
        assert!(parse_check(&[
            "--model".into(),
            "m".into(),
            "--trait".into(),
            "refusals".into(),
            "--mode".into(),
            "transmogrify".into(),
        ])
        .is_err());
        assert!(parse_check(&[
            "--model".into(),
            "m".into(),
            "--trait".into(),
            "refusals".into(),
            "--max-new".into(),
            "4".into(),
        ])
        .is_err());
        // The valid minimum for both verbs must parse.
        assert!(parse_extract(&[
            "--model".into(),
            "m".into(),
            "--trait".into(),
            "refusals".into()
        ])
        .is_ok());
        assert!(parse_check(&[
            "--model".into(),
            "m".into(),
            "--trait".into(),
            "refusals".into(),
            "--vector".into(),
            "d.gguf".into()
        ])
        .is_ok());
    }

    /// START:END parsing: ordered, 0-based, integer.
    #[test]
    fn layers_parse_orders_and_refuses_reversed() {
        assert_eq!(parse_layers("1:64").expect("parses"), (1, 64));
        assert!(parse_layers("64:1").is_err());
        assert!(parse_layers("40").is_err());
        assert!(parse_layers("10:").is_err());
        assert!(parse_layers("-1:20").is_err());
    }

    /// The packs listing's summary wrap must not split a word and must
    /// honor the indent for every continuation line.
    #[test]
    fn text_wrap_indents_continuations() {
        let wrapped = text_wrap(
            "Extracts the direction that separates prompts which tend to draw a \
             refusal from ordinary helpful requests on this checkpoint.",
            4,
        );
        let lines: Vec<&str> = wrapped.split('\n').collect();
        assert!(lines.len() >= 2, "long text wraps");
        for line in &lines[1..] {
            assert!(line.starts_with("    "), "continuation is indented: {line}");
        }
    }
}
