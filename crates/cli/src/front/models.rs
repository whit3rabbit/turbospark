//! Model names, `run`, `pull` and `rm` for the `turbospark` front door.
//!
//! The front door owns the user's spelling of a model (a tag like
//! `qwen3.6:35b`, an alias, or a path) and translates it to the alias the
//! peer binaries take. Every install and delete still runs inside
//! `turbospark-model`, so there is one implementation of the walk, its
//! free-space check and its confirmation prompt.

use std::io::{IsTerminal, Read, Write};
use std::process::{Command, ExitCode, Stdio};

use catalog::{Catalog, EntryKind, Store};

use super::{exit_code_of, find_peer_binary, run_peer};

/// Tells `turbospark-check` to skip its resolved-request printout
/// (`src/main.rs`). Set by every front-door path that calls it, so `run`
/// prints the answer and not twenty lines of settings first.
pub const HIDE_RESOLVED_REQUEST_ENV: &str = "TURBOSPARK_HIDE_RESOLVED_REQUEST";

/// `turbospark-check` with the front door's environment.
pub fn check_command() -> Command {
    let mut command = Command::new(find_peer_binary("turbospark-check"));
    command.env(HIDE_RESOLVED_REQUEST_ENV, "1");
    command
}

/// Runs `turbospark-check` with stdio inherited; its status code.
pub fn run_check(args: &[String]) -> i32 {
    match check_command().args(args).status() {
        Ok(status) => super::status_code(status),
        Err(e) => {
            ui::error(&format!("cannot run turbospark-check: {e}"));
            1
        }
    }
}
use crate::ui;

/// What a user-typed model name turned out to be.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolved {
    /// A path, or a name only `turbospark-check` can judge: passed through.
    Verbatim(String),
    /// A curated row, by alias, and whether it is installed.
    Catalog { alias: String, installed: bool },
}

/// Paths are the caller's: `turbospark-check` resolves a directory before
/// any alias, and so does this.
pub fn looks_like_path(name: &str) -> bool {
    name.contains('/') || name.starts_with('.') || name.starts_with('~')
}

pub fn open_catalog() -> Option<(Store, Catalog)> {
    let store = Store::default_store().ok()?;
    let catalog = Catalog::load(store.root()).ok()?;
    Some((store, catalog))
}

pub fn resolve(name: &str, store: &Store, catalog: &Catalog) -> Resolved {
    if looks_like_path(name) || store.resolve(name).is_some() {
        return Resolved::Verbatim(name.to_string());
    }
    match catalog.resolve_name(name) {
        Some(entry) => Resolved::Catalog {
            alias: entry.alias.clone(),
            installed: store.resolve(&entry.alias).is_some(),
        },
        None => Resolved::Verbatim(name.to_string()),
    }
}

/// The alias for `name` when it is a tag, else `name` unchanged. For flags
/// like `serve --model` whose value is otherwise passed through.
pub fn alias_for(name: &str) -> String {
    let Some((store, catalog)) = open_catalog() else {
        return name.to_string();
    };
    match resolve(name, &store, &catalog) {
        Resolved::Catalog { alias, .. } => alias,
        Resolved::Verbatim(name) => name,
    }
}

/// Rewrites the value of every `--model X` / `--model=X` in `args` through
/// [`alias_for`].
pub fn rewrite_model_flags(args: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(args.len());
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == "--model" {
            out.push(arg.clone());
            if let Some(value) = iter.next() {
                out.push(alias_for(value));
            }
        } else if let Some(value) = arg.strip_prefix("--model=") {
            out.push(format!("--model={}", alias_for(value)));
        } else {
            out.push(arg.clone());
        }
    }
    out
}

fn interactive() -> bool {
    std::io::stdin().is_terminal() && std::io::stderr().is_terminal()
}

/// `[Y/n]` on stderr. Anything but an explicit no is a yes.
fn confirm(question: &str) -> bool {
    let paint = ui::Paint::err();
    eprint!("  {} {question} {} ", paint.accent("?"), paint.dim("[Y/n]"));
    let _ = std::io::stderr().flush();
    let mut answer = String::new();
    if std::io::stdin().read_line(&mut answer).is_err() {
        return false;
    }
    !matches!(answer.trim().to_lowercase().as_str(), "n" | "no")
}

/// Makes sure `name` can be opened, pulling a curated model first the way
/// `ollama run` does. Returns the `--model` value to hand
/// `turbospark-check`, or the exit code to stop with.
pub fn ensure_installed(name: &str, yes: bool) -> Result<String, ExitCode> {
    let Some((store, catalog)) = open_catalog() else {
        return Ok(name.to_string());
    };
    match resolve(name, &store, &catalog) {
        Resolved::Verbatim(name) => {
            if !looks_like_path(&name) && store.resolve(&name).is_none() {
                ui::error(&unknown_model(&catalog, &name));
                return Err(ExitCode::FAILURE);
            }
            Ok(name)
        }
        Resolved::Catalog {
            alias,
            installed: true,
        } => Ok(alias),
        Resolved::Catalog {
            alias,
            installed: false,
        } => {
            let entry = catalog.get(&alias).expect("resolved from this catalog");
            if entry.kind != EntryKind::Model {
                ui::error(&format!(
                    "{alias} is a {} and cannot run on its own",
                    entry.kind.as_str()
                ));
                return Err(ExitCode::FAILURE);
            }
            let size = catalog::human_bytes(entry.download_bytes);
            if !yes && interactive() {
                if !confirm(&format!("{alias} is not installed. Download {size} now?")) {
                    ui::note(&format!(
                        "nothing downloaded. `turbospark pull {alias}` later."
                    ));
                    return Err(ExitCode::from(130));
                }
            } else {
                ui::note(&format!(
                    "{alias} is not installed: downloading {size} first"
                ));
            }
            let code = run_peer("turbospark-model", &["pull".to_string(), alias.clone()]);
            if code != 0 {
                return Err(exit_code_of(code));
            }
            Ok(alias)
        }
    }
}

/// The message for a name that is neither installed nor in the catalog.
pub fn unknown_model(catalog: &Catalog, name: &str) -> String {
    let stem = name.split(':').next().unwrap_or(name);
    let near: Vec<String> = catalog
        .find(stem)
        .iter()
        .map(|e| e.alias.clone())
        .take(6)
        .collect();
    if near.is_empty() {
        format!("no model named {name:?}. `turbospark search` lists the catalog, `turbospark ls` what is installed.")
    } else {
        format!(
            "no model named {name:?}. Close: {}. `turbospark search` lists the catalog.",
            near.join(", ")
        )
    }
}

// ---------------------------------------------------------------- run

/// The flags `turbospark run` reads itself. Everything else goes to
/// `turbospark-check` untouched.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct RunPlan {
    pub model: String,
    /// Encode the prompt verbatim, without the chat template.
    pub raw: bool,
    pub yes: bool,
    pub prompt: Option<String>,
    /// The caller named a mode (`--chat`, `--prompt`, `--messages-file`)
    /// and gets exactly that.
    pub explicit_mode: Vec<String>,
    pub passthrough: Vec<String>,
}

/// Splits `run`'s arguments. The first non-flag after the model is the
/// prompt; the rest are `turbospark-check` flags, so a flag's VALUE has to
/// be recognized as one: `--temperature 0.7` must not become the prompt.
pub fn plan_run(model: &str, rest: &[String]) -> RunPlan {
    let mut plan = RunPlan {
        model: model.to_string(),
        ..RunPlan::default()
    };
    let mut i = 0;
    while i < rest.len() {
        let arg = &rest[i];
        match arg.as_str() {
            "--raw" => plan.raw = true,
            "--yes" | "-y" => plan.yes = true,
            "--chat" => plan.explicit_mode.push(arg.clone()),
            "--prompt" | "--messages-file" => {
                plan.explicit_mode.push(arg.clone());
                if let Some(value) = rest.get(i + 1) {
                    plan.explicit_mode.push(value.clone());
                    i += 1;
                }
            }
            flag if flag.starts_with('-') => {
                plan.passthrough.push(arg.clone());
                // A flag spelled `--name=value` carries its own value, and
                // a bare switch is followed by another flag or nothing.
                if !flag.contains('=') && takes_value(flag) {
                    if let Some(value) = rest.get(i + 1) {
                        plan.passthrough.push(value.clone());
                        i += 1;
                    }
                }
            }
            _ if plan.prompt.is_none() && plan.explicit_mode.is_empty() => {
                plan.prompt = Some(arg.clone())
            }
            _ => plan.passthrough.push(arg.clone()),
        }
        i += 1;
    }
    plan
}

/// Whether `turbospark-check` reads the token after `flag` as its value,
/// from `turbospark-invocation`'s own declaration table so the two cannot
/// drift. An unknown flag is taken as a switch: the engine refuses it
/// anyway, and guessing "valued" would swallow the prompt after it.
fn takes_value(flag: &str) -> bool {
    invocation::OPTIONS
        .iter()
        .find(|decl| decl.flag == flag)
        .is_some_and(|decl| decl.takes_value)
}

/// The JSON conversation `--messages-file` reads: one user turn.
pub fn one_turn(prompt: &str) -> String {
    serde_json::json!([{ "role": "user", "content": prompt }]).to_string()
}

/// The `turbospark-check` command line for `plan`, and the conversation to
/// write to its stdin when there is one.
pub fn check_args(plan: &RunPlan, model: &str) -> (Vec<String>, Option<String>) {
    let mut args = vec!["--model".to_string(), model.to_string()];
    let mut stdin_json = None;
    if !plan.explicit_mode.is_empty() {
        args.extend(plan.explicit_mode.iter().cloned());
    } else if let Some(prompt) = &plan.prompt {
        if plan.raw {
            args.extend(["--prompt".to_string(), prompt.clone()]);
        } else {
            args.extend(["--messages-file".to_string(), "/dev/stdin".to_string()]);
            stdin_json = Some(one_turn(prompt));
        }
    } else {
        args.push("--chat".to_string());
    }
    args.extend(plan.passthrough.iter().cloned());
    (args, stdin_json)
}

/// `turbospark run MODEL [PROMPT]`.
///
/// A prompt goes through the model's CHAT TEMPLATE, as one user turn fed to
/// `turbospark-check --messages-file` on its stdin. The old mapping to
/// `--prompt` encoded the text verbatim, so an instruction-tuned model
/// continued the user's sentence and read as a broken forward pass (root
/// AGENTS.md). `--raw` keeps that path for base models and kernel checks.
pub fn run(model: &str, rest: &[String]) -> ExitCode {
    let mut plan = plan_run(model, rest);
    // `echo text | turbospark run m` is a one-shot prompt, like ollama.
    if plan.prompt.is_none() && plan.explicit_mode.is_empty() && !std::io::stdin().is_terminal() {
        let mut text = String::new();
        if std::io::stdin().read_to_string(&mut text).is_ok() && !text.trim().is_empty() {
            plan.prompt = Some(text);
        }
    }
    let model = match ensure_installed(&plan.model, plan.yes) {
        Ok(model) => model,
        Err(code) => return code,
    };
    let (args, stdin_json) = check_args(&plan, &model);
    let Some(json) = stdin_json else {
        return exit_code_of(run_check(&args));
    };
    let mut child = match check_command().args(&args).stdin(Stdio::piped()).spawn() {
        Ok(child) => child,
        Err(e) => {
            ui::error(&format!("cannot start turbospark-check: {e}"));
            return ExitCode::FAILURE;
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        // A write error means the child exited before reading; its own
        // status says why.
        let _ = stdin.write_all(json.as_bytes());
    }
    match child.wait() {
        Ok(status) => exit_code_of(super::status_code(status)),
        Err(e) => {
            ui::error(&format!("waiting for turbospark-check: {e}"));
            ExitCode::FAILURE
        }
    }
}

// ---------------------------------------------------------------- pull, rm

/// The `turbospark-model` arguments that install `name`, by what kind of
/// model it is. `None` when nothing knows the name.
pub fn pull_args(name: &str, catalog: &Catalog, extra: &[String]) -> Option<Vec<String>> {
    let mut args: Vec<String> = if let Some(entry) = catalog.resolve_name(name) {
        let verb = match entry.kind {
            EntryKind::Model => "pull",
            EntryKind::VisionTower => "pull-vision",
        };
        vec![verb.to_string(), entry.alias.clone()]
    } else if catalog::AudioCatalog::embedded()
        .ok()
        .is_some_and(|audio| audio.get(name).is_some())
    {
        vec!["pull-audio".to_string(), name.to_string()]
    } else if catalog::ImageCatalog::embedded()
        .ok()
        .is_some_and(|image| image.get(name).is_some())
    {
        vec![
            "pull-image".to_string(),
            "--alias".to_string(),
            name.to_string(),
        ]
    } else if name.split('@').next().unwrap_or(name).split('/').count() == 2 {
        // An `owner/repo[@rev]` the probe decides on. It needs an alias;
        // the repo's own name is the obvious one unless the caller gave one.
        let mut args = vec!["pull".to_string(), "--repo".to_string(), name.to_string()];
        if !extra
            .iter()
            .any(|a| a == "--alias" || a.starts_with("--alias="))
        {
            let repo = name.split('@').next().unwrap_or(name);
            let alias = repo.rsplit('/').next().unwrap_or(repo).to_lowercase();
            args.extend(["--alias".to_string(), alias]);
        }
        args
    } else {
        return None;
    };
    args.extend(extra.iter().cloned());
    Some(args)
}

pub fn pull(names: &[String], extra: &[String]) -> ExitCode {
    let Some((_, catalog)) = open_catalog() else {
        ui::error("cannot read the model store; set TURBOSPARK_HOME or check ~/.turbospark");
        return ExitCode::FAILURE;
    };
    let mut plans = Vec::with_capacity(names.len());
    // Every name is checked before the first byte moves, so a typo in the
    // third name does not surface after two long downloads.
    for name in names {
        match pull_args(name, &catalog, extra) {
            Some(args) => plans.push(args),
            None => {
                ui::error(&unknown_model(&catalog, name));
                return ExitCode::from(2);
            }
        }
    }
    for args in &plans {
        let code = run_peer("turbospark-model", args);
        if code != 0 {
            return exit_code_of(code);
        }
    }
    ExitCode::SUCCESS
}

pub fn rm(names: &[String], yes: bool) -> ExitCode {
    for name in names {
        let alias = alias_for(name);
        let mut args = vec!["rm".to_string(), alias];
        if yes {
            args.push("--yes".to_string());
        }
        let code = run_peer("turbospark-model", &args);
        if code != 0 {
            return exit_code_of(code);
        }
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_positional_prompt_is_the_prompt_and_flag_values_are_not() {
        let plan = plan_run(
            "gemma4",
            &strings(&["--temperature", "0.7", "hello there", "--max-new", "32"]),
        );
        assert_eq!(plan.prompt.as_deref(), Some("hello there"));
        assert_eq!(
            plan.passthrough,
            strings(&["--temperature", "0.7", "--max-new", "32"])
        );
        assert!(!plan.raw);
    }

    #[test]
    fn switches_do_not_swallow_the_prompt() {
        let plan = plan_run("gemma4", &strings(&["--quiet", "hi"]));
        assert_eq!(plan.prompt.as_deref(), Some("hi"));
        let plan = plan_run("gemma4", &strings(&["--raw", "-y", "hi"]));
        assert!(plan.raw && plan.yes);
        assert_eq!(plan.prompt.as_deref(), Some("hi"));
    }

    #[test]
    fn an_explicit_mode_is_kept_and_nothing_becomes_a_prompt() {
        let plan = plan_run("m", &strings(&["--messages-file", "c.json", "extra"]));
        assert_eq!(plan.explicit_mode, strings(&["--messages-file", "c.json"]));
        assert_eq!(plan.prompt, None);
        assert_eq!(plan.passthrough, strings(&["extra"]));
    }

    #[test]
    fn a_prompt_goes_through_the_chat_template_unless_raw() {
        let plan = plan_run("m", &strings(&["hi", "--max-new", "8"]));
        let (args, stdin) = check_args(&plan, "alias");
        assert_eq!(
            args,
            strings(&[
                "--model",
                "alias",
                "--messages-file",
                "/dev/stdin",
                "--max-new",
                "8"
            ])
        );
        assert_eq!(stdin.as_deref(), Some(one_turn("hi").as_str()));

        let raw = plan_run("m", &strings(&["hi", "--raw"]));
        let (args, stdin) = check_args(&raw, "alias");
        assert_eq!(args, strings(&["--model", "alias", "--prompt", "hi"]));
        assert_eq!(stdin, None);

        let chat = plan_run("m", &[]);
        assert_eq!(
            check_args(&chat, "a").0,
            strings(&["--model", "a", "--chat"])
        );
    }

    #[test]
    fn one_turn_is_a_user_message_the_engine_parses() {
        let json = one_turn("say \"hi\"\nplease");
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value[0]["role"], "user");
        assert_eq!(value[0]["content"], "say \"hi\"\nplease");
        assert_eq!(value.as_array().unwrap().len(), 1);
    }

    #[test]
    fn pull_routes_by_kind_and_resolves_tags() {
        let catalog = Catalog::embedded().unwrap();
        assert_eq!(
            pull_args("qwen3.6:35b", &catalog, &[]).unwrap(),
            strings(&["pull", "qwen36"])
        );
        assert_eq!(
            pull_args("qwen38-vision-tower", &catalog, &[]).unwrap(),
            strings(&["pull-vision", "qwen38-vision-tower"])
        );
        assert_eq!(
            pull_args("Org/Some-Model@abc", &catalog, &[]).unwrap(),
            strings(&[
                "pull",
                "--repo",
                "Org/Some-Model@abc",
                "--alias",
                "some-model"
            ])
        );
        assert_eq!(
            pull_args("org/m", &catalog, &strings(&["--alias", "mine"])).unwrap(),
            strings(&["pull", "--repo", "org/m", "--alias", "mine"])
        );
        assert!(pull_args("no-such-model", &catalog, &[]).is_none());
    }

    #[test]
    fn model_flags_are_rewritten_from_tags_and_paths_are_left_alone() {
        let args = strings(&["--port", "9000", "--model", "./x.gturbo"]);
        assert_eq!(rewrite_model_flags(&args), args);
    }
}
