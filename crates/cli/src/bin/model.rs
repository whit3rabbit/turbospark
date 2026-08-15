//! `turbospark-model`: the model catalog, the Hugging Face probe, and `pull`.
//!
//! A SECOND binary rather than subcommands on `turbospark-check`, and that is
//! a decision rather than convenience. `crates/invocation` is a pure, flat
//! option parser whose whole contract is "`--model` is required and exactly
//! one mode flag is set"; it carries a five-place rule for every new flag and
//! a hardcoded option-count assertion. A subcommand grammar does not belong
//! in it, and bending it into one would put a required `--model` in front of
//! a command whose entire job is that there is no model yet.
//!
//! This file is a shell: parse argv, call `turbospark-catalog`, print, pick
//! an exit code. Every decision it renders is computed there, the same way
//! `turbospark-check` delegates to `invocation`.

use std::process::ExitCode;

use catalog::{Catalog, Client, RepoRef, Store};

mod model_cmd;

const USAGE: &str = "\
turbospark-model -- find, inspect and install models

USAGE:
    turbospark-model <COMMAND> [OPTIONS]

COMMANDS:
    list [--filter TEXT]        curated models, marking the ones installed
    info <ALIAS>                one model in full, with its gate targets
    probe <REPO>[@REV]          what this engine makes of a Hugging Face repo,
                                reading headers only: no download
    pull <ALIAS>                install a curated model
    pull --repo <REPO>[@REV]    install any repo the probe accepts
    path <ALIAS>                print an install directory, for scripts
    rm <ALIAS>                  delete an install

OPTIONS:
    --out <DIR>                 install here instead of the default store
    --alias <NAME>              name a --repo pull (required for one)
    --file <NAME.gguf>          pick one file where a repo offers several
    --sidecar-repo <REPO>[@REV] take tokenizer files from another repo. A GGUF
                                carries llama.cpp's tokenizer, not an HF
                                tokenizer.json, so a GGUF pull needs this
    --filter <TEXT>             substring match for `list`
    --force                     install past a probe refusal
    --yes                       do not prompt before deleting
    --help                      print this text

ENVIRONMENT:
    TURBOSPARK_HOME             the store root (default ~/.turbospark)
    HF_TOKEN                    for gated repositories

NOTE: an install streams multi-GB weights and CANNOT RESUME. A failure
restarts the walk from the beginning.
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(Error::Usage(message)) => {
            eprintln!("{message}\n\n{USAGE}");
            ExitCode::from(2)
        }
        Err(Error::Failed(message)) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

/// The two failure kinds, which get different exit codes: a malformed
/// invocation is the caller's mistake (2, matching `turbospark-check`'s
/// invalid-invocation status) and everything else is a run that was asked for
/// correctly and did not work (1).
pub enum Error {
    Usage(String),
    Failed(String),
}

impl From<String> for Error {
    fn from(message: String) -> Self {
        Error::Failed(message)
    }
}

/// Parsed options, flat: every command reads the ones it needs and
/// [`Options::reject_unused`] refuses the rest, so a flag on the wrong
/// command is an error rather than a silent no-op.
#[derive(Debug, Default)]
pub struct Options {
    pub out: Option<String>,
    pub alias: Option<String>,
    pub file: Option<String>,
    pub sidecar_repo: Option<String>,
    pub filter: Option<String>,
    pub force: bool,
    pub yes: bool,
    /// Which options actually appeared, so [`Options::reject_unused`] can
    /// tell "not set" from "set to its default".
    seen: Vec<String>,
}

fn run(args: &[String]) -> Result<(), Error> {
    let Some(command) = args.first() else {
        println!("{USAGE}");
        return Ok(());
    };
    if command == "--help" || command == "-h" || command == "help" {
        println!("{USAGE}");
        return Ok(());
    }

    let (positionals, options) = parse(&args[1..])?;
    let store = Store::default_store().map_err(Error::Failed)?;
    let catalog = Catalog::load(store.root()).map_err(Error::Failed)?;
    let client = Client::new();

    match command.as_str() {
        "list" => {
            options.reject_unused(&["filter"])?;
            model_cmd::list(&catalog, &store, options.filter.as_deref());
            Ok(())
        }
        "info" => {
            options.reject_unused(&[])?;
            let alias = one_positional(&positionals, "info", "<ALIAS>")?;
            model_cmd::info(&catalog, &store, alias)
        }
        "probe" => {
            options.reject_unused(&["file", "sidecar-repo"])?;
            let target = one_positional(&positionals, "probe", "<REPO>[@REV]")?;
            let repo = RepoRef::parse(target).map_err(Error::Usage)?;
            let sidecars = parse_sidecar_repo(&options)?;
            model_cmd::probe(&client, &repo, options.file.as_deref(), sidecars.as_ref())
        }
        "pull" => model_cmd::pull(&catalog, &store, &client, &positionals, &options),
        "path" => {
            options.reject_unused(&[])?;
            let alias = one_positional(&positionals, "path", "<ALIAS>")?;
            model_cmd::path(&store, alias)
        }
        "rm" => {
            options.reject_unused(&["yes"])?;
            let alias = one_positional(&positionals, "rm", "<ALIAS>")?;
            model_cmd::remove(&store, alias, options.yes)
        }
        other => Err(Error::Usage(format!("unknown command {other:?}"))),
    }
}

fn parse_sidecar_repo(options: &Options) -> Result<Option<RepoRef>, Error> {
    match &options.sidecar_repo {
        Some(text) => RepoRef::parse(text).map(Some).map_err(Error::Usage),
        None => Ok(None),
    }
}

fn one_positional<'a>(
    positionals: &'a [String],
    command: &str,
    shape: &str,
) -> Result<&'a str, Error> {
    match positionals.len() {
        1 => Ok(&positionals[0]),
        0 => Err(Error::Usage(format!("{command} needs {shape}"))),
        n => Err(Error::Usage(format!(
            "{command} takes one argument, got {n}"
        ))),
    }
}

/// Split argv into positionals and options.
fn parse(args: &[String]) -> Result<(Vec<String>, Options), Error> {
    let mut positionals = Vec::new();
    let mut options = Options::default();
    let mut seen = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let token = &args[index];
        let value_for = |index: &mut usize, flag: &str| -> Result<String, Error> {
            *index += 1;
            args.get(*index)
                .cloned()
                .ok_or_else(|| Error::Usage(format!("{flag} needs a value")))
        };
        match token.as_str() {
            "--out" => {
                options.out = Some(value_for(&mut index, "--out")?);
                seen.push("out");
            }
            "--alias" => {
                options.alias = Some(value_for(&mut index, "--alias")?);
                seen.push("alias");
            }
            "--file" => {
                options.file = Some(value_for(&mut index, "--file")?);
                seen.push("file");
            }
            "--sidecar-repo" => {
                options.sidecar_repo = Some(value_for(&mut index, "--sidecar-repo")?);
                seen.push("sidecar-repo");
            }
            "--filter" => {
                options.filter = Some(value_for(&mut index, "--filter")?);
                seen.push("filter");
            }
            "--repo" => {
                // `--repo` is `pull`'s alternative to a positional alias, so
                // it lands in the positional list marked, keeping `pull`'s
                // two forms one code path.
                let value = value_for(&mut index, "--repo")?;
                positionals.push(format!("--repo={value}"));
            }
            "--force" => {
                options.force = true;
                seen.push("force");
            }
            "--yes" | "-y" => {
                options.yes = true;
                seen.push("yes");
            }
            "--help" | "-h" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other if other.starts_with('-') => {
                return Err(Error::Usage(format!("unknown option {other:?}")))
            }
            other => positionals.push(other.to_string()),
        }
        index += 1;
    }
    options.seen = seen.into_iter().map(str::to_string).collect();
    Ok((positionals, options))
}

impl Options {
    /// Refuse any option the current command does not read.
    ///
    /// Worth the few lines: the alternative is `--force` on `list` doing
    /// nothing, which reads as the command having considered and ignored it.
    fn reject_unused(&self, allowed: &[&str]) -> Result<(), Error> {
        for flag in &self.seen {
            if !allowed.contains(&flag.as_str()) {
                return Err(Error::Usage(format!(
                    "--{flag} does not apply to this command"
                )));
            }
        }
        Ok(())
    }
}
