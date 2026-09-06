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
    recommend                   what this machine should run, ranked
    pull <ALIAS>                install a curated model
    pull --repo <REPO>[@REV]    install any repo the probe accepts
    pull-vision <ALIAS>         install a curated vision-tower sidecar
    pull-vision --repo <REPO>[@REV] --alias <NAME>
                                install any repo's vision tower directly,
                                skipping the trunk probe (a tower has no
                                quantization block for it to check)
    path <ALIAS>                print an install directory, for scripts
    rm <ALIAS>                  delete an install
    auth [TOKEN]                manage Hugging Face credentials: print status,
                                set a token, or clear with --clear

OPTIONS:
    --out <DIR>                 install here instead of the default store
    --alias <NAME>              name a --repo pull (required for one)
    --file <NAME.gguf>          pick one file where a repo offers several,
                                or (with `pull-vision`) an explicit filename
                                for a repo whose shard index does not name
                                the vision tower's shard on its own
    --sidecar-repo <REPO>[@REV] take tokenizer files from another repo. A GGUF
                                carries llama.cpp's tokenizer, not an HF
                                tokenizer.json, so a GGUF pull needs this
    --reuse-trunk-from <ALIAS>  for a row naming an mtp source: read an
                                already-installed alias's resident trunk back
                                off disk instead of re-streaming it, so only
                                the head crosses the network. Refused unless
                                that install's recorded repo and revision
                                match this row's exactly
    --filter <TEXT>             substring match for `list`
    --context <N>               window to fit against for `recommend` (4096)
    --budget <BYTES>            override the memory probe for `recommend`
    --load-guard <TIER|BYTES>   how much of the machine a session may commit:
                                off, relaxed (default), balanced, strict, or a
                                size that caps what the engine allocates. MUST
                                match what the session will open with.
    --discover [N]              also rank the N most-downloaded GGUF repos on
                                Hugging Face, filtered through the probe
    --probe                     read every curated row's header too, which is
                                what turns `recommend`'s unknowns into
                                arithmetic. Slower: one header per row
    --hf-token <TOKEN>          Hugging Face API token override
    --clear                     clear saved Hugging Face token (for `auth`)
    --status                    verify and show authentication status (for `auth`)
    --set <TOKEN>               save Hugging Face token (for `auth`)
    --force                     install past a probe refusal
    --yes                       do not prompt before deleting
    --help                      print this text
    --version                   print the version

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
    pub context: Option<u32>,
    pub budget: Option<u64>,
    /// How much of the machine a session would be allowed to commit.
    ///
    /// **`recommend` MUST rank under the tier the session will OPEN with.**
    /// The two share a budget by construction, which is what makes a
    /// recommendation trustworthy; ranking under `relaxed` while sessions
    /// open under `strict` promises a fit the loader then refuses.
    pub load_guard: model_io::LoadGuard,
    pub discover: Option<usize>,
    pub probe: bool,
    pub alias: Option<String>,
    pub file: Option<String>,
    pub sidecar_repo: Option<String>,
    /// An already-installed alias to reuse the trunk's resident bytes from,
    /// instead of re-streaming them, when the row being pulled names an
    /// `mtp` source. Only meaningful with `pull`.
    pub reuse_trunk_from: Option<String>,
    pub filter: Option<String>,
    pub force: bool,
    pub yes: bool,
    pub hf_token: Option<String>,
    pub clear: bool,
    pub status: bool,
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
    // Ahead of `parse`, which would report it as an unknown command. Same
    // string as the other two binaries, from cargo rather than a literal:
    // every crate here inherits `version.workspace = true`, so this is the
    // version of whichever binary printed it.
    if command == "--version" || command == "-V" || command == "version" {
        println!("turbospark {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    let (positionals, options) = parse(&args[1..])?;
    let store = Store::default_store().map_err(Error::Failed)?;
    let catalog = Catalog::load(store.root()).map_err(Error::Failed)?;
    let client = Client::with_token(options.hf_token.clone());

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
            options.reject_unused(&["file", "sidecar-repo", "hf-token"])?;
            let target = one_positional(&positionals, "probe", "<REPO>[@REV]")?;
            let repo = RepoRef::parse(target).map_err(Error::Usage)?;
            let sidecars = parse_sidecar_repo(&options)?;
            model_cmd::probe(&client, &repo, options.file.as_deref(), sidecars.as_ref())
        }
        "recommend" => {
            options.reject_unused(&[
                "context",
                "budget",
                "load-guard",
                "discover",
                "probe",
                "hf-token",
            ])?;
            if !positionals.is_empty() {
                return Err(Error::Usage(
                    "recommend takes no arguments; it describes this machine".to_string(),
                ));
            }
            model_cmd::recommend(&catalog, &client, &options)
        }
        "pull" => {
            options.reject_unused(&[
                "out",
                "alias",
                "file",
                "sidecar-repo",
                "reuse-trunk-from",
                "force",
                "hf-token",
            ])?;
            model_cmd::pull(&catalog, &store, &client, &positionals, &options)
        }
        "pull-vision" => {
            options.reject_unused(&["out", "alias", "file", "force", "hf-token"])?;
            model_cmd::pull_vision(&catalog, &store, &client, &positionals, &options)
        }
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
        "auth" => {
            options.reject_unused(&["hf-token", "set", "clear", "status"])?;
            model_cmd::auth(&store, &positionals, &options)
        }
        other => Err(Error::Usage(format!("unknown command {other:?}"))),
    }
}

/// How many popular repositories a bare `--discover` scans.
const DEFAULT_DISCOVER_SCAN: usize = 20;

/// `36`, `36G`, `36GB`, `36GiB` -- all the same number, because the flag
/// exists to be typed by hand. A bare number is BYTES rather than gigabytes:
/// every other size in this tool's output is in bytes, and guessing the unit
/// on `--budget 36` would be a factor of a billion in whichever direction the
/// guess was wrong.
fn parse_bytes(text: &str) -> Option<u64> {
    let trimmed = text.trim();
    let digits_end = trimmed
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(trimmed.len());
    if digits_end == 0 {
        return None;
    }
    let (number, suffix) = trimmed.split_at(digits_end);
    let value: u64 = number.parse().ok()?;
    let scale: u64 = match suffix.trim().to_ascii_lowercase().as_str() {
        "" => 1,
        "k" | "kb" | "kib" => 1024,
        "m" | "mb" | "mib" => 1024 * 1024,
        "g" | "gb" | "gib" => 1024 * 1024 * 1024,
        _ => return None,
    };
    value.checked_mul(scale)
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
            "--reuse-trunk-from" => {
                options.reuse_trunk_from = Some(value_for(&mut index, "--reuse-trunk-from")?);
                seen.push("reuse-trunk-from");
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
            "--context" => {
                let value = value_for(&mut index, "--context")?;
                options.context =
                    Some(value.parse().map_err(|_| {
                        Error::Usage(format!("--context {value:?} is not a number"))
                    })?);
                seen.push("context");
            }
            // A tier word or a byte ceiling, one flag, for the reason
            // `crates/invocation`'s arm gives.
            "--load-guard" => {
                let value = value_for(&mut index, "--load-guard")?;
                options.load_guard = match model_io::LoadGuard::parse(&value) {
                    Some(g) => g,
                    None => match parse_bytes(&value) {
                        Some(n) if n > 0 => model_io::LoadGuard::Custom {
                            max_counted_bytes: n,
                        },
                        _ => {
                            return Err(Error::Usage(format!(
                                "--load-guard {value:?} is not off, relaxed, balanced, \
                                 strict or a size"
                            )))
                        }
                    },
                };
                seen.push("load-guard");
            }
            "--budget" => {
                let value = value_for(&mut index, "--budget")?;
                options.budget =
                    Some(parse_bytes(&value).ok_or_else(|| {
                        Error::Usage(format!("--budget {value:?} is not a size"))
                    })?);
                seen.push("budget");
            }
            "--discover" => {
                // The count is OPTIONAL, so a bare `--discover` has to not
                // swallow the next token. Nothing else in this parser has an
                // optional value, which is why it is spelled out here rather
                // than going through `value_for`.
                let count = match args.get(index + 1) {
                    Some(next) if next.parse::<usize>().is_ok() => {
                        index += 1;
                        next.parse().expect("checked")
                    }
                    _ => DEFAULT_DISCOVER_SCAN,
                };
                options.discover = Some(count);
                seen.push("discover");
            }
            "--probe" => {
                options.probe = true;
                seen.push("probe");
            }
            "--force" => {
                options.force = true;
                seen.push("force");
            }
            "--yes" | "-y" => {
                options.yes = true;
                seen.push("yes");
            }
            "--hf-token" => {
                options.hf_token = Some(value_for(&mut index, "--hf-token")?);
                seen.push("hf-token");
            }
            "--set" => {
                options.hf_token = Some(value_for(&mut index, "--set")?);
                seen.push("set");
            }
            "--clear" => {
                options.clear = true;
                seen.push("clear");
            }
            "--status" => {
                options.status = true;
                seen.push("status");
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
