//! `turbospark`: the one command a user types.
//!
//! A front door in the shape of ollama and kapitsa: `run`, `pull`, `ls`,
//! `rm`, `serve`, `launch`. It owns the command tree, help, model-name
//! spelling (tags like `qwen3.6:35b`) and the terminal presentation, and it
//! decides nothing about models: generation runs in `turbospark-check`,
//! installs and the catalog views in `turbospark-model`, the server in
//! `turbospark-server`, media in their own binaries. Those peers keep their
//! own command lines and exact output, since CI, the app bundle scripts and
//! docs call them directly.
//!
//! Exit status: 0 success, 1 a valid command that failed, 2 a usage error,
//! 130 when the user cancelled; a peer's own status is passed through.

use std::path::PathBuf;
use std::process::{Command, ExitCode, ExitStatus};

use clap::builder::styling::{AnsiColor, Styles};
use clap::{Args, CommandFactory, FromArgMatches, Parser, Subcommand};

#[path = "../agent.rs"]
mod agent;
#[path = "../daemon.rs"]
mod daemon;
#[path = "../ui/mod.rs"]
mod ui;

#[path = "../front/models.rs"]
mod models;
#[path = "../front/server.rs"]
mod server;

#[derive(Parser)]
#[command(
    name = "turbospark",
    version,
    about = "Run open models on Apple silicon",
    disable_help_subcommand = true,
    args_conflicts_with_subcommands = true
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Cmd>,
}

/// Arguments handed to a peer binary as they are, `--help` included.
#[derive(Args)]
struct Passthrough {
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    args: Vec<String>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Chat with a model, or answer one prompt and exit
    #[command(after_help = RUN_AFTER_HELP)]
    Run {
        /// A tag (qwen3.6:35b), an alias (qwen36) or an install path
        model: String,
        /// The prompt, then any turbospark-check flag (--temperature 0.7,
        /// --max-new 256, --image page.png, --raw, --yes)
        #[arg(
            trailing_var_arg = true,
            allow_hyphen_values = true,
            value_name = "PROMPT|FLAGS"
        )]
        rest: Vec<String>,
    },
    /// Download models: text, vision, image or audio, or any owner/repo
    Pull(PullArgs),
    /// Installed models
    #[command(visible_alias = "list")]
    Ls {
        #[arg(long)]
        json: bool,
        /// Only names containing this text
        #[arg(long)]
        filter: Option<String>,
    },
    /// Browse the model catalog
    Search {
        /// Match names, descriptions and repos
        text: Vec<String>,
        #[arg(long)]
        json: bool,
    },
    /// One model in full
    Show {
        name: String,
        #[arg(long)]
        json: bool,
        /// Add full revision, sidecars, gates and the install path
        #[arg(short, long)]
        verbose: bool,
    },
    /// Delete installed models
    #[command(visible_alias = "remove")]
    Rm {
        #[arg(required = true)]
        names: Vec<String>,
        /// Do not ask before deleting
        #[arg(short, long)]
        yes: bool,
    },
    /// This machine, and which models fit it
    Doctor {
        #[arg(long)]
        json: bool,
        /// Context window to plan for, in tokens
        #[arg(long, value_name = "TOKENS")]
        context: Option<u32>,
        /// Plan for this much memory instead of what the machine reports
        #[arg(long, value_name = "SIZE")]
        budget: Option<String>,
    },
    /// Serve OpenAI, Anthropic and Ollama APIs in this terminal
    #[command(disable_help_flag = true)]
    Serve(Passthrough),
    /// Start the server in the background (`start AGENT` launches an agent)
    #[command(disable_help_flag = true)]
    Start(Passthrough),
    /// Stop the background server
    Stop,
    /// Restart the background server, optionally with new options
    #[command(disable_help_flag = true)]
    Restart(Passthrough),
    /// Whether the background server is running, and its endpoints
    Status {
        #[arg(long)]
        json: bool,
    },
    /// The model the background server is running
    Ps {
        #[arg(long)]
        json: bool,
    },
    /// Start a coding agent on a local model: claude, codex, opencode,
    /// gemini, grok, hermes, openclaw, dsh
    #[command(disable_help_flag = true, after_help = LAUNCH_AFTER_HELP)]
    Launch {
        agent: String,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Generate an image (turbospark image --help)
    #[command(disable_help_flag = true)]
    Image(Passthrough),
    /// Generate music (turbospark music --help)
    #[command(disable_help_flag = true)]
    Music(Passthrough),
    /// Text to speech with Fish Audio Drama 3 (turbospark speak --help)
    #[command(disable_help_flag = true)]
    Speak(Passthrough),
    /// Speak in a cloned voice; needs --consent (turbospark clone --help)
    #[command(disable_help_flag = true)]
    Clone(Passthrough),
    /// Page images to text with Core ML, macOS 15+ (turbospark ocr --help)
    #[command(disable_help_flag = true)]
    Ocr(Passthrough),
    /// Rewrite a draft so it reads like a person wrote it
    #[command(disable_help_flag = true)]
    Humanize(Passthrough),
    /// Trait steering: extract a behavior direction from a model's own
    /// activations and check it (turbospark steer --help)
    #[command(disable_help_flag = true)]
    Steer(Passthrough),
    /// Throughput and memory benchmarks
    #[command(disable_help_flag = true)]
    Bench(Passthrough),
    /// Advanced model management: probe, path, auth, library, move, and
    /// every pull form (turbospark model --help)
    #[command(disable_help_flag = true)]
    Model(Passthrough),
    /// Shell completions: turbospark completions zsh > ~/.zfunc/_turbospark
    Completions { shell: clap_complete::Shell },

    // Earlier spellings, kept so scripts keep working. Hidden from help.
    #[command(hide = true, disable_help_flag = true)]
    Info(Passthrough),
    #[command(hide = true, disable_help_flag = true)]
    Recommend(Passthrough),
    #[command(hide = true, disable_help_flag = true)]
    Probe(Passthrough),
    #[command(hide = true, disable_help_flag = true)]
    Path(Passthrough),
    #[command(hide = true, disable_help_flag = true)]
    Auth(Passthrough),
    #[command(hide = true, name = "list-audio", disable_help_flag = true)]
    ListAudio(Passthrough),
    #[command(hide = true, name = "list-speech", disable_help_flag = true)]
    ListSpeech(Passthrough),
    #[command(hide = true, name = "pull-audio", disable_help_flag = true)]
    PullAudio(Passthrough),

    /// `turbospark MODEL AGENT`, or an unknown command.
    #[command(external_subcommand)]
    External(Vec<String>),
}

#[derive(Args)]
struct PullArgs {
    /// Tags, aliases, or owner/repo[@revision]
    #[arg(required = true)]
    names: Vec<String>,
    /// Install into this library (`turbospark model library list`)
    #[arg(long)]
    library: Option<String>,
    /// Install into this directory instead of the store
    #[arg(long, value_name = "DIR")]
    out: Option<String>,
    /// Name for an owner/repo pull (default: the repo's name)
    #[arg(long)]
    alias: Option<String>,
    /// Pick one GGUF file where a repo offers several
    #[arg(long, value_name = "NAME.gguf")]
    file: Option<String>,
    /// Take tokenizer files from another repo (needed for a GGUF pull)
    #[arg(long, value_name = "REPO[@REV]")]
    sidecar_repo: Option<String>,
    /// Accept one of an audio model's terms by id
    #[arg(long, value_name = "ID")]
    accept: Vec<String>,
    /// Install past a probe refusal
    #[arg(long)]
    force: bool,
}

impl PullArgs {
    fn extra(&self) -> Vec<String> {
        let mut extra = Vec::new();
        let mut valued = |flag: &str, value: &Option<String>| {
            if let Some(value) = value {
                extra.extend([flag.to_string(), value.clone()]);
            }
        };
        valued("--library", &self.library);
        valued("--out", &self.out);
        valued("--alias", &self.alias);
        valued("--file", &self.file);
        valued("--sidecar-repo", &self.sidecar_repo);
        for id in &self.accept {
            extra.extend(["--accept".to_string(), id.clone()]);
        }
        if self.force {
            extra.push("--force".to_string());
        }
        extra
    }
}

const RUN_AFTER_HELP: &str = "\
Examples:
  turbospark run qwen3.6                       chat in the terminal
  turbospark run qwen3.6 \"why is the sky blue\"  one answer, then exit
  cat notes.md | turbospark run gemma4          the piped text is the prompt
  turbospark run qwen3-vl \"read this\" --image page.png
  turbospark run tinyllama \"Once upon\" --raw    no chat template

A model that is not installed is downloaded first (asks in a terminal; --yes
skips the question). Every turbospark-check flag works here: run
`turbospark run MODEL --help` for the full list.";

const LAUNCH_AFTER_HELP: &str = "\
Examples:
  turbospark launch claude --model qwen3.6
  turbospark launch codex --model gemma4 -- --full-auto
  turbospark qwen3.6 claude                     the same, model first

--model starts (or restarts) the background server on that model.
--dry-run prints the launch without running it. Arguments after -- go to
the agent as they are.";

fn styles() -> Styles {
    Styles::styled()
        .header(AnsiColor::White.on_default().bold())
        .usage(AnsiColor::White.on_default().bold())
        .literal(AnsiColor::Green.on_default())
        .placeholder(AnsiColor::BrightBlack.on_default())
        .error(AnsiColor::Red.on_default().bold())
        .valid(AnsiColor::Green.on_default())
        .invalid(AnsiColor::Yellow.on_default())
}

/// The top-level help, grouped by what a user is doing. clap lists
/// subcommands in one flat block, so this one screen is written by hand;
/// every subcommand's own `--help` is still clap's.
fn root_help() -> String {
    let p = ui::Paint::out();
    let groups: &[(&str, &[(&str, &str)])] = &[
        (
            "Models",
            &[
                (
                    "run MODEL [PROMPT]",
                    "chat, or answer one prompt; downloads first",
                ),
                (
                    "pull NAME...",
                    "download models: text, vision, image, audio",
                ),
                ("ls", "installed models"),
                ("search [TEXT]", "the catalog"),
                ("show NAME", "one model in full"),
                ("rm NAME...", "delete installed models"),
                ("doctor", "this machine and what fits it"),
            ],
        ),
        (
            "Server and agents",
            &[
                (
                    "serve",
                    "OpenAI, Anthropic and Ollama APIs in this terminal",
                ),
                (
                    "start | stop | restart",
                    "the same server in the background",
                ),
                ("status | ps", "whether it runs, on what, and where"),
                ("launch AGENT", "claude, codex, opencode, gemini, grok, ..."),
            ],
        ),
        (
            "Create",
            &[
                ("image", "generate an image"),
                ("music", "generate music"),
                ("speak", "text to speech"),
                ("clone", "speech in a cloned voice (needs --consent)"),
                ("ocr PAGE...", "page images to text"),
                ("humanize MODEL DRAFT", "rewrite a draft to read as human"),
            ],
        ),
        (
            "More",
            &[
                ("bench", "throughput and memory benchmarks"),
                (
                    "model ...",
                    "probe, path, auth, library, move, every pull form",
                ),
                (
                    "steer ...",
                    "extract behavior directions from a model, check them",
                ),
                ("completions SHELL", "zsh, bash, fish, elvish or powershell"),
            ],
        ),
    ];
    let width = groups
        .iter()
        .flat_map(|(_, rows)| rows.iter().map(|(c, _)| c.len()))
        .max()
        .unwrap_or(0);
    let mut out = String::new();
    out.push_str(&format!(
        "{} {}\n{}\n\n{} turbospark [COMMAND] [OPTIONS]\n",
        p.bold("turbospark"),
        p.dim(env!("CARGO_PKG_VERSION")),
        "Run open models on Apple silicon: chat, serve an API, drive a coding agent.",
        p.bold("Usage:"),
    ));
    for (title, rows) in groups {
        out.push_str(&format!("\n{}\n", p.bold(title)));
        for (command, about) in *rows {
            out.push_str(&format!(
                "  {}{}  {about}\n",
                p.accent(command),
                " ".repeat(width - command.len())
            ));
        }
    }
    out.push_str(&format!(
        "\n{}\n",
        p.dim(
            "MODEL is a tag (qwen3.6:35b), an alias (qwen36) or an install path.\n\
             turbospark COMMAND --help for options. turbospark with no command shows\n\
             what is installed. NO_COLOR=1 turns color off."
        )
    ));
    out
}

// ---------------------------------------------------------------- peers

pub fn find_peer_binary(name: &str) -> PathBuf {
    if let Ok(mut exe) = std::env::current_exe() {
        exe.pop();
        let candidate = exe.join(name);
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from(name)
}

/// A child's status as a shell would report it: its code, or 128 plus the
/// signal that ended it (130 for Ctrl-C).
pub fn status_code(status: ExitStatus) -> i32 {
    if let Some(code) = status.code() {
        return code;
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return 128 + signal;
        }
    }
    1
}

pub fn exit_code_of(code: i32) -> ExitCode {
    ExitCode::from(u8::try_from(code).unwrap_or(1))
}

/// Runs a peer with stdio inherited and returns its status code.
pub fn run_peer(binary_name: &str, args: &[String]) -> i32 {
    let bin = find_peer_binary(binary_name);
    match Command::new(&bin).args(args).status() {
        Ok(status) => status_code(status),
        Err(e) => {
            ui::error(&format!(
                "cannot run {}: {e}. It ships beside `turbospark`; reinstall if it is missing.",
                bin.display()
            ));
            1
        }
    }
}

fn peer(binary_name: &str, args: &[String]) -> ExitCode {
    exit_code_of(run_peer(binary_name, args))
}

fn model_peer(verb: &str, args: &[String]) -> ExitCode {
    let mut all = vec![verb.to_string()];
    all.extend_from_slice(args);
    peer("turbospark-model", &all)
}

fn renamed(old: &str, new: &str) {
    ui::note(&format!("`turbospark {old}` is now `turbospark {new}`"));
}

fn or_fail(result: Result<(), String>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            ui::error(&e);
            ExitCode::FAILURE
        }
    }
}

// ---------------------------------------------------------------- commands

fn handle_image(args: &[String]) -> ExitCode {
    let mut image_args = Vec::with_capacity(args.len() + 2);
    if !matches!(args.first().map(String::as_str), Some("generate" | "pack")) {
        image_args.push("generate".to_string());
    }
    image_args.extend_from_slice(args);
    if args.is_empty() {
        image_args.push("--help".to_string());
    }
    peer(
        "turbospark-image",
        &models::rewrite_model_flags(&image_args),
    )
}

fn handle_humanize(args: &[String]) -> ExitCode {
    if args.len() < 2 || args[0] == "--help" || args[0] == "-h" {
        println!("Usage: turbospark humanize MODEL DRAFT [FLAGS]\n");
        println!("Rewrites a draft with the humanizer model (a raw-completion rewriter:");
        println!("no chat template, the model's own sampling recipe, EOS-only stop).");
        println!("DRAFT is a UTF-8 file path, or - to read stdin. The rewrite goes to");
        println!("stdout and the audit to stderr.\n");
        println!("Examples:");
        println!("  turbospark humanize humanizer draft.txt");
        println!("  cat draft.txt | turbospark humanize humanizer -");
        println!("  turbospark humanize humanizer draft.txt --max-new 512 --seed 3");
        return if args.len() < 2 && args.first().map(String::as_str) != Some("--help") {
            ExitCode::from(2)
        } else {
            ExitCode::SUCCESS
        };
    }
    // The model goes first, the draft becomes `--humanize-file`, and every
    // other flag passes through (`--max-new` and `--seed` are the ones the
    // flow reads; the sampling flags are forced by the flow).
    let model = match models::ensure_installed(&args[0], false) {
        Ok(model) => model,
        Err(code) => return code,
    };
    let mut check_args = vec!["--model".to_string(), model];
    check_args.push("--humanize-file".to_string());
    check_args.push(args[1].clone());
    check_args.extend_from_slice(&args[2..]);
    exit_code_of(models::run_check(&check_args))
}

fn handle_start(args: &[String]) -> ExitCode {
    if let Some(first) = args.first() {
        if agent::is_agent(first) {
            let rest = models::rewrite_model_flags(&args[1..]);
            return or_fail(agent::start_agent(first, &rest));
        }
    }
    if args.iter().any(|a| a == "--help" || a == "-h") {
        return peer("turbospark-server", &["--help".to_string()]);
    }
    let result = daemon::start(&models::rewrite_model_flags(args));
    if result.is_ok() {
        // The daemon's own lines name the pid and log; the card adds the
        // endpoints a client needs.
        return server::status(false);
    }
    or_fail(result)
}

/// Edit distance, for "did you mean" on an unknown command.
fn distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cur = row[j + 1];
            row[j + 1] = (prev + usize::from(ca != *cb)).min(row[j] + 1).min(cur + 1);
            prev = cur;
        }
    }
    row[b.len()]
}

fn handle_external(args: &[String]) -> ExitCode {
    let Some(first) = args.first() else {
        return ExitCode::from(2);
    };
    // `turbospark MODEL AGENT ...` is `launch AGENT --model MODEL ...`. Only
    // an AGENT in the second position triggers it, so `turbospark gemma4 hi`
    // stays an unknown command rather than becoming a launch.
    if let Some(agent_name) = args.get(1).filter(|a| agent::is_agent(a.as_str())) {
        let mut start_args = vec![agent_name.clone(), "--model".to_string(), first.to_string()];
        start_args.extend_from_slice(&args[2..]);
        return handle_start(&start_args);
    }
    let names: Vec<String> = Cli::command()
        .get_subcommands()
        .filter(|c| !c.is_hide_set())
        .map(|c| c.get_name().to_string())
        .collect();
    let near = names
        .iter()
        .map(|n| (distance(first, n), n))
        .filter(|(d, _)| *d <= 2)
        .min();
    let hint = match near {
        Some((_, name)) => format!(" Did you mean `turbospark {name}`?"),
        None if models::open_catalog().is_some_and(|(_, c)| c.resolve_name(first).is_some()) => {
            format!(" To chat with it: `turbospark run {first}`.")
        }
        None => " `turbospark --help` lists the commands.".to_string(),
    };
    ui::error(&format!("unknown command: {first}.{hint}"));
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let help = root_help();
    let command = Cli::command().styles(styles()).override_help(help);
    let matches = command.get_matches();
    let cli = match Cli::from_arg_matches(&matches) {
        Ok(cli) => cli,
        Err(e) => e.exit(),
    };
    let Some(command) = cli.command else {
        return model_peer("home", &[]);
    };
    match command {
        Cmd::Run { model, rest } => models::run(&model, &rest),
        Cmd::Pull(args) => models::pull(&args.names, &args.extra()),
        Cmd::Ls { json, filter } => {
            let mut args = Vec::new();
            if json {
                args.push("--json".to_string());
            }
            if let Some(filter) = filter {
                args.extend(["--filter".to_string(), filter]);
            }
            model_peer("ls", &args)
        }
        Cmd::Search { text, json } => {
            let mut args = text;
            if json {
                args.push("--json".to_string());
            }
            model_peer("search", &args)
        }
        Cmd::Show {
            name,
            json,
            verbose,
        } => {
            let mut args = vec![name];
            if json {
                args.push("--json".to_string());
            }
            if verbose {
                args.push("--verbose".to_string());
            }
            model_peer("show", &args)
        }
        Cmd::Rm { names, yes } => models::rm(&names, yes),
        Cmd::Doctor {
            json,
            context,
            budget,
        } => {
            let mut args = Vec::new();
            if json {
                args.push("--json".to_string());
            }
            if let Some(context) = context {
                args.extend(["--context".to_string(), context.to_string()]);
            }
            if let Some(budget) = budget {
                args.extend(["--budget".to_string(), budget]);
            }
            model_peer("doctor", &args)
        }
        Cmd::Serve(p) => peer("turbospark-server", &models::rewrite_model_flags(&p.args)),
        Cmd::Start(p) => handle_start(&p.args),
        Cmd::Stop => or_fail(daemon::stop()),
        Cmd::Restart(p) => or_fail(daemon::restart(&models::rewrite_model_flags(&p.args))),
        Cmd::Status { json } => server::status(json),
        Cmd::Ps { json } => server::ps(json),
        Cmd::Launch { agent: name, args } => {
            if !agent::is_agent(&name) {
                ui::error(&format!(
                    "unknown agent {name:?}. `turbospark launch --help` lists them."
                ));
                return ExitCode::from(2);
            }
            or_fail(agent::start_agent(
                &name,
                &models::rewrite_model_flags(&args),
            ))
        }
        Cmd::Image(p) => handle_image(&p.args),
        Cmd::Music(p) => peer("turbospark-music", &models::rewrite_model_flags(&p.args)),
        Cmd::Speak(p) => peer("turbospark-speak", &models::rewrite_model_flags(&p.args)),
        Cmd::Clone(p) => peer("turbospark-clone", &models::rewrite_model_flags(&p.args)),
        Cmd::Ocr(p) => peer("turbospark-ocr", &models::rewrite_model_flags(&p.args)),
        Cmd::Humanize(p) => handle_humanize(&p.args),
        Cmd::Steer(p) => peer("turbospark-steer", &p.args),
        Cmd::Bench(p) => peer("turbospark-bench", &p.args),
        Cmd::Model(p) => peer("turbospark-model", &p.args),
        Cmd::Completions { shell } => {
            let mut command = Cli::command();
            clap_complete::generate(shell, &mut command, "turbospark", &mut std::io::stdout());
            ExitCode::SUCCESS
        }
        Cmd::Info(p) => {
            renamed("info", "show");
            model_peer("info", &p.args)
        }
        Cmd::Recommend(p) => {
            renamed("recommend", "doctor");
            model_peer("recommend", &p.args)
        }
        Cmd::Probe(p) => model_peer("probe", &p.args),
        Cmd::Path(p) => model_peer("path", &p.args),
        Cmd::Auth(p) => model_peer("auth", &p.args),
        Cmd::ListAudio(p) => model_peer("list-audio", &p.args),
        Cmd::ListSpeech(p) => model_peer("list-speech", &p.args),
        Cmd::PullAudio(p) => model_peer("pull-audio", &p.args),
        Cmd::External(args) => handle_external(&args),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_command_tree_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn distance_is_levenshtein() {
        assert_eq!(distance("pul", "pull"), 1);
        assert_eq!(distance("satus", "status"), 1);
        assert_eq!(distance("serve", "serve"), 0);
        assert_eq!(distance("abc", "xyz"), 3);
    }

    #[test]
    fn run_keeps_the_prompt_and_flags_in_order() {
        let cli = Cli::try_parse_from([
            "turbospark",
            "run",
            "gemma4",
            "hello",
            "--temperature",
            "0.7",
        ])
        .unwrap();
        let Some(Cmd::Run { model, rest }) = cli.command else {
            panic!("run did not parse as run");
        };
        assert_eq!(model, "gemma4");
        assert_eq!(rest, ["hello", "--temperature", "0.7"]);
    }

    #[test]
    fn passthrough_commands_keep_help_for_the_peer() {
        let cli = Cli::try_parse_from(["turbospark", "image", "--help"]).unwrap();
        let Some(Cmd::Image(p)) = cli.command else {
            panic!("image did not parse");
        };
        assert_eq!(p.args, ["--help"]);
    }

    #[test]
    fn model_then_agent_is_external() {
        let cli = Cli::try_parse_from(["turbospark", "qwen3.6", "claude", "--dry-run"]).unwrap();
        let Some(Cmd::External(args)) = cli.command else {
            panic!("model-first launch did not reach the external arm");
        };
        assert_eq!(args, ["qwen3.6", "claude", "--dry-run"]);
    }
}
