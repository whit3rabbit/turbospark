//! `turbospark-model`'s argument surface and exit codes.
//!
//! Every case here is offline: no command that reaches the network is
//! exercised, so this runs in the default suite. The two exit codes are the
//! contract worth pinning -- 2 for a malformed invocation and 1 for a run
//! that was asked for correctly and did not work -- because a script doing
//! `probe X && pull X` depends on the difference.

use std::process::Command;

fn bin() -> Command {
    let mut path = std::env::current_exe().expect("test binary path");
    path.pop(); // deps/
    path.pop();
    path.push("turbospark-model");
    let mut command = Command::new(path);
    // A store of its own, so a developer's real installs cannot make this
    // pass or fail.
    command.env(
        "TURBOSPARK_HOME",
        std::env::temp_dir().join(format!("turbospark-model-cli-{}", std::process::id())),
    );
    // Same reasoning, for the machine's CHIP: `recommend_fits_against_the_
    // named_context` asserts against gemma4's measured row, which was taken
    // on "Apple M4 Max" (models.json). The real Metal device name is
    // whatever chip runs this suite -- a real Mac's own marketing name, or a
    // CI runner's virtualized one ("Apple Paravirtual device", which matches
    // no measured row at all) -- so without this override the test's result
    // depends on which machine happens to run it rather than on the code.
    command.env("TURBOSPARK_TEST_CHIP", "Apple M4 Max");
    command
}

fn run(args: &[&str]) -> (i32, String, String) {
    let out = bin().args(args).output().expect("the binary runs");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

#[test]
fn no_arguments_prints_usage_and_succeeds() {
    let (code, stdout, _) = run(&[]);
    assert_eq!(code, 0);
    assert!(stdout.contains("USAGE"), "{stdout}");
    assert!(stdout.contains("pull"), "{stdout}");
}

#[test]
fn help_prints_usage() {
    for flag in ["--help", "-h", "help"] {
        let (code, stdout, _) = run(&[flag]);
        assert_eq!(code, 0, "{flag}");
        assert!(stdout.contains("USAGE"), "{flag}: {stdout}");
    }
}

/// All three spellings print the version, and the string matches the other
/// two binaries' to the character -- each reads `CARGO_PKG_VERSION` itself
/// rather than sharing a helper, so nothing but a test pins them together.
/// It also has to be handled AHEAD of the subcommand match, which would
/// otherwise report `--version` as an unknown command.
#[test]
fn version_prints_the_workspace_version() {
    for flag in ["--version", "-V", "version"] {
        let (code, stdout, _) = run(&[flag]);
        assert_eq!(code, 0, "{flag}");
        assert_eq!(
            stdout.trim(),
            format!("turbospark {}", env!("CARGO_PKG_VERSION")),
            "{flag}"
        );
        assert!(
            !stdout.contains("USAGE"),
            "{flag} must not print the usage table: {stdout}"
        );
    }
}

/// The usage text has to warn about the thing a user cannot discover any
/// other way until it costs them twenty minutes.
#[test]
fn the_usage_text_says_a_pull_cannot_resume() {
    let (_, stdout, _) = run(&["--help"]);
    assert!(
        stdout.contains("CANNOT RESUME"),
        "the no-resume caveat must be in the usage text: {stdout}"
    );
}

#[test]
fn list_prints_the_catalog_with_its_status_legend() {
    let (code, stdout, _) = run(&["list"]);
    assert_eq!(code, 0);
    assert!(stdout.contains("gemma4"), "{stdout}");
    assert!(stdout.contains("ALIAS"), "{stdout}");
    // The legend is not decoration: `verified` and `runs` mean specific
    // things and a bare column would invite reading them as a quality
    // ranking.
    assert!(stdout.contains("docs/BENCHMARKS.md"), "{stdout}");
}

#[test]
fn list_filters() {
    let (code, stdout, _) = run(&["list", "--filter", "ternary"]);
    assert_eq!(code, 0);
    assert!(stdout.contains("ternary27b"), "{stdout}");
    assert!(!stdout.contains("gemma4 "), "filtered out: {stdout}");

    let (code, stdout, _) = run(&["list", "--filter", "no-such-model"]);
    assert_eq!(code, 0, "an empty result is not an error");
    assert!(stdout.contains("no models match"), "{stdout}");
}

#[test]
fn info_prints_a_row_in_full() {
    let (code, stdout, _) = run(&["info", "ternary27b"]);
    assert_eq!(code, 0);
    assert!(
        stdout.contains("prism-ml/Ternary-Bonsai-27B-mlx-2bit"),
        "{stdout}"
    );
    assert!(stdout.contains("merges.txt"), "the sidecar list: {stdout}");
    assert!(
        stdout.contains("ternary_quality_gate"),
        "the gates: {stdout}"
    );
    assert!(
        stdout.contains("not installed") || stdout.contains("pull"),
        "{stdout}"
    );
}

/// A floating row has to say so where somebody reads it, because its frozen
/// numbers stop meaning anything the moment the publisher re-uploads.
#[test]
fn info_marks_a_row_pinned_at_main_as_floating() {
    let (_, stdout, _) = run(&["info", "tinyllama"]);
    assert!(stdout.contains("FLOATS"), "{stdout}");
    let (_, pinned, _) = run(&["info", "gemma4"]);
    assert!(
        !pinned.contains("FLOATS"),
        "a sha-pinned row must not be marked as floating: {pinned}"
    );
}

/// An unknown alias suggests near matches rather than sending the reader to
/// `list`. Cheap, and it turns the commonest typo into a one-line fix.
#[test]
fn an_unknown_alias_fails_with_suggestions() {
    let (code, _, stderr) = run(&["info", "gemma"]);
    assert_eq!(code, 1, "a correct invocation naming a missing model is 1");
    assert!(stderr.contains("Did you mean"), "{stderr}");
    assert!(stderr.contains("gemma4"), "{stderr}");

    let (code, _, stderr) = run(&["info", "zzzz-nothing-like-this"]);
    assert_eq!(code, 1);
    assert!(
        stderr.contains("list"),
        "no near match: point at list: {stderr}"
    );
}

#[test]
fn path_fails_rather_than_printing_nothing_for_an_uninstalled_model() {
    // Failing matters more than the message: `--model $(turbospark-model path
    // X)` would otherwise expand to an empty string and run with `--model ''`.
    let (code, stdout, _) = run(&["path", "gemma4"]);
    assert_eq!(code, 1);
    assert!(stdout.trim().is_empty(), "nothing on stdout: {stdout:?}");
}

#[test]
fn a_malformed_invocation_exits_two_and_prints_usage() {
    for args in [
        vec!["not-a-command"],
        vec!["info"],
        vec!["info", "a", "b"],
        vec!["probe"],
        vec!["pull"],
        vec!["list", "--filter"],
        vec!["list", "--nonsense"],
    ] {
        let (code, _, stderr) = run(&args);
        assert_eq!(code, 2, "{args:?} should be a usage error: {stderr}");
        assert!(stderr.contains("USAGE"), "{args:?}: {stderr}");
    }
}

/// An option that the current command does not read is an ERROR, not a
/// silent no-op. `--force` on `list` doing nothing reads as the command
/// having considered and ignored it.
#[test]
fn an_option_on_the_wrong_command_is_refused() {
    let (code, _, stderr) = run(&["list", "--force"]);
    assert_eq!(code, 2);
    assert!(stderr.contains("does not apply"), "{stderr}");
}

#[test]
fn pull_by_repo_requires_an_alias_and_refuses_both_forms_at_once() {
    let (code, _, stderr) = run(&["pull", "--repo", "owner/name"]);
    assert_eq!(code, 2);
    assert!(stderr.contains("--alias"), "{stderr}");

    let (code, _, stderr) = run(&["pull", "gemma4", "--repo", "owner/name", "--alias", "x"]);
    assert_eq!(code, 2);
    assert!(stderr.contains("not both"), "{stderr}");
}

#[test]
fn a_malformed_repo_reference_is_a_usage_error() {
    for bad in ["notowneronly", "owner/name/extra", "owner/name@"] {
        let (code, _, stderr) = run(&["probe", bad]);
        assert_eq!(code, 2, "{bad:?}: {stderr}");
        assert!(
            stderr.contains("owner/name") || stderr.contains("revision"),
            "{stderr}"
        );
    }
}

/// `recommend` is offline by DEFAULT, which is what makes it testable here at
/// all -- `--discover` and `--probe` are the two arms that reach the network
/// and neither is exercised in this file.
///
/// `--budget` is what lets the case run on any machine: without it the answer
/// depends on how much memory the test runner has, which is a test that passes
/// or fails on the hardware rather than on the code.
#[test]
fn recommend_ranks_the_catalog_against_a_named_budget() {
    let (code, stdout, _) = run(&["recommend", "--budget", "36GiB"]);
    assert_eq!(code, 0, "{stdout}");
    assert!(stdout.contains("36.0 GiB of memory"), "{stdout}");
    assert!(stdout.contains("4096-token context"), "{stdout}");
    // The two size columns are the point of the table; a single one would be
    // wrong for one of the two shapes this engine has.
    assert!(
        stdout.contains("ALLOCS") && stdout.contains("ON DISK"),
        "{stdout}"
    );
    assert!(stdout.contains("gemma4"), "{stdout}");
}

/// **The budget is a MEMORY size and a bare number is bytes.** Guessing
/// gigabytes on `--budget 36` would be wrong by a factor of a billion in
/// whichever direction the guess missed, and the failure would be silent: a
/// 36-byte budget refuses everything and a 36 GiB one fits everything, both
/// plausibly.
#[test]
fn the_budget_flag_reads_sizes_and_refuses_nonsense() {
    let (code, stdout, _) = run(&["recommend", "--budget", "38654705664"]);
    assert_eq!(code, 0);
    assert!(stdout.contains("36.0 GiB of memory"), "{stdout}");

    for bad in ["thirty-six", "36 furlongs", "-1", ""] {
        let (code, _, stderr) = run(&["recommend", "--budget", bad]);
        assert_eq!(code, 2, "{bad:?} should be a usage error");
        assert!(stderr.contains("is not a size"), "{bad:?}: {stderr}");
    }
}

/// A window is a parameter because a footprint is a footprint at one window
/// (AGENTS.md Gotcha 40), and the table has to say which one it used.
#[test]
fn recommend_fits_against_the_named_context() {
    let (code, stdout, _) = run(&["recommend", "--budget", "36GiB", "--context", "8192"]);
    assert_eq!(code, 0, "{stdout}");
    assert!(stdout.contains("8192-token context"), "{stdout}");
    // gemma4's measured peak was taken at 4,096, so at 8,192 it must be
    // reported rather than applied -- silently quoting it would understate KV.
    assert!(
        stdout.contains("taken at 4096 context, not 8192"),
        "{stdout}"
    );
}

/// `recommend` describes the machine, so a positional argument is a
/// misunderstanding worth naming rather than ignoring.
#[test]
fn recommend_takes_no_positional_argument() {
    let (code, _, stderr) = run(&["recommend", "gemma4"]);
    assert_eq!(code, 2);
    assert!(stderr.contains("takes no arguments"), "{stderr}");
}

/// Every other command refuses the flags it does not read, and these are no
/// exception: `--context` on a `pull` would read as a considered-and-ignored
/// option rather than as a mistake.
#[test]
fn the_recommend_flags_do_not_apply_to_other_commands() {
    for (command, flag) in [
        ("list", "--context"),
        ("info", "--budget"),
        ("list", "--probe"),
    ] {
        let (code, _, stderr) = run(&[command, flag, "4096"]);
        assert_eq!(code, 2, "{command} {flag}");
        assert!(
            stderr.contains("does not apply to this command"),
            "{command} {flag}: {stderr}"
        );
    }
}

/// A bare `--discover` must not swallow the next token. Nothing else in this
/// parser has an optional value, so the case is worth pinning: `--discover
/// --budget 36GiB` has to mean the default scan and a 36 GiB budget, not a
/// scan of `--budget`.
/// `list` marks a vision-tower row distinguishably from a trunk row: the
/// KIND column reads `vision-tower` rather than `model` (vision memory
/// sidecar, part A5, Task 6's `qwen38-vision-tower` row).
#[test]
fn list_marks_a_vision_tower_row_distinguishably() {
    let (code, stdout, _) = run(&["list"]);
    assert_eq!(code, 0, "{stdout}");
    assert!(stdout.contains("qwen38-vision-tower"), "{stdout}");
    assert!(stdout.contains("vision-tower"), "{stdout}");
    assert!(stdout.contains("KIND"), "{stdout}");
}

/// `info` on a vision-tower row still prints in full -- nothing about the
/// new field changes the print path for an ordinary lookup.
#[test]
fn info_prints_a_vision_tower_row() {
    let (code, stdout, _) = run(&["info", "qwen38-vision-tower"]);
    assert_eq!(code, 0, "{stdout}");
    assert!(stdout.contains("preprocessor_config.json"), "{stdout}");
}

/// An uninstalled row's hint names the command that would ACTUALLY install
/// it: `pull-vision` for a tower, `pull` for a model. Printing the wrong one
/// for a tower row would send a reader into `pull`'s own "is a model row,
/// not a vision-tower row" refusal.
#[test]
fn info_hints_pull_vision_for_an_uninstalled_tower_and_pull_for_a_model() {
    let (_, tower, _) = run(&["info", "qwen38-vision-tower"]);
    assert!(tower.contains("pull-vision qwen38-vision-tower"), "{tower}");

    let (_, model, _) = run(&["info", "ternary27b"]);
    assert!(model.contains("pull ternary27b"), "{model}");
    assert!(!model.contains("pull-vision"), "{model}");
}

/// `pull-vision --repo` needs `--alias`, the same shape `pull --repo` has.
#[test]
fn pull_vision_by_repo_requires_an_alias_and_refuses_both_forms_at_once() {
    let (code, _, stderr) = run(&["pull-vision", "--repo", "owner/name"]);
    assert_eq!(code, 2, "{stderr}");
    assert!(stderr.contains("--alias"), "{stderr}");

    let (code, _, stderr) = run(&[
        "pull-vision",
        "qwen38-vision-tower",
        "--repo",
        "owner/name",
        "--alias",
        "x",
    ]);
    assert_eq!(code, 2, "{stderr}");
    assert!(stderr.contains("not both"), "{stderr}");
}

/// `pull-vision` with no arguments at all is a usage error, not a silent
/// no-op.
#[test]
fn pull_vision_with_no_arguments_is_a_usage_error() {
    let (code, _, stderr) = run(&["pull-vision"]);
    assert_eq!(code, 2, "{stderr}");
    assert!(stderr.contains("--repo"), "{stderr}");
}

/// `pull-vision` on an alias that exists but is a MODEL row (not a
/// vision-tower one) is refused by name, pointing at `pull` instead --
/// caught before any network call, so this stays in the offline suite.
#[test]
fn pull_vision_on_a_model_alias_is_refused_and_points_at_pull() {
    let (code, _, stderr) = run(&["pull-vision", "gemma4"]);
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("not a vision-tower row"), "{stderr}");
    assert!(
        stderr.contains("pull gemma4") || stderr.contains("`pull "),
        "{stderr}"
    );
}

/// An unknown alias to `pull-vision` gets the same near-match suggestion
/// every other alias lookup does.
#[test]
fn pull_vision_on_an_unknown_alias_suggests_near_matches() {
    // A substring of the real alias, not a superstring: `find` matches by
    // `alias.contains(needle)`, so a typo that ADDS characters (like a
    // doubled trailing letter) is never a substring of the shorter real
    // alias and would find nothing to suggest.
    let (code, _, stderr) = run(&["pull-vision", "qwen38-visio"]);
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("qwen38-vision-tower"), "{stderr}");
}

#[test]
fn a_malformed_pull_vision_repo_reference_is_a_usage_error() {
    let (code, _, stderr) = run(&["pull-vision", "--repo", "notowneronly", "--alias", "x"]);
    assert_eq!(code, 2, "{stderr}");
    assert!(
        stderr.contains("owner/name") || stderr.contains("revision"),
        "{stderr}"
    );
}

#[test]
fn a_bare_discover_does_not_swallow_the_next_flag() {
    // Parse only: `--budget 1` makes every row refuse, so nothing reaches the
    // network before the table is printed... except discovery itself, which
    // does. So this asserts the PARSE via the usage path instead.
    let (code, _, stderr) = run(&["list", "--discover", "--filter", "gemma"]);
    assert_eq!(code, 2, "{stderr}");
    assert!(
        stderr.contains("--discover does not apply"),
        "the flag parsed as a flag rather than eating --filter: {stderr}"
    );
}

#[test]
fn auth_status_runs_cleanly() {
    let (code, stdout, _) = run(&["auth"]);
    assert_eq!(code, 0);
    assert!(
        stdout.contains("Hugging Face token") || stdout.contains("No Hugging Face token found"),
        "{stdout}"
    );
}

#[test]
fn auth_clear_succeeds() {
    let (code, stdout, _) = run(&["auth", "--clear"]);
    assert_eq!(code, 0);
    assert!(stdout.contains("token cleared"), "{stdout}");
}

#[test]
fn auth_rejects_unused_flags() {
    let (code, _, stderr) = run(&["auth", "--context", "4096"]);
    assert_eq!(code, 2);
    assert!(
        stderr.contains("--context does not apply to this command"),
        "{stderr}"
    );
}

#[test]
fn auth_rejects_tokens_in_process_arguments() {
    for args in [
        &["auth", "hf_secret"] as &[&str],
        &["auth", "--set", "hf_secret"],
        &["pull", "tinyllama", "--hf-token", "hf_secret"],
    ] {
        let (code, _, stderr) = run(args);
        assert_eq!(code, 2, "{args:?}: {stderr}");
        assert!(stderr.contains("USAGE"), "{args:?}: {stderr}");
    }
}

#[test]
fn help_does_not_advertise_token_arguments() {
    let (_, stdout, _) = run(&["--help"]);
    assert!(!stdout.contains("--hf-token"), "{stdout}");
    assert!(!stdout.contains("--set <TOKEN>"), "{stdout}");
    assert!(!stdout.contains("auth [TOKEN]"), "{stdout}");
}
