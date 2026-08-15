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
