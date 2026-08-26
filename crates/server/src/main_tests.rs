use std::path::PathBuf;

use super::args::{parse_model_args, short_circuit, BindMode, ModelArgs};
use super::bind::tailnet_host;

fn parse(argv: &[&str]) -> Result<Option<ModelArgs>, String> {
    let owned: Vec<String> = argv.iter().map(|s| s.to_string()).collect();
    parse_model_args(&owned)
}

#[test]
fn legacy_positional_mode_is_left_alone() {
    assert!(parse(&["/tmp/tok", "9000"]).unwrap().is_none());
}

/// `--model` reaches the catalog store, so one install serves this
/// binary and `turbospark-check` under one alias.
///
/// The assertion is deliberately on a name that is NOT a directory:
/// swapping `resolve_model_arg` back for `PathBuf::from` leaves every
/// path case passing, because for a real path the two agree. Only the
/// alias arm can tell them apart, and this is the cheapest form of it
/// -- the "default install location that happens to exist" arm, which
/// needs no `installed.json` and no model.
///
/// It is the only test in this binary that touches `TURBOSPARK_HOME`,
/// which is what keeps it safe under the default parallel test threads.
#[cfg(target_os = "macos")]
#[test]
fn an_alias_resolves_to_its_install_directory() {
    let root = std::env::temp_dir().join(format!(
        "turbospark-server-alias-{}-{}",
        std::process::id(),
        line!()
    ));
    let installed = root.join("models").join("some-alias.gturbo");
    std::fs::create_dir_all(&installed).unwrap();
    std::env::set_var("TURBOSPARK_HOME", &root);

    assert_eq!(
        catalog::resolve_model_arg("some-alias"),
        installed,
        "an alias should resolve to its install directory"
    );
    // The property the resolution order exists to protect: a bare name
    // that is also a real directory is that directory, never the alias.
    assert_ne!(
        catalog::resolve_model_arg("some-alias"),
        PathBuf::from("some-alias"),
        "resolution must not be a pass-through for a known alias"
    );

    std::env::remove_var("TURBOSPARK_HOME");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn power_flags_default_to_unset_and_parse_their_documented_values() {
    // Unset rather than `performance`: the Low Power Mode default is
    // resolved at model open, where the OS can be asked.
    let d = parse(&["--model", "/tmp/m"]).unwrap().unwrap();
    assert_eq!(d.power_profile, None);
    assert_eq!(d.max_tokens_per_sec, None);

    let o = parse(&[
        "--model",
        "/tmp/m",
        "--power-profile",
        "efficiency",
        "--max-tokens-per-sec",
        "7.5",
    ])
    .unwrap()
    .unwrap();
    assert_eq!(o.power_profile, Some(runtime::PowerProfile::Efficiency));
    assert_eq!(o.max_tokens_per_sec, Some(7.5));
}

#[test]
fn bad_power_flag_values_are_rejected() {
    assert!(parse(&["--model", "/tmp/m", "--power-profile", "turbo"]).is_err());
    for bad in ["0", "-1", "abc", "inf"] {
        assert!(
            parse(&["--model", "/tmp/m", "--max-tokens-per-sec", bad]).is_err(),
            "expected {bad} to be rejected"
        );
    }
}

#[test]
fn model_mode_defaults_and_overrides() {
    let d = parse(&["--model", "/tmp/m"]).unwrap().unwrap();
    // BOTH sized knobs default to `None`, i.e. `auto`: the slot count is
    // sized against this machine and this install and never drops below
    // the shipped 16, and the context window is sized against the
    // checkpoint's trained context and what memory holds.
    assert_eq!(
        (d.port, d.max_context, d.expert_cache_slots),
        (8080, None, None)
    );
    let o = parse(&[
        "--model",
        "/tmp/m",
        "--port",
        "9",
        "--max-context",
        "1024",
        "--expert-cache-slots",
        "32",
    ])
    .unwrap()
    .unwrap();
    assert_eq!(
        (o.port, o.max_context, o.expert_cache_slots),
        (9, Some(1024), Some(32))
    );
    // `auto` is accepted by name as well as by omission, and is the one
    // value the allowed-set check must not reject.
    let a = parse(&["--model", "/tmp/m", "--expert-cache-slots", "auto"])
        .unwrap()
        .unwrap();
    assert_eq!(a.expert_cache_slots, None);
    assert!(parse(&["--model", "/tmp/m", "--expert-cache-slots", "20"]).is_err());

    // The context window takes the same `auto`-or-a-number grammar, and
    // unlike the slot count it has no allowed set to check against: every
    // positive value is a legal KV allocation.
    let c = parse(&["--model", "/tmp/m", "--max-context", "auto"])
        .unwrap()
        .unwrap();
    assert_eq!(c.max_context, None);
    // Zero is refused rather than read as `auto`: a window of zero admits
    // no prompt, and the flag already has a spelling for "you decide".
    assert!(parse(&["--model", "/tmp/m", "--max-context", "0"]).is_err());
    assert!(parse(&["--model", "/tmp/m", "--max-context", "lots"]).is_err());
}

/// **A flag-led invocation must not fall through to the scripted mode.**
/// The mode test used to be `args[0] == "--model"`, so
/// `--port 8080 --model X` was read as a positional TOKENIZER DIRECTORY
/// named `--port` and died with a filesystem error about a path nobody
/// typed. Anything starting with `-` belongs to this parser.
#[test]
fn a_flag_in_any_position_stays_out_of_the_scripted_mode() {
    // Mis-ordered but complete: parsed, not mistaken for a directory.
    let ordered = parse(&["--port", "9", "--model", "/tmp/m"])
        .unwrap()
        .unwrap();
    assert_eq!((ordered.port, ordered.model.as_str()), (9, "/tmp/m"));
    // A misspelled flag gets the usage text rather than a tokenizer error.
    let err = parse(&["--modle", "/tmp/m"]).unwrap_err();
    assert!(err.contains("unknown option"), "{err}");
    // And a real positional path still reaches the scripted mode.
    assert!(parse(&["/tmp/tokenizer-dir"]).unwrap().is_none());
}

/// `--help` and `--version` are handled ahead of the flag loop, because
/// that loop advances two tokens per flag and would eat what follows.
/// Whichever is reached first in a left-to-right scan wins.
#[test]
fn help_and_version_short_circuit_before_anything_is_parsed() {
    assert!(short_circuit(&owned(&["--help"]))
        .unwrap()
        .contains("usage:"));
    assert!(short_circuit(&owned(&["-h"])).unwrap().contains("usage:"));
    // Reachable past other flags, and NOT consuming a value.
    assert!(short_circuit(&owned(&["--model", "/tmp/m", "--help"])).is_some());
    assert!(short_circuit(&owned(&["--model", "/tmp/m"])).is_none());
    assert!(short_circuit(&owned(&["/tmp/dir"])).is_none());
}

/// The version line matches `turbospark-check`'s to the character. The
/// two are produced independently (this binary reads `CARGO_PKG_VERSION`
/// directly rather than depending on the parser crate for one format
/// string), so nothing but this pins them together.
#[test]
fn the_version_line_matches_the_clis() {
    let ours = short_circuit(&owned(&["--version"])).unwrap();
    assert_eq!(ours, format!("turbospark {}\n", env!("CARGO_PKG_VERSION")));
    assert!(short_circuit(&owned(&["-V"])).unwrap() == ours);
}

fn owned(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

#[test]
fn bad_model_mode_arguments_are_rejected() {
    // A slot count outside the allowed set would panic the runtime
    // config setter, so it has to fail here instead.
    assert!(parse(&["--model", "/tmp/m", "--expert-cache-slots", "12"]).is_err());
    assert!(parse(&["--model", "/tmp/m", "--port", "70000"]).is_err());
    assert!(parse(&["--model"]).is_err());
    assert!(parse(&["--model", "/tmp/m", "--nope", "1"]).is_err());
    assert!(parse(&["--model", "/tmp/m", "--bind", "lan"]).is_err());
}

#[test]
fn bind_mode_defaults_to_loopback() {
    assert_eq!(
        parse(&["--model", "/tmp/m"]).unwrap().unwrap().bind,
        BindMode::Loopback
    );
    let t = parse(&["--model", "/tmp/m", "--bind", "tailnet"])
        .unwrap()
        .unwrap();
    assert_eq!(t.bind, BindMode::Tailnet);
    assert_eq!(BindMode::Loopback.host().unwrap(), "127.0.0.1");
}

#[test]
fn tailnet_host_accepts_exactly_one_in_range_address() {
    assert_eq!(tailnet_host("100.64.0.1\n").unwrap(), "100.64.0.1");
    assert_eq!(
        tailnet_host("100.127.255.254\n").unwrap(),
        "100.127.255.254"
    );
}

#[test]
fn tailnet_host_never_falls_back() {
    // Empty, ambiguous, out-of-range, IPv6, and malformed all fail rather
    // than widening to a loopback/LAN/wildcard bind.
    assert!(tailnet_host("").is_err());
    assert!(tailnet_host("  \n").is_err());
    assert!(tailnet_host("100.64.0.1 100.64.0.2\n").is_err());
    assert!(tailnet_host("100.63.0.1").is_err());
    assert!(tailnet_host("100.128.0.1").is_err());
    assert!(tailnet_host("192.168.1.5").is_err());
    assert!(tailnet_host("127.0.0.1").is_err());
    assert!(tailnet_host("0.0.0.0").is_err());
    assert!(tailnet_host("fd7a:115c:a1e0::1").is_err());
    assert!(tailnet_host("100.64.0").is_err());
    assert!(tailnet_host("100.64.0.256").is_err());
    assert!(tailnet_host("100.064.0.1").is_err());
    assert!(tailnet_host("100.64.0.1;rm -rf /").is_err());
}

// --- directional steering (docs/OBLITERATION.md) --------------------------
//
// The CLI's half of these six flags is covered by `crates/invocation`'s parse
// tests. THIS parser is a second implementation of the same precedence rules
// against a different type, and until now none of the cases above touched it,
// so the two could drift with nothing to catch it. The order-dependence case
// below is the one no other file can cover at all: `crates/invocation`
// collects every flag before resolving anything, while this loop applies the
// layer range as it goes and has to repair it in two places.

#[test]
fn steering_defaults_to_off_and_reads_no_file() {
    let d = parse(&["--model", "/tmp/m"]).unwrap().unwrap();
    assert!(d.steering.set.is_none());
    assert!(!d.steering.is_active());
    assert_eq!(d.steering.alpha, 0.0);
}

/// A steering PARAMETER without `--steering` is refused rather than ignored.
/// Without a direction set the process serves every request unsteered, so the
/// flag would be a command line saying one thing while the server does
/// another -- on the one axis here that changes the TOKENS.
///
/// All five, because they reach the parsed policy by three different routes:
/// `--steering-mode` and `--steering-scale` are held aside, `--steering-target`
/// and `--steering-gate` are written straight into it, and
/// `--steering-layers` is held aside AND replayed.
#[test]
fn a_steering_parameter_without_a_vector_is_refused() {
    for (flag, value) in [
        ("--steering-mode", "renorm"),
        ("--steering-scale", "0.8"),
        ("--steering-layers", "30:40"),
        ("--steering-target", "2.5"),
        ("--steering-gate", "0.5"),
    ] {
        let err = parse(&["--model", "/tmp/m", flag, value])
            .expect_err(&format!("{flag} alone should be refused"));
        assert!(
            err.contains(flag) && err.contains("--steering"),
            "message should name the flag and what it needs: {err}"
        );
    }
}

/// The rejection has to be SPELLED from the accepted set. This message named
/// three modes for a release after `renorm` landed, so a caller who misspelled
/// the fourth was told it did not exist.
///
/// TWO ASSERTIONS, because the first one alone is SELF-REFERENTIAL and was
/// measured to be: the message is built from `STEERING_MODE_NAMES`, so
/// iterating that same list only proves the message was not hand-written, and
/// shortening the list leaves this green (checked -- it is `crates/core`'s
/// `the_mode_names_are_exactly_what_parse_accepts` that reddens there). The
/// literal is what pins the regression that actually happened, and the loop
/// is what pins the mechanism that prevents it recurring. Neither replaces the
/// other.
#[test]
fn an_unknown_steering_mode_names_every_mode_it_accepts() {
    let err = parse(&["--model", "/tmp/m", "--steering-mode", "renrom"])
        .expect_err("a misspelled mode should be refused");
    assert!(
        err.contains("renorm"),
        "the fourth mode is the one this message lost once, and reads: {err}"
    );
    for name in foundation::STEERING_MODE_NAMES {
        assert!(
            err.contains(name),
            "the rejection should name {name}, and reads: {err}"
        );
    }
}

#[cfg(target_os = "macos")]
fn write_vector(tag: &str, layers: usize, mode: Option<foundation::SteeringMode>) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-server-steer-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("v.gguf");
    // Blocks 1..=layers: `direction.0` has no name in this format, so a
    // control vector cannot carry an edit for block 0.
    let dirs: std::collections::BTreeMap<usize, Vec<f32>> = (1..=layers)
        .map(|l| (l, (0..8).map(|i| (l * 8 + i) as f32 * 0.125).collect()))
        .collect();
    std::fs::write(
        &path,
        repack::control_vector::write_control_vector(&dirs, "qwen35", mode).unwrap(),
    )
    .unwrap();
    path
}

/// Precedence, stated the same way `crates/cli`'s `resolve_steering` states
/// it: the flag wins over the file's declared mode, the file's over the
/// default. A set present with no `--steering-scale` means FULL strength, not
/// the zero `SteeringPolicy::off()` carries -- that last one is the trap,
/// since inheriting the default would load a vector and apply the identity.
#[cfg(target_os = "macos")]
#[test]
fn the_mode_comes_from_the_flag_then_the_file_then_the_default() {
    let declared = write_vector("declared", 4, Some(foundation::SteeringMode::Renorm));
    let bare = write_vector("bare", 4, None);
    let p = |args: &[&str]| parse(args).unwrap().unwrap().steering;

    let from_file = p(&[
        "--model",
        "/tmp/m",
        "--steering",
        declared.to_str().unwrap(),
    ]);
    assert_eq!(from_file.mode, foundation::SteeringMode::Renorm);
    assert_eq!(from_file.alpha, 1.0, "a loaded vector means full strength");

    let from_flag = p(&[
        "--model",
        "/tmp/m",
        "--steering",
        declared.to_str().unwrap(),
        "--steering-mode",
        "add",
    ]);
    assert_eq!(from_flag.mode, foundation::SteeringMode::Add);

    let defaulted = p(&["--model", "/tmp/m", "--steering", bare.to_str().unwrap()]);
    assert_eq!(defaulted.mode, foundation::SteeringMode::Ablate);
}

/// `--steering-layers` must survive BOTH orders, and this loop is the only
/// place in the workspace where that is a live question: it restricts the set
/// as the flag arrives, so the range has to be replayed when the vector comes
/// second and applied on arrival when it comes first. Getting one half wrong
/// steers the layers the caller excluded, which is fluent and wrong.
#[cfg(target_os = "macos")]
#[test]
fn a_layer_range_applies_whichever_side_of_the_vector_it_is_given() {
    let v = write_vector("range", 6, None);
    let path = v.to_str().unwrap();
    let covered = |args: &[&str]| {
        let s = parse(args).unwrap().unwrap().steering;
        let set = s.set.expect("a vector was given");
        (
            set.covered_layers(),
            set.layer(0).is_some(),
            set.layer(3).is_some(),
        )
    };

    let after = covered(&[
        "--model",
        "/tmp/m",
        "--steering",
        path,
        "--steering-layers",
        "2:4",
    ]);
    let before = covered(&[
        "--model",
        "/tmp/m",
        "--steering-layers",
        "2:4",
        "--steering",
        path,
    ]);
    assert_eq!(after, (3, false, true));
    assert_eq!(before, after, "the range must not depend on flag order");
}

/// An inverted range is refused here rather than silently selecting nothing,
/// matching `crates/invocation`'s `parse_layer_range`.
#[cfg(target_os = "macos")]
#[test]
fn an_inverted_layer_range_is_refused() {
    let v = write_vector("inverted", 6, None);
    assert!(parse(&[
        "--model",
        "/tmp/m",
        "--steering",
        v.to_str().unwrap(),
        "--steering-layers",
        "4:2",
    ])
    .is_err());
}
