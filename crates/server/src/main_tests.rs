use std::path::PathBuf;

use super::args::{parse_model_args, short_circuit, tailnet_host, BindMode, ModelArgs};

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
