---
description: >-
  Pure CLI argument parsing for TurboSpark: a token list in, one typed outcome
  out, with no I/O.
icon: box
---

# Argument parsing

Use this crate when you want TurboSpark's flag grammar and validation inside your own command-line tool. You pass in the tokens, and it returns exactly one of four outcomes: a validated `InvocationRequest`, a help request, a version request, or a typed `ParseFailure`. It reads no files, no environment variables, and no process state, so the same token list always gives the same result. That also makes it easy to test parsing without touching the process.

The crate checks values, not the world. It does not confirm that a model path exists, that an image decodes, or that a checkpoint accepts a reasoning level. Those checks belong to the layer that opens the model. For each flag's meaning and default, see the [option reference](options.md).

## Add the crate

The package name is `turbospark-invocation`, and the Rust import path is `turbospark_invocation`. Inside this workspace, sibling crates import it under a short alias. External users depend on the package name and write `turbospark_invocation` in code.

Add the package from crates.io once the version you need is published. Until then, depend on a checkout by path or by git:

```toml
[dependencies]
turbospark-invocation = { git = "https://github.com/whit3rabbit/turbospark" }
```

The crate depends on `turbospark-core`, so that package arrives as a transitive dependency. You do not need to import it for the types on this page.

## Parse a token list

This example compiles against the crate in this checkout and runs its assertions. It parses one valid invocation, shows a mutually exclusive failure, and maps an invalid invocation to the exit status.

```rust
use turbospark_invocation::{
    exit_status, parse, render_usage, ExitStatus, InvocationRequest, Mode, ParseFailure,
    ParseOutcome,
};

/// Parse a token list the way a CLI front end would. Help and version print
/// their text and return `Ok(None)`.
fn request_from(tokens: &[&str]) -> Result<Option<InvocationRequest>, ParseFailure> {
    let owned: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
    match parse(&owned) {
        ParseOutcome::Success(req) => Ok(Some(req)),
        ParseOutcome::Help => {
            print!("{}", render_usage());
            Ok(None)
        }
        ParseOutcome::Version => {
            print!("{}", turbospark_invocation::render_version());
            Ok(None)
        }
        ParseOutcome::Failure(failure) => Err(failure),
    }
}

fn main() {
    let req = request_from(&["--model", "models/demo", "--prompt", "Hello", "--max-new", "256"])
        .expect("valid invocation")
        .expect("not a help request");
    assert_eq!(req.model, "models/demo");
    assert_eq!(req.mode, Mode::Prompt("Hello".to_string()));
    assert_eq!(req.max_new, 256);
    assert_eq!(req.top_k, 64);
    println!("ok: model={} max_new={}", req.model, req.max_new);

    let failure = request_from(&["--model", "models/demo", "--prompt", "x", "--chat"])
        .expect_err("prompt and chat are mutually exclusive");
    assert!(matches!(
        failure,
        ParseFailure::MutuallyExclusive { first: "--prompt", second: "--chat" }
    ));

    let bad = parse(&[
        "--model".to_string(),
        "m".to_string(),
        "--top-k".to_string(),
        "999".to_string(),
    ]);
    assert_eq!(exit_status(&bad), ExitStatus::InvalidInvocation);
    assert_eq!(exit_status(&bad).code(), 2);
}
```

The `Mode` enum and the request fields are public, so a caller can match on them directly. The `request_from` helper is only a convenience for this page.

## The four outcomes

[`parse`](https://docs.rs/turbospark-invocation/latest/turbospark_invocation/fn.parse.html) returns a [`ParseOutcome`](https://docs.rs/turbospark-invocation/latest/turbospark_invocation/enum.ParseOutcome.html). Every call ends in one of these variants, and no partially filled result is ever returned.

| Variant                      | Meaning                                  | Payload                                         |
| ---------------------------- | ---------------------------------------- | ----------------------------------------------- |
| `Success(InvocationRequest)` | The tokens form a valid invocation.      | The validated request, with defaults filled in. |
| `Help`                       | `--help` was reached during the scan.    | None.                                           |
| `Version`                    | `--version` was reached during the scan. | None.                                           |
| `Failure(ParseFailure)`      | The tokens are invalid.                  | The typed failure.                              |

Tokens are read left to right. A value-taking option always consumes the next token as its value, even when that token looks like a flag. So `--prompt --help` sets the prompt to the text `--help` and does not request help. Whichever of `--help` or `--version` appears first wins.

The whole-invocation checks run only after the scan finishes with no failure and no help request. They run in a fixed order, and only the first failure is reported.

## Handle a failure

[`ParseFailure`](https://docs.rs/turbospark-invocation/latest/turbospark_invocation/enum.ParseFailure.html) has six variants. Match on the variant and its payload, not on message text. The message wording is not part of the contract.

| Variant             | When it happens                                                                           | Payload           |
| ------------------- | ----------------------------------------------------------------------------------------- | ----------------- |
| `UnknownOption`     | A token matches no documented flag.                                                       | `token`           |
| `MissingValue`      | A value-taking flag is the last token.                                                    | `option`          |
| `InvalidValue`      | A value fails its type, range, or membership check, or a steering or system-message rule. | `option`, `value` |
| `MissingRequired`   | `--model` is absent.                                                                      | `option`          |
| `MutuallyExclusive` | Two mode-selecting options appear together.                                               | `first`, `second` |
| `NoModeSelected`    | None of `--prompt`, `--messages-file`, `--humanize-file`, or `--chat` appears.            | None.             |

Two helpers turn an outcome into process-level decisions. [`exit_status`](https://docs.rs/turbospark-invocation/latest/turbospark_invocation/fn.exit_status.html) returns an `ExitStatus`, where `Success` and `Help` and `Version` map to exit code 0, and `InvalidInvocation` maps to exit code 2. [`stream_routing`](https://docs.rs/turbospark-invocation/latest/turbospark_invocation/fn.stream_routing.html) says which text belongs on the primary stream and which on the diagnostic stream. Help and version text go to the primary stream, and a failure goes to the diagnostic stream with usage text attached.

The diagnostic text that `stream_routing` builds uses the `Debug` output of the failure. That is useful for developers, but it is not polished copy. If your users see these messages, match on `ParseFailure`, and write your own sentences.

## Rules worth knowing

* `--model` is required, and it is stored as an opaque string.
* `--prompt`, `--messages-file`, `--humanize-file`, and `--chat` are pairwise exclusive. Exactly one must appear. `--system` is allowed only with `--chat`.
* Steering parameters without `--steering` are refused, not ignored. So are surplus per-vector values and an inverted `--steering-layers` range. A run that asked for steering and quietly did not apply it would measure the wrong thing.
* Opaque paths such as `--image`, `--steering`, and `--vision-sidecar` are not opened here. The string `auto` in `--vision-sidecar` is read as a literal path.
* The `--reasoning` set is the union of levels used across checkpoints. This crate cannot tell which checkpoint you have, so a level it accepts can still be refused when the model opens.
* `--max-context auto`, `--expert-cache-slots auto`, and `--prefill-chunk auto` are kept as keywords. Resolving them needs the install and the machine, so the runtime does that work.

## Types you will touch

Every public item is in the [crate docs](https://docs.rs/turbospark-invocation/latest/turbospark_invocation/).

* [`parse`](https://docs.rs/turbospark-invocation/latest/turbospark_invocation/fn.parse.html) and [`ParseOutcome`](https://docs.rs/turbospark-invocation/latest/turbospark_invocation/enum.ParseOutcome.html): the entry point and its result.
* [`InvocationRequest`](https://docs.rs/turbospark-invocation/latest/turbospark_invocation/struct.InvocationRequest.html) and [`Mode`](https://docs.rs/turbospark-invocation/latest/turbospark_invocation/enum.Mode.html): the validated value and the selected input mode.
* [`ParseFailure`](https://docs.rs/turbospark-invocation/latest/turbospark_invocation/enum.ParseFailure.html): the typed failure.
* [`OPTIONS`](https://docs.rs/turbospark-invocation/latest/turbospark_invocation/constant.OPTIONS.html) and [`OptionDecl`](https://docs.rs/turbospark-invocation/latest/turbospark_invocation/struct.OptionDecl.html): the flag table that both the parser and the usage text read.
* [`render_usage`](https://docs.rs/turbospark-invocation/latest/turbospark_invocation/fn.render_usage.html), [`render_version`](https://docs.rs/turbospark-invocation/latest/turbospark_invocation/fn.render_version.html), and [`VERSION`](https://docs.rs/turbospark-invocation/latest/turbospark_invocation/constant.VERSION.html): usage and version text. `VERSION` is the workspace version, read from cargo at build time.
* [`exit_status`](https://docs.rs/turbospark-invocation/latest/turbospark_invocation/fn.exit_status.html), [`stream_routing`](https://docs.rs/turbospark-invocation/latest/turbospark_invocation/fn.stream_routing.html), and [`ExitStatus`](https://docs.rs/turbospark-invocation/latest/turbospark_invocation/enum.ExitStatus.html): process mapping as plain data.
* [`steering_knob`](https://docs.rs/turbospark-invocation/latest/turbospark_invocation/fn.steering_knob.html): the one definition of how per-vector steering values pair with `--steering` paths.

## Platform and scope

The crate has no platform-specific code, and it does not depend on Metal or a model file. You can build and test it wherever the workspace toolchain runs. The workspace sets `rust-version = "1.82"`, and this crate inherits that value.

Parsing only proves the flags are well formed. It does not prove that a run will decode, that a model supports a feature, or that a measurement is valid. Check the [CLI reference](https://github.com/whit3rabbit/turbospark/blob/main/docs/CLI.md) for what each flag does at run time, and the [tool calling reference](https://github.com/whit3rabbit/turbospark/blob/main/docs/TOOL_CALLING.md) for the server side of the same options.

## Next

* [Option reference](options.md): every flag, its value shape, and its default.
* [Rust library overview](../): the other crates in the workspace.
* [Testing rules](https://github.com/whit3rabbit/turbospark/blob/main/docs/TESTING.md): how this repository tests parsing.
