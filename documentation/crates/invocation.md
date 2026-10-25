---
description: "Pure command-line invocation parsing: an ordered token list in, a validated request, a help or version short-circuit, or one of six typed failures out."
icon: paper-plane
---

# invocation

`crates/invocation` (package `turbospark-invocation`) translates an ordered
list of command-line argument tokens into a validated `InvocationRequest`, a
help or version short-circuit, or one of six distinguishable typed failures.
Parsing is a pure function of the token list: no filesystem, network,
environment, or process access happens here, no state is held between calls,
and the same tokens always produce the same outcome. The `cli` crate is the
direct consumer; it reads argv, calls `parse`, and applies the exit-status
and stream-routing decisions this crate also defines.

## What it does

| Responsibility | Where |
| --- | --- |
| The left-to-right token scan and whole-invocation checks | `src/parser.rs` (`parse`, `ParseOutcome`) |
| The validated request value and its option value types | `src/request.rs` (`InvocationRequest`, `Mode`, `MaxContext`, `Speculation`, ...) |
| The closed set of six typed parse failures | `src/failure.rs` (`ParseFailure`) |
| Option declarations and the options metadata table | `src/options.rs` (`OptionDecl`, `OPTIONS`) |
| Exit status and stream routing decisions | `src/diagnostics.rs` (`exit_status`, `stream_routing`) |
| Help and usage text formatting | `src/usage.rs` (`render_usage`, `render_version`, `VERSION`) |

Numeric validation reads the allowed-value sets from `turbospark-core`'s
`runtime_config` module (cache slots and chunk sizes), so there is exactly
one copy of each range in the workspace.

## Where it sits

`turbospark-invocation` depends only on `turbospark-core` (aliased
`foundation`) and is consumed directly by `cli`, whose binary is a thin
entry point over `parse`. The server keeps its own request parser for its
JSON boundary but validates its numeric knobs against the same `core`
constants, which is why the ranges live in `core` rather than here.

## Public API tour

Root re-exports from `crates/invocation/src/lib.rs`:

| Item | Path | Meaning |
| --- | --- | --- |
| `parse(&[String]) -> ParseOutcome` | `src/parser.rs` | The one entry point; exactly one of four outcomes |
| `ParseOutcome` | `src/parser.rs` | `Success(InvocationRequest)`, `Help`, `Version`, `Failure(ParseFailure)` |
| `InvocationRequest` | `src/request.rs` | Model path, mode, sampling knobs, stop strings, steering, speculation, power, and more |
| `Mode` | `src/request.rs` | `Prompt(String)`, `MessagesFile(String)`, `HumanizeFile(String)`, `Chat` |
| `ParseFailure` | `src/failure.rs` | `UnknownOption`, `MissingValue`, `InvalidValue`, `MissingRequired`, `MutuallyExclusive`, `NoModeSelected` |
| `exit_status`, `stream_routing`, `ExitStatus`, `StreamRouting` | `src/diagnostics.rs` | Pure mapping from an outcome to a process exit code (0 or 2) and primary/diagnostic stream text |
| `OptionDecl`, `OPTIONS` | `src/options.rs` | The declared option table behind help output |
| `render_usage`, `render_version`, `VERSION` | `src/usage.rs` | Help text, version line, and the crate version |
| `steering_knob` | `src/request.rs` | The one definition of how repeatable steering knobs pair with `--steering` paths |
| `DEFAULT_MAX_NEW` and the other defaults | `src/request.rs` | Documented defaults (max-new 1024, temperature 0.2, top-k 64, top-p 0.95) |

Value types such as `MaxContext`, `ExpertCacheSlots`, `ExpertResidency`,
`Speculation`, `SpeculativeDrafter`, `KvBits`, `PrefillChunk`, `LoadGuard`,
`ReadAheadMode`, `PowerProfile`, and `ReasoningEffort` each parse and reject
their own allowed spellings in `src/request.rs`. `ReasoningEffort` here is a
deliberate union of every checkpoint's accepted spellings; a checkpoint that
rejects one names its own set in the template error, because this crate
cannot look at the install.

## Examples

### Parse a minimal invocation and read the defaults

Adapted from `crates/invocation/tests/parse_outcomes.rs`.

```rust
use turbospark_invocation::{parse, InvocationRequest, Mode, ParseOutcome};

fn tok(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

let outcome = parse(&tok(&["--model", "m.bin", "--prompt", "hello"]));
match outcome {
    ParseOutcome::Success(req) => {
        let req: InvocationRequest = req;
        assert_eq!(req.model, "m.bin");
        assert_eq!(req.mode, Mode::Prompt("hello".to_string()));
        // Documented defaults for everything not supplied.
        assert_eq!(req.max_new, 1024);
        assert_eq!(req.top_k, 64);
    }
    other => panic!("expected success, got {other:?}"),
}
```

### Match a typed failure without parsing message strings

Each `ParseFailure` variant carries the offending option (and value) as
structured payload, from `crates/invocation/src/failure.rs`.

```rust
use turbospark_invocation::{parse, ParseFailure, ParseOutcome};

fn tok(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

let outcome = parse(&tok(&["--model", "m.bin", "--prompt", "hi", "--temperature", "hot"]));
match outcome {
    ParseOutcome::Failure(ParseFailure::InvalidValue { option, value }) => {
        assert_eq!(option, "--temperature");
        assert_eq!(value, "hot");
    }
    _ => panic!("expected InvalidValue"),
}
```

The other five variants cover the remaining refusal shapes, including two
mode-selecting options at once (`MutuallyExclusive`) and no mode at all
(`NoModeSelected`).

### Route exit status and streams

`exit_status` and `stream_routing` return plain data; the process entry
point applies them. From `crates/invocation/src/diagnostics.rs`.

```rust
use turbospark_invocation::{exit_status, parse, stream_routing, ExitStatus, ParseOutcome};

fn tok(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

let help = parse(&tok(&["--help"]));
assert_eq!(exit_status(&help), ExitStatus::Success);
assert_eq!(exit_status(&help).code(), 0);
let routing = stream_routing(&help);
assert!(routing.primary.is_some()); // usage on the primary stream

let bad = parse(&tok(&["--nope"]));
assert_eq!(exit_status(&bad), ExitStatus::InvalidInvocation);
assert_eq!(exit_status(&bad).code(), 2);
assert!(stream_routing(&bad).diagnostic.is_some());
```

## Gotchas

{% hint style="warning" %}
Adding a flag is a cross-surface change. Update the value type in
`request.rs`, the parser, the defaults, the help text, the wire shape if a
server surface mirrors it, and the tests together, as the crate's `AGENTS.md`
requires. Unknown values must fail explicitly; never add a silent fallback
default.
{% endhint %}

{% hint style="warning" %}
A value-taking option always consumes the very next token as its value,
whatever that token looks like, and a help token short-circuits the instant
it is reached, ahead of every whole-invocation check. Do not reorder the
whole-invocation checks either: they run in one fixed order after a complete
scan, and only the first failure is reported.
{% endhint %}

## Related pages

- [core](core.md)
- [The CLI](../command-line-and-local-api/cli.md)
- [The Local Server](../command-line-and-local-api/server.md)
- [Environment Variables](../advanced/environment-variables.md)
