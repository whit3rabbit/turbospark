---
description: "Fit a conversation into a length bound by dropping its oldest eligible turns."
icon: box
---

# turbospark-window-fit

Use `fit_conversation_window` when a chat history may be longer than the room you have for a prompt. The function removes the oldest eligible turns until a length you measure falls under a bound you choose. It is deterministic, performs no I/O, and keeps no state between calls.

The crate does not count tokens. You pass a measurement callback, and the crate asks it how long the retained conversation is. Every decision follows from that number and the bound.

{% hint style="warning" %}
A result that is still at or over the bound is not an error. The function returns it as a normal outcome, so check `has_room_for_generation()` before you send the prompt.
{% endhint %}

## Add the crate

The package is `turbospark-window-fit`. Your Rust code imports it as `turbospark_window_fit`.

Add the package from crates.io once the version you need is published. Until then, use a path or git dependency from a checkout of the repository. A path dependency looks like this:

```toml
[dependencies]
turbospark-window-fit = { path = "../turbospark/crates/window-fit" }
```

Inside the TurboSpark workspace, the CLI and FFI crates import this package under the alias `window_fit`. That alias is internal to the workspace. In your own code, depend on the package name and import `turbospark_window_fit`.

## Quick example

This example fits a four-turn history that starts with a system instruction. The stand-in measurement counts bytes. A real caller counts tokens from its tokenizer.

```rust
use turbospark_window_fit::{fit_conversation_window, WindowFitOutcome};

#[derive(Clone, Debug, PartialEq)]
struct Turn {
    role: &'static str,
    text: &'static str,
}

fn main() {
    let turns = vec![
        Turn { role: "system", text: "Answer in one sentence." },
        Turn { role: "user", text: "What is a window?" },
        Turn { role: "assistant", text: "A bounded slice of history." },
        Turn { role: "user", text: "Give an example." },
    ];

    // Stand-in for a real tokenizer. Measure the slice you are given.
    let measure = |slice: &[Turn]| -> u64 {
        slice.iter().map(|t| t.text.len() as u64).sum()
    };

    let outcome: WindowFitOutcome<Turn> = fit_conversation_window(&turns, true, 40, measure);

    println!("kept {} turns", outcome.retained_turns().len());
    println!("removed {}", outcome.removed_turn_count());
    println!("measured {}", outcome.measured_length());
    println!("room for reply: {}", outcome.has_room_for_generation());
    for turn in outcome.retained_turns() {
        println!("{}: {}", turn.role, turn.text);
    }
}
```

The output is:

```text
kept 2 turns
removed 2
measured 39
room for reply: true
system: Answer in one sentence.
user: Give an example.
```

The system turn stays because `has_leading_instruction` is `true`. The last turn stays because it is the newest. The two middle turns go, oldest first, and the remaining 39 units fall under the bound of 40.

## What the caller measures

The `measure` callback receives the whole retained conversation as one `&[T]` and returns one `u64`. Use the same unit for the return value and for `bound`. Tokens are the usual choice, but the crate does not care which unit you pick.

Measure what the model will receive. That includes the chat template, any system preamble, and any tool definitions in the prompt. The CLI measures with the reasoning level the turn renders with, because a count without it under-counts the prompt. The FFI includes the offered tools in every slice, because a budget that ignores them keeps turns the real prompt has no room for.

Choose the bound yourself. The crate does not reserve room for the reply. If you need `N` units for the reply, pass the context window minus `N`. The CLI passes the resolved context window, the same value its KV cache was sized with, rather than the value the user requested.

The callback runs once before any removal and once after each removal. Removing `k` turns makes `k + 1` calls, and each call sees the whole remaining conversation. The cost grows with the number of removals, so time your tokenizer on a realistic history before you put this in a hot path.

The callback returns a `u64`, so it cannot report an error. The workspace FFI returns `u64::MAX` when a render fails. That value is over every bound, so the function removes eligible turns until none remain. If `measured_length()` is `u64::MAX`, the retained turns still fail to render, so check the render path rather than the bound.

## Fitting rules

The function runs these steps in order:

1. Measure the whole conversation.
2. While the measured length is at or over the bound and a turn is eligible, remove the oldest eligible turn and measure again.
3. Return the retained turns, the last measured length, the removed count, and whether the length is under the bound.

Two turns are pinned. The newest turn is the last element of `turns`, whatever its role, and it is never removed. When `has_leading_instruction` is `true`, `turns[0]` is pinned too. Every other turn is eligible, and removal always starts with the oldest one.

| Turns | Leading instruction | Removable indices |
| --- | --- | --- |
| 3 | no | 0, 1 |
| 3 | yes | 1 |
| 2 | yes | none |
| 1 | no | none |
| 0 | no | none |

A conversation with nothing eligible comes back unchanged, even when it is still over the bound.

The function copies the input slice into an owned `Vec<T>`. That is why `T` needs `Clone`.

## The outcome

`fit_conversation_window` returns a `WindowFitOutcome<T>`. Its accessors are:

| Method | Returns | Meaning |
| --- | --- | --- |
| `retained_turns()` | `&[T]` | The kept turns, in original order. |
| `measured_length()` | `u64` | The last measurement. This is the initial measurement when nothing was removed, and the measurement taken after the last removal otherwise. |
| `removed_turn_count()` | `usize` | How many turns were removed. The count does not say which ones, so keep an ID on each turn if you need that. |
| `has_room_for_generation()` | `bool` | `true` when `measured_length()` is under the bound. |

`WindowFitOutcome<T>` derives `Debug`, `Clone`, `PartialEq`, and `Eq`. Its fields are private, and only the crate can construct one. You always get an outcome from `fit_conversation_window`.

## Refusals

The function returns the outcome directly, not a `Result`. A prompt that still does not fit comes back with `has_room_for_generation()` set to `false`. Your code decides whether to refuse the request, ask the user to shorten the newest message, or raise the budget.

```rust
use turbospark_window_fit::fit_conversation_window;

#[derive(Clone)]
struct Turn {
    text: &'static str,
}

fn prepare(turns: &[Turn], bound: u64) -> Result<Vec<Turn>, String> {
    let outcome = fit_conversation_window(turns, true, bound, |s: &[Turn]| {
        s.iter().map(|t| t.text.len() as u64).sum()
    });
    if !outcome.has_room_for_generation() {
        return Err(format!(
            "{} units measured, bound {bound}; {} turns kept",
            outcome.measured_length(),
            outcome.retained_turns().len()
        ));
    }
    Ok(outcome.retained_turns().to_vec())
}
```

With a bound of 5 on the example history above, `prepare` returns `Err`. The message reports 39 units measured against a bound of 5, with 2 turns kept. The pinned turns alone are over the bound, so the function cannot shorten the prompt further.

## Platform and dependencies

- The crate has no dependencies. `Cargo.toml` has no `[dependencies]` table.
- The crate has no `cfg` gates and no platform checks. It builds on any target the Rust toolchain supports.
- The minimum Rust version is 1.82, from the workspace `rust-version`. The edition is 2021.
- `src/` contains no `unsafe` blocks.
- The callback runs on the calling thread. The function adds no `Send` or `Sync` bounds of its own.
- The repository's benchmark docs have no timing rows for this crate. Measure the cost with your own tokenizer.

## Where to go next

- [API reference on docs.rs](https://docs.rs/turbospark-window-fit/latest/turbospark_window_fit/) lists every public item.
- [Tokenizer crate on docs.rs](https://docs.rs/turbospark-tokenizer/latest/turbospark_tokenizer/) is the usual source for the measurement.
- [CLI chat call site](https://github.com/whit3rabbit/turbospark/blob/main/crates/cli/src/chat.rs) shows the fit inside a real loop, including the reasoning-level measurement.
- [Rust library overview](../../rust.md) covers the other crates in the workspace.
