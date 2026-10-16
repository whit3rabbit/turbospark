---
description: >-
  Pick the next token ID from a logit vector with validated sampling settings,
  using the turbospark-selection crate.
icon: box
---

# Sampling

`turbospark-selection` turns one row of model logits into one token ID. Use it when you already run a model on the host and want TurboSpark's sampling rules (temperature, top-k, top-p, min-p, repetition penalty, presence penalty, and seeded draws) without pulling in the runtime. The crate is pure Rust with one dependency, `turbospark-core`, which provides the logit and token types.

The package name is `turbospark-selection`. Rust code imports it as `turbospark_selection`. Add the package from crates.io once the version you need is published, or use a path or git dependency from a checkout. Inside this workspace, some crates refer to `turbospark-core` through the alias `foundation`. That alias is workspace-internal, so your code imports from `turbospark_core`.

```toml
[dependencies]
turbospark-selection = { path = "../turbospark/crates/selection" }
turbospark-core = { path = "../turbospark/crates/core" }
```

## Quick start

`select` takes a logit view, a validated `ShapingConfig`, the token history, and the step position. It returns one `TokenId`, which is `i32`.

```rust
use turbospark_core::{LogitValue, LogitsView, TokenId};
use turbospark_selection::{select, ShapingConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ShapingConfig::new(0.7, 40, Some(0.95), 1.1, Some(42))?;
    let scores: Vec<LogitValue> = [2.0f32, 0.5, -1.0, 1.5]
        .into_iter()
        .map(LogitValue::from_f32)
        .collect();
    let history: Vec<TokenId> = vec![];
    let token = select(LogitsView::new(&scores), &config, &history, 0)?;
    println!("{token}");
    Ok(())
}
```

`LogitValue` is an alias for `half::f16`, so `LogitValue::from_f32` works without adding `half` to your own `Cargo.toml`.

## Decode loop

Call `select` once per generated token. Append each result to `history`, and pass the number of tokens generated so far as `position`.

```rust
use turbospark_core::{LogitValue, LogitsView, TokenId};
use turbospark_selection::{select, ShapingConfig};

fn model_logits(step: usize) -> Vec<LogitValue> {
    (0..8)
        .map(|i| LogitValue::from_f32(((i + step) % 8) as f32 * 0.25))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ShapingConfig::new(0.8, 0, None, 1.0, Some(7))?
        .with_presence_penalty(0.5)?
        .with_frequency_penalty(0.2)?
        .with_min_p(0.05)?;

    let mut history: Vec<TokenId> = vec![5, 9, 2];
    for step in 0..4u64 {
        let scores = model_logits(step as usize);
        let next = select(LogitsView::new(&scores), &config, &history, step)?;
        history.push(next);
    }
    println!("{history:?}");
    Ok(())
}
```

{% hint style="warning" %}
`position` does two jobs. It seeds the draw, and it marks where the generated suffix starts inside `history`.

Presence and frequency penalties count only the last `position` tokens. If you pass the prompt length, or a count that does not match what you appended, those penalties land on the wrong tokens and seeded replays diverge. `position` is clamped to `history.len()` rather than trusted, so a too-large value counts all of `history` as generated.
{% endhint %}

## What `select` does

`select` runs these steps in order. The source is in [`choose.rs`](https://github.com/whit3rabbit/turbospark/blob/main/crates/selection/src/choose.rs).

1. Reject an empty score slice and any non-finite score.
2. If `temperature` is exactly `0.0`, return the highest score. Ties go to the lowest index. The seed, penalties, and truncation are ignored.
3. Apply the repetition penalty to every distinct token in `history`, prompt included.
4. Apply presence and frequency penalties to the generated suffix only.
5. Convert scores to probabilities with a stable softmax, then rank them.
6. Apply top-k as a cap on the ranked list, then top-p over that list, then min-p against the top probability.
7. Raise each survivor's probability to the power `1 / temperature`, then renormalize.
8. Draw one survivor using a uniform value derived from the seed and `position`.

A temperature above `0.0`, however small, takes the sampling path. Only the exact value `0.0` is greedy.

## Determinism

With `Some(seed)`, the same seed, scores, history, position, and config always return the same token. Each position derives its own value, so replaying a sequence in order reproduces it token for token. With `None`, the draw is seeded from the system clock, so runs differ.

## Errors

Every fallible call returns `SelectionError`, which has one public field, `reason: String`. Its `Display` output starts with `invalid selection input:`. The wording of `reason` is not a stable contract, so branch on the error itself, not on its text.

| Call                                              | Fails when                                                        |
| ------------------------------------------------- | ----------------------------------------------------------------- |
| `ShapingConfig::new`                              | Temperature, top-k, top-p, or repetition penalty is out of range. |
| `with_presence_penalty`, `with_frequency_penalty` | The value is non-finite or outside `[-2, 2]`.                     |
| `with_min_p`                                      | The value is non-finite or outside `[0, 1)`.                      |
| `select`, `shaped_distribution`                   | The score slice is empty, or a score is NaN or infinite.          |

Each `with_*` method consumes the config and returns a new one, so a failed step ends the chain with an error. Build the config again with corrected values.

## Public surface

The crate root re-exports the items below. Other items are reachable through the public modules (`penalty`, `truncation`, `derive`, and so on), but this reference does not cover them.

| Item                                                                                                                    | Kind   | Purpose                                                                         |
| ----------------------------------------------------------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------------- |
| [`select`](https://docs.rs/turbospark-selection/latest/turbospark_selection/fn.select.html)                             | fn     | Draw one token ID from shaped logits.                                           |
| [`shaped_distribution`](https://docs.rs/turbospark-selection/latest/turbospark_selection/fn.shaped_distribution.html)   | fn     | Return the surviving tokens and their probabilities that `select` samples from. |
| [`residual`](https://docs.rs/turbospark-selection/latest/turbospark_selection/fn.residual.html)                         | fn     | Compute the normalized `max(p - q, 0)` used for speculative correction.         |
| [`ShapingConfig`](https://docs.rs/turbospark-selection/latest/turbospark_selection/struct.ShapingConfig.html)           | struct | Validated sampling settings. Copyable.                                          |
| [`ShapedDistribution`](https://docs.rs/turbospark-selection/latest/turbospark_selection/struct.ShapedDistribution.html) | struct | An explicit support with probabilities.                                         |
| [`SelectionError`](https://docs.rs/turbospark-selection/latest/turbospark_selection/struct.SelectionError.html)         | struct | Configuration and input rejection. Implements `std::error::Error`.              |

The `shaping` module also exports `MAX_TOP_K` (256) and `DEFAULT_REPETITION_PENALTY` (1.0).

## Platform and limits

* The crate has no `unsafe` blocks in its source.
* It uses `std` (hash maps, `thread_local!`, and the system clock), so it does not build for `no_std` targets.
* Scratch buffers live in a `thread_local!`. Each thread keeps its own buffers, sized to the vocabulary, and reuses them across calls.
* `top_k` is capped at 256. The crate does not impose a vocabulary size limit.
* The workspace sets `rust-version = "1.82"`.

## Next steps

* [Shaping configuration](shaping-config.md): every setting, its accepted range, and the penalty rules.
* [Rejection sampling](rejection-sampling.md): using `shaped_distribution` and `residual` for speculative decoding.
* [Rust library overview](../): the other crates in the workspace.
