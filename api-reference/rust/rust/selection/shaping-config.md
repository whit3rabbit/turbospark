---
description: "Field-by-field reference for ShapingConfig: accepted ranges, defaults, builder methods, and penalty rules."
icon: sliders
---

# Shaping configuration

`ShapingConfig` holds every sampling setting for one `select` call. Each value is checked when you set it, so a config you hold is always in range. Build it once and reuse it across steps, or build a new one per request. The type is `Copy`, so passing it by value costs nothing.

Create the base config with `ShapingConfig::new`. Add presence, frequency, and min-p settings with the `with_*` builder methods. Each builder returns `Result<ShapingConfig, SelectionError>`.

## Settings

| Setting | Set with | Accepted values | Default | Notes |
| --- | --- | --- | --- | --- |
| `temperature` | `new`, 1st argument | Finite, `>= 0.0` | None, required | `0.0` selects greedily. Any positive value samples. |
| `top_k` | `new`, 2nd argument | `0` to `256` | `0` (off) | Caps the candidate list by rank. |
| `top_p` | `new`, 3rd argument | `None`, or finite in `(0, 1]` | `None` (off) | Keeps the smallest prefix whose probability mass reaches the threshold. |
| `repetition_penalty` | `new`, 4th argument | Finite, `> 0.0` | `1.0` (no effect) | Applies to every distinct token in the full history. |
| `seed` | `new`, 5th argument | `Option<u64>` | `None` | `None` draws from the clock, so runs differ. |
| `presence_penalty` | `with_presence_penalty` | Finite, in `[-2, 2]` | `0.0` (no effect) | Flat subtraction, once per distinct generated token. |
| `frequency_penalty` | `with_frequency_penalty` | Finite, in `[-2, 2]` | `0.0` (no effect) | Subtraction scaled by how often the token appears in the generated suffix. |
| `min_p` | `with_min_p` | Finite, in `[0, 1)` | Off | Drops candidates below `min_p` times the top probability. |

The source for these checks is [`shaping.rs`](https://github.com/whit3rabbit/turbospark/blob/main/crates/selection/src/shaping.rs). Its doc comments give the reasons behind each range.

## Example

This example builds a config, then reports which setting failed. It reuses `SelectionError` as the common error type.

```rust
use turbospark_selection::{SelectionError, ShapingConfig};

fn build_config(temperature: f64, min_p: f64) -> Result<ShapingConfig, SelectionError> {
    ShapingConfig::new(temperature, 40, Some(0.9), 1.0, Some(1234))?
        .with_presence_penalty(0.0)?
        .with_frequency_penalty(0.0)?
        .with_min_p(min_p)
}

fn main() {
    match build_config(0.7, 1.0) {
        Ok(config) => println!("min_p: {:?}", config.min_p()),
        Err(e) => println!("rejected: {e}"),
    }
}
```

This prints `rejected: invalid selection input: min_p must be finite and in [0, 1), got 1`.

## Rules that are easy to miss

- **Top-p without top-k:** Setting `top_k` to `0` with a `top_p` below `1.0` is ordinary nucleus sampling. The top-k cap is skipped.
- **`min_p` of `0.0`:** it means off. The crate stores it as `None`, so `min_p()` returns `None` after `with_min_p(0.0)`.
- **`min_p` of `1.0`:** it is refused. A threshold of 1.0 would keep only the single most likely token. For greedy decoding, use `temperature` of `0.0` instead.
- **Negative penalties:** The presence and frequency ranges include negative values, which push the model toward repetition.
- **Repetition penalty above 1.0:** it discourages repeats. Positive scores are divided by the penalty, and negative scores are multiplied by it. A value below 1.0 boosts repeated tokens.

## Penalty scope

The repetition penalty reads the full `history` slice, prompt included. Presence and frequency penalties read only the generated suffix, which is the last `position` entries of `history`.

The source documents this as a deliberate choice, so that a prompt cannot suppress tokens the model needs to quote back. The source also notes that this differs from OpenAI's convention, which counts prompt tokens too. Check this if you port penalty settings from an OpenAI client.

## Getters

Each setting has a getter with the same name: `temperature()`, `top_k()`, `top_p()`, `repetition_penalty()`, `seed()`, `presence_penalty()`, `frequency_penalty()`, and `min_p()`. The method `is_deterministic()` returns `true` only when `temperature()` equals `0.0`.

## Next steps

- [Overview](README.md): the `select` pipeline and how `position` is used.
- [Rejection sampling](rejection-sampling.md): shaping a draft and target model under the same config.
